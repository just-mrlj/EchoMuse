"""Metadata decisions and real refresh logic, without runtime dependencies."""
import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import em_oww_metadata as metadata
import em_oww_models


@pytest.mark.parametrize("values", [{}, {"wake_word": ""}, {"wake_word": None}, {"wake_word": []}, {"language": "invalid code"}])
def test_stock_fallback(values):
    assert metadata.from_metadata("hey_jarvis_v0.1", values) == metadata.WakeWord("hey jarvis")


def test_custom_identity_does_not_change_prediction_key():
    name = "/models/not_the_phrase.onnx"
    assert metadata.from_metadata(name, {"wake_word": " Hallo Welt ", "language": "de_DE"}) == metadata.WakeWord("Hallo Welt", ("de-DE",))
    assert em_oww_models.prediction_key(name) == "not_the_phrase"


def _init_of(class_name):
    source = Path(__file__).resolve().parents[1] / "em_esphome.py"
    tree = ast.parse(source.read_text())
    class_def = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_def.body
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")


def test_the_server_is_built_with_its_wake_offer():
    """The offer HA reads is resolved (names and all) before the server
    exists, so its first read is already correct (#112)."""
    constructor = _init_of("DeviceESPhomeServer")
    positional = constructor.args.posonlyargs + constructor.args.args
    required = positional[:len(positional) - len(constructor.args.defaults)]
    assert "wake_offer" in {argument.arg for argument in required}
    assignments = [
        node for node in ast.walk(constructor)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(t, ast.Attribute) and t.attr == "wake_offer"
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Name)
    assert assignments[0].value.id == "wake_offer"


def test_the_satellite_reads_the_offer_rather_than_copying_it():
    """A copy taken at connect is what HA's read-back after its own write
    would see stale; the satellite reads its server's offer at request time."""
    args = {a.arg for a in _init_of("EchoMuseSatellite").args.args}
    assert not args & {"oww_model_id", "oww_model_info", "wake_offer"}


def _update_wake_words():
    source = Path(__file__).resolve().parents[1] / "em_esphome.py"
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == "update_wake_words")
    return source, function


def test_a_changed_offer_bounces_ha_only_once_it_has_read_the_old_one():
    import em_wake_models as wm
    source, function = _update_wake_words()
    satellite = SimpleNamespace(disconnect=Mock())
    old = wm.Offer((wm.Advertised("robot", "Old"),), ("robot",), 4)
    new = wm.Offer((wm.Advertised("robot", "Hallo", ("de",)),), ("robot",), 4)
    server = SimpleNamespace(wake_offer=old, wake_offer_told=None,
                             get_satellite=lambda: satellite)
    scope = {"get_server": lambda device: server, "log": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)

    # HA has not read anything yet: it will read the new offer when it does.
    asyncio.run(scope["update_wake_words"]("device", new))
    assert server.wake_offer == new
    satellite.disconnect.assert_not_called()

    # It read the old one; a re-trained model's new name must reach it.
    server.wake_offer, server.wake_offer_told = old, old
    asyncio.run(scope["update_wake_words"]("device", new))
    satellite.disconnect.assert_called_once()

    # What it read is still right — HA's own change, read back — no bounce.
    server.wake_offer_told = new
    asyncio.run(scope["update_wake_words"]("device", new))
    satellite.disconnect.assert_called_once()


def test_a_newer_resolution_wins_when_naming_finishes_out_of_order(monkeypatch):
    import sys
    import em_wake_models as wm
    import em_wake_sync
    told = []
    monkeypatch.setitem(sys.modules, "em_esphome", SimpleNamespace(
        update_wake_words=lambda device_id, offer: told.append(offer) or asyncio.sleep(0),
        get_wake_offer=lambda device_id: None))
    async def push(_event):
        return None
    monkeypatch.setitem(sys.modules, "em_api", SimpleNamespace(_push_event=push))
    first = wm.Resolution(("a",), ("a",), "dashboard", ("a",))
    second = wm.Resolution(("b",), ("b",), "ha", ("b",))
    live = SimpleNamespace(device_id="device", wake_resolution=first,
                           oww_threshold=0.5)

    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        async def offer_for(res):
            if res is first:
                started.set()
                await release.wait()
            return wm.Offer(active=res.active)
        monkeypatch.setattr(em_wake_sync, "offer_for", offer_for)
        slow = asyncio.create_task(em_wake_sync.publish("device", live, first))
        await started.wait()
        live.wake_resolution = second
        await em_wake_sync.publish("device", live, second)
        release.set()
        await slow
    asyncio.run(run())
    assert [o.active for o in told] == [("b",)]


def test_missing_and_unreadable_models_fall_back(tmp_path, monkeypatch):
    path = tmp_path / "hey_robot.onnx"
    assert metadata.resolve(str(path)).name == "hey robot"
    path.write_bytes(b"not an onnx model")
    def fail(*args):
        raise RuntimeError("unreadable metadata")
    monkeypatch.setattr(metadata, "_read", fail)
    assert metadata.resolve(str(path)).name == "hey robot"


def test_file_replacement_changes_reader_identity(tmp_path, monkeypatch):
    path = tmp_path / "robot.onnx"
    path.write_bytes(b"first")
    reader = Mock(return_value={"wake_word": "First"})
    monkeypatch.setattr(metadata, "_read", reader)
    assert metadata.resolve(str(path)).name == "First"
    old = reader.call_args
    path.write_bytes(b"a different model")
    reader.return_value = {"wake_word": "Second"}
    assert metadata.resolve(str(path)).name == "Second"
    assert reader.call_args != old
