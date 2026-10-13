"""
em_wake_models: more than one wake word per Echo (#112).

Home Assistant gives a satellite two wake word pickers, each beside its own
pipeline, and routes a run by the `wake_word_phrase` the satellite sends. The
rules here come from what HA reads (esphome/select.py, entry_data.py,
assist_satellite.py, dev 2026-10-13): options keyed by display NAME, the union
of both pickers sent uncapped, pickers hidden at a maximum below 1. Tests are
written from those edges rather than from typical input.
"""

import concurrent.futures
import math

import pytest

import em_barge
import em_config_types as T
import em_db as db
import em_dbwriter
import em_oww_models
import em_wake_models as wm

STOCK = [("hey_jarvis_v0.1", "hey jarvis"), ("alexa_v0.1", "alexa"),
         ("hey_mycroft_v0.1", "hey mycroft"), ("hey_rhasspy_v0.1", "hey rhasspy")]


# ── Names: unique, and never moved once HA has seen them ─────────────────────

def test_stock_names_are_their_phrases():
    names, _ = wm.assign_names({}, STOCK)
    assert names == dict(STOCK)


def test_a_newcomer_that_sorts_earlier_never_renames_an_existing_model():
    """HA restores its picker by NAME. If an upload that sorts earlier took the
    plain name, the existing model would be renamed and HA's picker would
    silently read "No wake word"."""
    names, reg = wm.assign_names({}, STOCK + [("zeta_robot", "hey robot")])
    assert names["zeta_robot"] == "hey robot"
    # alpha_robot sorts first and arrives first in the candidate list.
    names, reg = wm.assign_names(reg, STOCK + [("alpha_robot", "hey robot"),
                                               ("zeta_robot", "hey robot")])
    assert names["zeta_robot"] == "hey robot"
    assert names["alpha_robot"] == "hey robot (alpha_robot)"
    # And again, in any order: nothing moves.
    again, _ = wm.assign_names(reg, [("zeta_robot", "hey robot"),
                                     ("alpha_robot", "hey robot")] + STOCK)
    assert again == names


def test_a_custom_model_named_like_a_stock_one_is_suffixed():
    names, _ = wm.assign_names({}, STOCK + [("my_alexa", "alexa")])
    assert names["alexa_v0.1"] == "alexa"
    assert names["my_alexa"] == "alexa (my_alexa)"


def test_names_are_unique_even_against_a_literal_suffix():
    names, _ = wm.assign_names({}, [("a", "x"), ("b", "x (c)"), ("c", "x")])
    assert len(set(names.values())) == 3


def test_a_retrained_phrase_is_renamed_and_a_deleted_model_drops_out():
    names, reg = wm.assign_names({}, STOCK + [("robot", "hey robot")])
    names, reg = wm.assign_names(reg, STOCK + [("robot", "hallo roboter")])
    assert names["robot"] == "hallo roboter"
    names, reg = wm.assign_names(reg, STOCK)
    assert "robot" not in reg


def test_a_damaged_registry_is_rebuilt_not_trusted():
    names, _ = wm.assign_names({"alexa_v0.1": "not a dict", "x": None}, STOCK)
    assert names == dict(STOCK)
    names, _ = wm.assign_names("garbage", STOCK)
    assert names == dict(STOCK)


# ── How many at once ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("device_max, expected", [(None, 1), (0, 1), (-3, 1), (1, 1),
                                                   (2, 2), ("x", 1)])
def test_an_echo_detecting_its_own_runs_what_its_firmware_says(device_max, expected):
    """Firmware that says nothing runs one; never below 1, since HA hides
    both pickers at 0."""
    assert wm.max_active(mode="on", device_max=device_max) == expected


@pytest.mark.parametrize("mode", ["off", "shadow", "unknown"])
def test_the_controller_limit_applies_wherever_it_scores(mode):
    assert wm.max_active(mode=mode, device_max=2) == wm.CONTROLLER_MAX
    assert wm.CONTROLLER_MAX >= 2, "HA's two pickers must both fit"


# ── HA's request → what is stored ────────────────────────────────────────────

AVAILABLE = {"hey_jarvis_v0.1", "alexa_v0.1", "hey_mycroft_v0.1", "robot"}


def test_exactly_the_dashboard_model_clears_the_choice():
    """HA has no "follow the dashboard" option; this is how to get back."""
    assert wm.override_from_request(["hey_jarvis_v0.1"], available=AVAILABLE,
                                    default_id="hey_jarvis_v0.1") is None
    assert wm.override_from_request(["hey_jarvis_v0.1", "hey_jarvis_v0.1"],
                                    available=AVAILABLE, default_id="hey_jarvis_v0.1") is None


def test_duplicates_and_unknown_ids_go_and_order_stays():
    got = wm.override_from_request(["alexa_v0.1", "okay_nabu", "alexa_v0.1", "hey_jarvis_v0.1"],
                                   available=AVAILABLE, default_id="hey_jarvis_v0.1")
    assert got == ["alexa_v0.1", "hey_jarvis_v0.1"]


def test_one_other_model_is_a_choice():
    assert wm.override_from_request(["robot"], available=AVAILABLE,
                                    default_id="hey_jarvis_v0.1") == ["robot"]


# ── What an Echo listens for ─────────────────────────────────────────────────

INSTALLED = {"hey_jarvis_v0.1": "hey_jarvis_v0.1", "alexa_v0.1": "alexa_v0.1",
             "robot": "/data/oww_models/robot.onnx"}


def test_no_choice_follows_the_dashboard():
    r = wm.resolve(default_model="alexa_v0.1", override=None, installed=INSTALLED, max_n=4)
    assert (r.active, r.models, r.source, r.dropped) == (
        ("alexa_v0.1",), ("alexa_v0.1",), "dashboard", ())


def test_ha_choice_maps_ids_back_to_model_files():
    r = wm.resolve(default_model="hey_jarvis_v0.1", override=["robot", "alexa_v0.1"],
                   installed=INSTALLED, max_n=4)
    assert r.source == "ha"
    assert r.active == ("robot", "alexa_v0.1")
    assert r.models == ("/data/oww_models/robot.onnx", "alexa_v0.1")


def test_more_than_the_maximum_is_dropped_and_said_so():
    """HA does not cap what it sends; an Echo told more than it can run would
    show a wake word in HA that never fires."""
    r = wm.resolve(default_model="hey_jarvis_v0.1", override=["robot", "alexa_v0.1"],
                   installed=INSTALLED, max_n=1)
    assert r.active == ("robot",)
    assert r.dropped == (("alexa_v0.1", wm.OVER_MAX),)


def test_a_vanished_model_falls_back_to_the_dashboard_and_stays_visible():
    r = wm.resolve(default_model="hey_jarvis_v0.1", override=["gone"],
                   installed=INSTALLED, max_n=4)
    assert r.active == ("hey_jarvis_v0.1",)
    assert r.source == "ha" and r.dropped == (("gone", wm.NOT_INSTALLED),)


def test_the_dashboard_model_is_used_as_configured():
    """A custom file whose stem equals a stock id: the configured path wins
    for the dashboard's own model, so the device is never switched under it."""
    r = wm.resolve(default_model="/x/alexa_v0.1.onnx", override=None,
                   installed=INSTALLED, max_n=4)
    assert r.models == ("/x/alexa_v0.1.onnx",)


def test_max_is_never_below_one():
    r = wm.resolve(default_model="alexa_v0.1", override=None, installed=INSTALLED, max_n=0)
    assert r.max == 1 and r.active == ("alexa_v0.1",)


def test_what_is_stored_is_what_has_pickers_will_show():
    """HA resets a picker whose wake word it does not read back as active. A
    choice kept beyond that would, once the maximum rises, have the Echo
    listen for a word whose phrase matches no picker — the wrong assistant."""
    over = wm.resolve(default_model="hey_jarvis_v0.1", override=["robot", "alexa_v0.1"],
                      installed=INSTALLED, max_n=1)
    assert wm.settled(over, "hey_jarvis_v0.1") == ["robot"]
    gone = wm.resolve(default_model="hey_jarvis_v0.1", override=["gone"],
                      installed=INSTALLED, max_n=4)
    assert wm.settled(gone, "hey_jarvis_v0.1") is None
    both = wm.resolve(default_model="hey_jarvis_v0.1", override=["hey_jarvis_v0.1", "robot"],
                      installed=INSTALLED, max_n=1)
    assert wm.settled(both, "hey_jarvis_v0.1") is None, "only the default left: follow it"
    dash = wm.resolve(default_model="alexa_v0.1", override=None, installed=INSTALLED, max_n=4)
    assert wm.settled(dash, "alexa_v0.1") is None


# ── Thresholds per model ─────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, True, "0.3", math.nan, math.inf, 0, -0.1, 1.5])
def test_a_threshold_that_is_not_a_bar_falls_back(value):
    assert wm.threshold_for("robot", {"robot": value}, 0.5) == 0.5


def test_own_threshold_and_fallback():
    assert wm.threshold_for("robot", {"robot": 0.3}, 0.5) == 0.3
    assert wm.threshold_for("alexa_v0.1", {"robot": 0.3}, 0.5) == 0.5
    assert wm.threshold_for("robot", None, 0.5) == 0.5


def test_the_barge_bar_only_ever_lowers():
    b = wm.bars(["a", "b"], {"a": 0.2}, 0.5, lowered_to=0.25)
    assert b == {"a": 0.2, "b": 0.25}


def test_one_model_is_judged_exactly_as_before():
    assert wm.judge({"a": 0.6}, {"a": 0.5}) == ("a", 0.6, 0.5)
    assert wm.judge({"a": 0.4}, {"a": 0.5}) == ("a", 0.4, 0.5)


def test_each_model_is_judged_against_its_own_bar():
    """The custom model at 0.35 clears its own 0.3 while the stock one at 0.45
    misses the default 0.5: the custom one fired, though it scored lower."""
    assert wm.judge({"robot": 0.35, "alexa": 0.45},
                    {"robot": 0.3, "alexa": 0.5}) == ("robot", 0.35, 0.3)


def test_of_two_that_cross_the_higher_score_wins():
    assert wm.judge({"a": 0.7, "b": 0.9}, {"a": 0.5, "b": 0.5})[0] == "b"


def test_with_none_crossed_the_nearest_to_its_bar_is_the_near_miss():
    assert wm.judge({"a": 0.2, "b": 0.25}, {"a": 0.22, "b": 0.5})[0] == "a"


# ── The phrase HA routes on ──────────────────────────────────────────────────

NAMES = {"hey_jarvis_v0.1": "hey jarvis", "robot": "hey robot (robot)"}


def test_the_phrase_is_the_advertised_name_of_the_model_that_fired():
    assert wm.phrase("robot", NAMES, ("hey_jarvis_v0.1", "robot")) == "hey robot (robot)"


def test_a_wake_naming_no_model_is_the_first_wake_words():
    """Firmware that does not report which model fired runs one: the first."""
    assert wm.phrase(None, NAMES, ("hey_jarvis_v0.1", "robot")) == "hey jarvis"


def test_an_unnamed_model_falls_back_to_its_display_name():
    assert wm.phrase("hey_mycroft_v0.1", NAMES, ()) == em_oww_models.display_name("hey_mycroft_v0.1")
    assert wm.phrase(None, {}, ()) == ""


# ── What firmware that runs one model is sent ────────────────────────────────

def test_one_model_firmware_is_told_the_first_wake_word_at_its_own_threshold():
    r = wm.resolve(default_model="hey_jarvis_v0.1", override=["robot", "alexa_v0.1"],
                   installed=INSTALLED, max_n=1)
    cfg = {"owwModel": "hey_jarvis_v0.1", "owwThreshold": 0.5, "other": 1}
    out = wm.for_device(cfg, r, {"robot": 0.3}, 0.5)
    assert out == {"owwModel": "/data/oww_models/robot.onnx", "owwThreshold": 0.3, "other": 1}
    assert cfg["owwModel"] == "hey_jarvis_v0.1", "the stored config is not edited"


def test_an_unchanged_config_is_passed_through():
    r = wm.resolve(default_model="alexa_v0.1", override=None, installed=INSTALLED, max_n=4)
    cfg = {"owwModel": "alexa_v0.1", "owwThreshold": 0.5}
    assert wm.for_device(cfg, r, {}, 0.5) is cfg


# ── What the dashboard is told ───────────────────────────────────────────────

def test_the_view_shows_who_chose_and_what_was_not_applied():
    r = wm.resolve(default_model="hey_jarvis_v0.1", override=["robot", "alexa_v0.1"],
                   installed=INSTALLED, max_n=1)
    v = wm.view(r, NAMES, {"robot": 0.3}, 0.5)
    assert v["source"] == "ha" and v["max"] == 1
    assert v["active"] == [{"id": "robot", "name": "hey robot (robot)",
                            "threshold": 0.3, "ownThreshold": True}]
    assert v["dropped"] == [{"id": "alexa_v0.1", "name": "alexa",
                             "reason": wm.OVER_MAX}]


# ── Barge-in over several wake words ─────────────────────────────────────────

def test_decide_many_with_one_model_is_decide():
    for in_playback in (True, False):
        for score, prev in ((0.3, 0.3), (0.3, 0.1), (0.6, 0.0), (0.1, 0.9)):
            one = em_barge.decide(score=score, prev_score=prev, in_playback=in_playback,
                                  barge_threshold=0.25, wake_threshold=0.5)
            many = em_barge.decide_many(scores={"a": score}, prev_scores={"a": prev},
                                        in_playback=in_playback, barge_threshold=0.25,
                                        wake_thresholds={"a": 0.5})
            assert many.fired == one.fired and many.note == one.note


def test_two_frames_must_be_the_same_model():
    """One frame of each half-matching model is not a wake word."""
    many = em_barge.decide_many(scores={"a": 0.1, "b": 0.3}, prev_scores={"a": 0.3, "b": 0.1},
                                in_playback=True, barge_threshold=0.25,
                                wake_thresholds={"a": 0.5, "b": 0.5})
    assert not many.fired


def test_while_thinking_each_model_has_its_own_threshold_and_the_highest_names_it():
    many = em_barge.decide_many(scores={"a": 0.35, "b": 0.45}, prev_scores={},
                                in_playback=False, barge_threshold=0.25,
                                wake_thresholds={"a": 0.3, "b": 0.5})
    assert many.fired and many.model == "a" and many.threshold == 0.3
    many = em_barge.decide_many(scores={"a": 0.6, "b": 0.9}, prev_scores={},
                                in_playback=False, barge_threshold=0.25,
                                wake_thresholds={"a": 0.5, "b": 0.5})
    assert many.model == "b" and many.score == 0.9


# ── Config type and ceiling ──────────────────────────────────────────────────

def test_owwthresholds_is_an_object_of_numbers():
    assert T.KINDS["owwThresholds"] == T.FLOAT_MAP
    assert T.fits(T.FLOAT_MAP, {}) and T.fits(T.FLOAT_MAP, {"a": 0.3, "b": 1})
    for bad in ([0.3], {"a": "0.3"}, {"a": True}, {"a": math.nan}, {1: 0.3}, None):
        assert not T.fits(T.FLOAT_MAP, bad), bad


def test_a_per_model_threshold_that_cannot_fire_is_clamped():
    out = db._clamp_wake_threshold({"owwThresholds": {"a": 1.0, "b": 0.4}}, "test")
    assert out["owwThresholds"] == {"a": db.OWW_THRESHOLD_MAX, "b": 0.4}
    same = {"owwThresholds": {"a": 0.4}}
    assert db._clamp_wake_threshold(same, "test") is same


def test_a_model_ha_has_active_cannot_be_deleted(tmp_path):
    path = str(tmp_path / "robot.onnx")
    users = em_oww_models.in_use_by(path, {"global": {"owwModel": "alexa_v0.1"}},
                                    {"G0900000AAAA": ["alexa_v0.1", "robot"]})
    assert users == ["G0900000AAAA (Home Assistant)"]


# ── Stored: atomic, and through the one writer ───────────────────────────────

@pytest.fixture()
def fresh_db(tmp_path):
    db.init(str(tmp_path / "test.db"))
    yield db
    if db._conn is not None:
        db._conn.close()
        db._conn = None


def test_the_choice_round_trips_and_clears(fresh_db):
    assert db.get_wake_word_override("dev1") is None
    db.set_wake_word_override("dev1", ["robot", "alexa_v0.1"])
    assert db.get_wake_word_override("dev1") == ["robot", "alexa_v0.1"]
    db.set_wake_word_override("dev1", [])
    assert db.get_wake_word_override("dev1") is None
    db.set_wake_word_override("dev1", ["robot"])
    db.set_wake_word_override("dev1", None)
    assert db.get_wake_word_overrides() == {}


def test_bad_json_reads_as_no_choices(fresh_db):
    with db._tx() as conn:
        conn.execute("INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                     ("wake_words_ha", "{not json"))
    assert db.get_wake_word_overrides() == {}


def test_two_echoes_changing_at_once_lose_neither_write(fresh_db):
    """One blob for every device: without the read-modify-write in one tx, the
    second writer would overwrite the first. Written the way the controller
    writes it — through em_dbwriter — from many threads at once."""
    ids = [f"dev{i:02d}" for i in range(24)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = list(pool.map(
            lambda d: em_dbwriter.submit(db.set_wake_word_override, d, [f"m_{d}"]), ids))
    for f in futures:
        f.result(timeout=10)
    assert db.get_wake_word_overrides() == {d: [f"m_{d}"] for d in ids}


def test_names_are_stored_once_and_kept(fresh_db):
    first = db.assign_wake_word_names(STOCK + [("zeta", "hey robot")])
    later = db.assign_wake_word_names(STOCK + [("alpha", "hey robot"), ("zeta", "hey robot")])
    assert first["zeta"] == later["zeta"] == "hey robot"
    assert later["alpha"] == "hey robot (alpha)"


def test_deleting_a_device_forgets_its_choice(fresh_db):
    with db._tx() as conn:
        conn.execute("INSERT INTO devices (device_id, label, ip, approved) VALUES (?, ?, ?, 1)",
                     ("dev1", "Kitchen", "10.0.0.1"))
    db.set_wake_word_override("dev1", ["robot"])
    db.set_wake_word_override("dev2", ["alexa_v0.1"])
    db.delete_device("dev1")
    assert db.get_wake_word_overrides() == {"dev2": ["alexa_v0.1"]}
