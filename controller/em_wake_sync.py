"""
em_wake_sync.py — carries em_wake_models' answers to the Echo and to HA (#112)

em_wake_models decides; this reads what it decides from (the installed
models, HA's stored choice, the names HA has already been given) and hands
the answer on: to the live Device's mirrors, to the ESPHome satellite's
configuration, and to the dashboard. Both em_controller and em_api call it,
which is why it is not a function in either.
"""

from __future__ import annotations

import asyncio
import logging

import em_db as db
import em_dbwriter
import em_oww_assets
import em_oww_metadata
import em_oww_models
import em_shadow
import em_wake_models as wm

log = logging.getLogger("echomuse.wakewords")  # not __name__ (#378)

DEFAULT_MODEL = "hey_jarvis_v0.1"

# Advertised wake words, built once and rebuilt after an upload or delete.
# Reading a custom model's name loads it in onnxruntime, so this is not done
# per connection.
_advertised: tuple[wm.Advertised, ...] | None = None
_advertised_lock: asyncio.Lock | None = None


def installed_models() -> dict[str, str]:
    """
    Every wake word this controller can offer: {prediction key: owwModel}.

    Stock first, in em_oww_assets' order, then custom models by filename. A
    custom model whose stem equals a stock name cannot be told apart from it
    by openwakeword, which keys its predictions by stem, so the stock model
    keeps that id.
    """
    out = {m: m for m in em_oww_assets.STOCK_MODELS}
    for entry in em_oww_models.scan():
        out.setdefault(entry["name"], entry["path"])
    return out


def invalidate() -> None:
    """Forget the advertised list; the next read rebuilds it. After an upload
    or a delete, before telling the devices."""
    global _advertised
    _advertised = None


async def advertised() -> tuple[wm.Advertised, ...]:
    """The installed wake words with their stable names and languages."""
    global _advertised, _advertised_lock
    if _advertised_lock is None:
        _advertised_lock = asyncio.Lock()
    async with _advertised_lock:
        if _advertised is None:
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, installed_models)
            infos = await loop.run_in_executor(
                None, lambda: {i: em_oww_metadata.resolve(m) for i, m in models.items()})
            # Through the one writer, read-modify-write in one tx, so a name
            # once given is never given again or lost.
            names = await asyncio.wrap_future(em_dbwriter.submit(
                db.assign_wake_word_names, [(i, infos[i].name) for i in models]))
            _advertised = tuple(
                wm.Advertised(i, names[i], tuple(infos[i].languages)) for i in models)
        return _advertised


def mode_of(live, config: dict) -> str:
    """Where this device's wake word is scored. Offline, the configured mode is
    taken against firmware assumed unable to trigger, like effective_mode
    does for a device that has not said."""
    if live is not None:
        return live.oww_on_device
    return em_shadow.effective_mode(config.get("owwOnDevice"), False)


def resolve_sync(device_id: str, config: dict, *, mode: str,
                 device_max: int | None) -> tuple[wm.Resolution, dict[str, str]]:
    """The device's wake words, and the installed models they were chosen from."""
    installed = installed_models()
    res = wm.resolve(
        default_model=(config.get("owwModel") or DEFAULT_MODEL),
        override=db.get_wake_word_override(device_id),
        installed=installed,
        max_n=wm.max_active(mode=mode, device_max=device_max),
    )
    return res, installed


async def resolve(device_id: str, live, config: dict, *,
                  mode: str | None = None) -> wm.Resolution:
    """Resolve off the loop, and record it on the live device if there is one.
    `mode` is the incoming one when a save is changing it, since the maximum
    depends on it and the live mirror is updated after the push."""
    mode = mode or mode_of(live, config)
    res, installed = await asyncio.get_event_loop().run_in_executor(
        None, lambda: resolve_sync(device_id, config, mode=mode,
                                   device_max=getattr(live, "oww_max_models", None)))
    if live is not None:
        set_live(live, res, installed)
        if res.source == "ha" and res.dropped:
            # HA's pickers drop what they do not read back as active, so the
            # stored choice follows them (em_wake_models.settled).
            em_dbwriter.submit(db.set_wake_word_override, device_id, wm.settled(
                res, em_oww_models.prediction_key(config.get("owwModel") or DEFAULT_MODEL)))
    return res


def set_live(live, res: wm.Resolution, installed: dict[str, str]) -> None:
    """The live device's view: what the controller scores, and what HA's
    picker callback resolves against without touching the disk."""
    live.wake_resolution = res
    live.wake_installed = installed
    live.oww_models = res.models


def wanted_models(device_id: str) -> list[str]:
    """The models a device must carry for its wake words, offline or not."""
    cfg = db.get_effective_device_config(device_id) or {}
    res, _ = resolve_sync(device_id, cfg, mode=mode_of(None, cfg), device_max=None)
    return list(res.models)


async def offer_for(res: wm.Resolution) -> wm.Offer:
    """What the satellite tells HA. A configured model that is not installed
    (a custom file since moved) is still offered under its own name, as it
    always was, so HA's picker does not read "No wake word" for it."""
    adv = await advertised()
    known = {a.id for a in adv}
    extra = []
    for i, m in zip(res.active, res.models):
        if i not in known:
            info = await asyncio.get_event_loop().run_in_executor(None, em_oww_metadata.resolve, m)
            extra.append(wm.Advertised(i, info.name, tuple(info.languages)))
    return wm.Offer(advertised=adv + tuple(extra), active=res.active, max_active=res.max)


def _names(device_id: str) -> dict[str, str]:
    import em_esphome
    offer = em_esphome.get_wake_offer(device_id)
    return offer.names() if offer is not None else {}


def view_of(live) -> dict | None:
    """The dashboard's `wakeWords` for a live device, None if not resolved."""
    res = getattr(live, "wake_resolution", None)
    if res is None:
        return None
    return wm.view(res, _names(live.device_id), live.oww_thresholds, live.oww_threshold)


def view_for(device_id: str, live) -> dict | None:
    """`wakeWords` for any device: the live view, or for one that is offline
    what it will listen for when it returns — HA's stored choice included,
    since that is exactly what the fleet view must not hide."""
    if live is not None and getattr(live, "wake_resolution", None) is not None:
        return view_of(live)
    cfg = db.get_effective_device_config(device_id) or {}
    res, _ = resolve_sync(device_id, cfg, mode=mode_of(None, cfg), device_max=None)
    return wm.view(res, _names(device_id), cfg.get("owwThresholds") or {},
                   float(cfg.get("owwThreshold", 0.5)))


async def publish(device_id: str, live, res: wm.Resolution) -> None:
    """Hand a resolution to HA (bouncing it only if what it last read is now
    wrong) and to the dashboard."""
    import em_esphome
    offer = await offer_for(res)
    if live is not None and getattr(live, "wake_resolution", res) is not res:
        return  # a newer resolution was made while this one was being named
    await em_esphome.update_wake_words(device_id, offer)
    if live is not None:
        import em_api
        await em_api._push_event({
            "type": "device_update",
            "device_id": device_id,
            "state": {"wakeWords": view_of(live)},
        })


async def refresh_all(devices: dict) -> None:
    """After an upload or a delete: every device's HA offer is rebuilt."""
    invalidate()
    for device_id, live in list(devices.items()):
        try:
            cfg = await asyncio.get_event_loop().run_in_executor(
                None, db.get_effective_device_config, device_id)
            res = await resolve(device_id, live, cfg or {})
            await publish(device_id, live, res)
        except Exception as e:
            log.warning(f"[{device_id}] wake word refresh failed: {e}")
