"""
em_wake_models.py — more than one wake word per Echo (#112)

Home Assistant's ESPHome integration gives every voice satellite two wake
word pickers ("Wake word", "Wake word 2"), each beside its own pipeline
picker, and routes a run to the pipeline whose wake word picker reads the
same as the `wake_word_phrase` the satellite sends. So two wake words with
two agents needs only three things from us: advertise every model, honour
the list HA picks, and name the one that fired.

Who owns what (the split proposed on #112):

- The DASHBOARD owns which models are installed and the default
  (`owwModel`).
- HOME ASSISTANT owns which advertised models are active on each Echo. Its
  choice is stored apart from device config (em_db, `wake_words_ha`), the
  same reason `wake_word_off` is: a dashboard save would write back a stale
  copy. HA has no "follow the dashboard" option, so choosing exactly the
  dashboard's model clears the choice and the Echo follows the fleet again.
- The dashboard SHOWS an HA choice, so the fleet view never claims a device
  is doing something it is not (`view`).

Facts read from Home Assistant's source (dev, 2026-10-13), each one a rule
here:

- The picker's options are a dict keyed by display name
  (esphome/select.py: `{w.wake_word: w.id}`), so two models with the same
  name collapse into one option. Names are made unique, and HA restores a
  picker by NAME after a restart, so a name must also never change once HA
  has seen it: `assign_names`.
- HA sends the union of both pickers and does not cap it by
  `max_active_wake_words` on that path (only its websocket setup API
  checks). The cap is applied here: `resolve`.
- HA hides both pickers when `max_active_wake_words` is below 1, so the
  maximum is never reported as 0: `max_active`.
- Nothing in HA filters the picker by `trained_languages`; the languages
  are carried through and nothing more.

Pure: no I/O, no aiohttp, no db. em_wake_sync does the reading and writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import em_oww_models

# Models scored on the controller share openWakeWord's preprocessor and
# embedding (one `self.preprocessor` feeding a loop over `self.models` in
# openwakeword/model.py), so each extra model costs one small classifier head.
# Home Assistant can express two; four is what this side will honour.
CONTROLLER_MAX = 4

# Why a requested model is not active, as the dashboard reports it.
NOT_INSTALLED = "not_installed"
OVER_MAX = "over_max"


@dataclass(frozen=True)
class Advertised:
    """One wake word as Home Assistant sees it."""
    id: str
    name: str
    languages: tuple[str, ...] = ("en",)


@dataclass(frozen=True)
class Offer:
    """What a satellite tells Home Assistant: every model, the active ones,
    and how many may be active. Compared whole to decide whether HA has to
    be made to read it again."""
    advertised: tuple[Advertised, ...] = ()
    active: tuple[str, ...] = ()
    max_active: int = 1

    def names(self) -> dict[str, str]:
        return {a.id: a.name for a in self.advertised}

    def ids(self) -> set[str]:
        return {a.id for a in self.advertised}


@dataclass(frozen=True)
class Resolution:
    """Which models one Echo is listening for, and why."""
    # Prediction keys, in HA's order; never empty.
    active: tuple[str, ...]
    # The owwModel value for each active id (a stock name or a file path).
    models: tuple[str, ...]
    # "ha" when Home Assistant's choice is stored, else "dashboard".
    source: str
    # What was asked for: HA's stored choice, or the dashboard's model.
    requested: tuple[str, ...]
    # (id, reason) for each requested id that is not active.
    dropped: tuple[tuple[str, str], ...] = field(default=())
    max: int = 1


def assign_names(registry: dict, candidates: list[tuple[str, str]]) -> tuple[dict, dict]:
    """
    Unique display names that never move once assigned.

    `candidates` is [(id, base phrase)] for every installed model, stock models
    first. `registry` is what was assigned before, {id: {"base", "name"}}.
    Returns ({id: name}, new registry).

    A model already registered keeps its name while its base phrase is
    unchanged — whatever is installed later, and in whatever order. A newcomer
    takes its base phrase, or "<base> (<id>)" if that name is taken, so the
    suffix always lands on the newcomer. Sorting instead would let an upload
    that sorts earlier rename an existing model, and Home Assistant, which
    restores its pickers by name, would then silently show "No wake word".

    A model whose base phrase changed (re-trained with a new phrase) is
    renamed: that is its owner's change. Models no longer installed drop out.
    """
    registry = registry if isinstance(registry, dict) else {}
    wanted = {i: b for i, b in candidates}
    kept: dict[str, dict] = {}
    for i, entry in registry.items():
        if (i in wanted and isinstance(entry, dict)
                and entry.get("base") == wanted[i]
                and isinstance(entry.get("name"), str) and entry["name"]):
            kept[i] = {"base": wanted[i], "name": entry["name"]}
    taken = {e["name"] for e in kept.values()}
    out = dict(kept)
    for i, base in candidates:
        if i in out:
            continue
        name = base
        if name in taken:
            name = f"{base} ({i})"
            n = 2
            while name in taken:  # only if another model is literally named so
                name = f"{base} ({i} {n})"
                n += 1
        taken.add(name)
        out[i] = {"base": base, "name": name}
    names = {i: out[i]["name"] for i, _ in candidates}
    return names, {i: out[i] for i, _ in candidates}


def max_active(*, mode: str, device_max: int | None) -> int:
    """
    How many wake words this Echo can really listen for at once.

    An Echo detecting its own wake word (`on`) can run what its firmware says
    it can, and firmware that says nothing runs one. Anywhere else the
    controller scores, and its limit applies. Never below 1: Home Assistant
    hides the pickers at 0.
    """
    if mode == "on":
        try:
            return max(1, int(device_max or 1))
        except (TypeError, ValueError):
            return 1
    return CONTROLLER_MAX


def override_from_request(requested: list[str], *, available: set[str],
                          default_id: str) -> list[str] | None:
    """
    What to store for Home Assistant's choice. None clears it.

    Duplicates go (both pickers on one model), unknown ids go, order is kept.
    Exactly the dashboard's model means "follow the dashboard": HA has no
    other way to say it. The caller handles [] ("No wake word"), which is
    em_wakeword's, never this.
    """
    ids: list[str] = []
    for r in requested:
        if r in available and r not in ids:
            ids.append(r)
    if not ids or ids == [default_id]:
        return None
    return ids


def resolve(*, default_model: str, override: list[str] | None,
            installed: dict[str, str], max_n: int) -> Resolution:
    """
    The models an Echo listens for.

    `installed` maps prediction key -> owwModel value. The dashboard's model
    is always available under its own id, as configured, even if a stock
    model shares the stem. A stored choice naming a model that has since gone
    is reported as dropped, and the Echo falls back to the dashboard's model
    rather than to nothing.
    """
    default_id = em_oww_models.prediction_key(default_model)
    by_id = dict(installed)
    by_id[default_id] = default_model
    if override:
        requested, source = tuple(override), "ha"
    else:
        requested, source = (default_id,), "dashboard"
    n = max(1, int(max_n))
    active: list[str] = []
    dropped: list[tuple[str, str]] = []
    for i in requested:
        if i not in by_id:
            dropped.append((i, NOT_INSTALLED))
        elif len(active) >= n:
            dropped.append((i, OVER_MAX))
        else:
            active.append(i)
    if not active:
        active = [default_id]
    return Resolution(
        active=tuple(active),
        models=tuple(by_id[i] for i in active),
        source=source,
        requested=requested,
        dropped=tuple(dropped),
        max=n,
    )


def settled(res: Resolution, default_id: str) -> list[str] | None:
    """
    The stored choice once Home Assistant has read `res` back.

    HA's picker resets itself to "No wake word" when the wake word it shows is
    not among the active ones it reads (esphome/select.py,
    async_satellite_config_updated). So a choice over this Echo's maximum, or
    naming a model since removed, does not survive in HA, and must not survive
    here: kept, a later rise in the maximum (a change of where the wake word
    is detected) would make the Echo listen for a word HA's picker no longer
    shows, whose phrase then matches no pipeline and reaches the wrong
    assistant. Exactly the dashboard's model is no choice, as ever.
    """
    if res.source != "ha":
        return None
    ids = list(res.active)
    return None if ids == [default_id] else ids


def bars(ids, default: float, lowered_to: float | None = None) -> dict[str, float]:
    """Each model's bar for this frame. `lowered_to` is the barge-in bar while
    our own audio is playing; it only ever lowers a model's bar."""
    t = float(default)
    return {i: (min(t, lowered_to) if lowered_to is not None else t) for i in ids}


def judge(scores: dict[str, float], frame_bars: dict[str, float]) -> tuple[str, float, float]:
    """
    (model, score, bar) for one frame.

    Each model is judged against its bar. Of the models that crossed, the
    highest score wins. When none crossed, the one closest to its bar is
    reported, which is what the near-miss counters want. With one model both
    reduce to that model, so single-wake-word behaviour is unchanged.
    """
    crossed = [k for k, s in scores.items() if s >= frame_bars[k]]
    if crossed:
        k = max(crossed, key=lambda k: scores[k])
    else:
        k = max(scores, key=lambda k: scores[k] / frame_bars[k] if frame_bars[k] > 0 else scores[k])
    return k, scores[k], frame_bars[k]


def phrase(fired_id: str | None, names: dict[str, str], active) -> str:
    """
    The `wake_word_phrase` for a wake: the advertised name of the model that
    fired, which is what Home Assistant matches against each picker to choose
    a pipeline. A wake that names no model (firmware that does not report it)
    is the first active model's, which is the single-wake-word behaviour.
    """
    key = fired_id or (active[0] if active else "")
    if key in names:
        return names[key]
    return em_oww_models.display_name(key) if key else ""


def for_device(config: dict, resolution: Resolution) -> dict:
    """
    The config to push to firmware that runs ONE model.

    Such an Echo only reads `owwModel`, so it is sent the first wake word this
    device listens for. Without this an Echo detecting its own wake word would
    ignore Home Assistant's choice entirely.
    """
    model = resolution.models[0]
    if config.get("owwModel") == model:
        return config
    out = dict(config)
    out["owwModel"] = model
    return out


def view(resolution: Resolution, names: dict[str, str]) -> dict:
    """What the dashboard shows for one Echo (`wakeWords` in /api/devices)."""
    def _name(i: str) -> str:
        return names.get(i) or em_oww_models.display_name(i)

    return {
        "source": resolution.source,
        "active": [{"id": i, "name": _name(i)} for i in resolution.active],
        "dropped": [{"id": i, "name": _name(i), "reason": r}
                    for i, r in resolution.dropped],
        "max": resolution.max,
    }
