"""Sound intent -> profile -> small SFX registry -> deterministic asset -> MasterTimeline.

Every audio file here is a synthetic fixture written at test time; no production SFX exists or is used."""
from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import pytest

import video_edit_agent.sound as sound_pkg
from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.timeline import MasterTimeline, TrackType
from video_edit_agent.sound import mix, planner, resolver
from video_edit_agent.sound.intent import DEFAULT_EVENT_INTENT, EventType, SoundIntent, parse_intent
from video_edit_agent.sound.planner import VisualEvent, plan_sound
from video_edit_agent.sound.profile import PROFILES, get_profile
from video_edit_agent.sound.registry import (
    SfxAsset,
    SfxCategory,
    SfxRegistry,
    default_sfx_dir,
    load_registry,
    load_registry_dir,
)


def _wav(path: Path, seconds: float = 0.5) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * int(8000 * seconds))


def _asset(ident: str, category: SfxCategory, intents: list[SoundIntent], *, energy="low", style="clean",
           events: list[EventType] | None = None, duration=0.5) -> SfxAsset:
    return SfxAsset(id=ident, category=category, file=f"{ident}.wav", energy=energy, duration_s=duration,
                    style=style, allowed_intents=intents, recommended_events=events or [])


@pytest.fixture
def pack(tmp_path: Path) -> SfxRegistry:
    """A synthetic pack covering every intent, with real (silent) files."""
    assets = [
        _asset("whoosh_a", SfxCategory.SOFT_WHOOSH, [SoundIntent.SUBTLE_MOTION], events=[EventType.PUNCH_IN]),
        _asset("sweep_a", SfxCategory.TRANSITION_SWEEP, [SoundIntent.TRANSITION], events=[EventType.REPLACEMENT_IN]),
        _asset("tick_a", SfxCategory.TEXT_TICK_SOFT, [SoundIntent.ACCENT], events=[EventType.KEY_REVEAL], duration=0.2),
        _asset("hit_a", SfxCategory.IMPACT_SOFT, [SoundIntent.IMPACT], duration=0.7),
        _asset("hit_b", SfxCategory.IMPACT_SOFT, [SoundIntent.IMPACT], duration=0.7),
    ]
    for a in assets:
        _wav(tmp_path / a.file, a.duration_s)
    return SfxRegistry(assets=assets, root=str(tmp_path))


def _ev(ident: str, kind: EventType, start: float, **kw) -> VisualEvent:
    return VisualEvent(id=ident, type=kind, start=start, duration=kw.pop("duration", 0.42), **kw)


# 9. sound intent defaults to none ------------------------------------------------------------------------


def test_sound_intent_defaults_to_none():
    assert parse_intent(None) is SoundIntent.NONE
    assert parse_intent("nonsense") is SoundIntent.NONE
    assert Brand(name="x").sound.profile == "none"
    assert get_profile(None).name == "none"
    assert get_profile("unheard-of").silent
    # camera moves other than the first punch carry no sound unless somebody asks
    assert DEFAULT_EVENT_INTENT[EventType.PUNCH_OUT] is SoundIntent.NONE
    assert DEFAULT_EVENT_INTENT[EventType.RESET_TO_BASE] is SoundIntent.NONE


def test_a_silent_profile_never_schedules_anything(pack):
    events = [_ev("a", EventType.REPLACEMENT_IN, 3.0, intent=SoundIntent.IMPACT, importance=1.0)]
    out = plan_sound(events, get_profile("none"), pack)
    assert [d.status for d in out] == [planner.STATUS_NONE]


def test_intent_none_is_honoured_even_when_a_pack_exists(pack):
    out = plan_sound([_ev("a", EventType.KEY_REVEAL, 3.0, intent=SoundIntent.NONE)], get_profile("dynamic"), pack)
    assert out[0].status == planner.STATUS_NONE and out[0].asset_id is None


# 10. minimal is low density -----------------------------------------------------------------------------


def test_minimal_profile_is_low_density(pack):
    minimal, dynamic = get_profile("minimal"), get_profile("dynamic")
    assert minimal.min_spacing_s > dynamic.min_spacing_s
    assert minimal.max_per_minute < dynamic.max_per_minute
    assert minimal.importance_threshold > dynamic.importance_threshold
    assert "cartoon" in minimal.banned_styles and "aggressive_bass" in minimal.banned_styles
    # a busy minute of events: the minimal plan sounds far less often than the dynamic one
    events = [_ev(f"e{i}", EventType.REPLACEMENT_IN, 4.0 * i, importance=0.9) for i in range(15)]
    m = [d for d in plan_sound(events, minimal, pack) if d.status == planner.STATUS_SCHEDULED]
    d = [d for d in plan_sound(events, dynamic, pack) if d.status == planner.STATUS_SCHEDULED]
    assert len(m) <= minimal.max_per_minute < len(d) or len(m) < len(d)
    assert len(m) <= minimal.max_per_minute


def test_low_importance_events_are_dropped_under_minimal(pack):
    out = plan_sound([_ev("a", EventType.PUNCH_IN, 5.0, intent=SoundIntent.SUBTLE_MOTION, importance=0.2)],
                     get_profile("minimal"), pack)
    assert out[0].status == planner.STATUS_SUPPRESSED and out[0].reason == "below_importance_threshold"


# 11. the resolver is deterministic ---------------------------------------------------------------------


def test_resolver_is_deterministic(pack):
    p = get_profile("dynamic")
    picks = {resolver.resolve(p, SoundIntent.IMPACT, EventType.KEY_REVEAL, pack).asset_id for _ in range(20)}
    assert len(picks) == 1
    # a reordered registry still picks the same asset (ties break on category/id, not on order)
    flipped = SfxRegistry(assets=list(reversed(pack.assets)), root=pack.root)
    assert resolver.resolve(p, SoundIntent.IMPACT, EventType.KEY_REVEAL, flipped).asset_id == picks.pop()
    # an asset recommended for the event wins over a generic one
    r = resolver.resolve(p, SoundIntent.TRANSITION, EventType.REPLACEMENT_IN, pack)
    assert r.asset_id == "sweep_a"


def test_resolver_respects_profile_energy_and_banned_styles(tmp_path):
    loud = _asset("loud", SfxCategory.IMPACT_SOFT, [SoundIntent.IMPACT], energy="high")
    cartoon = _asset("boing", SfxCategory.IMPACT_SOFT, [SoundIntent.IMPACT], style="cartoon")
    for a in (loud, cartoon):
        _wav(tmp_path / a.file)
    reg = SfxRegistry(assets=[loud, cartoon], root=str(tmp_path))
    r = resolver.resolve(get_profile("minimal"), SoundIntent.IMPACT, EventType.KEY_REVEAL, reg)
    assert r.status == resolver.UNAVAILABLE and r.reason == resolver.REASON_NO_MATCH


# 12. a missing asset falls back to none without failing the render ------------------------------------------


def test_empty_registry_falls_back_to_none_without_error():
    out = plan_sound([_ev("a", EventType.REPLACEMENT_IN, 4.0, importance=0.9)], get_profile("minimal"), SfxRegistry())
    assert out[0].status == planner.STATUS_UNAVAILABLE
    assert out[0].reason == resolver.REASON_NO_REGISTRY
    assert planner.sfx_availability(out[0]) == "unavailable -> fallback to none"
    tl = MasterTimeline()
    planner.add_to_timeline(tl, out, SfxRegistry())  # never raises
    assert not [i for i in tl.items if i.type is TrackType.SFX]


def test_missing_file_falls_back_to_none(pack, tmp_path):
    (tmp_path / "sweep_a.wav").unlink()
    out = plan_sound([_ev("a", EventType.REPLACEMENT_IN, 4.0, importance=0.9)], get_profile("minimal"), pack)
    assert out[0].status == planner.STATUS_UNAVAILABLE and out[0].reason == resolver.REASON_FILE_MISSING


def test_an_unreadable_registry_file_is_an_empty_registry(tmp_path):
    (tmp_path / "registry.json").write_text("{not json", encoding="utf-8")
    assert load_registry_dir(tmp_path).empty
    assert load_registry(tmp_path / "nowhere").empty


def test_the_shipped_sfx_pack_is_empty_and_never_a_dependency():
    reg = load_registry(default_sfx_dir())
    assert reg.empty, "no production SFX exists; the registry ships empty"
    assert not [p for p in default_sfx_dir().iterdir() if p.suffix.lower() in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}]
    out = plan_sound([_ev("a", EventType.KEY_REVEAL, 4.0, importance=1.0)], get_profile("dynamic"), reg)
    assert out[0].status == planner.STATUS_UNAVAILABLE


# 13. SFX aligns to the visual event's timing -------------------------------------------------------------


def test_sfx_aligns_to_visual_event_timing(pack):
    p = get_profile("dynamic")
    motion = plan_sound([_ev("m", EventType.PUNCH_IN, 10.0, duration=0.42, intent=SoundIntent.SUBTLE_MOTION, importance=0.9)], p, pack)[0]
    assert motion.start == 10.0 and motion.duration <= 0.42 and motion.aligned_to == "motion_interval"
    cut = plan_sound([_ev("c", EventType.REPLACEMENT_IN, 10.0, duration=3.0, importance=0.9)], p, pack)[0]
    assert cut.aligned_to == "cut_lead_in" and cut.start < 10.0 <= cut.start + cut.duration + 0.5
    reveal = plan_sound([_ev("k", EventType.KEY_REVEAL, 10.0, importance=0.9)], p, pack)[0]
    assert reveal.aligned_to == "event_start" and reveal.start == 10.0
    hit = plan_sound([_ev("h", EventType.KEY_REVEAL, 10.0, duration=0.6, intent=SoundIntent.IMPACT, importance=0.9)], p, pack)[0]
    assert hit.aligned_to == "settle" and hit.start == pytest.approx(10.6)


def test_scheduled_sfx_is_written_to_the_master_timeline_with_its_timing(pack):
    events = [_ev("a", EventType.REPLACEMENT_IN, 12.0, duration=2.0, importance=0.9)]
    decisions = plan_sound(events, get_profile("minimal"), pack)
    tl = MasterTimeline()
    planner.add_to_timeline(tl, decisions, pack)
    planner.add_to_timeline(tl, decisions, pack)  # idempotent
    sfx = [i for i in tl.items if i.type is TrackType.SFX]
    assert len(sfx) == 1
    assert sfx[0].start == decisions[0].start and sfx[0].duration == decisions[0].duration
    assert sfx[0].source.endswith("sweep_a.wav")


# 14. repeated decorative SFX are suppressed --------------------------------------------------------------


def test_repeated_decorative_sfx_is_suppressed(pack):
    p = get_profile("minimal")
    events = [_ev("a", EventType.REPLACEMENT_IN, 10.0, importance=0.9), _ev("b", EventType.REPLACEMENT_IN, 14.0, importance=0.9),
              _ev("c", EventType.REPLACEMENT_IN, 40.0, importance=0.9)]
    out = {d.event_id: d for d in plan_sound(events, p, pack)}
    assert out["a"].status == planner.STATUS_SCHEDULED
    assert out["b"].status == planner.STATUS_SUPPRESSED and out["b"].reason == "repeated_event"
    assert out["c"].status == planner.STATUS_SCHEDULED  # far enough away to be a fresh moment


def test_sounds_never_stack_on_one_beat(pack):
    events = [_ev("a", EventType.REPLACEMENT_IN, 10.0, importance=0.9, user_locked=True),
              _ev("b", EventType.KEY_REVEAL, 10.2, importance=0.9, user_locked=True)]
    out = {d.event_id: d for d in plan_sound(events, get_profile("dynamic"), pack)}
    assert out["b"].status == planner.STATUS_SUPPRESSED and out["b"].reason == "stacked_on_the_same_beat"


def test_minimum_spacing_between_different_sounds(pack):
    events = [_ev("a", EventType.REPLACEMENT_IN, 10.0, importance=0.9), _ev("b", EventType.KEY_REVEAL, 12.0, importance=0.9)]
    out = {d.event_id: d for d in plan_sound(events, get_profile("minimal"), pack)}
    assert out["b"].status == planner.STATUS_SUPPRESSED and out["b"].reason == "min_spacing"


def test_planning_is_deterministic(pack):
    events = [_ev(f"e{i}", EventType.REPLACEMENT_IN, 5.0 * i, importance=0.9) for i in range(8)]
    a = [d.model_dump() for d in plan_sound(events, get_profile("dynamic"), pack)]
    b = [d.model_dump() for d in plan_sound(list(reversed(events)), get_profile("dynamic"), pack)]
    assert a == b


# 15. voice has the highest mix priority ---------------------------------------------------------------


def test_voice_has_the_highest_mix_priority():
    assert mix.MIX_PRIORITY == ("voice", "content_audio", "music", "sfx")
    assert mix.priority("voice") < mix.priority("content_audio") < mix.priority("music") < mix.priority("sfx")
    for name, profile in PROFILES.items():
        for intent in SoundIntent:
            if intent is SoundIntent.NONE:
                continue
            gain = mix.sfx_gain_db(profile, intent)
            assert gain <= mix.VOICE_GAIN_DB - mix.MIN_HEADROOM_DB, (name, intent)
    assert mix.voice_protected([-20.0, -14.0])
    assert not mix.voice_protected([-3.0])


def test_the_sfx_filter_is_trimmed_faded_and_delayed():
    f = mix.sfx_filter(2, start_s=1.5, duration_s=0.4, gain_db=-20.0, label="s2")
    assert "atrim=0:0.400" in f and "afade=t=in" in f and "afade=t=out" in f
    assert "volume=-20.0dB" in f and "adelay=1500|1500" in f and f.endswith("[s2]")


# 16. no brand-specific SFX is required ---------------------------------------------------------------


def test_no_brand_specific_sfx_is_required(pack):
    brand = Brand(name="anything")
    assert brand.sound.profile == "none"  # a brand carries a profile NAME only
    assert set(Brand.model_fields["sound"].annotation.model_fields) == {"profile"}
    # the same registry serves every profile; nothing is keyed by brand
    for name in ("minimal", "dynamic"):
        out = plan_sound([_ev("a", EventType.REPLACEMENT_IN, 10.0, importance=0.9)], get_profile(name), pack)
        assert out[0].status == planner.STATUS_SCHEDULED
    # an optional brand drop-in is merged in, but is never required
    root = Path(str(pack.root))
    assert load_registry(root / "empty", brand_dir=root / "no-such-brand").empty
    drop = root / "brand" / "sfx"
    drop.mkdir(parents=True)
    (drop / "registry.json").write_text(json.dumps({"assets": [pack.assets[0].model_dump(mode="json")]}), encoding="utf-8")
    assert [a.id for a in load_registry(root / "empty", brand_dir=root / "brand").assets] == ["whoosh_a"]


# 17. no client hardcoding in the generic modules -------------------------------------------------------------

_SRC = Path(sound_pkg.__file__).parents[1]
GENERIC = [
    "sound/intent.py", "sound/profile.py", "sound/registry.py", "sound/resolver.py", "sound/mix.py",
    "sound/planner.py", "sound/__init__.py", "direction/vocabulary.py", "direction/replacement.py",
    "direction/camera.py", "direction/transitions.py", "direction/speed.py", "direction/suitability.py",
    "direction/director.py", "direction/__init__.py", "captions/modes.py", "review/edit_plan_direction.py",
]
FORBIDDEN = [
    r"#0a3d62", r"#8e44ad", r"#f1c40f", r"video-ad-editor", r"LI-SEP", r"\b38\.94\b",
    r"\b45\.38\b", r"\b52\.06\b", r"\b58\.10?\b", r"\b64\.96\b",
]


@pytest.mark.parametrize("rel", GENERIC)
def test_no_brand_or_video_specific_values_in_the_generic_modules(rel: str):
    src = (_SRC / rel).read_text(encoding="utf-8")
    hits = [pat for pat in FORBIDDEN if re.search(pat, src, flags=re.IGNORECASE)]
    assert hits == [], f"{rel} hardcodes {hits}"
