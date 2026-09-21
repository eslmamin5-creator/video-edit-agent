"""Phase 1.2: the Visual Rhythm Engine and the Semantic Enhancement Director. Planning only -- nothing here
renders. Synthetic content only: no project, brand or reference specifics."""
from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest

from tests.conftest import make_transcript, make_word
from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript
from video_edit_agent.direction import (
    Beat,
    BeatKind,
    BehindSubjectEvidence,
    RhythmPolicy,
    assess_behind_subject,
    behind_subject_eligibility,
    build_hierarchy,
    direct,
    plan_rhythm,
    semantic_enhancement_options,
)
from video_edit_agent.direction.reset_grammar import never_accumulates
from video_edit_agent.direction.rhythm import (
    EXCURSIONS,
    SAFE_QUALITY,
    SEMANTIC_ENHANCEMENTS,
    VOCABULARY,
    Occupied,
    SemanticHint,
    entries,
    find_boundaries,
    headline_allowed,
    inside_word,
)
from video_edit_agent.review import edit_plan_direction as ed
from video_edit_agent.review.edit_plan_chat import format_rhythm_map
from video_edit_agent.review.edit_plan_rhythm import attach_rhythm

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"
TEXTS = [
    "one two three four five six seven eight", "we should plan first then act", "but instead of waiting we go now",
    "the result holds in the end", "so keep it simple and steady",
]
FULL = BehindSubjectEvidence(
    mask_quality=True, readable_occlusion=True, phrase_timing=True, shot_composition=True, caption_hierarchy=True,
    visual_value=0.9, head_hair_integrity=True, meaningful_occlusion=True,
)
EXCURSION_NAMES = {s.value for s in EXCURSIONS}


def _long(n: int = 24, seg: float = 3.4) -> Transcript:
    return make_transcript([(TEXTS[i % len(TEXTS)], i * seg, (i + 1) * seg) for i in range(n)])


def _gappy() -> Transcript:
    """Words with real silences between them (so a boundary could fall inside a word if the engine were careless)."""
    segs, t = [], 0.0
    for i in range(18):
        words = []
        for j, w in enumerate(TEXTS[i % len(TEXTS)].split()):
            words.append(make_word(w, t, t + 0.31))
            t += 0.31 + (0.42 if j == 2 else 0.04)
        segs.append(Segment(id=f"s{i}", start=words[0].start, end=words[-1].end, text=" ".join(w.word for w in words), words=words))
        t += 0.1
    return Transcript(provider="test", language="auto", duration=t, segments=segs)


def _no_boundaries(seconds: float = 30.0) -> Transcript:
    """One long unbroken run of speech: not a single safe place to move."""
    n = int(seconds / 0.5)
    words = [make_word(f"alpha{i}", i * 0.5, (i + 1) * 0.5) for i in range(n)]
    return Transcript(provider="test", language="auto", duration=seconds,
                      segments=[Segment(id="s0", start=0.0, end=seconds, text=" ".join(w.word for w in words), words=words)])


def _lead_rows(plan):
    return [g[0] for g in entries(plan.rows) if g[0].source == "rhythm" and g[0].state in EXCURSION_NAMES]


def _edl(t: Transcript) -> EDL:
    clips = [EDLClip(source_file="a.mp4", source_in=s.start, source_out=s.end, timeline_in=s.start, timeline_out=s.end,
                     caption_refs=[s.id]) for s in t.segments]
    return EDL(width=720, height=1280, clips=clips)


def _plan(t: Transcript):
    beats = [Beat(start=0.0, end=t.duration, kind=BeatKind.PLAIN)]
    plan = ed.build_directed_plan(beats, direct(beats), t, _edl(t))
    rhythm = attach_rhythm(plan, t, _edl(t), start=0.0)
    return plan, rhythm


# ---- A. the rhythm engine: low semantic dependency ----------------------------------------
def test_1_rhythm_works_with_low_semantic_confidence():
    t = _long()
    weak = [SemanticHint(start=0.0, end=t.duration, kind="key_claim", confidence=0.1)]
    assert len(plan_rhythm(t, hints=weak).excursions()) >= 4


def test_2_camera_variation_does_not_need_semantic_confidence():
    plan = plan_rhythm(_long())  # no semantic reading at all
    assert len(set(plan.excursions())) >= 3
    assert plan.to_camera_plan().events


def test_3_same_input_gives_the_same_sequence():
    assert plan_rhythm(_long()).model_dump() == plan_rhythm(_long()).model_dump()


def test_4_no_random_movement():
    text = (SRC / "direction" / "rhythm.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+random\b|\bnumpy\.random\b|\buuid\b|\btime\.time\(", text, re.MULTILINE)


def test_5_events_snap_to_safe_phrase_boundaries():
    t = _long()
    safe = {b.t for b in find_boundaries(t) if b.quality >= SAFE_QUALITY}
    leads = _lead_rows(plan_rhythm(t))
    assert leads and all(round(r.start, 3) in safe for r in leads)
    assert all(r.boundary_quality is not None and r.boundary_quality >= SAFE_QUALITY for r in leads)


def test_6_no_event_starts_mid_word():
    t = _gappy()
    plan = plan_rhythm(t)
    assert plan.excursions()
    events = [r for r in plan.rows if r.state not in ("hold", "base")]  # a hold/base row merely continues after an ease
    assert events and not [r.start for r in events if inside_word(r.start, t)]
    assert not [b.t for b in find_boundaries(t) if inside_word(b.t, t)]


def test_7_the_refresh_window_is_soft_not_exact():
    plan = plan_rhythm(_long())
    gaps = [round(b - a, 1) for a, b in pairwise(sorted({*plan.changes}))]
    assert len(set(gaps)) > 2  # not a metronome
    # with no phrase boundary at all nothing abrupt happens; only a smooth move may end the hold (Phase 1.3)
    long_run = plan_rhythm(_no_boundaries())
    assert not {"punch_in"} & set(long_run.excursions())
    assert all(r.motion_class == "smooth" for r in long_run.rows if r.moving)


def test_8_time_alone_cannot_create_semantic_graphics():
    assert semantic_enhancement_options(None, None) == []
    assert semantic_enhancement_options("key_claim", None, phrase="only this") == []
    plan, _rhythm = _plan(_long(60))  # a long stretch of speech with no semantic reading whatsoever
    assert plan.rhythm
    assert all(r.semantic_enhancement == "none" and r.semantic_options == [] for r in plan.rhythm)
    assert {r.state for r in plan.rhythm} <= set(VOCABULARY)
    assert all(s.treatment == "stay_on_speaker" for s in plan.slots)


def test_9_a_visual_hold_plus_a_safe_boundary_may_trigger_low_risk_variation():
    plan = plan_rhythm(_long(8))
    lead = _lead_rows(plan)[0]
    assert lead.state in {"slow_push", "reframe_left", "reframe_right", "lower_subject"}  # low-risk, not a punch
    assert plan.decisions[0].hold_before >= RhythmPolicy().min_hold_s


# ---- anti-pattern memory ------------------------------------------------------------------
def test_10_the_last_three_states_affect_selection():
    exc = plan_rhythm(_long()).excursions()
    assert len(set(exc)) >= 4
    for i in range(3, len(exc)):
        assert exc[i] not in exc[i - 3:i]


def test_11_repeated_punch_in_is_suppressed():
    t = _long()
    exc = plan_rhythm(t).excursions()
    assert exc.count("punch_in") <= max(1, len(exc) // 4)
    assert all(a != b for a, b in pairwise(exc))
    pinned = plan_rhythm(t, occupied=[Occupied(start=6.0, end=12.0, label="punch_in", pinned=True, changes=(6.0, 10.0))])
    assert pinned.excursions()[0] != "punch_in"  # the user's own punch is remembered


def test_12_anti_pattern_prevention_is_deterministic():
    runs = [plan_rhythm(_long()).excursions() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert not any(a == b == c for a, b, c in zip(runs[0], runs[0][1:], runs[0][2:], strict=False))


def test_13_reset_to_base_occurs_after_emphasis():
    plan = plan_rhythm(_long())
    groups = [g for g in entries(plan.rows) if g[0].source == "rhythm" and g[0].state in EXCURSION_NAMES]
    assert groups
    for g in groups:
        assert g[-1].state in ("reset_to_base", "slow_pull")
        assert (g[-1].zoom_to, g[-1].anchor_x) == (1.0, 0.5)
        assert g[-1].end - g[0].start >= RhythmPolicy().excursion_min_s - 1e-6


def test_14_zoom_values_do_not_accumulate():
    plan = plan_rhythm(_long(40))
    assert never_accumulates(plan.to_camera_plan())
    levels = {1.0, *(RhythmPolicy().level(s) for s in EXCURSIONS)}
    assert max(r.zoom_to for r in plan.rows) <= RhythmPolicy().max_zoom
    assert {round(r.zoom_from, 4) for r in plan.rows} <= levels
    assert all(r.zoom_to == 1.0 for r in plan.rows if r.state == "reset_to_base")


# ---- lower_subject ------------------------------------------------------------------------
def _lower_row(face):
    plan = plan_rhythm(_long(40), face_box=face)
    return plan, next((r for r in plan.rows if r.state == "lower_subject"), None)


def test_15_lower_subject_creates_safe_top_space():
    plan, row = _lower_row((0.30, 0.30, 0.40, 0.15))  # a measured face with room to lower
    assert row is not None and row.anchor_y == 0.0 and row.zoom_to == RhythmPolicy().lower_zoom
    assert re.search(r"top space|headroom", row.composition_reason, re.IGNORECASE)
    assert row.executable  # a measured face makes it executable (Phase 1.3)
    assert plan.to_camera_plan(face_box=(0.30, 0.30, 0.40, 0.15)).events
    assert all(e.anchor_x is not None for e in plan.to_camera_plan(face_box=(0.30, 0.30, 0.40, 0.15)).events)
    # a face that would end up under the caption band is never lowered
    assert "lower_subject" not in plan_rhythm(_long(40), face_box=(0.30, 0.55, 0.40, 0.40)).excursions()


def test_16_lower_subject_does_not_require_headline_text():
    _, row = _lower_row(None)
    assert row is not None  # planned even with nothing to say
    assert row.semantic_enhancement == "none" and row.primary_layer == "speaker"
    assert not headline_allowed("lower_subject", None, None, None)
    assert not headline_allowed("lower_subject", "key_claim", 0.2, "two words")


# ---- B. the semantic side -----------------------------------------------------------------
def test_17_a_semantic_headline_requires_semantic_evidence():
    assert "primary_headline_typography" in semantic_enhancement_options("key_claim", 0.8, phrase="keep it simple")
    assert "primary_headline_typography" not in semantic_enhancement_options("key_claim", 0.3, phrase="keep it simple")
    assert "primary_headline_typography" not in semantic_enhancement_options("support", 0.9, phrase="keep it simple")
    assert "primary_headline_typography" not in semantic_enhancement_options("key_claim", 0.9, phrase="a phrase far too long to headline")
    assert set(semantic_enhancement_options("key_claim", 0.9, phrase="keep it simple")) <= set(SEMANTIC_ENHANCEMENTS)


# ---- C. Behind-Subject stays first-class, gated, fail-closed -------------------------------
def _eligible(**over):
    args = {"phrase": "keep it simple", "semantic_kind": "key_claim", "semantic_confidence": 0.8, "evidence": FULL,
            "rhythm_state": "base", **over}
    return behind_subject_eligibility(**args)


def test_18_behind_subject_remains_available():
    verdict = _eligible()
    assert verdict.status == "eligible" and verdict.requires_approval  # offered, never auto-approved
    assert "behind_subject_text" in SEMANTIC_ENHANCEMENTS
    assert build_hierarchy("behind_subject_text", gate_passed=True).headline_placement == "behind_subject"


def test_19_behind_subject_requires_the_suitability_gate():
    assert _eligible(evidence=None).status == "not_eligible"
    assert _eligible(evidence=BehindSubjectEvidence()).unproven
    for missing in ("caption_hierarchy", "phrase_timing", "shot_composition", "readable_occlusion", "visual_value"):
        assert _eligible(evidence=FULL.model_copy(update={missing: None})).status == "not_eligible", missing
    assert _eligible(phrase=None).status == "not_applicable"
    assert _eligible(phrase="this phrase has far too many words").status == "not_eligible"
    assert _eligible(rhythm_state="punch_in").status == "not_eligible"
    assert _eligible(caption_competing=True).status == "not_eligible"
    assert _eligible(better_simpler_treatment=True).status == "not_eligible"
    assert build_hierarchy("behind_subject_text", gate_passed=False).headline_placement != "behind_subject"


def test_20_behind_subject_cannot_bypass_subject_integrity():
    for check in ("mask_quality", "head_hair_integrity", "meaningful_occlusion"):
        for value in (False, None):
            assert _eligible(evidence=FULL.model_copy(update={check: value})).status == "not_eligible", (check, value)
    assert not assess_behind_subject(FULL.model_copy(update={"head_hair_integrity": False})).suitable
    assert assess_behind_subject(FULL).suitable


def test_21_behind_subject_reduces_caption_hierarchy_when_active():
    h = build_hierarchy("behind_subject_text", gate_passed=True)
    assert h.caption_role == "reduced" and h.primary_layer == "headline" and not h.duplication_risks()


def test_22_low_semantic_confidence_cannot_trigger_behind_subject():
    assert _eligible(semantic_confidence=0.2).status == "not_eligible"
    assert _eligible(semantic_confidence=None, semantic_kind=None).status == "not_eligible"
    t = _long()
    weak = [SemanticHint(start=0.0, end=t.duration, kind="key_claim", confidence=0.2)]
    assert not [r for r in plan_rhythm(t, hints=weak).rows if r.semantic_enhancement == "behind_subject_text"]
    filled, _ = _plan(t)
    assert not [r for r in filled.rhythm if r.behind_subject == "eligible"]  # nothing measured, so nothing eligible


# ---- what is never automatic ---------------------------------------------------------------
def test_23_no_automatic_speaker_replacement():
    plan, rhythm = _plan(_long(60))
    assert all(s.speaker_visible for s in plan.slots)
    assert all(r.speaker_visibility == "full" for r in plan.rhythm)
    for kind in ("key_claim", "process_list"):
        assert "speaker_replacement" not in semantic_enhancement_options(kind, 1.0, phrase="keep it")
    assert not set(rhythm.excursions()) & {"speaker_replacement", "full_screen_graphic", "real_broll", "user_broll"}


def test_24_no_automatic_generated_visual():
    for kind in ("key_claim", "payoff", "contrast", "question", "topic_shift", "process_list", "support"):
        assert "generated_visual" not in semantic_enhancement_options(kind, 1.0, phrase="keep it simple")
    plan, _ = _plan(_long(60))
    assert not any("generated" in s.treatment for s in plan.slots)
    assert not any(r.semantic_enhancement == "generated_visual" for r in plan.rhythm)


# ---- generic, deterministic, reviewable ----------------------------------------------------
GENERIC = ["direction/rhythm.py", "review/edit_plan_rhythm.py", "direction/suitability.py"]


def test_25_no_hard_reference_timestamps():
    for name in GENERIC:
        text = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"\b\d{1,2}:\d{2}\b", text), name
        assert not re.search(r"\b\d{2,3}\.\d{2}\b", text), f"{name}: a hardcoded second mark"
    policy = RhythmPolicy()
    assert policy.refresh_min_s < policy.refresh_max_s < policy.attention_guard_s  # soft targets, not exact intervals


def test_26_no_project_specific_content_in_the_generic_modules():
    for name in GENERIC:
        text = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"#[0-9a-fA-F]{6}\b", text), name
        assert "video-ad-editor" not in text and not re.search(r"[A-Za-z]:[\\/]", text), name
        assert "محمد" not in text and "رفعت" not in text, name


# ---- review map ---------------------------------------------------------------------------
def test_the_review_map_carries_every_required_field_and_never_blocks_review():
    plan, _ = _plan(_long())
    assert plan.rhythm
    row = next(r for r in plan.rhythm if r.state != "base")
    for field in ("start", "end", "transcript_context", "boundary_kind", "boundary_after", "state", "semantic_enhancement",
                  "history_reason", "composition_reason", "reset_plan", "primary_layer", "caption_role", "speaker_visibility",
                  "behind_subject", "sound_intent", "sound_status", "approval_status"):
        assert getattr(row, field) not in (None, ""), field
    assert row.behind_subject in ("eligible", "not_eligible", "not_applicable")
    text = format_rhythm_map(plan)
    for label in ("Spoken:", "Phrase boundary:", "Why now:", "Composition:", "Return:", "Semantic:", "Primary:", "Behind-subject:", "Sound:"):
        assert label in text, label
    assert not [s for s in plan.slots if s.number and s.status.value == "pending_review" and s.treatment == "stay_on_speaker"]


def test_a_director_or_user_window_is_left_alone_and_the_hold_is_reported_honestly():
    occupied = [Occupied(start=10.0, end=17.0, label="punch_in", pinned=True, changes=(10.0, 14.0))]
    plan = plan_rhythm(_long(), occupied=occupied)
    assert not [r for r in plan.rows if r.source == "rhythm" and r.state != "base" and r.start < 17.0 - 1e-6 and r.end > 8.8]
    seconds, a, b = plan.longest_unchanged_hold()
    assert seconds == pytest.approx(b - a)
