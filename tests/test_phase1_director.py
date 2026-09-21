"""Phase 1 director logic: semantic beats, the SOFT fatigue governor, reset-to-base camera grammar,
sound honesty and the primary-visual hierarchy. Planning only -- nothing here renders. Synthetic
content only: no project, brand or reference specifics."""
from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

from tests.conftest import make_transcript
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.direction import (
    Beat,
    BeatKind,
    SemanticKind,
    TreatmentHistory,
    detect_semantic_beats,
    direct,
    to_director_beats,
)
from video_edit_agent.direction.director import LOW_CONFIDENCE
from video_edit_agent.direction.hierarchy import build_hierarchy
from video_edit_agent.direction.history import TIME_CAP, VARIATION_TRIGGER
from video_edit_agent.direction.reset_grammar import never_accumulates
from video_edit_agent.direction.vocabulary import STRONG_PRIMARY
from video_edit_agent.review import edit_plan_direction as ed
from video_edit_agent.review.edit_plan import SlotStatus
from video_edit_agent.sound.profile import get_profile
from video_edit_agent.sound.registry import SfxRegistry

LINES = [
    ("Why does this keep happening?", 0.0, 4.0),
    ("The important thing is that we must plan first.", 4.0, 9.0),
    ("But instead of waiting we act now.", 9.0, 14.0),
    ("First we list the steps, then we test them.", 14.0, 20.0),
    ("That's why the result holds in the end.", 20.0, 26.0),
]
SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"
GENERIC = [
    "direction/semantic_beats.py", "direction/history.py", "direction/reset_grammar.py", "direction/hierarchy.py",
    "direction/director.py", "direction/transitions.py", "review/edit_plan_direction.py",
]


def _transcript():
    return make_transcript(list(LINES))


def _edl() -> EDL:
    clips = [
        EDLClip(source_file="a.mp4", source_in=s, source_out=e, timeline_in=s, timeline_out=e, caption_refs=[f"s{i}"])
        for i, (_, s, e) in enumerate(LINES)
    ]
    return EDL(width=720, height=1280, clips=clips)


def _emphasis(start: float, end: float, **kw) -> Beat:
    return Beat(start=start, end=end, kind=BeatKind.EMPHASIS, importance=0.8, **kw)


def _plain(start: float, end: float, **kw) -> Beat:
    return Beat(start=start, end=end, kind=BeatKind.PLAIN, **kw)


# ---- Delta 1: semantic beats --------------------------------------------------------------
def test_the_approved_transcript_is_never_modified():
    t = _transcript()
    before = t.model_dump_json()
    beats = detect_semantic_beats(t)
    result = direct(to_director_beats(beats))
    assert result.decisions
    assert t.model_dump_json() == before


def test_semantic_beat_detection_is_deterministic():
    a = detect_semantic_beats(_transcript())
    b = detect_semantic_beats(_transcript())
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert {x.kind for x in a} - {SemanticKind.SUPPORT}  # and it actually finds something


def test_beat_timing_stays_inside_the_approved_transcript_timing():
    t = _transcript()
    beats = detect_semantic_beats(t)
    starts = {s.start for s in t.segments} | {w.start for s in t.segments for w in s.words}
    ends = {s.end for s in t.segments} | {w.end for s in t.segments for w in s.words}
    assert beats
    for b in beats:
        assert t.segments[0].start <= b.start < b.end <= t.segments[-1].end
        assert b.start in starts and b.end in ends  # only ever an existing boundary
    for prev, nxt in pairwise(beats):
        assert prev.end <= nxt.start + 1e-6  # tiled, never overlapping


def test_low_confidence_classification_never_forces_a_strong_treatment():
    unsure = _emphasis(0.0, 8.0, semantic_kind=SemanticKind.KEY_CLAIM, semantic_confidence=LOW_CONFIDENCE - 0.1)
    d = direct([unsure]).decisions[0]
    assert d.treatment == "speaker_static" and d.camera.value == "static"
    assert any("low-confidence" in n for n in d.notes)
    # the detector's own fallback: an unsure beat becomes plain support, and plain gets no treatment
    ambiguous = detect_semantic_beats(make_transcript([("we did a thing here", 0.0, 4.0), ("and then it was fine", 4.0, 8.0)]))
    assert all(b.kind is SemanticKind.SUPPORT or b.confidence >= LOW_CONFIDENCE for b in ambiguous)
    for db in to_director_beats([b for b in ambiguous if b.confidence < LOW_CONFIDENCE]):
        assert db.kind is BeatKind.PLAIN


# ---- Delta 2: history + SOFT fatigue ------------------------------------------------------
def test_fatigue_is_a_soft_signal_not_a_timer():
    scores = []
    for since in (5.8, 6.0, 6.2, 6.8, 7.2):
        h = TreatmentHistory(origin=0.0)
        scores.append(h.read(since, "punch_in", semantic_reason=False, alternative_available=True).variation_score)
    assert max(scores) - min(scores) < 0.05  # nothing steps at 6s or 7s
    assert scores == sorted(scores)  # it leans, gently
    h = TreatmentHistory(origin=0.0)
    for since in (10.0, 60.0, 600.0):
        r = h.read(since, "punch_in", semantic_reason=False, alternative_available=True)
        assert not r.variation_desirable  # time alone never raises the flag
        assert r.variation_score <= TIME_CAP + 1e-9 < VARIATION_TRIGGER


def test_a_long_static_state_may_remain_static_when_there_is_no_meaningful_alternative():
    beats = [_plain(i * 20.0, (i + 1) * 20.0) for i in range(4)]  # 80 s of the speaker
    result = direct(beats)
    assert all(d.treatment == "speaker_static" and not d.variation_desirable for d in result.decisions)
    h = TreatmentHistory(origin=0.0)
    r = h.read(90.0, "speaker_static", semantic_reason=True, alternative_available=False)
    assert not r.variation_desirable and "no meaningful alternative" in r.variation_reason


def test_treatment_history_affects_recommendations():
    fresh = TreatmentHistory(origin=0.0)
    used = TreatmentHistory(origin=0.0)
    used.record(0.0, 6.0, "punch_in", "punch_in", True)
    a = fresh.read(8.0, "punch_in", semantic_reason=True, alternative_available=True)
    b = used.read(8.0, "punch_in", semantic_reason=True, alternative_available=True)
    assert b.repetition_count == 1 and a.repetition_count == 0
    assert b.previous_used_strong_emphasis and not a.previous_used_strong_emphasis
    assert b.variation_score < a.variation_score
    assert b.previous_treatment == "punch_in" and b.previous_camera == "punch_in"


def test_a_repeated_punch_in_is_suppressed_when_it_was_used_recently():
    r = direct([_emphasis(0.0, 8.0), _emphasis(8.0, 16.0), _emphasis(16.0, 24.0)])
    treatments = [d.treatment for d in r.decisions]
    assert treatments[0] == "punch_in"
    assert treatments.count("punch_in") < 3  # emphasis does not repeat back to back
    assert any("already used" in n or "previous beat already used" in n for d in r.decisions for n in d.notes)


def test_a_reviewer_choice_is_never_softened_by_repetition():
    def second(**kw):
        return direct([_emphasis(0.0, 8.0), _plain(8.0, 20.0), _emphasis(20.0, 28.0, **kw)]).decisions[2]

    assert second().treatment != "punch_in"  # the planner's own repeat gives way to the speaker
    assert second(prefer="punch_in", pinned=True).treatment == "punch_in"  # yours does not


# ---- Delta 3: reset-to-base ---------------------------------------------------------------
def test_a_punch_or_push_schedules_its_release():
    r = direct([_emphasis(0.0, 8.0), _plain(8.0, 16.0)])
    story = r.decisions[0].camera_story
    assert story is not None and r.decisions[0].camera.value == "punch_in"
    assert story.incoming_state == "base" and story.event == "punch_in"
    assert story.release == "reset_to_base" and story.executed_as
    assert story.sequence[0].startswith("base") and any("reset" in s for s in story.sequence)
    push = direct([Beat(start=0.0, end=12.0, kind=BeatKind.PERSONAL), _plain(12.0, 20.0)]).decisions[0]
    assert push.camera.value == "slow_push" and push.camera_story is not None
    assert push.camera_story.release == "slow_pull"


def test_zoom_state_does_not_accumulate():
    r = direct([_emphasis(0.0, 8.0), _emphasis(14.0, 22.0), _plain(22.0, 30.0), _emphasis(36.0, 44.0)])
    assert never_accumulates(r.camera)
    assert all(d.camera_story is not None and not d.camera_story.accumulates for d in r.decisions)
    assert all(ev.zoom_to <= r.camera.policy.max_zoom + 1e-6 for ev in r.camera.events)


# ---- replacement is judged on meaning, not on elapsed time --------------------------------
def test_speaker_replacement_is_not_triggered_by_elapsed_time_alone():
    beats = [_plain(i * 30.0, (i + 1) * 30.0) for i in range(6)]  # three minutes, nothing to show
    r = direct(beats)
    assert all(d.speaker_visible and d.klass != "replacement" for d in r.decisions)
    h = TreatmentHistory(origin=0.0)
    assert not h.read(180.0, "speaker_static", semantic_reason=False, alternative_available=True).variation_desirable


def test_replacement_density_has_no_hard_global_cap():
    beats = []
    for i in range(3):  # a replacement, the speaker back, a replacement...: about half the time is replaced
        beats += [Beat(start=i * 12.0, end=i * 12.0 + 6.0, kind=BeatKind.CONCEPT), _plain(i * 12.0 + 6.0, i * 12.0 + 12.0)]
    r = direct(beats)
    hidden = sum(d.end - d.start for d in r.decisions if not d.speaker_visible)
    total = sum(d.end - d.start for d in r.decisions)
    assert hidden / total > 0.15  # nothing clips it to a global budget
    # ...and a plan with none is equally valid
    assert not any(not d.speaker_visible for d in direct([_plain(0.0, 20.0)]).decisions)
    for name in ("director.py", "replacement.py"):
        assert not re.search(r"\b0?\.15\b|\b15\s*%", (SRC / "direction" / name).read_text(encoding="utf-8"))


# ---- Delta 4: sound honesty ---------------------------------------------------------------
def test_a_sound_intent_with_an_empty_registry_reports_unavailable_fallback_none():
    beats = [_emphasis(0.0, 8.0), _plain(8.0, 16.0)]
    r = direct(beats, sound_profile=get_profile("minimal"), registry=SfxRegistry())
    assert any(d.sound and d.sound.intent != "none" for d in r.decisions)
    plan = ed.build_directed_plan(beats, r, _transcript(), _edl(), sound_profile="minimal", registry=SfxRegistry())
    audible = [s for s in plan.slots if s.sound_intent not in {"none"}]
    assert audible
    for s in audible:
        assert s.sound_status == "unavailable_fallback_none"
        assert s.sound_status != "scheduled"
        assert s.sound_intent != "none"  # the intent is kept, honestly
        assert "no" in (s.sfx_availability or "").lower() or "unavailable" in (s.sfx_availability or "").lower()


# ---- Delta 5: primary visual hierarchy ----------------------------------------------------
def test_primary_visual_hierarchy_metadata_is_present():
    beats = [_emphasis(0.0, 8.0), Beat(start=8.0, end=14.0, kind=BeatKind.KEY_PHRASE), _plain(14.0, 26.0)]
    r = direct(beats)
    plan = ed.build_directed_plan(beats, r, _transcript(), _edl())
    assert plan.slots
    for s in plan.slots:
        pl = s.planning
        assert pl is not None
        assert pl.primary_layer in {"speaker", "replacement_visual", "headline"}
        assert pl.headline_role in {"none", "primary_headline_typography"}
        assert pl.speaker_visibility in {"full", "partial", "hidden"}
        assert s.caption_behavior in {"normal", "reduced"}
    for d in r.decisions:
        assert d.hierarchy is not None
        assert (d.hierarchy.caption_role == "reduced") == (d.treatment in STRONG_PRIMARY)
        assert not d.hierarchy.duplication_risks()


def test_behind_subject_placement_needs_the_gate_and_is_never_the_default():
    unproven = build_hierarchy("behind_subject_text", True, gate_passed=None)
    failed = build_hierarchy("behind_subject_text", True, gate_passed=False)
    passed = build_hierarchy("behind_subject_text", True, gate_passed=True)
    assert unproven.headline_placement == "top" and failed.headline_placement == "top"
    assert passed.headline_placement == "behind_subject"
    assert build_hierarchy("speaker_static").headline_placement is None


# ---- honesty about a conflict with the reviewer's own answer ------------------------------
def test_an_answered_slot_is_kept_or_flagged_never_silently_overwritten():
    beats = [_emphasis(0.0, 8.0), _plain(8.0, 16.0)]
    first = ed.build_directed_plan(beats, direct(beats), _transcript(), _edl())
    answered = first.slots[0]
    answered.status = SlotStatus.APPROVED
    # same plan again: the approval carries over, no conflict
    again = ed.build_directed_plan(beats, direct(beats), _transcript(), _edl(), decided=[answered])
    assert again.slots[0].status is SlotStatus.APPROVED and not again.slots[0].review_required
    # the planner now disagrees (a plain beat): the answer is flagged, not approved, not dropped
    other = [_plain(0.0, 8.0), _plain(8.0, 16.0)]
    flagged = ed.build_directed_plan(other, direct(other), _transcript(), _edl(), decided=[answered])
    slot = flagged.slots[0]
    assert slot.review_required and slot.status is SlotStatus.PENDING_REVIEW and answered.treatment in slot.review_note


# ---- nothing specific to a reference, a brand or a project is baked in --------------------
def _generic_sources() -> dict[str, str]:
    return {name: (SRC / name).read_text(encoding="utf-8") for name in GENERIC}


def test_no_reference_specific_timestamps_colours_or_text_are_hardcoded():
    for name, text in _generic_sources().items():
        assert not re.search(r"#[0-9a-fA-F]{6}\b", text), f"{name}: a hardcoded colour"
        assert not re.search(r"\breference\s+[ab]\b|\bref_?[ab]\b", text, re.IGNORECASE), f"{name}: names a specific reference"
        assert not re.search(r"\b\d{1,2}:\d{2}\b", text), f"{name}: a hardcoded timestamp"


def test_no_brand_specific_values_in_the_generic_modules():
    for name, text in _generic_sources().items():
        assert "video-ad-editor" not in text and not re.search(r"[A-Za-z]:[\\/]", text), f"{name}: a hardcoded path"
        assert "محمد" not in text and "رفعت" not in text, f"{name}: a person's name"


def test_a_proposal_right_before_the_reviewers_own_camera_move_gives_way_to_it():
    r = direct([_emphasis(0.0, 1.5), _emphasis(1.5, 9.0, prefer="punch_in", pinned=True)])
    first, second = r.decisions
    assert first.treatment == "speaker_static" and any("reviewer's own camera move" in n for n in first.notes)
    assert second.treatment == "punch_in" and second.camera.value == "punch_in"
