"""Phase 1.3: visual rhythm becomes executable. One canonical camera timeline, abrupt vs smooth boundaries, smooth returns,
lower_subject geometry, user-pinned Behind-Subject evidence. Synthetic content only: no project, brand or reference specifics."""
from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest

from tests.test_phase12_rhythm import FULL, _edl, _gappy, _lead_rows, _long, _no_boundaries, _plan
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.direction import RhythmPolicy, behind_subject_eligibility, plan_rhythm
from video_edit_agent.direction.camera import (
    BASE_ANCHOR_X,
    BASE_ZOOM,
    CameraEvent,
    CameraMove,
    EventStatus,
    MotionClass,
)
from video_edit_agent.direction.camera_timeline import (
    BASE_ANCHOR_Y,
    EXECUTABLE_MOVES,
    apply_timeline,
    build_camera_timeline,
    legacy_events,
    review_mismatches,
    sample_geometry,
    static_claims,
)
from video_edit_agent.direction.reset_grammar import never_accumulates
from video_edit_agent.direction.rhythm import (
    SAFE_QUALITY,
    Boundary,
    RhythmPlan,
    RhythmState,
    _rows_for,
    find_boundaries,
    inside_word,
)
from video_edit_agent.review.edit_plan import EditPlanSlot
from video_edit_agent.review.edit_plan_rhythm import attach_rhythm

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"
FACE = (0.30, 0.30, 0.40, 0.15)  # a measured face with room to lower (x, y, w, h, normalised)
S = RhythmState


def _hand(state: S, t: float = 4.0, ret: float = 8.0, *, face=FACE, end: float = 20.0) -> RhythmPlan:
    """A one-excursion plan built by the engine's own row builder, so the timeline sees exactly what the engine emits."""
    policy = RhythmPolicy()
    rows = _rows_for(state, t, Boundary(t=ret + 1.0, kind="clause", quality=0.65, after="x"), ret, policy, face)
    for r in rows:
        r.number = 1
    return RhythmPlan(rows=rows, policy=policy, start=0.0, end=end)


def _uniform_edl(seconds: float = 20.0, step: float = 2.0) -> EDL:
    n = int(seconds / step)
    clips = [EDLClip(source_file="a.mp4", source_in=i * step, source_out=(i + 1) * step, timeline_in=i * step, timeline_out=(i + 1) * step)
             for i in range(n)]
    return EDL(width=720, height=1280, clips=clips)


def _times(lo: float, hi: float, step: float = 0.05) -> list[float]:
    n = round((hi - lo) / step)
    return [round(lo + i * step, 3) for i in range(n)]


def _long_timeline():
    plan = plan_rhythm(_long(40), face_box=FACE)
    return plan, build_camera_timeline(plan, face_box=FACE)


# 1 -----------------------------------------------------------------------------------------
def test_1_a_rhythm_camera_event_becomes_an_executable_timeline_event():
    plan, tl = _long_timeline()
    assert tl.executable
    leads = _lead_rows(plan)
    assert leads
    for row in leads:
        ev = next(e for e in tl.executable if e.number == row.number and e.start == pytest.approx(row.motion_start))
        assert ev.owner == "rhythm" and ev.status is EventStatus.EXECUTABLE
        assert ev.zoom_to == pytest.approx(row.zoom_to) and ev.anchor_x == pytest.approx(row.anchor_x)
        assert row.status == "executable" and row.executable  # the review row mirrors the event that renders


# 2 -----------------------------------------------------------------------------------------
def test_2_a_static_legacy_camera_event_is_superseded_when_rhythm_owns_the_interval():
    plan = _hand(S.REFRAME_LEFT)
    old = CameraEvent(start=2.0, end=12.0, move=CameraMove.STATIC, zoom_to=BASE_ZOOM, owner="legacy", status_reason="the plan's slot camera=static")
    zoom = CameraEvent(start=5.0, end=5.4, move=CameraMove.PUNCH_IN, zoom_to=1.1, owner="legacy")
    tl = build_camera_timeline(plan, face_box=FACE, legacy=[old, zoom])
    assert {e.move for e in tl.superseded} == {CameraMove.STATIC, CameraMove.PUNCH_IN}
    assert all(e.status is EventStatus.SUPERSEDED and e.superseded_by for e in tl.superseded)
    assert all(e.owner == "rhythm" for e in tl.events)  # a superseded legacy event is never live
    assert plan.rows[0].superseded_legacy and tl.summary()["superseded"] == 2
    # ... but outside the stretch the timeline owns, the older system keeps its event
    scoped = build_camera_timeline(plan, face_box=FACE, legacy=[old.model_copy(update={"start": 30.0, "end": 31.0})], scope=(0.0, 20.0))
    assert not scoped.superseded


# 3 -----------------------------------------------------------------------------------------
def test_3_no_double_camera_execution():
    _, tl = _long_timeline()
    live = sorted(tl.executable, key=lambda e: e.start)
    assert all(b.start >= a.end - 1e-3 for a, b in pairwise(live))
    assert not [i for i in tl.issues if "double camera execution" in i]
    # two executable events that overlap are reported, never silently rendered twice
    twin = _hand(S.SLOW_PUSH, t=4.0, ret=9.0)
    twin.rows += [r.model_copy(update={"number": 2, "start": r.start + 0.5, "end": r.end + 0.5, "motion_start": (r.motion_start or r.start) + 0.5,
                                       "motion_end": (r.motion_end or r.end) + 0.5}) for r in _hand(S.REFRAME_RIGHT, t=4.0, ret=9.0).rows]
    bad = build_camera_timeline(twin, face_box=FACE)
    assert [i for i in bad.issues if "double camera execution" in i] and not bad.render_ready


# 4 -----------------------------------------------------------------------------------------
def test_4_an_abrupt_move_requires_a_stronger_boundary_than_a_smooth_move():
    weak = _no_boundaries()  # only word gaps (0.30): no phrase, clause or pause
    assert {b.kind for b in find_boundaries(weak)} <= {"word_gap", "segment_edge"}
    plan = plan_rhythm(weak)
    moving = [r for r in plan.rows if r.moving and r.source == "rhythm"]
    assert moving and all(r.motion_class == "smooth" for r in moving)
    assert not {"punch_in"} & set(plan.excursions())
    rich = plan_rhythm(_long(40))
    abrupt = [r for r in rich.rows if r.source == "rhythm" and r.moving and r.motion_class == "abrupt" and r.state != "reset_to_base"]
    assert abrupt and all(r.boundary_quality is not None and r.boundary_quality >= SAFE_QUALITY for r in abrupt)


# 5 -----------------------------------------------------------------------------------------
def test_5_a_smooth_move_may_start_at_a_safe_word_boundary_inside_a_clause():
    t = _no_boundaries()
    plan = plan_rhythm(t)
    leads = _lead_rows(plan)
    assert leads
    low = [r for r in leads if r.boundary_quality is not None and r.boundary_quality < SAFE_QUALITY]
    assert low, "with no phrase boundary a smooth move must be allowed on a word boundary"
    assert all(r.motion_class == "smooth" and not inside_word(r.start, t) for r in low)
    assert all(r.state in {"slow_push", "reframe_left", "reframe_right", "lower_subject"} for r in low)


# 6 -----------------------------------------------------------------------------------------
def test_6_no_event_starts_mid_word():
    t = _gappy()
    plan = plan_rhythm(t, face_box=FACE)
    tl = build_camera_timeline(plan, face_box=FACE)
    assert tl.events
    assert not [e.start for e in tl.events if inside_word(e.start, t)]
    assert not [r.start for r in plan.rows if r.source == "rhythm" and r.moving and inside_word(r.start, t)]


# 7 -----------------------------------------------------------------------------------------
def test_7_an_extended_hold_may_trigger_a_smooth_move_without_semantic_confidence():
    t = _no_boundaries(30.0)
    plan = plan_rhythm(t)  # no semantic reading, no phrase boundary at all
    assert plan.excursions()
    assert all(r.semantic_enhancement == "none" for r in plan.rows)
    seconds, _, _ = plan.longest_unchanged_hold()
    assert seconds < t.duration  # the hold was ended by a smooth move, not left for the whole run
    first = _lead_rows(plan)[0]
    assert first.start <= RhythmPolicy().attention_guard_s + 1e-6  # ... inside the extended-hold window, not on a timer cut


# 8 -----------------------------------------------------------------------------------------
def test_8_time_alone_still_does_not_trigger_a_semantic_treatment():
    plan = plan_rhythm(_no_boundaries(60.0))
    assert plan.excursions()
    assert all(r.semantic_enhancement == "none" and r.semantic_options == [] for r in plan.rows)
    filled, _ = _plan(_long(60))
    assert all(r.semantic_enhancement == "none" for r in filled.rhythm)
    assert all(s.treatment == "stay_on_speaker" for s in filled.slots)


# 9 -----------------------------------------------------------------------------------------
def test_9_a_smooth_reset_has_duration_start_and_end_states():
    plan = plan_rhythm(_long(40), face_box=FACE)
    smooth_returns = [r for r in plan.rows if r.source == "rhythm" and r.state in {"reset_to_base", "slow_pull"} and r.motion_class == "smooth"]
    assert smooth_returns
    p = plan.policy
    for r in smooth_returns:
        assert r.motion_start is not None and r.motion_end is not None
        assert p.return_min_s - 1e-6 <= r.motion_end - r.motion_start <= max(p.return_max_s, p.slow_ease_s) + 1e-6
        assert r.zoom_from > BASE_ZOOM and r.zoom_to == BASE_ZOOM  # start state and target state are both stored
        assert r.anchor_x_from is not None
    tl = build_camera_timeline(plan, face_box=FACE)
    back = [e for e in tl.executable if e.move in {CameraMove.RESET_TO_BASE, CameraMove.SLOW_PULL} and e.motion_class is MotionClass.SMOOTH]
    assert back and all(e.zoom_from is not None and e.zoom_from > e.zoom_to and e.motion_end > e.motion_start for e in back)


# 10 ----------------------------------------------------------------------------------------
def test_10_a_reset_returns_exactly_to_the_canonical_base():
    for state in (S.REFRAME_LEFT, S.REFRAME_RIGHT, S.LOWER_SUBJECT, S.SLOW_PUSH, S.PUNCH_IN):
        plan = _hand(state)
        tl = build_camera_timeline(plan, face_box=FACE)
        edl = _uniform_edl()
        apply_timeline(edl, tl, face_box=FACE)
        settle = max(e.end for e in tl.executable)
        after = sample_geometry(edl, _times(settle + 0.05, 19.9, 0.25), face_box=FACE)
        assert after, state
        for row in after:
            assert row["zoom"] == BASE_ZOOM and row["anchor_x"] == BASE_ANCHOR_X and row["anchor_y"] == BASE_ANCHOR_Y, (state, row)
        for row in sample_geometry(edl, [0.1, 1.0], face_box=FACE):  # ... and before the move the framing is the same base
            assert row["zoom"] == BASE_ZOOM and row["anchor_x"] == BASE_ANCHOR_X and row["anchor_y"] == BASE_ANCHOR_Y


# 11 ----------------------------------------------------------------------------------------
def test_11_lower_subject_increases_measurable_headroom():
    plan = _hand(S.LOWER_SUBJECT)
    tl = build_camera_timeline(plan, face_box=FACE)
    ev = next(e for e in tl.executable if e.move is CameraMove.LOWER_SUBJECT)
    assert ev.anchor_y == 0.0 and ev.scale == RhythmPolicy().lower_zoom and ev.anchor_x == BASE_ANCHOR_X
    assert ev.headroom_gain is not None and ev.headroom_gain >= RhythmPolicy().min_headroom_gain
    assert ev.safe_headroom == pytest.approx(FACE[1] + ev.headroom_gain, abs=1e-3)
    edl = _uniform_edl()
    apply_timeline(edl, tl, face_box=FACE)
    held = sample_geometry(edl, [ev.end + 0.5], face_box=FACE)[0]
    base = sample_geometry(edl, [0.5], face_box=FACE)[0]
    assert held["face_top"] - base["face_top"] >= RhythmPolicy().min_headroom_gain - 1e-6
    # a face that would end up under the caption band is blocked, never faked with a token crop
    low_face = (0.30, 0.55, 0.40, 0.40)
    blocked = build_camera_timeline(_hand(S.LOWER_SUBJECT, face=low_face), face_box=low_face)
    assert not any(e.move is CameraMove.LOWER_SUBJECT for e in blocked.executable)


# 12 ----------------------------------------------------------------------------------------
def test_12_lower_subject_preserves_the_head_and_face_safe_area():
    plan = _hand(S.LOWER_SUBJECT)
    tl = build_camera_timeline(plan, face_box=FACE)
    edl = _uniform_edl()
    apply_timeline(edl, tl, face_box=FACE)
    rows = sample_geometry(edl, _times(0.0, 19.95, 0.05), 720, 1280, face_box=FACE)
    p = RhythmPolicy()
    for r in rows:
        x0, y0, x1, y1 = r["window"]
        assert y0 >= -1e-6 and y1 <= 1280 + 1e-6 and x0 >= -1e-6 and x1 <= 720 + 1e-6  # the crop stays inside the source frame
        assert r["face_top"] >= FACE[1] - 1e-6  # never less room above the head than the base framing has
        assert 0.0 < r["face_top"] and r["face_bottom"] <= p.caption_safe_top + 1e-6  # nothing cut off, face above the captions
        assert 0.0 <= r["face_left"] and r["face_right"] <= 1.0


# 13 ----------------------------------------------------------------------------------------
def test_13_lower_subject_can_animate_and_reset():
    plan = _hand(S.LOWER_SUBJECT, t=4.0, ret=9.0)
    tl = build_camera_timeline(plan, face_box=FACE)
    edl = _uniform_edl()
    apply_timeline(edl, tl, face_box=FACE)
    rows = sample_geometry(edl, _times(3.5, 12.0, 0.05), face_box=FACE)
    ys = [r["anchor_y"] for r in rows]
    assert ys[0] == BASE_ANCHOR_Y and ys[-1] == BASE_ANCHOR_Y and min(ys) == 0.0  # base -> lowered -> exactly base
    inside = [r for r in rows if 4.0 < r["t"] < 4.9]
    assert inside and all(0.0 < r["anchor_y"] < BASE_ANCHOR_Y for r in inside[1:-1])  # a glide, not a jump
    steps = [abs(b["face_top"] - a["face_top"]) for a, b in pairwise(rows)]
    assert max(steps) < 0.02  # no visible jump between two 50 ms samples


# 14 ----------------------------------------------------------------------------------------
def test_14_reframe_left_and_right_are_executable():
    for state, sign in ((S.REFRAME_LEFT, -1), (S.REFRAME_RIGHT, 1)):
        tl = build_camera_timeline(_hand(state), face_box=FACE)
        ev = next(e for e in tl.executable if e.move.value == state.value)
        assert state.value in {m.value for m in EXECUTABLE_MOVES} and tl.render_ready
        assert (ev.anchor_x - BASE_ANCHOR_X) * sign > 0
        edl = _uniform_edl()
        apply_timeline(edl, tl, face_box=FACE)
        held = sample_geometry(edl, [ev.end + 0.4], face_box=FACE)[0]
        assert (held["anchor_x"] - BASE_ANCHOR_X) * sign > 0 and held["zoom"] > BASE_ZOOM


# 15 ----------------------------------------------------------------------------------------
def test_15_slow_push_slow_pull_punch_and_reset_are_executable():
    tl = build_camera_timeline(_hand(S.SLOW_PUSH), face_box=FACE)
    assert {e.move for e in tl.executable} >= {CameraMove.SLOW_PUSH, CameraMove.SLOW_PULL}
    edl = _uniform_edl()
    apply_timeline(edl, tl, face_box=FACE)
    push = next(e for e in tl.executable if e.move is CameraMove.SLOW_PUSH)
    mid = sample_geometry(edl, [(push.start + push.end) / 2, push.end - 0.01], face_box=FACE)
    assert BASE_ZOOM < mid[0]["zoom"] < mid[1]["zoom"] <= push.zoom_to + 1e-6  # it drifts in, it does not jump
    punch = build_camera_timeline(_hand(S.PUNCH_IN), face_box=FACE)
    assert {e.move for e in punch.executable} >= {CameraMove.PUNCH_IN, CameraMove.RESET_TO_BASE}
    assert punch.render_ready and tl.render_ready


# 16 ----------------------------------------------------------------------------------------
def test_16_no_accumulated_zoom():
    plan, tl = _long_timeline()
    assert never_accumulates(plan.to_camera_plan(face_box=FACE))
    t = _long(40)
    edl = _edl(t)
    apply_timeline(edl, tl, face_box=FACE)
    rows = sample_geometry(edl, _times(0.0, t.duration - 0.05, 0.1), face_box=FACE)
    assert max(r["zoom"] for r in rows) <= RhythmPolicy().max_zoom + 1e-9
    # every excursion ends on base: the framing sampled just after each settle is exactly the canonical base
    for ev in tl.executable:
        if ev.move in {CameraMove.RESET_TO_BASE, CameraMove.SLOW_PULL, CameraMove.PUNCH_OUT}:
            row = sample_geometry(edl, [ev.end + 0.02], face_box=FACE)
            if row and not any(o.start <= ev.end + 0.02 < o.end for o in tl.executable if o is not ev):
                assert row[0]["zoom"] == BASE_ZOOM and row[0]["anchor_x"] == BASE_ANCHOR_X and row[0]["anchor_y"] == BASE_ANCHOR_Y


# 17 ----------------------------------------------------------------------------------------
def _pin(**over):
    args = {"phrase": "keep it simple", "semantic_kind": "key_claim", "semantic_confidence": 0.1, "evidence": FULL, "rhythm_state": "base",
            "semantic_source": "user_pinned", **over}
    return behind_subject_eligibility(**args)


def test_17_a_user_pinned_semantic_treatment_survives_low_model_confidence():
    pinned = _pin()
    assert pinned.status == "eligible" and pinned.semantic_source == "user_pinned"
    assert pinned.candidate == "user_pinned_technical_gate_passed" and pinned.requires_approval  # a pin is never an approval
    assert _pin(semantic_kind=None, semantic_confidence=None).status == "eligible"
    auto = _pin(semantic_source="auto")  # the same low confidence without a pin is still rejected
    assert auto.status == "not_eligible" and "semantic_clarity" in auto.failed and auto.candidate == ""


# 18 ----------------------------------------------------------------------------------------
def test_18_a_user_pinned_behind_subject_still_requires_the_technical_gate():
    assert _pin(evidence=None).status == "not_eligible"
    assert _pin(evidence=None).candidate == "user_pinned_pending_technical_gate"
    for check in ("mask_quality", "head_hair_integrity", "meaningful_occlusion", "readable_occlusion", "phrase_timing", "shot_composition",
                  "caption_hierarchy"):
        for value in (False, None):
            v = _pin(evidence=FULL.model_copy(update={check: value}))
            assert v.status == "not_eligible" and v.candidate == "user_pinned_pending_technical_gate", (check, value)
    assert _pin(phrase="this phrase has far too many words").status == "not_eligible"
    assert _pin(rhythm_state="punch_in").status == "not_eligible"
    assert _pin(caption_competing=True).status == "not_eligible"
    assert _pin(better_simpler_treatment=True).status == "not_eligible"


# 19 ----------------------------------------------------------------------------------------
def test_19_behind_subject_blocks_incompatible_camera_movement():
    window = [(3.0, 12.0)]
    for state in (S.PUNCH_IN, S.REFRAME_LEFT, S.REFRAME_RIGHT, S.SLOW_PUSH):
        tl = build_camera_timeline(_hand(state), face_box=FACE, behind_subject=window)
        moving = [e for e in tl.events if e.move.value == state.value]
        assert moving and all(e.status is EventStatus.BLOCKED and "Behind-Subject" in e.status_reason for e in moving), state
        assert not [e for e in tl.executable if e.owner == "rhythm" and e.start < 12.0]
    # a very subtle, slow move and a pre-validated lower_subject stay compatible; outside the window nothing changes
    subtle = _hand(S.SLOW_PUSH)
    for r in subtle.rows:
        r.zoom_to = 1.02 if r.state == "slow_push" else r.zoom_to
        r.zoom_from = 1.02 if r.state == "slow_pull" else r.zoom_from
    assert all(e.status is EventStatus.EXECUTABLE for e in build_camera_timeline(subtle, face_box=FACE, behind_subject=window).events)
    lowered = build_camera_timeline(_hand(S.LOWER_SUBJECT), face_box=FACE, behind_subject=window)
    assert any(e.move is CameraMove.LOWER_SUBJECT and e.status is EventStatus.EXECUTABLE for e in lowered.events)
    away = build_camera_timeline(_hand(S.PUNCH_IN), face_box=FACE, behind_subject=[(15.0, 18.0)])
    assert all(e.status is EventStatus.EXECUTABLE for e in away.events)


# 20 ----------------------------------------------------------------------------------------
def test_20_planning_only_events_keep_ready_for_final_render_false():
    raise_tl = build_camera_timeline(_hand(S.RAISE_SUBJECT), face_box=FACE)
    assert raise_tl.planning_only and not raise_tl.render_ready
    assert all(e.status is EventStatus.PLANNING_ONLY for e in raise_tl.events)
    assert any("raise_subject" in b for b in raise_tl.blockers)
    blind = build_camera_timeline(_hand(S.LOWER_SUBJECT, face=None), face_box=None)  # no measured face -> cannot be proven safe
    assert blind.planning_only and not blind.render_ready
    plan, _ = _plan(_long(60))
    assert plan.camera_timeline is not None
    plan.camera_timeline = raise_tl
    assert plan.render_blockers() and not plan.camera_render_ready
    ready = build_camera_timeline(_hand(S.REFRAME_LEFT), face_box=FACE)
    assert ready.render_ready


# 21 ----------------------------------------------------------------------------------------
def _numbered_plan(seconds: float = 60.0, step: float = 12.0):
    """A plan whose slots are numbered (as reviewable slots are) and all claim `camera=static`."""
    t = _long(int(seconds / 3.4) + 1)
    plan, _ = _plan(t)
    plan.slots = [EditPlanSlot(number=i + 1, timeline_start=i * step, timeline_end=(i + 1) * step, recommended="stay_on_speaker",
                               treatment="stay_on_speaker", camera="static") for i in range(int(seconds / step))]
    rhythm = attach_rhythm(plan, t, _edl(t), face_box=FACE)
    return plan, rhythm, t


def test_21_the_review_state_equals_the_executable_state():
    plan, _, t = _numbered_plan()
    tl = plan.camera_timeline
    assert tl is not None and tl.executable
    assert review_mismatches(plan.slots, tl) == []
    moving = [s for s in plan.slots if s.camera_owner == "rhythm"]
    assert moving and all(s.camera != "static" and s.camera_events for s in moving)
    # the legacy `static` claims that the timeline owns are superseded explicitly, never left to coexist
    assert tl.superseded and all(e.status is EventStatus.SUPERSEDED for e in tl.superseded)
    assert all(e.owner == "legacy" for e in static_claims(plan.slots) + legacy_events(_edl(t)))
    # a review that still says `static` over an executable move is reported and the plan is not render-ready
    victim = moving[0].model_copy(update={"camera": "static", "camera_owner": ""})
    assert review_mismatches([victim], tl)
    plan.slots[plan.slots.index(moving[0])] = victim
    assert any("review says camera=static" in b for b in plan.render_blockers()) and not plan.camera_render_ready


# 22 ----------------------------------------------------------------------------------------
FORBIDDEN = [r"#0a3d62", r"#8e44ad", r"#f1c40f", r"video-ad-editor", r"LI-SEP", r"value\s+حقيقية", r"\b15\.94\b",
             r"\b25\.42\b", r"\b9\.48\b"]
GENERIC = ["direction/camera_timeline.py", "direction/rhythm.py", "direction/camera.py", "direction/suitability.py", "review/edit_plan_rhythm.py",
           "render/reframe.py"]


@pytest.mark.parametrize("rel", GENERIC)
def test_22_no_project_specific_values_in_the_generic_implementation(rel: str):
    src = (SRC / rel).read_text(encoding="utf-8")
    hits = [pat for pat in FORBIDDEN if re.search(pat, src, flags=re.IGNORECASE)]
    assert hits == [], f"{rel} hardcodes {hits}"


# 23 ----------------------------------------------------------------------------------------
def test_23_settled_static_slots_mirror_the_executable_camera():
    """A settled slot (number 0, "no special treatment") that claims `static` must show the moving event that will render."""
    t = _long(20)
    plan, _ = _plan(t)
    plan.slots = [EditPlanSlot(number=0, timeline_start=i * 12.0, timeline_end=(i + 1) * 12.0, recommended="stay_on_speaker",
                               treatment="stay_on_speaker", camera="static", settled_reason="no special treatment") for i in range(5)]
    attach_rhythm(plan, t, _edl(t), face_box=FACE)
    tl = plan.camera_timeline
    assert tl is not None and tl.executable
    assert review_mismatches(plan.slots, tl) == []
    moving = [s for s in plan.slots if s.camera_owner == "rhythm"]
    assert moving and all(s.camera != "static" and s.camera_events for s in moving)


# 24 ----------------------------------------------------------------------------------------
def test_24_a_long_wait_scores_lower_than_a_reachable_phrase_boundary_without_a_hard_timer():
    from video_edit_agent.direction.rhythm import timing_score
    p = RhythmPolicy()
    tail = [timing_score(h, p) for h in (6.0, 7.0, 8.0, 9.5, 12.0, 30.0)]
    assert all(a >= b for a, b in pairwise(tail)) and tail[0] > tail[3]
    assert min(tail) > 0.0  # never zero: a long hold with nothing better remains a valid outcome (no fixed timer)
