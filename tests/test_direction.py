"""The constrained visual director: vocabulary, speaker replacement, camera grammar, speed,
transitions and the behind-subject editorial gate. Synthetic beats only -- nothing project-specific."""
from __future__ import annotations

from itertools import pairwise

import pytest

from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.core.timeline import TrackType
from video_edit_agent.direction import (
    Beat,
    BeatKind,
    SpeakerReplacement,
    direct,
)
from video_edit_agent.direction.camera import (
    BASE_ZOOM,
    CameraMove,
    CameraPolicy,
    CameraRequest,
    apply_to_edl,
    peak_zoom,
    plan_camera,
    zoom_trajectory,
)
from video_edit_agent.direction.speed import PACING_PRIORITY, SpeedEvent, plan_speed, renderable
from video_edit_agent.direction.suitability import BehindSubjectEvidence, assess_behind_subject
from video_edit_agent.direction.transitions import (
    DIRECT,
    STYLIZED_EXECUTABLE,
    TransitionReason,
    TransitionStyle,
    select_transition,
)
from video_edit_agent.direction.vocabulary import (
    REPLACEMENT_TREATMENTS,
    SPEAKER_TREATMENTS,
    VOCABULARY,
    default_speaker_visible,
    treatment_class,
)
from video_edit_agent.sound.profile import get_profile


def _beat(start: float, end: float, kind: BeatKind = BeatKind.PLAIN, **kw) -> Beat:
    return Beat(start=start, end=end, kind=kind, **kw)


_GOOD = BehindSubjectEvidence(mask_quality=True, readable_occlusion=True, phrase_timing=True,
                              shot_composition=True, caption_hierarchy=True, visual_value=0.9)

# 1. the director picks from the constrained vocabulary ---------------------------------------------------


def test_director_only_picks_from_the_approved_vocabulary():
    beats = [_beat(0, 6), _beat(6, 12, BeatKind.EMPHASIS), _beat(12, 18, BeatKind.CONCEPT), _beat(18, 24, BeatKind.DATA),
             _beat(24, 30, BeatKind.KEY_PHRASE), _beat(30, 36, BeatKind.CONCRETE), _beat(36, 44, BeatKind.PERSONAL),
             _beat(44, 50, BeatKind.TOPIC_SHIFT)]
    result = direct(beats)
    assert len(result.decisions) == len(beats)
    for d in result.decisions:
        assert d.treatment in VOCABULARY
        assert all(a in VOCABULARY for a in d.alternatives)
    assert {"speaker", "replacement"} <= {d.klass for d in result.decisions}


def test_a_treatment_outside_the_vocabulary_is_never_executed():
    d = direct([_beat(0, 6, prefer="spinning_3d_cube")]).decisions[0]
    assert d.treatment == "speaker_static" and "not in the approved vocabulary" in d.reason


def test_the_vocabulary_has_the_specified_members():
    assert SPEAKER_TREATMENTS == {"speaker_static", "punch_in", "punch_out", "reframe", "slow_push", "hold", "reset_to_base"}
    assert REPLACEMENT_TREATMENTS == {"real_broll", "user_broll", "motion_graphic", "kinetic_typography", "illustration",
                                      "generated_visual", "graphic_data_scene", "full_screen_text_scene"}


def test_plain_delivery_stays_on_the_speaker_with_no_move_and_no_sound():
    d = direct([_beat(0, 8)], sound_profile=get_profile("dynamic")).decisions[0]
    assert d.treatment == "speaker_static" and d.camera is CameraMove.STATIC and d.speaker_visible
    assert d.sound is None
    assert d.transition.effective is TransitionStyle.DIRECT_CUT


def test_behind_subject_is_optional_and_gated():
    beat = _beat(10, 14, BeatKind.EMPHASIS)
    assert direct([beat]).decisions[0].treatment == "punch_in"  # no evidence -> not chosen
    weak = _beat(10, 14, BeatKind.EMPHASIS, evidence=_GOOD.model_copy(update={"mask_quality": False}))
    dec = direct([weak]).decisions[0]
    assert dec.treatment == "punch_in" and "mask_quality" in dec.reason
    assert direct([_beat(10, 14, BeatKind.EMPHASIS, evidence=_GOOD)]).decisions[0].treatment == "behind_subject_text"


@pytest.mark.parametrize("field", ["mask_quality", "readable_occlusion", "phrase_timing", "shot_composition", "caption_hierarchy"])
def test_editorial_gate_fails_closed(field):
    unmeasured = _GOOD.model_copy(update={field: None})
    failed = _GOOD.model_copy(update={field: False})
    for ev in (unmeasured, failed):
        verdict = assess_behind_subject(ev)
        assert not verdict.suitable and field in (verdict.failed + verdict.unproven)
    assert assess_behind_subject(_GOOD).suitable
    assert not assess_behind_subject(_GOOD.model_copy(update={"visual_value": 0.2})).suitable
    assert not assess_behind_subject(BehindSubjectEvidence()).suitable


def test_concrete_beat_without_footage_never_falls_back_to_generation():
    d = direct([_beat(0, 6, BeatKind.CONCRETE, visual_concept="a place")]).decisions[0]
    assert d.treatment == "speaker_static" and not d.needs_generation
    assert "generated_visual" in d.alternatives  # offered, never chosen
    with_footage = direct([_beat(0, 6, BeatKind.CONCRETE, has_footage=True)]).decisions[0]
    assert with_footage.treatment == "real_broll" and with_footage.needs_asset and not with_footage.speaker_visible


def test_abstract_concepts_get_a_simple_graphic_not_forced_footage():
    d = direct([_beat(0, 6, BeatKind.CONCEPT)]).decisions[0]
    assert d.treatment == "motion_graphic" and not d.needs_asset and not d.needs_generation


def test_the_director_is_deterministic():
    beats = [_beat(0, 6), _beat(6, 12, BeatKind.CONCEPT), _beat(20, 26, BeatKind.EMPHASIS, importance=0.9)]
    a = direct(beats, sound_profile=get_profile("minimal")).model_dump()
    b = direct(list(reversed(beats)), sound_profile=get_profile("minimal")).model_dump()
    assert a == b


def test_the_speaker_returns_before_another_replacement():
    beats = [_beat(0, 6, BeatKind.CONCEPT), _beat(6, 12, BeatKind.DATA), _beat(12, 18, BeatKind.KEY_PHRASE)]
    kinds = [(d.klass, d.speaker_visible) for d in direct(beats).decisions]
    hidden_run = [not visible and klass == "replacement" for klass, visible in kinds]
    assert not any(a and b for a, b in pairwise(hidden_run))


# 2. speaker replacement is distinct from B-roll --------------------------------------------------------


def test_speaker_replacement_is_distinct_from_broll():
    graphic = SpeakerReplacement(start=10, end=14, kind="motion_graphic")
    footage = SpeakerReplacement(start=10, end=14, kind="user_broll")
    assert graphic.source.value == "authored" and footage.source.value == "footage"
    assert not graphic.needs_asset and footage.needs_asset
    # a replacement that is not B-roll needs no asset and is executable on its own
    assert graphic.executable and graphic.effective == "motion_graphic"
    # the voice never stops, whatever kind of visual replaces the speaker
    assert graphic.voice_continues
    with pytest.raises(ValueError):
        SpeakerReplacement(start=10, end=14, kind="motion_graphic", voice_continues=False)
    # only replacement kinds qualify; a camera move or a speaker shot is not a replacement
    for bad in ("punch_in", "speaker_static", "behind_subject_text", "made_up"):
        with pytest.raises(ValueError):
            SpeakerReplacement(start=0, end=2, kind=bad)
    assert treatment_class("motion_graphic") == "replacement" and treatment_class("real_broll") == "replacement"
    assert treatment_class("punch_in") == "speaker" and treatment_class("behind_subject_text") == "overlay"
    assert not default_speaker_visible("full_screen_text_scene")


def test_replacement_without_its_asset_falls_back_to_the_speaker():
    missing = SpeakerReplacement(start=5, end=9, kind="real_broll")
    assert not missing.executable and missing.effective == "speaker_static"
    supplied = SpeakerReplacement(start=5, end=9, kind="real_broll", asset_provided=True)
    assert supplied.effective == "real_broll"


def test_replacement_becomes_a_timeline_item_above_the_speaker_without_touching_the_voice():
    item = SpeakerReplacement(start=5, end=9, kind="motion_graphic").to_timeline_item("r1")
    assert item.type in {TrackType.MOTION_GRAPHICS, TrackType.BROLL}
    assert item.start == 5 and item.duration == 4
    assert item.type is not TrackType.AUDIO


# 19. generated assets are never triggered without explicit approval -------------------------------------------


def test_generated_visuals_need_explicit_approval():
    gen = SpeakerReplacement(start=5, end=9, kind="generated_visual")
    assert gen.needs_generation and not gen.executable and gen.effective == "speaker_static"
    approved = SpeakerReplacement(start=5, end=9, kind="generated_visual", generation_approved=True)
    assert approved.executable and approved.effective == "generated_visual"
    # the director never picks a generated visual on its own, whatever the beat kind
    for kind in BeatKind:
        d = direct([_beat(0, 8, kind, visual_concept="anything")]).decisions[0]
        assert d.treatment != "generated_visual" and not d.needs_generation, kind


# 3. punch-out / reset is supported --------------------------------------------------------------------


def test_punch_out_and_reset_are_supported_and_return_to_base():
    plan = plan_camera([
        CameraRequest(start=10, end=13, move=CameraMove.PUNCH_IN),
        CameraRequest(start=13, end=15, move=CameraMove.PUNCH_OUT),
        CameraRequest(start=30, end=33, move=CameraMove.PUNCH_IN),
        CameraRequest(start=34, end=36, move=CameraMove.RESET_TO_BASE),
    ])
    moves = [e.move for e in plan.events]
    assert CameraMove.PUNCH_OUT in moves and CameraMove.RESET_TO_BASE in moves
    assert zoom_trajectory(plan, 20.0) == pytest.approx(BASE_ZOOM, abs=1e-6)
    assert zoom_trajectory(plan, 60.0) == pytest.approx(BASE_ZOOM, abs=1e-6)


def test_emphasis_is_reset_after_the_beat_even_if_nobody_asks():
    plan = plan_camera([CameraRequest(start=10, end=12, move=CameraMove.PUNCH_IN)])
    assert plan.events[-1].move is CameraMove.RESET_TO_BASE and plan.events[-1].zoom_to == BASE_ZOOM


def test_static_is_a_valid_camera_answer():
    plan = plan_camera([CameraRequest(start=0, end=10, move=CameraMove.STATIC)])
    assert plan.events == [] and plan.dropped == []


# 4. no cumulative zoom drift --------------------------------------------------------------------------


def test_no_cumulative_zoom_drift():
    policy = CameraPolicy()
    requests = [CameraRequest(start=5.0 * i, end=5.0 * i + 2.0, move=CameraMove.PUNCH_IN) for i in range(40)]
    plan = plan_camera(requests, policy)
    assert peak_zoom(plan) <= policy.max_zoom
    assert all(e.zoom_to <= policy.max_zoom for e in plan.events)
    for t in [x * 0.25 for x in range(900)]:
        assert BASE_ZOOM - 1e-6 <= zoom_trajectory(plan, t) <= policy.max_zoom + 1e-6
    assert zoom_trajectory(plan, 1000.0) == pytest.approx(BASE_ZOOM, abs=1e-6)
    # the same move ten times over reaches the same absolute level every time (targets, not increments)
    levels = {e.zoom_to for e in plan.events if e.move is CameraMove.PUNCH_IN}
    assert levels == {policy.punch_in_zoom}


def test_a_move_on_top_of_a_held_emphasis_is_dropped_not_stacked():
    plan = plan_camera([CameraRequest(start=10, end=20, move=CameraMove.PUNCH_IN), CameraRequest(start=11, end=15, move=CameraMove.SLOW_PUSH)])
    assert peak_zoom(plan) == CameraPolicy().punch_in_zoom
    assert plan.dropped and plan.dropped[0].move is CameraMove.SLOW_PUSH


def test_moves_are_spaced_out():
    plan = plan_camera([CameraRequest(start=10, end=11, move=CameraMove.PUNCH_IN),
                        CameraRequest(start=11.5, end=12.5, move=CameraMove.PUNCH_IN)])
    assert len([e for e in plan.events if e.move is CameraMove.PUNCH_IN]) == 1
    assert plan.dropped[0].reason in {"too close to the previous move", "a previous emphasis has not been reset yet"}


def test_camera_plan_is_written_to_the_edl_through_the_existing_punch_in_model():
    edl = EDL(width=720, height=1280, clips=[
        EDLClip(source_file="a.mp4", source_in=0, source_out=30, timeline_in=0, timeline_out=30, caption_refs=["s0"]),
    ])
    plan = plan_camera([CameraRequest(start=10, end=13, move=CameraMove.PUNCH_IN)])
    apply_to_edl(edl, plan)
    ramps = edl.clips[0].reframe.ramps if edl.clips[0].reframe else []
    assert ramps, "the move is expressed as ordinary zoom ramps, like the approved punch-in"
    assert max(r.zoom_to for r in ramps) <= CameraPolicy().max_zoom


# 5. speed-up is not automatic -------------------------------------------------------------------------------


def test_speed_up_is_never_automatic():
    assert plan_speed() == [] and plan_speed(None) == []
    beats = [_beat(0, 8), _beat(8, 16, BeatKind.EMPHASIS), _beat(16, 30, BeatKind.PERSONAL)]
    assert not hasattr(direct(beats), "speed")
    assert PACING_PRIORITY[0] == "remove_dead_air" and PACING_PRIORITY[-1] == "explicit_speed_change"


def test_a_speed_change_needs_a_justification_and_approval():
    with pytest.raises(ValueError):
        SpeedEvent(start=0, end=5, factor=1.1, justification="  ")
    for factor in (1.5, 0.5):
        with pytest.raises(ValueError):
            SpeedEvent(start=0, end=5, factor=factor, justification="x")
    ev = SpeedEvent(start=0, end=5, factor=1.1, justification="a long pause the speaker asked to tighten")
    assert renderable([ev]) == []  # unapproved: never plays
    assert renderable([ev.model_copy(update={"approved": True})]) != []


# 6. default transition is a direct/invisible cut ---------------------------------------------------------


def test_default_transition_is_a_direct_cut():
    assert select_transition().style is TransitionStyle.DIRECT_CUT
    assert DIRECT.effective is TransitionStyle.DIRECT_CUT and not DIRECT.stylized
    beats = [_beat(0, 6, BeatKind.CONCEPT), _beat(6, 12), _beat(12, 18, BeatKind.DATA)]
    for d in direct(beats).decisions:
        assert d.transition.effective is TransitionStyle.DIRECT_CUT


def test_stylized_transitions_need_a_justification_and_are_kept_apart():
    assert select_transition(preferred="whip").style is TransitionStyle.DIRECT_CUT  # no reason
    assert select_transition("unknown_reason").style is TransitionStyle.DIRECT_CUT
    ok = select_transition(TransitionReason.MOTION_CONTINUITY, at=20.0)
    assert ok.style is TransitionStyle.WHIP and ok.stylized
    assert select_transition(TransitionReason.MOTION_CONTINUITY, "blur").style is TransitionStyle.DIRECT_CUT  # wrong style for the reason
    close = select_transition(TransitionReason.SEMANTIC_CHANGE, at=22.0, previous_stylized_at=20.0)
    assert close.style is TransitionStyle.DIRECT_CUT


def test_a_justified_stylized_transition_is_recorded_but_the_direct_cut_plays():
    assert not STYLIZED_EXECUTABLE
    beat = _beat(10, 16, BeatKind.CONCEPT, transition_reason=TransitionReason.DELIBERATE_REVEAL)
    d = direct([beat]).decisions[0]
    assert d.transition.stylized and d.transition.effective is TransitionStyle.DIRECT_CUT


# density of replacements/camera moves in a realistic run ----------------------------------------------------


def test_a_long_run_stays_sparse():
    beats = [_beat(6.0 * i, 6.0 * i + 6.0, BeatKind.EMPHASIS if i % 2 else BeatKind.PLAIN, importance=0.8) for i in range(20)]
    result = direct(beats)
    moves = [e for e in result.camera.events if e.move not in {CameraMove.RESET_TO_BASE, CameraMove.PUNCH_OUT}]
    assert len(moves) <= len(beats) // 2
    starts = [e.start for e in moves]
    assert all(b - a >= CameraPolicy().min_gap_s for a, b in pairwise(starts))
