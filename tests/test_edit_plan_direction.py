"""The directed edit plan: the review state records the visual AND sound decisions, sound edits are
honest, generation is never triggered without an explicit approval, and the chat speaks the
per-slot language. Synthetic content only -- nothing project-specific."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from tests.conftest import make_transcript
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.direction import Beat, BeatKind, direct
from video_edit_agent.review import edit_plan as ep
from video_edit_agent.review import edit_plan_commands as cmd
from video_edit_agent.review import edit_plan_direction as ed
from video_edit_agent.review import state as rs
from video_edit_agent.review.edit_plan import SlotStatus
from video_edit_agent.review.edit_plan_chat import EditPlanChat
from video_edit_agent.sound.intent import EventType, SoundIntent
from video_edit_agent.sound.profile import get_profile
from video_edit_agent.sound.registry import SfxAsset, SfxCategory, SfxRegistry
from video_edit_agent.transcription.router import save_transcript

LINES = [
    ("أهلا بيكم في الفيديو", 0.0, 6.0),
    ("الـvalues دي لازم تبقى حقيقية مش كلام", 6.0, 12.0),
    ("بنرتب الأفكار في مجموعات واضحة", 12.0, 18.0),
    ("وبنشتغل مع الفريق كل يوم", 18.0, 24.0),
    ("ولازم الـdemand يبقى واضح قبل أي حاجة", 24.0, 30.0),
]
BEATS = [
    Beat(start=0.0, end=6.0, kind=BeatKind.PLAIN),
    Beat(start=6.0, end=12.0, kind=BeatKind.EMPHASIS, importance=0.8),
    Beat(start=12.0, end=18.0, kind=BeatKind.CONCEPT, visual_concept="hands arranging blank cards"),
    Beat(start=18.0, end=24.0, kind=BeatKind.CONCRETE, visual_concept="the team at work"),
    Beat(start=24.0, end=30.0, kind=BeatKind.KEY_PHRASE),
]


def _edl() -> EDL:
    clips = [
        EDLClip(source_file="a.mp4", source_in=s, source_out=e, timeline_in=s, timeline_out=e, caption_refs=[f"s{i}"])
        for i, (_, s, e) in enumerate(LINES)
    ]
    return EDL(width=720, height=1280, clips=clips)


def _transcript():
    return make_transcript(list(LINES))


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 4000)


@pytest.fixture
def pack(tmp_path: Path) -> SfxRegistry:
    """A synthetic pack (silent real files) so the 'scheduled' path can be exercised."""
    assets = [
        SfxAsset(id="whoosh_a", category=SfxCategory.SOFT_WHOOSH, file="whoosh_a.wav", energy="low", duration_s=0.5,
                 style="clean", allowed_intents=[SoundIntent.SUBTLE_MOTION, SoundIntent.ACCENT],
                 recommended_events=[EventType.PUNCH_IN]),
        SfxAsset(id="sweep_a", category=SfxCategory.TRANSITION_SWEEP, file="sweep_a.wav", energy="low", duration_s=0.5,
                 style="clean", allowed_intents=[SoundIntent.TRANSITION, SoundIntent.ACCENT],
                 recommended_events=[EventType.REPLACEMENT_IN]),
        SfxAsset(id="tick_a", category=SfxCategory.TEXT_TICK_SOFT, file="tick_a.wav", energy="low", duration_s=0.3,
                 style="clean", allowed_intents=[SoundIntent.ACCENT], recommended_events=[EventType.KEY_REVEAL]),
    ]
    for a in assets:
        _wav(tmp_path / a.file)
    return SfxRegistry(assets=assets, root=str(tmp_path))


def _plan(profile: str = "minimal", registry: SfxRegistry | None = None, **kw) -> ep.EditPlan:
    result = direct(BEATS, sound_profile=get_profile(profile), registry=registry)
    return ed.build_directed_plan(BEATS, result, _transcript(), _edl(), sound_profile=profile, registry=registry, **kw)


def _project(tmp_path: Path, plan: ep.EditPlan) -> Path:
    edit = tmp_path / "edit"
    (edit / "review").mkdir(parents=True)
    save_transcript(_transcript(), edit / "transcript_unified.json")
    (edit / "edl.json").write_text(_edl().model_dump_json(), encoding="utf-8")
    ep.save_plan(plan, edit / "review")
    return edit


# 18. review state records visual + sound decisions -------------------------------------------------------


def test_every_beat_becomes_a_slot_that_records_the_visual_and_the_sound_decision():
    plan = _plan()
    assert [(s.timeline_start, s.timeline_end) for s in plan.slots] == [(b.start, b.end) for b in BEATS]
    for slot in plan.slots:
        assert slot.directed and slot.beat_kind and slot.camera and slot.transition
        assert slot.sound_intent in ep.SOUND_INTENTS and slot.sound_status in {"none", "suppressed", "unavailable_fallback_none", "scheduled"}
        assert slot.caption_behavior in {"normal", "reduced"}
    assert plan.sound_profile == "minimal"
    assert plan.slots[0].spoken_context.startswith("أهلا")  # the approved transcript, verbatim


def test_plain_speaker_beats_are_settled_and_the_rest_wait_for_the_reviewer():
    plan = _plan()
    plain = plan.slots[0]
    assert plain.status is SlotStatus.APPROVED and plain.number == 0 and plain.speaker_visible
    assert plain.sound_intent == "none" and plain.transition == "direct_cut"
    rest = [s for s in plan.slots if s.number > 0]
    assert rest and all(s.status is SlotStatus.PENDING_REVIEW for s in rest)
    assert [s.number for s in rest] == list(range(1, len(rest) + 1))
    # a concrete beat with no footage stays on the speaker (settled) rather than inventing a visual
    no_footage = plan.slots[3]
    assert no_footage.treatment == "stay_on_speaker" and no_footage.number == 0 and no_footage.speaker_visible


def test_replacement_slots_hide_the_speaker_and_have_no_camera():
    plan = _plan()
    replaced = [s for s in plan.slots if not s.speaker_visible]
    assert replaced and all(s.camera == "n/a" for s in replaced)


def test_the_plan_round_trips_with_the_full_decision(tmp_path: Path):
    plan = _plan()
    ep.save_plan(plan, tmp_path)
    back = ep.load_plan(tmp_path)
    assert back is not None and back.model_dump() == plan.model_dump()


def test_kept_slots_are_carried_over_untouched():
    kept = ep.EditPlanSlot(timeline_start=-1.0, timeline_end=0.0, treatment="punch_in", recommended="punch_in",
                           status=SlotStatus.APPROVED, settled_reason="approved earlier")
    plan = _plan(kept=[kept])
    first = plan.slots[0]
    assert first.timeline_start == -1.0 and first.status is SlotStatus.APPROVED and first.settled_reason == "approved earlier"
    assert len(plan.slots) == len(BEATS) + 1


def test_building_a_plan_never_approves_generation_or_final_render(tmp_path: Path):
    plan = _plan()
    assert not any(s.generation_approved for s in plan.slots)
    edit = _project(tmp_path, plan)
    assert rs.load_review_state(edit / "review").ready_for_final_render is False


# 19. generated assets are never triggered without explicit approval ---------------------------------------


def test_generated_visuals_are_never_chosen_or_approved_by_default():
    plan = _plan()
    assert all(s.treatment != "generated_broll" for s in plan.slots)
    assert not any(s.needs_generation for s in plan.slots)


def test_generation_needs_its_own_explicit_approval(tmp_path: Path):
    plan = _plan()
    slot = plan.slots[2]
    ep.set_treatment(slot, "generated_broll")
    assert slot.needs_generation and not slot.generation_approved
    assert slot.status is not SlotStatus.GENERATION_APPROVED
    ep.approve_slot(slot)  # accepting the treatment is not permission to generate
    assert not slot.generation_approved and slot.needs_generation
    ep.approve_generation(slot)
    assert slot.generation_approved and slot.status is SlotStatus.GENERATION_APPROVED


def test_generation_approval_is_refused_on_a_directed_non_generated_slot():
    plan = _plan()
    with pytest.raises(ValueError):
        ep.approve_generation(plan.slots[1])


def test_changing_away_from_generated_withdraws_the_generation_approval():
    plan = _plan()
    slot = plan.slots[2]
    ep.set_treatment(slot, "generated_broll")
    ep.approve_generation(slot)
    ep.set_treatment(slot, "motion_graphic")
    assert not slot.generation_approved


# sound outcomes are honest -----------------------------------------------------------------------------


def test_with_no_registry_every_sound_is_honestly_unavailable_or_none():
    plan = _plan()
    for slot in plan.slots:
        if slot.sound_intent == "none":
            assert slot.sound_status == "none"
        else:
            assert slot.sound_status == "unavailable_fallback_none"
            assert "no" in (slot.sfx_availability or "").lower() or "unavailable" in (slot.sfx_availability or "").lower()


def test_a_real_pack_schedules_a_sound_for_a_chosen_intent(pack):
    plan = _plan(registry=pack)
    slot = plan.slots[2]
    ed.set_slot_sound(plan, slot, "accent", pack)
    assert slot.sound_locked and slot.sound_intent == "accent"
    assert slot.sound_status in {"scheduled", "suppressed"}  # profile density may still decline it
    ed.set_slot_sound(plan, slot, "none", pack)
    assert slot.sound_status == "none" and slot.sound_intent == "none"


def test_a_user_sound_choice_is_recorded_as_theirs_and_never_changes_the_visual(pack):
    plan = _plan(registry=pack)
    slot = plan.slots[1]
    before = (slot.treatment, slot.camera, slot.status)
    ed.set_slot_sound(plan, slot, "impact", pack)
    assert slot.sound_locked and slot.sound_intent == "impact"
    assert (slot.treatment, slot.camera, slot.status) == before


def test_unknown_sound_intent_is_rejected():
    plan = _plan()
    with pytest.raises(ValueError):
        ep.set_sound_intent(plan.slots[1], "boom")


def test_syncing_is_deterministic(pack):
    a, b = _plan(registry=pack), _plan(registry=pack)
    assert a.model_dump() == b.model_dump()
    ed.sync_sound(a, pack)
    assert a.model_dump() == b.model_dump()


def test_a_silent_profile_keeps_every_slot_silent(pack):
    plan = _plan("none", pack)
    assert all(s.sound_status in {"none", "suppressed", "unavailable_fallback_none"} for s in plan.slots)
    assert not any(s.sound_status == "scheduled" for s in plan.slots)


# retargeting after the reviewer changes the visual ---------------------------------------------------


def test_changing_the_treatment_updates_speaker_camera_and_captions():
    plan = _plan()
    slot = plan.slots[1]
    ep.set_treatment(slot, "motion_graphic")
    ed.retarget(plan, slot)
    assert slot.speaker_visible is False and slot.camera == "n/a" and slot.caption_behavior == "reduced"
    ep.set_treatment(slot, "stay_on_speaker")
    ed.retarget(plan, slot)
    assert slot.speaker_visible is True and slot.camera == "static" and slot.caption_behavior == "normal"
    ep.set_treatment(slot, "punch_in")
    ed.retarget(plan, slot)
    assert slot.camera == "punch_in" and slot.speaker_visible is True


def test_retarget_ignores_slots_that_were_not_directed():
    slot = ep.EditPlanSlot(timeline_start=0, timeline_end=1, treatment="punch_in", recommended="punch_in")
    plan = ep.EditPlan(slots=[slot])
    before = slot.model_dump()
    assert ed.retarget(plan, slot) is slot and slot.model_dump() == before


# grammar ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("message, number, intent", [
    ("4 من غير sound", 4, "none"),
    ("4 no sound", 4, "none"),
    ("4 sound accent", 4, "accent"),
    ("4 sound accent بدل transition", 4, "accent"),
    ("4 sound accent instead of transition", 4, "accent"),
    ("2 sfx impact", 2, "impact"),
    ("3 صوت خفيف", 3, "subtle_motion"),
])
def test_sound_grammar(message, number, intent):
    got = cmd.parse(message)
    assert got.kind == cmd.KIND_SET_SOUND and got.numbers == (number,) and got.sound == intent


@pytest.mark.parametrize("message", ["4 موافق", "4 خليه speaker", "اعتمد الباقي", "4 transition", "4 بلاش"])
def test_visual_replies_are_not_mistaken_for_sound(message):
    assert cmd.find_sound(cmd.parse(message).raw if hasattr(cmd.parse(message), "raw") else message) is None or \
        cmd.parse(message).kind != cmd.KIND_SET_SOUND


@pytest.mark.parametrize("message, treatment", [
    ("2 illustration", "illustration"),
    ("2 full screen text", "full_screen_text_scene"),
    ("2 data graphic", "graphic_data_scene"),
])
def test_new_visual_treatments_are_understood(message, treatment):
    got = cmd.parse(message)
    assert got.kind == cmd.KIND_SET_TREATMENT and got.treatment == treatment


# chat ------------------------------------------------------------------------------------------------


def _chat(tmp_path: Path, lang: str = "en", plan: ep.EditPlan | None = None) -> tuple[EditPlanChat, Path]:
    edit = _project(tmp_path, plan or _plan())
    return EditPlanChat(edit, lang=lang), edit


def _has_every_review_line(text: str) -> None:
    for head in ("Visual:", "Speaker:", "Camera:", "Transition:", "Sound intent:", "SFX availability:", "Caption behavior:"):
        assert head in text, head


def test_a_slot_shows_the_full_review_block(tmp_path: Path):
    chat, _ = _chat(tmp_path)
    reply = chat.reply("2").text
    _has_every_review_line(reply)


def test_natural_sound_replies_update_the_plan_and_confirm(tmp_path: Path):
    chat, edit = _chat(tmp_path)
    out = chat.reply("2 من غير sound").text
    assert "2" in out
    assert ep.load_plan(edit / "review").slot(2).sound_intent == "none"  # type: ignore[union-attr]
    out = chat.reply("2 sound accent").text
    slot = ep.load_plan(edit / "review").slot(2)  # type: ignore[union-attr]
    assert slot.sound_intent == "accent" and slot.sound_locked
    assert "accent" in out.lower()


def test_a_sound_reply_never_touches_the_visual_or_generation(tmp_path: Path):
    chat, edit = _chat(tmp_path)
    before = ep.load_plan(edit / "review").slot(3).model_dump()  # type: ignore[union-attr]
    chat.reply("3 sound accent")
    after = ep.load_plan(edit / "review").slot(3).model_dump()  # type: ignore[union-attr]
    for key in ("treatment", "camera", "status", "generation_approved", "speaker_visible"):
        assert after[key] == before[key]
    assert rs.load_review_state(edit / "review").ready_for_final_render is False


def test_keeping_the_speaker_works_in_natural_language(tmp_path: Path):
    chat, edit = _chat(tmp_path)
    chat.reply("3 خليه speaker")
    slot = ep.load_plan(edit / "review").slot(3)  # type: ignore[union-attr]
    assert slot.treatment == "stay_on_speaker" and slot.speaker_visible and slot.camera == "static"
    assert not slot.generation_approved


def test_approving_the_rest_never_approves_generation(tmp_path: Path):
    plan = _plan()
    ep.set_treatment(plan.slots[2], "generated_broll")
    chat, edit = _chat(tmp_path, plan=plan)
    chat.reply("اعتمد الباقي")
    slots = ep.load_plan(edit / "review").slots  # type: ignore[union-attr]
    assert not any(s.generation_approved for s in slots)
    assert rs.load_review_state(edit / "review").ready_for_final_render is False


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_the_reviewer_is_never_asked_for_a_filename(tmp_path: Path, lang: str):
    chat, _ = _chat(tmp_path, lang)
    texts = [chat.reply(m).text for m in ("2", "3", "2 sound accent", "help", "3 sound impact")]
    for text in texts:
        low = text.lower()
        assert ".wav" not in low and ".mp3" not in low and "file name" not in low and "filename" not in low


def test_a_slot_without_direction_has_no_sound_control(tmp_path: Path):
    plan = ep.EditPlan(slots=[ep.EditPlanSlot(number=1, timeline_start=0, timeline_end=3, treatment="punch_in",
                                              recommended="punch_in", spoken_context="x")])
    chat, _ = _chat(tmp_path, plan=plan)
    out = chat.reply("1 sound accent").text
    assert "no sound control" in out


def test_the_saved_plan_file_records_the_sound_profile_and_decisions(tmp_path: Path):
    _, edit = _chat(tmp_path)
    raw = json.loads((edit / "review" / "edit_plan.json").read_text(encoding="utf-8"))
    assert raw["sound_profile"] == "minimal"
    assert {"sound_intent", "sound_status", "camera", "transition", "speaker_visible", "caption_behavior"} <= set(raw["slots"][0])
