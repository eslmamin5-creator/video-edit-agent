"""Chat-first edit-plan / B-roll review: grammar, per-slot state, generation
approval kept separate, text options tied to the approved transcript, and the
final-approval gate. Synthetic content only -- nothing project-specific."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_transcript
from video_edit_agent.broll.treatment import Treatment, TreatmentDecision, load_decisions
from video_edit_agent.cli.main import app
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.review import corrections as corr
from video_edit_agent.review import edit_plan as ep
from video_edit_agent.review import edit_plan_commands as cmd
from video_edit_agent.review import state as rs
from video_edit_agent.review.edit_plan import SlotStatus
from video_edit_agent.review.edit_plan_chat import EditPlanChat, generation_risk, open_edit_plan
from video_edit_agent.review.schemas import TranscriptCorrection
from video_edit_agent.transcription.router import save_transcript

LINES = [
    ("أهلا بيكم في الفيديو", 0.0, 6.0),
    ("الـvalues دي لازم تبقى حقيقية مش كلام", 6.0, 12.0),
    ("بنرتب الأفكار في مجموعات واضحة", 12.0, 18.0),
    ("وبنشتغل مع الفريق كل يوم", 18.0, 24.0),
    ("ولازم الـdemand يبقى واضح قبل أي حاجة", 24.0, 30.0),
]

DECISIONS = [
    {"timeline_start": 0.0, "timeline_end": 6.0, "treatment": "kinetic_typography", "reason": "opening title"},
    {"timeline_start": 6.0, "timeline_end": 12.0, "treatment": "punch_in", "reason": "emphasis"},
    {"timeline_start": 12.0, "timeline_end": 18.0, "treatment": "generated_broll", "reason": "abstract idea",
     "visual_concept": "overhead shot of hands arranging blank cards", "prompt": "Overhead documentary shot.",
     "fallback": "motion_graphic", "alternatives": ["motion_graphic", "local_broll"]},
    {"timeline_start": 18.0, "timeline_end": 24.0, "treatment": "local_broll", "reason": "real team footage",
     "visual_concept": "the team at work", "fallback": "punch_in"},
    {"timeline_start": 24.0, "timeline_end": 30.0, "treatment": "behind_subject_text", "reason": "key term",
     "text_options": ["الـdemand", "واضح قبل", "كلمة مخترعة"]},
]


def _project(tmp_path: Path, decisions=DECISIONS) -> Path:
    edit = tmp_path / "edit"
    (edit / "review").mkdir(parents=True)
    save_transcript(make_transcript(list(LINES)), edit / "transcript_unified.json")
    clips = [
        EDLClip(source_file="a.mp4", source_in=s, source_out=e, timeline_in=s, timeline_out=e, caption_refs=[f"s{i}"])
        for i, (_, s, e) in enumerate(LINES)
    ]
    (edit / "edl.json").write_text(EDL(width=720, height=1280, clips=clips).model_dump_json(), encoding="utf-8")
    (edit / "review" / "broll_editorial.json").write_text(json.dumps({"decisions": decisions}), encoding="utf-8")
    (edit / "review" / "review_state.json").write_text(json.dumps({"text_treatments": [{
        "treatment": "hook_title", "visual_status": "approved", "copy_status": "approved",
    }]}), encoding="utf-8")  # the opening title was approved in the hook review
    return edit


def _chat(edit: Path) -> EditPlanChat:
    open_edit_plan(edit, settled={"punch_in": "approved earlier"})
    return EditPlanChat(edit, lang="en")


def _slot(edit: Path, n: int) -> ep.EditPlanSlot:
    slot = ep.load_plan(edit / "review").slot(n)  # type: ignore[union-attr]
    assert slot is not None
    return slot


# 1. grammar ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("message, kind, numbers", [
    ("1 موافق", cmd.KIND_APPROVE, (1,)),
    ("2 ok", cmd.KIND_APPROVE, (2,)),
    ("اعتمد الباقي", cmd.KIND_APPROVE_REST, ()),
    ("approve the rest", cmd.KIND_APPROVE_REST, ()),
    ("2 بلاش", cmd.KIND_REJECT, (2,)),
    ("3 ولّد", cmd.KIND_APPROVE_GENERATION, (3,)),
    ("3 generate", cmd.KIND_APPROVE_GENERATION, (3,)),
])
def test_grammar_kinds(message, kind, numbers):
    got = cmd.parse(message)
    assert (got.kind, got.numbers) == (kind, numbers)


@pytest.mark.parametrize("message, treatment", [
    ("2 خليه speaker", "stay_on_speaker"),
    ("3 بدل generated اعمله motion graphic", "motion_graphic"),
    ("3 instead of generated make it a motion graphic", "motion_graphic"),
    ("4 استخدم B-roll محلي", "local_broll"),
    ("4 use local b-roll", "local_broll"),
    ("1 خليه punch-in", "punch_in"),
    ("5 اعمله kinetic typography", "kinetic_typography"),
    ("5 نص ورا المتحدث", "behind_subject_text"),
])
def test_grammar_treatments(message, treatment):
    got = cmd.parse(message)
    assert got.kind == cmd.KIND_SET_TREATMENT and got.treatment == treatment


@pytest.mark.parametrize("message, view", [
    ("وريني بس الحاجات اللي محتاجة asset", cmd.VIEW_ASSET),
    ("وريني الـgenerated فقط", cmd.VIEW_GENERATED),
    ("وريني الكل", cmd.VIEW_ALL),
    ("show only what needs an asset", cmd.VIEW_ASSET),
])
def test_grammar_views(message, view):
    got = cmd.parse(message)
    assert got.kind == cmd.KIND_SHOW and got.view == view


def test_grammar_options_and_text():
    assert cmd.parse("3 اختار الخيار B") == cmd.Command(cmd.KIND_CHOOSE_OPTION, numbers=(3,), option=1)
    assert cmd.parse("اختار الخيار B").option == 1
    assert cmd.parse("5 choose option C").option == 2
    got = cmd.parse("5 النص: value حقيقية")
    assert got.kind == cmd.KIND_SET_TEXT and got.numbers == (5,) and got.text == "value حقيقية"
    assert cmd.parse('5 text "demand"').text == "demand"


def test_negated_generation_is_not_an_approval():
    assert cmd.parse("3 don't generate").kind != cmd.KIND_APPROVE_GENERATION
    assert cmd.parse("3 ما تولدش").kind != cmd.KIND_APPROVE_GENERATION


# 2. building the plan -----------------------------------------------------------------------------------


def test_plan_settles_hook_punch_in_and_numbers_the_rest(tmp_path: Path):
    edit = _project(tmp_path)
    plan = open_edit_plan(edit, settled={"punch_in": "approved in the punch-in review"})
    assert [s.number for s in plan.slots] == [0, 0, 1, 2, 3]
    assert plan.slots[0].settled_reason == "approved in the hook review"
    assert plan.slots[1].settled_reason == "approved in the punch-in review"
    assert all(s.status is SlotStatus.PENDING_REVIEW for s in plan.pending())
    assert len(plan.pending()) == 3


def test_plan_uses_the_approved_transcript_not_the_raw_one(tmp_path: Path):
    edit = _project(tmp_path)
    corr.save_corrections(edit / "review", [
        TranscriptCorrection(segment_id="s2", corrected_text="بنرتب الأفكار في مجموعات تانية"),
    ])
    raw_before = (edit / "transcript_unified.json").read_bytes()
    reply = _chat(edit).reply("1")
    assert "مجموعات تانية" in reply.text
    assert (edit / "transcript_unified.json").read_bytes() == raw_before


def test_text_options_must_come_from_the_spoken_sentence(tmp_path: Path):
    edit = _project(tmp_path)
    plan = open_edit_plan(edit)
    text_slot = next(s for s in plan.slots if s.treatment == "behind_subject_text")
    assert text_slot.text_options == ["الـdemand", "واضح قبل"]  # the invented option is dropped


def test_text_options_are_derived_when_the_planner_gave_none(tmp_path: Path):
    decisions = [{"timeline_start": 24.0, "timeline_end": 30.0, "treatment": "behind_subject_text"}]
    plan = open_edit_plan(_project(tmp_path, decisions))
    assert plan.slots[0].text_options and all(ep.occurs_in(o, LINES[4][0]) for o in plan.slots[0].text_options)


def test_generated_slot_shows_concept_prompt_and_rules_without_generating(tmp_path: Path):
    edit = _project(tmp_path)
    text = _chat(edit).reply("show 1").text
    assert "Needs asset: YES" in text and "AI generation: YES" in text
    assert "overhead shot of hands arranging blank cards" in text
    assert "Overhead documentary shot." in text and "no readable text" in text.lower()
    assert "AI-artifact risk:" in text and "nothing has been generated" in text
    assert "Generation: NOT approved" in text
    assert not (edit / ".cache").exists() and not list(edit.glob("**/*.mp4"))


def test_generation_risk_heuristic():
    assert generation_risk("a person talking to camera") == "high"
    assert generation_risk("overhead shot of hands arranging blank cards") == "medium"
    assert generation_risk("slow drifting light through a window") == "low"


# 3. decisions -------------------------------------------------------------------------------------------


def test_approving_a_generated_slot_does_not_approve_generation(tmp_path: Path):
    edit = _project(tmp_path)
    reply = _chat(edit).reply("1 موافق")
    slot = _slot(edit, 1)
    assert slot.status is SlotStatus.APPROVED and slot.generation_approved is False
    assert "NOT approved" in reply.text and reply.changed
    assert slot.effective_treatment == "motion_graphic"  # the fallback until generation is allowed


def test_explicit_generation_approval_is_per_slot_and_only_for_generated(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    chat.reply("1 ولّد")
    assert _slot(edit, 1).generation_approved and _slot(edit, 1).status is SlotStatus.GENERATION_APPROVED
    assert _slot(edit, 1).effective_treatment == "generated_broll"
    assert "nothing to allow" in chat.reply("2 generate").text
    assert _slot(edit, 2).generation_approved is False
    # the review state is untouched: nothing here approves the render
    assert rs.load_review_state(edit / "review").ready_for_final_render is False


def test_switching_away_from_generated_clears_generation_approval(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    chat.reply("1 generate")
    chat.reply("1 بدل generated اعمله motion graphic")
    slot = _slot(edit, 1)
    assert slot.treatment == "motion_graphic" and slot.status is SlotStatus.CHANGED
    assert slot.generation_approved is False and slot.recommended == "generated_broll"


def test_local_broll_never_blocks_and_uses_its_fallback(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    text = chat.reply("show 2").text
    assert "Footage needed: the team at work" in text and "If you have none: Punch-in / reframe" in text
    chat.reply("2 موافق")
    slot = _slot(edit, 2)
    assert slot.status is SlotStatus.ASSET_REQUIRED and not slot.blocking
    assert slot.effective_treatment == "punch_in"


def test_text_slot_needs_exact_text_before_approval(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    reply = chat.reply("3 موافق")
    assert "exact text" in reply.text and "A) الـdemand" in reply.text
    assert _slot(edit, 3).status is SlotStatus.PENDING_REVIEW
    chat.reply("3 اختار الخيار B")
    slot = _slot(edit, 3)
    assert slot.text == "واضح قبل" and slot.status is SlotStatus.APPROVED


def test_typed_text_is_kept_verbatim_and_flagged_when_not_spoken(tmp_path: Path):
    edit = _project(tmp_path)
    reply = EditPlanChat(_prepared(edit), lang="en").reply("3 النص: كلمة ما اتقالتش")
    slot = _slot(edit, 3)
    assert slot.text == "كلمة ما اتقالتش" and slot.status is SlotStatus.APPROVED
    assert "not from the spoken sentence" in reply.text


def _prepared(edit: Path) -> Path:
    open_edit_plan(edit, settled={"punch_in": "approved earlier"})
    return edit


def test_reject_and_no_treatment_fall_back_to_speaker_and_stay_rejected(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    chat.reply("1 بلاش")
    chat.reply("1 موافق")  # approving afterwards must not resurrect the treatment
    slot = _slot(edit, 1)
    assert slot.status is SlotStatus.REJECTED and slot.effective_treatment == "stay_on_speaker"


def test_approve_rest_skips_generation_and_unresolved_text(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    reply = chat.reply("اعتمد الباقي")
    assert _slot(edit, 1).status is SlotStatus.APPROVED and not _slot(edit, 1).generation_approved
    assert _slot(edit, 2).status is SlotStatus.ASSET_REQUIRED
    assert _slot(edit, 3).status is SlotStatus.PENDING_REVIEW  # text still to choose
    assert "3" in reply.text and "exact text" in reply.text.lower()


def test_multiple_entries_in_one_message(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    chat.reply("1 بدل generated اعمله motion graphic\n2 استخدم B-roll محلي\n3 اختار الخيار A")
    assert _slot(edit, 1).treatment == "motion_graphic"
    assert _slot(edit, 2).status is SlotStatus.ASSET_REQUIRED
    assert _slot(edit, 3).text == "الـdemand"


def test_filters(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    asset = chat.reply("وريني بس الحاجات اللي محتاجة asset").text
    assert "[1]" in asset and "[2]" in asset and "[3]" not in asset
    generated = chat.reply("وريني الـgenerated فقط").text
    assert "[1]" in generated and "[2]" not in generated
    everything = chat.reply("وريني الكل").text
    assert "[–]" in everything and "approved earlier" in everything


def test_unknown_number_and_ambiguous_ok_are_answered_not_applied(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    assert "no decision 9" in chat.reply("9 موافق").text.lower()
    assert not chat.reply("موافق").changed  # three pending slots: which one?
    assert all(s.status is SlotStatus.PENDING_REVIEW for s in ep.load_plan(edit / "review").pending())


def test_bare_ok_applies_to_the_slot_just_shown(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    chat.reply("2 show")
    assert chat.reply("موافق").changed is False or _slot(edit, 2).status is not SlotStatus.PENDING_REVIEW


# 4. gate and pipeline -----------------------------------------------------------------------------------


def test_final_approval_is_refused_while_a_slot_is_pending(tmp_path: Path):
    edit = _project(tmp_path)
    _chat(edit)
    with pytest.raises(rs.UnresolvedReviewItems, match="edit-plan"):
        rs.approve(edit / "review")
    assert rs.load_review_state(edit / "review").ready_for_final_render is False


def test_final_approval_passes_once_every_slot_is_decided(tmp_path: Path):
    edit = _project(tmp_path)
    chat = _chat(edit)
    chat.reply("1 بلاش\n2 موافق\n3 اختار الخيار A")
    assert rs.approve(edit / "review").ready_for_final_render is True


def test_approve_is_unchanged_without_an_edit_plan(tmp_path: Path):
    review = tmp_path / "review"
    review.mkdir()
    assert rs.approve(review).ready_for_final_render is True


def test_effective_decisions_never_generate_without_explicit_approval(tmp_path: Path):
    edit = _project(tmp_path)
    review = edit / "review"
    decisions = load_decisions(review)
    assert effective(review, decisions)[2].treatment is Treatment.GENERATED_BROLL  # no plan: untouched
    chat = _chat(edit)
    chat.reply("1 موافق")
    assert effective(review, decisions)[2].treatment is Treatment.MOTION_GRAPHIC  # fallback
    chat.reply("1 ولّد")
    assert effective(review, decisions)[2].treatment is Treatment.GENERATED_BROLL
    chat.reply("1 بلاش")
    assert effective(review, decisions)[2].treatment is Treatment.STAY_ON_SPEAKER


def effective(review: Path, decisions: list[TreatmentDecision]) -> list[TreatmentDecision]:
    return ep.effective_decisions(review, decisions)


# 5. CLI -------------------------------------------------------------------------------------------------


def test_review_plan_cli_opens_and_answers(tmp_path: Path):
    edit = _project(tmp_path)
    runner = CliRunner()
    opened = runner.invoke(app, ["review-plan", str(edit), "--open", "--lang", "en", "--settled", "punch_in=approved earlier"])
    assert opened.exit_code == 0, opened.output
    assert "Recommended:" in opened.output and "Status: PENDING" in opened.output
    answered = runner.invoke(app, ["review-plan", str(edit), "1 generate", "--lang", "en"])
    assert answered.exit_code == 0 and "generation approved" in answered.output
    assert _slot(edit, 1).generation_approved
    assert rs.load_review_state(edit / "review").ready_for_final_render is False
