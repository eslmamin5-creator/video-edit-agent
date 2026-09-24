"""Hook-copy approval rule: raw/unresolved ASR is never promoted into hook copy,
and the look of a text treatment is approved separately from its words.
Synthetic brand, transcript and text throughout -- nothing here is project-specific."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_transcript
from video_edit_agent.brand.schema import Brand
from video_edit_agent.cli.main import app
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.motion.director import build_motion_plan, plan_hook
from video_edit_agent.review import state as rs
from video_edit_agent.review import text_copy
from video_edit_agent.review.schemas import (
    ApprovalStatus,
    CopySource,
    ReviewApprovalState,
    UnresolvedTranscriptItem,
)

W, H = 720, 1280
ASR = "raw asr words the reviewer has not confirmed"
PLACEHOLDER = "HOOK TEXT - PENDING REVIEW"


def _brand() -> Brand:
    return Brand.model_validate({"name": "acme", "colors": {"primary": "#123456", "secondary": "#FEDCBA", "accent": "#22AA66"}})


def _edl() -> EDL:
    return EDL(width=W, height=H, clips=[
        EDLClip(source_file="a.mp4", source_in=0.0, source_out=8.0, timeline_in=0.0, timeline_out=8.0),
    ])


def _transcript():
    return make_transcript([(ASR, 0.0, 3.0), ("second segment that is fine", 3.2, 6.0)])


def _open(segment_id: str = "s0", n: int = 1) -> UnresolvedTranscriptItem:
    return UnresolvedTranscriptItem(segment_id=segment_id, segment=n, asr_text=ASR, reason="not confirmed")


def _state(*open_items: UnresolvedTranscriptItem) -> ReviewApprovalState:
    return ReviewApprovalState(unresolved_transcript=list(open_items))


def _plan(state: ReviewApprovalState, *, final: bool = False):
    return plan_hook(_edl(), _transcript(), _brand(), None, state, for_final_render=final)


def _recorded(tmp_path: Path, *open_items: UnresolvedTranscriptItem) -> Path:
    """A review dir whose state holds the planned hook record."""
    for item in open_items:
        rs.flag_unresolved(tmp_path, item)
    rs.record_text_treatment(tmp_path, _plan(rs.load_review_state(tmp_path)).review)
    return tmp_path


# 1. unresolved ASR cannot become final hook copy automatically ------------------------------------


def test_unresolved_asr_never_becomes_hook_copy_in_a_final_render():
    plan = _plan(_state(_open()), final=True)
    assert plan.spec is None  # no copy layer at all
    assert plan.review.copy_status is ApprovalStatus.PENDING_REVIEW
    assert plan.review.approved_copy is None
    assert plan.review.blocking_reason == "unresolved transcript"
    specs = build_motion_plan(_edl(), _transcript(), brand=_brand(), hook=plan)
    assert not [s for s in specs if s.kind.value == "hook_title"]


def test_review_preview_shows_the_marked_placeholder_not_the_asr_text():
    plan = _plan(_state(_open()))
    assert plan.spec.text == PLACEHOLDER
    assert ASR.split()[0] not in plan.spec.text
    assert plan.review.placeholder_used_in_preview is True
    assert plan.review.visual_status is ApprovalStatus.PENDING_REVIEW  # the plan is kept, just not approved yet
    assert plan.spec.extra["color"]  # the visual treatment is still planned


def test_pending_copy_reports_its_source_segment_and_blocking_reason():
    review = _plan(_state(_open())).review
    assert review.source_segment_ids == ["s0"]
    assert review.source_segments == [1]
    assert review.blocking_reason == "unresolved transcript"
    assert review.proposed_asr_text == ASR  # kept for the reviewer only


def test_confirmed_transcript_text_is_valid_hook_copy():
    plan = _plan(_state(_open("s1", 2)), final=True)  # s1 is open, but the hook comes from s0
    assert plan.spec.text == ASR
    assert plan.review.copy_source is CopySource.APPROVED_TRANSCRIPT


def test_a_segment_flagged_after_copy_was_taken_from_it_withdraws_the_copy(tmp_path: Path):
    rs.record_text_treatment(tmp_path, _plan(_state()).review)
    assert rs.load_review_state(tmp_path).hook_copy_status == "approved"
    rs.flag_unresolved(tmp_path, _open())
    after = rs.load_review_state(tmp_path)
    assert after.hook_copy_status == "pending_review"
    assert after.treatment("hook_title").approved_copy is None


# 2. approved visual + pending copy is a valid review state ------------------------------------------


def test_approved_visual_with_pending_copy_is_a_valid_persisted_state(tmp_path: Path):
    _recorded(tmp_path, _open())
    state = rs.approve_visual(tmp_path, "hook_title", {"color": "#FEDCBA", "centerY": 400, "maxWidth": 600})
    raw = json.loads((tmp_path / "review_state.json").read_text(encoding="utf-8"))
    hook = raw["text_treatments"][0]
    assert hook["visual_status"] == "approved"
    assert hook["copy_status"] == "pending_review"
    assert raw["hook_copy_status"] == "pending_review"
    assert hook["source_segments"] == [1]
    assert hook["blocking_reason"] == "unresolved transcript"
    assert hook["placeholder_used_in_preview"] is True
    assert raw["ready_for_final_render"] is False
    assert rs.load_review_state(tmp_path) == state  # round-trips


def test_pending_hook_copy_blocks_final_approval(tmp_path: Path):
    _recorded(tmp_path, _open())
    rs.approve_visual(tmp_path, "hook_title")
    rs.resolve_unresolved(tmp_path, "s0")  # the transcript is no longer the obstacle...
    with pytest.raises(rs.UnresolvedReviewItems, match="hook_title"):  # ...the hook copy still is
        rs.approve(tmp_path)
    assert rs.is_ready_for_final_render(tmp_path) is False


# 3. correcting hook copy does not reset visual approval ----------------------------------------------


def test_correcting_the_copy_keeps_the_visual_approval_and_props(tmp_path: Path):
    _recorded(tmp_path, _open())
    props = {"color": "#FEDCBA", "centerY": 400, "maxWidth": 600, "plate": None, "outline": None, "shadow": ""}
    rs.approve_visual(tmp_path, "hook_title", props)
    state = rs.submit_copy(tmp_path, "hook_title", "Approved words")
    hook = state.treatment("hook_title")
    assert hook.visual_status is ApprovalStatus.APPROVED
    assert hook.visual_props == props
    assert hook.copy_status is ApprovalStatus.APPROVED
    assert hook.blocking_reason is None and hook.placeholder_used_in_preview is False


def test_a_copy_change_reuses_the_approved_visual_instead_of_redesigning(tmp_path: Path):
    _recorded(tmp_path, _open())
    props = {"color": "#FEDCBA", "centerY": 360, "maxWidth": 600, "plate": None, "outline": None, "shadow": ""}
    rs.approve_visual(tmp_path, "hook_title", props)
    rs.submit_copy(tmp_path, "hook_title", "Short words")
    plan = _plan(rs.load_review_state(tmp_path))
    assert plan.spec.extra == props  # same colour, placement, width, plate -- nothing re-derived
    assert plan.review.layout_fit_issue is None


def test_new_text_that_does_not_fit_moves_only_the_minimum_and_says_so():
    props = {"color": "#FEDCBA", "centerY": 1000, "maxWidth": 600, "plate": None, "outline": None, "shadow": "s"}
    fit = text_copy.refit_copy(props, "words " * 6, W, H)
    assert fit.issue and fit.adjusted == ["centerY"]
    assert {k: v for k, v in fit.props.items() if k != "centerY"} == {k: v for k, v in props.items() if k != "centerY"}
    assert fit.props["centerY"] < props["centerY"]
    huge = text_copy.refit_copy({**props, "centerY": 400}, "words " * 200, W, H)
    assert "taller than" in huge.issue and huge.props["color"] == props["color"]


# 4. user-supplied / approved copy replaces the placeholder -------------------------------------------


@pytest.mark.parametrize("source", [CopySource.USER_SUPPLIED, CopySource.APPROVED_REWRITE])
def test_user_copy_replaces_the_placeholder_in_preview_and_final(tmp_path: Path, source: CopySource):
    _recorded(tmp_path, _open())
    rs.submit_copy(tmp_path, "hook_title", "The words the user chose", source)
    state = rs.load_review_state(tmp_path)
    for final in (False, True):
        plan = _plan(state, final=final)
        assert plan.spec.text == "The words the user chose"
        assert plan.review.copy_source is source
        assert plan.review.placeholder_used_in_preview is False
    assert state.hook_copy_status == "approved"


def test_user_copy_wins_even_though_the_transcript_segment_is_still_unresolved(tmp_path: Path):
    _recorded(tmp_path, _open())
    rs.submit_copy(tmp_path, "hook_title", "Chosen")
    assert _plan(rs.load_review_state(tmp_path), final=True).spec.text == "Chosen"
    # ...but the unresolved transcript itself still blocks overall approval
    with pytest.raises(rs.UnresolvedReviewItems, match="unresolved transcript"):
        rs.approve(tmp_path)


def test_with_everything_confirmed_final_approval_goes_through(tmp_path: Path):
    _recorded(tmp_path, _open())
    rs.submit_copy(tmp_path, "hook_title", "Chosen")
    rs.resolve_unresolved(tmp_path, "s0")
    assert rs.approve(tmp_path).ready_for_final_render is True


def test_copy_from_the_transcript_cannot_be_submitted_as_user_copy(tmp_path: Path):
    _recorded(tmp_path, _open())
    with pytest.raises(ValueError):
        rs.submit_copy(tmp_path, "hook_title", "x", CopySource.APPROVED_TRANSCRIPT)
    with pytest.raises(ValueError):
        rs.submit_copy(tmp_path, "hook_title", "   ")
    with pytest.raises(KeyError):
        rs.submit_copy(tmp_path, "no_such_treatment", "x")


# 5. raw ASR remains unchanged ---------------------------------------------------------------------------


def test_raw_asr_is_never_modified_by_planning_or_copy_approval(tmp_path: Path):
    transcript = _transcript()
    before = transcript.model_dump_json()
    item = _open()
    state = _state(item)
    plan_hook(_edl(), transcript, _brand(), None, state)
    _recorded(tmp_path, _open())
    rs.approve_visual(tmp_path, "hook_title")
    rs.submit_copy(tmp_path, "hook_title", "Different words entirely")
    plan_hook(_edl(), transcript, _brand(), None, rs.load_review_state(tmp_path), for_final_render=True)
    assert transcript.model_dump_json() == before
    assert item.asr_text == ASR
    kept = rs.load_review_state(tmp_path)
    assert kept.treatment("hook_title").proposed_asr_text == ASR  # still the reviewer's reference
    assert kept.unresolved_transcript[0].asr_text == ASR


# 6. nothing project-specific is required ---------------------------------------------------------------


def test_the_rule_needs_no_brand_and_names_none():
    plan = plan_hook(_edl(), _transcript(), None, None, _state(_open()))
    assert plan.spec.text == PLACEHOLDER and plan.spec.extra == {}
    assert plan.review.blocking_reason == "unresolved transcript"


def test_it_works_for_other_treatments_and_other_languages():
    review = text_copy.decide_copy(
        "lower_third", "Ceci est du texte", ["s7"], unresolved=[_open("s7", 8)], source_segments=[8],
    ).review
    assert review.copy_status is ApprovalStatus.PENDING_REVIEW
    assert review.placeholder_text == "LOWER TEXT - PENDING REVIEW"
    ok = text_copy.decide_copy("lower_third", "Ceci est du texte", ["s7"], unresolved=[])
    assert ok.final_text == "Ceci est du texte"


def test_logic_modules_contain_no_project_specific_values():
    root = Path(text_copy.__file__).parent
    for name in ("text_copy.py", "state.py", "schemas.py"):
        source = (root / name).read_text(encoding="utf-8").lower().replace("neutral", "")
        for needle in ("#0a3d62", "#8e44ad", "#f1c40f", "d:\\", "segment 1", '"s0"'):
            assert needle not in source, (name, needle)


# CLI -------------------------------------------------------------------------------------------------


def test_cli_approves_visual_and_copy_separately(tmp_path: Path):
    edit = tmp_path / "edit"
    review = edit / "review"
    _recorded(review, _open())
    (edit / "motion_plan.json").write_text(json.dumps([
        {"spec": {"kind": "hook_title", "extra": {"color": "#FEDCBA", "centerY": 400}}},
    ]), encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(app, ["review-approve-visual", str(edit)]).exit_code == 0
    hook = rs.load_review_state(review).treatment("hook_title")
    assert hook.visual_status is ApprovalStatus.APPROVED and hook.copy_status is ApprovalStatus.PENDING_REVIEW
    assert hook.visual_props == {"color": "#FEDCBA", "centerY": 400}
    assert runner.invoke(app, ["review-copy", str(edit), "Words from the user"]).exit_code == 0
    hook = rs.load_review_state(review).treatment("hook_title")
    assert hook.visual_status is ApprovalStatus.APPROVED and hook.approved_copy == "Words from the user"
