"""Chat-first transcript review: presenter, natural-language grammar, corrections,
approval states, audio on demand, paging, and the hook-copy separation.
Synthetic Arabic/English text throughout -- nothing here is project-specific."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_transcript
from video_edit_agent.brand.schema import Brand
from video_edit_agent.cli.main import app
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.motion.director import plan_hook
from video_edit_agent.review import audio_clip
from video_edit_agent.review import chat_commands as cmd
from video_edit_agent.review import corrections as corr
from video_edit_agent.review import state as rs
from video_edit_agent.review import transcript_chat as tc
from video_edit_agent.review.chat_session import TranscriptReviewChat, open_transcript_review
from video_edit_agent.review.schemas import (
    ApprovalStatus,
    ReviewApprovalState,
    SegmentReviewStatus,
)
from video_edit_agent.transcription.router import load_transcript, save_transcript

S = SegmentReviewStatus
LINES = [
    ("إزاي بقى إنت كأجينسي تبقى فاهم إن الـvalues دي حقيقية", 0.0, 6.5),
    ("ما تكونش آراء شخصية من صاحب البراند نفسه", 6.5, 10.4),
    ("الأخمس أضلاع خيف سيد", 10.4, 12.3),
    ("أنا بحب الـservice بتاعتكم جدًا", 12.3, 15.0),
    ("ده كلام سليم تمامًا", 15.0, 17.0),
]


def _project(tmp_path: Path, *, low: dict[int, list[int]] | None = None, lines=LINES) -> Path:
    """An edit/ dir with a raw transcript; `low` maps segment index -> word indexes with low confidence."""
    edit = tmp_path / "edit"
    edit.mkdir()
    transcript = make_transcript(list(lines))
    for seg_i, word_is in (low if low is not None else {2: [2]}).items():
        for wi in word_is:
            transcript.segments[seg_i].words[wi].confidence = 0.54
    save_transcript(transcript, edit / "transcript_unified.json")
    return edit


def _chat(edit: Path, **kw) -> TranscriptReviewChat:
    return TranscriptReviewChat(edit, **kw)


def _raw_bytes(edit: Path) -> bytes:
    return (edit / "transcript_unified.json").read_bytes()


# 1. numbered, human-readable presenter ---------------------------------------------------------------


def test_presenter_shows_numbered_segments_with_timestamps_and_status(tmp_path: Path):
    edit = _project(tmp_path)
    text = _chat(edit).reply("الكل").text
    assert "[01] 00:00.00–00:06.50" in text
    assert "[03] 00:10.40–00:12.30" in text
    assert "Status: APPROVED" in text and "Status: NEEDS REVIEW" in text
    assert LINES[2][0] in text
    assert not re.search(r"\bs\d+\b", text)  # internal ids are never shown


def test_low_confidence_words_are_listed_with_their_score(tmp_path: Path):
    text = _chat(_project(tmp_path)).reply().text
    assert "Low-confidence: خيف (0.54)" in text


# 2. low-confidence filter -------------------------------------------------------------------------------


def test_default_view_and_suspicious_request_show_only_open_segments(tmp_path: Path):
    edit = _project(tmp_path)
    for message in (None, "وريني بس الجمل اللي فيها شك", "show only suspicious"):
        text = _chat(edit).reply(message).text
        assert "[03]" in text
        assert "[01]" not in text and "[05]" not in text


def test_review_specific_numbers_only(tmp_path: Path):
    text = _chat(_project(tmp_path)).reply("راجعلي 1 و4 بس").text
    assert "[01]" in text and "[04]" in text and "[02]" not in text and "[03]" not in text


# 3. correction by segment number persists ----------------------------------------------------------------


def test_sentence_correction_by_number_persists_and_approves(tmp_path: Path):
    edit = _project(tmp_path)
    open_transcript_review(edit / "review", load_transcript(edit / "transcript_unified.json"))
    fixed = "الأضلاع الخمس خيرة سيد"
    reply = _chat(edit).reply(f"الجملة 3: {fixed}")
    assert reply.changed
    saved = corr.load_corrections(edit / "review")
    assert [(c.segment_id, c.corrected_text, c.word_index) for c in saved] == [("s2", fixed, None)]
    state = rs.load_review_state(edit / "review")
    assert state.unresolved_transcript == []
    assert [(d.segment_id, d.status) for d in state.segment_reviews] == [("s2", S.APPROVED)]


def test_one_message_can_answer_several_segments_one_per_line(tmp_path: Path):
    """Regression: `9: ...\\n\\n14: ...` used to be stored whole as the text of segment 9."""
    edit = _project(tmp_path, low={2: [2], 3: [1], 4: [0]})
    open_transcript_review(edit / "review", load_transcript(edit / "transcript_unified.json"))
    reply = _chat(edit).reply("3: الأضلاع الخمس، Five Cs.\n\n4: أنا بحب الـservice جدًا.\n\n5: ده كلام سليم، تمامًا.")
    saved = {c.segment_id: c.corrected_text for c in corr.load_corrections(edit / "review")}
    assert saved == {"s2": "الأضلاع الخمس، Five Cs.", "s3": "أنا بحب الـservice جدًا.", "s4": "ده كلام سليم، تمامًا."}
    assert not any("\n" in t or re.search(r"\d\s*:", t) for t in saved.values())
    state = rs.load_review_state(edit / "review")
    assert state.unresolved_transcript == []
    assert {d.segment_id: d.status for d in state.segment_reviews} == {"s2": S.APPROVED, "s3": S.APPROVED, "s4": S.APPROVED}
    assert reply.changed
    for n in (3, 4, 5):
        assert f"[{n:02d}]" in reply.text  # one confirmation per segment


def test_split_message_only_cuts_at_numbered_lines():
    assert cmd.split_message("9: الجملة الأولى\n14: الجملة التانية") == ["9: الجملة الأولى", "14: الجملة التانية"]
    assert cmd.split_message("٩ صح\n١٤ صح") == ["٩ صح", "١٤ صح"]
    assert cmd.split_message("9: جملة طويلة\nبتكمل في سطر تاني\n14 صح") == ["9: جملة طويلة بتكمل في سطر تاني", "14 صح"]
    assert cmd.split_message("9: جملة واحدة\nبس على سطرين") == ["9: جملة واحدة\nبس على سطرين"]
    assert cmd.split_message("9 و14 و15 صح") == ["9 و14 و15 صح"]


# 4. word-level replacement persists -----------------------------------------------------------------------


def test_word_replacement_changes_only_that_word_and_keeps_timing(tmp_path: Path):
    edit = _project(tmp_path)
    before = load_transcript(edit / "transcript_unified.json").segments[3]
    _chat(edit).reply("في الجملة 4 غير service إلى سيرفس")
    after = corr.apply_corrections(
        load_transcript(edit / "transcript_unified.json"), corr.load_corrections(edit / "review"),
    ).segments[3]
    assert after.text == "أنا بحب الـسيرفس بتاعتكم جدًا"  # only the word changed; the attached "الـ" stays
    assert (after.start, after.end) == (before.start, before.end)
    assert len(after.words) == len(before.words)
    assert [(w.start, w.end) for w in after.words] == [(w.start, w.end) for w in before.words]
    assert [w.word for i, w in enumerate(after.words) if i != 2] == [w.word for i, w in enumerate(before.words) if i != 2]


def test_word_replacement_without_a_number_finds_the_only_segment_that_has_it(tmp_path: Path):
    edit = _project(tmp_path)
    reply = _chat(edit).reply("غير service إلى سيرفس")
    assert reply.changed and "[04]" in reply.text


def test_ambiguous_or_missing_word_is_not_guessed(tmp_path: Path):
    edit = _project(tmp_path, lines=[("a b a", 0.0, 3.0), ("c d", 3.0, 5.0)], low={})
    ambiguous = _chat(edit).reply("في الجملة 1 غير a إلى x")
    assert not ambiguous.changed and corr.load_corrections(edit / "review") == []
    missing = _chat(edit).reply("غير zzz إلى x")
    assert not missing.changed and corr.load_corrections(edit / "review") == []


def test_replace_phrase_keeps_surrounding_punctuation():
    assert corr.replace_phrase("قال، service هي الأهم.", "service", "سيرفس") == ("قال، سيرفس هي الأهم.", 1)
    assert corr.replace_phrase("the value, really", "value", "قيمة") == ("the قيمة, really", 1)


# 5. sentence-level replacement ----------------------------------------------------------------------------


def test_sentence_replacement_preserves_segment_span_and_leaves_others_alone(tmp_path: Path):
    edit = _project(tmp_path)
    new = "كلام جديد بالكامل هنا خلاص يعني"
    _chat(edit).reply(f"5: {new}")
    effective = corr.apply_corrections(
        load_transcript(edit / "transcript_unified.json"), corr.load_corrections(edit / "review"),
    )
    raw = load_transcript(edit / "transcript_unified.json")
    assert effective.segments[4].text == new
    assert (effective.segments[4].start, effective.segments[4].end) == (raw.segments[4].start, raw.segments[4].end)
    assert [s.text for s in effective.segments[:4]] == [s.text for s in raw.segments[:4]]


def test_a_new_sentence_replaces_earlier_word_level_corrections(tmp_path: Path):
    edit = _project(tmp_path)
    _chat(edit).reply("في الجملة 4 غير service إلى سيرفس")
    _chat(edit).reply("4: الجملة الجديدة كلها")
    saved = corr.load_corrections(edit / "review")
    assert [(c.segment_id, c.word_index) for c in saved] == [("s3", None)]


# 6. raw ASR untouched ----------------------------------------------------------------------------------------


def test_no_chat_action_modifies_the_raw_transcript(tmp_path: Path):
    edit = _project(tmp_path)
    before = _raw_bytes(edit)
    chat = _chat(edit)
    for message in ("3: جملة", "في الجملة 4 غير service إلى سيرفس", "1 صح", "اعتمد الباقي", "التالي"):
        chat.reply(message)
    assert _raw_bytes(edit) == before


# 7. dialect and code-switching stay verbatim -----------------------------------------------------------------


@pytest.mark.parametrize("sentence", [
    "إزاي بقى إنت كأجينسي تبقى فاهم إن الـvalues اللي بتتقال دي حاجات حقيقية؟",
    "مش هيبقى كده خالص يا عم، ده بيتعمل مع الـ brand بتاعنا",
    "we need the زبون to feel it، مش كده؟",
])
def test_dictated_text_is_stored_verbatim(tmp_path: Path, sentence: str):
    edit = _project(tmp_path)
    assert cmd.parse(f"الجملة 2: {sentence}").text == sentence
    _chat(edit).reply(f"الجملة 2: {sentence}")
    assert corr.load_corrections(edit / "review")[0].corrected_text == sentence


def test_replacement_words_are_used_verbatim():
    c = cmd.parse("في الجملة 9 غير كلمة service إلى سيرفس")
    assert (c.kind, c.numbers, c.old, c.new) == (cmd.KIND_REPLACE_WORDS, (9,), "service", "سيرفس")
    assert cmd.parse("غير service إلى سيرفس").numbers == ()


# 8. states survive reload ---------------------------------------------------------------------------------------


def test_states_survive_a_fresh_session(tmp_path: Path):
    edit = _project(tmp_path)
    open_transcript_review(edit / "review", load_transcript(edit / "transcript_unified.json"))
    _chat(edit).reply("3: جملة مصححة")
    _chat(edit).reply("في الجملة 4 غير service إلى سيرفس")
    views = {v.number: v for v in _chat(edit)._views()}  # a new object: everything is read from disk
    assert views[3].status is S.APPROVED and views[3].text == "جملة مصححة"
    assert views[4].status is S.CORRECTED_PENDING_APPROVAL and "سيرفس" in views[4].text
    assert views[1].status is S.APPROVED


def test_paging_cursor_survives_a_fresh_session(tmp_path: Path):
    edit = _project(tmp_path, lines=[(f"جملة رقم {i}", i, i + 1) for i in range(12)])
    _chat(edit, page_size=4).reply("الكل")
    text = _chat(edit, page_size=4).reply("التالي").text
    assert "[05]" in text and "[01]" not in text


# 9. unresolved / pending block approval -------------------------------------------------------------------------


def test_unresolved_segments_block_final_approval(tmp_path: Path):
    edit = _project(tmp_path)
    flagged = open_transcript_review(edit / "review", load_transcript(edit / "transcript_unified.json"))
    assert flagged == [3]
    assert tc.blocking_numbers(_chat(edit)._views()) == [3]
    with pytest.raises(rs.UnresolvedReviewItems):
        rs.approve(edit / "review")
    _chat(edit).reply("3 صح")
    assert rs.approve(edit / "review").ready_for_final_render is True


def test_a_word_fix_keeps_blocking_until_the_user_says_ok(tmp_path: Path):
    edit = _project(tmp_path, low={})
    _chat(edit).reply("في الجملة 4 غير service إلى سيرفس")
    with pytest.raises(rs.UnresolvedReviewItems):
        rs.approve(edit / "review")
    _chat(edit).reply("4 صح")
    assert rs.approve(edit / "review").ready_for_final_render is True


def test_open_review_is_idempotent_and_skips_decided_segments(tmp_path: Path):
    edit = _project(tmp_path)
    transcript = load_transcript(edit / "transcript_unified.json")
    assert open_transcript_review(edit / "review", transcript) == [3]
    _chat(edit).reply("3 صح")
    assert open_transcript_review(edit / "review", transcript) == []


def test_approve_the_rest_lists_exactly_what_it_approved(tmp_path: Path):
    edit = _project(tmp_path, low={2: [2], 3: [1]})
    open_transcript_review(edit / "review", load_transcript(edit / "transcript_unified.json"))
    reply = _chat(edit).reply("اعتمد الباقي")
    assert "3, 4" in reply.text
    assert tc.remaining_numbers(_chat(edit)._views()) == []


# 10. audio is optional and on demand ---------------------------------------------------------------------------------


def test_no_audio_is_made_during_a_normal_review(tmp_path: Path, monkeypatch):
    edit = _project(tmp_path)

    def boom(*a, **k):  # pragma: no cover - would fail the test
        raise AssertionError("audio must not be cut during a normal review")

    monkeypatch.setattr("video_edit_agent.review.chat_session.ensure_clip", boom)
    chat = _chat(edit)
    for message in (None, "الكل", "3 صح", "3: جملة"):
        chat.reply(message)
    assert not (edit / "review" / "audio").exists()


def test_audio_is_cut_only_when_asked_and_returned_in_the_reply(tmp_path: Path, monkeypatch):
    edit = _project(tmp_path)
    calls = []

    def fake(review_dir, number, start, end, source, **kw):
        calls.append((number, start, end))
        clip = review_dir / "audio" / audio_clip.clip_name(number, start, end)
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"RIFF")
        return clip

    monkeypatch.setattr("video_edit_agent.review.chat_session.ensure_clip", fake)
    reply = _chat(edit).reply("اسمعني الجملة 3")
    assert calls == [(3, 10.4, 12.3)]
    assert [p.name for p in reply.audio_paths] == ["seg_03_10.40-12.30.wav"]
    assert "[03]" in reply.text and LINES[2][0] in reply.text
    # "the last segment I looked at" is remembered for a bare request
    assert _chat(edit).reply("مش فاكر قلت ايه هنا").audio_paths


def test_missing_source_audio_is_reported_not_raised(tmp_path: Path):
    edit = _project(tmp_path)
    reply = _chat(edit).reply("play segment 3", )
    assert reply.audio_paths == [] and "no source audio" in reply.text.lower() or "ما قدرتش" in reply.text


def test_an_existing_clip_is_reused(tmp_path: Path):
    review = tmp_path / "review"
    clip = review / "audio" / audio_clip.clip_name(2, 1.0, 2.0)
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"RIFF")
    assert audio_clip.ensure_clip(review, 2, 1.0, 2.0, None) == clip


# 11. project data lives in state, never in generic modules ---------------------------------------------------------


GENERIC = ["transcript_chat", "chat_commands", "chat_session", "audio_clip", "corrections", "text_copy"]


def test_generic_modules_hold_no_project_specific_values():
    root = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent" / "review"
    forbidden = [
        "#0a3d62", "#8e44ad", "#f1c40f", r"\bd:[\\/]", r"\bs0\b", r"\bs4\b",
        "الـvalues اللي بتتقال", "بس عشان تبدأ تعمل كده", "السيرفس هي قيمة",
    ]
    for name in GENERIC:
        source = (root / f"{name}.py").read_text(encoding="utf-8").lower()
        for token in forbidden:
            pattern = token if token.startswith("\\b") else re.escape(token)
            assert not re.search(pattern, source), f"{name}.py mentions {token!r}"


# 12. long reviews can be paged and focused ---------------------------------------------------------------------------


def test_long_review_is_paged_focused_and_navigable(tmp_path: Path):
    lines = [(f"جملة رقم {i + 1}", float(i), i + 1.0) for i in range(40)]
    edit = _project(tmp_path, lines=lines, low={12: [0], 30: [1]})
    chat = _chat(edit, page_size=10)
    everything = chat.reply("show all").text
    assert "[10]" in everything and "[11]" not in everything and "صفحة 1 من 4" in everything
    assert "[11]" in chat.reply("next").text
    assert "[01]" in chat.reply("previous").text
    suspicious = chat.reply("وريني بس الجمل اللي فيها شك").text
    assert "[13]" in suspicious and "[31]" in suspicious and "[01]" not in suspicious
    ranged = chat.reply("من 20 إلى 22").text  # a bare range also focuses
    assert "[20]" in ranged or "[20]" in chat.reply("show 20 to 22").text
    shown = chat.reply("show 20 to 22").text
    assert all(f"[{n}]" in shown for n in (20, 21, 22)) and "[19]" not in shown and "[23]" not in shown


def test_out_of_range_numbers_are_reported(tmp_path: Path):
    edit = _project(tmp_path)
    reply = _chat(edit).reply("99 صح")
    assert not reply.changed and "99" in reply.text and corr.load_corrections(edit / "review") == []


# 13. transcript approval does not approve hook copy ---------------------------------------------------------------


def _hook(state: ReviewApprovalState, transcript):
    edl = EDL(width=720, height=1280, clips=[
        EDLClip(source_file="a.mp4", source_in=0.0, source_out=8.0, timeline_in=0.0, timeline_out=8.0),
    ])
    brand = Brand.model_validate({"name": "acme", "colors": {"primary": "#123456", "secondary": "#FEDCBA", "accent": "#22AA66"}})
    return plan_hook(edl, transcript, brand, None, state)


def test_hook_copy_stays_pending_after_its_source_segment_is_approved(tmp_path: Path):
    edit = _project(tmp_path, low={0: [1]})
    review = edit / "review"
    open_transcript_review(review, load_transcript(edit / "transcript_unified.json"))
    plan = _hook(rs.load_review_state(review), load_transcript(edit / "transcript_unified.json"))
    rs.record_text_treatment(review, plan.review)
    rs.approve_visual(review, "hook_title", None)

    _chat(edit).reply("1: إزاي بقى إنت كأجينسي تبقى فاهم إن الـvalues دي حقيقية")  # transcript approved
    state = rs.load_review_state(review)
    plan = _hook(state, corr.apply_corrections(load_transcript(edit / "transcript_unified.json"), corr.load_corrections(review)))
    hook = plan.review
    assert hook.copy_status is ApprovalStatus.PENDING_REVIEW
    assert hook.approved_copy is None and hook.placeholder_used_in_preview is True
    assert state.treatment("hook_title").visual_status is ApprovalStatus.APPROVED
    assert "الـvalues" not in (plan.spec.text if plan.spec else "")  # the sentence is never auto-promoted


def test_a_rewrite_is_only_proposed_until_the_user_approves_it(tmp_path: Path):
    edit = _project(tmp_path, low={})
    review = edit / "review"
    rs.record_text_treatment(review, _hook(ReviewApprovalState(), load_transcript(edit / "transcript_unified.json")).review)
    state = rs.propose_copy(review, "hook_title", "عنوان مقترح")
    hook = state.treatment("hook_title")
    assert hook.proposed_copy == "عنوان مقترح"
    assert hook.approved_copy is None or hook.approved_copy != "عنوان مقترح"
    assert state.hook_copy_status != "approved" or hook.approved_copy != "عنوان مقترح"
    with pytest.raises(ValueError):
        rs.propose_copy(review, "hook_title", "  ")


def test_correcting_a_segment_withdraws_transcript_derived_hook_copy(tmp_path: Path):
    edit = _project(tmp_path, low={})
    review = edit / "review"
    transcript = load_transcript(edit / "transcript_unified.json")
    rs.record_text_treatment(review, _hook(ReviewApprovalState(), transcript).review)
    assert rs.load_review_state(review).hook_copy_status == "approved"
    _chat(edit).reply("في الجملة 1 غير الـvalues إلى القيم")
    assert rs.load_review_state(review).hook_copy_status == "pending_review"


# CLI -----------------------------------------------------------------------------------------------------------


def test_review_chat_cli_round_trip(tmp_path: Path):
    edit = _project(tmp_path)
    runner = CliRunner()
    shown = runner.invoke(app, ["review-chat", str(edit), "--open", "--lang", "en"])
    assert shown.exit_code == 0, shown.output
    assert "Transcript Review" in shown.output and "[03]" in shown.output and "UNRESOLVED" in shown.output
    fixed = runner.invoke(app, ["review-chat", str(edit), "3: تمام كده", "--lang", "en"])
    assert fixed.exit_code == 0 and "saved and approved" in fixed.output
    assert json.loads((edit / "review" / "transcript_corrections.json").read_text(encoding="utf-8"))["corrections"][0]["corrected_text"] == "تمام كده"
    assert runner.invoke(app, ["review-approve", str(edit)]).exit_code == 0
