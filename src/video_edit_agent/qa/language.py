"""Language QA layer (spec sections 33, 37, 38): the NON-NEGOTIABLE dialect
preservation check plus caption/transcript text sanity checks. Runs fully
offline -- no API key required.
"""
from __future__ import annotations

from video_edit_agent.core.schemas import QAIssue, QASeverity, Transcript
from video_edit_agent.language.dialect_guard import check_no_dialect_substitution


def check_dialect_preserved(source_transcript: Transcript, final_transcript: Transcript) -> list[QAIssue]:
    """Runs the dialect guard segment-by-segment. A violation here is always
    ERROR severity and never auto-repairable by rewriting text -- the only
    correct repair is to restore the source segment verbatim (spec section 8)."""
    issues: list[QAIssue] = []
    final_by_id = {seg.id: seg for seg in final_transcript.segments}

    for source_seg in source_transcript.segments:
        final_seg = final_by_id.get(source_seg.id)
        if final_seg is None:
            continue
        result = check_no_dialect_substitution(source_seg.text, final_seg.text)
        if not result.ok:
            issues.append(
                QAIssue(
                    category="language",
                    severity=QASeverity.ERROR,
                    message=result.reason,
                    timeline_at=source_seg.start,
                    auto_repairable=True,  # repairable by restoring source_seg.text verbatim
                )
            )
    return issues


def check_no_empty_segments(transcript: Transcript) -> list[QAIssue]:
    issues: list[QAIssue] = []
    for seg in transcript.segments:
        if not seg.text.strip():
            issues.append(
                QAIssue(
                    category="language", severity=QASeverity.WARNING,
                    message=f"Empty transcript segment at {seg.start:.2f}s", timeline_at=seg.start,
                    auto_repairable=False,
                )
            )
    return issues


def run_language_qa(source_transcript: Transcript, final_transcript: Transcript) -> list[QAIssue]:
    issues = check_dialect_preserved(source_transcript, final_transcript)
    issues.extend(check_no_empty_segments(final_transcript))
    return issues
