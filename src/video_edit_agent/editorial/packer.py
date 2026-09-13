"""Transcript packing (spec sections 11, 14): a compact, human/LLM-reviewable
`takes_packed.md` — avoids dumping raw JSON or thousands of frames."""
from __future__ import annotations

from video_edit_agent.core.schemas import Transcript
from video_edit_agent.editorial.takes import TakeVerdict


def pack_takes(transcript: Transcript, verdicts: list[TakeVerdict]) -> str:
    by_id = {v.segment_id: v for v in verdicts}
    lines = ["# Takes\n", f"_provider: {transcript.provider} · language: {transcript.language}_\n"]
    for seg in transcript.segments:
        v = by_id.get(seg.id)
        mark = "KEEP" if (v is None or v.keep) else f"CUT ({v.reason})"
        lines.append(f"- `[{seg.start:6.2f}-{seg.end:6.2f}]` **{mark}** — {seg.text}")
    return "\n".join(lines) + "\n"


def write_takes_packed(transcript: Transcript, verdicts: list[TakeVerdict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pack_takes(transcript, verdicts), encoding="utf-8")
