"""Language / dialect detection (signal only — never used to rewrite text)."""
from __future__ import annotations

from video_edit_agent.core.schemas import Transcript
from video_edit_agent.language.arabic import dialect_marker_hits, is_arabic_text


def detect_interface_language(user_text: str) -> str:
    """Used to decide which language the AGENT responds in (spec section 9),
    never to alter the video transcript."""
    return "ar" if is_arabic_text(user_text) else "en"


def detect_dominant_dialect(transcript: Transcript) -> str:
    """Best-effort label for the transcript's dominant Arabic dialect, purely
    informational (e.g. for choosing a caption font or interface tone hint).
    Never used to alter `transcript.segments[*].text`."""
    if not is_arabic_text(transcript.full_text):
        return "en" if transcript.full_text else "unknown"

    hits = {"egyptian": 0, "gulf_saudi": 0, "msa": 0}
    for seg in transcript.segments:
        for k, v in dialect_marker_hits(seg.text).items():
            hits[k] += v

    if all(v == 0 for v in hits.values()):
        return "ar"  # generic/unmarked Arabic
    return max(hits, key=hits.get)
