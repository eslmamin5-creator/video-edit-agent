"""Dialect preservation guard (spec sections 8, 37) — NON-NEGOTIABLE.

TRANSCRIBE. DO NOT TRANSLATE. DO NOT REWRITE. DO NOT FORMALIZE.
DO NOT LOCALIZE INTO ANOTHER DIALECT.

This guard does not "fix" anything — it only detects and reports suspicious
dialect substitution if some transform ever produces a `final_text` that
diverges from the `source_text` the audio actually contains. The source
audio (and its verbatim transcript) always has final authority; if this
guard fires, the correct action is to keep `source_text`, never to keep
whichever text "looks more correct".
"""
from __future__ import annotations

from dataclasses import dataclass

from video_edit_agent.language.arabic import dialect_marker_hits, normalize_for_matching


class DialectGuardViolation(RuntimeError):
    pass


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""


def check_no_dialect_substitution(source_text: str, final_text: str) -> GuardResult:
    """Returns ok=False if `final_text` introduces dialect markers from a
    *different* dialect family than `source_text` contained, while removing
    the source's own markers -- a strong signal of unrequested normalization
    (e.g. Egyptian "عايز" silently rewritten to Gulf "أبي" or MSA "أريد")."""

    if normalize_for_matching(source_text) == normalize_for_matching(final_text):
        return GuardResult(ok=True)

    src_hits = dialect_marker_hits(source_text)
    dst_hits = dialect_marker_hits(final_text)

    src_dialects = {k for k, v in src_hits.items() if v > 0}
    dst_dialects = {k for k, v in dst_hits.items() if v > 0}

    if not src_dialects:
        return GuardResult(ok=True)  # nothing to preserve a claim about

    lost = src_dialects - dst_dialects
    gained = dst_dialects - src_dialects
    if lost and gained:
        return GuardResult(
            ok=False,
            reason=(
                f"Text changed from a passage with {sorted(src_dialects)} markers to one with "
                f"{sorted(dst_dialects)} markers — looks like unrequested dialect conversion, "
                f"not a verbatim transcript edit."
            ),
        )
    return GuardResult(ok=True)


def enforce(source_text: str, final_text: str) -> None:
    result = check_no_dialect_substitution(source_text, final_text)
    if not result.ok:
        raise DialectGuardViolation(result.reason)
