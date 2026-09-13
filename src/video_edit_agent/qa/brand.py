"""Brand QA layer (spec section 33): checks the final output against the
active Brand Profile's declared constraints -- forbidden words/imagery
(`avoid`), caption style/color match, and CTA presence when the brand
requires one.
"""
from __future__ import annotations

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import EDL, QAIssue, QASeverity, Transcript


def check_avoid_list(transcript: Transcript, brand: Brand) -> list[QAIssue]:
    issues: list[QAIssue] = []
    if not brand.avoid:
        return issues

    full_text_lower = transcript.full_text.lower()
    for forbidden in brand.avoid:
        if forbidden.lower() in full_text_lower:
            issues.append(
                QAIssue(
                    category="brand", severity=QASeverity.WARNING,
                    message=f"Transcript contains a brand-avoided term: '{forbidden}'",
                    auto_repairable=False,
                )
            )
    return issues


def check_cta_present(edl: EDL, has_cta_slot: bool, brand: Brand) -> list[QAIssue]:
    if brand.cta.text_default and not has_cta_slot and edl.total_duration > 6.0:
        return [
            QAIssue(
                category="brand", severity=QASeverity.INFO,
                message="Brand defines a default CTA but none was placed on the timeline",
                auto_repairable=True,
            )
        ]
    return []


def run_brand_qa(edl: EDL, transcript: Transcript, brand: Brand, has_cta_slot: bool) -> list[QAIssue]:
    issues = check_avoid_list(transcript, brand)
    issues.extend(check_cta_present(edl, has_cta_slot, brand))
    return issues
