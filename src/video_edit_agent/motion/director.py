"""Motion director (spec section 17): turns editorial/transcript signals into
a list of `AnimationSpec` slots on the post-cut timeline. This is
intentionally conservative -- it only proposes graphics where there's a clear
signal (a new speaker, a spoken number, an opening/closing beat) -- and it
never blocks the pipeline: any exception here should be treated by the caller
as "no motion graphics for this project", not a hard failure.
"""
from __future__ import annotations

import re

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import EDL, AnimationKind, AnimationSpec, Transcript
from video_edit_agent.motion.legibility import plan_title_treatment
from video_edit_agent.subject.framing import FrameAnalysis

_NUMBER_RE = re.compile(r"\b(\d[\d,]*\.?\d*%?)\b")
_MIN_HOOK_GAP_S = 0.5
_MAX_STAT_SLOTS = 3


def _words_in_window(transcript: Transcript, start: float, end: float) -> list[str]:
    return [w.word for w in transcript.words if start <= w.start < end]


def _hook_extra(brand: Brand | None, text: str, edl: EDL, analysis: FrameAnalysis | None) -> dict:
    """Legibility treatment for the hook title (see `motion/legibility.py`):
    colour, backing plate and placement derived from the Brand Profile and, when
    available, the footage analysis. Without a brand the title keeps the
    renderer's default look."""
    if brand is None:
        return {}
    treatment = plan_title_treatment(brand, text, edl.width, edl.height, analysis=analysis)
    return {**treatment.as_props(), "legibility": {
        "foreground_source": treatment.foreground_source,
        "contrast": treatment.contrast,
        "reasons": treatment.reasons,
    }}


def build_motion_plan(
    edl: EDL, transcript: Transcript, brand: Brand | None = None, cta_text: str | None = None,
    analysis: FrameAnalysis | None = None,
) -> list[AnimationSpec]:
    """Propose a small, high-confidence set of motion graphic slots:

    - a `hook_title` in the opening seconds, using the first spoken clause
    - a `lower_third` whenever the active speaker changes
    - up to `_MAX_STAT_SLOTS` `stat_counter` slots where a number is spoken
    - a closing `cta` in the final seconds -- ONLY when CTA text was explicitly
      provided (`cta_text`, e.g. from an approved edit plan) or the Brand
      Profile defines one. CTA is NONE by default: no generic
      call-to-action text is ever invented.
    """
    specs: list[AnimationSpec] = []
    if not edl.clips:
        return specs

    total_duration = edl.total_duration

    first_clip = edl.clips[0]
    hook_words = _words_in_window(transcript, first_clip.source_in, first_clip.source_in + 4.0)
    hook_text = " ".join(hook_words[:8]).strip()
    if hook_text:
        specs.append(
            AnimationSpec(
                kind=AnimationKind.HOOK_TITLE,
                timeline_start=0.0,
                timeline_end=min(3.0, total_duration),
                text=hook_text,
                extra=_hook_extra(brand, hook_text, edl, analysis),
            )
        )

    last_speaker: str | None = None
    for clip in edl.clips:
        if clip.speaker and clip.speaker != last_speaker:
            specs.append(
                AnimationSpec(
                    kind=AnimationKind.LOWER_THIRD,
                    timeline_start=clip.timeline_in,
                    timeline_end=min(clip.timeline_in + 3.5, clip.timeline_out),
                    text=clip.speaker,
                )
            )
            last_speaker = clip.speaker

    stat_slots = 0
    for clip in edl.clips:
        if stat_slots >= _MAX_STAT_SLOTS:
            break
        window_words = _words_in_window(transcript, clip.source_in, clip.source_out)
        joined = " ".join(window_words)
        match = _NUMBER_RE.search(joined)
        if match:
            specs.append(
                AnimationSpec(
                    kind=AnimationKind.STAT_COUNTER,
                    timeline_start=clip.timeline_in,
                    timeline_end=min(clip.timeline_in + 3.0, clip.timeline_out),
                    value=match.group(1),
                    text=joined[:60],
                    # Baseline Recovery Milestone item 7: a stat counter reads as
                    # a big background statistic the speaker stands in front of
                    # (documentary/social-video convention), unlike LOWER_THIRD/
                    # CTA, which are clean foreground chyron/UI elements that
                    # would look broken if partially occluded by the subject.
                    behind_subject=True,
                )
            )
            stat_slots += 1

    cta_text = cta_text or (brand.cta_text if brand is not None else None)
    if cta_text and total_duration > 6.0:
        specs.append(
            AnimationSpec(
                kind=AnimationKind.CTA,
                timeline_start=max(0.0, total_duration - 3.0),
                timeline_end=total_duration,
                text=cta_text,
                # The CTA composition has no default label; the only text it
                # ever shows is the explicitly provided one.
                extra={"actionLabel": cta_text},
            )
        )

    return specs
