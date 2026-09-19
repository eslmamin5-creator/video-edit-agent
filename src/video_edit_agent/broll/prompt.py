"""Default generated-B-roll prompt rules (Baseline Recovery Milestone item 2).

The acceptance run's generated B-roll looked like cheap/plastic AI stock
imagery and produced malformed Arabic/English text baked into the pixels.
This module centralizes the default style + negative-prompt rules so both
`providers/gemini_image.py` and `providers/veo.py` build the same
photorealism-first, no-readable-text prompt instead of each hand-rolling its
own `f"{concept}. Cinematic, {aspect_ratio} aspect ratio."` template (which
had neither realism guidance nor a text prohibition).

These rules are appended unconditionally -- even when a Brand Profile
supplies its own `broll_aesthetic` -- because "no readable text in generated
imagery" is a correctness requirement (spec item 2), not a stylistic choice
a brand should be able to opt out of. Intentional on-screen text belongs in
the motion/typography layer, never inside the generated image/video itself.
"""
from __future__ import annotations

from video_edit_agent.core.schemas import BrollPlanItem

REALISM_RULES = (
    "Photorealistic documentary/commercial footage look, natural real-world "
    "lighting, realistic skin and material texture, believable camera optics "
    "(natural depth of field, no artificial lens distortion), subtle "
    "realistic color grading, plausible real-world imperfections (grain, "
    "minor asymmetry, natural framing). Avoid a glossy/plastic AI-generated "
    "look, avoid excessive HDR/bloom/glow, avoid an artificial stock-photo "
    "sheen, avoid cheap stock-AI composition. No brand logos unless explicitly requested."
)

NO_TEXT_RULES = (
    "No readable text of any kind: no Arabic text, no English text, no "
    "letters, no numbers, no signage, no labels, no logos, no captions, no "
    "fake UI, no document text. If a screen, sign, or document appears in "
    "frame, it must show only abstract, non-readable visual information "
    "(blurred, out of focus, or generic patterns) -- never legible glyphs."
)


def build_broll_prompt(item: BrollPlanItem) -> str:
    """Builds the final image/video-gen prompt for one resolved B-roll slot:
    the item's own concept/brand-styled prompt (or a sane default) plus the
    mandatory realism + no-text rules, always appended last so they cannot be
    silently dropped by an upstream override."""
    base = item.prompt or f"{item.recommended_visual}. Cinematic, {item.aspect_ratio} aspect ratio."
    return f"{base} {REALISM_RULES} {NO_TEXT_RULES}"


def looks_like_usable_asset(path_size_bytes: int, *, min_bytes: int = 2048) -> bool:
    """Cheapest possible quality gate (spec item 2: "safest minimal quality
    rejection", explicitly not a CV/OCR subsystem): rejects obviously broken
    generation output (zero-byte or truncated files a failed/filtered
    generation call can produce) before it is allowed to become an
    `Overlay`. This does not detect malformed on-screen text -- that would
    need real OCR/vision, which is out of scope for this milestone; the
    actual text-safety guarantee here is the prompt-level `NO_TEXT_RULES`
    above, applied before generation."""
    return path_size_bytes >= min_bytes
