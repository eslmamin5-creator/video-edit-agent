"""Arabic text utilities shared by the caption engine and the dialect guard."""
from __future__ import annotations

import re
import unicodedata

ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

# Dialect marker words: small, high-precision indicator sets used for
# detection and for regression testing (spec sections 8, 37). These are
# *signals*, never a normalization/translation table — no code path may use
# this to rewrite one dialect into another.
EGYPTIAN_MARKERS = {"عايز", "عاوز", "دلوقتي", "إزيك", "ازيك", "كده", "مش", "خلاص", "قوي", "أهو"}
GULF_SAUDI_MARKERS = {"أبي", "ابي", "أبغى", "ابغى", "الحين", "وايد", "شلونك", "زين", "يبيلك", "أسوي", "اسوي"}
MSA_MARKERS = {"الآن", "أريد", "سوف", "لماذا", "كيف حالك"}


def is_arabic_text(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic_count = sum(1 for c in letters if ARABIC_RANGE.match(c))
    return arabic_count / len(letters) > 0.5


def has_code_switch(text: str) -> bool:
    return bool(ARABIC_RANGE.search(text)) and bool(LATIN_RANGE.search(text))


def normalize_for_matching(text: str) -> str:
    """NFC-normalize and strip tatweel/diacritics only for *comparison*
    purposes (dialect detection, dedup). Never used to alter stored text."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"ـ", "", text)  # tatweel
    text = re.sub(r"[ً-ٰٟ]", "", text)  # harakat
    return text


def dialect_marker_hits(text: str) -> dict[str, int]:
    norm = normalize_for_matching(text)
    tokens = set(re.findall(r"[؀-ۿ]+", norm))
    return {
        "egyptian": len(tokens & EGYPTIAN_MARKERS),
        "gulf_saudi": len(tokens & GULF_SAUDI_MARKERS),
        "msa": len(tokens & MSA_MARKERS),
    }


def shape_for_display(text: str) -> str:
    """Apply Arabic letter shaping + BiDi reordering for rendering into
    caption images/subtitles (spec section 15). Requires arabic-reshaper and
    python-bidi; degrades to plain text if unavailable rather than failing
    the whole render."""
    if not is_arabic_text(text) and not has_code_switch(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        return text
