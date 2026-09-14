"""Deterministic, offline script analysis (Creator spec section 4).

No LLM/cloud call is required or used here: language is detected via the
Arabic Unicode block ratio, hook/CTA/emotional-beat detection uses simple
keyword lists (Arabic + English), statistics are extracted with a regex
applied to the actual script text only (never invented), and duration is
estimated from a word-count/reading-rate heuristic.
"""
from __future__ import annotations

import re

from video_edit_agent.agents.creator.schemas import ScriptAnalysis

_ARABIC_RANGE = re.compile(r"[؀-ۿ]")
_STAT_RE = re.compile(r"\b\d+([.,]\d+)?\s*%?\b")

_CTA_KEYWORDS_EN = ["subscribe", "follow", "click", "sign up", "buy now", "learn more", "visit", "download", "try"]
_CTA_KEYWORDS_AR = ["اشترك", "تابعنا", "اضغط", "سجل", "اشتري", "تعرف على المزيد", "زور", "حمل", "جرب"]

_EMOTION_KEYWORDS_EN = {
    "excitement": ["amazing", "incredible", "exciting", "wow"],
    "urgency": ["now", "today", "limited", "hurry", "don't miss"],
    "concern": ["problem", "risk", "danger", "warning", "struggle"],
    "reassurance": ["easy", "simple", "safe", "guaranteed", "trusted"],
}
_EMOTION_KEYWORDS_AR = {
    "excitement": ["مذهل", "رائع", "مثير"],
    "urgency": ["الآن", "اليوم", "محدود", "بسرعة"],
    "concern": ["مشكلة", "خطر", "تحذير"],
    "reassurance": ["سهل", "بسيط", "آمن", "موثوق"],
}

_WORDS_PER_SECOND = 2.3  # average speaking rate heuristic


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?؟،\n])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    arabic = sum(1 for c in letters if _ARABIC_RANGE.match(c))
    return "ar" if arabic / len(letters) > 0.3 else "en"


def _find_cta(sentences: list[str], keywords: list[str]) -> str:
    for sentence in reversed(sentences):
        lowered = sentence.lower()
        if any(k in lowered for k in keywords):
            return sentence
    return sentences[-1] if sentences else ""


def _extract_statistics(text: str) -> list[str]:
    stats = []
    for sentence in _split_sentences(text):
        if _STAT_RE.search(sentence):
            stats.append(sentence)
    return stats


def analyze_script(text: str, *, title: str = "") -> ScriptAnalysis:
    sentences = _split_sentences(text)
    language = _detect_language(text)
    cta_keywords = _CTA_KEYWORDS_AR if language == "ar" else _CTA_KEYWORDS_EN
    emotion_keywords = _EMOTION_KEYWORDS_AR if language == "ar" else _EMOTION_KEYWORDS_EN

    hook = sentences[0] if sentences else ""
    cta = _find_cta(sentences, cta_keywords) if sentences else ""
    key_message = sentences[len(sentences) // 2] if sentences else ""
    supporting_points = [s for s in sentences if s not in (hook, cta, key_message)]
    statistics = _extract_statistics(text)

    lowered = text.lower()
    emotional_beats = [emotion for emotion, kws in emotion_keywords.items() if any(k in lowered for k in kws)]

    word_count = len(text.split())
    estimated_duration = round(word_count / _WORDS_PER_SECOND, 1) if word_count else 0.0

    visual_opportunities = []
    if statistics:
        visual_opportunities.append("data_viz")
    if len(sentences) >= 4:
        visual_opportunities.append("scene_breakdown")
    if cta:
        visual_opportunities.append("cta_card")

    return ScriptAnalysis(
        title=title,
        topic=hook,
        language=language,
        target_tone="informative",
        hook=hook,
        key_message=key_message,
        supporting_points=supporting_points,
        examples=[],
        statistics=statistics,
        cta=cta,
        emotional_beats=emotional_beats,
        estimated_target_duration=estimated_duration,
        visual_opportunities=visual_opportunities,
    )
