"""Semantic beat detector: planning metadata derived from the APPROVED transcript.

    Approved Transcript -> Semantic Beats -> (director) -> reviewable plan

A semantic beat is a stretch of the approved transcript that does one job in the
argument: it opens a topic, states a key claim, asks a question, sets up a
contrast, walks through steps, pays something off, or simply supports what came
before. The detector only ever READS the transcript:

* it never rewrites, normalises or translates text (spoken dialect is kept
  verbatim; the comparison-only folding below is never stored),
* every beat boundary is an existing segment boundary or a gap between two
  existing words, so beat timing always stays inside the approved timing,
* it is deterministic: identical transcript -> identical beats, no randomness,
  no clock, no model call.

Confidence is explicit. Each kind is scored from generic discourse cues
(sentence-final marks, interrogatives, contrast/sequence/conclusion connectives,
emphasis words, a pause before the unit). The winner's confidence is its score
minus half of the runner-up's. Below `LOW_CONFIDENCE` the beat is reported as
plain SUPPORT (its weaker guess kept as `candidate`) and `to_director_beats`
gives it NO special treatment: a guess never invents a visual by itself.

The cue lexicons are generic discourse markers (Arabic dialect/MSA and English),
not the vocabulary of any project, brand or video. They are signals for scoring,
never a rewriting or translation table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import Transcript, Word
from video_edit_agent.direction.director import HIGH_CONFIDENCE, LOW_CONFIDENCE, Beat, BeatKind
from video_edit_agent.direction.transitions import TransitionReason, TransitionStyle
from video_edit_agent.language.arabic import normalize_for_matching

MIN_BEAT_S = 2.0  # a shorter sentence is merged into its neighbour
MIN_BEAT_WORDS = 3
CUT_TOLERANCE_S = 0.3  # how far a forced boundary may sit from a sentence boundary
PAUSE_BEFORE_S = 0.8  # a gap this long before a unit is a (weak) topic-shift cue

_SENTENCE_END = re.compile(r"[.!?؟…]+[\"'”’)\]]*$")
_QUESTION_END = re.compile(r"[?؟]+[\"'”’)\]]*$")


class SemanticKind(str, Enum):
    TOPIC_SHIFT = "topic_shift"
    KEY_CLAIM = "key_claim"
    QUESTION = "question"
    CONTRAST = "contrast"
    PROCESS_LIST = "process_list"
    PAYOFF = "payoff"
    SUPPORT = "support"  # supporting explanation (also what a low-confidence guess falls back to)


# ties are broken in this fixed order, so equal scores stay deterministic
_PRIORITY = (
    SemanticKind.QUESTION, SemanticKind.CONTRAST, SemanticKind.PAYOFF,
    SemanticKind.KEY_CLAIM, SemanticKind.PROCESS_LIST, SemanticKind.TOPIC_SHIFT,
)

# --- generic discourse cues. Matched on a comparison-only fold of the text (alef/ya/ta-marbuta
# variants, tashkeel and tatweel are ignored for matching; the stored text is never folded). ---
_INTERROGATIVES: dict[str, float] = {  # leading interrogatives
    "هل": 0.5, "ليه": 0.45, "ليش": 0.45, "لماذا": 0.5, "كيف": 0.45, "ازاي": 0.45, "امتي": 0.4, "فين": 0.35,
    "why": 0.5, "how": 0.4, "what": 0.35, "which": 0.35, "when": 0.35, "where": 0.35, "is": 0.2, "do": 0.2, "does": 0.2, "can": 0.2,
}
_PHRASE_CUES: dict[SemanticKind, dict[str, float]] = {
    SemanticKind.CONTRAST: {
        "لكن": 0.6, "ولكن": 0.6, "لاكن": 0.6, "بل": 0.5, "بالعكس": 0.6, "علي العكس": 0.6, "في المقابل": 0.55, "بدل ما": 0.5,
        "بدلا من": 0.5, "عكس": 0.4, "بس": 0.3, "مع ذلك": 0.45, "غير كده": 0.3,
        "but": 0.5, "however": 0.6, "instead": 0.5, "rather than": 0.55, "on the other hand": 0.6, "whereas": 0.5,
    },
    SemanticKind.PAYOFF: {
        "في النهايه": 0.65, "الخلاصه": 0.65, "خلاصه": 0.55, "عشان كده": 0.6, "علشان كده": 0.6, "لذلك": 0.55, "ولذلك": 0.55,
        "بالتالي": 0.55, "النتيجه": 0.55, "وده معناه": 0.55, "يبقي": 0.3, "اخيرا": 0.5,
        "therefore": 0.55, "that's why": 0.6, "thats why": 0.6, "in the end": 0.65, "in conclusion": 0.7, "finally": 0.5, "so that": 0.3,
    },
    SemanticKind.KEY_CLAIM: {
        "اهم": 0.6, "الاهم": 0.65, "اهميه": 0.6, "مهم": 0.5, "لازم": 0.35, "اساس": 0.45, "السر": 0.55, "الحقيقه": 0.5,
        "مستحيل": 0.5, "ابدا": 0.4, "دايما": 0.3, "مفتاح": 0.5,
        "important": 0.55, "the key": 0.55, "must": 0.35, "never": 0.4, "always": 0.3, "the truth": 0.5, "the secret": 0.55,
    },
    SemanticKind.PROCESS_LIST: {
        "اولا": 0.55, "ثانيا": 0.55, "ثالثا": 0.55, "الخطوه": 0.55, "خطوه": 0.45, "بعدين": 0.4, "بعد كده": 0.4, "وبعدها": 0.4,
        "بعدها": 0.35, "ثم": 0.35, "المرحله": 0.45, "اول حاجه": 0.5, "تاني حاجه": 0.5,
        "first": 0.4, "second": 0.5, "third": 0.5, "then": 0.3, "next": 0.35, "step": 0.45,
    },
    SemanticKind.TOPIC_SHIFT: {
        "طيب": 0.4, "تعالي نشوف": 0.6, "خلينا": 0.4, "نيجي": 0.4, "ننتقل": 0.6, "الموضوع التاني": 0.6, "نقطه تانيه": 0.55,
        "اما": 0.3, "تعالوا": 0.4, "خلونا": 0.4,
        "let's move": 0.6, "moving on": 0.6, "next topic": 0.6, "now let's": 0.55, "so what about": 0.5, "now": 0.25,
    },
}
_STEM_KINDS = {SemanticKind.KEY_CLAIM}  # importance stems also appear with affixes (الأهمية, وأهمها ...)
_LIST_COMMAS = 3  # this many commas in one unit read as an enumeration
_ENUMERATION_CUE = 0.35
_EMBEDDED_QUESTION = 0.5
_LEADING_NUMBER = re.compile(r"^\s*[\d٠-٩]+\s*[).\-:،]")


def _fold(text: str) -> str:
    """Comparison-only folding. Never stored, never shown."""
    t = normalize_for_matching(text).lower()
    for src, dst in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه"), ("’", "'"), ("‘", "'")):
        t = t.replace(src, dst)
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class SemanticBeat(BaseModel):
    start: float
    end: float
    first_segment: int  # 0-based index into `transcript.segments`
    last_segment: int
    kind: SemanticKind
    confidence: float
    candidate: SemanticKind | None = None  # the weaker guess, when `kind` fell back to SUPPORT
    cues: list[str] = Field(default_factory=list)
    text: str = ""  # the approved words, verbatim
    pause_before_s: float = 0.0

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def confidence_label(self) -> str:
        return confidence_label(self.confidence)

    @property
    def low_confidence(self) -> bool:
        return self.candidate is not None or self.confidence < LOW_CONFIDENCE

    @property
    def segment_numbers(self) -> list[int]:
        """1-based segment numbers, the way the review plan refers to them."""
        return list(range(self.first_segment + 1, self.last_segment + 2))


def confidence_label(value: float) -> str:
    return "high" if value >= HIGH_CONFIDENCE else "medium" if value >= LOW_CONFIDENCE else "low"


@dataclass
class _Unit:
    words: list[Word]
    text: str
    start: float
    end: float
    segment: int
    ends_question: bool


def _units(transcript: Transcript) -> list[_Unit]:
    """Sentence-sized units from the words' own punctuation and timing (whole segments where a
    segment carries no word timing)."""
    units: list[_Unit] = []
    for si, seg in enumerate(transcript.segments):
        words = [w for w in seg.words if w.word.strip()]
        if not words:
            units.append(_Unit([], seg.text.strip(), seg.start, seg.end, si, bool(_QUESTION_END.search(seg.text.strip()))))
            continue
        cut = [i for i, w in enumerate(words) if _SENTENCE_END.search(w.word.strip())]
        if not cut or cut[-1] != len(words) - 1:
            cut.append(len(words) - 1)
        lo = 0
        for c in cut:
            chunk = words[lo:c + 1]
            lo = c + 1
            if not chunk:
                continue
            text = seg.text.strip() if len(chunk) == len(words) else " ".join(w.word.strip() for w in chunk)
            units.append(_Unit(chunk, text, chunk[0].start, chunk[-1].end, si, bool(_QUESTION_END.search(chunk[-1].word.strip()))))
    return units


def _small(group: list[_Unit]) -> bool:
    words = sum(len(u.words) or len(u.text.split()) for u in group)
    return (group[-1].end - group[0].start) < MIN_BEAT_S or words < MIN_BEAT_WORDS


def _cut_between(prev: _Unit, nxt: _Unit, boundaries: tuple[float, ...]) -> bool:
    """True when a forced boundary (an already-decided window edge) falls between two units."""
    return any(prev.end - CUT_TOLERANCE_S <= t <= nxt.start + CUT_TOLERANCE_S for t in boundaries)


def _merge_small(units: list[_Unit], boundaries: tuple[float, ...] = ()) -> list[list[_Unit]]:
    """Groups of consecutive units; a unit too short to carry a treatment joins a neighbour,
    but never across a forced boundary."""
    groups: list[list[_Unit]] = []
    for u in units:
        if groups and _small(groups[-1]) and not groups[-1][-1].ends_question and not _cut_between(groups[-1][-1], u, boundaries):
            groups[-1].append(u)
        else:
            groups.append([u])
    if len(groups) > 1 and _small(groups[-1]) and not _cut_between(groups[-2][-1], groups[-1][0], boundaries):
        tail = groups.pop()
        groups[-1].extend(tail)
    return groups


def _score(text: str, *, ends_question: bool, pause_before: float, first: bool, last: bool) -> tuple[dict[SemanticKind, float], dict[SemanticKind, list[str]]]:
    folded = _fold(text)
    tokens = folded.split()
    padded = f" {folded} "
    scores: dict[SemanticKind, float] = {k: 0.0 for k in _PRIORITY}
    why: dict[SemanticKind, list[str]] = {k: [] for k in _PRIORITY}

    def add(kind: SemanticKind, weight: float, reason: str) -> None:
        scores[kind] = min(1.0, scores[kind] + weight)
        why[kind].append(reason)

    if ends_question:
        add(SemanticKind.QUESTION, 0.9, "ends with a question mark")
    for lead in tokens[:2]:
        weight = _INTERROGATIVES.get(lead)
        if weight:
            add(SemanticKind.QUESTION, weight, f"opens with the interrogative «{lead}»")
            break
    else:
        if "هل" in tokens[2:] and not ends_question:  # a yes/no question posed mid-sentence
            add(SemanticKind.QUESTION, _EMBEDDED_QUESTION, "poses a yes/no question («هل») inside the sentence")
    for kind, table in _PHRASE_CUES.items():
        for phrase, weight in table.items():
            hit = f" {phrase} " in padded
            if not hit and kind in _STEM_KINDS and " " not in phrase:
                hit = any(t.startswith((phrase, "ال" + phrase, "و" + phrase)) for t in tokens)
            if hit:
                add(kind, weight, f"cue «{phrase}»")
    if len(re.findall(r"[,،]", text)) >= _LIST_COMMAS:
        add(SemanticKind.PROCESS_LIST, _ENUMERATION_CUE, "an enumeration (several commas)")
    if _LEADING_NUMBER.search(text):
        add(SemanticKind.PROCESS_LIST, 0.45, "opens with a number marker")
    if pause_before >= PAUSE_BEFORE_S and not first:
        add(SemanticKind.TOPIC_SHIFT, 0.3, f"a {pause_before:.1f}s pause before it")
    if last:
        add(SemanticKind.PAYOFF, 0.25, "closes the video")
    return scores, why


def _classify(scores: dict[SemanticKind, float], why: dict[SemanticKind, list[str]]) -> tuple[SemanticKind, float, SemanticKind | None, list[str]]:
    ranked = sorted(_PRIORITY, key=lambda k: (-scores[k], _PRIORITY.index(k)))
    best = ranked[0]
    top = scores[best]
    if top <= 0:
        return SemanticKind.SUPPORT, 0.5, None, ["no structural cue: it supports the surrounding idea"]
    confidence = round(max(0.0, top - 0.5 * scores[ranked[1]]), 3)
    if confidence < LOW_CONFIDENCE:
        return SemanticKind.SUPPORT, confidence, best, [*why[best], f"low confidence ({confidence:.2f}): treated as supporting explanation"]
    return best, confidence, None, why[best]


def detect_semantic_beats(transcript: Transcript, *, boundaries: tuple[float, ...] = ()) -> list[SemanticBeat]:
    """The semantic beats of `transcript` (time-ordered, contiguous, inside the transcript's own
    timing). The transcript is not modified. `boundaries` are times a beat must not straddle (the
    edges of windows that were already decided): a cut is honoured where it meets a sentence
    boundary, so an approved window always lines up with the beats around it."""
    units = _units(transcript)
    if not units:
        return []
    groups = _merge_small(units, tuple(boundaries))
    beats: list[SemanticBeat] = []
    previous_end: float | None = None
    for gi, group in enumerate(groups):
        first, last = group[0], group[-1]
        text = " ".join(u.text for u in group).strip()
        pause = round(max(0.0, first.start - previous_end), 3) if previous_end is not None else 0.0
        scores, why = _score(text, ends_question=last.ends_question, pause_before=pause, first=gi == 0, last=gi == len(groups) - 1)
        kind, confidence, candidate, cues = _classify(scores, why)
        beats.append(SemanticBeat(
            start=first.start, end=last.end, first_segment=first.segment, last_segment=last.segment,
            kind=kind, confidence=confidence, candidate=candidate, cues=cues, text=text, pause_before_s=pause,
        ))
        previous_end = last.end
    _tile(beats, transcript)
    return beats


def _tile(beats: list[SemanticBeat], transcript: Transcript) -> None:
    """Closes the gaps between beats so they tile the transcript: a boundary between two segments is
    the existing segment boundary; a boundary inside a segment sits mid-way through the silence
    between the two words. The outer edges are the transcript's own first/last segment times."""
    for a, b in pairwise(beats):
        if a.last_segment != b.first_segment:
            edge = max(a.end, min(transcript.segments[b.first_segment].start, b.start))
        else:
            edge = (a.end + b.start) / 2
        a.end = b.start = round(edge, 3)
    beats[0].start = round(min(beats[0].start, transcript.segments[beats[0].first_segment].start), 3)
    beats[-1].end = round(max(beats[-1].end, transcript.segments[beats[-1].last_segment].end), 3)


# --------------------------------------------------------------------------
# semantic beats -> director beats
# --------------------------------------------------------------------------

_TREATED = {  # semantic kind -> (director kind, base importance, confidence gain)
    SemanticKind.KEY_CLAIM: (BeatKind.EMPHASIS, 0.5, 0.4),
    SemanticKind.PAYOFF: (BeatKind.EMPHASIS, 0.5, 0.35),
    SemanticKind.CONTRAST: (BeatKind.EMPHASIS, 0.45, 0.3),
    SemanticKind.PROCESS_LIST: (BeatKind.CONCEPT, 0.45, 0.3),
}


def to_director_beats(beats: list[SemanticBeat]) -> list[Beat]:
    """Director beats for `beats`. Only a confident, meaningful classification asks for anything
    beyond the speaker; QUESTION / SUPPORT / any low-confidence guess stay plain. A topic shift
    records a light-leak transition INTENT (recorded only: the renderer plays a direct cut)."""
    out: list[Beat] = []
    for i, b in enumerate(beats):
        kind, importance = BeatKind.PLAIN, 0.35
        reason = style = None
        if not b.low_confidence and b.kind in _TREATED:
            kind, base, gain = _TREATED[b.kind]
            importance = round(min(1.0, base + gain * b.confidence), 3)
        elif b.kind is SemanticKind.QUESTION and not b.low_confidence:
            importance = 0.45
        elif b.kind is SemanticKind.TOPIC_SHIFT and not b.low_confidence and i > 0:
            reason, style, importance = TransitionReason.SEMANTIC_CHANGE, TransitionStyle.LIGHT_LEAK, 0.5
        out.append(Beat(
            start=b.start, end=b.end, kind=kind, importance=importance, text=b.text,
            transition_reason=reason, transition_style=style,
            semantic_kind=b.kind.value, semantic_confidence=b.confidence, semantic_cues=list(b.cues),
        ))
    return out


__all__ = [
    "HIGH_CONFIDENCE", "LOW_CONFIDENCE", "SemanticBeat", "SemanticKind", "confidence_label",
    "detect_semantic_beats", "to_director_beats",
]
