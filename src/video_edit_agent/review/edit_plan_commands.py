"""Understands what the user says about the edit plan, in Arabic or English.

Same idea as `chat_commands` (no rigid syntax; the agent hands the message over
and gets a `Command`), for the editorial treatments. Text the user dictates is
returned verbatim. Recognised (examples, not an exhaustive grammar):

    1 موافق                            approve slot 1's recommended treatment
    2 خليه speaker                     set_treatment (stay on speaker)
    3 بدل generated اعمله motion graphic   set_treatment (motion graphic, NOT generated)
    4 استخدم B-roll محلي               set_treatment (local B-roll)
    5 اختار الخيار B                   choose_option (keyword option B)
    5 النص: value حقيقية               set_text (verbatim)
    3 ولّد / 3 generate                approve_generation (explicit, per slot)
    2 بلاش                             reject (no treatment)
    اعتمد الباقي                        approve_rest
    وريني بس الحاجات اللي محتاجة asset   show (asset)
    وريني الـgenerated فقط              show (generated)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from video_edit_agent.review.chat_commands import (
    _APPROVE_REST,
    _APPROVE_WORDS,
    _FILLER,
    _HELP,
    _LABEL,
    _SHOW_VERB,
    _bare,
    _norm,
    _numbers,
    split_message,
)

KIND_SHOW = "show"
KIND_APPROVE = "approve"
KIND_APPROVE_REST = "approve_rest"
KIND_SET_TREATMENT = "set_treatment"
KIND_REJECT = "reject"
KIND_CHOOSE_OPTION = "choose_option"
KIND_SET_TEXT = "set_text"
KIND_APPROVE_GENERATION = "approve_generation"
KIND_SET_SOUND = "set_sound"
KIND_HELP = "help"
KIND_UNKNOWN = "unknown"

VIEW_PENDING = "pending"
VIEW_ALL = "all"
VIEW_ASSET = "asset"
VIEW_GENERATED = "generated"
VIEW_NUMBERS = "numbers"


@dataclass(frozen=True)
class Command:
    kind: str
    numbers: tuple[int, ...] = ()
    treatment: str | None = None  # set_treatment: a `Treatment` value or "no_treatment"
    sound: str | None = None  # set_sound: a sound intent (none | subtle_motion | transition | accent | impact)
    option: int | None = None  # choose_option: 0-based (A=0)
    text: str | None = None  # set_text: verbatim
    view: str | None = None


# Ordered so that a longer/more specific phrase wins where they overlap
# ("motion graphic" before "graphic"). Matched on `_bare` (hamza/diacritics folded, lower-case).
_TREATMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("behind_subject_text", (r"behind[\s-]*(?:the\s+)?(?:subject|speaker|person)(?:\s*text)?|behind[\s-]*text|"
                            r"(?:ورا|وراء|خلف)\s*(?:ال)?(?:متحدث|شخص|سبيكر|speaker|راجل|بني ادم)|نص\s*(?:ورا|خلف)(?:\s*(?:ال)?(?:متحدث|شخص|سبيكر|speaker))?")),
    ("full_screen_text_scene", r"full[\s-]*screen\s*(?:text|typography|title)|نص\s*(?:ملء|كامل)\s*(?:ال)?شاش\w*|شاشه\s*كامله\s*نص"),
    ("graphic_data_scene", r"data\s*(?:scene|graphics?|viz|visuali[sz]ation)|infographics?|انفوجراف\w*|رسم\s*بياني|chart"),
    ("illustration", r"illustrat\w*|ايلستريشن|رسمه\s*توضيحي\w*|رسم\s*توضيحي"),
    ("kinetic_typography", r"kinetic(?:\s*typography)?|typography|كينتيك|تايبوجراف\w*|نص\s*متحرك|كلام\s*متحرك"),
    ("motion_graphic", r"motion[\s-]*graphics?|موشن(?:\s*جرافيك)?|جرافيك|graphics?|animation|انيميشن|رسم\s*متحرك"),
    ("punch_in", r"punch[\s-]*in|بانش(?:\s*ان)?|زوو?م|zoom|re-?frame|ريفريم"),
    ("local_broll", (r"local(?:\s*b[\s-]*roll)?|b[\s-]*roll\s*(?:محلي|بتاعي|بتاعنا|من\s*عندي)|محلي\w*|"
                    r"(?:my|our|own|user)(?:\s*own)?\s*(?:footage|clips?|video)|(?:فيديو|footage|لقطات?)\s*(?:بتاع\w*|من\s*عندي)")),
    ("generated_broll", r"generated|ai[\s-]*(?:generated|b[\s-]*roll|video|footage)|مولد\w*|توليد\s*ai|ذكاء\s*اصطناعي"),
    ("stay_on_speaker", (r"stay\s+on\s+(?:the\s+)?speaker|speaker|سبيكر|(?:على|علي)\s*(?:ال)?متحدث|المتحدث|"
                        r"من\s*غير\s*(?:اي\s*)?(?:b[\s-]*roll|قطع)")),
)
_TREATMENT_RE = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in _TREATMENT_PATTERNS]
# "instead of X" / "not X": the mention right after these is what the user is moving AWAY from.
_REPLACED = re.compile(
    r"(?:بدل(?:ا)?(?:\s+من)?|عوض(?:ا)?(?:\s+عن)?|مش|مش\s+عايز|instead\s+of|rather\s+than|not|no\s+more|without)\s*(?:ال)?",
    re.IGNORECASE,
)
_NO_TREATMENT = re.compile(
    r"no\s+treatment|without\s+(?:any\s+)?(?:treatment|edit\w*)|من\s*غير\s*(?:اي\s*)?(?:treatment|تدخل|تعديل|حاجه)|"
    r"بلاش|شيل\w*|احذف\w*|remove|skip|drop|delete|مش\s*محتاج\w*|ماحتاج\w*|no\s+need",
    re.IGNORECASE,
)
_REJECT = re.compile(r"(?:^|\s)(?:رفض\w*|reject\w*|مش\s*موافق|غير\s*موافق|disapprove\w*|لا\s+شكرا|no\s+thanks?)(?:\s|$)", re.IGNORECASE)
_GENERATE_OK = re.compile(
    r"(?<!\w)(?:generate(?!d)|go\s+ahead\s+and\s+generate|ولد(?:ها|ه)?|ولد\s+ده|اعتمد\s+(?:ال)?توليد|approve\s+generation|"
    r"generation\s+(?:ok|approved)|(?:سمحت|اسمح)\s+ب(?:ال)?توليد)(?!\w)",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"(?<!\w)(?:لا|ما|مش|متعملش|ماتعملش|ماتولدش|متولدش|don'?t|do\s+not|not|never|no)(?!\w)", re.IGNORECASE)
_OPTION = re.compile(
    r"(?:الخيار|الاختيار|خيار|اختار\w*|اختر|option|choice|choose|pick|variant)\s*(?:الخيار|option|رقم|no\.?|#)?\s*"
    r"(?P<o>[a-cA-C]|[أابج]|\d)(?![\w-])",
    re.IGNORECASE,
)
_BARE_OPTION = re.compile(r"^\s*(?P<n>\d+)\s*[:\-–]?\s*(?P<o>[a-cA-Cأبج])\s*[.!]?\s*$")
_TEXT_KEY = re.compile(
    rf"^\s*(?:{_LABEL}\s*)?(?P<n>\d+)\s*(?:(?:ال)?(?:نص|كلمه|كلمة|عبار[ةه]|text|keyword|phrase|word)\s*)"
    r"(?:[:：=]\s*|\s+(?=[\"“«'`]))(?P<t>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_ALL = re.compile(r"(?:\ball\b|الكل|كلهم|كل\s*(?:الحاجات|حاجه|الاماكن|التعديلات|الفتحات)|everything|show\s+all)", re.IGNORECASE)
_ASSET = re.compile(
    r"(?:asset|assets|اصل|اصول|محتاج\w*\s*(?:footage|فيديو|لقطات?)|footage|needs?\s+(?:an?\s+)?(?:asset|footage))", re.IGNORECASE)
_GENERATED_VIEW = re.compile(r"(?:generated|generate\b|مولد\w*|ai\b|توليد)", re.IGNORECASE)
_APPROVAL_PLAIN = _APPROVE_WORDS | {"تمام", "اوكي", "اوك", "حلو", "جميل", "عاجبني", "عاجبه", "yep", "yeah", "sure", "great", "perfect"}
_LETTER = {"a": 0, "b": 1, "c": 2, "أ": 0, "ا": 0, "ب": 1, "ج": 2}


def _unquote(text: str) -> str:
    text = text.strip()
    pairs = {'"': '"', "'": "'", "“": "”", "«": "»", "`": "`"}
    if len(text) >= 2 and pairs.get(text[0]) == text[-1]:
        return text[1:-1].strip()
    return text


def _is_approval(bare: str) -> bool:
    tokens = [t for t in re.findall(r"[^\W\d_]+|#", bare.lower()) if t not in _FILLER]
    return bool(tokens) and all(t in _APPROVAL_PLAIN for t in tokens)


def find_treatment(bare: str) -> str | None:
    """The treatment the message asks FOR: mentions right after a 'instead of' are
    ignored, and the last remaining mention wins ("بدل generated اعمله motion graphic")."""
    text = bare
    spans: list[tuple[int, int]] = []
    for m in _REPLACED.finditer(text):
        for _, rx in _TREATMENT_RE:
            hit = rx.match(text, m.end())
            if hit:
                spans.append((m.start(), hit.end()))
    keep = "".join(text[i] if not any(a <= i < b for a, b in spans) else " " for i in range(len(text)))
    found = [(hit.start(), hit.end(), name) for name, rx in _TREATMENT_RE for hit in rx.finditer(keep)]
    # "data graphic" is one treatment, not "data" + "graphic": a mention inside a longer one is part of it.
    found = [h for h in found if not any(o[2] != h[2] and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0]) for o in found)]
    hits = [(start, name) for start, _end, name in found]
    # "speaker" is also a passing noun ("punch in on the speaker"): it only counts when nothing else was asked for.
    specific = [h for h in hits if h[1] != "stay_on_speaker"]
    return max(specific or hits, default=(0, None))[1]


_SOUND_WORD = re.compile(r"(?<![\w])(?:sounds?|sfx|sound\s*effects?|effects?|ساوند|سواند|صوت(?:ي|يه)?|اصوات|مؤثر\w*|موثر\w*)(?![\w])", re.IGNORECASE)
_NO_SOUND = re.compile(
    r"(?:من\s*غير|بدون|بلاش|مفيش|شيل\w*|احذف\w*|no|without|remove|skip|drop|mute|silent|off)\s*(?:اي\s*)?(?:ال)?"
    r"(?:sounds?|sfx|sound\s*effects?|effects?|ساوند|سواند|صوت|اصوات|مؤثر\w*|موثر\w*)", re.IGNORECASE)
_NO_SOUND_TRAILING = re.compile(r"(?:sounds?|sfx|ساوند|سواند|صوت)\s*(?:off|none|بلاش|مش\s*محتاج\w*)", re.IGNORECASE)
_SOUND_INTENT_PATTERNS = (
    ("subtle_motion", r"subtle(?:[\s_-]*motion)?|motion[\s_-]*sound|خفيف\w*|هادي"),
    ("transition", r"transition|whoosh|sweep|انتقال\w*|ترانزيشن"),
    ("accent", r"accents?|tick|اكسنت|أكسنت|تيك"),
    ("impact", r"impact|hit|thud|امباكت|إمباكت"),
)
_SOUND_INTENT_RE = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in _SOUND_INTENT_PATTERNS]


def find_sound(bare: str) -> str | None:
    """The sound intent a message asks for, or None when it is not about sound.

    Sound is only mentioned with a sound word (`sound`, `sfx`, `صوت` ...), so a bare `transition`
    stays a visual word. "without/no sound" -> `none`; "sound accent instead of transition" -> `accent`
    (a mention right after `instead of` is what the user is moving away from)."""
    if not _SOUND_WORD.search(bare):
        return None
    if _NO_SOUND.search(bare) or _NO_SOUND_TRAILING.search(bare):
        return "none"
    keep = list(bare)
    for m in _REPLACED.finditer(bare):
        for _, rx in _SOUND_INTENT_RE:
            hit = rx.match(bare, m.end())
            if hit:
                keep[m.start():hit.end()] = " " * (hit.end() - m.start())
    text = "".join(keep)
    hits = [(hit.start(), name) for name, rx in _SOUND_INTENT_RE for hit in rx.finditer(text)]
    return max(hits)[1] if hits else None


def parse(message: str) -> Command:
    raw = message.strip()
    if not raw:
        return Command(KIND_SHOW, view=VIEW_PENDING)
    norm = _norm(raw)
    bare = _bare(raw).lower()
    numbers = _numbers(bare)

    m = _TEXT_KEY.match(norm)
    if m:
        return Command(KIND_SET_TEXT, numbers=(int(m.group("n")),), text=_unquote(raw[m.start("t"):m.end("t")]))

    if _HELP.match(bare):
        return Command(KIND_HELP)
    if _APPROVE_REST.search(bare):
        return Command(KIND_APPROVE_REST)

    if _SHOW_VERB.search(bare):
        if _ALL.search(bare) and not numbers:
            return Command(KIND_SHOW, view=VIEW_ALL)
        if _ASSET.search(bare):
            return Command(KIND_SHOW, view=VIEW_ASSET)
        if _GENERATED_VIEW.search(bare):
            return Command(KIND_SHOW, view=VIEW_GENERATED)
        if numbers:
            return Command(KIND_SHOW, numbers=numbers, view=VIEW_NUMBERS)
        return Command(KIND_SHOW, view=VIEW_PENDING)

    m = _BARE_OPTION.match(norm)
    if m:
        return Command(KIND_CHOOSE_OPTION, numbers=(int(m.group("n")),), option=_LETTER[m.group("o").strip().lower()])
    m = _OPTION.search(bare)
    if m:
        letter = m.group("o").lower()
        option = _LETTER[letter] if letter in _LETTER else int(letter) - 1
        slot = _numbers(bare[:m.start("o")] + " " + bare[m.end("o"):])[:1] if letter.isdigit() else numbers[:1]
        if option >= 0:
            return Command(KIND_CHOOSE_OPTION, numbers=slot, option=option)

    sound = find_sound(bare)
    if sound is not None:
        return Command(KIND_SET_SOUND, numbers=numbers[:1], sound=sound)

    negated = bool(_NEGATION.search(bare))
    if _GENERATE_OK.search(bare) and not negated:
        return Command(KIND_APPROVE_GENERATION, numbers=numbers[:1])

    target = find_treatment(bare)
    if target:
        return Command(KIND_SET_TREATMENT, numbers=numbers[:1], treatment=target)
    if _REJECT.search(f" {bare} ") or _NO_TREATMENT.search(bare):
        return Command(KIND_REJECT, numbers=numbers)
    if _is_approval(bare):
        return Command(KIND_APPROVE, numbers=numbers)
    if numbers and not re.search(r"[^\W\d_]", bare):
        return Command(KIND_SHOW, numbers=numbers, view=VIEW_NUMBERS)
    return Command(KIND_UNKNOWN, text=raw)


def split_entries(message: str) -> list[str]:
    """One message may decide several slots (one per line: `1 موافق` / `2 خليه speaker`)."""
    return split_message(message)


__all__ = ["Command", "find_sound", "find_treatment", "parse", "split_entries"]
