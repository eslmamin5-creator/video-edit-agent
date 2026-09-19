"""Understands what the user says about the transcript, in Arabic or English.

No rigid command syntax: the agent hands the user's message here and gets back
a `Command`. Text the user dictates (a corrected sentence, the replacement word)
is returned VERBATIM -- nothing is translated, formalized or "improved". Only
numbers are normalised (Arabic-Indic digits -> 0-9) so sentence numbers match.

Recognised (examples, not an exhaustive grammar):

    1 صح                              approve 1            -> approve
    اعتمد الباقي / approve the rest    -> approve_rest
    الجملة 5: <sentence>  /  5: <s>    -> replace_text
    الجملة 14 كلها المفروض تكون: <s>   -> replace_text
    في الجملة 9 غير كلمة X إلى Y       -> replace_words (segment 9)
    غير service إلى سيرفس              -> replace_words (any segment)
    اسمعني الجملة 15 / play 15         -> audio
    وريني بس الجمل اللي فيها شك        -> show (suspicious only)
    راجعلي 9 و14 و15 بس               -> show (those segments)
    الكل / show all / من 3 إلى 7       -> show (all / range)
    التالي / السابق / next / previous  -> next / previous
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KIND_SHOW = "show"
KIND_NEXT = "next"
KIND_PREVIOUS = "previous"
KIND_APPROVE = "approve"
KIND_APPROVE_REST = "approve_rest"
KIND_REPLACE_TEXT = "replace_text"
KIND_REPLACE_WORDS = "replace_words"
KIND_AUDIO = "audio"
KIND_HELP = "help"
KIND_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Command:
    kind: str
    numbers: tuple[int, ...] = ()
    view: str | None = None  # show: "pending" | "all" | "range" | "numbers"
    text: str | None = None  # replace_text: the full corrected sentence, verbatim
    old: str | None = None  # replace_words
    new: str | None = None  # replace_words, verbatim
    everywhere: bool = False  # replace_words: apply to every segment that has it


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_HAMZA = str.maketrans("أإآ", "ااا")
_MARKS = re.compile("[ً-ْـ]")  # tashkeel + tatweel: ignored when matching keywords

_LABEL = r"(?:الجمل[ةه]|جمل[ةه]|سيجمنت|segments?|sentences?|seg|lines?|رقم|#)"
_SEP = r"[:：,،\-–=]"
_VERB = r"(?:غي[ً-ْ]*ر|بد[ً-ْ]*ل|استبدل|change|replace|swap)"
_CONNECT = r"(?:إلى|الى|إلي|الي|ل|to|with|into|by|ب|->|→|حط|خلي|خليها|اكتب|اكتبها)"
_REPLACE = re.compile(
    rf"^\s*(?:(?:في\s+)?{_LABEL}\s*(?P<n1>\d+)\s*{_SEP}?\s*|(?P<n2>\d+)\s*{_SEP}\s*)?"
    rf"(?:(?:in|at)\s+{_LABEL}\s*(?P<n3>\d+)\s*{_SEP}?\s*)?"
    rf"{_VERB}\s+(?:(?:كلم[ةه]|عبار[ةه]|the\s+word|the\s+phrase|word|phrase)\s+)?"
    rf"(?P<old>.+?)\s+{_CONNECT}\s+(?P<new>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REPLACE_TAIL = re.compile(
    rf"\s+(?:(?:في|in|at)\s+)?(?:(?P<all>(?:كل|كلها|all)\s+(?:الجمل|مكان|segments|places)|everywhere)|"
    rf"{_LABEL}\s*(?P<n>\d+))\s*$",
    re.IGNORECASE,
)
_TEXT_COLON = re.compile(
    rf"^\s*(?:{_LABEL}\s*)?(?P<n>\d+)\s*(?:(?:كلها|كله|كاملة|بالكامل|whole|entire)\s*)?"
    rf"(?:(?:المفروض|لازم|ينفع|should|must)\s*(?:تكون|يكون|تبقى|be|read)?\s*)?"
    r"(?:[:：=]|->|→)\s*(?P<text>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TEXT_SHOULD = re.compile(
    rf"^\s*(?:{_LABEL}\s*)?(?P<n>\d+)\s*(?:(?:كلها|كله|كاملة|بالكامل|whole|entire)\s*)?"
    r"(?:المفروض|لازم|ينفع|should|must)\s*(?:تكون|يكون|تبقى|be|read)\s+(?P<text>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

_APPROVE_WORDS = {
    "صح", "صحيح", "صحيحه", "تمام", "اعتمد", "اعتمدي", "اعتماد", "اعتمدها", "موافق", "ماشي", "كويس",
    "سليمه", "مظبوطه", "مظبوط", "ok", "okay", "approve", "approved", "correct", "good", "yes",
    "right", "fine", "lgtm", "accept", "confirmed", "confirm",
}
_FILLER = {
    "و", "and", "،", ",", ":", "-", "كلها", "كله", "all", "بس", "only", "the", "this", "that", "is",
    "are", "دي", "ده", "دى", "it", "كمان", "also", "please", "لو", "سمحت", "segment", "segments",
    "sentence", "sentences", "seg", "line", "lines", "رقم", "#", "جمله", "الجمله", "جمل", "الجمل", "سيجمنت",
    "جملة", "الجملة",
    "to", "من", "الي", "الى",
}
_APPROVE_REST = re.compile(
    r"(?:اعتمد|اعتمدي|وافق|approve|accept)\s+(?:باقي|الباقي|البقيه|البقية|الباقيه|الكل|كله|الجمل\s+الباقيه|"
    r"(?:the\s+)?(?:rest|remaining|others|everything|all))|(?:الباقي|الباقيه)\s+(?:صح|تمام|كله)",
    re.IGNORECASE,
)
_AUDIO = re.compile(
    r"(?:اسمعني|سمعني|اسمع|شغللي|شغلي|شغل|شغلني|play|listen|hear|audio|صوت|مش\s+فاكر|فاكر\s+قلت|"
    r"الجزء\s+ده|الجزء\s+دا)",
    re.IGNORECASE,
)
_NEXT = re.compile(r"^\s*(?:التالي|التاليه|الجاي|الجايه|بعده|كمل|كمّل|next|more|continue)\s*[.!؟?]*\s*$", re.IGNORECASE)
_PREV = re.compile(r"^\s*(?:السابق|السابقه|اللي\s+قبل|قبله|previous|prev|back)\s*[.!؟?]*\s*$", re.IGNORECASE)
_SUSPICIOUS = re.compile(
    r"(?:شك|مشكوك|الشك|suspicious|low[\s-]?confidence|questionable|needs?\s+review|مش\s+متاكد|مش\s+واضح)",
    re.IGNORECASE,
)
_SHOW_VERB = re.compile(r"(?:وريني|ورّيني|اعرض|اعرضلي|عرض|راجعلي|راجع|show|list|review|display|see)", re.IGNORECASE)
_ALL = re.compile(r"(?:كل\s+الجمل|الكل|كلهم|show\s+all|\ball\b|everything|كله)", re.IGNORECASE)
_RANGE = re.compile(
    r"(?:من\s*)?(?P<a>\d+)\s*(?:-|–|إلى|الى|إلي|الي|to|through|until|لغاية|لحد)\s*(?P<b>\d+)", re.IGNORECASE,
)
_HELP = re.compile(r"^\s*(?:help|مساعده|مساعدة|ساعدني|\?|؟)\s*$", re.IGNORECASE)


def _norm(message: str) -> str:
    """Digits to 0-9 and diacritics/tatweel dropped -- 1:1 in length except for the latter."""
    return message.translate(_DIGITS)


def _bare(message: str) -> str:
    return _MARKS.sub("", _norm(message)).translate(_HAMZA)


def _numbers(text: str) -> tuple[int, ...]:
    """All segment numbers in `text`, with `3-7` / `من 3 إلى 7` expanded, no duplicates, in order."""
    found: list[int] = []
    span_done: list[tuple[int, int]] = []
    for m in _RANGE.finditer(text):
        lo, hi = sorted((int(m.group("a")), int(m.group("b"))))
        if hi - lo <= 500:
            found.extend(range(lo, hi + 1))
            span_done.append(m.span())
    for m in re.finditer(r"\d+", text):
        if not any(a <= m.start() < b for a, b in span_done):
            found.append(int(m.group()))
    return tuple(dict.fromkeys(found))


def _unquote(text: str) -> str:
    text = text.strip()
    pairs = {'"': '"', "'": "'", "“": "”", "«": "»", "`": "`"}
    if len(text) >= 2 and pairs.get(text[0]) == text[-1]:
        return text[1:-1].strip()
    return text


def _only_approval(text: str) -> bool:
    """True when, besides numbers and labels, the message says nothing but 'ok'."""
    tokens = re.findall(r"[^\W\d_]+|#", _bare(text).lower())
    words = [t for t in tokens if t not in _FILLER]
    return bool(words) and all(t in _APPROVE_WORDS for t in words) and bool(re.search(r"\d", text))


def parse(message: str) -> Command:
    raw = message.strip()
    if not raw:
        return Command(KIND_SHOW, view=None)
    norm = _norm(raw)
    bare = _bare(raw)

    m = _REPLACE.match(norm)
    if m:
        old, new = _unquote(raw[m.start("old"):m.end("old")]), raw[m.start("new"):m.end("new")]
        number = next((int(m.group(g)) for g in ("n1", "n2", "n3") if m.group(g)), None)
        everywhere = False
        tail = _REPLACE_TAIL.search(new)
        if tail:
            new = new[:tail.start()]
            if tail.group("all"):
                everywhere = True
            elif number is None:
                number = int(tail.group("n"))
        return Command(
            KIND_REPLACE_WORDS, numbers=(number,) if number else (), old=old, new=_unquote(new),
            everywhere=everywhere,
        )

    if _APPROVE_REST.search(bare):
        return Command(KIND_APPROVE_REST)

    if _only_approval(raw):
        return Command(KIND_APPROVE, numbers=_numbers(norm))

    for pattern in (_TEXT_COLON, _TEXT_SHOULD):
        m = pattern.match(norm)
        if m:
            return Command(KIND_REPLACE_TEXT, numbers=(int(m.group("n")),), text=raw[m.start("text"):m.end("text")])

    if _NEXT.match(bare):
        return Command(KIND_NEXT)
    if _PREV.match(bare):
        return Command(KIND_PREVIOUS)
    if _HELP.match(bare):
        return Command(KIND_HELP)
    if _AUDIO.search(bare):
        return Command(KIND_AUDIO, numbers=_numbers(norm))

    numbers = _numbers(norm)
    if _SUSPICIOUS.search(bare):
        return Command(KIND_SHOW, view="pending")
    if _SHOW_VERB.search(bare):
        rng = _RANGE.search(norm)
        if rng:
            lo, hi = sorted((int(rng.group("a")), int(rng.group("b"))))
            return Command(KIND_SHOW, numbers=(lo, hi), view="range")
        if numbers:
            return Command(KIND_SHOW, numbers=numbers, view="numbers")
        if _ALL.search(bare):
            return Command(KIND_SHOW, view="all")
        return Command(KIND_SHOW, view="pending")
    if _ALL.search(bare) and not numbers:
        return Command(KIND_SHOW, view="all")
    return Command(KIND_UNKNOWN, text=raw)
