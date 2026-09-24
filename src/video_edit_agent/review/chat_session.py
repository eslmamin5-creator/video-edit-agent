"""One conversation turn of the chat-first transcript review.

The agent passes the user's message to `TranscriptReviewChat.reply` and shows
`ChatReply.text` (plus any `audio_paths`) in the chat. Everything the user says
is understood by `chat_commands.parse`; everything they see is built by
`transcript_chat`; what changes is persisted through the existing correction
and review-state modules, so a reload (or a new conversation) starts exactly
where the user left off.

Approval semantics (one rule, documented for users in `format_help`):

- the user types a whole corrected sentence, or says a segment is right
  -> `approved` (they confirmed the exact wording);
- the user fixes only a word or phrase -> `corrected_pending_approval`
  (they have not seen the result yet; it keeps blocking final approval until
  they say "ok");
- nothing is ever approved by silence, and unresolved / pending segments block
  `review-approve`.

Raw ASR (`transcript_unified.json`) is never written. Text the user dictates is
stored verbatim -- no translation, no formalizing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.core.schemas import Transcript
from video_edit_agent.review import chat_commands as cmd
from video_edit_agent.review import corrections as corr
from video_edit_agent.review import state as review_state
from video_edit_agent.review import transcript_chat as tc
from video_edit_agent.review.audio_clip import AudioClipError, ensure_clip, find_source_audio
from video_edit_agent.review.schemas import (
    SUSPICIOUS_CONFIDENCE_THRESHOLD,
    UnresolvedTranscriptItem,
)
from video_edit_agent.transcription.router import load_transcript

TRANSCRIPT_FILENAME = "transcript_unified.json"
CURSOR_FILENAME = "chat_cursor.json"


@dataclass
class ChatReply:
    text: str
    changed: bool = False  # review state or corrections were written
    audio_paths: list[Path] = field(default_factory=list)
    note: str = ""  # the confirmation / explanation line(s) above the list
    body: str = ""  # the list of segments below it


def open_transcript_review(review_dir: Path, transcript: Transcript) -> list[int]:
    """Starts a chat review: every segment with low-confidence words that the
    user has not decided yet is flagged `unresolved`, so it blocks approval until
    they answer. Returns the 1-based numbers flagged. Idempotent."""
    state = review_state.load_review_state(review_dir)
    decided = {d.segment_id for d in state.segment_reviews}
    open_ids = {i.segment_id for i in state.unresolved_transcript}
    corrected = {c.segment_id for c in corr.load_corrections(review_dir)}
    flagged: list[int] = []
    for number, seg in enumerate(transcript.segments, start=1):
        if seg.id in decided | open_ids | corrected:
            continue
        low = [w for w in seg.words if w.confidence < SUSPICIOUS_CONFIDENCE_THRESHOLD]
        if not low:
            continue
        reason = "low-confidence: " + ", ".join(f"{w.word} ({w.confidence:.2f})" for w in low)
        review_state.flag_unresolved(
            review_dir,
            UnresolvedTranscriptItem(segment_id=seg.id, segment=number, asr_text=seg.text, reason=reason),
        )
        flagged.append(number)
    return flagged


def format_help(lang: str = "ar") -> str:
    if lang == "ar":
        return (
            "تقدر تردّ عادي، مثلًا:\n"
            "  1 صح                              ← اعتماد جملة\n"
            "  5: <الجملة الصحيحة كاملة>          ← استبدال الجملة (بتتعتمد على طول)\n"
            "  في الجملة 9 غير كلمة X إلى Y       ← تصحيح كلمة (بتستنى موافقتك بعدها)\n"
            "  اسمعني الجملة 15                   ← مقطع صوتي للجملة دي بس\n"
            "  وريني بس الجمل اللي فيها شك | راجعلي 9 و14 و15 بس | الكل | من 3 إلى 7\n"
            "  التالي / السابق                    ← صفحات\n"
            "  اعتمد الباقي                       ← اعتماد اللي معروض قدامك"
        )
    return (
        "Just reply naturally, e.g.:\n"
        "  1 ok                               approve a segment\n"
        "  5: <the full correct sentence>     replace it (approved right away)\n"
        "  in segment 9 change X to Y         fix a word (waits for your ok)\n"
        "  play segment 15                    short audio clip of that segment only\n"
        "  show only the suspicious ones | review 9 and 14 and 15 | show all | 3 to 7\n"
        "  next / previous                    pages\n"
        "  approve the rest                   approve what is on screen"
    )


class TranscriptReviewChat:
    def __init__(
        self, edit_dir: Path, *, lang: str = "ar", page_size: int = tc.DEFAULT_PAGE_SIZE,
        source_media: Path | None = None,
    ) -> None:
        self.edit_dir = Path(edit_dir)
        self.review_dir = self.edit_dir / "review"
        self.lang = lang
        self.page_size = page_size
        self.source_media = source_media

    # ---- loading -------------------------------------------------------------
    def _raw(self) -> Transcript:
        return load_transcript(self.edit_dir / TRANSCRIPT_FILENAME)

    def _views(self) -> list[tc.SegmentView]:
        effective = corr.apply_corrections(self._raw(), corr.load_corrections(self.review_dir))
        return tc.build_views(
            effective, corr.load_corrections(self.review_dir), review_state.load_review_state(self.review_dir),
        )

    def _load_cursor(self) -> dict:
        path = self.review_dir / CURSOR_FILENAME
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_cursor(self, spec: tc.ViewSpec, shown: list[tc.SegmentView], last: int | None) -> None:
        cursor = {
            "view": spec.kind, "numbers": list(spec.numbers), "page": spec.page,
            "last_shown": [v.number for v in shown], "last_segment": last,
        }
        self.review_dir.mkdir(parents=True, exist_ok=True)
        (self.review_dir / CURSOR_FILENAME).write_text(json.dumps(cursor, indent=2), encoding="utf-8")

    def _spec(self, cursor: dict) -> tc.ViewSpec:
        return tc.ViewSpec(
            kind=cursor.get("view", tc.VIEW_PENDING), numbers=tuple(cursor.get("numbers", ())),
            page=int(cursor.get("page", 0)),
        )

    # ---- one turn --------------------------------------------------------------
    def reply(self, message: str | None = None) -> ChatReply:
        entries = cmd.split_message(message or "")
        if len(entries) == 1:
            return self._reply_one(entries[0])
        # several segments answered in one message: apply each in order, then
        # show every confirmation followed by ONE refreshed list.
        replies = [self._reply_one(entry) for entry in entries]
        notes = [r.note for r in replies if r.note]
        audio = [p for r in replies for p in r.audio_paths]
        body = replies[-1].body
        return ChatReply(
            text="\n\n".join([*notes, body] if body else notes),
            changed=any(r.changed for r in replies), audio_paths=audio, note="\n\n".join(notes), body=body,
        )

    def _reply_one(self, message: str) -> ChatReply:
        views = self._views()
        cursor = self._load_cursor()
        command = cmd.parse(message)
        ar = self.lang == "ar"

        numbered = command.kind in _NUMBERED and command.view != tc.VIEW_RANGE
        bad = [n for n in command.numbers if numbered and not 1 <= n <= len(views)]
        if bad:
            return self._present(views, self._spec(cursor), cursor, prefix=(
                f"مفيش جملة رقم {', '.join(map(str, bad))} (الجمل من 1 لـ {len(views)})." if ar
                else f"There is no segment {', '.join(map(str, bad))} (segments run 1-{len(views)})."
            ))

        handler = {
            cmd.KIND_SHOW: self._show, cmd.KIND_NEXT: self._page, cmd.KIND_PREVIOUS: self._page,
            cmd.KIND_APPROVE: self._approve, cmd.KIND_APPROVE_REST: self._approve_rest,
            cmd.KIND_REPLACE_TEXT: self._replace_text, cmd.KIND_REPLACE_WORDS: self._replace_words,
            cmd.KIND_AUDIO: self._audio, cmd.KIND_HELP: self._help,
        }.get(command.kind)
        if handler is None:
            return self._present(views, self._spec(cursor), cursor, prefix=(
                "ما فهمتش الطلب. " + format_help(self.lang) if ar else "I did not get that. " + format_help(self.lang)
            ))
        return handler(command, views, cursor)

    # ---- presentation ------------------------------------------------------------
    def _present(
        self, views: list[tc.SegmentView], spec: tc.ViewSpec, cursor: dict, *, prefix: str = "",
        changed: bool = False, last: int | None = None, audio: list[Path] | None = None,
    ) -> ChatReply:
        text, shown, page = tc.format_review(views, spec, lang=self.lang, page_size=self.page_size)
        self._save_cursor(
            tc.ViewSpec(spec.kind, spec.numbers, page), shown, last if last is not None else cursor.get("last_segment"),
        )
        return ChatReply(
            text=f"{prefix}\n\n{text}" if prefix else text, changed=changed, audio_paths=audio or [],
            note=prefix, body=text,
        )

    def _after_change(
        self, prefix: str, cursor: dict, last: int | None, audio: list[Path] | None = None,
    ) -> ChatReply:
        """Confirmation line plus the refreshed list of what is still open."""
        views = self._views()
        spec = tc.ViewSpec(tc.VIEW_PENDING) if cursor.get("view") in (None, tc.VIEW_PENDING) else self._spec(cursor)
        return self._present(views, spec, cursor, prefix=prefix, changed=True, last=last, audio=audio)

    def _show(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        spec = tc.ViewSpec(command.view or tc.VIEW_PENDING, command.numbers)
        return self._present(views, spec, cursor)

    def _page(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        spec = self._spec(cursor)
        step = 1 if command.kind == cmd.KIND_NEXT else -1
        return self._present(views, tc.ViewSpec(spec.kind, spec.numbers, spec.page + step), cursor)

    def _help(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        return self._present(views, self._spec(cursor), cursor, prefix=format_help(self.lang))

    # ---- approvals -----------------------------------------------------------------
    def _approve(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        numbers = list(command.numbers) or ([cursor["last_segment"]] if cursor.get("last_segment") else [])
        if not numbers:
            msg = "أعتمد أنهي جملة؟ اكتب رقمها، مثلًا: 9 صح" if self.lang == "ar" else "Which segment? e.g. `9 ok`"
            return self._present(views, self._spec(cursor), cursor, prefix=msg)
        for n in numbers:
            review_state.resolve_unresolved(self.review_dir, views[n - 1].segment_id)
        label = "، ".join(map(str, numbers)) if self.lang == "ar" else ", ".join(map(str, numbers))
        msg = f"✔ اتعتمدت: {label}" if self.lang == "ar" else f"✔ Approved: {label}"
        return self._after_change(msg, cursor, numbers[-1])

    def _approve_rest(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        spec = self._spec(cursor)
        scope = tc.select_view(views, spec) if spec.kind != tc.VIEW_PENDING else views
        rest = [v for v in scope if v.is_pending]
        if not rest:
            msg = "مفيش حاجة مفتوحة في اللي معروض." if self.lang == "ar" else "Nothing open in what is on screen."
            return self._present(views, spec, cursor, prefix=msg)
        for v in rest:
            review_state.resolve_unresolved(self.review_dir, v.segment_id)
        label = ", ".join(str(v.number) for v in rest)
        msg = f"✔ اتعتمدت الباقي: {label}" if self.lang == "ar" else f"✔ Approved the rest: {label}"
        return self._after_change(msg, cursor, rest[-1].number)

    # ---- corrections -----------------------------------------------------------------
    def _replace_text(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        number, text = command.numbers[0], (command.text or "").strip()
        view = views[number - 1]
        raw = self._raw().segments[number - 1]
        if text == raw.text.strip():
            corr.clear_corrections(self.review_dir, view.segment_id)  # same as the ASR: nothing to store
        else:
            corr.set_segment_text(self.review_dir, view.segment_id, text)
        review_state.resolve_unresolved(self.review_dir, view.segment_id)
        msg = (
            f"✔ [{number:02d}] اتحفظت واتعتمدت:\n{text}" if self.lang == "ar"
            else f"✔ [{number:02d}] saved and approved:\n{text}"
        )
        return self._after_change(msg, cursor, number)

    def _replace_words(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        old, new = command.old or "", command.new or ""
        ar = self.lang == "ar"
        if command.everywhere:
            targets = [v for v in views if corr.replace_phrase(v.text, old, new)[1] >= 1]
        elif command.numbers:
            targets = [views[command.numbers[0] - 1]]
        else:
            candidates = [v for v in views if corr.replace_phrase(v.text, old, new)[1] >= 1]
            targets = candidates if len(candidates) == 1 else []
            if not targets:
                where = ", ".join(str(v.number) for v in candidates)
                if candidates:
                    msg = (
                        f"«{old}» موجودة في أكتر من جملة ({where}). قولّي أنهي جملة، مثلًا: في الجملة {candidates[0].number} غير {old} إلى {new}" if ar
                        else f"“{old}” appears in several segments ({where}). Say which one, e.g. in segment {candidates[0].number} change {old} to {new}"
                    )
                else:
                    msg = f"ما لقيتش «{old}» في أي جملة." if ar else f"I could not find “{old}” in any segment."
                return self._present(views, self._spec(cursor), cursor, prefix=msg)

        done: list[int] = []
        skipped: list[str] = []
        for view in targets:
            fresh, count = corr.replace_phrase(view.text, old, new)
            if fresh is None:
                skipped.append(
                    (f"{view.number}: «{old}» ظاهرة {count} مرات" if count > 1 else f"{view.number}: مش لاقي «{old}»")
                    if ar else (f"{view.number}: “{old}” appears {count} times" if count > 1 else f"{view.number}: “{old}” not found")
                )
                continue
            corr.set_segment_text(self.review_dir, view.segment_id, fresh)
            review_state.mark_corrected_pending(self.review_dir, view.segment_id)
            done.append(view.number)

        lines: list[str] = []
        if done:
            new_views = {v.number: v for v in self._views()}
            for n in done:
                lines.append(f"✔ [{n:02d}] {new_views[n].text}")
            head = (
                f"اتصلحت «{old}» ← «{new}» في: {', '.join(map(str, done))}. الجملة بعد التعديل تحت؛ قولّي «صح» لو تمام."
                if ar else
                f"Changed “{old}” → “{new}” in: {', '.join(map(str, done))}. Result below; say ok to approve."
            )
            lines.insert(0, head)
        if skipped:
            lines.append(("محتاجة توضيح: " if ar else "Needs your call: ") + "; ".join(skipped))
        return self._after_change("\n".join(lines), cursor, done[-1] if done else cursor.get("last_segment")) if done else \
            self._present(views, self._spec(cursor), cursor, prefix="\n".join(lines))

    # ---- audio (on demand only) ---------------------------------------------------------
    def _audio(self, command: cmd.Command, views: list[tc.SegmentView], cursor: dict) -> ChatReply:
        ar = self.lang == "ar"
        numbers = list(command.numbers) or ([cursor["last_segment"]] if cursor.get("last_segment") else [])
        if not numbers:
            msg = "أسمّعك أنهي جملة؟ مثلًا: اسمعني الجملة 9" if ar else "Which segment? e.g. `play segment 9`"
            return self._present(views, self._spec(cursor), cursor, prefix=msg)
        source = find_source_audio(self.edit_dir, self.source_media)
        clips: list[Path] = []
        problems: list[str] = []
        for n in numbers[:3]:
            v = views[n - 1]
            try:
                clips.append(ensure_clip(self.review_dir, n, v.start, v.end, source))
            except AudioClipError as exc:
                problems.append(f"{n}: {exc}")
        lines = []
        for n, clip in zip(numbers, clips, strict=False):
            v = views[n - 1]
            lines.append(
                f"🔊 [{n:02d}] {tc.format_timestamp(v.start)}–{tc.format_timestamp(v.end)}\n{v.text}"
            )
        if problems:
            lines.append(("ما قدرتش أعمل المقطع: " if ar else "Could not make the clip: ") + "; ".join(problems))
        if len(numbers) > 3:
            lines.append("(3 مقاطع في المرة)" if ar else "(3 clips at a time)")
        return self._present(
            views, self._spec(cursor), cursor, prefix="\n".join(lines), last=numbers[0], audio=clips,
        )


_NUMBERED = (cmd.KIND_APPROVE, cmd.KIND_REPLACE_TEXT, cmd.KIND_REPLACE_WORDS, cmd.KIND_AUDIO, cmd.KIND_SHOW)

__all__ = ["ChatReply", "TranscriptReviewChat", "format_help", "open_transcript_review"]
