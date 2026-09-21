"""Chat-first edit-plan review: what the user sees and how they answer.

`open_edit_plan` builds `<review>/edit_plan.json` from the project's editorial
decisions and the APPROVED transcript. `EditPlanChat.reply` takes the user's
message, applies it through `review.edit_plan`, and returns text meant to be
shown in the conversation (no JSON, no internal ids).

Approval semantics (documented for users in `format_help`):

- `N موافق` accepts the recommended treatment; it does NOT allow AI generation.
- AI generation needs its own explicit answer for that slot (`N ولّد`).
- a slot whose treatment shows text needs the exact text approved (pick an
  option or type it) before the slot is approved.
- local B-roll never blocks: without footage the slot uses its fallback.
- nothing here ever sets `ready_for_final_render`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from video_edit_agent.broll.prompt import NO_TEXT_RULES, REALISM_RULES
from video_edit_agent.broll.treatment import load_decisions
from video_edit_agent.core.schemas import EDL
from video_edit_agent.review import edit_plan as ep
from video_edit_agent.review import edit_plan_commands as cmd
from video_edit_agent.review import edit_plan_direction as direction
from video_edit_agent.review import state as review_state
from video_edit_agent.review import text_copy
from video_edit_agent.review.chat_session import ChatReply
from video_edit_agent.review.corrections import apply_corrections, load_corrections
from video_edit_agent.review.edit_plan import EditPlan, EditPlanSlot, SlotStatus
from video_edit_agent.review.schemas import ApprovalStatus
from video_edit_agent.sound.registry import SfxRegistry, default_sfx_dir, load_registry
from video_edit_agent.transcription.router import load_transcript

TRANSCRIPT_FILENAME = "transcript_unified.json"
EDL_FILENAME = "edl.json"
HOOK_END_S = 3.0  # the hook title's window (see motion.director.plan_hook)

_LABEL = {
    "stay_on_speaker": ("Stay on speaker", "البقاء على المتحدث"),
    "punch_in": ("Punch-in / reframe", "Punch-in / reframe"),
    "kinetic_typography": ("Kinetic typography", "Kinetic typography"),
    "behind_subject_text": ("Behind-subject text", "نص ورا المتحدث"),
    "motion_graphic": ("Motion graphic", "Motion graphic"),
    "local_broll": ("Local / your own B-roll", "B-roll محلي (بتاعك)"),
    "generated_broll": ("AI-generated B-roll", "B-roll مولَّد بالـAI"),
    "illustration": ("Illustration", "Illustration (رسم توضيحي)"),
    "graphic_data_scene": ("Data graphic scene", "مشهد بيانات/انفوجراف"),
    "full_screen_text_scene": ("Full-screen text scene", "مشهد نص بملء الشاشة"),
    "no_treatment": ("No treatment", "من غير تدخل"),
}
_SEES = {
    "stay_on_speaker": ("The speaker stays on screen, uncut.", "المتحدث فضل على الشاشة من غير أي تدخل."),
    "punch_in": ("The same shot with a slow, slight zoom.", "نفس اللقطة بزووم بطيء وخفيف."),
    "kinetic_typography": ("A short phrase animates on screen over the speaker.", "جملة/كلمة قصيرة بتتحرك على الشاشة فوق المتحدث."),
    "behind_subject_text": ("One large keyword sits behind the speaker.", "كلمة كبيرة ورا المتحدث."),
    "motion_graphic": ("A simple animated graphic (shapes/lines, no invented text or numbers).", "رسم متحرك بسيط (أشكال/خطوط من غير نص أو أرقام مخترعة)."),
    "local_broll": ("A cutaway of your own real footage while the speaker keeps talking.", "قطع على لقطات حقيقية بتاعتك والمتحدث بيكمّل كلام."),
    "generated_broll": ("An AI-generated cutaway while the speaker keeps talking.", "قطع على لقطة مولَّدة بالـAI والمتحدث بيكمّل كلام."),
    "illustration": ("A simple illustration replaces the speaker while the voice continues.", "رسم توضيحي بسيط بيحل مكان المتحدث والصوت مكمّل."),
    "graphic_data_scene": ("A data/graphic scene (real figures only) replaces the speaker while the voice continues.", "مشهد بيانات (من أرقام حقيقية بس) بيحل مكان المتحدث والصوت مكمّل."),
    "full_screen_text_scene": ("A full-screen text scene replaces the speaker while the voice continues.", "مشهد نص بملء الشاشة بيحل مكان المتحدث والصوت مكمّل."),
    "no_treatment": ("Nothing added; the plain speaker shot.", "مفيش إضافة؛ لقطة المتحدث زي ما هي."),
}
_INTENT_LABEL = {
    "none": ("none", "من غير"),
    "subtle_motion": ("subtle motion", "حركة خفيفة"),
    "transition": ("transition", "انتقال"),
    "accent": ("accent", "أكسنت"),
    "impact": ("impact", "امباكت"),
}
_CAMERA_LABEL = {
    "static": "static", "punch_in": "punch-in", "punch_out": "punch-out", "slow_push": "slow push",
    "reframe_left": "reframe left", "reframe_right": "reframe right", "reset_to_base": "reset to base",
    "n/a": "n/a (speaker replaced)",
}
_STATUS = {
    SlotStatus.PENDING_REVIEW: "PENDING",
    SlotStatus.APPROVED: "APPROVED",
    SlotStatus.CHANGED: "CHANGED (your choice)",
    SlotStatus.REJECTED: "REJECTED (no treatment)",
    SlotStatus.ASSET_REQUIRED: "ASSET REQUIRED (fallback until you supply it)",
    SlotStatus.GENERATION_APPROVED: "GENERATION APPROVED",
}
_HIGH_RISK = re.compile(r"\b(?:face|faces|person|people|crowd|team|man|woman|portrait|screen|dashboard|monitor|laptop|document|chart|graph|sign|logo|text|handwrit\w*|newspaper|book)\b", re.IGNORECASE)
_MEDIUM_RISK = re.compile(r"\b(?:hands?|fingers?|cards?|papers?|sheets?|pen|writing|typing|arranging)\b", re.IGNORECASE)


def label(treatment: str, lang: str = "ar") -> str:
    return _LABEL.get(treatment, (treatment, treatment))[1 if lang == "ar" else 0]


def generation_risk(concept: str | None) -> str:
    """low / medium / high chance of visible AI artifacts for `concept` (generic keyword heuristic:
    faces, screens and documents are worst; hands and small manipulated objects next)."""
    text = concept or ""
    if _HIGH_RISK.search(text):
        return "high"
    if _MEDIUM_RISK.search(text):
        return "medium"
    return "low"


def timestamp(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes):02d}:{sec:05.2f}"


def span(slot: EditPlanSlot) -> str:
    return f"{timestamp(slot.timeline_start)}–{timestamp(slot.timeline_end)}"


# --------------------------------------------------------------------------
# Opening the plan
# --------------------------------------------------------------------------


def open_edit_plan(
    edit_dir: Path, *, settled: dict[str, str] | None = None, rebuild: bool = False,
) -> EditPlan:
    """Creates `review/edit_plan.json` from the editorial decisions (unless it exists;
    `rebuild=True` starts over). Slots already approved elsewhere are settled:
    `settled` maps a treatment to why (e.g. {"punch_in": "approved in the punch-in review"})."""
    edit_dir, review_dir = Path(edit_dir), Path(edit_dir) / "review"
    existing = None if rebuild else ep.load_plan(review_dir)
    if existing is not None:
        return _refreshed(existing, edit_dir)
    transcript = apply_corrections(load_transcript(edit_dir / TRANSCRIPT_FILENAME), load_corrections(review_dir))
    edl = EDL.model_validate(json.loads((edit_dir / EDL_FILENAME).read_text(encoding="utf-8")))
    hook = review_state.load_review_state(review_dir).treatment(text_copy.HOOK_TREATMENT)
    plan = ep.build_edit_plan(
        transcript, edl, load_decisions(review_dir), settled=settled, hook_window=(0.0, HOOK_END_S),
        hook_approved=bool(hook and hook.visual_status is ApprovalStatus.APPROVED and hook.copy_status is ApprovalStatus.APPROVED),
    )
    ep.save_plan(plan, review_dir)
    return plan


def _refreshed(plan: EditPlan, edit_dir: Path) -> EditPlan:
    review_dir = edit_dir / "review"
    transcript = apply_corrections(load_transcript(edit_dir / TRANSCRIPT_FILENAME), load_corrections(review_dir))
    edl = EDL.model_validate(json.loads((edit_dir / EDL_FILENAME).read_text(encoding="utf-8")))
    return ep.refresh_context(plan, transcript, edl)


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def format_slot(slot: EditPlanSlot, lang: str = "ar") -> str:
    ar = lang == "ar"
    seg = ("سيجمنت " if ar else "segment ") + ", ".join(map(str, slot.segments)) if slot.segments else ""
    head = f"[{slot.number}] {span(slot)}" if not slot.settled else f"[–] {span(slot)}"
    lines = [f"{head}  ({slot.duration:g}s{'، ' if ar else ', '}{seg})" if seg else f"{head}  ({slot.duration:g}s)"]
    lines.append(f"Spoken idea: «{slot.spoken_context}»")
    rec = label(slot.treatment, lang)
    if slot.treatment != slot.recommended:
        rec += (f"  (كان المقترح: {label(slot.recommended, lang)})" if ar else f"  (was recommended: {label(slot.recommended, lang)})")
    lines.append(f"Recommended: {rec}" if slot.status is SlotStatus.PENDING_REVIEW else f"Treatment: {rec}")
    if slot.reason:
        lines.append(f"Why: {slot.reason}")
    lines.append(f"What you would see: {_SEES[slot.treatment][0 if not ar else 1]}")
    if slot.directed:
        lines.extend(_direction_lines(slot, ar))
    lines.append(f"Needs asset: {'YES' if slot.needs_asset else 'NO'}")
    lines.append(f"AI generation: {'YES' if slot.needs_generation else 'NO'}")
    if slot.treatment == "generated_broll":
        lines.extend(_generated_lines(slot, ar))
    elif slot.treatment == "local_broll":
        lines.extend(_local_lines(slot, ar))
    if slot.treatment in ep.TEXT_TREATMENTS:
        lines.extend(_text_lines(slot, ar))
    if slot.alternatives and slot.status is SlotStatus.PENDING_REVIEW:
        lines.append(("بدائل: " if ar else "Alternatives: ") + " / ".join(label(a, lang) for a in slot.alternatives))
    lines.append(f"Status: {_STATUS[slot.status]}" if not slot.settled else f"Status: {_STATUS[slot.status]} — {slot.settled_reason}")
    return "\n".join(lines)


def _direction_lines(slot: EditPlanSlot, ar: bool) -> list[str]:
    """The per-beat direction block: what the speaker does, the camera, the cut, the sound and the captions."""
    i = 1 if ar else 0
    speaker = (("visible" if not ar else "ظاهر") if slot.speaker_visible
               else ("hidden, voice continues" if not ar else "مخفي والصوت مكمّل"))
    caption = slot.caption_mode if slot.caption_mode != "unchanged" else ("current" if not ar else "زي الحالي")
    caption += f", {slot.caption_behavior}" if slot.caption_behavior != "normal" else ""
    mine = (" (your choice)" if not ar else " (اختيارك)") if slot.sound_locked else ""
    return [
        f"Visual: {label(slot.treatment, 'ar' if ar else 'en')}",
        f"Speaker: {speaker}",
        f"Camera: {_CAMERA_LABEL.get(slot.camera, slot.camera)}",
        f"Transition: {direction.transition_label(slot)}",
        f"Sound intent: {_INTENT_LABEL.get(slot.sound_intent, (slot.sound_intent,) * 2)[i]}{mine}",
        f"SFX availability: {slot.sfx_availability or 'none'}",
        f"Caption behavior: {caption}",
        *[f"Note: {n}" for n in slot.direction_notes],
    ]


def _generated_lines(slot: EditPlanSlot, ar: bool) -> list[str]:
    drafted = slot.draft_prompt or slot.visual_concept
    concept = slot.visual_concept if slot.recommended == "generated_broll" else drafted  # the concept belongs to the recommendation
    risk = generation_risk(drafted)
    out = [
        f"Concept: {concept or '—'}",
        f"Insertion: {'قطع مباشر (cutaway) والصوت بتاع المتحدث مكمّل' if ar else 'straight cutaway, speaker audio continues'} · ~{slot.duration:g}s",
        f"AI-artifact risk: {risk.upper()}",
        f"Draft prompt: {slot.draft_prompt or slot.visual_concept or '—'} {REALISM_RULES} {NO_TEXT_RULES}",
        ("(الـprompt ده مسودة؛ لسه ما اتولّدش حاجة.)" if ar else "(this is a draft; nothing has been generated.)"),
    ]
    if slot.generation_approved:
        out.append("Generation: APPROVED by you" if not ar else "التوليد: معتمد منك")
    else:
        out.append("Generation: NOT approved" if not ar else "التوليد: لسه مش معتمد")
    return out


def _local_lines(slot: EditPlanSlot, ar: bool) -> list[str]:
    fb = label(slot.fallback or "stay_on_speaker", "ar" if ar else "en")
    return [
        f"Footage needed: {slot.visual_concept or '—'} · ~{slot.duration:g}s",
        f"If you have none: {fb}" if not ar else f"لو مفيش عندك footage: {fb}",
    ]


def _text_lines(slot: EditPlanSlot, ar: bool) -> list[str]:
    if slot.text:
        return [f"On-screen text: «{slot.text}»"]
    if not slot.text_options:
        return ["On-screen text: not chosen — send it as `N النص: ...`" if not ar else "النص: لسه ماتحددش — ابعته كده `N النص: ...`"]
    opts = [f"  {chr(65 + i)}) {o}" for i, o in enumerate(slot.text_options)]
    return ["Text options (from what the speaker said):", *opts]


def summary(plan: EditPlan, lang: str = "ar") -> str:
    reviewable = plan.reviewable()
    open_n = len(plan.pending())
    done = len(reviewable) - open_n
    if lang == "ar":
        return f"متبقي {open_n} قرار من {len(reviewable)} (اتحسم {done})."
    return f"{open_n} decision(s) open of {len(reviewable)} ({done} decided)."


def format_help(lang: str = "ar") -> str:
    if lang == "ar":
        return (
            "تقدر تردّ عادي، مثلًا:\n"
            "  1 موافق                        ← اعتماد الـtreatment المقترح (مش بيسمح بالتوليد)\n"
            "  2 خليه speaker                 ← تغيير الـtreatment\n"
            "  3 بدل generated اعمله motion graphic\n"
            "  4 استخدم B-roll محلي           ← لو معندكش footage بيتطبق الـfallback\n"
            "  5 اختار الخيار B | 5 النص: <كلمة>   ← نص ورا المتحدث\n"
            "  3 ولّد                          ← إذن صريح بتوليد B-roll الـslot ده بس\n"
            "  4 من غير sound | 4 sound accent | 4 sound accent بدل transition   ← نية الصوت للـslot\n"
            "  2 بلاش                          ← من غير أي تدخل\n"
            "  اعتمد الباقي | وريني بس الحاجات اللي محتاجة asset | وريني الـgenerated فقط | وريني الكل"
        )
    return (
        "Just reply naturally, e.g.:\n"
        "  1 ok                           accept the recommended treatment (does not allow generation)\n"
        "  2 keep the speaker             change the treatment\n"
        "  3 instead of generated make it a motion graphic\n"
        "  4 use local B-roll             without footage the fallback applies\n"
        "  5 choose option B | 5 text: <word>   text behind the speaker\n"
        "  3 generate                     explicit permission to generate THIS slot only\n"
        "  4 no sound | 4 sound accent | 4 sound accent instead of transition   sound intent for the slot\n"
        "  2 skip                         no treatment\n"
        "  approve the rest | show only what needs an asset | show only the generated | show all"
    )


# --------------------------------------------------------------------------
# The conversation
# --------------------------------------------------------------------------


class EditPlanChat:
    def __init__(self, edit_dir: Path, *, lang: str = "ar") -> None:
        self.edit_dir = Path(edit_dir)
        self.review_dir = self.edit_dir / "review"
        self.lang = lang

    def _plan(self) -> EditPlan:
        plan = ep.load_plan(self.review_dir)
        if plan is None:
            plan = open_edit_plan(self.edit_dir)
        return _refreshed(plan, self.edit_dir)

    def _save(self, plan: EditPlan) -> None:
        ep.save_plan(plan, self.review_dir)

    def reply(self, message: str | None = None) -> ChatReply:
        entries = cmd.split_entries(message or "")
        replies = [self._reply_one(e) for e in entries]
        if len(replies) == 1:
            return replies[0]
        notes = [r.note for r in replies if r.note]
        body = replies[-1].body
        return ChatReply(
            text="\n\n".join([*notes, body] if body else notes), changed=any(r.changed for r in replies),
            note="\n\n".join(notes), body=body,
        )

    # ---- one entry ---------------------------------------------------------------
    def _reply_one(self, message: str) -> ChatReply:
        plan = self._plan()
        command = cmd.parse(message)
        ar = self.lang == "ar"
        if command.kind == cmd.KIND_UNKNOWN:
            return self._show(plan, cmd.Command(cmd.KIND_SHOW, view=cmd.VIEW_PENDING), prefix=(
                "ما فهمتش الطلب.\n" if ar else "I did not get that.\n") + format_help(self.lang))
        if command.kind == cmd.KIND_HELP:
            return self._show(plan, cmd.Command(cmd.KIND_SHOW, view=cmd.VIEW_PENDING), prefix=format_help(self.lang))
        if command.kind == cmd.KIND_SHOW:
            bad = self._bad_numbers(plan, command.numbers)
            return self._show(plan, command, prefix=bad)
        if command.kind == cmd.KIND_APPROVE_REST:
            return self._approve_rest(plan)

        targets = list(command.numbers) or self._default_target(plan)
        if not targets:
            return self._show(plan, cmd.Command(cmd.KIND_SHOW, view=cmd.VIEW_PENDING), prefix=(
                "أنهي رقم؟ مثلًا: 1 موافق" if ar else "Which number? e.g. `1 ok`"))
        bad = self._bad_numbers(plan, tuple(targets))
        if bad:
            return self._show(plan, cmd.Command(cmd.KIND_SHOW, view=cmd.VIEW_PENDING), prefix=bad)
        notes: list[str] = []
        for n in targets:
            slot = plan.slot(n)
            assert slot is not None
            notes.append(self._apply(plan, slot, command))
        plan.last_slot = targets[-1]
        self._save(plan)
        return self._show(plan, cmd.Command(cmd.KIND_SHOW, view=cmd.VIEW_PENDING), prefix="\n".join(notes), changed=True)

    def _default_target(self, plan: EditPlan) -> list[int]:
        if plan.last_slot and plan.slot(plan.last_slot):
            return [plan.last_slot]
        pending = plan.pending()
        return [pending[0].number] if len(pending) == 1 else []

    def _bad_numbers(self, plan: EditPlan, numbers: tuple[int, ...]) -> str:
        top = max((s.number for s in plan.slots), default=0)
        bad = [n for n in numbers if plan.slot(n) is None]
        if not bad:
            return ""
        joined = ", ".join(map(str, bad))
        return (f"مفيش قرار رقم {joined} (الأرقام من 1 لـ {top})." if self.lang == "ar"
                else f"There is no decision {joined} (numbers run 1-{top}).")

    def _apply(self, plan: EditPlan, slot: EditPlanSlot, command: cmd.Command) -> str:
        ar = self.lang == "ar"
        n = slot.number
        kind = command.kind
        if kind == cmd.KIND_APPROVE:
            if slot.needs_text:
                return self._needs_text(slot)
            ep.approve_slot(slot)
            return self._confirm(slot)
        if kind == cmd.KIND_SET_TREATMENT:
            ep.set_treatment(slot, command.treatment or "")
            direction.retarget(plan, slot, self._registry())
            if slot.needs_text:
                return self._needs_text(slot, changed=True)
            return self._confirm(slot)
        if kind == cmd.KIND_SET_SOUND:
            if not slot.directed:
                return (f"[{n}] الـslot ده مالوش تحكم في الصوت." if ar else f"[{n}] this slot has no sound control.")
            direction.set_slot_sound(plan, slot, command.sound or "none", self._registry())
            return self._sound_confirm(slot)
        if kind == cmd.KIND_REJECT:
            ep.reject_slot(slot)
            direction.retarget(plan, slot, self._registry())
            return (f"✔ [{n}] اتلغى — من غير أي تدخل (المتحدث زي ما هو)." if ar
                    else f"✔ [{n}] no treatment — the plain speaker shot stays.")
        if kind == cmd.KIND_CHOOSE_OPTION:
            idx = command.option if command.option is not None else -1
            if slot.treatment not in ep.TEXT_TREATMENTS or not 0 <= idx < len(slot.text_options):
                return (f"[{n}] مفيش خيار بالشكل ده للـslot ده." if ar else f"[{n}] no such option for this slot.")
            ep.set_text(slot, slot.text_options[idx])
            return self._confirm(slot)
        if kind == cmd.KIND_SET_TEXT:
            text = command.text or ""
            ep.set_text(slot, text)
            note = ""
            if not ep.occurs_in(text, slot.spoken_context):
                note = (" (الكلمة دي مش من كلام الجملة — اتعتمدت لأنك كتبتها بنفسك)" if ar
                        else " (this is not from the spoken sentence — accepted because you typed it)")
            if slot.treatment not in ep.TEXT_TREATMENTS:
                return (f"[{n}] الـtreatment الحالي ({label(slot.treatment, self.lang)}) مش بيعرض نص." if ar
                        else f"[{n}] the current treatment ({label(slot.treatment, self.lang)}) shows no text.")
            return self._confirm(slot) + note
        if kind == cmd.KIND_APPROVE_GENERATION:
            try:
                ep.approve_generation(slot)
            except ValueError:
                return (f"[{n}] الـslot ده مش B-roll مولَّد، فمفيش توليد أوافق عليه." if ar
                        else f"[{n}] this slot is not generated B-roll, so there is nothing to allow.")
            return (f"✔ [{n}] التوليد معتمد للـslot ده بس. لسه ما اتولّدش حاجة." if ar
                    else f"✔ [{n}] generation approved for this slot only. Nothing has been generated.")
        return f"[{n}] ?"

    def _registry(self) -> SfxRegistry:
        return load_registry(default_sfx_dir())

    def _sound_confirm(self, slot: EditPlanSlot) -> str:
        ar = self.lang == "ar"
        n, i = slot.number, 1 if ar else 0
        intent = _INTENT_LABEL.get(slot.sound_intent, (slot.sound_intent,) * 2)[i]
        line = f"✔ [{n}] " + (f"الصوت: {intent}" if ar else f"Sound: {intent}")
        if slot.sound_intent != "none":
            line += f" — SFX: {slot.sfx_availability or 'none'}"
        return line

    def _needs_text(self, slot: EditPlanSlot, *, changed: bool = False) -> str:
        ar = self.lang == "ar"
        head = (f"[{slot.number}] {label(slot.treatment, self.lang)}: محتاج أعتمد النص بالظبط الأول، وبعدها الـslot يتعتمد."
                if ar else f"[{slot.number}] {label(slot.treatment, self.lang)}: I need the exact text approved first.")
        return head + "\n" + "\n".join(_text_lines(slot, ar))

    def _confirm(self, slot: EditPlanSlot) -> str:
        ar = self.lang == "ar"
        n, t = slot.number, label(slot.treatment, self.lang)
        line = f"✔ [{n}] {t}" + (f" — «{slot.text}»" if slot.text else "")
        if slot.treatment == "generated_broll" and not slot.generation_approved:
            line += ("\n   التوليد لسه مش معتمد — قول «" + f"{n} ولّد" + "» لو عايز تسمح بيه. لحد وقتها الـfallback هو اللي بيتطبق: "
                     + label(slot.fallback or "stay_on_speaker", "ar")) if ar else (
                "\n   Generation is NOT approved yet — say “" + f"{n} generate" + "” to allow it. Until then the fallback applies: "
                + label(slot.fallback or "stay_on_speaker", "en"))
        if slot.status is SlotStatus.ASSET_REQUIRED:
            line += ("\n   محتاج footage بتاعك؛ لو مفيش، الـfallback: " + label(slot.fallback or "stay_on_speaker", "ar")) if ar else (
                "\n   Needs your footage; without it the fallback applies: " + label(slot.fallback or "stay_on_speaker", "en"))
        return line

    # ---- bulk ----------------------------------------------------------------------
    def _approve_rest(self, plan: EditPlan) -> ChatReply:
        ar = self.lang == "ar"
        done, blocked, notes = [], [], []
        for slot in plan.pending():
            if slot.needs_text:
                blocked.append(slot.number)
                continue
            ep.approve_slot(slot)
            done.append(slot.number)
            if slot.treatment == "generated_broll" and not slot.generation_approved:
                notes.append(self._confirm(slot))
        self._save(plan)
        lines = []
        if done:
            lines.append(f"✔ اتعتمد: {', '.join(map(str, done))}" if ar else f"✔ Approved: {', '.join(map(str, done))}")
        if blocked:
            lines.append(f"محتاجين النص الأول: {', '.join(map(str, blocked))}" if ar
                          else f"Need the exact text first: {', '.join(map(str, blocked))}")
        if not done and not blocked:
            lines.append("مفيش قرارات مفتوحة." if ar else "No open decisions.")
        return self._show(plan, cmd.Command(cmd.KIND_SHOW, view=cmd.VIEW_PENDING), prefix="\n".join([*lines, *notes]), changed=bool(done))

    # ---- views ---------------------------------------------------------------------
    def _select(self, plan: EditPlan, command: cmd.Command) -> list[EditPlanSlot]:
        if command.view == cmd.VIEW_ALL:
            return list(plan.slots)
        if command.view == cmd.VIEW_NUMBERS:
            return [s for s in plan.slots if s.number in command.numbers]
        if command.view == cmd.VIEW_ASSET:
            return [s for s in plan.reviewable() if s.needs_asset]
        if command.view == cmd.VIEW_GENERATED:
            return [s for s in plan.reviewable() if s.needs_generation]
        return plan.pending()

    def _show(self, plan: EditPlan, command: cmd.Command, *, prefix: str = "", changed: bool = False) -> ChatReply:
        ar = self.lang == "ar"
        slots = self._select(plan, command)
        if slots:
            body = "\n\n".join(format_slot(s, self.lang) for s in slots)
        elif command.view == cmd.VIEW_PENDING:
            body = "مفيش قرارات مفتوحة دلوقتي." if ar else "No decisions are open right now."
        else:
            body = "مفيش حاجة بالشكل ده." if ar else "Nothing matches."
        footer = summary(plan, self.lang)
        text = "\n\n".join(part for part in (prefix, body, footer) if part)
        return ChatReply(text=text, changed=changed, note=prefix, body=f"{body}\n\n{footer}")


__all__ = [
    "EditPlanChat", "format_help", "format_slot", "generation_risk", "label", "open_edit_plan", "summary",
]
