"""Edit-plan review state: one reviewable decision per editorial intervention.

Workflow (generic, any project):

    EDIT PLAN -> SHOW TREATMENTS IN CHAT -> USER APPROVES / CHANGES
      -> SAVE DECISIONS -> SOURCE / GENERATE ASSETS -> QUALITY GATE -> COMPOSITION

The user therefore never discovers a B-roll or motion decision for the first
time in a finished render. Generated B-roll is ONE treatment among several and
is never the default; it also needs its own explicit approval
(`generation_approved`), separate from approving the treatment.

Slots come from the project's editorial decisions (`broll_editorial.json`),
are mapped through the EDL onto the approved transcript segments, and are
stored in `<review>/edit_plan.json`. Approving nothing here ever sets
`ready_for_final_render`; the final gate (`review.state.approve`) only refuses
while a slot is still pending.

Nothing in this module is brand- or project-specific.
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from video_edit_agent.broll.treatment import BROLL_TREATMENTS, Treatment, TreatmentDecision
from video_edit_agent.core.schemas import EDL, Transcript

PLAN_FILENAME = "edit_plan.json"
NO_TREATMENT = "no_treatment"
TREATMENTS = tuple(t.value for t in Treatment) + (NO_TREATMENT,)
TEXT_TREATMENTS = frozenset({Treatment.BEHIND_SUBJECT_TEXT.value, Treatment.KINETIC_TYPOGRAPHY.value})
_GENERATED = Treatment.GENERATED_BROLL.value
_LOCAL = Treatment.LOCAL_BROLL.value
_MIN_OVERLAP_S = 0.05
# Treatments a slot may quietly fall back to when its asset is missing or
# generation is not approved; the reviewer sees the fallback before deciding.
_DEFAULT_FALLBACK = {
    _LOCAL: Treatment.STAY_ON_SPEAKER.value,
    _GENERATED: Treatment.STAY_ON_SPEAKER.value,
}


class SlotStatus(str, Enum):
    PENDING_REVIEW = "pending_review"  # nothing decided yet; blocks final approval
    APPROVED = "approved"  # the user accepted the recommended treatment
    CHANGED = "changed"  # the user picked a different treatment
    REJECTED = "rejected"  # the user does not want an intervention here
    ASSET_REQUIRED = "asset_required"  # treatment decided, external footage still missing (fallback applies)
    GENERATION_APPROVED = "generation_approved"  # the user explicitly allowed AI generation for this slot


class EditPlanSlot(BaseModel):
    number: int = 0  # 1-based, what the user types; 0 = settled elsewhere / no special treatment
    timeline_start: float
    timeline_end: float
    segments: list[int] = Field(default_factory=list)  # 1-based transcript segment numbers
    spoken_context: str = ""  # approved transcript text, refreshed on load
    recommended: str  # the planner's treatment (never rewritten)
    treatment: str  # the current treatment
    reason: str = ""
    visual_concept: str | None = None
    fallback: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    status: SlotStatus = SlotStatus.PENDING_REVIEW
    settled_reason: str | None = None  # why this slot is not asked about
    draft_prompt: str | None = None  # generated_broll: what a provider would receive (shown, not sent)
    generation_approved: bool = False
    text_options: list[str] = Field(default_factory=list)
    text: str | None = None  # the approved on-screen text (text treatments)

    @property
    def duration(self) -> float:
        return round(self.timeline_end - self.timeline_start, 2)

    @property
    def settled(self) -> bool:
        return self.number == 0

    @property
    def needs_asset(self) -> bool:
        return self.treatment in {t.value for t in BROLL_TREATMENTS}

    @property
    def needs_generation(self) -> bool:
        return self.treatment == _GENERATED

    @property
    def needs_text(self) -> bool:
        return self.treatment in TEXT_TREATMENTS and not self.text

    @property
    def blocking(self) -> bool:
        return not self.settled and self.status is SlotStatus.PENDING_REVIEW

    @property
    def effective_treatment(self) -> str:
        """What the composition would actually do right now: a slot whose
        footage is missing or whose generation is not approved uses its fallback."""
        if self.status is SlotStatus.REJECTED or self.treatment == NO_TREATMENT:
            return Treatment.STAY_ON_SPEAKER.value
        if self.treatment == _GENERATED and not self.generation_approved:
            return self.fallback or _DEFAULT_FALLBACK[_GENERATED]
        if self.treatment == _LOCAL and self.status is SlotStatus.ASSET_REQUIRED:
            return self.fallback or _DEFAULT_FALLBACK[_LOCAL]
        return self.treatment


class EditPlan(BaseModel):
    slots: list[EditPlanSlot] = Field(default_factory=list)
    last_slot: int | None = None  # the slot the user was last looking at ("ok" alone means this one)

    def slot(self, number: int) -> EditPlanSlot | None:
        return next((s for s in self.slots if s.number == number and number > 0), None)

    def reviewable(self) -> list[EditPlanSlot]:
        return [s for s in self.slots if not s.settled]

    def pending(self) -> list[EditPlanSlot]:
        return [s for s in self.slots if s.blocking]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def plan_path(review_dir: Path) -> Path:
    return review_dir / PLAN_FILENAME


def load_plan(review_dir: Path) -> EditPlan | None:
    path = plan_path(review_dir)
    if not path.exists():
        return None
    return EditPlan.model_validate_json(path.read_text(encoding="utf-8"))


def save_plan(plan: EditPlan, review_dir: Path) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    path = plan_path(review_dir)
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return path


def open_pending_slots(review_dir: Path) -> list[EditPlanSlot]:
    """Slots that still block final approval (empty when no edit plan exists)."""
    plan = load_plan(review_dir)
    return plan.pending() if plan else []


# --------------------------------------------------------------------------
# Building the plan
# --------------------------------------------------------------------------

_MARKS = re.compile("[ً-ْـ]")


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKS.sub("", text).translate(str.maketrans("أإآ", "ااا")).lower()).strip()


def occurs_in(option: str, text: str) -> bool:
    """True when `option` is literally part of `text` (diacritics/case/spacing ignored):
    a text treatment may only show words the speaker actually said."""
    return bool(option.strip()) and _fold(option) in _fold(text)


_STOP = {"زي", "ما", "اللي", "ده", "دي", "دا", "في", "من", "على", "علي", "إن", "ان", "هو", "هي", "the", "a", "of"}
_LATIN = re.compile(r"[A-Za-z]{3,}")
_TOKEN = re.compile(r"[^\s،,.؟?!:؛;…]+")


def keyword_options(text: str, limit: int = 3) -> list[str]:
    """Short candidate keywords taken verbatim from `text`: each Latin
    (code-switched) term with the word right after it when that word is an
    ordinary content word, e.g. `value حقيقية`. Dialect is never translated."""
    tokens = _TOKEN.findall(text)
    found: list[str] = []
    for i, tok in enumerate(tokens):
        if not _LATIN.search(tok):
            continue
        span = tok
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if nxt and nxt not in _STOP and not _LATIN.fullmatch(nxt.strip("ـ")) and len(nxt) >= 3:
            span = f"{tok} {nxt}"
        span = span.strip("ـ")
        if span and span not in found:
            found.append(span)
    return found[:limit]


def _segments_for(start: float, end: float, edl: EDL, transcript: Transcript) -> list[int]:
    """1-based numbers of the transcript segments spoken in timeline [start, end], via the EDL."""
    number = {seg.id: n for n, seg in enumerate(transcript.segments, start=1)}
    found: list[int] = []
    for clip in edl.clips:
        if min(end, clip.timeline_out) - max(start, clip.timeline_in) > _MIN_OVERLAP_S:
            found.extend(number[ref] for ref in clip.caption_refs if ref in number)
    return sorted(dict.fromkeys(found))


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_edit_plan(
    transcript: Transcript,
    edl: EDL,
    decisions: list[TreatmentDecision],
    *,
    settled: dict[str, str] | None = None,
    hook_window: tuple[float, float] | None = None,
    hook_approved: bool = False,
) -> EditPlan:
    """The reviewable plan for `decisions`, in time order.

    `transcript` must be the APPROVED (corrected) transcript so the reviewer sees
    what the speaker actually said. A slot is *settled* (never asked, number 0)
    when its treatment is `stay_on_speaker` (no special treatment), when its
    treatment appears in `settled` ({treatment: why}, e.g. approved in an earlier
    review step), or when it is the opening kinetic title covered by an approved
    hook. Everything else is numbered 1..N and starts as `pending_review`."""
    settled = settled or {}
    slots: list[EditPlanSlot] = []
    for d in sorted(decisions, key=lambda d: d.timeline_start):
        segs = _segments_for(d.timeline_start, d.timeline_end, edl, transcript)
        context = " ".join(transcript.segments[n - 1].text.strip() for n in segs)
        treatment = d.treatment.value
        settled_reason: str | None = None
        if treatment == Treatment.STAY_ON_SPEAKER.value:
            settled_reason = "no special treatment"
        elif treatment in settled:
            settled_reason = settled[treatment]
        elif (
            hook_approved and hook_window and treatment == Treatment.KINETIC_TYPOGRAPHY.value
            and _overlap(d.timeline_start, d.timeline_end, *hook_window) > 0
        ):
            settled_reason = "approved in the hook review"
        options = [o for o in d.text_options if occurs_in(o, context)]
        if treatment in TEXT_TREATMENTS and not options and not d.text_options:
            options = keyword_options(context)
        slots.append(EditPlanSlot(
            timeline_start=round(d.timeline_start, 3), timeline_end=round(d.timeline_end, 3),
            segments=segs, spoken_context=context, recommended=treatment, treatment=treatment,
            reason=d.reason, visual_concept=d.visual_concept,
            fallback=(d.fallback.value if d.fallback else _DEFAULT_FALLBACK.get(treatment)),
            alternatives=[a.value for a in d.alternatives],
            status=SlotStatus.APPROVED if settled_reason else SlotStatus.PENDING_REVIEW,
            settled_reason=settled_reason,
            draft_prompt=d.prompt, text_options=options,
        ))
    number = 0
    for slot in slots:
        if slot.settled_reason is None:
            number += 1
            slot.number = number
    return EditPlan(slots=slots)


def refresh_context(plan: EditPlan, transcript: Transcript, edl: EDL) -> EditPlan:
    """Re-reads the spoken context from the current approved transcript (a
    correction made after the plan was built must show up in the review)."""
    for slot in plan.slots:
        segs = _segments_for(slot.timeline_start, slot.timeline_end, edl, transcript)
        slot.segments = segs
        slot.spoken_context = " ".join(transcript.segments[n - 1].text.strip() for n in segs)
    return plan


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


def _after_choice(slot: EditPlanSlot, status_if_ok: SlotStatus) -> SlotStatus:
    """The status a slot gets once its treatment is chosen, given what it still lacks."""
    if slot.needs_text:
        return SlotStatus.PENDING_REVIEW  # exact on-screen text must be approved first
    if slot.treatment == _LOCAL:
        return SlotStatus.ASSET_REQUIRED
    if slot.generation_approved and slot.treatment == _GENERATED:
        return SlotStatus.GENERATION_APPROVED
    return status_if_ok


def approve_slot(slot: EditPlanSlot) -> EditPlanSlot:
    """The user accepts the CURRENT treatment. Generation is NOT thereby approved.
    A slot the user turned down stays rejected."""
    if slot.treatment == NO_TREATMENT or slot.status is SlotStatus.REJECTED:
        slot.treatment, slot.status = NO_TREATMENT, SlotStatus.REJECTED
        return slot
    slot.status = _after_choice(slot, SlotStatus.APPROVED if slot.treatment == slot.recommended else SlotStatus.CHANGED)
    return slot


def set_treatment(slot: EditPlanSlot, treatment: str) -> EditPlanSlot:
    """The user picked `treatment`. Picking it is a decision (status `changed`,
    or `approved` when it equals the recommendation)."""
    if treatment not in TREATMENTS:
        raise ValueError(f"unknown treatment '{treatment}'")
    if treatment == NO_TREATMENT:
        slot.treatment, slot.status, slot.generation_approved = NO_TREATMENT, SlotStatus.REJECTED, False
        return slot
    if treatment != _GENERATED:
        slot.generation_approved = False  # generation was for the generated treatment only
    slot.treatment = treatment
    slot.status = _after_choice(slot, SlotStatus.APPROVED if treatment == slot.recommended else SlotStatus.CHANGED)
    return slot


def reject_slot(slot: EditPlanSlot) -> EditPlanSlot:
    slot.treatment, slot.status, slot.generation_approved = NO_TREATMENT, SlotStatus.REJECTED, False
    return slot


def approve_generation(slot: EditPlanSlot) -> EditPlanSlot:
    """Explicit permission to generate this slot's B-roll. Only valid while the
    slot's treatment is generated B-roll; nothing is generated here."""
    if slot.treatment != _GENERATED:
        raise ValueError("this slot is not generated B-roll")
    slot.generation_approved = True
    slot.status = SlotStatus.GENERATION_APPROVED
    return slot


def set_text(slot: EditPlanSlot, text: str) -> EditPlanSlot:
    """The exact on-screen text the user approved (verbatim). Approves the slot's
    treatment along with it."""
    slot.text = text.strip()
    return approve_slot(slot) if slot.treatment in TEXT_TREATMENTS else slot


# --------------------------------------------------------------------------
# Feeding the decisions back to the planner
# --------------------------------------------------------------------------


def effective_decisions(review_dir: Path, decisions: list[TreatmentDecision]) -> list[TreatmentDecision]:
    """`decisions` with the user's edit-plan choices applied (no plan on disk -> unchanged).

    A generated-B-roll slot the user has not explicitly approved for generation
    falls back to its fallback treatment, so approving a plan can never
    silently trigger paid generation."""
    plan = load_plan(review_dir)
    if plan is None:
        return decisions
    out: list[TreatmentDecision] = []
    for d in decisions:
        best = max(plan.slots, key=lambda s: _overlap(d.timeline_start, d.timeline_end, s.timeline_start, s.timeline_end), default=None)
        if best is None or _overlap(d.timeline_start, d.timeline_end, best.timeline_start, best.timeline_end) <= 0:
            out.append(d)
            continue
        out.append(d.model_copy(update={"treatment": Treatment(best.effective_treatment)}))
    return out
