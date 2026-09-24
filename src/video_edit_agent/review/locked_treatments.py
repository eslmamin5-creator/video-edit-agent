"""Locked Treatment Specs (Phase 1.5.3): the approved visual recipe of a major treatment.

Approval used to preserve only status, text and timing; the final renderer then re-derived the
look (font size, placement, camera, occlusion) from generic defaults, so the render drifted from
the micro-preview the user actually approved. A `LockedTreatment` freezes the recipe the approved
preview was rendered with. The final render consumes it directly; discovery only fills gaps no lock
owns, and a lock that cannot be honoured fails loudly (`LockedTreatmentFidelityError`) instead of
silently degrading into a different treatment.

Nothing here is client-specific: the recipe values live in the project's own lock file
(`<review dir>/locked_treatments.json`), never in this module.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

LOCK_FILENAME = "locked_treatments.json"
RENDERER_VERSION = "1.5.3"
SCHEMA_VERSION = 1

TREATMENT_HEADLINE = "lower_subject_semantic"
TREATMENT_BEHIND = "behind_subject_text"
LOCKABLE_TREATMENTS = (TREATMENT_HEADLINE, TREATMENT_BEHIND)

APPROVED = "approved"
PENDING = "pending_review"
REJECTED = "rejected"
INVALIDATED = "invalidated"

BEHIND_LAYER_ORDER = ("base_video", "text_graphic", "subject_cutout", "captions")
HEADLINE_LAYER_ORDER = ("base_video", "captions", "headline")

FIDELITY_ERROR_CODE = "LOCKED_TREATMENT_FIDELITY_ERROR"


class LockedTreatmentFidelityError(RuntimeError):
    """A locked treatment cannot be reproduced faithfully; the render must not fall back silently."""

    def __init__(self, treatment_id: str, problems: Sequence[str]):
        self.treatment_id, self.problems = treatment_id, list(problems)
        super().__init__(f"{FIDELITY_ERROR_CODE}: {treatment_id}: " + "; ".join(self.problems))


# --------------------------------------------------------------------------
# Subtype recipes
# --------------------------------------------------------------------------


class HeadlineRecipe(BaseModel):
    """`lower_subject_semantic`: a headline above the speaker + a lower_subject camera move."""

    font_family: str = ""
    font_px: int
    font_weight: str = "bold"
    y_px: float  # top of the headline line box (ASS \an8), frame pixels
    alignment: str = "top_center"
    text_position: str = "top"
    slide_px: float = 14.0
    settle_from: float = 0.94
    fade_in: tuple[float, float]  # timeline seconds
    fade_out: tuple[float, float]
    caption_role: str = "reduced"
    caption_reduction_window: tuple[float, float]  # captions are reduced only inside this window
    caption_reduction_scale: float | None = None  # None = the standard REDUCED behaviour
    # camera: base -> lower_subject (t0..t1) -> hold (t1..t2) -> reset_to_base (t2..t3)
    camera_owner: str = "composition"
    camera_state: str = "lower_subject"
    changes: tuple[float, float, float, float]
    geometry: dict[str, Any]  # LowerSubjectGeometry.model_dump()
    zoom: float
    anchor_x: float
    anchor_y: float
    face_box: tuple[float, float, float, float] | None = None
    head_top: float | None = None
    text_bbox: tuple[float, float, float, float] | None = None  # normalized (x0, y0, x1, y1) of the headline ink
    layer_order: tuple[str, ...] = HEADLINE_LAYER_ORDER

    @property
    def entry_duration(self) -> float:
        return round(self.changes[1] - self.changes[0], 3)

    @property
    def hold_duration(self) -> float:
        return round(self.changes[2] - self.changes[1], 3)

    @property
    def reset_duration(self) -> float:
        return round(self.changes[3] - self.changes[2], 3)


class BehindSubjectRecipe(BaseModel):
    """`behind_subject_text`: keyword text drawn behind the speaker via a refined subject cutout."""

    font_family: str = ""
    font_px: int
    font_weight: str = "bold"
    lines: tuple[str, ...]
    center_x: float
    center_y: float
    bounds: tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels
    color: str
    opacity: float = 1.0
    outline: dict[str, Any] | None = None
    shadow: str = ""
    enter_sec: float = 0.0
    exit_sec: float = 0.0
    alignment: str = "center"
    caption_role: str = "normal"
    mask_mode: str = "subject_cutout_refined"
    layer_order: tuple[str, ...] = BEHIND_LAYER_ORDER
    occlusion: dict[str, Any] = Field(default_factory=dict)  # the measured visible/occluded relationship
    phrase_timing: dict[str, Any] = Field(default_factory=dict)
    legibility: dict[str, Any] = Field(default_factory=dict)
    camera_owner: str = "base"
    camera_state: str = "base"
    zoom: float = 1.0
    anchor_x: float = 0.5
    anchor_y: float = 0.3
    slot: int | None = None


class LockedTreatment(BaseModel):
    treatment_id: str
    treatment_type: str
    approval_status: str = PENDING
    text: str
    source_segment_ids: list[int] = Field(default_factory=list)
    start: float
    end: float
    headline: HeadlineRecipe | None = None
    behind: BehindSubjectRecipe | None = None
    brand_tokens: dict[str, Any] = Field(default_factory=dict)
    renderer_version: str = RENDERER_VERSION
    locked_from_preview: bool = True
    locked_preview_path: str = ""
    locked_preview_sha256: str = ""
    preview_timeline_start: float = 0.0  # timeline second at which the approved preview clip starts
    locked_at: str = ""
    revision: int = 1
    draft_of: str | None = None  # the approved lock a pending draft would replace
    notes: list[str] = Field(default_factory=list)

    @property
    def recipe(self) -> HeadlineRecipe | BehindSubjectRecipe | None:
        return self.headline if self.treatment_type == TREATMENT_HEADLINE else self.behind

    @property
    def is_active(self) -> bool:
        return self.approval_status == APPROVED

    def window(self) -> tuple[float, float]:
        return self.start, self.end


class LockStore(BaseModel):
    schema_version: int = SCHEMA_VERSION
    locks: list[LockedTreatment] = Field(default_factory=list)

    def get(self, treatment_id: str) -> LockedTreatment | None:
        return next((lk for lk in self.locks if lk.treatment_id == treatment_id), None)

    def active(self, treatment_type: str | None = None) -> list[LockedTreatment]:
        """Approved locks only: pending drafts and rejected/invalidated locks never render."""
        return [lk for lk in self.locks if lk.is_active and (treatment_type is None or lk.treatment_type == treatment_type)]

    def owned_windows(self) -> list[tuple[float, float]]:
        return [lk.window() for lk in self.active()]

    def owns(self, start: float, end: float) -> bool:
        return any(s < end and start < e for s, e in self.owned_windows())


# --------------------------------------------------------------------------
# Persistence (canonical location, deterministic bytes)
# --------------------------------------------------------------------------


def lock_path(review_dir: Path) -> Path:
    return Path(review_dir) / LOCK_FILENAME


def serialize(store: LockStore) -> str:
    ordered = LockStore(locks=sorted(store.locks, key=lambda lk: (lk.start, lk.treatment_id)))
    return json.dumps(ordered.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def save_locks(review_dir: Path, store: LockStore) -> Path:
    path = lock_path(review_dir)
    text = serialize(store)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return path  # idempotent: an unchanged store is never rewritten
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_locks(review_dir: Path) -> LockStore:
    path = lock_path(review_dir)
    if not path.exists():
        return LockStore()
    return LockStore.model_validate_json(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Lifecycle: approve -> lock, edit -> pending_review, reject -> invalidated
# --------------------------------------------------------------------------


def approve_from_preview(
    store: LockStore, lock: LockedTreatment, preview_path: Path | str, *, now: str | None = None,
) -> LockedTreatment:
    """The user approved the micro-preview rendered from `lock`'s recipe: freeze it. Approval REPLACES any earlier
    lock (or pending draft) of the same treatment id, so the approved state and the recipe cannot drift apart."""
    stamped = lock.model_copy(update={
        "approval_status": APPROVED, "locked_from_preview": True, "locked_preview_path": str(preview_path),
        "locked_preview_sha256": _sha256(Path(preview_path)), "locked_at": now or _now(), "draft_of": None,
        "renderer_version": RENDERER_VERSION,
    })
    previous = store.get(lock.treatment_id)
    if previous is not None:
        ignore = {"revision", "locked_at", "notes"}
        if previous.is_active and previous.model_dump(exclude=ignore) == stamped.model_dump(exclude=ignore):
            return previous  # re-approving an identical recipe changes nothing (idempotent)
        stamped.revision = previous.revision + 1
    store.locks = [lk for lk in store.locks if lk.treatment_id != lock.treatment_id] + [stamped]
    return stamped


def edit_lock(store: LockStore, treatment_id: str, mutate: Callable[[LockedTreatment], None]) -> LockedTreatment:
    """A revision of a locked treatment. The approval is invalidated immediately: the edited draft is
    `pending_review` and cannot render until a fresh preview of it is approved."""
    current = store.get(treatment_id)
    if current is None:
        raise KeyError(f"no locked treatment {treatment_id!r}")
    draft = current.model_copy(deep=True)
    mutate(draft)
    draft.approval_status, draft.draft_of = PENDING, current.locked_preview_sha256 or current.locked_preview_path or "lock"
    draft.locked_at = ""
    draft.notes = [*draft.notes, f"revision of rev {current.revision}"]
    store.locks = [lk for lk in store.locks if lk.treatment_id != treatment_id] + [draft]
    return draft


def reject(store: LockStore, treatment_id: str) -> LockedTreatment:
    """A rejected preview invalidates the lock: nothing of it may render."""
    current = store.get(treatment_id)
    if current is None:
        raise KeyError(f"no locked treatment {treatment_id!r}")
    updated = current.model_copy(update={"approval_status": REJECTED, "locked_at": ""})
    store.locks = [lk for lk in store.locks if lk.treatment_id != treatment_id] + [updated]
    return updated


# --------------------------------------------------------------------------
# Revision requests ("bigger headline", "lower it", ...): a draft recipe change, never raw JSON
# --------------------------------------------------------------------------

Operation = Literal["scale_font", "shift_y", "to_behind_subject", "scale_zoom", "remove_motion", "extend_hold", "approve"]

_REVISION_RULES: tuple[tuple[Operation, float, re.Pattern[str]], ...] = (
    ("approve", 0.0, re.compile(r"اعتمد|اعتمدها|approve", re.IGNORECASE)),
    ("remove_motion", 0.0, re.compile(r"شيل\s*الحرك|remove\s+(the\s+)?(motion|move)|no\s+motion", re.IGNORECASE)),
    ("to_behind_subject", 0.0, re.compile(r"behind[\s-]*subject|ورا\s*(ال)?(شخص|متحدث)", re.IGNORECASE)),
    ("scale_zoom", 0.5, re.compile(r"خفف\s*(ال)?\s*zoom|less\s+zoom|reduce\s+(the\s+)?zoom", re.IGNORECASE)),
    ("extend_hold", 1.0, re.compile(r"النص\s*أطول|خلي\s*النص\s*أطول|longer\s+(text|hold)|hold\s+longer", re.IGNORECASE)),
    ("scale_font", 1.2, re.compile(r"كبر|أكبر|bigger|larger|increase\s+(the\s+)?(size|font)", re.IGNORECASE)),
    ("scale_font", 0.85, re.compile(r"صغر|أصغر|smaller|decrease\s+(the\s+)?(size|font)", re.IGNORECASE)),
    ("shift_y", 0.03, re.compile(r"نزل|انزل|lower\s+it|move\s+down", re.IGNORECASE)),
    ("shift_y", -0.03, re.compile(r"طلع|فوق|move\s+up|raise", re.IGNORECASE)),
)


def parse_revision(request: str) -> tuple[Operation, float] | None:
    """Maps a short user request (Arabic dialect or English) to one recipe operation and its amount."""
    for op, amount, pattern in _REVISION_RULES:
        if pattern.search(request):
            return op, amount
    return None


def apply_revision(store: LockStore, treatment_id: str, request: str, *, frame_height: int = 1920) -> LockedTreatment:
    """Applies one revision request to the treatment's draft recipe. The result is `pending_review`: render a new
    micro-preview from it, then `approve_from_preview` (or `reject`)."""
    parsed = parse_revision(request)
    if parsed is None:
        raise ValueError(f"unrecognised revision request: {request!r}")
    op, amount = parsed
    if op == "approve":
        raise ValueError("approval needs the rendered preview: use approve_from_preview")

    def mutate(lk: LockedTreatment) -> None:
        h, b = lk.headline, lk.behind
        if op == "scale_font":
            if h is not None:
                h.font_px = round(h.font_px * amount)
            if b is not None:
                b.font_px = round(b.font_px * amount)
        elif op == "shift_y":
            if h is not None:
                h.y_px = round(h.y_px + amount * frame_height, 3)
            if b is not None:
                dy = amount * frame_height
                b.center_y = round(b.center_y + dy, 1)
                b.bounds = (b.bounds[0], b.bounds[1] + dy, b.bounds[2], b.bounds[3] + dy)
        elif op == "scale_zoom" and h is not None:
            h.zoom = round(1.0 + (h.zoom - 1.0) * amount, 4)
            h.geometry = {**h.geometry, "zoom": h.zoom}
        elif op == "remove_motion" and h is not None:
            h.zoom, h.anchor_x = 1.0, 0.5
            h.geometry = {**h.geometry, "zoom": 1.0, "anchor_x": 0.5}
            h.camera_state = "base"
        elif op == "extend_hold":
            t0, t1, t2, t3 = h.changes if h is not None else (0, 0, 0, 0)
            if h is not None:
                h.changes = (t0, t1, round(t2 + amount, 3), round(t3 + amount, 3))
                h.fade_out = (round(h.fade_out[0] + amount, 3), round(h.fade_out[1] + amount, 3))
                h.caption_reduction_window = (h.caption_reduction_window[0], round(h.caption_reduction_window[1] + amount, 3))
                lk.end = round(lk.end + amount, 3)
            elif b is not None:
                b.exit_sec = round(b.exit_sec + amount, 3)
                lk.end = round(lk.end + amount, 3)
        elif op == "to_behind_subject":
            lk.notes.append("requested treatment change: behind_subject_text (needs a fresh placement preview)")

    return edit_lock(store, treatment_id, mutate)


# --------------------------------------------------------------------------
# Validation + consumption by the final renderer
# --------------------------------------------------------------------------


def validate_lock(lock: LockedTreatment, *, total_duration: float | None = None, frame_size: tuple[int, int] | None = None) -> list[str]:
    """Hard incompatibilities that make a lock impossible to reproduce. Empty list = renderable."""
    problems: list[str] = []
    if lock.treatment_type not in LOCKABLE_TREATMENTS:
        problems.append(f"unsupported treatment type {lock.treatment_type!r}")
    if not lock.text.strip():
        problems.append("locked text is empty")
    if lock.end <= lock.start:
        problems.append(f"locked window {lock.start}..{lock.end} is empty")
    if total_duration is not None and lock.end > total_duration + 0.05:
        problems.append(f"locked window ends at {lock.end} after the timeline ({total_duration:.2f}s)")
    r = lock.recipe
    if r is None:
        problems.append("locked recipe is missing for its treatment type")
        return problems
    if r.font_px <= 0:
        problems.append("locked font size is not positive")
    if isinstance(r, HeadlineRecipe):
        if not (r.changes[0] < r.changes[1] <= r.changes[2] < r.changes[3]):
            problems.append(f"camera timing {r.changes} is not ordered entry<hold<reset")
        if r.geometry.get("status", "ok") != "ok":
            problems.append("locked lower_subject geometry was rejected")
        if r.zoom < 1.0:
            problems.append("locked zoom is below 1.0")
        if frame_size is not None and not 0 <= r.y_px < frame_size[1]:
            problems.append(f"headline y {r.y_px} is outside the frame")
    else:
        if not r.lines or not any(ln.strip() for ln in r.lines):
            problems.append("locked behind-subject lines are empty")
        order = list(r.layer_order)
        if "subject_cutout" not in order or "text_graphic" not in order or order.index("subject_cutout") < order.index("text_graphic"):
            problems.append("layer order must draw the subject cutout above the text")
        x0, y0, x1, y1 = r.bounds
        if x1 <= x0 or y1 <= y0:
            problems.append("locked text bounds are empty")
        if frame_size is not None and (x0 < 0 or y0 < 0 or x1 > frame_size[0] or y1 > frame_size[1]):
            problems.append("locked text bounds leave the frame")
    if lock.locked_preview_path and lock.locked_preview_sha256 and not Path(lock.locked_preview_path).exists():
        problems.append(f"approved preview {lock.locked_preview_path} is missing (provenance broken)")
    return problems


def require_renderable(lock: LockedTreatment, **kwargs: Any) -> LockedTreatment:
    problems = validate_lock(lock, **kwargs)
    if not lock.is_active:
        problems.append(f"lock is {lock.approval_status}, not approved")
    if problems:
        raise LockedTreatmentFidelityError(lock.treatment_id, problems)
    return lock


def headline_composition(lock: LockedTreatment):
    """The lower_subject composition of a headline lock: its rows come from the locked geometry and times, with no
    discovery, no re-measurement and no re-ranking."""
    from video_edit_agent.direction.composition import (
        LowerSubjectComposition,
        LowerSubjectGeometry,
        _rows,
        hierarchy,
    )
    from video_edit_agent.direction.rhythm import RhythmPolicy

    r = lock.headline
    if lock.treatment_type != TREATMENT_HEADLINE or r is None:
        raise LockedTreatmentFidelityError(lock.treatment_id, ["not a headline lock"])
    geometry = LowerSubjectGeometry(**r.geometry)
    t0, t1, t2, t3 = r.changes
    return LowerSubjectComposition(
        start=t0, end=t3, rows=_rows(geometry, t0, t1, t2, t3, RhythmPolicy()), changes=[t0, t1, t2, t3], phrase=lock.text,
        phrase_start=lock.start, phrase_end=lock.end, headline_in=r.fade_in, headline_out=r.fade_out, geometry=geometry,
        hierarchy=hierarchy(), semantic_source="locked", approval_status=APPROVED,
    )


def headline_spec(lock: LockedTreatment):
    from video_edit_agent.captions.headline import HeadlineSpec

    r = lock.headline
    if r is None:
        raise LockedTreatmentFidelityError(lock.treatment_id, ["not a headline lock"])
    return HeadlineSpec(
        text=lock.text, y_px=r.y_px, fade_in=r.fade_in, fade_out=r.fade_out, font_px=r.font_px, slide_px=r.slide_px,
        settle_from=r.settle_from,
    )


def behind_subject_spec(lock: LockedTreatment):
    """The exact `AnimationSpec` the approved behind-subject preview was rendered from."""
    from video_edit_agent.core.schemas import AnimationKind, AnimationSpec

    r = lock.behind
    if lock.treatment_type != TREATMENT_BEHIND or r is None:
        raise LockedTreatmentFidelityError(lock.treatment_id, ["not a behind-subject lock"])
    payload = {
        "lines": list(r.lines), "fontPx": r.font_px, "centerX": r.center_x, "centerY": r.center_y, "color": r.color,
        "opacity": r.opacity, "outline": r.outline, "shadow": r.shadow, "enterSec": r.enter_sec, "exitSec": r.exit_sec,
        "slot": r.slot, "phrase_timing": r.phrase_timing,
        "placement": {"lines": list(r.lines), "font_px": r.font_px, "bounds": list(r.bounds), **r.occlusion},
        "legibility": r.legibility, "decision": "behind_subject", "locked_treatment": lock.treatment_id,
    }
    return AnimationSpec(
        kind=AnimationKind.BEHIND_TEXT, timeline_start=lock.start, timeline_end=lock.end, text=lock.text, behind_subject=True,
        extra=payload,
    )


def locks_for_render(review_dir: Path, *, total_duration: float | None = None, frame_size: tuple[int, int] | None = None) -> list[LockedTreatment]:
    """The approved locks of a project, each verified renderable (raises `LockedTreatmentFidelityError` otherwise)."""
    return [require_renderable(lk, total_duration=total_duration, frame_size=frame_size) for lk in load_locks(review_dir).active()]


__all__ = [
    "APPROVED",
    "FIDELITY_ERROR_CODE",
    "INVALIDATED",
    "LOCK_FILENAME",
    "PENDING",
    "REJECTED",
    "BehindSubjectRecipe",
    "HeadlineRecipe",
    "LockStore",
    "LockedTreatment",
    "LockedTreatmentFidelityError",
    "apply_revision",
    "approve_from_preview",
    "behind_subject_spec",
    "edit_lock",
    "headline_composition",
    "headline_spec",
    "load_locks",
    "lock_path",
    "locks_for_render",
    "parse_revision",
    "reject",
    "require_renderable",
    "save_locks",
    "serialize",
    "validate_lock",
]
