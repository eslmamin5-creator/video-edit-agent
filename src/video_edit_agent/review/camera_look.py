"""The automatic Visual Rhythm camera as a reviewable treatment (Phase 1.6).

Every project gets subtle camera motion. The user judges it on a short preview and can ask for more or less of it
("خفف الزوم" / "زود الزوم" / "شيل الحركة دي"). The lifecycle is the same as a Locked Treatment Spec:

    draft -> pending_review (a revision) -> approved + locked -> final_render

A pending revision never renders (the final render uses the approved version, or the automatic default while nothing
is approved), and a revision of an approved version invalidates that approval until the new preview is approved again.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from video_edit_agent.review.locked_treatments import LockedTreatmentFidelityError, parse_revision

LOOK_FILENAME = "camera_motion.json"
TREATMENT_ID = "camera_motion"
DRAFT, PENDING, APPROVED, FINAL = "draft", "pending_review", "approved", "final_render"
MAX_SCALE = 2.0
ZOOM_TOL = 0.02  # planned peak zoom vs the approved one


class CameraVersion(BaseModel):
    zoom_scale: float = 1.0  # 1.0 = the automatic rhythm, 0 = no zoom at all
    revision: int = 1
    status: str = DRAFT
    preview_path: str = ""
    preview_sha256: str = ""
    window: tuple[float, float] = (0.0, 0.0)  # the timeline seconds the preview shows
    peak_zoom: float = 1.0  # planned zoom peak inside the window, measured when the preview was rendered
    locked_at: str = ""

    @property
    def is_active(self) -> bool:
        return self.status in (APPROVED, FINAL)


class CameraLook(BaseModel):
    schema_version: int = 1
    current: CameraVersion = Field(default_factory=CameraVersion)  # what the user is looking at
    approved: CameraVersion | None = None  # the locked version the final render follows
    history: list[CameraVersion] = Field(default_factory=list)  # earlier versions, oldest first ("undo")

    @property
    def pending(self) -> bool:
        return self.current.status == PENDING

    def render_scale(self) -> float:
        """The scale the final render uses: the approved version, else the automatic default. Never a pending draft."""
        return self.approved.zoom_scale if self.approved is not None else 1.0


def look_path(review_dir: Path) -> Path:
    return Path(review_dir) / LOOK_FILENAME


def load_look(review_dir: Path) -> CameraLook:
    path = look_path(review_dir)
    if not path.exists():
        return CameraLook()
    return CameraLook.model_validate_json(path.read_text(encoding="utf-8"))


def save_look(review_dir: Path, look: CameraLook) -> Path:
    path = look_path(review_dir)
    text = json.dumps(look.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def revise(look: CameraLook, request: str) -> CameraVersion:
    """One natural-language request -> a new pending_review version. An approved version is invalidated by the
    revision (it stays in the history)."""
    parsed = parse_revision(request)
    if parsed is None or parsed[0] not in ("scale_zoom", "remove_motion"):
        raise ValueError(f"not a camera-motion request: {request!r}")
    op, amount = parsed
    scale = 0.0 if op == "remove_motion" else min(MAX_SCALE, round(look.current.zoom_scale * amount, 4))
    look.history.append(look.current)
    look.approved = None
    look.current = CameraVersion(zoom_scale=scale, revision=look.current.revision + 1, status=PENDING)
    return look.current


def attach_preview(look: CameraLook, path: Path, window: tuple[float, float], peak_zoom: float) -> None:
    look.current = look.current.model_copy(update={
        "preview_path": str(Path(path).resolve()), "preview_sha256": _sha256(path), "window": (round(window[0], 3), round(window[1], 3)),
        "peak_zoom": round(peak_zoom, 4),
    })


def preview_current(look: CameraLook) -> bool:
    p = look.current.preview_path
    return bool(p) and Path(p).exists() and _sha256(Path(p)) == look.current.preview_sha256


def approve(look: CameraLook, *, now: str | None = None) -> CameraVersion:
    """The user approved the preview of `look.current`: lock it. Approving an unchanged, already-approved version
    changes nothing."""
    cur = look.current
    if not preview_current(look):
        raise LockedTreatmentFidelityError(TREATMENT_ID, ["the preview of this version is missing; render it before approving"])
    if look.approved is not None and look.approved.zoom_scale == cur.zoom_scale and look.approved.revision == cur.revision:
        return look.approved
    if look.approved is not None:
        look.history.append(look.approved)
    locked = cur.model_copy(update={"status": APPROVED, "locked_at": now or _now()})
    look.current = locked
    look.approved = locked
    return locked


def undo(look: CameraLook) -> CameraVersion:
    """Back to the version before the current one (its approval, if it had one, is restored with it)."""
    if not look.history:
        raise KeyError("no earlier version")
    prev = look.history.pop()
    look.current = prev
    look.approved = prev if prev.is_active else None
    return prev


def mark_rendered(look: CameraLook) -> None:
    if look.approved is not None and look.approved.status != FINAL:
        look.approved = look.approved.model_copy(update={"status": FINAL})
        if look.current.revision == look.approved.revision:
            look.current = look.approved


def validate_for_render(look: CameraLook) -> list[str]:
    """Problems that make the approved version irreproducible (missing preview provenance)."""
    a = look.approved
    if a is None:
        return []
    problems = []
    if not a.preview_path or not Path(a.preview_path).exists():
        problems.append("approved preview is missing")
    elif _sha256(Path(a.preview_path)) != a.preview_sha256:
        problems.append("approved preview changed after approval")
    if not (0.0 <= a.zoom_scale <= MAX_SCALE) or math.isnan(a.zoom_scale):
        problems.append("zoom scale out of range")
    return problems


__all__ = [
    "APPROVED", "DRAFT", "FINAL", "PENDING", "TREATMENT_ID", "CameraLook", "CameraVersion", "approve", "attach_preview",
    "load_look", "mark_rendered", "preview_current", "revise", "save_look", "undo", "validate_for_render",
]
