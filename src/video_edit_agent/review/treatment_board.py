"""The review screen as data (Phase 1.6): one card per major treatment, in plain language.

Each card has a preview, a friendly name, a short reason, a status and the actions Approve / Change / Remove. Nothing
here exposes scores or internal file formats; revisions are natural-language requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video_edit_agent.render import treatment_preview as tpreview
from video_edit_agent.review import camera_look
from video_edit_agent.review import locked_treatments as locks

ACTIONS = ("Approve", "Change", "Remove")
WAITING = "Waiting for your OK"
_STATUS_TEXT = {
    "draft": "Proposed", "pending_review": WAITING, "approved": "Approved", "final_render": "Approved",
    "rejected": "Removed",
}
_LOCK_NAMES = {locks.TREATMENT_HEADLINE: "Headline that sits with the speaker", locks.TREATMENT_BEHIND: "Text behind the speaker"}
_REMOVE_MOTION = "شيل الحركة دي"


@dataclass
class BoardItem:
    treatment_id: str
    name: str
    reason: str
    status: str
    preview: Path | None
    actions: tuple[str, ...] = ACTIONS


def review_dir_of(edit_dir: Path) -> Path:
    edit_dir = Path(edit_dir).resolve()
    return edit_dir if edit_dir.name == "review" else edit_dir / "review"


def items(edit_dir: Path) -> list[BoardItem]:
    review_dir = review_dir_of(edit_dir)
    out: list[BoardItem] = []
    look = camera_look.load_look(review_dir)
    cur = look.current
    if cur.zoom_scale > 0 or cur.preview_path:
        reason = ("No camera movement." if cur.zoom_scale == 0 else
                  "Subtle zoom moves keep the picture alive without distracting from the speaker.")
        out.append(BoardItem(
            camera_look.TREATMENT_ID, "Camera movement", reason, _STATUS_TEXT.get(cur.status, cur.status),
            Path(cur.preview_path) if cur.preview_path else None,
        ))
    for lk in locks.load_locks(review_dir).locks:
        if lk.approval_status == locks.REJECTED:
            continue
        preview = None
        if lk.approval_status == locks.PENDING:
            p = tpreview.headline_preview_path(review_dir, lk) if lk.treatment_type == locks.TREATMENT_HEADLINE else None
            preview = p if p is not None and p.exists() else None
        elif lk.locked_preview_path:
            preview = Path(lk.locked_preview_path)
        out.append(BoardItem(
            lk.treatment_id, _LOCK_NAMES.get(lk.treatment_type, "Text treatment"),
            "A key phrase from the speech, shown large so viewers catch the main point.",
            _STATUS_TEXT.get(lk.approval_status, lk.approval_status), preview,
        ))
    return out


def _target_for(review_dir: Path, request: str) -> str:
    parsed = locks.parse_revision(request)
    if parsed is None:
        raise ValueError("I did not understand that request. Try, for example: كبر الهيدر, خفف الزوم, نزله شوية, شيل الحركة دي.")
    if parsed[0] in ("scale_zoom", "remove_motion"):
        return camera_look.TREATMENT_ID
    live = [lk for lk in locks.load_locks(review_dir).locks if lk.approval_status != locks.REJECTED]
    if not live:
        raise ValueError("There is no headline or text treatment in this video to change.")
    return live[0].treatment_id


def revise(edit_dir: Path, request: str, *, frame_size: tuple[int, int] = (1080, 1920)) -> str:
    """Applies one natural-language revision. The old approval no longer counts: the result waits for a new preview
    and a new OK. Returns the treatment id that changed."""
    review_dir = review_dir_of(edit_dir)
    target = _target_for(review_dir, request)
    if target == camera_look.TREATMENT_ID:
        look = camera_look.load_look(review_dir)
        camera_look.revise(look, request)
        camera_look.save_look(review_dir, look)
        return target
    store = locks.load_locks(review_dir)
    if locks.parse_revision(request)[0] == "undo":
        locks.undo(store, target)
    else:
        locks.apply_revision(store, target, request, frame_width=frame_size[0], frame_height=frame_size[1])
    locks.save_locks(review_dir, store)
    return target


def undo(edit_dir: Path) -> str:
    """Back to the version before the latest change."""
    review_dir = review_dir_of(edit_dir)
    store = locks.load_locks(review_dir)
    for lk in store.locks:
        if any(h.treatment_id == lk.treatment_id for h in store.history):
            locks.undo(store, lk.treatment_id)
            locks.save_locks(review_dir, store)
            return lk.treatment_id
    look = camera_look.load_look(review_dir)
    if look.history:
        camera_look.undo(look)
        camera_look.save_look(review_dir, look)
        return camera_look.TREATMENT_ID
    raise KeyError("There is no earlier version to go back to.")


def approve(edit_dir: Path) -> list[str]:
    """Approves every treatment whose preview is ready: locks it to exactly what the preview shows. Returns the ids
    approved. A treatment without a preview yet is left alone (run `edit` again to render its preview)."""
    review_dir = review_dir_of(edit_dir)
    approved: list[str] = []
    look = camera_look.load_look(review_dir)
    if camera_look.preview_current(look) and (look.approved is None or look.approved.revision != look.current.revision):
        camera_look.approve(look)
        camera_look.save_look(review_dir, look)
        approved.append(camera_look.TREATMENT_ID)
    store = locks.load_locks(review_dir)
    changed = False
    for lk in list(store.locks):
        if lk.approval_status != locks.PENDING or lk.treatment_type != locks.TREATMENT_HEADLINE:
            continue
        p = tpreview.headline_preview_path(review_dir, lk)
        if not p.exists():
            continue
        locks.approve_from_preview(store, lk, p)
        approved.append(lk.treatment_id)
        changed = True
    if changed:
        locks.save_locks(review_dir, store)
    return approved


def remove(edit_dir: Path, treatment_id: str) -> None:
    review_dir = review_dir_of(edit_dir)
    if treatment_id == camera_look.TREATMENT_ID:
        look = camera_look.load_look(review_dir)
        camera_look.revise(look, _REMOVE_MOTION)
        camera_look.save_look(review_dir, look)
        return
    store = locks.load_locks(review_dir)
    locks.reject(store, treatment_id)
    locks.save_locks(review_dir, store)


def waiting_for_preview(edit_dir: Path) -> list[str]:
    """Treatments that changed and whose new preview has not been rendered yet."""
    return [i.treatment_id for i in items(edit_dir) if i.status == WAITING and i.preview is None]


__all__ = ["ACTIONS", "BoardItem", "approve", "items", "remove", "review_dir_of", "revise", "undo", "waiting_for_preview"]
