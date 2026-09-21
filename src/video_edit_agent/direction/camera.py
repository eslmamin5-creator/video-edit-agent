"""Camera grammar for a locked-off talking head.

Moves are tied to semantic beats, never to a fixed interval, and they are
ABSOLUTE: every move targets a zoom level measured from the base framing, so
zoom can never accumulate (five punch-ins in a row still peak at the policy's
punch-in level). Emphasis is followed by a reset/punch-out; static is a valid
choice; there is no continuous fake camera motion.

    static          nothing moves
    punch_in        ease to the punch-in level
    punch_out       ease back out (to base)
    slow_push       a slow drift to a small level over the beat, then reset
    reframe_left/right  a slight zoom with the anchor shifted (needs room), then reset
    reset_to_base   ease back to the base framing
    slow_pull       the gradual glide back to base after a slow_push
    lower_subject   top-anchored zoom: the subject sits lower in the frame (real headroom, head never cropped)
    raise_subject   planning-only: the renderer cannot pad above the source frame

Spacing and density rules drop moves that are too close or too many; every
drop is recorded with its reason. The plan is realised through the EXISTING
`Reframe`/`ZoomRamp` model (`render.reframe`), which keeps the crop window
face-safe; `apply_to_edl` only replaces the fixed-cycle assignment when called.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import DEFAULT_ZOOM_ANCHOR_Y, EDL, Reframe, ZoomRamp
from video_edit_agent.render import reframe as reframe_math

BASE_ZOOM = 1.0
BASE_ANCHOR_X = 0.5


class CameraMove(str, Enum):
    STATIC = "static"
    PUNCH_IN = "punch_in"
    PUNCH_OUT = "punch_out"
    SLOW_PUSH = "slow_push"
    REFRAME_LEFT = "reframe_left"
    REFRAME_RIGHT = "reframe_right"
    RESET_TO_BASE = "reset_to_base"
    SLOW_PULL = "slow_pull"
    LOWER_SUBJECT = "lower_subject"
    RAISE_SUBJECT = "raise_subject"


class EventStatus(str, Enum):
    """Whether the renderer will actually draw an event. Only `executable` events can be rendered."""

    PLANNING_ONLY = "planning_only"  # proposed, but the renderer cannot (or is not yet proven to) realise it
    EXECUTABLE = "executable"
    BLOCKED = "blocked"  # would render, but a safety rule forbids it
    SUPERSEDED = "superseded"  # an older event that a newer owner replaced; it never renders


class MotionClass(str, Enum):
    ABRUPT = "abrupt"  # perceptually discrete: wants a phrase boundary
    SMOOTH = "smooth"  # gradual: may begin at any word boundary
    NONE = "none"


class CameraPolicy(BaseModel):
    """How much camera a project wants (derived from a motion-energy word, not a brand)."""

    punch_in_zoom: float = 1.08
    slow_push_zoom: float = 1.05
    reframe_zoom: float = 1.05
    reframe_shift: float = 0.05  # anchor_x offset, still clamped face-safe by the renderer
    max_zoom: float = 1.14
    ease_s: float = 0.3
    max_hold_s: float = 4.0  # emphasis is reset after at most this long
    min_gap_s: float = 3.0  # between the starts of two moves (resets excluded)
    max_moves_per_10s: int = 1


POLICIES: dict[str, CameraPolicy] = {
    "low": CameraPolicy(punch_in_zoom=1.05, slow_push_zoom=1.03, reframe_zoom=1.04, min_gap_s=5.0, max_hold_s=3.5),
    "medium": CameraPolicy(),
    "high": CameraPolicy(punch_in_zoom=1.10, max_zoom=1.14, min_gap_s=2.0, max_moves_per_10s=2),
}


def policy_for(energy: str | None) -> CameraPolicy:
    return POLICIES.get((energy or "medium").lower(), POLICIES["medium"])


class CameraRequest(BaseModel):
    """What a semantic beat asks of the camera."""

    start: float
    end: float
    move: CameraMove = CameraMove.STATIC
    importance: float = 0.5


class CameraEvent(BaseModel):
    """One camera move. `start`/`end` are the motion itself (start = motion start, end = settle); `zoom_from` /
    `anchor_*_from` are the state it starts from and `zoom_to` / `anchor_*` the ABSOLUTE state it settles on."""

    start: float  # the ramp begins here (timeline seconds)
    end: float  # ... and reaches `zoom_to` here
    move: CameraMove
    zoom_to: float  # ABSOLUTE zoom level
    anchor_x: float = BASE_ANCHOR_X
    beat_start: float | None = None
    beat_end: float | None = None
    zoom_from: float | None = None
    anchor_x_from: float | None = None
    anchor_y: float = DEFAULT_ZOOM_ANCHOR_Y
    anchor_y_from: float | None = None
    easing: str = "smoothstep"
    status: EventStatus = EventStatus.EXECUTABLE
    status_reason: str = ""
    owner: str = "rhythm"  # rhythm | pinned | director | legacy
    motion_class: MotionClass = MotionClass.ABRUPT
    boundary_kind: str = ""
    boundary_quality: float | None = None
    number: int = 0  # the rhythm entry this event belongs to
    safe_headroom: float | None = None  # normalised face-top space after the move (None: no measured face)
    headroom_gain: float | None = None
    face_bottom: float | None = None
    superseded_by: str = ""

    @property
    def scale(self) -> float:
        return self.zoom_to

    @property
    def motion_start(self) -> float:
        return self.start

    @property
    def motion_end(self) -> float:
        return self.end


class DroppedMove(BaseModel):
    start: float
    move: CameraMove
    reason: str


class CameraPlan(BaseModel):
    events: list[CameraEvent] = Field(default_factory=list)
    dropped: list[DroppedMove] = Field(default_factory=list)
    policy: CameraPolicy = Field(default_factory=CameraPolicy)


_MOVES_THAT_ZOOM = {CameraMove.PUNCH_IN, CameraMove.SLOW_PUSH, CameraMove.REFRAME_LEFT, CameraMove.REFRAME_RIGHT}


def _target(move: CameraMove, policy: CameraPolicy) -> float:
    level = {
        CameraMove.PUNCH_IN: policy.punch_in_zoom, CameraMove.SLOW_PUSH: policy.slow_push_zoom,
        CameraMove.REFRAME_LEFT: policy.reframe_zoom, CameraMove.REFRAME_RIGHT: policy.reframe_zoom,
    }.get(move, BASE_ZOOM)
    return round(min(level, policy.max_zoom), 4)


def plan_camera(requests: list[CameraRequest], policy: CameraPolicy | None = None) -> CameraPlan:
    """Sequences the beats' camera requests into absolute, non-accumulating events."""
    policy = policy or CameraPolicy()
    plan = CameraPlan(policy=policy)
    state = {"zoom": BASE_ZOOM, "anchor": BASE_ANCHOR_X}
    last_move_start: float | None = None
    move_starts: list[float] = []

    def emit(start: float, end: float, move: CameraMove, zoom: float, anchor: float, beat: CameraRequest) -> None:
        plan.events.append(CameraEvent(
            start=round(start, 3), end=round(end, 3), move=move, zoom_to=zoom, anchor_x=anchor,
            beat_start=beat.start, beat_end=beat.end,
        ))
        state["zoom"], state["anchor"] = zoom, anchor

    def reset_if_due(now: float) -> None:
        """Emphasis never survives its beat: reset any earlier zoom that is still held at `now`."""
        pending = plan.events[-1] if plan.events else None
        if pending is None or state["zoom"] <= BASE_ZOOM + 1e-6:
            return
        due = min((pending.beat_end or now), pending.end + policy.max_hold_s, now)
        # a beat that starts at 0.0 is a real start: `or` would treat it as missing
        beat = CameraRequest(start=due if pending.beat_start is None else pending.beat_start, end=due if pending.beat_end is None else pending.beat_end)
        emit(max(due, pending.end), max(due, pending.end) + policy.ease_s, CameraMove.RESET_TO_BASE, BASE_ZOOM, BASE_ANCHOR_X, beat)

    for req in sorted(requests, key=lambda r: (r.start, -r.importance)):
        if req.move is CameraMove.STATIC:
            reset_if_due(req.start)
            continue
        if req.move in {CameraMove.PUNCH_OUT, CameraMove.RESET_TO_BASE}:
            if state["zoom"] > BASE_ZOOM + 1e-6:
                emit(req.start, req.start + policy.ease_s, req.move, BASE_ZOOM, BASE_ANCHOR_X, req)
            else:
                plan.dropped.append(DroppedMove(start=req.start, move=req.move, reason="already at base framing"))
            continue
        reset_if_due(req.start)
        if state["zoom"] > BASE_ZOOM + 1e-6:  # a previous emphasis still holds: never stack another on top
            plan.dropped.append(DroppedMove(start=req.start, move=req.move, reason="a previous emphasis has not been reset yet"))
            continue
        if last_move_start is not None and req.start - last_move_start < policy.min_gap_s:
            plan.dropped.append(DroppedMove(start=req.start, move=req.move, reason="too close to the previous move"))
            continue
        if sum(1 for s in move_starts if 0 <= req.start - s < 10.0) >= policy.max_moves_per_10s:
            plan.dropped.append(DroppedMove(start=req.start, move=req.move, reason="too many moves in 10 seconds"))
            continue
        target = _target(req.move, policy)
        anchor = BASE_ANCHOR_X
        if req.move is CameraMove.REFRAME_LEFT:
            anchor = BASE_ANCHOR_X - policy.reframe_shift
        elif req.move is CameraMove.REFRAME_RIGHT:
            anchor = BASE_ANCHOR_X + policy.reframe_shift
        length = max(policy.ease_s, min(req.end - req.start, policy.max_hold_s)) if req.move is CameraMove.SLOW_PUSH else policy.ease_s
        emit(req.start, req.start + length, req.move, target, anchor, req)
        last_move_start = req.start
        move_starts.append(req.start)
    if plan.events:
        reset_if_due(float("inf"))
    return plan


def zoom_trajectory(plan: CameraPlan, t: float) -> float:
    """The zoom level at timeline second `t` (eased exactly as the renderer eases it)."""
    zoom = BASE_ZOOM
    for ev in plan.events:
        span = max(ev.end - ev.start, 1e-3)
        zoom += (ev.zoom_to - zoom_before(plan, ev)) * reframe_math._ease((t - ev.start) / span, "smoothstep")
    return zoom


def zoom_before(plan: CameraPlan, event: CameraEvent) -> float:
    index = plan.events.index(event)
    return plan.events[index - 1].zoom_to if index else BASE_ZOOM


def peak_zoom(plan: CameraPlan) -> float:
    return max((e.zoom_to for e in plan.events), default=BASE_ZOOM)


def apply_to_edl(edl: EDL, plan: CameraPlan, *, face_box: tuple[float, float, float, float] | None = None) -> None:
    """Replaces the per-clip framing of `edl` with `plan`, in place, through the existing
    `Reframe`/`ZoomRamp` model. A move that straddles a cut is split across both clips."""
    for clip in edl.clips:
        lo, hi = clip.timeline_in, clip.timeline_out
        ramps: list[ZoomRamp] = []
        level = BASE_ZOOM
        zoom_start = BASE_ZOOM
        anchor = BASE_ANCHOR_X
        for ev in plan.events:
            if ev.end <= lo:  # finished before this clip: it only sets the starting level
                level = zoom_start = ev.zoom_to
                anchor = ev.anchor_x
            elif ev.start < hi:
                ramps.append(ZoomRamp(start_s=round(ev.start - lo, 3), end_s=round(ev.end - lo, 3), zoom_to=ev.zoom_to))
                anchor = ev.anchor_x if ev.zoom_to > BASE_ZOOM else anchor
                level = ev.zoom_to
        clip.zoom = max([zoom_start, *(r.zoom_to for r in ramps)])
        clip.reframe = Reframe(zoom_start=zoom_start, ramps=ramps, anchor_x=anchor if level > BASE_ZOOM or ramps else BASE_ANCHOR_X, face_box=face_box)


__all__ = [
    "BASE_ANCHOR_X",
    "BASE_ZOOM",
    "POLICIES",
    "CameraEvent",
    "CameraMove",
    "CameraPlan",
    "CameraPolicy",
    "CameraRequest",
    "DroppedMove",
    "EventStatus",
    "MotionClass",
    "apply_to_edl",
    "peak_zoom",
    "plan_camera",
    "policy_for",
    "zoom_trajectory",
]
