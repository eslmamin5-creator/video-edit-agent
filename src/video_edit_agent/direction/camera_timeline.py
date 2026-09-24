"""The ONE canonical camera pipeline.

    EditPlan / rhythm decisions  ->  camera timeline (this module)  ->  renderer (`render.reframe`)

Every camera move the renderer will draw is an event on a `CameraTimeline`. The timeline is built from three
sources and nothing else reaches the renderer:

    rhythm    the visual rhythm engine's excursions (reframe, slow push/pull, punch, lower_subject, resets)
    pinned    camera decisions the semantic director made or the user approved (they keep their interval)
    legacy    the older fixed-cycle EDL zoom and the plan's `static` slot claims: marked SUPERSEDED wherever the
              timeline owns the interval, so two systems never move the same picture

Each event carries an explicit status:

    executable     the renderer draws exactly this event (proven geometry, face-safe)
    planning_only  proposed, but the renderer cannot (or is not yet proven to) realise it: never drawn
    blocked        it would render but a safety rule forbids it (Behind-Subject is active, unsafe headroom, conflict)
    superseded     an older event a newer owner replaced: never drawn

The plan is render-ready only while the review-visible camera state equals the executable events and no
planning-only event remains. Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Protocol

from pydantic import BaseModel, Field

from video_edit_agent.core.schemas import DEFAULT_ZOOM_ANCHOR_Y, EDL, EDLClip, Reframe, ZoomRamp
from video_edit_agent.direction.camera import (
    BASE_ANCHOR_X,
    BASE_ZOOM,
    CameraEvent,
    CameraMove,
    CameraPlan,
    EventStatus,
    MotionClass,
)
from video_edit_agent.direction.rhythm import RhythmPlan, RhythmRow, row_event
from video_edit_agent.render import reframe as reframe_math

BASE_ANCHOR_Y = DEFAULT_ZOOM_ANCHOR_Y
FaceBox = tuple[float, float, float, float]
_EPS = 1e-3
_BACK = {CameraMove.RESET_TO_BASE, CameraMove.SLOW_PULL, CameraMove.PUNCH_OUT}
# What Behind-Subject text tolerates while it is the primary visual: a hold, a very small reframe, or a pre-validated lower_subject
BEHIND_SUBJECT_MAX_ZOOM_DELTA = 0.03

# Every move the timeline can turn into pixels. raise_subject is NOT here: the renderer cannot pad above the source frame.
EXECUTABLE_MOVES = frozenset({
    CameraMove.REFRAME_LEFT, CameraMove.REFRAME_RIGHT, CameraMove.SLOW_PUSH, CameraMove.SLOW_PULL, CameraMove.PUNCH_IN,
    CameraMove.PUNCH_OUT, CameraMove.RESET_TO_BASE, CameraMove.LOWER_SUBJECT,
})
PLANNING_ONLY_MOVES = frozenset({CameraMove.RAISE_SUBJECT})


class CameraSlot(Protocol):
    """The part of an edit-plan slot the reconciliation needs."""

    number: int
    timeline_start: float
    timeline_end: float
    camera: str


class CameraTimeline(BaseModel):
    """The single source of truth for camera execution."""

    start: float = 0.0  # the stretch this timeline owns; the EDL before `start` (the hook) is left as it is
    end: float = 0.0
    events: list[CameraEvent] = Field(default_factory=list)  # live: executable / planning_only / blocked, by start
    superseded: list[CameraEvent] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)  # anything that makes the plan not render-ready
    face_box: FaceBox | None = None

    @property
    def executable(self) -> list[CameraEvent]:
        return [e for e in self.events if e.status is EventStatus.EXECUTABLE]

    @property
    def planning_only(self) -> list[CameraEvent]:
        return [e for e in self.events if e.status is EventStatus.PLANNING_ONLY]

    @property
    def blocked(self) -> list[CameraEvent]:
        return [e for e in self.events if e.status is EventStatus.BLOCKED]

    @property
    def blockers(self) -> list[str]:
        out = list(self.issues)
        out += [f"{e.move.value} at {e.start:.2f}s is planning-only: {e.status_reason or 'not renderable'}" for e in self.planning_only]
        return out

    @property
    def render_ready(self) -> bool:
        """True only when every live visual event is executable (blocked and superseded events never render, so
        they cannot make the picture differ from the review) and no consistency issue remains."""
        return not self.blockers

    def summary(self) -> dict[str, int]:
        return {s.value: sum(1 for e in self.events if e.status is s) + (len(self.superseded) if s is EventStatus.SUPERSEDED else 0)
                for s in EventStatus}


# --------------------------------------------------------------------------
# Building the timeline
# --------------------------------------------------------------------------


def _intervals(rows: Sequence[RhythmRow]) -> dict[int, tuple[float, float]]:
    """(start, end) of each rhythm-owned excursion: its move, hold and return together."""
    out: dict[int, tuple[float, float]] = {}
    for r in rows:
        if r.source != "rhythm" or r.state == "base":
            continue
        lo, hi = out.get(r.number, (r.start, r.end))
        out[r.number] = (min(lo, r.start), max(hi, r.end))
    return out


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > _EPS


def _classify(ev: CameraEvent, rhythm: RhythmPlan, face_box: FaceBox | None) -> None:
    """Sets the status of a rhythm event from what the renderer can prove."""
    policy = rhythm.policy
    if ev.move in PLANNING_ONLY_MOVES:
        ev.status, ev.status_reason = EventStatus.PLANNING_ONLY, "raise_subject is planning-only: the renderer cannot pad above the source frame"
    elif ev.move is CameraMove.LOWER_SUBJECT:
        if face_box is None:
            ev.status, ev.status_reason = EventStatus.PLANNING_ONLY, "lower_subject needs a measured face box before it can be proven safe"
        elif ev.headroom_gain is None or ev.headroom_gain < policy.min_headroom_gain - 1e-9:
            ev.status, ev.status_reason = EventStatus.BLOCKED, f"lower_subject would add only {ev.headroom_gain or 0.0:.3f} of headroom (< {policy.min_headroom_gain:.2f})"
        elif ev.face_bottom is not None and ev.face_bottom > policy.caption_safe_top + 1e-9:
            ev.status, ev.status_reason = EventStatus.BLOCKED, f"the lowered face would reach {ev.face_bottom:.2f}, inside the caption band"
        else:
            ev.status, ev.status_reason = EventStatus.EXECUTABLE, f"top space +{ev.headroom_gain:.3f}; face bottom {ev.face_bottom:.2f} above the caption band"
    elif ev.move in EXECUTABLE_MOVES:
        ev.status = EventStatus.EXECUTABLE
    else:
        ev.status, ev.status_reason = EventStatus.PLANNING_ONLY, f"{ev.move.value} has no renderer mapping"


def behind_subject_compatible(ev: CameraEvent) -> tuple[bool, str]:
    """May this event run while Behind-Subject text is the primary visual? A hold (no event), a very subtle stable
    reframe and a pre-validated lower_subject are fine; a punch, a rapid move or any large zoom is not."""
    if ev.motion_class is MotionClass.ABRUPT or ev.move in (CameraMove.PUNCH_IN, CameraMove.PUNCH_OUT):
        return False, "a punch/abrupt move would compete with the text behind the subject"
    if ev.move is CameraMove.LOWER_SUBJECT:
        return (ev.status is EventStatus.EXECUTABLE), "lower_subject is allowed only when pre-validated"
    delta = abs(ev.zoom_to - (ev.zoom_from if ev.zoom_from is not None else BASE_ZOOM))
    if delta > BEHIND_SUBJECT_MAX_ZOOM_DELTA + 1e-9:
        return False, f"a {delta:.3f} zoom move is too large while text sits behind the subject"
    if ev.end - ev.start < 0.6 and delta > 1e-6:
        return False, "a rapid move would shake the text while it sits behind the subject"
    return True, ""


def _rhythm_events(rhythm: RhythmPlan, face_box: FaceBox | None) -> list[CameraEvent]:
    events = []
    for r in rhythm.rows:
        if r.source != "rhythm" or not r.moving:
            continue
        ev = row_event(r, face_box)
        ev.owner = "rhythm"
        _classify(ev, rhythm, face_box)
        events.append(ev)
    return events


def _mirror(rhythm: RhythmPlan, events: Iterable[CameraEvent], superseded_by: dict[int, list[str]]) -> None:
    """The review rows show the status of the event that will (or will not) render."""
    worst: dict[int, CameraEvent] = {}
    rank = {EventStatus.BLOCKED: 3, EventStatus.PLANNING_ONLY: 2, EventStatus.EXECUTABLE: 1}
    for e in events:
        cur = worst.get(e.number)
        if cur is None or rank.get(e.status, 0) > rank.get(cur.status, 0):
            worst[e.number] = e
    for r in rhythm.rows:
        if r.source != "rhythm":
            continue
        e = worst.get(r.number)
        if e is not None:
            r.status, r.status_reason, r.executable = e.status.value, e.status_reason, e.status is EventStatus.EXECUTABLE
            if r.state == "lower_subject":
                r.safe_headroom, r.headroom_gain, r.face_bottom = e.safe_headroom, e.headroom_gain, e.face_bottom
        r.superseded_legacy = list(superseded_by.get(r.number, []))


def _describe(ev: CameraEvent) -> str:
    return f"{ev.owner}:{ev.move.value}@{ev.start:.2f}-{ev.end:.2f}"


def build_camera_timeline(
    rhythm: RhythmPlan,
    *,
    face_box: FaceBox | None = None,
    legacy: Sequence[CameraEvent] = (),
    pinned: Sequence[CameraEvent] = (),
    behind_subject: Sequence[tuple[float, float]] = (),
    scope: tuple[float, float] | None = None,
) -> CameraTimeline:
    """Normalises the rhythm decisions, the pinned/director camera events and the legacy events into one timeline.

    * a legacy event inside `scope` is SUPERSEDED (recorded, never rendered): rhythm owns the whole picture there,
      whether an excursion is planned at that moment or the framing simply stays on base;
    * a pinned event keeps its interval; a rhythm event that overlaps one is BLOCKED (it never overrides your decision);
    * a rhythm event that overlaps an active Behind-Subject window is BLOCKED unless it is compatible;
    * two live executable events never overlap in time (no double camera execution)."""
    lo, hi = scope if scope is not None else (rhythm.start, rhythm.end)
    tl = CameraTimeline(start=lo, end=hi, face_box=face_box)
    live = _rhythm_events(rhythm, face_box)
    keep_pinned = [p.model_copy(update={"owner": p.owner if p.owner in ("pinned", "director") else "pinned"}) for p in pinned]

    for ev in live:  # a pinned decision wins its interval
        if any(_overlaps(ev.start, ev.end, p.start, p.end) for p in keep_pinned):
            ev.status, ev.status_reason = EventStatus.BLOCKED, "overlaps a camera decision that is pinned by you / the semantic director"
    lowered = {e.number for e in live if e.move is CameraMove.LOWER_SUBJECT and e.status is EventStatus.EXECUTABLE}
    for ev in live:  # Behind-Subject is the primary visual while it is active
        if ev.status is EventStatus.BLOCKED:
            continue
        for b0, b1 in behind_subject:
            if _overlaps(ev.start, ev.end, b0, b1):
                # the glide back out of a pre-validated lower_subject belongs to that composition
                ok, why = (True, "") if ev.number in lowered and ev.move in _BACK and ev.motion_class is MotionClass.SMOOTH else behind_subject_compatible(ev)
                if not ok:
                    ev.status, ev.status_reason = EventStatus.BLOCKED, f"Behind-Subject is active {b0:.2f}-{b1:.2f}s: {why}"
                break

    for ev in live:  # an excursion is all-or-nothing: if its move cannot render, its return has nothing to return from
        worst = next((o for o in live if o.number == ev.number and o.status is not EventStatus.EXECUTABLE), None)
        if worst is not None and ev.status is EventStatus.EXECUTABLE:
            ev.status, ev.status_reason = worst.status, f"part of an excursion whose move is {worst.status.value}: {worst.status_reason}"

    owned = _intervals(rhythm.rows)
    superseded_by: dict[int, list[str]] = {}
    for old in legacy:
        if old.owner in ("pinned", "director"):
            continue
        if not (lo - _EPS <= old.start < hi):  # outside the stretch the timeline owns: the older system still owns it
            continue
        stop = max(old.end, old.start + _EPS)
        holder = next((n for n, (a, b) in owned.items() if _overlaps(old.start, stop, a, b)), None)
        by = f"rhythm#{holder}" if holder is not None else "camera_timeline (framing stays on base here)"
        tl.superseded.append(old.model_copy(update={
            "status": EventStatus.SUPERSEDED, "superseded_by": by,
            "status_reason": f"the canonical camera timeline owns {old.start:.2f}-{max(old.end, old.start):.2f}s",
        }))
        if holder is not None:
            superseded_by.setdefault(holder, []).append(_describe(old))

    tl.events = sorted([*live, *keep_pinned], key=lambda e: (e.start, e.end))
    _mirror(rhythm, live, superseded_by)
    tl.issues = validate(tl)
    return tl


def validate(tl: CameraTimeline) -> list[str]:
    """Consistency problems that make the plan not render-ready: two live events that would move the picture at the same
    time (a double camera execution), or a superseded event that is still live."""
    out: list[str] = []
    running = sorted((e for e in tl.events if e.status is EventStatus.EXECUTABLE), key=lambda e: (e.start, e.end))
    for a, b in pairwise(running):
        if b.start < a.end - _EPS:
            out.append(f"double camera execution: {_describe(a)} overlaps {_describe(b)}")
    for e in tl.events:
        if e.status is EventStatus.SUPERSEDED:
            out.append(f"a superseded event is still live: {_describe(e)}")
    return out


# --------------------------------------------------------------------------
# Review == executable
# --------------------------------------------------------------------------

_FAMILY = {CameraMove.PUNCH_OUT: CameraMove.RESET_TO_BASE, CameraMove.SLOW_PULL: CameraMove.RESET_TO_BASE}


def _family(move: CameraMove | str) -> CameraMove:
    m = CameraMove(move)
    return _FAMILY.get(m, m)


def slot_events(timeline: CameraTimeline, slot: CameraSlot) -> list[CameraEvent]:
    """The executable events that move the picture inside `slot`."""
    return [e for e in timeline.executable if _overlaps(e.start, e.end, slot.timeline_start, slot.timeline_end)]


def review_mismatches(slots: Iterable[CameraSlot], timeline: CameraTimeline) -> list[str]:
    """Where the camera the review shows differs from the event the renderer will draw. Slots that carry no camera
    claim ('n/a', a replaced speaker) are not part of the comparison."""
    out: list[str] = []
    for s in slots:
        if s.camera in ("n/a", ""):
            continue
        actual = slot_events(timeline, s)
        if s.camera == "static":
            if actual:
                out.append(f"slot {s.number}: review says camera=static but the timeline will move the camera "
                           f"({', '.join(_describe(e) for e in actual)})")
        elif not any(_family(e.move) is _family(s.camera) for e in actual):
            out.append(f"slot {s.number}: review says camera={s.camera} but no executable event draws it")
    return out


def reconcile_slots(slots: Iterable[CameraSlot], timeline: CameraTimeline) -> None:
    """Makes each slot show the camera that will actually render. A slot the timeline owns (`camera_owner == 'rhythm'`, or
    a `static` claim that overlaps a rhythm event) shows the event sequence; a slot with a pinned camera keeps it.
    Mutates the slots (they carry `camera`, `camera_owner` and `camera_events`)."""
    for s in slots:
        if s.camera in ("n/a", ""):  # settled slots (number 0) are reconciled too: a "static" claim must not contradict a moving rhythm event
            continue
        if s.camera != "static" and getattr(s, "camera_owner", "") != "rhythm":
            continue  # a pinned / director camera keeps its own claim
        actual = [e for e in slot_events(timeline, s) if e.owner == "rhythm"]
        if hasattr(s, "camera_events"):
            s.camera_events = [f"{e.move.value} {e.start:.2f}-{e.end:.2f}s -> zoom {e.zoom_to:.3f}" for e in actual]
        if actual:
            s.camera = next((e for e in actual if e.move not in _BACK), actual[0]).move.value
            if hasattr(s, "camera_owner"):
                s.camera_owner = "rhythm"
        else:
            s.camera = "static"
            if hasattr(s, "camera_owner"):
                s.camera_owner = ""


# --------------------------------------------------------------------------
# Legacy events (the older fixed-cycle EDL zoom and static slot claims)
# --------------------------------------------------------------------------


def legacy_events(edl: EDL) -> list[CameraEvent]:
    """The camera the EDL already carries (the fixed-cycle punch-in assignment), as legacy events."""
    out: list[CameraEvent] = []
    for clip in edl.clips:
        lo = clip.timeline_in
        rf = reframe_math.resolve_reframe(clip)
        if rf is None:
            continue
        if abs(rf.zoom_start - 1.0) > 1e-6:
            out.append(CameraEvent(start=lo, end=lo, move=CameraMove.PUNCH_IN, zoom_to=rf.zoom_start, anchor_x=rf.anchor_x, anchor_y=rf.anchor_y,
                                   owner="legacy", status=EventStatus.EXECUTABLE, status_reason="legacy EDL zoom held from the clip start"))
        level = rf.zoom_start
        for ramp in rf.ramps:
            move = CameraMove.PUNCH_IN if ramp.zoom_to > level + 1e-6 else CameraMove.RESET_TO_BASE
            out.append(CameraEvent(start=round(lo + ramp.start_s, 3), end=round(lo + ramp.end_s, 3), move=move, zoom_to=ramp.zoom_to,
                                   zoom_from=level, anchor_x=rf.anchor_x, anchor_y=rf.anchor_y, owner="legacy", status_reason="legacy EDL zoom ramp"))
            level = ramp.zoom_to
    return out


def static_claims(slots: Iterable[CameraSlot]) -> list[CameraEvent]:
    """Each slot that claims `camera=static` as a (do-nothing) legacy event, so the claim is superseded explicitly wherever
    the timeline owns the interval instead of coexisting with it."""
    return [CameraEvent(start=s.timeline_start, end=s.timeline_end, move=CameraMove.STATIC, zoom_to=BASE_ZOOM, owner="legacy",
                        status_reason="the plan's slot camera=static") for s in slots if s.camera == "static" and s.number]


# --------------------------------------------------------------------------
# Applying the timeline to the renderer
# --------------------------------------------------------------------------


def _target(ev: CameraEvent) -> tuple[float, float, float]:
    """(zoom, anchor_x, anchor_y) the event settles on (a return settles on the canonical base exactly)."""
    if ev.move in _BACK:
        return BASE_ZOOM, BASE_ANCHOR_X, BASE_ANCHOR_Y
    return ev.zoom_to, ev.anchor_x, ev.anchor_y


def _clip_reframe(clip: EDLClip, events: Sequence[CameraEvent], face_box: FaceBox | None) -> Reframe | None:
    lo, hi = clip.timeline_in, clip.timeline_out
    cur = (BASE_ZOOM, BASE_ANCHOR_X, BASE_ANCHOR_Y)
    start_state = cur
    ramps: list[ZoomRamp] = []
    for ev in events:
        if ev.end <= lo + _EPS:  # finished before this clip: it only sets the state the clip starts from
            cur = start_state = _target(ev)
        elif ev.start < hi - _EPS:
            z, ax, ay = _target(ev)
            ramps.append(ZoomRamp(
                start_s=round(ev.start - lo, 3), end_s=round(ev.end - lo, 3), zoom_to=z, easing=ev.easing,
                anchor_x_to=None if abs(ax - cur[1]) < 1e-6 else ax, anchor_y_to=None if abs(ay - cur[2]) < 1e-6 else ay))
            cur = (z, ax, ay)
    if not ramps and abs(start_state[0] - BASE_ZOOM) < 1e-6:
        return None
    return Reframe(zoom_start=start_state[0], ramps=ramps, anchor_x=start_state[1], anchor_y=start_state[2], face_box=face_box,
                   owner="camera_timeline")


def apply_timeline(edl: EDL, timeline: CameraTimeline, *, face_box: FaceBox | None = None) -> None:
    """Replaces the framing of every clip the timeline owns (`timeline.start` onwards) with the executable events, in place,
    through the existing `Reframe`/`ZoomRamp` model. Clips before `timeline.start` (the hook) are untouched. Legacy
    zoom on an owned clip is removed: the timeline is the only camera."""
    events = sorted(timeline.executable, key=lambda e: (e.start, e.end))
    box = face_box if face_box is not None else timeline.face_box
    for clip in edl.clips:
        if clip.timeline_in < timeline.start - _EPS or clip.timeline_in >= timeline.end - _EPS:
            continue
        rf = _clip_reframe(clip, events, box)
        clip.reframe = rf
        clip.zoom = max([rf.zoom_start, *(r.zoom_to for r in rf.ramps)]) if rf is not None else 1.0


def timeline_from_camera_plan(plan: CameraPlan, owner: str = "pinned") -> list[CameraEvent]:
    """The director's own (pinned) camera events as timeline events: they keep their interval and are executable."""
    return [e.model_copy(update={"owner": owner, "status": EventStatus.EXECUTABLE}) for e in plan.events]


def sample_geometry(edl: EDL, times: Iterable[float], width: int = 1080, height: int = 1920,
                    face_box: FaceBox | None = None) -> list[dict]:
    """The crop window the renderer shows at each timeline second, evaluated with the renderer's own math: zoom, anchor,
    the window and (with a face box) the face's position and margins inside the output frame."""
    rows = []
    clips = sorted(edl.clips, key=lambda c: c.timeline_in)
    for t in times:
        clip = next((c for c in clips if c.timeline_in - 1e-9 <= t < c.timeline_out), None)
        if clip is None:
            continue
        rf = reframe_math.resolve_reframe(clip)
        local = t - clip.timeline_in
        if rf is None:
            z, ax, ay = 1.0, BASE_ANCHOR_X, BASE_ANCHOR_Y
        else:
            z = reframe_math.zoom_at(rf, local)
            ax, ay = reframe_math.anchor_at(rf, local)
        row = {"t": round(t, 3), "zoom": z, "anchor_x": ax, "anchor_y": ay,
               "window": reframe_math.crop_window(rf, local, width, height) if rf is not None else (0.0, 0.0, float(width), float(height))}
        if face_box is not None:
            left, top, right, bottom = reframe_math.face_frame(face_box, z, ax, ay)
            row.update(face_left=left, face_top=top, face_right=right, face_bottom=bottom)
        rows.append(row)
    return rows


__all__ = [
    "BASE_ANCHOR_Y", "BEHIND_SUBJECT_MAX_ZOOM_DELTA", "EXECUTABLE_MOVES", "PLANNING_ONLY_MOVES", "CameraSlot", "CameraTimeline",
    "apply_timeline", "behind_subject_compatible", "build_camera_timeline", "legacy_events", "reconcile_slots", "review_mismatches",
    "sample_geometry", "slot_events", "static_claims", "timeline_from_camera_plan", "validate",
]
