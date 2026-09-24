"""Reset-to-base planning grammar: every emphasis is a round trip.

    base -> punch/push -> hold -> reset/release

The camera planner (`direction.camera`) already emits ABSOLUTE, non-accumulating
events (each move targets a zoom level measured from the base framing, and every
emphasis is followed by a reset event). This module reads that plan back as a
per-beat STORY the reviewer can see before anything renders:

    incoming camera state   where the framing is when the beat starts
    treatment camera event  what this beat asks of the camera (and the absolute level)
    release                 how and when it returns

    punch_in    -> reset_to_base
    slow_push   -> slow_pull / reset   (planned as a slow release; the camera plan
                                        executes it as its reset_to_base ease)
    reframe     -> reset_to_base
    replacement -> return_to_base      (the speaker comes back at the base framing)

Planning metadata only: nothing here moves a camera or touches the renderer.
Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from video_edit_agent.direction.camera import (
    BASE_ZOOM,
    CameraEvent,
    CameraMove,
    CameraPlan,
    zoom_trajectory,
)

NONE = "none"
RESET_TO_BASE = "reset_to_base"
SLOW_PULL = "slow_pull"
RETURN_TO_BASE = "return_to_base"

_EPS = 1e-6
_RELEASING = {CameraMove.RESET_TO_BASE, CameraMove.PUNCH_OUT, CameraMove.SLOW_PULL}


class CameraStory(BaseModel):
    incoming_zoom: float = BASE_ZOOM
    incoming_state: str = "base"  # base | held (an earlier emphasis is still on screen)
    event: str = CameraMove.STATIC.value  # the treatment's camera move
    event_zoom_to: float | None = None  # ABSOLUTE level the move reaches (never relative)
    event_at: float | None = None
    peak_zoom: float = BASE_ZOOM
    release: str = NONE  # reset_to_base | slow_pull | return_to_base | none
    release_at: float | None = None
    release_s: float | None = None
    release_zoom: float = BASE_ZOOM
    executed_as: str | None = None  # what the camera plan actually emits for the release
    sequence: list[str] = Field(default_factory=list)  # base -> punch -> hold -> reset, spelled out
    accumulates: bool = False  # always False: zoom never stacks

    def summary(self) -> str:
        if self.event == CameraMove.STATIC.value and self.release == NONE:
            return f"{self.incoming_state} framing, no camera event"
        return " -> ".join(self.sequence)


def _find_release(plan: CameraPlan, beat_start: float) -> CameraEvent | None:
    return next((e for e in plan.events if e.move in _RELEASING and e.beat_start is not None and abs(e.beat_start - beat_start) < 1e-3), None)


def _find_event(plan: CameraPlan, beat_start: float) -> CameraEvent | None:
    return next((e for e in plan.events if e.move not in _RELEASING and e.beat_start is not None and abs(e.beat_start - beat_start) < 1e-3), None)


def camera_story(plan: CameraPlan, *, start: float, end: float, move: CameraMove, replaced: bool = False) -> CameraStory:
    """The camera story of the beat [start, end) whose treatment asked for `move`, read from `plan`
    (the deterministic camera plan the director produced). `replaced`: the speaker is hidden."""
    incoming = round(zoom_trajectory(plan, start - 1e-3), 4) if plan.events else BASE_ZOOM
    held = incoming > BASE_ZOOM + _EPS
    story = CameraStory(incoming_zoom=incoming, incoming_state=f"held at {incoming:.2f}" if held else "base")
    if replaced:
        story.event = "n/a"
        story.release, story.release_at, story.release_zoom = RETURN_TO_BASE, round(end, 3), BASE_ZOOM
        story.sequence = [story.incoming_state, "speaker replaced (voice continues)", f"return_to_base at {end:.2f}s (base framing)"]
        return story
    story.event = move.value
    if move is CameraMove.STATIC:
        return story
    ev = _find_event(plan, start)
    if ev is None:
        story.event = f"{move.value} (dropped by the camera plan)"
        return story
    story.event_zoom_to, story.event_at, story.peak_zoom = ev.zoom_to, ev.start, ev.zoom_to
    rel = _find_release(plan, start)
    story.sequence = [story.incoming_state, f"{move.value} to {ev.zoom_to:.2f} at {ev.start:.2f}s", "hold"]
    if rel is not None:
        _fill_release(story, rel, move)
        story.sequence.append(f"{story.release} to {story.release_zoom:.2f} at {rel.start:.2f}s")
    else:
        story.sequence.append("no release planned")
    return story


def _fill_release(story: CameraStory, rel: CameraEvent, move: CameraMove) -> None:
    story.release = SLOW_PULL if move is CameraMove.SLOW_PUSH else RESET_TO_BASE
    story.release_at, story.release_s, story.release_zoom = rel.start, round(rel.end - rel.start, 3), rel.zoom_to
    story.executed_as = f"{rel.move.value} ease {story.release_s:g}s"


def never_accumulates(plan: CameraPlan) -> bool:
    """True when no event asks for a level above the policy's own maximum, and every zoom-in is
    entered from the base framing (an emphasis is reset before the next one starts)."""
    level = BASE_ZOOM
    for ev in plan.events:
        if ev.zoom_to > plan.policy.max_zoom + _EPS:
            return False
        if ev.move not in _RELEASING and level > BASE_ZOOM + _EPS:
            return False
        level = ev.zoom_to
    return True


__all__ = [
    "RESET_TO_BASE", "RETURN_TO_BASE", "SLOW_PULL", "CameraStory", "camera_story", "never_accumulates",
]
