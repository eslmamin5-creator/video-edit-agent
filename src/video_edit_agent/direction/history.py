"""Treatment history and the SOFT visual-fatigue governor.

The director plans beat by beat, but a good edit is judged across beats: what the
viewer has just seen decides whether one more punch-in helps or wearies. This
module remembers what the plan has already done and turns it into a `FatigueReading`
for the next beat:

    previous treatment / camera state           what the viewer saw last
    seconds since a meaningful visual change    how long the picture has been "the same"
    seconds the speaker has stayed unchanged    framing/replacement of the speaker itself
    recent repetition                           how often this treatment was used lately
    previous beat used strong emphasis          variation just happened
    a meaningful alternative exists             something worth switching to

The governor is SOFT and advisory. It composes a `variation_score` from several
signals and only raises `variation_desirable`; it never forces a punch, a
replacement, a graphic or a transition, and there is NO time threshold anywhere:

* elapsed time contributes a ramp that is capped BELOW the trigger, so time alone
  can never raise the flag; a semantic reason (a beat that means something) is
  structurally required,
* the flag additionally needs a meaningful alternative to exist; without one the
  reading says so and the shot stays static however long it has run,
* speaker/body motion, a recent strong emphasis and a recently repeated
  treatment all LOWER the desire for variation.

A long, meaningful static shot may therefore exceed any reference duration.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from video_edit_agent.direction.camera import CameraMove
from video_edit_agent.direction.vocabulary import STRONG_PRIMARY, SpeakerTreatment

VARIATION_TRIGGER = 0.5  # the score at which variation becomes "desirable"
TIME_CAP = 0.35  # what elapsed time can contribute at most (< VARIATION_TRIGGER)
NON_SEMANTIC_CAP = 0.45  # everything that is not about meaning, together (< VARIATION_TRIGGER)
SEMANTIC_WEIGHT = 0.35  # a beat that means something (confident claim / contrast / payoff / steps / shift)
FATIGUE_RAMP_S = 30.0  # the (soft) scale of the time ramp: it never steps, it only leans
REPEATED_STATIC_WEIGHT = 0.1  # per recent beat that was also a plain static speaker shot (capped)
MOTION_RELIEF = 0.25  # what a visibly moving speaker takes off the score (scaled by 0..1 motion)
STRONG_RELIEF = 0.3  # the previous beat was strong: variation just happened
REPEAT_RELIEF = 0.3  # the candidate treatment was used again very recently
RECENT_BEATS = 3  # "recently" for repetition, in beats (not seconds)

_CALM = frozenset({
    SpeakerTreatment.SPEAKER_STATIC.value, SpeakerTreatment.HOLD.value,
    SpeakerTreatment.RESET_TO_BASE.value, SpeakerTreatment.PUNCH_OUT.value,
})
_CAMERA_ZOOMS = frozenset({CameraMove.PUNCH_IN.value, CameraMove.SLOW_PUSH.value, CameraMove.REFRAME_LEFT.value, CameraMove.REFRAME_RIGHT.value})


class HistoryEntry(BaseModel):
    start: float
    end: float
    treatment: str
    camera: str = CameraMove.STATIC.value
    speaker_visible: bool = True
    strong: bool = False  # the beat used a strong primary treatment or a camera emphasis
    eventful: bool = False  # something visible happened (not a plain static speaker shot)
    pinned: bool = False  # the reviewer chose it


class FatigueReading(BaseModel):
    previous_treatment: str | None = None
    previous_camera: str | None = None
    seconds_since_visual_change: float = 0.0
    speaker_unchanged_s: float = 0.0
    repetition_count: int = 0  # recent beats that used the candidate treatment
    recent_static_run: int = 0  # consecutive most-recent plain static speaker beats
    previous_used_strong_emphasis: bool = False
    alternative_available: bool = False
    semantic_reason: bool = False
    variation_score: float = 0.0
    variation_desirable: bool = False
    variation_reason: str = ""
    signals: list[str] = Field(default_factory=list)


def is_eventful(treatment: str, camera: str, speaker_visible: bool) -> bool:
    return treatment not in _CALM or camera in _CAMERA_ZOOMS or not speaker_visible


class TreatmentHistory:
    """What the plan has done so far, in time order. Feed it every decided beat with `record`."""

    def __init__(self, origin: float = 0.0) -> None:
        self.origin = origin
        self.entries: list[HistoryEntry] = []

    def record(self, start: float, end: float, treatment: str, camera: str = CameraMove.STATIC.value,
               speaker_visible: bool = True, *, pinned: bool = False) -> HistoryEntry:
        entry = HistoryEntry(
            start=start, end=end, treatment=treatment, camera=camera, speaker_visible=speaker_visible,
            strong=treatment in STRONG_PRIMARY or camera in _CAMERA_ZOOMS,
            eventful=is_eventful(treatment, camera, speaker_visible), pinned=pinned,
        )
        self.entries.append(entry)
        return entry

    # -- facts about the past -------------------------------------------------------------
    @property
    def previous(self) -> HistoryEntry | None:
        return self.entries[-1] if self.entries else None

    def repetitions(self, treatment: str) -> int:
        return sum(1 for e in self.entries[-RECENT_BEATS:] if e.treatment == treatment)

    def seconds_since_change(self, now: float) -> float:
        """Time since the picture last visibly changed: the start of the last eventful beat, or the end
        of a beat that replaced the speaker (the speaker coming back is a change too)."""
        moments = [self.origin]
        for e in self.entries:
            if e.eventful:
                moments.append(e.start if e.speaker_visible else e.end)
        return round(max(0.0, now - max(moments)), 3)

    def speaker_unchanged_s(self, now: float) -> float:
        """Time since the SPEAKER's own framing last changed (a camera move, or a replacement ending)."""
        moments = [self.origin]
        for e in self.entries:
            if e.camera in _CAMERA_ZOOMS:
                moments.append(e.start)
            if not e.speaker_visible:
                moments.append(e.end)
        return round(max(0.0, now - max(moments)), 3)

    def static_run(self) -> int:
        n = 0
        for e in reversed(self.entries):
            if e.eventful:
                break
            n += 1
        return n

    # -- the soft governor ----------------------------------------------------------------
    def read(
        self, now: float, candidate: str, *, semantic_reason: bool = False, alternative_available: bool = False,
        speaker_motion: float = 0.0,
    ) -> FatigueReading:
        """Everything the director should know before deciding the beat starting at `now` (whose
        candidate treatment is `candidate`). Advisory only."""
        prev = self.previous
        since = self.seconds_since_change(now)
        unchanged = self.speaker_unchanged_s(now)
        repeats = self.repetitions(candidate)
        reading = FatigueReading(
            previous_treatment=prev.treatment if prev else None, previous_camera=prev.camera if prev else None,
            seconds_since_visual_change=since, speaker_unchanged_s=unchanged, repetition_count=repeats,
            recent_static_run=self.static_run(), previous_used_strong_emphasis=bool(prev and prev.strong),
            alternative_available=alternative_available, semantic_reason=semantic_reason,
        )
        signals: list[str] = []
        time_part = round(TIME_CAP * min(1.0, max(since, unchanged) / FATIGUE_RAMP_S), 3)
        quiet = round(min(0.2, REPEATED_STATIC_WEIGHT * max(0, reading.recent_static_run - 1)), 3)
        motion = max(0.0, min(1.0, speaker_motion))
        non_semantic = min(NON_SEMANTIC_CAP, time_part + quiet)
        if time_part > 0:
            signals.append(f"the picture has been the same for {since:.0f}s (a soft lean, never a trigger)")
        if quiet > 0:
            signals.append(f"{reading.recent_static_run} plain speaker beats in a row")
        score = non_semantic
        if semantic_reason:
            score += SEMANTIC_WEIGHT
            signals.append("this beat means something (confident semantic type)")
        if motion:
            score -= MOTION_RELIEF * motion
            signals.append("the speaker is visibly moving (natural variation already)")
        if reading.previous_used_strong_emphasis:
            score -= STRONG_RELIEF
            signals.append("the previous beat already used strong emphasis")
        if repeats:
            score -= REPEAT_RELIEF
            signals.append(f"'{candidate}' was used {repeats}x in the last {RECENT_BEATS} beats")
        reading.variation_score = round(max(0.0, score), 3)
        reading.signals = signals
        reading.variation_desirable = reading.variation_score >= VARIATION_TRIGGER and semantic_reason and alternative_available
        if reading.variation_desirable:
            reading.variation_reason = "variation desirable: " + "; ".join(signals)
        elif reading.variation_score >= VARIATION_TRIGGER and not alternative_available:
            reading.variation_reason = "a fresh look would help, but there is no meaningful alternative: the speaker stays"
        elif not semantic_reason:
            reading.variation_reason = ("no variation needed: nothing on this beat calls for a change"
                                        + (" (long unchanged stretches are fine when the speaker is the best visual)" if since > 0 else ""))
        else:
            reading.variation_reason = "no variation needed: " + ("; ".join(signals) if signals else "the current treatment is enough")
        return reading


__all__ = [
    "FATIGUE_RAMP_S", "NON_SEMANTIC_CAP", "RECENT_BEATS", "TIME_CAP", "VARIATION_TRIGGER", "FatigueReading",
    "HistoryEntry", "TreatmentHistory", "is_eventful",
]
