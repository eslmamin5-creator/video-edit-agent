"""Mix priority and simple SFX gain rules. Not a DAW.

    voice > essential content audio > music > SFX

SFX never reduces the intelligibility of speech: their nominal level sits well
below the voice, they get short fades so they never click, and music may be
ducked under them. The final loudness normalization stays the existing
`render.audio.loudnorm_filter` pass; nothing here replaces it.

Nothing here is brand- or project-specific.
"""
from __future__ import annotations

from video_edit_agent.sound.intent import SoundIntent
from video_edit_agent.sound.profile import SoundProfile

# Highest priority first.
MIX_PRIORITY = ("voice", "content_audio", "music", "sfx")
VOICE_GAIN_DB = 0.0
MIN_HEADROOM_DB = 12.0  # an SFX sits at least this far below the voice
FADE_MS = 25
DUCK_MUSIC_DB = -3.0

# Relative to the profile's nominal SFX level.
_INTENT_OFFSET_DB = {
    SoundIntent.SUBTLE_MOTION: -3.0,
    SoundIntent.TRANSITION: 0.0,
    SoundIntent.ACCENT: -2.0,
    SoundIntent.IMPACT: 0.0,
}


def priority(source: str) -> int:
    """0 is the highest priority (voice)."""
    return MIX_PRIORITY.index(source)


def sfx_gain_db(profile: SoundProfile, intent: SoundIntent) -> float:
    """SFX level in dB for `intent` under `profile`, never closer to the voice than the headroom."""
    level = profile.gain_db + _INTENT_OFFSET_DB.get(intent, 0.0)
    return round(min(level, VOICE_GAIN_DB - MIN_HEADROOM_DB), 1)


def voice_protected(gains_db: list[float], voice_gain_db: float = VOICE_GAIN_DB) -> bool:
    """True when every SFX gain leaves the voice its headroom."""
    return all(g <= voice_gain_db - MIN_HEADROOM_DB for g in gains_db)


def sfx_filter(index: int, *, start_s: float, duration_s: float, gain_db: float, label: str) -> str:
    """The ffmpeg filter that trims an SFX input, fades it in/out shortly, applies its gain and
    places it at `start_s` on the timeline, ready for `amix` (ffmpeg args are built by the caller
    through `core.media.run`, never a shell string)."""
    fade = min(FADE_MS / 1000.0, duration_s / 2)
    delay_ms = max(0, round(start_s * 1000))
    return (
        f"[{index}:a]atrim=0:{duration_s:.3f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:d={fade:.3f},afade=t=out:st={max(0.0, duration_s - fade):.3f}:d={fade:.3f},"
        f"volume={gain_db:.1f}dB,adelay={delay_ms}|{delay_ms}[{label}]"
    )


__all__ = [
    "DUCK_MUSIC_DB", "FADE_MS", "MIN_HEADROOM_DB", "MIX_PRIORITY", "VOICE_GAIN_DB", "priority", "sfx_filter",
    "sfx_gain_db", "voice_protected",
]
