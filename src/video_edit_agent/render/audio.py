"""Audio smoothing helpers (spec section 11): short fades around cuts to
avoid clicks/pops, plus optional loudness normalization."""
from __future__ import annotations


def afade_filter(index: int, fade_in_ms: int, fade_out_ms: int, clip_duration: float) -> str:
    """Build an `afade` filter chain for one clip's audio stream."""
    parts = []
    if fade_in_ms > 0:
        parts.append(f"afade=t=in:st=0:d={fade_in_ms / 1000:.3f}")
    if fade_out_ms > 0:
        start = max(0.0, clip_duration - fade_out_ms / 1000)
        parts.append(f"afade=t=out:st={start:.3f}:d={fade_out_ms / 1000:.3f}")
    return ",".join(parts)


def loudnorm_filter(target_lufs: float = -16.0) -> str:
    """One-pass loudnorm — good enough for short-form social content; a
    proper two-pass measure/apply flow can be added later without changing
    the EDL/render contract."""
    return f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
