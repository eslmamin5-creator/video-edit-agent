"""Real loudness measurement + normalization planning (Phase 2 Finalization
spec sections 7-8): measures each scene's actual mean audio volume via
ffmpeg's `volumedetect`, and only asks for a `loudnorm` pass on scenes that
measurably drift from a conservative streaming-style target -- scenes
already close to it are left untouched, so intentional dynamics are never
destroyed for a scene that didn't need fixing.

`required` (deviates from target) and `applied` (a real `loudnorm` filter
will run) are tracked separately so the report never claims normalization
happened when it didn't, or was needed when it wasn't (spec section 20's
"do not fake it in the report" standard, extended to loudness).
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from video_edit_agent.agents.assembler.schemas import SceneInventoryItem
from video_edit_agent.core.media import run

TARGET_DB = -16.0
DEVIATION_THRESHOLD_DB = 2.0
# Below this, a scene's audio is effectively digital silence (e.g. a
# synthesized silent track). `loudnorm` measures such input as -inf/NaN
# internally and can poison the encoder with NaN samples, so silence this
# quiet is left untouched rather than "normalized" toward the target.
SILENCE_FLOOR_DB = -60.0

_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


class LoudnessDecision(BaseModel):
    scene_id: str
    mean_volume_db: float | None = None
    target_db: float = TARGET_DB
    required: bool = False
    applied: bool = False
    reason: str = ""


def _measure_mean_volume(path: str) -> float | None:
    result = run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        timeout=120,
    )
    match = _MEAN_VOLUME_RE.search(result.stderr or "")
    if not match:
        return None
    return float(match.group(1))


def plan_loudness(items: list[SceneInventoryItem]) -> list[LoudnessDecision]:
    decisions: list[LoudnessDecision] = []
    for item in items:
        if not item.has_audio:
            decisions.append(
                LoudnessDecision(scene_id=item.id, reason="scene has no audio track; nothing to normalize")
            )
            continue

        mean_db = _measure_mean_volume(item.path)
        if mean_db is None:
            decisions.append(
                LoudnessDecision(scene_id=item.id, reason="could not measure loudness; leaving audio untouched")
            )
            continue

        if mean_db <= SILENCE_FLOOR_DB:
            decisions.append(
                LoudnessDecision(
                    scene_id=item.id,
                    mean_volume_db=mean_db,
                    target_db=TARGET_DB,
                    required=False,
                    applied=False,
                    reason=(
                        f"mean volume {mean_db:.1f}dB is effectively silence "
                        f"(below {SILENCE_FLOOR_DB:.1f}dB floor); leaving audio untouched"
                    ),
                )
            )
            continue

        deviation = abs(mean_db - TARGET_DB)
        required = deviation > DEVIATION_THRESHOLD_DB
        decisions.append(
            LoudnessDecision(
                scene_id=item.id,
                mean_volume_db=mean_db,
                target_db=TARGET_DB,
                required=required,
                applied=required,
                reason=(
                    f"mean volume {mean_db:.1f}dB deviates {deviation:.1f}dB from target "
                    f"{TARGET_DB:.1f}dB; applying conservative loudnorm"
                    if required
                    else f"mean volume {mean_db:.1f}dB is within {DEVIATION_THRESHOLD_DB:.1f}dB "
                    f"of target; no change needed"
                ),
            )
        )
    return decisions
