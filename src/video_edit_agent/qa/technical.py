"""Technical QA layer (spec section 33): deterministic, local, no-API-key
checks on the rendered output -- these always run, even fully offline.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.media import probe
from video_edit_agent.core.schemas import EDL, QAIssue, QASeverity

MIN_AUDIO_DB_FLOOR = -60.0
MAX_DURATION_DRIFT_S = 0.5


def check_output_exists(output_path: Path) -> list[QAIssue]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return [
            QAIssue(
                category="technical",
                severity=QASeverity.ERROR,
                message=f"Render output is missing or empty: {output_path}",
                auto_repairable=False,
            )
        ]
    return []


def check_duration_matches_edl(output_path: Path, edl: EDL) -> list[QAIssue]:
    issues: list[QAIssue] = []
    try:
        info = probe(output_path)
    except Exception as exc:  # noqa: BLE001
        return [
            QAIssue(
                category="technical", severity=QASeverity.ERROR,
                message=f"Could not probe render output: {exc}", auto_repairable=False,
            )
        ]

    drift = abs(info.duration - edl.total_duration)
    if drift > MAX_DURATION_DRIFT_S:
        issues.append(
            QAIssue(
                category="technical", severity=QASeverity.WARNING,
                message=f"Rendered duration ({info.duration:.2f}s) drifts from EDL total ({edl.total_duration:.2f}s) "
                f"by {drift:.2f}s",
                auto_repairable=True,
            )
        )
    return issues


def check_has_audio_track(output_path: Path) -> list[QAIssue]:
    try:
        info = probe(output_path)
    except Exception as exc:  # noqa: BLE001
        return [
            QAIssue(
                category="technical", severity=QASeverity.ERROR,
                message=f"Could not probe render output for audio track: {exc}", auto_repairable=False,
            )
        ]

    if not info.has_audio:
        return [
            QAIssue(
                category="technical", severity=QASeverity.ERROR,
                message="Rendered output has no audio track", auto_repairable=False,
            )
        ]
    return []


def run_technical_qa(output_path: Path, edl: EDL) -> list[QAIssue]:
    issues = check_output_exists(output_path)
    if issues:
        return issues  # nothing else can be checked if the file doesn't exist

    issues.extend(check_duration_matches_edl(output_path, edl))
    issues.extend(check_has_audio_track(output_path))
    return issues
