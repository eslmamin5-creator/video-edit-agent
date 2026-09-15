"""Assembler-specific QA (spec section 28). Reuses the shared technical QA
primitives (`qa.technical.run_technical_qa`) for the checks that already
apply to any EDL-rendered output (missing/empty file, duration drift,
missing audio track), and adds Assembler-only structural checks: missing
scene references, duplicated scene references, timeline gaps/overlaps,
aspect-ratio mismatch against the requested target, and output codec/
playability via ffprobe.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from video_edit_agent.agents.assembler.schemas import SceneInventoryItem
from video_edit_agent.core.media import MediaError, probe
from video_edit_agent.core.schemas import EDL, QAIssue, QASeverity
from video_edit_agent.qa.technical import run_technical_qa

GAP_TOLERANCE_S = 0.05
OVERLAP_TOLERANCE_S = 0.01


def check_scene_coverage(edl: EDL, expected_items: list[SceneInventoryItem]) -> list[QAIssue]:
    issues: list[QAIssue] = []
    referenced_paths = []
    for clip in edl.clips:
        p = Path(clip.source_file)
        # Muxed silent-audio copies live under cache/<id>_with_silent_audio.mp4;
        # match by the original scene id embedded in the filename when the
        # exact path isn't a literal scene source (spec section 28: missing/
        # duplicated clip detection must survive that offline-only rewrite).
        referenced_paths.append(p.resolve())

    missing = [i for i in expected_items if not any(_refers_to_scene(rp, i) for rp in referenced_paths)]
    for m in missing:
        issues.append(
            QAIssue(
                category="technical", severity=QASeverity.ERROR,
                message=f"Scene '{m.filename}' is missing from the rendered EDL", auto_repairable=False,
            )
        )

    seen: dict[str, int] = {}
    for i in expected_items:
        count = sum(1 for rp in referenced_paths if _refers_to_scene(rp, i))
        if count > 1:
            seen[i.filename] = count
    for filename, count in seen.items():
        issues.append(
            QAIssue(
                category="technical", severity=QASeverity.WARNING,
                message=f"Scene '{filename}' appears {count} times in the rendered EDL", auto_repairable=False,
            )
        )
    return issues


def _refers_to_scene(resolved_path: Path, item: SceneInventoryItem) -> bool:
    if resolved_path == Path(item.path).resolve():
        return True
    return resolved_path.name.startswith(f"{item.id}_")


def check_timeline_continuity(edl: EDL) -> list[QAIssue]:
    issues: list[QAIssue] = []
    ordered = sorted(edl.clips, key=lambda c: c.timeline_in)
    for a, b in pairwise(ordered):
        gap = b.timeline_in - a.timeline_out
        if gap > GAP_TOLERANCE_S:
            issues.append(
                QAIssue(
                    category="technical", severity=QASeverity.ERROR,
                    message=f"Timeline gap of {gap:.2f}s between clips ending at {a.timeline_out:.2f}s and "
                    f"starting at {b.timeline_in:.2f}s",
                    timeline_at=a.timeline_out, auto_repairable=False,
                )
            )
        elif gap < -OVERLAP_TOLERANCE_S:
            # A negative gap is expected -- not an error -- when it matches a
            # real rendered crossfade's overlap region (Phase 2 Finalization
            # spec section 3); only flag it when it exceeds what `b`'s own
            # transition duration accounts for.
            expected_overlap = b.transition_duration_s
            if -gap > expected_overlap + OVERLAP_TOLERANCE_S:
                issues.append(
                    QAIssue(
                        category="technical", severity=QASeverity.ERROR,
                        message=f"Impossible timeline overlap of {-gap:.2f}s between clips at {a.timeline_out:.2f}s",
                        timeline_at=a.timeline_out, auto_repairable=False,
                    )
                )
    return issues


def check_aspect_ratio(output_path: Path, edl: EDL) -> list[QAIssue]:
    if not output_path.exists():
        return []
    try:
        info = probe(output_path)
    except MediaError as exc:
        return [
            QAIssue(
                category="technical", severity=QASeverity.ERROR,
                message=f"Could not probe render output for aspect ratio: {exc}", auto_repairable=False,
            )
        ]
    if info.width != edl.width or info.height != edl.height:
        return [
            QAIssue(
                category="technical", severity=QASeverity.ERROR,
                message=f"Output dimensions {info.width}x{info.height} do not match target "
                f"{edl.width}x{edl.height}",
                auto_repairable=False,
            )
        ]
    return []


def check_playable(output_path: Path) -> list[QAIssue]:
    if not output_path.exists():
        return []
    try:
        probe(output_path)
    except MediaError as exc:
        return [
            QAIssue(
                category="technical", severity=QASeverity.ERROR,
                message=f"Rendered output is not a valid/playable video: {exc}", auto_repairable=False,
            )
        ]
    return []


def run_assembler_qa(output_path: Path, edl: EDL, expected_items: list[SceneInventoryItem]) -> list[QAIssue]:
    issues = list(run_technical_qa(output_path, edl))
    issues.extend(check_scene_coverage(edl, expected_items))
    issues.extend(check_timeline_continuity(edl))
    issues.extend(check_playable(output_path))
    issues.extend(check_aspect_ratio(output_path, edl))
    return issues
