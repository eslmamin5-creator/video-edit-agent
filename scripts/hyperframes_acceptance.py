"""Baseline Recovery Milestone item 4: real HyperFrames render acceptance
(spec section 6).

This script does NOT assume HyperFrames works because the adapter file
exists. It checks whether Node.js/npm are genuinely available (the real
integration surface -- see `motion/hyperframes/adapter.py`'s module
docstring for the full due-diligence trail on why there is no importable
Python `hyperframes` SDK), and -- only if they are -- performs one real,
non-mocked render through the actual adapter (`hyperframes_adapter.render`,
the same function `motion/router.py` calls) against a small title-card
`AnimationSpec`. It then verifies the output is not just non-empty but
genuinely alpha-transparent (`pix_fmt=yuva444p12le` via ffprobe) and
composites correctly as an overlay through this project's own
`render/composition.py` filter graph -- the two properties this milestone
actually cares about, not just "a file got written".

If Node/npm are missing, it reports BLOCKED BY ENVIRONMENT with the precise
reason and does NOT silently test the Simple engine instead and call that
"HyperFrames verified".

Usage:
    .venv/Scripts/python.exe scripts/hyperframes_acceptance.py [output_dir]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from video_edit_agent.core.schemas import EDL, AnimationKind, AnimationSpec, EDLClip
from video_edit_agent.core.verification_store import record_verified
from video_edit_agent.motion.hyperframes import adapter as hyperframes_adapter
from video_edit_agent.render.composition import Overlay, RenderPlan, build_filter_complex


def _make_base_clip(path: Path, seconds: float = 3.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s=1920x1080:d={seconds:.1f}:r=30",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(seconds),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _ffprobe_pix_fmt(path: Path) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "scratch" / "hyperframes_acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Checking for a real HyperFrames integration surface (Node.js + npm)...")
    available = hyperframes_adapter.is_available()
    print(f"hyperframes_adapter.is_available() -> {available}")

    if not available:
        print()
        print("BLOCKED BY ENVIRONMENT: Node.js and/or npm are not available on PATH (or")
        print("the bundled composition template is missing). The real HyperFrames")
        print("integration surface is its published npm CLI (`npx hyperframes@<pinned>`)")
        print("-- see motion/hyperframes/adapter.py's module docstring for the full")
        print("due-diligence trail on why no Python SDK is used.")
        print()
        print("The adapter and motion router are already correctly architected for this:")
        print("  - HyperFramesUnavailable is raised immediately, with a clear message.")
        print("  - motion/router.py falls back to the next engine (see")
        print("    tests/test_motion_router.py::test_hyperframes_request_falls_back_and_is_traceable).")
        return 3

    spec = AnimationSpec(
        kind=AnimationKind.HOOK_TITLE,
        timeline_start=0.0,
        timeline_end=2.5,
        text="HyperFrames Acceptance",
        subtext="real render + real alpha overlay",
    )

    result_path = out_dir / "hyperframes_acceptance.mov"
    try:
        hyperframes_adapter.render(
            spec, project_root=out_dir, output_path=result_path, fps=30, slot_id="acceptance", offline=False
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: HyperFrames reported available but render raised: {e}")
        return 1

    if not result_path.exists() or result_path.stat().st_size == 0:
        print(f"FAILED: render() returned {result_path} but it is missing or empty")
        return 1
    print(f"render() produced a non-empty file at {result_path} ({result_path.stat().st_size} bytes)")

    try:
        pix_fmt = _ffprobe_pix_fmt(result_path)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: could not ffprobe the render output: {e}")
        return 1
    print(f"ffprobe pix_fmt -> {pix_fmt}")
    if "yuva" not in pix_fmt:
        print(f"FAILED: output is not alpha-transparent (pix_fmt={pix_fmt}); cannot composite as an overlay")
        return 1

    # Real end-to-end proof: the output must composite through this project's
    # OWN filter-graph builder, not a hand-rolled ffmpeg command.
    base_clip = out_dir / "acceptance_base.mp4"
    _make_base_clip(base_clip)
    edl = EDL(
        width=1920, height=1080, fps=30,
        clips=[EDLClip(source_file=str(base_clip), source_in=0.0, source_out=3.0, timeline_in=0.0, timeline_out=3.0)],
    )
    plan = RenderPlan(edl=edl, overlays=[Overlay(path=result_path, start=0.0, end=2.5)])
    inputs, filter_complex, maps = build_filter_complex(plan)
    v_label, a_label = maps.strip("[]").split("][")
    composited = out_dir / "acceptance_composited.mp4"
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{v_label}]", "-map", f"[{a_label}]",
        "-c:v", "libx264", "-c:a", "aac", str(composited),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not composited.exists():
        print(f"FAILED: real composition.py overlay pipeline failed:\n{result.stderr[-2000:]}")
        return 1

    print(f"VERIFIED: HyperFrames produced a real, alpha-transparent, composited output at {composited}")
    record_verified(
        "hyperframes",
        f"real render + alpha-overlay composite through render/composition.py at {composited}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
