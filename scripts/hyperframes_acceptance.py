"""V1.1 hardening: real HyperFrames render acceptance attempt (spec section 6).

This script does NOT assume HyperFrames works because the adapter file
exists. It inspects the actual adapter (`motion/hyperframes/adapter.py`),
checks whether a real, working HyperFrames SDK is importable, and -- only if
one genuinely is -- attempts one small real render (a 1080x1920, ~3s title
card) and probes the output. If no real SDK is available it reports
BLOCKED BY ENVIRONMENT with the precise reason, and does NOT silently test
the Simple engine instead and call that "HyperFrames verified".

Due-diligence finding (recorded here so it isn't lost): `pip index versions
hyperframes` resolves to a real PyPI package named "hyperframes" (0.0.1), but
its published metadata describes it as a pandas-like N-dimensional DataFrame
/ labeled-array library (dependencies: pandas, numpy only) by an unrelated
author -- it has no rendering, composition, or `render_from_spec` API of any
kind. Installing it would make `is_available()` return True (a bare
`import hyperframes` succeeds) while `render()` would immediately fail with
an AttributeError, which is exactly the false-positive capability report the
hardening spec forbids. It was deliberately NOT installed.

There is no other package on PyPI, and no vendored/local copy in this repo,
providing an actual "HyperFrames" motion-graphics engine. This is therefore
a real, structural environment limitation, not a bug in this project's
adapter or router code -- both are already correctly architected to degrade
gracefully (see tests/test_motion_router.py).

Usage:
    .venv/Scripts/python.exe scripts/hyperframes_acceptance.py [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from video_edit_agent.core.schemas import AnimationKind, AnimationSpec  # noqa: E402
from video_edit_agent.core.verification_store import record_verified  # noqa: E402
from video_edit_agent.motion.hyperframes import adapter as hyperframes_adapter  # noqa: E402


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "scratch" / "hyperframes_acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Checking for a real, working HyperFrames SDK (not just a same-named package)...")
    available = hyperframes_adapter.is_available()
    print(f"hyperframes_adapter.is_available() -> {available}")

    if not available:
        print()
        print("BLOCKED BY ENVIRONMENT: no `hyperframes` package is installed, and due")
        print("diligence (see this script's module docstring) confirmed the only PyPI")
        print("package with that name is an unrelated N-dimensional DataFrame library")
        print("with no rendering API -- it was deliberately NOT installed, since doing")
        print("so would make is_available() falsely report success.")
        print()
        print("The adapter and motion router are already correctly architected for this:")
        print("  - HyperFramesUnavailable is raised immediately, with a clear message.")
        print("  - motion/router.py falls back to the next engine (see")
        print("    tests/test_motion_router.py::test_hyperframes_request_falls_back_and_is_traceable).")
        return 3

    # Only reached if a real SDK is genuinely installed in this environment.
    spec = AnimationSpec(
        kind=AnimationKind.HOOK_TITLE,
        timeline_start=0.0,
        timeline_end=3.0,
        text="HyperFrames Acceptance",
        engine_hint=None,
    )
    try:
        result_path = hyperframes_adapter.render(spec, out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: HyperFrames reported available but render raised: {e}")
        return 1

    if not result_path.exists() or result_path.stat().st_size == 0:
        print(f"FAILED: render() returned {result_path} but it is missing or empty")
        return 1

    print(f"VERIFIED: HyperFrames produced a real, non-empty output at {result_path}")
    record_verified("hyperframes", f"real render acceptance produced non-empty output at {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
