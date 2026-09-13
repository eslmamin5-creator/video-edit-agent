"""V1.1 hardening (spec section 7): motion engine fallback must be real and
traceable. When a requested engine is unavailable/fails, `render_motion()`
must fall through to a working engine, the project must still complete, and
the substitution must never be silent -- `MotionPlanItem.fallback_log` must
record exactly which engines were tried and rejected before the one that
actually produced output.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import AnimationKind, AnimationSpec, MotionEngine
from video_edit_agent.motion.router import render_motion


def test_hyperframes_request_falls_back_and_is_traceable(tmp_path: Path):
    # HyperFrames is requested explicitly via engine_hint but has no real,
    # working SDK installed in this environment (see hardening due-diligence:
    # the PyPI package named "hyperframes" is an unrelated dataframe library,
    # not a motion-graphics engine) -- so this must fail and fall through.
    spec = AnimationSpec(
        kind=AnimationKind.BOX,
        timeline_start=0.0,
        timeline_end=2.0,
        text="MARKET",
        engine_hint=MotionEngine.HYPERFRAMES,
    )

    item = render_motion(spec, project_root=tmp_path, output_dir=tmp_path / "out", slot_id="fallback_test")

    # The project must still complete -- some engine must have produced output.
    assert item.engine_used is not None
    assert item.engine_used != MotionEngine.HYPERFRAMES
    assert item.output_path is not None
    assert Path(item.output_path).exists()
    assert Path(item.output_path).stat().st_size > 0

    # No silent substitution: the rejected attempt must be traceable.
    assert len(item.fallback_log) >= 1
    assert any(entry.startswith("hyperframes:") for entry in item.fallback_log)


def test_engine_actually_used_matches_priority_after_hyperframes_removed(tmp_path: Path):
    # BOX's default priority is [SIMPLE, REMOTION, HYPERFRAMES]; forcing the
    # hint to HYPERFRAMES reorders it to [HYPERFRAMES, SIMPLE, REMOTION], so
    # once HyperFrames is rejected the very next engine tried is SIMPLE (the
    # guaranteed, dependency-free fallback per spec section 43) -- confirm the
    # router didn't skip straight to a later engine without trying it.
    spec = AnimationSpec(
        kind=AnimationKind.BOX,
        timeline_start=0.0,
        timeline_end=2.0,
        text="MARKET",
        engine_hint=MotionEngine.HYPERFRAMES,
    )

    item = render_motion(spec, project_root=tmp_path, output_dir=tmp_path / "out2", slot_id="fallback_test2")

    assert item.engine_used == MotionEngine.SIMPLE
    assert item.fallback_log == [entry for entry in item.fallback_log if entry.startswith("hyperframes:")]
