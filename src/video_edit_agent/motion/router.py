"""Motion engine router (spec section 17).

Given an `AnimationSpec`, picks an engine in priority order and always
degrades to the `simple` PIL engine as a last resort so a missing optional
dependency (Node, Manim, HyperFrames) never fails the render (spec section 43).

Per-kind tendencies (spec section 17): kinetic typography / lower-thirds /
CTAs favor Remotion (real React animation, brand-themeable); technical
diagrams/data-viz favor Manim; anything can be forced onto a specific engine
via `AnimationSpec.engine_hint` or a brand's `preferred_engine`.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.schemas import AnimationKind, AnimationSpec, MotionEngine, MotionPlanItem
from video_edit_agent.motion.hyperframes import adapter as hyperframes_adapter
from video_edit_agent.motion.manim import adapter as manim_adapter
from video_edit_agent.motion.remotion import adapter as remotion_adapter
from video_edit_agent.motion.simple import engine as simple_engine

_DIAGRAM_KINDS = {AnimationKind.DIAGRAM, AnimationKind.DATA_VIZ}
_REMOTION_PREFERRED_KINDS = {
    AnimationKind.HOOK_TITLE,
    AnimationKind.LOWER_THIRD,
    AnimationKind.STAT_COUNTER,
    AnimationKind.QUOTE,
    AnimationKind.COMPARISON,
    AnimationKind.FEATURE_CARD,
    AnimationKind.CTA,
    AnimationKind.LOGO_REVEAL,
    AnimationKind.METRIC_HIGHLIGHT,
    AnimationKind.PRODUCT_CALLOUT,
    AnimationKind.TIMELINE_GRAPHIC,
}


def _default_priority(kind: AnimationKind) -> list[MotionEngine]:
    if kind in _DIAGRAM_KINDS:
        return [MotionEngine.MANIM, MotionEngine.HYPERFRAMES, MotionEngine.REMOTION, MotionEngine.SIMPLE]
    if kind in _REMOTION_PREFERRED_KINDS:
        return [MotionEngine.REMOTION, MotionEngine.HYPERFRAMES, MotionEngine.SIMPLE]
    return [MotionEngine.SIMPLE, MotionEngine.REMOTION, MotionEngine.HYPERFRAMES]


def _engine_priority(spec: AnimationSpec, brand: Brand | None) -> list[MotionEngine]:
    priority = _default_priority(spec.kind)
    preferred = brand.motion.preferred_engine if brand and brand.motion else None
    if spec.engine_hint:
        priority = [spec.engine_hint] + [e for e in priority if e != spec.engine_hint]
    elif preferred:
        try:
            preferred_engine = MotionEngine(preferred)
        except ValueError:
            preferred_engine = None
        if preferred_engine:
            priority = [preferred_engine] + [e for e in priority if e != preferred_engine]
    if MotionEngine.SIMPLE not in priority:
        priority.append(MotionEngine.SIMPLE)
    return priority


def render_motion(
    spec: AnimationSpec,
    project_root: Path,
    output_dir: Path,
    brand: Brand | None = None,
    fps: int = 30,
    slot_id: str = "slot",
) -> MotionPlanItem:
    """Try each engine in priority order; return a MotionPlanItem recording
    which engine actually produced output (or the last error if the `simple`
    engine itself somehow failed, which should not normally happen)."""
    fallback_log: list[str] = []

    for engine in _engine_priority(spec, brand):
        try:
            if engine == MotionEngine.REMOTION:
                out_path = output_dir / f"{slot_id}_remotion.webm"
                result = remotion_adapter.render(spec, project_root, out_path, fps=fps, slot_id=slot_id)
            elif engine == MotionEngine.MANIM:
                result = manim_adapter.render(spec, output_dir)
            elif engine == MotionEngine.HYPERFRAMES:
                result = hyperframes_adapter.render(spec, output_dir)
            else:
                out_path = output_dir / f"{slot_id}_simple.png"
                result = simple_engine.render_animation(spec, out_path)

            return MotionPlanItem(
                spec=spec, engine_used=engine, output_path=str(result), fallback_log=fallback_log
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the next engine
            fallback_log.append(f"{engine.value}: {exc}")
            continue

    return MotionPlanItem(
        spec=spec, engine_used=None, output_path=None, error=fallback_log[-1] if fallback_log else None,
        fallback_log=fallback_log,
    )
