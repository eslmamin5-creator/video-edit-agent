"""HyperFrames adapter (spec section 18).

HyperFrames is treated as a first-class OPTIONAL engine: it is never
installed eagerly, and this adapter is the only place that knows anything
about its implementation details. If the `hyperframes` package isn't
installed, `render()` raises `HyperFramesUnavailable` and the motion router
(spec section 17) falls back to another engine — the pipeline never fails
because of this.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import AnimationSpec


class HyperFramesUnavailable(RuntimeError):
    pass


class HyperFramesRenderError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import hyperframes  # noqa: F401
    except ImportError:
        return False
    return True


def render(spec: AnimationSpec, output_dir: Path) -> Path:
    """Receive a structured AnimationSpec, create/render a composition, and
    return a path to a compositable (ideally alpha-channel) output. Source
    files are persisted under `output_dir` for later editing."""
    if not is_available():
        raise HyperFramesUnavailable(
            "hyperframes is not installed. It is an optional engine — the motion router will "
            "use Remotion/Manim/the simple engine instead. Install it explicitly if you want "
            "kinetic-typography-grade HyperFrames output."
        )

    try:
        import hyperframes  # type: ignore
    except ImportError as e:
        raise HyperFramesUnavailable(str(e)) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"{spec.kind.value}_source.hf.json"
    source_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    try:
        # The real hyperframes composition/render API is intentionally kept
        # behind this single call site so a future SDK version only needs a
        # change here, not throughout the motion pipeline.
        result_path = hyperframes.render_from_spec(spec.model_dump(mode="json"), str(output_dir))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 - report cleanly, never crash the render
        raise HyperFramesRenderError(f"HyperFrames render failed: {e}") from e

    return Path(result_path)
