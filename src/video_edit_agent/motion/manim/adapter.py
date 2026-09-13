"""Manim adapter (spec sections 17, 43) — technical diagrams, data
visualizations, math/scientific animations. Lazy import; graceful degrade if
`manim` isn't installed."""
from __future__ import annotations

import tempfile
from pathlib import Path

from video_edit_agent.core.media import run
from video_edit_agent.core.schemas import AnimationKind, AnimationSpec


class ManimUnavailable(RuntimeError):
    pass


class ManimRenderError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        import manim  # noqa: F401
    except ImportError:
        return False
    return True


_SCENE_TEMPLATE = '''
from manim import *

class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#00000000"
        title = Text({text!r}, font_size=54, color=WHITE)
        self.play(FadeIn(title, shift=UP * 0.3), run_time=0.6)
        self.wait({hold:.2f})
        self.play(FadeOut(title), run_time=0.4)
'''


def render(spec: AnimationSpec, output_dir: Path) -> Path:
    if not is_available():
        raise ManimUnavailable(
            "manim is not installed (pip install video-edit-agent[motion]). The motion router "
            "will use another engine instead."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    hold = max(0.5, spec.timeline_end - spec.timeline_start - 1.0)
    scene_code = _SCENE_TEMPLATE.format(text=spec.text or spec.value or "", hold=hold)

    with tempfile.TemporaryDirectory() as tmp:
        scene_path = Path(tmp) / "scene.py"
        scene_path.write_text(scene_code, encoding="utf-8")
        result = run(
            [
                "python", "-m", "manim", "render", "-ql", "--transparent",
                "-o", "output.mov", str(scene_path), "GeneratedScene",
            ],
            timeout=300,
        )
        if result.returncode != 0:
            raise ManimRenderError(f"manim render failed: {result.stderr.strip()[-2000:]}")

        rendered = next(Path(tmp).rglob("output.mov"), None)
        if rendered is None:
            raise ManimRenderError("manim did not produce an output file")

        dest = output_dir / f"{spec.kind.value}_{int(spec.timeline_start * 1000)}.mov"
        dest.write_bytes(rendered.read_bytes())
        return dest
