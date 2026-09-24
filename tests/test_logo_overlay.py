"""Baseline Recovery Milestone item 5: brand logo composition. A brand's
logo image, dropped into `brands/<name>/logos/` per `examples/README.md`,
must actually reach the render plan as a persistent top-right watermark --
not just sit in a folder nobody reads."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.pipeline import _brand_logo_overlay
from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.render.composition import Overlay, RenderPlan, build_filter_complex


def _edl(width: int = 1080, height: int = 1920, duration: float = 5.0) -> EDL:
    return EDL(
        version=1, fps=30.0, width=width, height=height,
        clips=[EDLClip(source_file="bg.mp4", source_in=0.0, source_out=duration, timeline_in=0.0, timeline_out=duration)],
    )


def test_brand_logo_overlay_spans_the_full_timeline_top_right():
    edl = _edl()
    brand = Brand(name="acme")
    overlay = _brand_logo_overlay(Path("logo.png"), brand, edl)

    assert overlay.path == Path("logo.png")
    assert overlay.start == 0.0
    assert overlay.end == edl.total_duration
    assert overlay.x == f"W-w-{round(0.04 * edl.width)}"
    assert overlay.scale_width == round(0.15 * edl.width)


def test_brand_logo_overlay_respects_brand_safe_zone_top_margin():
    edl = _edl()
    brand = Brand(name="acme")
    brand.safe_zones["top"] = 0.1
    overlay = _brand_logo_overlay(Path("logo.png"), brand, edl)
    assert overlay.y == str(round(0.1 * edl.height))


def test_scale_width_overlay_produces_a_scale_filter_stage():
    edl = _edl()
    plan = RenderPlan(edl=edl, overlays=[Overlay(path=Path("logo.png"), start=0.0, end=5.0, scale_width=162)])
    _, filters, _ = build_filter_complex(plan)
    assert "scale=162:-1" in filters
    assert filters.count("overlay=x=") == 1


def test_scale_width_is_ignored_when_scale_to_canvas_is_set():
    edl = _edl()
    plan = RenderPlan(
        edl=edl,
        overlays=[Overlay(path=Path("bg.png"), start=0.0, end=5.0, scale_to_canvas=True, scale_width=162)],
    )
    _, filters, _ = build_filter_complex(plan)
    assert "scale=162:-1" not in filters
    assert f"scale={edl.width}:{edl.height}:force_original_aspect_ratio=increase" in filters


def test_webm_overlay_input_forces_vp8_decode_to_preserve_alpha():
    """A plain `-i file.webm` silently drops the alpha plane and decodes as
    opaque yuv420p. Remotion's motion-graphic overlays (HOOK_TITLE/CTA) are
    alpha-transparent VP8 webm (motion/remotion/adapter.py pins
    `--codec=vp8 --pixel-format=yuva420p`); without a matching forced VP8
    decode here they silently composite as a fully opaque black rectangle
    covering the whole canvas, hiding the speaker underneath for the
    overlay's entire duration (Baseline Recovery Milestone regression,
    caught only by visual inspection of a full acceptance render, not by
    any existing unit test)."""
    edl = _edl()
    plan = RenderPlan(
        edl=edl,
        overlays=[Overlay(path=Path("motion0_remotion.webm"), start=0.0, end=3.0)],
    )
    inputs, _, _ = build_filter_complex(plan)
    webm_idx = inputs.index("-i", inputs.index("bg.mp4") + 1)
    assert inputs[webm_idx - 2:webm_idx] == ["-c:v", "libvpx"]


def test_non_webm_overlay_input_does_not_force_a_video_decoder():
    edl = _edl()
    plan = RenderPlan(edl=edl, overlays=[Overlay(path=Path("logo.png"), start=0.0, end=5.0)])
    inputs, _, _ = build_filter_complex(plan)
    assert "-c:v" not in inputs
