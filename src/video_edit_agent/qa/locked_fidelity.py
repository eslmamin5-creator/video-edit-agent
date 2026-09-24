"""Approved preview vs final render (Phase 1.5.3): visual-recipe fidelity of a locked treatment.

Not codec equality: the checks compare the RECIPE (headline font size / y in the burned ASS, camera zoom + anchor
in the EDL, behind-subject placement in the motion plan) exactly, and representative frames of the approved preview
and the final render by region, with tolerances that absorb encoder noise.
"""
from __future__ import annotations

import json
import math
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.core.media import run
from video_edit_agent.review.locked_treatments import (
    FIDELITY_ERROR_CODE,
    LockedTreatment,
    LockedTreatmentFidelityError,
)

ZOOM_TOL = 0.005
ANCHOR_TOL = 0.005
Y_TOL_PX = 1.0
FRAME_MAD_TOL = 12.0  # mean absolute luma difference (encoder noise is ~4; a different treatment is far more)
TEXT_REGION_MAD_TOL = 16.0


@dataclass
class FidelityReport:
    treatment_id: str
    checks: dict[str, dict] = field(default_factory=dict)

    def record(self, name: str, ok: bool, **detail: object) -> None:
        self.checks[name] = {"ok": bool(ok), **detail}

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks.values())

    @property
    def failures(self) -> list[str]:
        return [f"{name}: {c}" for name, c in self.checks.items() if not c["ok"]]

    def assert_ok(self) -> FidelityReport:
        if not self.ok:
            raise LockedTreatmentFidelityError(self.treatment_id, self.failures)
        return self


# ---- recipe (artifact) checks ----------------------------------------------------------------------------------------


def _headline_event(ass_text: str) -> tuple[float, float] | None:
    """(font size of the Headline style, y of the Headline event)."""
    style = re.search(r"^Style:\s*Headline,[^,]*,(\d+)", ass_text, re.MULTILINE)
    event = next((ln for ln in ass_text.splitlines() if ln.startswith("Dialogue:") and ",Headline," in ln), None)
    if style is None or event is None:
        return None
    y = re.search(r"\\(?:pos|move)\(\s*[-\d.]+\s*,\s*([-\d.]+)", event)
    return float(style.group(1)), float(y.group(1)) if y else float("nan")


def check_headline_recipe(lock: LockedTreatment, ass_text: str, report: FidelityReport) -> None:
    recipe = lock.headline
    found = _headline_event(ass_text)
    report.record("headline_event_present", found is not None)
    if recipe is None or found is None:
        return
    size, y = found
    report.record("headline_font_size", int(size) == recipe.font_px, expected=recipe.font_px, actual=size)
    # the event eases by `slide_px` over the fade-in, so its first \move y is within the slide of the locked y
    report.record("headline_position", math.isnan(y) or abs(y - recipe.y_px) <= recipe.slide_px + Y_TOL_PX, expected=recipe.y_px, actual=y)


def check_camera_recipe(lock: LockedTreatment, edl, report: FidelityReport) -> None:
    from video_edit_agent.direction.camera_timeline import sample_geometry

    recipe = lock.headline
    if recipe is None:
        return
    t1, t2, t3 = recipe.changes[1:]
    hold, after = sample_geometry(edl, [round((t1 + t2) / 2, 3), t3 + 0.05], edl.width, edl.height)[:2]
    report.record("camera_zoom_at_hold", abs(hold["zoom"] - recipe.zoom) <= ZOOM_TOL, expected=recipe.zoom, actual=hold["zoom"])
    report.record(
        "camera_anchor_at_hold",
        abs(hold["anchor_x"] - recipe.anchor_x) <= ANCHOR_TOL and abs(hold["anchor_y"] - recipe.anchor_y) <= ANCHOR_TOL,
        expected=[recipe.anchor_x, recipe.anchor_y], actual=[hold["anchor_x"], hold["anchor_y"]],
    )
    report.record("camera_reset_to_base", abs(after["zoom"] - 1.0) <= ZOOM_TOL, actual=after["zoom"])


def check_behind_recipe(lock: LockedTreatment, motion_plan_path: Path, report: FidelityReport) -> None:
    recipe = lock.behind
    path = Path(motion_plan_path)
    items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    spec = next((it["spec"] for it in items if it["spec"].get("behind_subject") and abs(it["spec"]["timeline_start"] - lock.start) < 0.01), None)
    report.record("behind_spec_present", spec is not None)
    if recipe is None or spec is None:
        return
    extra = spec["extra"]
    report.record("behind_font_size", extra.get("fontPx") == recipe.font_px, expected=recipe.font_px, actual=extra.get("fontPx"))
    report.record(
        "behind_position",
        abs(extra.get("centerX", -1) - recipe.center_x) < 0.05 and abs(extra.get("centerY", -1) - recipe.center_y) < 0.05,
        expected=[recipe.center_x, recipe.center_y], actual=[extra.get("centerX"), extra.get("centerY")],
    )
    report.record("behind_window", abs(spec["timeline_end"] - lock.end) < 0.01, expected=lock.end, actual=spec["timeline_end"])
    report.record("behind_text", spec["text"] == lock.text)
    report.record("behind_style", extra.get("color") == recipe.color and extra.get("outline") == recipe.outline)
    order = list(recipe.layer_order)
    report.record("behind_layer_order", order.index("subject_cutout") > order.index("text_graphic"))


# ---- frame checks ----------------------------------------------------------------------------------------------------


def extract_frame(video: Path, t: float, out_png: Path) -> Path:
    result = run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(video), "-frames:v", "1", str(out_png)], timeout=120)
    if result.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"frame extract failed at {t}s: {result.stderr.strip()[-300:]}")
    return out_png


def region_mad(a_png: Path, b_png: Path, region: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)) -> float:
    """Mean absolute luma difference inside a normalized (x0, y0, x1, y1) region of two frames."""
    import numpy as np
    from PIL import Image

    def luma(path: Path):
        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)

    a, b = luma(a_png), luma(b_png)
    if a.shape != b.shape:
        return float("inf")
    h, w = a.shape
    x0, y0, x1, y1 = int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)
    return float(np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1]).mean())


def _text_region(lock: LockedTreatment, frame: tuple[int, int] = (1080, 1920)) -> tuple[float, float, float, float]:
    if lock.behind is not None:
        x0, y0, x1, y1 = lock.behind.bounds
        return x0 / frame[0], y0 / frame[1], x1 / frame[0], y1 / frame[1]
    if lock.headline is not None and lock.headline.text_bbox is not None:
        return lock.headline.text_bbox
    return 0.0, 0.0, 1.0, 0.35


def check_frames(
    lock: LockedTreatment, preview: Path, final: Path, report: FidelityReport, *,
    fractions: Sequence[float] = (0.35, 0.5, 0.65), final_offset: float = 0.0, workdir: Path | None = None,
) -> None:
    """Preview vs final frames at fractions of the treatment window (equal timeline seconds: the preview clip starts at
    `lock.preview_timeline_start`; `final_offset` = seconds an intro card shifts the final render)."""
    region = _text_region(lock)
    with tempfile.TemporaryDirectory() as tmp:
        work = workdir or Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        for i, frac in enumerate(fractions):
            t = lock.start + (lock.end - lock.start) * frac
            p = extract_frame(preview, t - lock.preview_timeline_start, work / f"prev_{lock.treatment_id}_{i}.png")
            f = extract_frame(final, t + final_offset, work / f"final_{lock.treatment_id}_{i}.png")
            full, text = region_mad(p, f), region_mad(p, f, region)
            report.record(f"frame_full_{frac}", full <= FRAME_MAD_TOL, mad=round(full, 2))
            report.record(f"frame_text_region_{frac}", text <= TEXT_REGION_MAD_TOL, mad=round(text, 2))


def verify_locked_render(
    lock: LockedTreatment, *, ass_path: Path | None = None, edl=None, motion_plan_path: Path | None = None,
    preview: Path | None = None, final: Path | None = None, final_offset: float = 0.0,
) -> FidelityReport:
    """Everything supplied is checked; `report.assert_ok()` raises LOCKED_TREATMENT_FIDELITY_ERROR on any failure."""
    report = FidelityReport(lock.treatment_id)
    if lock.headline is not None:
        if ass_path is not None:
            check_headline_recipe(lock, Path(ass_path).read_text(encoding="utf-8"), report)
        if edl is not None:
            check_camera_recipe(lock, edl, report)
    if lock.behind is not None and motion_plan_path is not None:
        check_behind_recipe(lock, motion_plan_path, report)
    if preview is not None and final is not None:
        check_frames(lock, preview, final, report, final_offset=final_offset)
    return report


__all__ = ["FIDELITY_ERROR_CODE", "FidelityReport", "check_frames", "region_mad", "verify_locked_render"]
