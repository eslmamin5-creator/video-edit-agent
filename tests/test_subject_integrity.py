"""Behind-subject compositing integrity.

The text may disappear behind the speaker; the speaker must never disappear behind
the text. These tests build synthetic scenes (a face ellipse with dark hair on a
uniform backdrop, a bright test graphic) and hold the layer stack, the subject-RGB
invariant, the hair-safe matte and the quality gate to that contract.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.test_behind_subject_text import (
    W,
    _clip,
    _compose,
    _edl,
    _fakes,
    _plans,
    _spec,
    loader,
)
from video_edit_agent.motion import behind_subject as bs
from video_edit_agent.subject import refine
from video_edit_agent.subject.integrity import (
    MatteGatePolicy,
    assess_matte,
    check_integrity,
    composite_behind,
    feather_width,
    outward_growth_px,
)

SH, SW = 480, 400  # small enough to be fast, tall enough that the hair band (1.2% of the height) is a few px
BACKDROP = (200, 205, 210)
SKIN = (205, 150, 120)
HAIR = (25, 20, 18)


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame, true silhouette, hair fringe): a uniform backdrop, a face ellipse with a
    dark hair cap that reaches a few px past it, and shoulders. The fringe is the hair
    outside the face: what a low-resolution segmenter tends to miss."""
    import cv2

    frame = np.zeros((SH, SW, 3), np.uint8)
    frame[:] = BACKDROP
    face = np.zeros((SH, SW), np.uint8)
    cv2.ellipse(face, (200, 200), (80, 104), 0, 0, 360, 1, -1)
    cap = np.zeros((SH, SW), np.uint8)
    cv2.ellipse(cap, (200, 198), (84, 106), 0, 180, 360, 1, -1)  # the hair above the hairline
    cap[200:, :] = 0
    body = np.zeros((SH, SW), np.uint8)
    body[300:, 60:340] = 1
    silhouette = (face | cap | body) > 0
    frame[silhouette] = SKIN
    frame[cap > 0] = HAIR
    frame[body > 0] = (60, 80, 140)
    return frame, silhouette, (cap > 0) & ~(face > 0)


def _soft_mask(silhouette: np.ndarray, miss: np.ndarray | None = None) -> np.ndarray:
    """What a low-resolution segmenter returns: the silhouette, blurred, and (optionally)
    missing the hair, whose confidence stays below 0.5."""
    import cv2

    m = silhouette.astype(np.float32)
    if miss is not None:
        m = np.where(miss, 0.30, m)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 3.0), 0.0, 1.0).astype(np.float32)


def _graphic(box=(80, 120, 320, 260), colour=(255, 40, 200)) -> tuple[np.ndarray, np.ndarray]:
    """A bright, fully opaque test graphic (rgb, alpha)."""
    rgb = np.zeros((SH, SW, 3), np.uint8)
    alpha = np.zeros((SH, SW), np.float32)
    x0, y0, x1, y1 = box
    rgb[y0:y1, x0:x1] = colour
    alpha[y0:y1, x0:x1] = 1.0
    return rgb, alpha


# -- 1 / 2. the subject composites above the text, with its original pixels ------


def test_the_subject_composites_above_a_bright_graphic():
    frame, silhouette, _ = _scene()
    g_rgb, g_a = _graphic()
    alpha = silhouette.astype(np.float32)
    out = composite_behind(frame, g_rgb, g_a, frame, alpha)
    assert np.array_equal(out[silhouette], frame[silhouette])  # the subject is untouched
    shown = (g_a > 0) & ~silhouette
    assert np.array_equal(out[shown], g_rgb[shown])  # the graphic shows everywhere else
    assert not np.array_equal(out[silhouette & (g_a > 0)], g_rgb[silhouette & (g_a > 0)])  # ... and not over the subject


def test_subject_rgb_is_the_original_where_alpha_is_opaque():
    frame, silhouette, _ = _scene()
    g_rgb, g_a = _graphic()
    alpha = _soft_mask(silhouette)
    out = composite_behind(frame, g_rgb, g_a, frame, alpha)
    report = check_integrity(frame, alpha, g_a, out)
    assert report.protected_px > 0 and report.graphic_px_in_protected > 0  # the graphic really is behind the subject
    assert report.passed and report.leaked_px == 0
    assert report.rgb_max_error <= (1 - 0.98) * 255  # only the graphic a 98%-opaque edge pixel lets through


def test_the_subject_is_rebuilt_from_the_source_never_from_the_background():
    frame, silhouette, _ = _scene()
    g_rgb, g_a = _graphic()
    alpha = silhouette.astype(np.float32)
    tinted = np.clip(frame.astype(np.int16) - 40, 0, 255).astype(np.uint8)  # a graded / darkened background layer
    out = composite_behind(tinted, g_rgb, g_a, frame, alpha)
    assert np.array_equal(out[silhouette], frame[silhouette])


# -- 3. text cannot punch a hole ---------------------------------------------


def test_a_graphic_passing_behind_the_subject_never_punches_a_hole():
    frame, silhouette, _ = _scene()
    alpha = _soft_mask(silhouette)
    for x in range(-120, SW, 40):  # the bright graphic sweeps across the whole silhouette
        g_rgb, g_a = _graphic(box=(max(0, x), 60, max(1, x + 120), 400))
        out = composite_behind(frame, g_rgb, g_a, frame, alpha)
        report = check_integrity(frame, alpha, g_a, out)
        assert report.leaked_px == 0 and report.holes_px == 0
        inside = alpha >= 0.98
        diff = np.abs(out[inside].astype(np.int16) - frame[inside].astype(np.int16))
        assert diff.max() <= (1 - 0.98) * 255  # nothing but the 2% a near-opaque edge pixel lets through


def test_the_integrity_check_catches_a_graphic_drawn_over_the_subject():
    frame, silhouette, _ = _scene()
    g_rgb, g_a = _graphic()
    alpha = silhouette.astype(np.float32)
    wrong = frame.astype(np.float32) * (1 - g_a[..., None]) + g_rgb.astype(np.float32) * g_a[..., None]  # graphic on top
    report = check_integrity(frame, alpha, g_a, np.rint(wrong).astype(np.uint8))
    assert not report.passed and report.leaked_px > 0 and report.rgb_max_error > 100


def test_the_integrity_check_reports_a_hole_in_the_matte():
    frame, silhouette, _ = _scene()
    alpha = silhouette.astype(np.float32)
    alpha[160:200, 180:220] = 0.0  # a hole in the middle of the head
    assert check_integrity(frame, alpha, np.zeros_like(alpha), frame).holes_px == 40 * 40


# -- 4. head / hair safety, no halo ------------------------------------------


def test_hair_the_model_missed_is_recovered_and_opaque():
    frame, silhouette, hair = _scene()
    raw = _soft_mask(silhouette, miss=hair)  # the segmenter gives the hair only 0.30
    assert raw[hair].mean() < 0.5
    alpha, report = refine.refine_matte(raw, frame)
    assert report.evidence_used and report.added_area > 0
    assert (alpha[hair] >= 0.98).mean() > 0.9  # the fringe is subject, opaque
    assert (alpha[silhouette] >= 0.98).mean() > 0.97


def test_the_refined_matte_has_no_large_halo():
    frame, silhouette, hair = _scene()
    alpha, _ = refine.refine_matte(_soft_mask(silhouette, miss=hair), frame)
    growth = outward_growth_px(alpha, silhouette)
    assert growth <= 4.0  # a feather, not a halo
    band = refine.DEFAULT_PARAMS.band_px(SH)
    assert growth < band


def test_an_accurate_matte_on_a_uniform_backdrop_does_not_grow():
    frame, silhouette, _ = _scene()
    exact, report = refine.refine_matte(silhouette.astype(np.float32), frame)
    assert report.added_area == 0  # nothing differs from the backdrop outside the core
    assert outward_growth_px(exact, silhouette) <= 2.0  # just the feather
    alpha, report = refine.refine_matte(_soft_mask(silhouette), frame)  # a blurred mask rounds the corners
    assert report.added_frac < 0.02
    assert outward_growth_px(alpha, silhouette) <= 3.0
    assert 0.5 <= feather_width(alpha) <= 8.0


def test_the_matte_never_shrinks_the_subject():
    frame, silhouette, hair = _scene()
    raw = _soft_mask(silhouette, miss=hair)
    alpha, _ = refine.refine_matte(raw, frame)
    assert (alpha >= np.where(raw > 0.5, 1.0, 0.0) - 1e-6).all()  # everything the model was sure of stays opaque


# -- 5. determinism ----------------------------------------------------------


def test_the_matte_and_its_temporal_stabilisation_are_deterministic():
    frame, silhouette, hair = _scene()
    rng = np.random.default_rng(7)
    masks = [np.clip(_soft_mask(silhouette, miss=hair) + rng.normal(0, 0.05, (SH, SW)), 0, 1).astype(np.float32) for _ in range(5)]
    a1 = [refine.refine_matte(m, frame)[0] for m in refine.stabilize_temporal(masks)]
    a2 = [refine.refine_matte(m, frame)[0] for m in refine.stabilize_temporal(masks)]
    assert all(np.array_equal(x, y) for x, y in zip(a1, a2, strict=True))
    r1, r2 = refine.refine_matte(masks[0], frame)[1], refine.refine_matte(masks[0], frame)[1]
    assert r1 == r2


def test_temporal_stabilisation_calms_flicker_in_the_refined_alpha():
    _, silhouette, _ = _scene()
    rng = np.random.default_rng(3)
    base = _soft_mask(silhouette)
    masks = [np.clip(base + rng.normal(0, 0.15, base.shape), 0, 1).astype(np.float32) for _ in range(6)]
    raw_iou = refine.mask_stability(masks)[0]
    stable_iou = refine.mask_stability(refine.stabilize_temporal(masks))[0]
    assert stable_iou >= raw_iou


# -- 6. a low-quality matte rejects behind-subject ----------------------------


def _report(**kw) -> refine.MatteReport:
    base = {"core_area": 10000, "added_area": 0, "evidence_pixels": 0, "soft_area": 500, "band_px": 23,
            "feather_sigma": 1.5, "evidence_used": True}
    return refine.MatteReport(**{**base, **kw})


def test_a_reliable_matte_passes_the_gate():
    frame, silhouette, hair = _scene()
    _, report = refine.refine_matte(_soft_mask(silhouette, miss=hair), frame)
    ok, detail = assess_matte([report] * 3)
    assert ok, detail


@pytest.mark.parametrize("bad", [
    _report(added_area=4000),  # the colour evidence had to add a body, not hair
    _report(soft_area=9000),  # the model was mostly unsure
    _report(core_area=0),  # nothing to protect
    _report(evidence_used=False),  # no picture to verify against
])
def test_an_unreliable_matte_is_rejected(bad):
    ok, detail = assess_matte([_report(), bad, _report()])
    assert not ok and detail


def test_no_matte_at_all_is_rejected():
    assert assess_matte([])[0] is False


def test_the_policy_limits_are_generic_shares():
    strict = MatteGatePolicy(max_added_frac=0.0)
    assert not assess_matte([_report(added_area=1)], strict)[0]
    assert assess_matte([_report(added_area=1)])[0]


def test_the_planner_rejects_behind_subject_when_the_matte_is_unreliable():
    def load(edl, start, end):
        data = loader()(edl, start, end)
        data.matte = [_report(added_area=6000)] * len(data.masks)
        return data

    (p,) = _plans(load=load)
    assert p.decision == bs.DECISION_FOREGROUND and not p.recommended  # text stays in front; nothing goes behind
    assert p.spec is not None and p.spec.behind_subject is False
    assert "matte_integrity" in p.reason
    assert not p.gate["matte_integrity"]["passed"]


def test_the_planner_keeps_behind_subject_when_the_matte_is_reliable():
    def load(edl, start, end):
        data = loader()(edl, start, end)
        data.matte = [_report()] * len(data.masks)
        return data

    (p,) = _plans(load=load)
    assert p.gate["matte_integrity"]["passed"] and p.spec is not None


# -- 7. preview and final render share one path -------------------------------


def test_preview_and_final_composite_through_the_same_function_with_the_refined_cutout(tmp_path: Path):
    from video_edit_agent.core import pipeline as pipeline_mod
    from video_edit_agent.render import micro_preview as mp

    assert pipeline_mod.compose_behind_subject is bs.compose_behind_subject is mp.compose_behind_subject
    edl = _edl(_clip(0, 8))
    calls, render_fn, cutout_fn = _fakes(tmp_path)
    _compose(edl, _spec(edl), tmp_path, render_fn, cutout_fn)
    assert calls[0][3]["refine"] is True  # the same matte maths whichever caller asks


def test_the_cutout_cache_is_keyed_by_the_matte_version(tmp_path: Path):
    from video_edit_agent.subject import compositor

    src = tmp_path / "s.mp4"
    src.write_bytes(b"footage")
    plain = compositor._cutout_cache_path(tmp_path, src, 1.0, 2.0, 10.0, (W, 1920), False)
    refined = compositor._cutout_cache_path(tmp_path, src, 1.0, 2.0, 10.0, (W, 1920), True)
    assert refine.REFINE_VERSION in refined.name and refined != plain


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_overlay_chain_matches_the_numpy_contract(tmp_path: Path):
    """The renderers stack layers with ffmpeg's `overlay` (graphic, then the subject cutout):
    that must produce what `composite_behind` says, so the tests on the maths hold for the video."""
    from PIL import Image

    frame, silhouette, _ = _scene()
    g_rgb, g_a = _graphic()
    alpha = _soft_mask(silhouette)
    graphic = np.dstack([g_rgb, np.rint(g_a * 255).astype(np.uint8)])
    cutout = np.dstack([frame, np.rint(alpha * 255).astype(np.uint8)])
    Image.fromarray(frame).save(tmp_path / "bg.png")
    Image.fromarray(graphic, "RGBA").save(tmp_path / "g.png")
    Image.fromarray(cutout, "RGBA").save(tmp_path / "c.png")
    out = tmp_path / "out.png"
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(tmp_path / "bg.png"), "-i", str(tmp_path / "g.png"),
         "-i", str(tmp_path / "c.png"), "-filter_complex",
         "[0:v]format=rgb24[b];[1:v]format=rgba[g];[2:v]format=rgba[c];[b][g]overlay=format=auto[t];[t][c]overlay=format=auto",
         "-frames:v", "1", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    got = np.array(Image.open(out).convert("RGB")).astype(np.int16)
    want = composite_behind(frame, g_rgb, g_a, frame, np.rint(alpha * 255) / 255.0).astype(np.int16)
    assert np.abs(got - want).max() <= 2
    assert check_integrity(frame, alpha, g_a, got.astype(np.uint8), tolerance=7.0).passed


# -- 8. nothing brand- or video-specific in the new generic module -------------


def test_the_generic_module_reads_no_brand_or_video_values():
    import re

    from tests.test_behind_subject_text import _SRC, FORBIDDEN

    text = (_SRC / "subject" / "integrity.py").read_text(encoding="utf-8")
    for pattern in FORBIDDEN:
        assert not re.search(pattern, text, re.IGNORECASE), pattern

