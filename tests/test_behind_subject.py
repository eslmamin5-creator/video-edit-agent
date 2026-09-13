"""V1.1 hardening (spec sections 8-9): Behind-Subject acceptance.

Two things are verified here, kept deliberately separate because they are
genuinely different in this codebase today:

1. Mask detection + caching (`subject/compositor.py`, `subject/mask_cache.py`)
   is real, working code -- it really runs mediapipe (or really reports it is
   unavailable) and really avoids recomputation on a repeated call.

2. `render/composition.py::build_filter_complex` never reads
   `Overlay.behind_subject` -- there is currently no code path anywhere in
   the repo that consumes a `CompositingPlan` to build an actual masked
   composite. This is captured as a characterization test, NOT a bug fix:
   fixing it is real new feature work, out of scope for a hardening-only
   pass (see scripts/behind_subject_acceptance.py for the full writeup).
   The test exists so this gap can never silently "become verified" by a
   future change that flips the flag without actually wiring the masking.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg
from video_edit_agent.render.composition import Overlay, RenderPlan, build_filter_complex
from video_edit_agent.subject import mask_cache
from video_edit_agent.subject.compositor import plan_behind_subject_overlay
from video_edit_agent.subject.detect import is_available as mediapipe_available


@requires_ffmpeg
def test_mask_cache_reuses_result_on_second_identical_call(sample_video: Path, tmp_path: Path):
    if not mediapipe_available():
        pytest.skip("mediapipe not installed -- graceful-fallback path is what runs in production here")

    cache_dir = tmp_path / "mask_cache"
    plan1 = plan_behind_subject_overlay(sample_video, 0.0, 1.0, cache_dir)
    cache_path = mask_cache.cache_path(cache_dir, sample_video, 0.0, 1.0, 6.0)

    # A cache entry is only ever written when real (non-empty) masks came
    # back; if this fixture's flat color video doesn't yield any, there's
    # nothing to reuse and the test only confirms deterministic behavior.
    if not cache_path.exists():
        plan2 = plan_behind_subject_overlay(sample_video, 0.0, 1.0, cache_dir)
        assert plan2.behind_subject == plan1.behind_subject
        assert plan2.reason == plan1.reason
        return

    mtime_before = cache_path.stat().st_mtime
    plan2 = plan_behind_subject_overlay(sample_video, 0.0, 1.0, cache_dir)
    mtime_after = cache_path.stat().st_mtime

    assert mtime_after == mtime_before, "second call must reuse the cached .npz, not rewrite it"
    assert plan2.behind_subject == plan1.behind_subject
    assert plan2.mask_frames_path == plan1.mask_frames_path


def test_missing_segmentation_dependency_falls_back_gracefully(tmp_path: Path, monkeypatch):
    """When mediapipe is unavailable, plan_behind_subject_overlay must return
    behind_subject=False with a reason, never raise."""
    import video_edit_agent.subject.detect as detect_module

    monkeypatch.setattr(detect_module, "is_available", lambda: False)
    fake_source = tmp_path / "does_not_matter.mp4"
    fake_source.write_bytes(b"not a real video, just needs to exist for content_hash()")
    plan = plan_behind_subject_overlay(fake_source, 0.0, 1.0, tmp_path / "cache")
    assert plan.behind_subject is False
    assert plan.reason


def test_render_filter_graph_does_not_yet_apply_behind_subject_masking():
    """Characterization test: `Overlay.behind_subject=True` currently has NO
    effect on the generated ffmpeg filter graph. Real masked compositing is
    not implemented in this repository yet -- see the V1.1 hardening report's
    Known Limitations. This test exists so a future change cannot silently
    start claiming "behind subject" works by merely setting the flag without
    actually altering the filter graph; when real masking is implemented,
    this test should be replaced with one that asserts the mask IS applied.
    """
    from video_edit_agent.core.schemas import EDL, EDLClip

    edl = EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[EDLClip(source_file="bg.mp4", source_in=0.0, source_out=2.0, timeline_in=0.0, timeline_out=2.0)],
    )
    plain = RenderPlan(edl=edl, overlays=[Overlay(path=Path("gfx.png"), start=0.0, end=2.0, behind_subject=False)])
    masked = RenderPlan(edl=edl, overlays=[Overlay(path=Path("gfx.png"), start=0.0, end=2.0, behind_subject=True)])

    _, filters_plain, _ = build_filter_complex(plain)
    _, filters_masked, _ = build_filter_complex(masked)

    assert filters_plain == filters_masked, (
        "behind_subject currently has no effect on the filter graph -- if this "
        "assertion starts failing, real masking has been implemented and this "
        "test's expectation (and the hardening report) should be updated"
    )
    assert "alphamerge" not in filters_masked and "maskedmerge" not in filters_masked
