"""EDL validation + persistence tests (spec section 12)."""
from __future__ import annotations

import pytest

from video_edit_agent.core.schemas import EDL, EDLClip
from video_edit_agent.editorial.edl import EDLValidationError, load, save, validate


def test_empty_edl_fails_validation():
    with pytest.raises(EDLValidationError, match="no clips"):
        validate(EDL(clips=[]))


def test_clip_with_zero_or_negative_source_duration_fails(sample_video):
    edl = EDL(clips=[
        EDLClip(source_file=str(sample_video), source_in=1.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0),
    ])
    with pytest.raises(EDLValidationError, match="source_out"):
        validate(edl)


def test_clip_with_zero_or_negative_timeline_duration_fails(sample_video):
    edl = EDL(clips=[
        EDLClip(source_file=str(sample_video), source_in=0.0, source_out=1.0, timeline_in=1.0, timeline_out=1.0),
    ])
    with pytest.raises(EDLValidationError, match="timeline_out"):
        validate(edl)


def test_missing_source_file_fails_validation(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    edl = EDL(clips=[
        EDLClip(source_file=str(missing), source_in=0.0, source_out=1.0, timeline_in=0.0, timeline_out=1.0),
    ])
    with pytest.raises(EDLValidationError, match="source file missing"):
        validate(edl)


def test_valid_edl_passes(sample_edl):
    validate(sample_edl)  # should not raise


def test_edl_save_and_load_round_trip(tmp_path, sample_edl):
    path = tmp_path / "edl.json"
    save(sample_edl, path)
    restored = load(path)
    assert restored.clips[0].source_file == sample_edl.clips[0].source_file
    assert restored.fps == sample_edl.fps
    validate(restored)  # re-renderable: loaded EDL must still validate
