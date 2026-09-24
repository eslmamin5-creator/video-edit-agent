"""Baseline Recovery Milestone item 4: focused tests for the real HyperFrames
adapter (`motion/hyperframes/adapter.py`).

The old adapter assumed a Python-importable `hyperframes` SDK that does not
exist. The real, verified integration surface is the published npm CLI
(`npx hyperframes@<pinned>`, see the adapter's module docstring for the
due-diligence trail). These tests exercise the adapter's own logic --
availability detection, per-kind layout selection, variable mapping, and the
offline/not-yet-cached guard -- without making a real network call (the real
end-to-end render is exercised by `scripts/hyperframes_acceptance.py`).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from video_edit_agent.core.schemas import AnimationKind, AnimationSpec
from video_edit_agent.motion.hyperframes import adapter as hf


def _spec(kind: AnimationKind, **kwargs) -> AnimationSpec:
    defaults = {"timeline_start": 0.0, "timeline_end": 3.0, "text": "Headline"}
    defaults.update(kwargs)
    return AnimationSpec(kind=kind, **defaults)


def test_layout_for_kind_maps_stat_and_lower_third_and_defaults_to_title():
    assert hf._layout_for_kind(_spec(AnimationKind.STAT_COUNTER)) == "stat"
    assert hf._layout_for_kind(_spec(AnimationKind.METRIC_HIGHLIGHT)) == "stat"
    assert hf._layout_for_kind(_spec(AnimationKind.LOWER_THIRD)) == "lower_third"
    assert hf._layout_for_kind(_spec(AnimationKind.HOOK_TITLE)) == "title"


def test_variables_for_spec_maps_text_subtext_value_and_duration():
    spec = _spec(AnimationKind.STAT_COUNTER, subtext="context", value="42", timeline_start=1.0, timeline_end=4.0)
    variables = hf._variables_for_spec(spec, fps=30)
    assert variables["title"] == "Headline"
    assert variables["subtitle"] == "context"
    assert variables["value"] == "42"
    assert variables["layout"] == "stat"
    assert variables["duration"] == pytest.approx(3.0)


def test_variables_for_spec_defaults_missing_subtext_and_value_to_empty_string():
    spec = _spec(AnimationKind.HOOK_TITLE)
    variables = hf._variables_for_spec(spec, fps=30)
    assert variables["subtitle"] == ""
    assert variables["value"] == ""


def test_is_available_requires_node_npm_and_template_dir():
    with patch("video_edit_agent.motion.hyperframes.adapter.detect_node") as node, \
         patch("video_edit_agent.motion.hyperframes.adapter.detect_npm") as npm:
        node.return_value.available = True
        npm.return_value.available = True
        assert hf.is_available() is True

        node.return_value.available = False
        assert hf.is_available() is False


def test_render_offline_without_cached_cli_raises_without_touching_subprocess(tmp_path: Path):
    with patch("video_edit_agent.motion.hyperframes.adapter.is_available", return_value=True), \
         patch("video_edit_agent.motion.hyperframes.adapter.subprocess.run") as run:
        spec = _spec(AnimationKind.HOOK_TITLE)
        with pytest.raises(hf.HyperFramesUnavailable):
            hf.render(spec, project_root=tmp_path, output_path=tmp_path / "out.webm", offline=True)
        run.assert_not_called()


def test_render_offline_succeeds_once_ready_marker_exists(tmp_path: Path):
    marker = hf._cli_ready_marker(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    output_path = tmp_path / "out.webm"

    def _fake_run(cmd, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-webm-bytes")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    with patch("video_edit_agent.motion.hyperframes.adapter.is_available", return_value=True), \
         patch("video_edit_agent.motion.hyperframes.adapter.shutil.which", return_value="npx"), \
         patch("video_edit_agent.motion.hyperframes.adapter.subprocess.run", side_effect=_fake_run):
        spec = _spec(AnimationKind.HOOK_TITLE)
        result = hf.render(spec, project_root=tmp_path, output_path=output_path, offline=True)

    assert result == output_path.resolve()
    assert result.exists()


def test_render_raises_render_error_on_nonzero_exit_when_online(tmp_path: Path):
    output_path = tmp_path / "out.webm"

    class _Result:
        returncode = 1
        stderr = "boom: chrome crashed"

    with patch("video_edit_agent.motion.hyperframes.adapter.is_available", return_value=True), \
         patch("video_edit_agent.motion.hyperframes.adapter.shutil.which", return_value="npx"), \
         patch("video_edit_agent.motion.hyperframes.adapter.subprocess.run", return_value=_Result()):
        spec = _spec(AnimationKind.HOOK_TITLE)
        with pytest.raises(hf.HyperFramesRenderError, match="chrome crashed"):
            hf.render(spec, project_root=tmp_path, output_path=output_path, offline=False)
