"""Phase 1.5.2: Windows-compatible standard MP4 export.

The generic render path must emit a conservative H.264/AAC MP4 (avc1, High@4.1, yuv420p,
AAC-LC 48 kHz stereo, faststart), validate the result before finalization, and change only
container/codec parameters -- never the visual timeline. Synthetic media only."""
from __future__ import annotations

import dataclasses
import inspect
import subprocess
from pathlib import Path

import pytest

from tests.conftest import requires_ffmpeg
from video_edit_agent.core import pipeline
from video_edit_agent.core.media import probe
from video_edit_agent.render import export, export_validation
from video_edit_agent.render.composition import RenderPlan
from video_edit_agent.render.export import (
    PRESETS,
    audio_encode_args,
    video_encode_args,
    video_filter_tail,
)
from video_edit_agent.render.export_validation import (
    ExportCompatibilityError,
    assert_export_compatible,
    check_metadata,
)
from video_edit_agent.render.ffmpeg import render

REEL = PRESETS["reel"]


def _flag(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _make_clip(path: Path, *, pix_fmt: str = "yuv420p", audio: str = "aac", width: int = 1080, height: int = 1920) -> Path:
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=s={width}x{height}:d=1:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-ac", "2", "-c:v", "libx264", "-pix_fmt", pix_fmt, "-c:a", audio,
        "-movflags", "+faststart", str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return path


# ---- 1-8. the standard encode definition ------------------------------------------------------
def test_1_standard_export_uses_h264():
    assert _flag(video_encode_args(REEL), "-c:v") == "libx264"


def test_2_standard_export_uses_avc1_tag():
    assert _flag(video_encode_args(REEL), "-tag:v") == "avc1"


def test_3_standard_export_uses_yuv420p():
    assert _flag(video_encode_args(REEL), "-pix_fmt") == "yuv420p"
    assert video_filter_tail(REEL).endswith("format=yuv420p")


def test_4_standard_export_uses_high_profile_level_4_1_and_bt709():
    args = video_encode_args(REEL)
    assert _flag(args, "-profile:v") == "high" and _flag(args, "-level:v") == "4.1"
    assert _flag(args, "-colorspace") == "bt709" and _flag(args, "-color_trc") == "bt709"
    assert "out_color_matrix=bt709" in video_filter_tail(REEL)


def test_5_standard_export_uses_aac():
    assert _flag(audio_encode_args(REEL), "-c:a") == "aac"
    assert _flag(audio_encode_args(REEL), "-b:a") == "192k"


def test_6_audio_sample_rate_is_48k():
    assert _flag(audio_encode_args(REEL), "-ar") == "48000"


def test_7_audio_is_stereo():
    assert _flag(audio_encode_args(REEL), "-ac") == "2"


@requires_ffmpeg
def test_8_render_output_is_faststart_and_fully_compatible(tmp_path: Path, sample_video, sample_edl):
    out = render(RenderPlan(edl=sample_edl, overlays=[], captions=None), tmp_path / "out.mp4", REEL)
    assert export_validation.moov_before_mdat(out) is True
    assert check_metadata(out, REEL) == []
    assert_export_compatible(out, REEL)
    video = next(s for s in export_validation._probe(out)["streams"] if s["codec_type"] == "video")
    assert video["pix_fmt"] == "yuv420p" and video["profile"] == "High" and video["level"] <= 41


# ---- 9-11. the validator ---------------------------------------------------------------------
@requires_ffmpeg
def test_9_validator_rejects_incompatible_pix_fmt(tmp_path: Path):
    clip = _make_clip(tmp_path / "yuv444.mp4", pix_fmt="yuv444p")
    problems = check_metadata(clip, REEL)
    assert any("pix_fmt" in p for p in problems)
    with pytest.raises(ExportCompatibilityError):
        assert_export_compatible(clip, REEL)


@requires_ffmpeg
def test_10_validator_rejects_non_aac_audio(tmp_path: Path):
    clip = _make_clip(tmp_path / "ac3.mp4", audio="ac3")
    assert any("audio codec" in p for p in check_metadata(clip, REEL))


@requires_ffmpeg
def test_11_validator_catches_decode_failure(tmp_path: Path, monkeypatch):
    clip = _make_clip(tmp_path / "ok.mp4")
    real_run = export_validation.run

    def failing_run(cmd, timeout=None):
        if cmd[0] == "ffmpeg" and "-f" in cmd and "-frames:v" not in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "[h264] error while decoding MB 3 4")
        return real_run(cmd, timeout=timeout)

    monkeypatch.setattr(export_validation, "run", failing_run)
    assert any("full decode failed" in p for p in export_validation.check_decode(clip))
    with pytest.raises(ExportCompatibilityError, match="decode"):
        assert_export_compatible(clip, REEL)


# ---- 12-14. finalization gate, explicit bypass, genericity -------------------------------------
def test_12_pipeline_finalization_runs_the_compatibility_gate_before_success():
    src = inspect.getsource(pipeline.run_pipeline)
    gate = src.index("assert_export_compatible(final_output")
    assert gate < src.index("memory.render_history.append(f\"Rendered")
    assert gate > src.index("compose_with_cards(")


@requires_ffmpeg
def test_13_nonstandard_export_bypasses_only_via_explicit_config(tmp_path: Path):
    clip = _make_clip(tmp_path / "yuv444.mp4", pix_fmt="yuv444p")
    with pytest.raises(ExportCompatibilityError):
        assert_export_compatible(clip, REEL)
    advanced = dataclasses.replace(REEL, standard=False)
    assert_export_compatible(clip, advanced)
    assert video_filter_tail(advanced) == ""
    assert "-pix_fmt" not in video_encode_args(advanced)


def test_14_no_project_specific_encoding_logic():
    for module in (export, export_validation):
        assert "client" not in inspect.getsource(module).lower()
    assert all(p.standard for p in PRESETS.values())


# ---- 15. export normalization does not touch the visual timeline --------------------------------
@requires_ffmpeg
def test_15_export_normalization_preserves_timeline_and_content(tmp_path: Path, sample_video, sample_edl):
    plan = RenderPlan(edl=sample_edl, overlays=[], captions=None)
    standard = render(plan, tmp_path / "standard.mp4", REEL)
    advanced = render(plan, tmp_path / "advanced.mp4", dataclasses.replace(REEL, standard=False))
    a, b = probe(standard), probe(advanced)
    assert (a.width, a.height, a.fps) == (b.width, b.height, b.fps)
    assert abs(a.duration - b.duration) < 0.05

    def frame(path: Path, name: str) -> bytes:
        png = tmp_path / name
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", "1.0", "-i", str(path), "-frames:v", "1",
             "-pix_fmt", "rgb24", "-f", "rawvideo", str(png)],
            check=True,
        )
        return png.read_bytes()

    fa, fb = frame(standard, "a.raw"), frame(advanced, "b.raw")
    assert len(fa) == len(fb)
    mean_abs = sum(abs(x - y) for x, y in zip(fa[::97], fb[::97])) / len(fa[::97])
    assert mean_abs < 6.0
