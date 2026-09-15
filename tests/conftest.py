"""Shared fixtures for the videoedit test suite."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript, Word

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A tiny synthetic 2s video with a silent audio track, generated purely
    from ffmpeg lavfi sources — no binary fixture file needed."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=320x568:d=2:r=30",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=2",
            "-shortest", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            str(out),
        ],
        capture_output=True, check=True, timeout=60,
    )
    return out


def make_word(word: str, start: float, end: float) -> Word:
    return Word(word=word, start=start, end=end, confidence=0.95)


def make_transcript(segments_text: list[tuple[str, float, float]], provider: str = "test") -> Transcript:
    """Builds a Transcript with one word per character-split token, evenly
    spaced across each segment's [start, end) window."""
    segments: list[Segment] = []
    for idx, (text, start, end) in enumerate(segments_text):
        tokens = text.split()
        words: list[Word] = []
        if tokens:
            step = (end - start) / len(tokens)
            for i, tok in enumerate(tokens):
                words.append(make_word(tok, start + i * step, start + (i + 1) * step))
        segments.append(Segment(id=f"s{idx}", start=start, end=end, text=text, words=words))
    return Transcript(provider=provider, language="auto", duration=segments[-1].end if segments else 0.0, segments=segments)


@pytest.fixture
def sample_transcript() -> Transcript:
    return make_transcript([
        ("hello there friend", 0.0, 1.5),
        ("this is a test", 1.5, 3.0),
    ])


def make_scene_video(
    out_path: Path, *, color: str, duration: float, width: int = 640, height: int = 360,
    fps: int = 30, with_audio: bool = True,
) -> Path:
    """A tiny synthetic scene video for Assembler fixtures -- solid color,
    optional silent audio track, generated purely from ffmpeg lavfi sources."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    inputs = ["-f", "lavfi", "-i", f"color=c={color}:s={width}x{height}:d={duration}:r={fps}"]
    if with_audio:
        inputs += ["-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={duration}"]
    cmd = ["ffmpeg", "-y", *inputs, "-shortest", "-pix_fmt", "yuv420p", "-c:v", "libx264"]
    if with_audio:
        cmd += ["-c:a", "aac"]
    cmd += [str(out_path)]
    subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    return out_path


@pytest.fixture
def assembler_scenes_dir(tmp_path: Path) -> Path:
    """5 legal synthetic scenes with differing colors, durations, audio
    presence, and (for one scene) a differing resolution/aspect ratio.
    Filenames are zero-padded (scene_01..scene_05) so a naive alphabetical
    sort would already agree with natural order here -- see
    `unordered_assembler_scenes_dir` for the alphabetical-vs-natural case."""
    d = tmp_path / "scenes"
    d.mkdir()
    specs = [
        ("scene_01_intro.mp4", "red", 1.0, 640, 360, True),
        ("scene_02_middle.mp4", "green", 1.5, 640, 360, True),
        ("scene_03_action.mp4", "blue", 1.0, 640, 360, False),
        ("scene_04_closeup.mp4", "yellow", 0.8, 480, 480, True),
        ("scene_05_outro.mp4", "purple", 1.2, 640, 360, True),
    ]
    for name, color, dur, w, h, audio in specs:
        make_scene_video(d / name, color=color, duration=dur, width=w, height=h, with_audio=audio)
    return d


@pytest.fixture
def unordered_assembler_scenes_dir(tmp_path: Path) -> Path:
    """Filenames where plain alphabetical sort would misorder scenes
    (clip10 < clip2 alphabetically) but natural filename order -- and thus
    the default preserve-order policy -- must keep clip1 < clip2 < clip10."""
    d = tmp_path / "scenes_unordered"
    d.mkdir()
    specs = [("clip1.mp4", "red"), ("clip2.mp4", "green"), ("clip10.mp4", "blue")]
    for name, color in specs:
        make_scene_video(d / name, color=color, duration=0.6, with_audio=True)
    return d


@pytest.fixture
def sample_edl(sample_video: Path) -> EDL:
    return EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[
            EDLClip(
                source_file=str(sample_video), source_in=0.0, source_out=2.0,
                timeline_in=0.0, timeline_out=2.0, caption_refs=["s0", "s1"],
            ),
        ],
    )
