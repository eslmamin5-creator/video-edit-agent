"""Shared fixtures for the videoedit test suite."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_edit_agent.core.schemas import EDL, EDLClip, Segment, Transcript, Word  # noqa: E402

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
