"""whisper.cpp compatibility provider (subprocess-based, spec section 5).

Invokes an on-PATH `whisper-cpp` binary producing JSON output. Fully local,
no Python bindings required.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from video_edit_agent.core.media import run
from video_edit_agent.core.schemas import Segment, Transcript, Word
from video_edit_agent.transcription.base import TranscriptionProvider, TranscriptionUnavailable


class WhisperCppProvider(TranscriptionProvider):
    name = "whisper.cpp"

    def __init__(self, model_path: str | None = None, binary: str = "whisper-cpp"):
        self.model_path = model_path
        self.binary = binary

    def is_available(self, *, offline: bool) -> tuple[bool, str]:
        if shutil.which(self.binary) is None:
            return False, f"'{self.binary}' not found on PATH"
        if not self.model_path:
            return False, "no whisper.cpp model path configured"
        return True, "OK"

    def transcribe(self, audio_path: Path, *, language: str = "auto", locale: str | None = None) -> Transcript:
        available, reason = self.is_available(offline=True)
        if not available:
            raise TranscriptionUnavailable(reason)

        out_prefix = audio_path.with_suffix("")
        lang = "auto" if language == "auto" else language.split("-")[0]
        result = run(
            [
                self.binary, "-m", self.model_path, "-f", str(audio_path),
                "-l", lang, "-oj", "-of", str(out_prefix), "-nt",
            ],
            timeout=1800,
        )
        json_path = out_prefix.with_suffix(".json")
        if result.returncode != 0 or not json_path.exists():
            raise TranscriptionUnavailable(f"whisper.cpp failed: {result.stderr.strip()[-500:]}")

        data = json.loads(json_path.read_text(encoding="utf-8"))
        segments = []
        for idx, seg in enumerate(data.get("transcription", [])):
            text = seg.get("text", "").strip()
            start = _ts_to_seconds(seg.get("offsets", {}).get("from", 0))
            end = _ts_to_seconds(seg.get("offsets", {}).get("to", 0))
            segments.append(Segment(id=f"s{idx}", start=start, end=end, text=text, words=[Word(word=text, start=start, end=end)]))

        return Transcript(
            provider=self.name,
            language=lang,
            locale=locale,
            duration=segments[-1].end if segments else 0.0,
            segments=segments,
            verbatim=True,
        )


def _ts_to_seconds(ms: float) -> float:
    return float(ms) / 1000.0
