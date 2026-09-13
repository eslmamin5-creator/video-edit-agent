"""OpenAI Whisper (local `openai-whisper` package) compatibility provider."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import Segment, Transcript, Word
from video_edit_agent.transcription.base import TranscriptionProvider, TranscriptionUnavailable


class OpenAIWhisperProvider(TranscriptionProvider):
    name = "openai-whisper"

    def __init__(self, model: str = "small"):
        self.model_size = model

    def is_available(self, *, offline: bool) -> tuple[bool, str]:
        try:
            import whisper  # noqa: F401
        except ImportError:
            return False, "openai-whisper not installed"
        return True, "OK"

    def transcribe(self, audio_path: Path, *, language: str = "auto", locale: str | None = None) -> Transcript:
        try:
            import whisper
        except ImportError as e:
            raise TranscriptionUnavailable("openai-whisper not installed") from e

        model = whisper.load_model(self.model_size)
        lang = None if language in ("auto", None) else language.split("-")[0]
        result = model.transcribe(str(audio_path), language=lang, word_timestamps=True, verbose=False)

        segments = []
        for idx, seg in enumerate(result.get("segments", [])):
            words = [
                Word(word=w["word"].strip(), start=w["start"], end=w["end"], confidence=float(w.get("probability", 1.0)))
                for w in seg.get("words", [])
            ]
            segments.append(Segment(id=f"s{idx}", start=seg["start"], end=seg["end"], text=seg["text"].strip(), words=words))

        return Transcript(
            provider=self.name,
            language=result.get("language", language),
            locale=locale,
            duration=segments[-1].end if segments else 0.0,
            segments=segments,
            verbatim=True,
        )
