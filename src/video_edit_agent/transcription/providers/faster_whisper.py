"""Local Faster-Whisper provider — the default, always-available transcriber
(spec sections 5, 24). Works fully offline once a model is downloaded."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.schemas import Segment, Transcript, Word
from video_edit_agent.transcription.base import TranscriptionProvider, TranscriptionUnavailable

_model_cache: dict[tuple[str, str, str], object] = {}


class FasterWhisperProvider(TranscriptionProvider):
    name = "faster-whisper"

    def __init__(self, model: str = "auto", device: str = "auto", compute_type: str = "auto"):
        self.model_size = "small" if model == "auto" else model
        self.device = device
        self.compute_type = compute_type

    def is_available(self, *, offline: bool) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, "faster-whisper not installed (pip install video-edit-agent[local])"
        return True, "OK"

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscriptionUnavailable("faster-whisper not installed") from e

        device = self.device if self.device != "auto" else "cpu"
        compute_type = self.compute_type if self.compute_type != "auto" else "int8"
        key = (self.model_size, device, compute_type)
        if key not in _model_cache:
            try:
                _model_cache[key] = WhisperModel(self.model_size, device=device, compute_type=compute_type)
            except Exception as e:
                raise TranscriptionUnavailable(
                    f"Local Whisper model '{self.model_size}' is not available locally and could not be "
                    f"prepared ({e}). Run `videoedit setup` first, or prepare it explicitly before going "
                    f"offline."
                ) from e
        return _model_cache[key]

    def transcribe(self, audio_path: Path, *, language: str = "auto", locale: str | None = None) -> Transcript:
        model = self._load_model()
        lang = None if language in ("auto", None) else language.split("-")[0]
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=lang,
            word_timestamps=True,
            vad_filter=False,  # keep fillers/false starts — verbatim is an editing signal
        )

        segments: list[Segment] = []
        for idx, seg in enumerate(segments_iter):
            words = [
                Word(word=w.word.strip(), start=w.start, end=w.end, confidence=float(w.probability))
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(id=f"s{idx}", speaker=None, start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
            )

        return Transcript(
            provider=self.name,
            language=info.language or (lang or "auto"),
            locale=locale,
            duration=info.duration or (segments[-1].end if segments else 0.0),
            speakers=[],
            segments=segments,
            verbatim=True,
        )
