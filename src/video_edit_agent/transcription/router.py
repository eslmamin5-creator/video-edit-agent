"""Transcription provider router (spec sections 5, 6, 24).

Auto priority: Gemini -> ElevenLabs -> Local Faster-Whisper. If no keys are
available, Local Faster-Whisper is used automatically without asking for a
key. In strict offline mode, only local providers are ever attempted.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from video_edit_agent.core.config import TranscriptionConfig
from video_edit_agent.core.schemas import Transcript
from video_edit_agent.transcription.base import TranscriptionProvider, TranscriptionUnavailable
from video_edit_agent.transcription.providers.elevenlabs import ElevenLabsProvider
from video_edit_agent.transcription.providers.faster_whisper import FasterWhisperProvider
from video_edit_agent.transcription.providers.gemini import GeminiTranscriptionProvider
from video_edit_agent.transcription.providers.openai_whisper import OpenAIWhisperProvider
from video_edit_agent.transcription.providers.whisper_cpp import WhisperCppProvider

logger = logging.getLogger(__name__)

OFFLINE_SAFE_PROVIDERS = {"faster-whisper", "openai-whisper", "whisper.cpp"}


def build_providers(cfg: TranscriptionConfig, gemini_model: str) -> dict[str, TranscriptionProvider]:
    return {
        "gemini": GeminiTranscriptionProvider(model=gemini_model),
        "elevenlabs": ElevenLabsProvider(),
        "faster-whisper": FasterWhisperProvider(
            model=cfg.local.model, device=cfg.local.device, compute_type=cfg.local.compute_type
        ),
        "openai-whisper": OpenAIWhisperProvider(),
        "whisper.cpp": WhisperCppProvider(),
    }


class TranscriptionRouter:
    def __init__(self, cfg: TranscriptionConfig, *, offline: bool, gemini_model: str = "gemini-3.5-flash"):
        self.cfg = cfg
        self.offline = offline
        self.providers = build_providers(cfg, gemini_model)

    def _candidate_order(self) -> list[str]:
        if self.cfg.provider != "auto":
            return [self.cfg.provider]
        order = list(self.cfg.priority)
        if "faster-whisper" not in order:
            order.append("faster-whisper")
        return order

    def transcribe(self, audio_path: Path, *, on_attempt=None) -> Transcript:
        errors: list[str] = []
        for name in self._candidate_order():
            provider = self.providers.get(name)
            if provider is None:
                continue
            if self.offline and name not in OFFLINE_SAFE_PROVIDERS:
                errors.append(f"{name}: skipped (offline mode)")
                continue
            available, reason = provider.is_available(offline=self.offline)
            if on_attempt:
                on_attempt(name, available, reason)
            if not available:
                errors.append(f"{name}: {reason}")
                continue
            try:
                return provider.transcribe(audio_path, language=self.cfg.language, locale=self._locale(self.cfg))
            except TranscriptionUnavailable as e:
                errors.append(f"{name}: {e}")
                continue

        raise TranscriptionUnavailable(
            "No transcription provider could run. Attempts:\n  " + "\n  ".join(errors)
        )

    @staticmethod
    def _locale(cfg: TranscriptionConfig) -> str | None:
        return None if cfg.locale in ("auto", None) else cfg.locale


def save_transcript(transcript: Transcript, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def load_transcript(path: Path) -> Transcript:
    return Transcript.model_validate(json.loads(path.read_text(encoding="utf-8")))
