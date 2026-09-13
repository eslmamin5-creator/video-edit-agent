"""Transcription provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from video_edit_agent.core.schemas import Transcript


class TranscriptionUnavailable(RuntimeError):
    """Raised by a provider when it cannot run right now (missing key, missing
    model, offline mode, etc). The router catches this and falls back."""


class TranscriptionProvider(ABC):
    name: str

    @abstractmethod
    def is_available(self, *, offline: bool) -> tuple[bool, str]:
        """Return (available, reason)."""

    @abstractmethod
    def transcribe(self, audio_path: Path, *, language: str = "auto", locale: str | None = None) -> Transcript:
        ...
