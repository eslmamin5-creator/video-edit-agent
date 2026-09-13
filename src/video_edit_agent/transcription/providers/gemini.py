"""Gemini transcription provider (spec section 6).

Uses verbatim mode by default — Smart transcription must never be used as the
editing source because it strips fillers/false-starts/repetitions, which are
valuable editorial signal (section 6).
"""
from __future__ import annotations

import json
from pathlib import Path

from video_edit_agent.core.config import get_gemini_key
from video_edit_agent.core.schemas import Segment, Transcript, Word
from video_edit_agent.transcription.base import TranscriptionProvider, TranscriptionUnavailable

VERBATIM_PROMPT = """Transcribe this audio VERBATIM.

Rules:
- Do NOT clean up disfluencies. Keep fillers, false starts, repetitions, and
  self-corrections exactly as spoken.
- Do NOT translate. Do NOT change dialect. Transcribe in the exact language
  and dialect spoken, including any code-switching between languages.
- Return ONLY strict JSON matching this shape, no prose, no markdown fences:
{"language": "...", "segments": [{"speaker": "...", "start": 0.0, "end": 0.0,
"text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.0}]}]}
"""


class GeminiTranscriptionProvider(TranscriptionProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-3.5-transcribe", locale_hint: str | None = None):
        self.model = model
        self.locale_hint = locale_hint

    def is_available(self, *, offline: bool) -> tuple[bool, str]:
        if offline:
            return False, "offline mode: cloud providers disabled"
        if get_gemini_key() is None:
            return False, "missing GEMINI_API_KEY"
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return False, "google-genai not installed (pip install video-edit-agent[gemini])"
        return True, "OK"

    def transcribe(self, audio_path: Path, *, language: str = "auto", locale: str | None = None) -> Transcript:
        available, reason = self.is_available(offline=False)
        if not available:
            raise TranscriptionUnavailable(reason)

        try:
            from google import genai
        except ImportError as e:
            raise TranscriptionUnavailable("google-genai not installed") from e

        client = genai.Client(api_key=get_gemini_key())
        prompt = VERBATIM_PROMPT
        if locale or self.locale_hint:
            prompt += f"\nExpected locale hint (do not force it if audio differs): {locale or self.locale_hint}\n"

        uploaded = client.files.upload(file=str(audio_path))
        response = client.models.generate_content(model=self.model, contents=[uploaded, prompt])

        text = (response.text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise TranscriptionUnavailable(f"Gemini returned non-JSON transcript: {e}") from e

        segments = []
        for idx, seg in enumerate(data.get("segments", [])):
            words = [Word(word=w["word"], start=w["start"], end=w["end"], confidence=1.0) for w in seg.get("words", [])]
            segments.append(
                Segment(
                    id=f"s{idx}",
                    speaker=seg.get("speaker"),
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"],
                    words=words,
                )
            )

        return Transcript(
            provider=self.name,
            language=data.get("language", language),
            locale=locale,
            duration=segments[-1].end if segments else 0.0,
            speakers=sorted({s.speaker for s in segments if s.speaker}),
            segments=segments,
            verbatim=True,
        )
