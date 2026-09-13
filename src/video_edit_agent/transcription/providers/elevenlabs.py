"""ElevenLabs speech-to-text provider (spec section 5)."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.core.config import get_elevenlabs_key
from video_edit_agent.core.schemas import Segment, Transcript, Word
from video_edit_agent.transcription.base import TranscriptionProvider, TranscriptionUnavailable


class ElevenLabsProvider(TranscriptionProvider):
    name = "elevenlabs"

    def is_available(self, *, offline: bool) -> tuple[bool, str]:
        if offline:
            return False, "offline mode: cloud providers disabled"
        if get_elevenlabs_key() is None:
            return False, "missing ELEVENLABS_API_KEY"
        try:
            import elevenlabs  # noqa: F401
        except ImportError:
            return False, "elevenlabs not installed (pip install video-edit-agent[elevenlabs])"
        return True, "OK"

    def transcribe(self, audio_path: Path, *, language: str = "auto", locale: str | None = None) -> Transcript:
        available, reason = self.is_available(offline=False)
        if not available:
            raise TranscriptionUnavailable(reason)

        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as e:
            raise TranscriptionUnavailable("elevenlabs not installed") from e

        client = ElevenLabs(api_key=get_elevenlabs_key())
        with audio_path.open("rb") as f:
            result = client.speech_to_text.convert(
                file=f,
                model_id="scribe_v1",
                language_code=None if language == "auto" else language,
                diarize=True,
                timestamps_granularity="word",
            )

        words_raw = getattr(result, "words", None) or []
        segments: list[Segment] = []
        if words_raw:
            # ElevenLabs returns a flat word stream; group into segments on
            # speaker change or a >0.6s gap, preserving verbatim text.
            current: list = []
            current_speaker = None
            for w in words_raw:
                speaker = getattr(w, "speaker_id", None)
                gap_break = current and (w.start - current[-1].end) > 0.6
                if current and (speaker != current_speaker or gap_break):
                    segments.append(_segment_from_words(len(segments), current, current_speaker))
                    current = []
                current_speaker = speaker
                current.append(w)
            if current:
                segments.append(_segment_from_words(len(segments), current, current_speaker))

        return Transcript(
            provider=self.name,
            language=getattr(result, "language_code", language) or language,
            locale=locale,
            duration=segments[-1].end if segments else 0.0,
            speakers=sorted({s.speaker for s in segments if s.speaker}),
            segments=segments,
            verbatim=True,
        )


def _segment_from_words(idx: int, words: list, speaker) -> Segment:
    return Segment(
        id=f"s{idx}",
        speaker=str(speaker) if speaker is not None else None,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words).strip(),
        words=[Word(word=w.text, start=w.start, end=w.end, confidence=1.0) for w in words],
    )
