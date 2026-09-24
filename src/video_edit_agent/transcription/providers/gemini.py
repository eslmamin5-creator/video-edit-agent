"""Gemini transcription provider (spec section 6).

Uses verbatim mode by default — Smart transcription must never be used as the
editing source because it strips fillers/false-starts/repetitions, which are
valuable editorial signal (section 6).
"""
from __future__ import annotations

import json
from pathlib import Path

from video_edit_agent.core.config import get_gemini_key
from video_edit_agent.core.media import MediaError, probe_duration
from video_edit_agent.core.schemas import Segment, Transcript, Word
from video_edit_agent.transcription.base import TranscriptionProvider, TranscriptionUnavailable

VERBATIM_PROMPT = """Transcribe this audio VERBATIM.

Rules:
- Do NOT clean up disfluencies. Keep fillers, false starts, repetitions, and
  self-corrections exactly as spoken.
- Do NOT translate. Do NOT change dialect. Transcribe in the exact language
  and dialect spoken, including any code-switching between languages.
- All "start"/"end" values are PLAIN DECIMAL SECONDS from the beginning of
  the audio, NEVER minutes:seconds notation. They must increase
  monotonically for the whole audio, including past the 1-minute mark.
  For example, one minute and 35 seconds into the audio must be written as
  95.0 — NOT as 1.35 or 1:35.
- Return ONLY strict JSON matching this shape, no prose, no markdown fences:
{"language": "...", "segments": [{"speaker": "...", "start": 0.0, "end": 0.0,
"text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.0}]}]}
"""

# How far behind the running "previous end" a raw timestamp may fall and
# still be trusted as plain decimal seconds (covers minor model jitter,
# e.g. a word's start landing fractionally before the prior word's end).
_JITTER_TOLERANCE_S = 1.0

# How far a transcript's final timestamp may exceed the probed media
# duration before it's clamped rather than rejected (covers the model
# slightly overestimating where the last syllable trails off).
_DURATION_OVERSHOOT_TOLERANCE_S = 3.0

# How much of the media's tail may go uncovered by the transcript before
# it's treated as a silent truncation rather than legitimate trailing
# silence/room tone.
_MIN_COVERAGE_FRACTION = 0.85

# The fractional check above is unreliable on very short clips, where a
# legitimate trailing pause can easily exceed 15% of the whole duration
# (e.g. a 2s clip with speech only in the first 0.8s). Only treat coverage
# as a truncation failure once the actually-uncovered tail is at least this
# many seconds long, regardless of what fraction of the total that is.
_MIN_UNCOVERED_TAIL_S = 5.0


class TimestampNormalizationError(TranscriptionUnavailable):
    """Raised when Gemini's returned timestamps cannot be safely resolved to
    a single canonical decimal-seconds representation."""


def _reinterpret_as_minute_dot_second(raw: float) -> float | None:
    """A model quirk on longer audio: despite the prompt, timestamps past
    the 1-minute mark sometimes get written as minute.second notation (e.g.
    "1.03" meaning 1:03 = 63.0s) instead of continuous decimal seconds. If
    `raw` parses as a plausible M.SS value (a non-negative two-digit
    fractional part under 60), returns the equivalent decimal seconds;
    otherwise None.
    """
    if raw < 0:
        return None
    minutes = int(raw)
    # Round to avoid float artifacts (e.g. 1.13 - 1 == 0.12999999999999989).
    seconds_part = round((raw - minutes) * 100)
    if seconds_part < 0 or seconds_part >= 60:
        return None
    return minutes * 60.0 + seconds_part


def _normalize_timestamp(raw: float, prev_end: float) -> float | None:
    """Resolves `raw` to canonical decimal seconds given the running
    "previous end" of the sequence so far. Trusts `raw` as-is when it's
    consistent with monotonic progression; otherwise tries the M.SS
    reinterpretation; otherwise returns None (unresolvable/ambiguous).
    """
    if raw < 0:
        return None
    if raw >= prev_end - _JITTER_TOLERANCE_S:
        return raw
    reinterpreted = _reinterpret_as_minute_dot_second(raw)
    if reinterpreted is not None and reinterpreted >= prev_end - _JITTER_TOLERANCE_S:
        return reinterpreted
    return None


def _normalize_word(raw_word: dict, prev_end: float) -> tuple[dict, float]:
    start = _normalize_timestamp(float(raw_word["start"]), prev_end)
    if start is None:
        raise TimestampNormalizationError(
            f"Gemini word {raw_word.get('word')!r} has an unresolvable start timestamp "
            f"{raw_word['start']!r} (previous end: {prev_end}s)"
        )
    end = _normalize_timestamp(float(raw_word["end"]), start)
    if end is None or end < start:
        raise TimestampNormalizationError(
            f"Gemini word {raw_word.get('word')!r} has an unresolvable or inverted end "
            f"timestamp {raw_word['end']!r} (normalized start: {start}s)"
        )
    return {**raw_word, "start": start, "end": end}, end


def _normalize_segment(raw_seg: dict, seg_id: str, prev_end: float) -> tuple[dict, float]:
    start = _normalize_timestamp(float(raw_seg["start"]), prev_end)
    if start is None:
        raise TimestampNormalizationError(
            f"Gemini segment {seg_id!r} has an unresolvable start timestamp "
            f"{raw_seg['start']!r} (previous segment's end: {prev_end}s) — sequence "
            "progression is not plausible"
        )
    end = _normalize_timestamp(float(raw_seg["end"]), start)
    if end is None or end < start:
        raise TimestampNormalizationError(
            f"Gemini segment {seg_id!r} has an unresolvable or inverted end timestamp "
            f"{raw_seg['end']!r} (normalized start: {start}s)"
        )

    words_norm = []
    word_prev = start
    for raw_word in raw_seg.get("words", []):
        word_norm, word_prev = _normalize_word(raw_word, word_prev)
        words_norm.append(word_norm)

    return {**raw_seg, "start": start, "end": end, "words": words_norm}, end


def normalize_transcript_timestamps(raw_segments: list[dict], *, media_duration: float | None) -> list[dict]:
    """Converts every "start"/"end" value in `raw_segments` (segment- and
    word-level) into one canonical representation — continuous decimal
    seconds from the start of the audio — defensively, without relying on
    the model having followed the prompt's formatting instruction.

    Raises TimestampNormalizationError if any timestamp is ambiguous /
    unresolvable, if the resolved sequence isn't monotonically plausible,
    if it lands far outside the known media duration, or if the resolved
    transcript's coverage silently truncates a substantial trailing portion
    of the source (see module-level tolerance constants). Callers should
    treat this as a failed transcription and fall back rather than build an
    EDL from unreliable timing.
    """
    normalized: list[dict] = []
    prev_end = 0.0
    for idx, raw_seg in enumerate(raw_segments):
        seg_norm, prev_end = _normalize_segment(raw_seg, f"s{idx}", prev_end)
        normalized.append(seg_norm)

    if not normalized:
        return normalized

    last_end = normalized[-1]["end"]

    if media_duration and media_duration > 0:
        if last_end > media_duration + _DURATION_OVERSHOOT_TOLERANCE_S:
            raise TimestampNormalizationError(
                f"Gemini transcript's final timestamp ({last_end}s) far exceeds the media "
                f"duration ({media_duration}s) even after normalization"
            )
        if last_end > media_duration:
            # Minor overshoot from the model rounding the last syllable's
            # trailing edge — clamp rather than reject.
            normalized[-1]["end"] = media_duration
            for w in normalized[-1].get("words", []):
                w["end"] = min(w["end"], media_duration)
            last_end = media_duration

        if (
            last_end < media_duration * _MIN_COVERAGE_FRACTION
            and (media_duration - last_end) > _MIN_UNCOVERED_TAIL_S
        ):
            raise TimestampNormalizationError(
                f"Gemini transcript covers only up to {last_end}s of a {media_duration}s "
                "source — a substantial trailing portion of the spoken content appears "
                "to have been silently truncated"
            )

    return normalized


class GeminiTranscriptionProvider(TranscriptionProvider):
    name = "gemini"

    # Default model must be a general multimodal model (flash/pro), not a
    # dedicated "-transcribe" model: dedicated ASR models return their
    # answer via a native `audio_transcription` response part (plain text,
    # no timestamps) and ignore the VERBATIM_PROMPT/JSON contract entirely,
    # so response.text comes back empty and word-level timing is lost.

    def __init__(self, model: str = "gemini-3.5-flash", locale_hint: str | None = None):
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

        try:
            media_duration = probe_duration(audio_path)
        except MediaError:
            media_duration = None

        raw_segments = data.get("segments", [])
        normalized_segments = normalize_transcript_timestamps(raw_segments, media_duration=media_duration)

        segments = []
        for idx, seg in enumerate(normalized_segments):
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
