"""Regression tests for the Gemini timestamp-drift bug (spec section 6 addendum).

On audio longer than ~60s, Gemini (gemini-3.5-flash) sometimes ignores the
VERBATIM_PROMPT's explicit "plain decimal seconds" instruction and drifts to
minute.second notation instead (e.g. "1.03" meaning 1:03 = 63.0s). JSON
parses that literally as the float 1.03, which is a huge *backward* jump
versus the previous segment's true end time — and if used as-is, produces an
EDL that references the wrong source frames (e.g. ~1s into the source
instead of ~63s) for the entire remainder of the audio.

This exact pattern was reproduced live against gemini-3.5-flash on a real
89.7s Egyptian-Arabic source video; the raw segment timestamps below are
copied verbatim from that response (see the "real captured drift" test).
"""
from __future__ import annotations

from itertools import pairwise

import pytest

from video_edit_agent.transcription.providers.gemini import (
    TimestampNormalizationError,
    _normalize_timestamp,
    _reinterpret_as_minute_dot_second,
    normalize_transcript_timestamps,
)


def _seg(start: float, end: float, text: str = "x", words: list[dict] | None = None) -> dict:
    return {"speaker": "0", "start": start, "end": end, "text": text, "words": words or []}


# ---------------------------------------------------------------------------
# _reinterpret_as_minute_dot_second / _normalize_timestamp (unit level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1.03, 63.0),
        (1.13, 73.0),
        (1.21, 81.0),
        (1.31, 91.0),
        (2.05, 125.0),
        (0.0, 0.0),
    ],
)
def test_reinterpret_as_minute_dot_second(raw, expected):
    assert _reinterpret_as_minute_dot_second(raw) == pytest.approx(expected)


def test_reinterpret_rejects_invalid_seconds_component():
    # A fractional part >= .60 cannot be MM:SS notation (no 60+ seconds).
    assert _reinterpret_as_minute_dot_second(1.75) is None


def test_normalize_timestamp_trusts_plain_seconds_when_monotonic():
    assert _normalize_timestamp(95.0, prev_end=53.0) == 95.0


def test_normalize_timestamp_reinterprets_backward_jump():
    # This is exactly the drift observed in production: after a segment
    # ending at 53.0s, the next raw value 1.03 is a backward jump unless
    # read as 1:03.
    assert _normalize_timestamp(1.03, prev_end=53.0) == pytest.approx(63.0)


def test_normalize_timestamp_unresolvable_returns_none():
    # Neither "0.5 seconds" nor any M.SS reading of 0.5 is consistent with
    # a previous end of 53.0s.
    assert _normalize_timestamp(0.5, prev_end=53.0) is None


# ---------------------------------------------------------------------------
# normalize_transcript_timestamps (integration level, no network)
# ---------------------------------------------------------------------------


def test_short_audio_under_60s_passes_through_unchanged():
    """No drift is possible/observed under a minute; values must be left as-is."""
    raw = [_seg(0.0, 10.0), _seg(10.0, 25.5), _seg(25.5, 40.0)]
    out = normalize_transcript_timestamps(raw, media_duration=41.0)
    assert [s["start"] for s in out] == [0.0, 10.0, 25.5]
    assert [s["end"] for s in out] == [10.0, 25.5, 40.0]


def test_real_captured_drift_pattern_normalizes_to_continuous_seconds():
    """Verbatim reproduction of the raw response captured from gemini-3.5-flash
    on the real 89.7s acceptance-test source (see module docstring)."""
    raw = [
        _seg(0.0, 10.5),
        _seg(10.5, 17.5),
        _seg(17.5, 26.0),
        _seg(26.0, 36.5),
        _seg(36.5, 45.5),
        _seg(45.5, 53.0),
        _seg(53.0, 1.03),  # drift begins: should resolve to 53.0 -> 63.0
        _seg(1.03, 1.13),  # 63.0 -> 73.0
        _seg(1.13, 1.21),  # 73.0 -> 81.0
        _seg(1.21, 1.31),  # 81.0 -> 91.0, clamped to media duration
    ]
    out = normalize_transcript_timestamps(raw, media_duration=89.708345)

    ends = [s["end"] for s in out]
    assert ends == pytest.approx([10.5, 17.5, 26.0, 36.5, 45.5, 53.0, 63.0, 73.0, 81.0, 89.708345])
    # Monotonic and strictly forward for the whole sequence.
    starts = [s["start"] for s in out]
    for a, b in zip(starts, ends):
        assert a <= b
    for prev, cur in pairwise(ends):
        assert cur >= prev
    # No segment references source frames near second 1 once drift is resolved.
    assert all(s["start"] > 5.0 or s is out[0] for s in out)


def test_word_level_timestamps_are_normalized_too():
    raw = [
        _seg(
            45.5, 53.0,
            words=[{"word": "a", "start": 45.5, "end": 47.0}, {"word": "b", "start": 47.0, "end": 53.0}],
        ),
        _seg(
            53.0, 1.03,
            words=[{"word": "c", "start": 53.0, "end": 1.0}, {"word": "d", "start": 1.0, "end": 1.03}],
        ),
    ]
    # media_duration=None here: this test is only about word-level
    # normalization, not the separate trailing-coverage check (covered by
    # test_truncated_trailing_coverage_is_rejected).
    out = normalize_transcript_timestamps(raw, media_duration=None)
    words = out[1]["words"]
    assert words[0]["start"] == pytest.approx(53.0)
    assert words[0]["end"] == pytest.approx(60.0)
    assert words[1]["start"] == pytest.approx(60.0)
    assert words[1]["end"] == pytest.approx(63.0)


def test_minor_end_overshoot_past_duration_is_clamped_not_rejected():
    raw = [_seg(0.0, 10.0), _seg(10.0, 91.0)]
    out = normalize_transcript_timestamps(raw, media_duration=89.708345)
    assert out[-1]["end"] == pytest.approx(89.708345)


def test_gross_overshoot_past_duration_is_rejected():
    raw = [_seg(0.0, 10.0), _seg(10.0, 500.0)]
    with pytest.raises(TimestampNormalizationError):
        normalize_transcript_timestamps(raw, media_duration=89.708345)


def test_ambiguous_unresolvable_timestamp_is_rejected_not_silently_used():
    """A value that is neither plausible as continuous seconds nor as an
    M.SS reinterpretation must fail loudly rather than build a broken EDL."""
    raw = [_seg(0.0, 53.0), _seg(0.5, 5.0)]
    with pytest.raises(TimestampNormalizationError):
        normalize_transcript_timestamps(raw, media_duration=89.708345)


def test_truncated_trailing_coverage_is_rejected():
    """If the resolved transcript stops well short of the media duration
    (e.g. the model just stopped transcribing), that's a silent-truncation
    failure distinct from the timestamp-format bug, and must also be caught."""
    raw = [_seg(0.0, 10.0), _seg(10.0, 20.0)]
    with pytest.raises(TimestampNormalizationError):
        normalize_transcript_timestamps(raw, media_duration=89.708345)


def test_short_clip_trailing_silence_is_not_treated_as_truncation():
    """A short clip whose speech ends well before the end still legitimately
    has silence/room tone in absolute terms even if that's a big fraction of
    a short clip (e.g. 2s clip, speech only in 0.4-0.8s -> 60% of the clip is
    trailing silence, but that's only 1.2s -- not a truncation)."""
    raw = [_seg(0.4, 0.8)]
    out = normalize_transcript_timestamps(raw, media_duration=2.0)
    assert out[-1]["end"] == pytest.approx(0.8)


def test_inverted_segment_is_rejected():
    raw = [_seg(0.0, 10.0), _seg(15.0, 12.0)]
    with pytest.raises(TimestampNormalizationError):
        normalize_transcript_timestamps(raw, media_duration=20.0)


def test_no_media_duration_still_normalizes_but_skips_duration_checks():
    raw = [_seg(53.0, 1.03), _seg(1.03, 1.13)]
    out = normalize_transcript_timestamps(raw, media_duration=None)
    assert out[0]["end"] == pytest.approx(63.0)
    assert out[1]["end"] == pytest.approx(73.0)
