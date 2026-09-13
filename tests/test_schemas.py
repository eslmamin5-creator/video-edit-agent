"""Unified transcript / EDL schema round-trip tests (spec section 7)."""
from __future__ import annotations

import json

from video_edit_agent.core.schemas import EDL, EDLClip, Transcript


def test_transcript_round_trips_through_json(sample_transcript):
    dumped = json.dumps(sample_transcript.model_dump(mode="json"), ensure_ascii=False)
    restored = Transcript.model_validate(json.loads(dumped))
    assert restored.provider == sample_transcript.provider
    assert restored.full_text == sample_transcript.full_text
    assert len(restored.words) == len(sample_transcript.words)


def test_transcript_is_provider_agnostic_shape(sample_transcript):
    # Nothing about the schema should depend on which provider produced it —
    # any provider string is a valid Transcript.
    for provider in ("gemini", "elevenlabs", "faster-whisper", "openai-whisper", "whisper.cpp"):
        t = sample_transcript.model_copy(update={"provider": provider})
        assert t.provider == provider
        assert t.full_text == sample_transcript.full_text


def test_edl_clip_duration_is_derived():
    clip = EDLClip(source_file="x.mp4", source_in=1.0, source_out=3.5, timeline_in=0.0, timeline_out=2.5)
    assert clip.duration == 2.5


def test_edl_total_duration_is_max_timeline_out():
    edl = EDL(clips=[
        EDLClip(source_file="a.mp4", source_in=0, source_out=1, timeline_in=0, timeline_out=1),
        EDLClip(source_file="a.mp4", source_in=1, source_out=3, timeline_in=1, timeline_out=3),
    ])
    assert edl.total_duration == 3


def test_edl_round_trips_through_json(sample_edl):
    dumped = json.dumps(sample_edl.model_dump(mode="json"))
    restored = EDL.model_validate(json.loads(dumped))
    assert restored.clips[0].source_file == sample_edl.clips[0].source_file
    assert restored.fps == sample_edl.fps
