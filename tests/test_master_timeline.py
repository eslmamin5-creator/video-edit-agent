"""MasterTimeline schema + EDL adapter (spec Phase 2 sections 5-6)."""
from __future__ import annotations

import json
from pathlib import Path

from video_edit_agent.core.schemas import EDL, CutReason, EDLClip, TransitionType
from video_edit_agent.core.timeline import (
    MasterTimeline,
    ProviderKind,
    TimelineItem,
    TrackType,
    edl_to_master_timeline,
    load,
    save,
)


def test_timeline_item_end_is_derived():
    item = TimelineItem(id="a", type=TrackType.VIDEO, start=1.0, duration=2.5, source="x.mp4")
    assert item.end == 3.5


def test_master_timeline_total_duration_is_max_item_end():
    timeline = MasterTimeline(
        items=[
            TimelineItem(id="a", type=TrackType.VIDEO, start=0.0, duration=2.0, source="a.mp4"),
            TimelineItem(id="b", type=TrackType.MUSIC, start=1.0, duration=5.0, source="b.mp3"),
        ]
    )
    assert timeline.total_duration == 6.0


def test_master_timeline_round_trips_through_json():
    timeline = MasterTimeline(
        workflow="creator",
        items=[TimelineItem(id="a", type=TrackType.MOTION_GRAPHICS, start=0.0, duration=1.0, source="gfx.png")],
    )
    dumped = json.dumps(timeline.model_dump(mode="json"), ensure_ascii=False)
    restored = MasterTimeline.model_validate(json.loads(dumped))
    assert restored.workflow == "creator"
    assert restored.items[0].type == TrackType.MOTION_GRAPHICS


def test_master_timeline_save_and_load_round_trip(tmp_path: Path):
    timeline = MasterTimeline(items=[TimelineItem(id="a", type=TrackType.VIDEO, start=0.0, duration=1.0, source="a.mp4")])
    path = tmp_path / "master_timeline.json"
    save(timeline, path)
    restored = load(path)
    assert restored == timeline


def test_edl_to_master_timeline_produces_one_video_and_audio_item_per_clip():
    edl = EDL(
        version=1, fps=30.0, width=1080, height=1920,
        clips=[
            EDLClip(
                source_file="a.mp4", source_in=1.0, source_out=3.0, timeline_in=0.0, timeline_out=2.0,
                speaker="host", reason=CutReason.MANUAL, transition_in=TransitionType.CROSSFADE,
            ),
            EDLClip(source_file="a.mp4", source_in=3.0, source_out=5.0, timeline_in=2.0, timeline_out=4.0),
        ],
    )

    timeline = edl_to_master_timeline(edl, workflow="editor")

    assert timeline.workflow == "editor"
    assert timeline.fps == edl.fps and timeline.width == edl.width and timeline.height == edl.height
    assert timeline.total_duration == edl.total_duration

    video_items = [i for i in timeline.items if i.type == TrackType.VIDEO]
    audio_items = [i for i in timeline.items if i.type == TrackType.AUDIO]
    assert len(video_items) == len(edl.clips) == 2
    assert len(audio_items) == len(edl.clips) == 2

    first = video_items[0]
    assert first.start == 0.0 and first.duration == 2.0
    assert first.in_point == 1.0 and first.out_point == 3.0
    assert first.speaker == "host"
    assert first.transition_in.type == TransitionType.CROSSFADE
    assert first.provenance.kind == ProviderKind.USER
    assert first.provenance.detail == "a.mp4"
