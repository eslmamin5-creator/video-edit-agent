"""Review-First ordering fix: `run_pipeline(review=True)` must only plan.
No Gemini/Veo B-roll generation, no motion rendering, no behind-subject
cutouts and no final render may happen before the user approves the plan,
while planned B-roll, planned motion and CTA detection still reach the
review artifacts. `review=False` must keep the normal render path."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_edit_agent.broll import planner as planner_mod
from video_edit_agent.core import pipeline as pipeline_mod
from video_edit_agent.core.media import MediaInfo
from video_edit_agent.core.schemas import (
    EDL,
    AnimationKind,
    AnimationSpec,
    BrollPlanItem,
    BrollSourceKind,
    EDLClip,
    MotionPlanItem,
)
from video_edit_agent.qa.repair import RepairResult


def _boom(name: str):
    def _raise(*_a, **_k):
        raise AssertionError(f"{name} must not be called")

    return _raise


@pytest.fixture
def three_clip_edl(sample_edl: EDL) -> EDL:
    """One clip per planned motion spec, since the timeline review reports one
    motion treatment per EDL clip."""
    src = sample_edl.clips[0].source_file
    clips = [
        EDLClip(source_file=src, source_in=float(i), source_out=i + 1.0, timeline_in=float(i),
                timeline_out=i + 1.0, caption_refs=[])
        for i in range(3)
    ]
    return sample_edl.model_copy(update={"clips": clips})


@pytest.fixture
def planned_specs() -> list[AnimationSpec]:
    return [
        AnimationSpec(kind=AnimationKind.HOOK_TITLE, timeline_start=0.0, timeline_end=1.0, text="hook"),
        AnimationSpec(
            kind=AnimationKind.STAT_COUNTER, timeline_start=1.0, timeline_end=2.0, text="stat",
            behind_subject=True,
        ),
        AnimationSpec(kind=AnimationKind.CTA, timeline_start=2.0, timeline_end=3.0, text="Join ACME"),
    ]


@pytest.fixture
def pipeline_env(monkeypatch, tmp_path: Path, sample_transcript, three_clip_edl: EDL, planned_specs):
    """Stubs cheap stages (probe/transcribe/EDL) but leaves plan_broll,
    the motion wiring and the review builders real."""
    sample_edl = three_clip_edl
    monkeypatch.setattr(pipeline_mod, "probe", lambda path, *_a, **_k: MediaInfo(
        path=Path(path), width=sample_edl.width, height=sample_edl.height, fps=sample_edl.fps,
        duration=sample_edl.total_duration, has_audio=True, video_codec="h264", audio_codec="aac",
    ))
    monkeypatch.setattr(pipeline_mod, "extract_audio", lambda *_a, **_k: None)

    class DummyRouter:
        def __init__(self, *_a, **_k):
            pass

        def transcribe(self, *_a, **_k):
            return sample_transcript

    monkeypatch.setattr(pipeline_mod, "TranscriptionRouter", DummyRouter)
    monkeypatch.setattr(pipeline_mod, "save_transcript", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "build_edl", lambda *_a, **_k: (sample_edl, []))
    monkeypatch.setattr(pipeline_mod, "write_takes_packed", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "validate_edl", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "write_captions", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "assert_export_compatible", lambda *_a, **_k: None)  # renderer is stubbed
    monkeypatch.setattr(pipeline_mod, "build_motion_plan", lambda *_a, **_k: list(planned_specs))
    monkeypatch.setattr(pipeline_mod, "resolve_brand_logo", lambda *_a, **_k: None)

    candidate = BrollPlanItem(
        timeline_start=0.2, timeline_end=1.2, purpose="context", spoken_concept="an office",
        recommended_visual="office establishing shot", prompt="a bright modern office",
    )
    monkeypatch.setattr(planner_mod, "select_broll_moments", lambda *_a, **_k: [candidate])
    monkeypatch.setattr(planner_mod.local, "find_broll", lambda item, *_a, **_k: item)

    # Preview frames are lightweight but need real media; not under test here.
    monkeypatch.setattr(pipeline_mod, "generate_preview_frames", _boom("generate_preview_frames"))

    monkeypatch.setattr(pipeline_mod, "run_technical_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_mod, "run_language_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_mod, "run_visual_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(pipeline_mod, "run_brand_qa", lambda *_a, **_k: [])
    monkeypatch.setattr(
        pipeline_mod, "run_repair_loop",
        lambda report, collect, repair_handlers: RepairResult(report=report, iterations_used=0, repairs_applied=[]),
    )

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"fake-source")
    return source_video


def test_review_true_never_renders_or_generates(monkeypatch, pipeline_env, planned_specs):
    monkeypatch.setattr(pipeline_mod, "render_motion", _boom("render_motion"))
    monkeypatch.setattr(pipeline_mod, "render_subject_cutout", _boom("render_subject_cutout"))
    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", _boom("render_ffmpeg"))
    monkeypatch.setattr(planner_mod.gemini_image, "generate_broll_image", _boom("gemini generation"))
    monkeypatch.setattr(planner_mod.veo, "generate_broll_video", _boom("veo generation"))

    # offline=False on purpose: review must force generation off by itself.
    result = pipeline_mod.run_pipeline(pipeline_env, offline=False, review=True)

    assert result.final_output is None
    assert result.ready_for_final_render is False
    review_dir = result.review_dir
    assert review_dir is not None
    for name in (
        "transcript_review.json", "caption_preview.json", "brand_summary.json",
        "timeline_review.json", "broll_review.json", "review_state.json",
    ):
        assert (review_dir / name).exists(), name

    # Motion specs are planned only.
    assert len(result.motion_plan) == len(planned_specs)
    assert all(m.engine_used is None and m.output_path is None for m in result.motion_plan)
    assert any(m.spec.behind_subject for m in result.motion_plan)

    timeline = json.loads((review_dir / "timeline_review.json").read_text(encoding="utf-8"))
    items = timeline["items"]
    assert [i["motion_treatment"] for i in items] == ["hook_title", "stat_counter", "cta"]
    assert [i["behind_subject"] for i in items] == [False, True, False]
    assert [i["cta_present"] for i in items] == [False, False, True]

    # B-roll is still planned (source recommendation + draft prompt), never generated.
    assert len(result.broll_plan) == 1
    assert result.broll_plan[0].source == BrollSourceKind.NONE
    broll = json.loads((review_dir / "broll_review.json").read_text(encoding="utf-8"))
    assert "office" in json.dumps(broll)
    assert "a bright modern office" in json.dumps(broll)

    brand_summary = json.loads((review_dir / "brand_summary.json").read_text(encoding="utf-8"))
    assert "Join ACME" in json.dumps(brand_summary, ensure_ascii=False)


def test_review_run_keeps_unresolved_transcript_items_open(monkeypatch, pipeline_env):
    """A review re-run must not forget what the user has not confirmed, must
    not approve B-roll generation, and must not flip the render gate."""
    from video_edit_agent.core.project import ProjectPaths
    from video_edit_agent.review import state as review_state
    from video_edit_agent.review.schemas import UnresolvedTranscriptItem

    monkeypatch.setattr(pipeline_mod, "render_motion", _boom("render_motion"))
    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", _boom("render_ffmpeg"))
    review_dir = ProjectPaths.for_source(pipeline_env).edit_dir / "review"
    review_state.flag_unresolved(review_dir, UnresolvedTranscriptItem(segment_id="s0", segment=1, reason="unclear"))

    result = pipeline_mod.run_pipeline(pipeline_env, offline=False, review=True)

    state = review_state.load_review_state(result.review_dir)
    assert [i.segment_id for i in state.unresolved_transcript] == ["s0"]
    assert state.ready_for_final_render is False
    assert state.broll_generation_approved is False
    assert result.final_output is None


def test_preview_frame_failure_is_a_warning_in_review_mode(monkeypatch, pipeline_env):
    """The lightweight preview step is the only media work allowed in review
    mode; a failure there is a warning, never a crash."""
    monkeypatch.setattr(pipeline_mod, "render_motion", _boom("render_motion"))
    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", _boom("render_ffmpeg"))

    result = pipeline_mod.run_pipeline(pipeline_env, offline=False, review=True)

    assert any("Preview frame generation skipped" in w for w in result.warnings)


def test_review_false_still_follows_normal_render_path(monkeypatch, pipeline_env, planned_specs, tmp_path: Path):
    calls: dict = {"motion": [], "cutout": 0, "render": 0, "allow_generation": []}

    def fake_render_motion(spec, *_a, **_k):
        calls["motion"].append(spec.kind)
        out = tmp_path / f"motion_{len(calls['motion'])}.webm"
        out.write_bytes(b"x")
        return MotionPlanItem(spec=spec, engine_used=None, output_path=str(out))

    def fake_cutout(*_a, **_k):
        calls["cutout"] += 1

    def fake_render(plan, output_path, _preset):
        calls["render"] += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp4")
        return output_path

    real_plan_broll = pipeline_mod.plan_broll

    def spy_plan_broll(*a, **k):
        calls["allow_generation"].append(k.get("allow_generation"))
        return real_plan_broll(*a, **k)

    monkeypatch.setattr(pipeline_mod, "render_motion", fake_render_motion)
    monkeypatch.setattr(pipeline_mod, "render_subject_cutout", fake_cutout)
    # A punched-in clip gets no cutout (it would not follow the reframe); keep this flow test unzoomed.
    monkeypatch.setattr(pipeline_mod, "plan_punch_ins", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", fake_render)
    monkeypatch.setattr(pipeline_mod, "plan_broll", spy_plan_broll)
    monkeypatch.setattr(planner_mod.gemini_image, "generate_broll_image", lambda item, *_a, **_k: item)
    monkeypatch.setattr(planner_mod.veo, "generate_broll_video", lambda item, *_a, **_k: item)

    result = pipeline_mod.run_pipeline(pipeline_env, offline=False, review=False)

    assert result.final_output is not None
    assert calls["render"] == 1
    assert len(calls["motion"]) == len(planned_specs)
    assert calls["cutout"] == 1  # the one behind_subject spec
    assert calls["allow_generation"] == [True]
    assert result.review_dir is None
    state = json.loads((pipeline_env.parent / "edit" / "review" / "review_state.json").read_text(encoding="utf-8"))
    assert state["bypassed"] is True  # the bypass is explicit and recorded


def test_review_mode_forces_generation_off_even_when_online(monkeypatch, pipeline_env):
    seen: list = []
    monkeypatch.setattr(pipeline_mod, "plan_broll", lambda *_a, **k: seen.append(k["allow_generation"]) or [])
    monkeypatch.setattr(pipeline_mod, "render_motion", _boom("render_motion"))
    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", _boom("render_ffmpeg"))

    pipeline_mod.run_pipeline(pipeline_env, offline=False, review=True)

    assert seen == [False]


def test_cta_detection_uses_enum_value_not_str(monkeypatch, pipeline_env):
    """Regression: `str(AnimationKind.CTA)` is 'AnimationKind.CTA', so the old
    `str(kind).endswith("cta")` check never matched and the brand summary lost
    the CTA text."""
    assert str(AnimationKind.CTA).endswith("cta") is False  # documents the original bug
    monkeypatch.setattr(pipeline_mod, "render_motion", _boom("render_motion"))
    monkeypatch.setattr(pipeline_mod, "render_ffmpeg", _boom("render_ffmpeg"))

    result = pipeline_mod.run_pipeline(pipeline_env, offline=True, review=True)

    summary = json.loads((result.review_dir / "brand_summary.json").read_text(encoding="utf-8"))
    assert "Join ACME" in json.dumps(summary, ensure_ascii=False)


def test_planned_motion_frames_get_a_placeholder_banner(tmp_path: Path, planned_specs):
    from PIL import Image

    from video_edit_agent.review.preview import annotate_planned_treatment

    items = [MotionPlanItem(spec=s) for s in planned_specs]
    labels = pipeline_mod._planned_motion_labels(
        items, {"hook": 0.5, "motion_graphic": 0.5, "behind_subject": 1.5, "cta": 2.5}
    )
    assert set(labels) == {"motion_graphic", "behind_subject", "cta"}
    assert "behind subject" in labels["behind_subject"]
    assert "not rendered" in labels["cta"]

    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (320, 568), (10, 10, 200)).save(frame)
    annotate_planned_treatment(frame, labels["cta"])
    with Image.open(frame) as img:
        assert img.getpixel((5, img.height - 40)) != (10, 10, 200)  # banner band drawn


def test_broll_review_shows_planned_visual_source_and_draft_prompt():
    """Review must show what would be generated (visual, source
    recommendation, draft prompt) even though nothing is generated yet."""
    from video_edit_agent.review.builder import build_broll_review

    later = BrollPlanItem(
        timeline_start=5.0, timeline_end=8.0, purpose="context", spoken_concept="later",
        recommended_visual="later visual",
    )
    earlier = BrollPlanItem(
        timeline_start=1.0, timeline_end=3.0, purpose="context", spoken_concept="an office",
        recommended_visual="office establishing shot", treatment="generated_broll", generate_later=True,
    )
    review = build_broll_review([later, earlier])

    first = review.items[0]
    assert (first.timeline_start, first.timeline_end) == (1.0, 3.0)  # chronological
    assert first.spoken_context == "an office"
    assert first.recommended_visual == "office establishing shot"
    assert first.source == "none"
    assert "generate" in first.source_recommendation
    assert first.asset_path is None
    assert "office establishing shot" in first.draft_prompt
    assert "No readable text" in first.draft_prompt
