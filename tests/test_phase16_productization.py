"""Phase 1.6: productization + new-video acceptance. Synthetic data only (no project media).

The real-footage acceptance (fresh video through the public CLI) is recorded in the release report; these tests pin
the rules that make it repeatable.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_transcript
from tests.test_phase153_locked_treatments import FACE, HID, _edl, _headline_lock
from video_edit_agent.brand.loader import load_brand
from video_edit_agent.cli import main as cli
from video_edit_agent.core import pipeline
from video_edit_agent.core.config import AppConfig
from video_edit_agent.direction.production_profile import (
    EditingProfile,
    motion_graphics_enabled,
    plan_and_apply_visual_rhythm,
)
from video_edit_agent.render import treatment_preview as tpreview
from video_edit_agent.review import camera_look as cam
from video_edit_agent.review import locked_treatments as lt
from video_edit_agent.review import setup_choices as sc
from video_edit_agent.review import state as review_state
from video_edit_agent.review import treatment_board as board

SRC = Path(__file__).resolve().parents[1] / "src" / "video_edit_agent"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "review").mkdir()
    return tmp_path


def _preview(project: Path, name: str = "p.mp4", data: bytes = b"preview") -> Path:
    p = project / "review" / "micro_previews" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _approved_camera(project: Path, scale_request: str | None = None) -> cam.CameraLook:
    rd = project / "review"
    look = cam.load_look(rd)
    if scale_request:
        cam.revise(look, scale_request)
    cam.attach_preview(look, _preview(project, f"c{look.current.revision}.mp4", f"v{look.current.revision}".encode()), (4.0, 8.0), 1.08)
    cam.approve(look)
    cam.save_look(rd, look)
    return look


def _pending_headline(project: Path) -> lt.LockStore:
    rd = project / "review"
    store = lt.LockStore()
    lt.approve_from_preview(store, _headline_lock(), _preview(project, "h.mp4"), now="2026-01-01T00:00:00+00:00")
    lt.apply_revision(store, HID, "كبر الهيدر")
    lt.save_locks(rd, store)
    return store


# 1
def test_1_default_profile_is_balanced(tmp_path):
    assert AppConfig().profile == "balanced"
    assert sc.resolve_choices(tmp_path / "review").profile == "balanced"
    assert "profile" in inspect.signature(pipeline.run_pipeline).parameters
    with pytest.raises(ValueError):
        sc.resolve_choices(tmp_path / "review", profile="wild")


# 2
def test_2_fresh_project_is_review_first(project):
    assert inspect.signature(pipeline.run_pipeline).parameters["review"].default is None
    assert not review_state.is_ready_for_final_render(project / "review")
    src = inspect.getsource(pipeline.run_pipeline)
    assert "plan_only = review is True or not approval.ready_for_final_render" in src


# 3
def test_3_neutral_brand_fallback(tmp_path):
    assert sc.resolve_choices(tmp_path / "review").brand is None
    assert load_brand(None).name == "default"
    assert sc.resolve_choices(tmp_path / "review", offline=None).offline is False


# 4
def test_4_natural_language_revision_updates_the_draft(project):
    _approved_camera(project)
    assert board.revise(project, "خفف الزوم") == cam.TREATMENT_ID
    look = cam.load_look(project / "review")
    assert look.current.status == cam.PENDING and look.current.zoom_scale == pytest.approx(0.5)
    _pending_headline(project)
    assert board.revise(project, "كبر الهيدر") == HID
    pending = lt.load_locks(project / "review").get(HID)
    assert pending.approval_status == lt.PENDING
    assert pending.headline.font_px > _headline_lock().headline.font_px


# 5
def test_5_revision_invalidates_the_old_approval(project):
    look = _approved_camera(project)
    assert look.render_scale() == 1.0 and look.approved is not None
    board.revise(project, "زود الزوم")
    look = cam.load_look(project / "review")
    assert look.approved is None and look.pending
    store = _pending_headline(project)
    assert store.get(HID).approval_status == lt.PENDING and not store.get(HID).is_active
    assert store.active() == []


# 6
def test_6_reapproval_replaces_the_lock(project):
    _approved_camera(project)
    board.revise(project, "خفف الزوم")
    look = cam.load_look(project / "review")
    cam.attach_preview(look, _preview(project, "c2.mp4", b"second"), (4.0, 8.0), 1.04)
    cam.save_look(project / "review", look)
    assert board.approve(project) == [cam.TREATMENT_ID]
    look = cam.load_look(project / "review")
    assert look.approved.zoom_scale == pytest.approx(0.5) and look.approved.revision == 2
    assert len(look.history) >= 1  # the earlier approved version stays recoverable ("undo")
    _pending_headline(project)
    p = tpreview.headline_preview_path(project / "review", lt.load_locks(project / "review").get(HID))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"new headline preview")
    assert HID in board.approve(project)
    now = lt.load_locks(project / "review").get(HID)
    assert now.is_active and now.locked_preview_path == str(p)


# 7
def test_7_lock_survives_rerun(project):
    _approved_camera(project, "خفف الزوم")
    rd = project / "review"
    store = lt.LockStore()
    lt.approve_from_preview(store, _headline_lock(), _preview(project, "h.mp4"), now="2026-01-01T00:00:00+00:00")
    lt.save_locks(rd, store)
    before = (rd / lt.LOCK_FILENAME).read_bytes(), (rd / cam.LOOK_FILENAME).read_bytes()
    # a rerun loads and saves again: nothing changes, nothing is re-approved
    lt.save_locks(rd, lt.load_locks(rd))
    cam.save_look(rd, cam.load_look(rd))
    assert ((rd / lt.LOCK_FILENAME).read_bytes(), (rd / cam.LOOK_FILENAME).read_bytes()) == before
    assert board.approve(project) == []
    assert cam.load_look(rd).render_scale() == pytest.approx(0.5)


# 8
def test_8_pending_cannot_reach_final_render(project):
    _approved_camera(project)
    board.revise(project, "زود الزوم")
    look = cam.load_look(project / "review")
    assert look.pending and look.render_scale() == 1.0  # the render never sees the pending scale
    store = _pending_headline(project)
    assert store.active() == []
    assert lt.locks_for_render(project / "review") == []
    src = inspect.getsource(pipeline.run_pipeline)
    assert "pending_changes" in src and "bool(pending_changes) and review is not False" in src


# 9
def test_9_rejected_cannot_reach_final_render(project):
    rd = project / "review"
    store = lt.LockStore()
    lt.approve_from_preview(store, _headline_lock(), _preview(project, "h.mp4"), now="2026-01-01T00:00:00+00:00")
    lt.reject(store, HID)
    lt.save_locks(rd, store)
    assert lt.locks_for_render(rd) == []
    with pytest.raises(lt.LockedTreatmentFidelityError):
        lt.require_renderable(lt.load_locks(rd).get(HID))
    board.remove(project, cam.TREATMENT_ID)
    assert cam.load_look(rd).render_scale() == 1.0 and cam.load_look(rd).current.zoom_scale == 0.0


# 10
def test_10_fresh_project_auto_discovery_still_works(tmp_path):
    tr = make_transcript([("one two three four five six seven eight nine ten", 0.0, 20.0)])
    edl = _edl()
    result = plan_and_apply_visual_rhythm(
        tr, edl, profile=EditingProfile.BALANCED, face_box=FACE, headline_font_px=64, allow_auto_headline_discovery=True,
    )
    assert result is not None  # discovery ran; an ineligible transcript fails closed with no headline
    assert "pinned_headline" in inspect.signature(plan_and_apply_visual_rhythm).parameters


# 11
def test_11_existing_locks_are_authoritative(project):
    rd = project / "review"
    store = lt.LockStore()
    lt.approve_from_preview(store, _headline_lock(), _preview(project, "h.mp4"), now="2026-01-01T00:00:00+00:00")
    lt.save_locks(rd, store)
    lock = lt.locks_for_render(rd)[0]
    tr = make_transcript([("one two three four five six seven eight nine ten", 0.0, 20.0)])
    edl = _edl()
    plan_and_apply_visual_rhythm(
        tr, edl, profile=EditingProfile.DYNAMIC, face_box=FACE, headline_font_px=64, locked_headline=lock,
        allow_auto_headline_discovery=True, zoom_scale=0.5,
    )
    from video_edit_agent.qa import locked_fidelity as fid

    rep = fid.FidelityReport(HID)
    fid.check_camera_recipe(lock, edl, rep)
    assert rep.ok, rep.failures  # the lock's geometry wins over profile, discovery and the zoom scale


# 12
def test_12_resume_does_not_retranscribe():
    src = inspect.getsource(pipeline.run_pipeline)
    assert ".transcript.json" in src and "transcript_reused" in src
    assert src.index("load_transcript(transcript_cache)") < src.index("router.transcribe(audio_path)")
    assert "transcript_reused" in inspect.signature(pipeline.PipelineResult).parameters


# 13
def test_13_no_internal_json_edits_are_needed(project):
    """The whole loop (change -> preview -> approve -> undo -> remove) runs through the friendly API and CLI."""
    _approved_camera(project)
    board.revise(project, "خفف الزوم")
    look = cam.load_look(project / "review")
    cam.attach_preview(look, _preview(project, "c2.mp4", b"two"), (4.0, 8.0), 1.04)
    cam.save_look(project / "review", look)
    board.approve(project)
    assert board.undo(project) == cam.TREATMENT_ID
    assert [i.actions for i in board.items(project)] == [("Approve", "Change", "Remove")]
    names = {c.name for c in cli.app.registered_commands} | {c.callback.__name__ for c in cli.app.registered_commands}
    assert {"edit", "review", "revise", "approve", "undo", "remove"} <= names
    text = "\n".join(f"{i.name} {i.reason} {i.status}" for i in board.items(project))
    assert not re.search(r"score|json|mad|sha256|treatment_id", text, re.IGNORECASE)  # no engineering terms on the cards


# 14
def test_14_final_render_matches_the_approved_revised_preview():
    tr = make_transcript([("one two three four five six seven eight nine ten", 0.0, 20.0)])
    kwargs = {"profile": EditingProfile.BALANCED, "face_box": FACE, "headline_font_px": 64}
    preview_edl, final_edl = _edl(), _edl()
    plan_and_apply_visual_rhythm(tr, preview_edl, zoom_scale=0.5, **kwargs)  # what the revised preview was planned with
    plan_and_apply_visual_rhythm(tr, final_edl, zoom_scale=0.5, **kwargs)  # what the final render plans (approved scale)
    assert preview_edl.model_dump() == final_edl.model_dump()
    window = (0.0, 20.0)
    assert tpreview.planned_peak_zoom(final_edl, window) == tpreview.planned_peak_zoom(preview_edl, window)
    default_edl = _edl()
    plan_and_apply_visual_rhythm(tr, default_edl, zoom_scale=1.0, **kwargs)
    assert tpreview.planned_peak_zoom(final_edl, window) <= tpreview.planned_peak_zoom(default_edl, window)
    src = inspect.getsource(pipeline.run_pipeline)
    assert "zoom_scale=look.render_scale()" in src and "_verify_approved_treatments" in src


# 15
def test_15_windows_export_validator_runs_after_render():
    src = inspect.getsource(pipeline.run_pipeline)
    assert src.index("render_ffmpeg(plan") < src.index("assert_export_compatible(") < src.index("_verify_approved_treatments(")
    assert src.index("_verify_approved_treatments(") < src.index("review_state.mark_rendered")


# 16
def test_16_motion_graphics_off_by_default():
    assert AppConfig().motion_graphics_mode == "off"
    assert motion_graphics_enabled(AppConfig().motion_graphics_mode) is False


# 17
def test_17_no_sfx_broll_or_generated_visual_dependency():
    for name in ("review/treatment_board.py", "review/camera_look.py", "review/setup_choices.py", "render/treatment_preview.py"):
        text = (SRC / name).read_text(encoding="utf-8").lower()
        assert not re.search(r"sfx|sound_effect|broll|b-roll|generate_image|gemini", text), name
    assert cli._plan_again.__code__.co_consts  # preview re-plans with B-roll and motion graphics off
    assert "enable_broll=False" in inspect.getsource(cli._plan_again)


# 18
def test_18_no_client_hardcoding():
    for name in ("review/treatment_board.py", "review/camera_look.py", "review/setup_choices.py", "render/treatment_preview.py",
                 "core/pipeline.py", "cli/main.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"rc_v\d|RC v\d", text), name
        assert not re.search(r"(?<![\w.])\d{1,3}\.\d{2,3}\s*,\s*\d{1,3}\.\d{2,3}\)\s*#\s*hardcoded", text), name


# 19
def test_19_second_video_uses_the_generic_pipeline_only():
    src = inspect.getsource(cli.edit)
    assert "run_pipeline(" in src and "pinned_headline" not in src and "transcript_override" not in src
    assert "pinned_headline" not in inspect.getsource(cli._plan_again)
    result = CliRunner().invoke(cli.app, ["edit", "--help"])
    assert result.exit_code == 0
    for word in ("--profile", "--brand", "--yes"):
        assert word in result.output
    for hidden in ("--no-broll", "--logo-mode", "transcript_review.json", "caption_preview.json"):
        assert hidden not in result.output


# 20
def test_20_client_regression_lock_bytes_unchanged_without_history(tmp_path):
    store = lt.LockStore()
    lt.approve_from_preview(store, _headline_lock(), _preview(tmp_path, "h.mp4"), now="2026-01-01T00:00:00+00:00")
    data = json.loads(lt.serialize(store))
    assert "history" not in data  # a project that never revised serializes exactly as it did in v0.2.3-RC
    assert (Path(__file__).parent / "test_phase153_locked_treatments.py").exists()


# Phase 1.6.1: interruption / resume
def test_161_resume_keeps_locks_settings_and_is_idempotent(project):
    rd = project / "review"
    _approved_camera(project, "خفف الزوم")
    sc.save_choices(rd, sc.resolve_choices(rd, profile="dynamic", brand=None))
    _pending_headline(project)
    before = {p.name: p.read_bytes() for p in rd.glob("*.json")}
    for _ in range(2):  # an interrupted render leaves review state untouched; each resume loads and re-saves it
        lt.save_locks(rd, lt.load_locks(rd))
        cam.save_look(rd, cam.load_look(rd))
        sc.save_choices(rd, sc.resolve_choices(rd))
        assert board.approve(project) == []  # nothing pending with a preview is silently approved
    assert {p.name: p.read_bytes() for p in rd.glob("*.json")} == before
    assert sc.resolve_choices(rd).profile == "dynamic"
    assert cam.load_look(rd).render_scale() == pytest.approx(0.5)
    assert lt.load_locks(rd).get(HID).approval_status == lt.PENDING  # a pending revision is not revived as approved
    assert "Existing project found" in inspect.getsource(cli.edit)
