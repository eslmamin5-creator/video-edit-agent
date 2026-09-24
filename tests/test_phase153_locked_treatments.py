"""Phase 1.5.3: approved treatment locking + final-render fidelity. Synthetic data only (no project media)."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tests.conftest import make_transcript
from video_edit_agent.captions.styles import CaptionStyle
from video_edit_agent.core import pipeline
from video_edit_agent.core.config import AppConfig
from video_edit_agent.core.schemas import EDL, AnimationKind, EDLClip
from video_edit_agent.direction.production_profile import (
    EditingProfile,
    burn_locked_headline,
    discover_and_verify_headline,
    motion_graphics_enabled,
    plan_and_apply_visual_rhythm,
)
from video_edit_agent.qa import locked_fidelity as fid
from video_edit_agent.review import locked_treatments as lt

W, H = 1080, 1920
GEOMETRY = {"status": "ok", "head_top_base": 0.2896, "face_top_base": 0.328, "face_bottom_base": 0.4115, "zoom": 1.2245, "anchor_x": 0.5, "anchor_y": 0.0, "base_headroom": 0.2896, "target_headroom": 0.3546, "delta": 0.065, "face_top_target": 0.4017, "face_bottom_target": 0.5039}
FACE = (0.4726, 0.328, 0.1474, 0.0835)
HID, BID = "lower_subject_semantic@x", "behind_subject_text@y"


def _edl(seconds: int = 20) -> EDL:
    return EDL(width=W, height=H, clips=[
        EDLClip(source_file="a.mp4", source_in=i * 2.0, source_out=(i + 1) * 2.0, timeline_in=i * 2.0, timeline_out=(i + 1) * 2.0)
        for i in range(seconds // 2)
    ])


def _headline_lock() -> lt.LockedTreatment:
    recipe = lt.HeadlineRecipe(
        font_family="Arial", font_px=190, y_px=201.6, slide_px=12.0, settle_from=0.96, fade_in=(5.0, 5.3), fade_out=(8.0, 8.3),
        caption_reduction_window=(5.0, 8.3), changes=(4.5, 5.5, 8.0, 9.0), geometry=GEOMETRY, zoom=1.2245, anchor_x=0.5,
        anchor_y=0.0, face_box=FACE, head_top=0.19,
    )
    return lt.LockedTreatment(
        treatment_id=HID, treatment_type=lt.TREATMENT_HEADLINE, text="Big Idea", start=5.0, end=8.3, headline=recipe,
    )


def _behind_lock() -> lt.LockedTreatment:
    recipe = lt.BehindSubjectRecipe(
        font_px=140, lines=("value", "real"), center_x=540.0, center_y=700.0, bounds=(100.0, 500.0, 980.0, 900.0), color="#FFFFFF",
        outline={"w": 4}, occlusion={"meaningful_occlusion": 0.31}, slot=3,
    )
    return lt.LockedTreatment(
        treatment_id=BID, treatment_type=lt.TREATMENT_BEHIND, text="value real", start=12.0, end=14.0, behind=recipe,
    )


@pytest.fixture
def preview(tmp_path: Path) -> Path:
    p = tmp_path / "preview.mp4"
    p.write_bytes(b"approved-preview-bytes")
    return p


def _approved(lock: lt.LockedTreatment, preview: Path, store: lt.LockStore | None = None) -> lt.LockStore:
    store = store or lt.LockStore()
    lt.approve_from_preview(store, lock, preview, now="2026-01-01T00:00:00+00:00")
    return store


def _base_ass() -> str:
    return (
        "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
        "Style: Default,Arial,64\n\n[Events]\nFormat: Layer, Start, End, Style, Text\n"
        "Dialogue: 0,0:00:06.00,0:00:07.00,Default,,0,0,0,,hello\n"
    )


def test_1_approved_preview_creates_a_lock(preview):
    lock = _approved(_headline_lock(), preview).get(HID)
    assert lock.is_active and lock.locked_from_preview and lock.locked_at


def test_2_lock_preserves_recipe_through_disk(preview, tmp_path):
    lt.save_locks(tmp_path, _approved(_headline_lock(), preview))
    back = lt.load_locks(tmp_path).get(HID)
    assert (back.text, back.start, back.end) == ("Big Idea", 5.0, 8.3)
    h = back.headline
    assert (h.font_px, h.y_px, h.zoom, h.anchor_y, h.changes) == (190, 201.6, 1.2245, 0.0, (4.5, 5.5, 8.0, 9.0))
    assert h.geometry == GEOMETRY and h.fade_in == (5.0, 5.3) and h.caption_role == "reduced"


def test_3_burn_uses_locked_font_and_position_not_defaults(preview):
    lock = _approved(_headline_lock(), preview).get(HID)
    out = burn_locked_headline(_base_ass(), _base_ass(), lock, CaptionStyle(name="t"), W)
    found = fid._headline_event(out)
    assert found is not None and int(found[0]) == 190
    rep = fid.FidelityReport(HID)
    fid.check_headline_recipe(lock, out, rep)
    assert rep.ok, rep.failures


def test_4_discovery_returns_the_lock_verbatim(preview):
    lock = _approved(_headline_lock(), preview).get(HID)
    wiring = discover_and_verify_headline(
        [], profile=EditingProfile.BALANCED, face_box=None, width=W, height=H, headline_font_px=64, locked=lock,
    )
    assert wiring.composition.phrase == "Big Idea" and wiring.composition.changes == [4.5, 5.5, 8.0, 9.0]
    assert wiring.composition.semantic_source == "locked"
    assert _approved(_behind_lock(), preview).owns(13.0, 15.0)


def test_5_pending_and_rejected_never_render(preview):
    store = _approved(_headline_lock(), preview)
    lt.edit_lock(store, HID, lambda lk: setattr(lk.headline, "font_px", 200))
    assert store.active() == []
    with pytest.raises(lt.LockedTreatmentFidelityError):
        lt.require_renderable(store.get(HID))
    lt.reject(store, HID)
    assert store.active() == [] and store.get(HID).approval_status == lt.REJECTED


def test_6_edit_returns_to_pending_review(preview):
    store = _approved(_headline_lock(), preview)
    draft = lt.apply_revision(store, HID, "كبر الهيدر")
    assert draft.approval_status == lt.PENDING and draft.headline.font_px == round(190 * 1.2) and draft.draft_of


def test_7_reapproval_replaces_lock(preview):
    store = _approved(_headline_lock(), preview)
    lt.apply_revision(store, HID, "نزله شوية")
    new = lt.approve_from_preview(store, store.get(HID), preview, now="2026-02-01T00:00:00+00:00")
    assert len(store.locks) == 1 and new.is_active and new.revision == 2 and new.headline.y_px > 201.6


def test_8_headline_size_and_placement_survive(preview):
    lock = _approved(_headline_lock(), preview).get(HID)
    size, y = fid._headline_event(burn_locked_headline(_base_ass(), _base_ass(), lock, CaptionStyle(name="t"), W))
    assert int(size) == lock.headline.font_px and abs(y - lock.headline.y_px) <= lock.headline.slide_px + fid.Y_TOL_PX


def test_9_lower_subject_geometry_survives_camera_apply(preview):
    lock = _approved(_headline_lock(), preview).get(HID)
    tr = make_transcript([("one two three four five six seven eight nine ten", 0.0, 20.0)])
    edl = _edl()
    plan_and_apply_visual_rhythm(
        tr, edl, profile=EditingProfile.BALANCED, face_box=FACE, headline_font_px=64, locked_headline=lock,
    )
    rep = fid.FidelityReport(HID)
    fid.check_camera_recipe(lock, edl, rep)
    assert rep.ok, rep.failures


def test_10_behind_recipe_survives_into_the_spec(preview, tmp_path):
    lock = _approved(_behind_lock(), preview).get(BID)
    spec = lt.behind_subject_spec(lock)
    assert spec.behind_subject and spec.extra["placement"]["meaningful_occlusion"] == 0.31
    plan = tmp_path / "motion_plan.json"
    plan.write_text(json.dumps([{"spec": spec.model_dump(mode="json")}]), encoding="utf-8")
    rep = fid.FidelityReport(BID)
    fid.check_behind_recipe(lock, plan, rep)
    assert rep.ok, rep.failures


def test_11_layer_order_is_cutout_above_text(preview):
    lock = _approved(_behind_lock(), preview).get(BID)
    order = list(lock.behind.layer_order)
    assert order.index("subject_cutout") > order.index("text_graphic")
    broken = lock.model_copy(deep=True)
    broken.behind.layer_order = ("base_video", "subject_cutout", "text_graphic", "captions")
    assert any("layer order" in p for p in lt.validate_lock(broken))


def test_12_broken_lock_fails_with_error_code(preview, tmp_path):
    bad = _headline_lock()
    bad.headline.changes = (9.0, 5.5, 8.0, 4.5)
    lt.save_locks(tmp_path, _approved(bad, preview))
    with pytest.raises(lt.LockedTreatmentFidelityError, match=lt.FIDELITY_ERROR_CODE):
        lt.locks_for_render(tmp_path, total_duration=20.0, frame_size=(W, H))
    with pytest.raises(lt.LockedTreatmentFidelityError):
        lt.require_renderable(_approved(_headline_lock(), preview).get(HID), total_duration=6.0)
    assert "locks_for_render" in inspect.getsource(pipeline.run_pipeline)  # the pipeline never catches it silently
    assert "except LockedTreatmentFidelityError" not in inspect.getsource(pipeline)


def test_13_provenance_points_to_preview(preview):
    lock = _approved(_headline_lock(), preview).get(HID)
    assert lock.locked_preview_path == str(preview)
    assert lock.locked_preview_sha256 == hashlib.sha256(preview.read_bytes()).hexdigest()
    preview.unlink()
    assert any("provenance" in p for p in lt.validate_lock(lock))


def test_14_serialization_is_deterministic(preview):
    a, b = lt.LockStore(), lt.LockStore()
    for store, order in ((a, (_headline_lock(), _behind_lock())), (b, (_behind_lock(), _headline_lock()))):
        for lk in order:
            lt.approve_from_preview(store, lk, preview, now="2026-01-01T00:00:00+00:00")
    assert lt.serialize(a) == lt.serialize(b)


def test_15_reapproving_identical_recipe_is_idempotent(preview, tmp_path):
    lt.save_locks(tmp_path, _approved(_headline_lock(), preview))
    first = lt.lock_path(tmp_path).read_bytes()
    again = lt.load_locks(tmp_path)
    lt.approve_from_preview(again, _headline_lock(), preview, now="2027-05-05T00:00:00+00:00")
    lt.save_locks(tmp_path, again)
    assert lt.lock_path(tmp_path).read_bytes() == first and again.get(HID).revision == 1


def test_16_no_lock_file_means_no_locks_and_discovery_stays(tmp_path):
    assert lt.load_locks(tmp_path).locks == [] and lt.locks_for_render(tmp_path, total_duration=20.0) == []
    wiring = discover_and_verify_headline([], profile=EditingProfile.BALANCED, face_box=FACE, width=W, height=H, headline_font_px=64)
    assert not wiring.ok


def test_17_motion_graphics_stay_off():
    assert not motion_graphics_enabled(AppConfig().motion_graphics_mode)


def test_18_locks_only_add_headline_and_behind_specs(preview):
    spec = lt.behind_subject_spec(_approved(_behind_lock(), preview).get(BID))
    assert spec.kind == AnimationKind.BEHIND_TEXT
    assert set(lt.LOCKABLE_TREATMENTS) == {"lower_subject_semantic", "behind_subject_text"}
    text = inspect.getsource(lt).lower()
    assert "sfx" not in text and "broll" not in text


def test_19_export_gate_still_wired():
    src = inspect.getsource(pipeline)
    assert "assert_export_compatible" in src and "locks_for_render" in src


def test_20_no_project_specific_hardcoding():
    for mod in (lt, fid):
        text = inspect.getsource(mod)
        for token in ("Five Cs", "five-cs", "haqiqia", "محمد", "حقيقية"):
            assert token not in text, f"{token!r} in {mod.__name__}"
