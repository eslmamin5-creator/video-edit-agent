"""Render-approval gate tests (Review-First Editing Workflow spec section 8):
final render must require explicit approval or an explicit bypass -- never
silently default to ready."""
from __future__ import annotations

from video_edit_agent.review.schemas import ReviewStage
from video_edit_agent.review.state import (
    approve,
    bypass,
    is_ready_for_final_render,
    load_review_state,
)


def test_fresh_project_is_not_ready_for_final_render(tmp_path):
    assert is_ready_for_final_render(tmp_path / "review") is False


def test_approve_flips_the_gate(tmp_path):
    review_dir = tmp_path / "review"
    approve(review_dir, note="looks good")
    assert is_ready_for_final_render(review_dir) is True
    state = load_review_state(review_dir)
    assert state.stage == ReviewStage.APPROVED
    assert state.bypassed is False
    assert "looks good" in state.notes


def test_bypass_flips_the_gate_and_records_it(tmp_path):
    review_dir = tmp_path / "review"
    bypass(review_dir, reason="--yes flag")
    state = load_review_state(review_dir)
    assert state.ready_for_final_render is True
    assert state.bypassed is True
    assert "--yes flag" in state.notes
