"""Baseline Recovery Milestone item 2: default generated-B-roll prompts must
request photorealism (no glossy/plastic AI look) and prohibit readable
text, and both the image and video generation providers must build their
prompt through the shared rules instead of a bare `f"{concept}. Cinematic"`
template with no safety guidance."""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.broll.prompt import build_broll_prompt, looks_like_usable_asset
from video_edit_agent.broll.providers import gemini_image, veo
from video_edit_agent.core.schemas import BrollPlanItem, BrollSourceKind


def _item(prompt: str | None = None) -> BrollPlanItem:
    return BrollPlanItem(
        timeline_start=0.0, timeline_end=2.0, purpose="context",
        spoken_concept="a busy office", recommended_visual="office establishing shot",
        prompt=prompt,
    )


def test_default_prompt_requests_photorealism_and_prohibits_text():
    generated = build_broll_prompt(_item())
    assert "office establishing shot" in generated
    for phrase in ("Photorealistic", "glossy/plastic", "No readable text"):
        assert phrase in generated


def test_prompt_rules_apply_even_with_brand_or_caller_supplied_prompt():
    generated = build_broll_prompt(_item(prompt="Concept. Style: warm and minimal."))
    assert "Style: warm and minimal." in generated
    assert "No readable text" in generated


def test_looks_like_usable_asset_rejects_near_empty_files():
    assert looks_like_usable_asset(0) is False
    assert looks_like_usable_asset(100) is False
    assert looks_like_usable_asset(50_000) is True


def test_gemini_image_provider_uses_shared_prompt_and_rejects_empty_output(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gemini_image, "is_available", lambda: True)
    captured = {}

    def fake_generate_image(prompt: str, output_path: Path):
        captured["prompt"] = prompt
        output_path.write_bytes(b"")  # simulates a failed/filtered generation
        return output_path

    monkeypatch.setattr(gemini_image, "_generate_image", fake_generate_image)
    result = gemini_image.generate_broll_image(_item(), tmp_path)

    assert "No readable text" in captured["prompt"]
    assert result.source == BrollSourceKind.NONE  # rejected: empty file


def test_gemini_image_provider_accepts_real_output(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gemini_image, "is_available", lambda: True)

    def fake_generate_image(prompt: str, output_path: Path):
        output_path.write_bytes(b"x" * 10_000)
        return output_path

    monkeypatch.setattr(gemini_image, "_generate_image", fake_generate_image)
    result = gemini_image.generate_broll_image(_item(), tmp_path)
    assert result.source == BrollSourceKind.GENERATED_IMAGE
    assert result.asset_path


def test_veo_provider_uses_shared_prompt(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(veo, "is_available", lambda: True)
    captured = {}

    def fake_generate_video(prompt: str, output_path: Path):
        captured["prompt"] = prompt
        output_path.write_bytes(b"x" * 10_000)
        return output_path

    monkeypatch.setattr(veo, "_generate_video", fake_generate_video)
    result = veo.generate_broll_video(_item(), tmp_path)

    assert "Photorealistic" in captured["prompt"]
    assert result.source == BrollSourceKind.GENERATED_VIDEO
