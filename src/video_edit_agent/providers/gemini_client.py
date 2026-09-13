"""Low-level Gemini client (spec section 21): visual analysis, image
generation, and Veo video generation, all behind capability detection so the
rest of the pipeline can call these functions unconditionally and just catch
`GeminiUnavailable` to degrade gracefully (spec section 43). No API key is
ever read from anywhere but `core.config.get_gemini_key()` (environment
only, never persisted to disk).
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from video_edit_agent.core.config import get_gemini_key


class GeminiUnavailable(RuntimeError):
    pass


class GeminiRequestError(RuntimeError):
    pass


def is_available() -> bool:
    if not get_gemini_key():
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def _client():
    if not is_available():
        raise GeminiUnavailable(
            "Gemini is not available: missing GEMINI_API_KEY or google-genai is not installed "
            "(pip install video-edit-agent[gemini]). The caller should fall back to a local/offline path."
        )
    from google import genai  # type: ignore

    return genai.Client(api_key=get_gemini_key())


def analyze_video_segment(video_path: Path, prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Sends a short video clip plus a text prompt to Gemini for visual
    analysis (e.g. QA visual checks, B-roll concept extraction) and returns
    the raw text response."""
    client = _client()
    try:
        video_bytes = video_path.read_bytes()
        response = client.models.generate_content(
            model=model,
            contents=[
                {"inline_data": {"mime_type": "video/mp4", "data": base64.b64encode(video_bytes).decode("ascii")}},
                prompt,
            ],
        )
        return response.text or ""
    except Exception as exc:  # noqa: BLE001
        raise GeminiRequestError(f"Gemini video analysis failed: {exc}") from exc


def generate_image(prompt: str, output_path: Path, model: str = "gemini-2.5-flash-image") -> Path:
    """Generates a single image from a text prompt and writes it to
    `output_path`. Used as a B-roll fallback source (spec section 20) when no
    matching local/user footage is found."""
    client = _client()
    try:
        response = client.models.generate_content(model=model, contents=[prompt])
        for part in response.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(inline.data)
                return output_path
        raise GeminiRequestError("Gemini response contained no image data")
    except GeminiRequestError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GeminiRequestError(f"Gemini image generation failed: {exc}") from exc


def generate_video(prompt: str, output_path: Path, model: str = "veo-3.0-generate-001", **kwargs: Any) -> Path:
    """Generates a short B-roll video clip via Veo. This is the most
    expensive/slowest B-roll fallback tier (spec section 20's priority
    order) -- callers should only reach here after user assets, local
    library, project assets, and generated images have all failed."""
    client = _client()
    try:
        operation = client.models.generate_videos(model=model, prompt=prompt, **kwargs)
        import time

        while not operation.done:
            time.sleep(5)
            operation = client.operations.get(operation)

        video = operation.response.generated_videos[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        client.files.download(file=video.video)
        video.video.save(str(output_path))
        return output_path
    except Exception as exc:  # noqa: BLE001
        raise GeminiRequestError(f"Veo video generation failed: {exc}") from exc
