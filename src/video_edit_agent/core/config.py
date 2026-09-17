"""Project + user configuration (spec section 25).

Config is layered: built-in defaults -> user config (~/.videoedit/config.yaml)
-> project config (<project>/edit/config.yaml) -> CLI flags. Nothing here ever
stores a secret; secrets always come from environment variables (section 26).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from video_edit_agent.core.env_file import resolve_secret

USER_CONFIG_DIR = Path(os.environ.get("VIDEOEDIT_HOME", Path.home() / ".videoedit"))
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.yaml"


class LocalTranscriptionConfig(BaseModel):
    engine: str = "faster-whisper"
    model: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"


class TranscriptionConfig(BaseModel):
    provider: str = "auto"
    priority: list[str] = Field(default_factory=lambda: ["gemini", "elevenlabs", "faster-whisper"])
    language: str = "auto"
    locale: str = "auto"
    mode: str = "verbatim"
    local: LocalTranscriptionConfig = Field(default_factory=LocalTranscriptionConfig)


class CloudConfig(BaseModel):
    enabled: bool = True


class GeminiConfig(BaseModel):
    transcription_model: str = "gemini-3.5-flash"
    vision_model: str = "gemini-3.5-flash"


class InterfaceConfig(BaseModel):
    language: str = "auto"  # auto | ar | en
    arabic_style: str = "simple"  # simple | egyptian | saudi | gulf | msa


class EditorialConfig(BaseModel):
    crossfade_ms: int = 30
    max_repair_loops: int = 2


class AppConfig(BaseModel):
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)
    editorial: EditorialConfig = Field(default_factory=EditorialConfig)
    offline: bool = False
    brand: str | None = None

    # ---- persistence -------------------------------------------------

    @classmethod
    def load(cls, project_dir: Path | None = None) -> AppConfig:
        data: dict[str, Any] = {}
        if USER_CONFIG_PATH.exists():
            data = _deep_merge(data, yaml.safe_load(USER_CONFIG_PATH.read_text(encoding="utf-8")) or {})
        if project_dir is not None:
            project_cfg = Path(project_dir) / "edit" / "config.yaml"
            if project_cfg.exists():
                data = _deep_merge(data, yaml.safe_load(project_cfg.read_text(encoding="utf-8")) or {})
        return cls.model_validate(data)

    def save_user(self) -> Path:
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        USER_CONFIG_PATH.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return USER_CONFIG_PATH

    def save_project(self, project_dir: Path) -> Path:
        edit_dir = Path(project_dir) / "edit"
        edit_dir.mkdir(parents=True, exist_ok=True)
        path = edit_dir / "config.yaml"
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------
# Secrets (spec section 26) — read-only accessors, never persisted anywhere.
# OS/process env wins; project-local `.env` (see core.env_file) is only a
# fallback. Neither of these ever mutates os.environ or logs a value.
# --------------------------------------------------------------------------


def get_gemini_key() -> str | None:
    return resolve_secret("GEMINI_API_KEY")


def get_elevenlabs_key() -> str | None:
    return resolve_secret("ELEVENLABS_API_KEY")
