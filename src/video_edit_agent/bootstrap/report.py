"""Setup state persistence (spec section 18) and plain-text report rendering.

`.runtime/setup_state.json` is the only file this package writes outside the
venv itself. It is non-secret metadata only -- never an API key, never a
token -- so it is safe to read back, log, or ship in a bug report as-is.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from video_edit_agent import __version__ as PACKAGE_VERSION
from video_edit_agent.bootstrap.runtime import runtime_dir

SETUP_STATE_FILENAME = "setup_state.json"

# Keys that must never appear in persisted state, defense-in-depth against a
# future field accidentally carrying a secret-shaped value.
_FORBIDDEN_KEY_SUBSTRINGS = ("key", "token", "secret", "password", "credential")


@dataclass
class SetupState:
    setup_version: str = PACKAGE_VERSION
    os: str = ""
    python_executable: str = ""
    python_version: str = ""
    profile: str = ""
    ffmpeg_version: str | None = None
    node_version: str | None = None
    npm_version: str | None = None
    capability_summary: dict[str, str] = field(default_factory=dict)
    last_verified_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def state_path(project_root: Path) -> Path:
    return runtime_dir(project_root) / SETUP_STATE_FILENAME


def assert_no_secrets(data: dict) -> None:
    """Defensive check used by save_state -- raises if any key name looks
    secret-shaped, so a future field addition can't silently start
    persisting credentials."""
    for key in data:
        lowered = key.lower()
        if any(bad in lowered for bad in _FORBIDDEN_KEY_SUBSTRINGS):
            raise ValueError(f"refusing to persist setup_state field that looks like a secret: {key!r}")


def save_state(project_root: Path, state: SetupState) -> Path:
    data = state.to_dict()
    assert_no_secrets(data)
    path = state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_state(project_root: Path) -> SetupState | None:
    path = state_path(project_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    known_fields = {f for f in SetupState.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return SetupState(**filtered)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def platform_summary() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"
