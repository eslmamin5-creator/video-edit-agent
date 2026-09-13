"""Records real, executed acceptance verifications (spec V1.1 hardening
section 18): `videoedit doctor` must never say a capability is "Available"
just because an adapter file exists or a package imports cleanly. This is a
tiny local JSON record that acceptance scripts (scripts/*_acceptance.py)
write to when they genuinely produce and probe a real render, so doctor can
show "Verified render: yes (2026-09-13)" distinctly from "Installed: yes".

No secrets are ever stored here -- only capability names, booleans, and
timestamps.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STORE_PATH = Path("edit") / ".cache" / "verified_capabilities.json"


def record_verified(name: str, detail: str = "", store_path: Path = DEFAULT_STORE_PATH) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_all(store_path)
    data[name] = {"verified": True, "detail": detail, "at": datetime.now(timezone.utc).isoformat()}
    store_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_all(store_path: Path = DEFAULT_STORE_PATH) -> dict:
    if not store_path.exists():
        return {}
    try:
        return json.loads(store_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def is_verified(name: str, store_path: Path = DEFAULT_STORE_PATH) -> tuple[bool, str]:
    entry = load_all(store_path).get(name)
    if not entry:
        return False, "not yet verified by a real acceptance run"
    at = entry.get("at", "")
    return bool(entry.get("verified")), f"verified {at}" if at else "verified"
