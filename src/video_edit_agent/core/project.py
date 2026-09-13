"""Project directory + project memory (spec sections 13, 14).

Every editing project lives next to its source media in an `edit/` folder.
Source media is never touched. `project.md` persists human-readable memory so
a future session (or a future agent) can resume reliably.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ProjectPaths:
    root: Path  # the folder containing the source video(s)
    edit_dir: Path
    cache_dir: Path
    verify_dir: Path

    @classmethod
    def for_source(cls, source: Path) -> "ProjectPaths":
        base = source if source.is_dir() else source.parent
        edit_dir = base / "edit"
        return cls(
            root=base,
            edit_dir=edit_dir,
            cache_dir=edit_dir / "cache",
            verify_dir=edit_dir / "verify",
        )

    def ensure(self) -> None:
        for d in (self.edit_dir, self.cache_dir, self.verify_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def transcript_raw(self) -> Path:
        return self.edit_dir / "transcript_raw.json"

    @property
    def transcript_unified(self) -> Path:
        return self.edit_dir / "transcript_unified.json"

    @property
    def transcript_txt(self) -> Path:
        return self.edit_dir / "transcript.txt"

    @property
    def takes_packed(self) -> Path:
        return self.edit_dir / "takes_packed.md"

    @property
    def edl_json(self) -> Path:
        return self.edit_dir / "edl.json"

    @property
    def project_md(self) -> Path:
        return self.edit_dir / "project.md"

    @property
    def master_srt(self) -> Path:
        return self.edit_dir / "master.srt"

    @property
    def preview_mp4(self) -> Path:
        return self.edit_dir / "preview.mp4"

    @property
    def final_mp4(self) -> Path:
        return self.edit_dir / "final.mp4"

    @property
    def visual_plan(self) -> Path:
        return self.edit_dir / "visual_plan.json"

    @property
    def motion_plan(self) -> Path:
        return self.edit_dir / "motion_plan.json"

    @property
    def broll_plan(self) -> Path:
        return self.edit_dir / "broll_plan.json"


@dataclass
class ProjectMemory:
    """Human-readable + machine-appendable memory backing `project.md`."""

    paths: ProjectPaths
    intent: str = ""
    preferences: dict[str, Any] = field(default_factory=dict)
    brand: str | None = None
    style: str | None = None
    decisions: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    source_inventory: list[str] = field(default_factory=list)
    render_history: list[str] = field(default_factory=list)
    qa_issues: list[str] = field(default_factory=list)
    outstanding: list[str] = field(default_factory=list)

    def log_decision(self, text: str) -> None:
        self.decisions.append(f"[{_now()}] {text}")

    def log_rejected(self, text: str) -> None:
        self.rejected.append(f"[{_now()}] {text}")

    def write(self) -> Path:
        self.paths.ensure()
        lines = [
            "# Project Memory",
            "",
            f"_Last updated: {_now()}_",
            "",
            "## Editorial Intent",
            self.intent or "_(none recorded)_",
            "",
            f"## Brand\n{self.brand or '_(none)_'}",
            "",
            f"## Style\n{self.style or '_(none)_'}",
            "",
            "## Preferences",
            _bullets([f"{k}: {v}" for k, v in self.preferences.items()]) or "_(none)_",
            "",
            "## Decisions",
            _bullets(self.decisions) or "_(none)_",
            "",
            "## Rejected Options",
            _bullets(self.rejected) or "_(none)_",
            "",
            "## Source Inventory",
            _bullets(self.source_inventory) or "_(none)_",
            "",
            "## Render History",
            _bullets(self.render_history) or "_(none)_",
            "",
            "## QA Issues",
            _bullets(self.qa_issues) or "_(none)_",
            "",
            "## Outstanding Work",
            _bullets(self.outstanding) or "_(none)_",
            "",
        ]
        self.paths.project_md.write_text("\n".join(lines), encoding="utf-8")
        return self.paths.project_md

    @classmethod
    def load_or_new(cls, paths: ProjectPaths) -> "ProjectMemory":
        # project.md is human-readable memory, not a strict machine format;
        # a JSON sidecar keeps it reliably resumable.
        sidecar = paths.edit_dir / ".project_memory.json"
        if sidecar.exists():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return cls(paths=paths, **data)
        return cls(paths=paths)

    def save(self) -> None:
        self.write()
        sidecar = self.paths.edit_dir / ".project_memory.json"
        payload = {
            "intent": self.intent,
            "preferences": self.preferences,
            "brand": self.brand,
            "style": self.style,
            "decisions": self.decisions,
            "rejected": self.rejected,
            "source_inventory": self.source_inventory,
            "render_history": self.render_history,
            "qa_issues": self.qa_issues,
            "outstanding": self.outstanding,
        }
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)
