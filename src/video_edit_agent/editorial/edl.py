"""EDL persistence + validation (spec section 12)."""
from __future__ import annotations

import json
from pathlib import Path

from video_edit_agent.core.schemas import EDL


class EDLValidationError(RuntimeError):
    pass


def validate(edl: EDL) -> None:
    if not edl.clips:
        raise EDLValidationError("EDL has no clips")
    for i, c in enumerate(edl.clips):
        if c.source_out <= c.source_in:
            raise EDLValidationError(f"clip {i}: source_out <= source_in")
        if c.timeline_out <= c.timeline_in:
            raise EDLValidationError(f"clip {i}: timeline_out <= timeline_in")
        if not Path(c.source_file).exists():
            raise EDLValidationError(f"clip {i}: source file missing: {c.source_file}")


def save(edl: EDL, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(edl.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> EDL:
    return EDL.model_validate(json.loads(path.read_text(encoding="utf-8")))
