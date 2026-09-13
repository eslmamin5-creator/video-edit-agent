"""Remotion motion engine adapter (spec section 19).

Copies the bundled template into a per-project cache directory (so
`node_modules` can be installed once and reused across renders), writes the
animation props as JSON, then shells out to `npx remotion render`. Any
failure (missing Node/npm, missing deps, render error) raises
`RemotionUnavailable` / `RemotionRenderError` so the motion router can fall
back to another engine -- this adapter never raises an unhandled exception
that would abort the whole pipeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from video_edit_agent.core.capability_router import detect_node, detect_npm
from video_edit_agent.core.schemas import AnimationSpec

_TEMPLATE_DIR = Path(__file__).parent / "template"


class RemotionUnavailable(Exception):
    pass


class RemotionRenderError(Exception):
    pass


def is_available() -> bool:
    return detect_node().available and detect_npm().available and _TEMPLATE_DIR.exists()


def _project_cache_dir(project_root: Path) -> Path:
    cache_dir = project_root / "edit" / ".cache" / "remotion"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _ensure_template_copied(project_root: Path) -> Path:
    cache_dir = _project_cache_dir(project_root)
    dest = cache_dir / "template"
    if not dest.exists():
        shutil.copytree(_TEMPLATE_DIR, dest, ignore=shutil.ignore_patterns("node_modules"))
    return dest


def _resolve(cmd: str) -> str:
    """Resolve a Node-toolchain command to its full path.

    On Windows, npm/npx are `.cmd` shims; `subprocess.run(["npm", ...])`
    without going through the shell raises `FileNotFoundError` because
    `CreateProcess` doesn't apply PATHEXT resolution the way cmd.exe does.
    `shutil.which` does that resolution for us on every platform."""
    resolved = shutil.which(cmd)
    if not resolved:
        raise RemotionUnavailable(f"'{cmd}' not found on PATH")
    return resolved


def _ensure_deps_installed(template_dest: Path) -> None:
    if (template_dest / "node_modules").exists():
        return
    result = subprocess.run(
        [_resolve("npm"), "install", "--no-audit", "--no-fund"],
        cwd=str(template_dest),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RemotionRenderError(f"npm install failed: {result.stderr[-2000:]}")


def _props_for_spec(spec: AnimationSpec) -> dict:
    """Map the generic AnimationSpec fields onto the props each Remotion
    composition expects (see Root.tsx defaultProps for the shape)."""
    base = {"text": spec.text, "name": spec.text, "title": spec.text, "metric": spec.text, "label": spec.text}
    if spec.subtext:
        base.update({"subtitle": spec.subtext, "description": spec.subtext, "attribution": spec.subtext})
    if spec.value is not None:
        base.update({"value": spec.value})
    base.update(spec.extra)
    return base


def render(spec: AnimationSpec, project_root: Path, output_path: Path, fps: int = 30, slot_id: str = "slot") -> Path:
    """Render a single AnimationSpec to a transparent WebM via Remotion.

    Raises RemotionUnavailable if Node/npm/template are missing, or
    RemotionRenderError if the render itself fails.
    """
    if not is_available():
        raise RemotionUnavailable("Node.js/npm or Remotion template not available")

    template_dest = _ensure_template_copied(project_root)
    _ensure_deps_installed(template_dest)

    # The render subprocess is launched with cwd=template_dest (below), so any
    # relative path handed to it on the command line resolves against that
    # directory, not the caller's cwd. project_root/output_path may well be
    # relative (e.g. `videoedit edit sample.mp4` from the project dir) --
    # resolve to absolute paths before building the command.
    output_path = output_path.resolve()
    props = _props_for_spec(spec)
    props_path = (template_dest / f"_props_{slot_id}.json").resolve()
    props_path.write_text(json.dumps(props, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration_frames = max(1, round((spec.timeline_end - spec.timeline_start) * fps))
    # Remotion composition ids may only contain [a-zA-Z0-9-]; AnimationKind
    # values are snake_case (e.g. "hook_title"), so translate to match the
    # hyphenated ids registered in template/src/Root.tsx.
    kind_value = spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind)
    composition_id = kind_value.replace("_", "-")
    cmd = [
        _resolve("npx"),
        "remotion",
        "render",
        "src/index.ts",
        composition_id,
        str(output_path),
        f"--props={props_path}",
        f"--duration-in-frames={duration_frames}",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(template_dest),
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemotionRenderError(f"Remotion render timed out: {exc}") from exc
    finally:
        props_path.unlink(missing_ok=True)

    if result.returncode != 0 or not output_path.exists():
        raise RemotionRenderError(f"Remotion render failed: {result.stderr[-2000:]}")

    return output_path
