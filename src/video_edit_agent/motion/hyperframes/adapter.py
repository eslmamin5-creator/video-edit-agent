"""HyperFrames motion engine adapter (spec section 18).

Baseline Recovery Milestone item 4 correction: there is no importable Python
`hyperframes` SDK -- the real `heygen-com/hyperframes` project is a Bun/Node
monorepo whose only stable, install-free integration surface is its
published npm CLI (`npx hyperframes@<pinned>`), confirmed runnable via
`hyperframes doctor`/`render --help` against this project's own ffmpeg/Chrome
install. This adapter shells out to that CLI against a bundled, hand-written
composition (`template/index.html`) built from the real HyperFrames
composition contract (`data-composition-id`, `data-composition-variables`,
`data-var-text`, a single paused GSAP timeline registered on
`window.__timelines`) -- see `heygen-com/hyperframes`
`skills/hyperframes-core/references/minimal-composition.md` and
`variables-and-media.md`.

Mirrors `motion/remotion/adapter.py`'s established pattern: copy the template
into a per-project cache dir once, resolve `.cmd` shims explicitly on
Windows, isolate the npm/npx package cache under that same per-project
directory (so a first render never touches the user's global npm cache or
installs anything system-wide), and raise `HyperFramesUnavailable` /
`HyperFramesRenderError` for the motion router to fall back on rather than
ever aborting the pipeline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from video_edit_agent.core.capability_router import detect_node, detect_npm
from video_edit_agent.core.schemas import AnimationSpec

_TEMPLATE_DIR = Path(__file__).parent / "template"

# Pinned for deterministic renders -- see adapter docstring for why `npx
# hyperframes@latest` is the wrong default (silent, unreviewed version drift).
_CLI_VERSION = "0.8.46"
_CLI_PACKAGE = f"hyperframes@{_CLI_VERSION}"

_STAT_KINDS = {"stat_counter", "metric_highlight"}
_LOWER_THIRD_KINDS = {"lower_third", "product_callout"}


class HyperFramesUnavailable(RuntimeError):
    pass


class HyperFramesRenderError(RuntimeError):
    pass


def is_available() -> bool:
    return detect_node().available and detect_npm().available and _TEMPLATE_DIR.exists()


def _project_cache_dir(project_root: Path) -> Path:
    cache_dir = project_root / "edit" / ".cache" / "hyperframes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _ensure_template_copied(project_root: Path) -> Path:
    cache_dir = _project_cache_dir(project_root)
    dest = cache_dir / "template"
    if not dest.exists():
        shutil.copytree(_TEMPLATE_DIR, dest)
    return dest


def _npm_cache_dir(project_root: Path) -> Path:
    npm_cache = _project_cache_dir(project_root) / "npm-cache"
    npm_cache.mkdir(parents=True, exist_ok=True)
    return npm_cache


def _cli_ready_marker(project_root: Path) -> Path:
    return _project_cache_dir(project_root) / f".cli_ready_{_CLI_VERSION}"


def _resolve(cmd: str) -> str:
    """Resolve a Node-toolchain command to its full path (see
    `remotion/adapter.py::_resolve` for the Windows `.cmd`-shim rationale)."""
    resolved = shutil.which(cmd)
    if not resolved:
        raise HyperFramesUnavailable(f"'{cmd}' not found on PATH")
    return resolved


def _layout_for_kind(spec: AnimationSpec) -> str:
    kind_value = spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind)
    if kind_value in _STAT_KINDS:
        return "stat"
    if kind_value in _LOWER_THIRD_KINDS:
        return "lower_third"
    return "title"


def _variables_for_spec(spec: AnimationSpec, fps: int) -> dict:
    duration = max(0.5, spec.timeline_end - spec.timeline_start)
    variables = {
        "title": spec.text,
        "subtitle": spec.subtext or "",
        "value": "" if spec.value is None else str(spec.value),
        "layout": _layout_for_kind(spec),
        "duration": duration,
    }
    accent = spec.extra.get("accent_color") or spec.extra.get("accent")
    if accent:
        variables["accent"] = accent
    return variables


def render(
    spec: AnimationSpec,
    project_root: Path,
    output_path: Path,
    fps: int = 30,
    slot_id: str = "slot",
    offline: bool = False,
) -> Path:
    """Render a single AnimationSpec to an alpha-transparent MOV (ProRes,
    yuva444p) via the real HyperFrames CLI.

    MOV/ProRes was chosen over WebM/VP9 after empirically verifying both:
    HyperFrames' WebM output reports `needsAlpha:true` and an ALPHA_MODE
    container tag, but this project's ffmpeg build only honours that alpha
    plane when the WebM input is decoded with an explicitly forced
    `-c:v libvpx-vp9` flag (the trick `heygen-com/hyperframes`'
    `skills/embedded-captions/scripts/render-and-composite.sh` itself relies
    on) -- a plain `-i file.webm` silently drops to opaque yuv420p. The MOV/
    ProRes output decodes as `yuva444p12le` via a plain `-i file.mov` with no
    special flags, which is what `render/composition.py`'s overlay filter
    graph actually does -- so MOV is the format that composites correctly
    with zero changes to the downstream pipeline.

    Raises HyperFramesUnavailable if Node/npm/the template are missing (or,
    in offline mode, if the CLI package isn't already cached locally), or
    HyperFramesRenderError if the render subprocess itself fails.
    """
    if not is_available():
        raise HyperFramesUnavailable("Node.js/npm or the HyperFrames template is not available")

    ready_marker = _cli_ready_marker(project_root)
    if offline and not ready_marker.exists():
        raise HyperFramesUnavailable(
            f"HyperFrames CLI ({_CLI_PACKAGE}) is not cached locally and --offline forbids "
            "network access to fetch it via npx."
        )

    template_dest = _ensure_template_copied(project_root)
    npx = _resolve("npx")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variables = _variables_for_spec(spec, fps)
    props_path = (_project_cache_dir(project_root) / f"_hf_variables_{slot_id}.json").resolve()
    props_path.write_text(json.dumps(variables, ensure_ascii=False, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["npm_config_cache"] = str(_npm_cache_dir(project_root))
    if offline:
        env["npm_config_offline"] = "true"
        env["npm_config_prefer_offline"] = "true"

    cmd = [
        npx,
        "--yes",
        _CLI_PACKAGE,
        "render",
        str(template_dest),
        "-o",
        str(output_path),
        "-f",
        str(fps),
        "--format",
        "mov",
        "--variables-file",
        str(props_path),
        "--quiet",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(template_dest),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HyperFramesRenderError(f"HyperFrames render timed out: {exc}") from exc
    finally:
        props_path.unlink(missing_ok=True)

    if result.returncode != 0 or not output_path.exists():
        stderr = result.stderr[-2000:]
        if offline:
            raise HyperFramesUnavailable(
                f"HyperFrames CLI is not cached locally and --offline forbids network access: {stderr}"
            )
        raise HyperFramesRenderError(f"HyperFrames render failed: {stderr}")

    ready_marker.touch()
    return output_path
