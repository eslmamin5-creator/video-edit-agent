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

from video_edit_agent.brand.schema import Brand
from video_edit_agent.core.capability_router import detect_node, detect_npm
from video_edit_agent.core.schemas import AnimationSpec

_FALLBACK_FONT_FAMILY = "Arial, 'Segoe UI', sans-serif"

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
    else:
        _sync_template_sources(dest)
    return dest


def _sync_template_sources(dest: Path) -> None:
    """A cached template copy predates later template changes (a new component, a
    fixed prop): refresh every bundled source file whose content differs, so a
    render never runs an older component than the one the Python side targets.
    `node_modules` and staged assets are left alone."""
    for src in (_TEMPLATE_DIR / "src").rglob("*"):
        if not src.is_file():
            continue
        target = dest / "src" / src.relative_to(_TEMPLATE_DIR / "src")
        if not target.exists() or target.read_bytes() != src.read_bytes():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


_FONT_WEIGHTS = (("extralight", 200), ("light", 300), ("semibold", 600), ("bold", 700), ("regular", 400))


def _font_weight(name: str) -> int:
    lowered = name.lower()
    return next((w for key, w in _FONT_WEIGHTS if key in lowered), 400)


def _stage_brand_fonts(template_dest: Path, brand: Brand | None) -> list[dict]:
    """Copies the brand's font files under the template's `public/fonts` and
    returns the `theme.fontFiles` descriptors, so Remotion draws with the brand
    typeface the captions use (libass gets the same directory) instead of a
    system fallback. Empty when the brand ships no font files."""
    if brand is None or not brand.arabic_font:
        return []
    from video_edit_agent.brand.loader import resolve_fonts_dir

    fonts_dir = resolve_fonts_dir(brand.name)
    if fonts_dir is None:
        return []
    target = template_dest / "public" / "fonts"
    target.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for font in sorted(fonts_dir.iterdir()):
        if font.is_file() and font.suffix.lower() in {".ttf", ".otf", ".woff", ".woff2"}:
            shutil.copy2(font, target / font.name)
            out.append({"family": brand.arabic_font, "file": f"fonts/{font.name}", "weight": _font_weight(font.name)})
    return out


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


def _ensure_deps_installed(template_dest: Path, offline: bool = False) -> None:
    if (template_dest / "node_modules").exists():
        return
    if offline:
        raise RemotionUnavailable(
            "Remotion node_modules are not cached and installing them requires network "
            "access, which is disallowed in --offline mode."
        )
    result = subprocess.run(
        [_resolve("npm"), "install", "--no-audit", "--no-fund"],
        cwd=str(template_dest),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise RemotionRenderError(f"npm install failed: {result.stderr[-2000:]}")


def _theme_props(brand: Brand | None, font_files: list[dict] | None = None) -> dict | None:
    """Builds the `BrandTheme` prop shape (template/src/theme.ts) from a
    Brand Profile. Returns None when no brand is available at all, letting
    the template's own neutral `defaultTheme` apply.

    Before this, `_props_for_spec` never emitted a `theme` key at all, so
    every Remotion-rendered motion graphic (lower thirds, CTAs, stat
    counters, ...) silently used `defaultTheme`'s hardcoded colors regardless
    of the active Brand Profile -- one of the root causes of the "brand
    colors not applied consistently" regression (Review-First Editing
    Workflow spec section 3/11-B)."""
    if brand is None:
        return None
    theme = {
        "primary": brand.colors.primary,
        "secondary": brand.colors.secondary,
        # No arbitrary yellow fallback (spec section 3): fall back to the
        # brand's own secondary color, never invent a strong accent.
        "accent": brand.colors.accent or brand.colors.secondary,
        "fontFamily": brand.arabic_font or _FALLBACK_FONT_FAMILY,
        "rtl": brand.captions.rtl,
    }
    if font_files:
        theme["fontFiles"] = font_files
    return theme


def _props_for_spec(spec: AnimationSpec, brand: Brand | None = None, font_files: list[dict] | None = None) -> dict:
    """Map the generic AnimationSpec fields onto the props each Remotion
    composition expects (see Root.tsx defaultProps for the shape)."""
    base = {"text": spec.text, "name": spec.text, "title": spec.text, "metric": spec.text, "label": spec.text}
    if spec.subtext:
        base.update({"subtitle": spec.subtext, "description": spec.subtext, "attribution": spec.subtext})
    if spec.value is not None:
        base.update({"value": spec.value})
    theme = _theme_props(brand, font_files)
    if theme is not None:
        base["theme"] = theme
    base.update(spec.extra)
    return base


def render(
    spec: AnimationSpec,
    project_root: Path,
    output_path: Path,
    fps: int = 30,
    slot_id: str = "slot",
    offline: bool = False,
    brand: Brand | None = None,
) -> Path:
    """Render a single AnimationSpec to a transparent WebM via Remotion.

    Raises RemotionUnavailable if Node/npm/template are missing, or
    RemotionRenderError if the render itself fails. In offline mode, this
    raises RemotionUnavailable rather than installing node_modules over the
    network, so the motion router falls back to a fully local engine.
    """
    if not is_available():
        raise RemotionUnavailable("Node.js/npm or Remotion template not available")

    template_dest = _ensure_template_copied(project_root)
    _ensure_deps_installed(template_dest, offline=offline)
    font_files = _stage_brand_fonts(template_dest, brand)

    # The render subprocess is launched with cwd=template_dest (below), so any
    # relative path handed to it on the command line resolves against that
    # directory, not the caller's cwd. project_root/output_path may well be
    # relative (e.g. `videoedit edit sample.mp4` from the project dir) --
    # resolve to absolute paths before building the command.
    output_path = output_path.resolve()
    props = _props_for_spec(spec, brand, font_files)
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
        # None of the template compositions paint a background, so the
        # canvas is transparent -- but Remotion only preserves that in the
        # exported file when explicitly told to encode alpha. Without these
        # two flags it silently composites the transparent content onto an
        # opaque black canvas at encode time (confirmed via ffprobe:
        # codec_name=vp8, pix_fmt=yuv420p with no alpha plane at all), which
        # then overlays as a full-canvas black rectangle hiding whatever is
        # underneath -- not a downstream decode issue, since there is no
        # alpha to decode in the first place.
        "--codec=vp8",
        "--pixel-format=yuva420p",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(template_dest),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RemotionRenderError(f"Remotion render timed out: {exc}") from exc
    finally:
        props_path.unlink(missing_ok=True)

    if result.returncode != 0 or not output_path.exists():
        raise RemotionRenderError(f"Remotion render failed: {result.stderr[-2000:]}")

    return output_path


def render_still(
    spec: AnimationSpec,
    project_root: Path,
    output_path: Path,
    *,
    frame: int = 0,
    slot_id: str = "still",
    offline: bool = False,
    brand: Brand | None = None,
) -> Path:
    """Render one frame of a composition to a transparent PNG (the same
    composition, props and fonts `render` uses). The behind-subject planner reads
    the real glyph shapes back from such a still."""
    if not is_available():
        raise RemotionUnavailable("Node.js/npm or Remotion template not available")
    template_dest = _ensure_template_copied(project_root)
    _ensure_deps_installed(template_dest, offline=offline)
    font_files = _stage_brand_fonts(template_dest, brand)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    props_path = (template_dest / f"_props_{slot_id}.json").resolve()
    props_path.write_text(json.dumps(_props_for_spec(spec, brand, font_files), ensure_ascii=False), encoding="utf-8")
    kind_value = spec.kind.value if hasattr(spec.kind, "value") else str(spec.kind)
    cmd = [
        _resolve("npx"), "remotion", "still", "src/index.ts", kind_value.replace("_", "-"), str(output_path),
        f"--props={props_path}", f"--frame={frame}", "--image-format=png",
    ]
    try:
        result = subprocess.run(cmd, cwd=str(template_dest), capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RemotionRenderError(f"Remotion still timed out: {exc}") from exc
    finally:
        props_path.unlink(missing_ok=True)
    if result.returncode != 0 or not output_path.exists():
        raise RemotionRenderError(f"Remotion still failed: {result.stderr[-2000:]}")
    return output_path
