"""Brand-driven intro/end-card renderer.

Everything visual comes from a `CardSpec` built from the active Brand Profile
(`brand/logo_policy.py`): background colors, logo asset, reveal style,
duration. This module contains no client-specific values.
"""
from __future__ import annotations

from pathlib import Path

from video_edit_agent.brand.logo_policy import CardSpec
from video_edit_agent.core.media import probe_duration, run


class EndCardError(RuntimeError):
    pass


def _hex_rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def write_background_png(spec: CardSpec, path: Path) -> Path:
    """Solid or vertical-gradient background from the card's brand colors."""
    from PIL import Image

    top, bottom = _hex_rgb(spec.background_top), _hex_rgb(spec.background_bottom)
    if top == bottom:
        img = Image.new("RGB", (spec.width, spec.height), top)
    else:
        column = Image.new("RGB", (1, spec.height))
        for y in range(spec.height):
            f = y / max(1, spec.height - 1)
            column.putpixel((0, y), tuple(round(t + (b - t) * f) for t, b in zip(top, bottom, strict=True)))
        img = column.resize((spec.width, spec.height))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _write_accent_png(spec: CardSpec, path: Path) -> Path:
    from PIL import Image

    w = max(2, round(spec.width * spec.logo_width_pct * 0.35))
    h = max(2, round(spec.height * 0.0035))
    img = Image.new("RGBA", (w, h), (*_hex_rgb(spec.accent or "#FFFFFF"), 255))
    img.save(path)
    return path


def _ramp(spec: CardSpec) -> float:
    return min(0.8, spec.duration * 0.4)


def _build_cmd(spec: CardSpec, workdir: Path) -> tuple[list[str], str]:
    """(ffmpeg input args, filter_complex ending in [v]) for one card."""
    if spec.logo_path is None:
        raise EndCardError("card has no logo asset")
    bg = write_background_png(spec, workdir / f"{spec.kind}_bg.png")
    fps = f"{spec.fps:g}"
    d = f"{spec.duration:.3f}"
    inputs = [
        "-loop", "1", "-framerate", fps, "-t", d, "-i", str(bg),
        "-loop", "1", "-framerate", fps, "-t", d, "-i", str(spec.logo_path),
    ]
    logo_w = round(spec.width * spec.logo_width_pct)
    ramp = _ramp(spec)
    if spec.reveal == "none":
        logo = f"[1:v]format=rgba,scale={logo_w}:-2[logo]"
    elif spec.reveal == "fade":
        logo = f"[1:v]format=rgba,scale={logo_w}:-2,fade=t=in:st=0.1:d={ramp:.3f}:alpha=1[logo]"
    else:  # subtle: opacity reveal plus a small scale settle (94% -> 100%)
        logo = (
            f"[1:v]format=rgba,scale=w='2*trunc({logo_w}*(0.94+0.06*min(t/{ramp:.3f},1))/2)':h=-2:eval=frame,"
            f"fade=t=in:st=0.1:d={ramp:.3f}:alpha=1[logo]"
        )
    graph = [logo, "[0:v][logo]overlay=x=(W-w)/2:y=(H-h)/2:format=auto[c1]"]
    last = "c1"
    if spec.accent_style == "underline" and spec.accent:
        acc = _write_accent_png(spec, workdir / f"{spec.kind}_accent.png")
        inputs += ["-loop", "1", "-framerate", fps, "-t", d, "-i", str(acc)]
        graph.append(f"[2:v]format=rgba,fade=t=in:st={ramp:.3f}:d={ramp:.3f}:alpha=1[acc]")
        from PIL import Image

        with Image.open(spec.logo_path) as im:
            logo_h = round(logo_w * im.height / im.width)
        y = round(spec.height / 2 + logo_h / 2 + spec.height * 0.03)
        graph.append(f"[c1][acc]overlay=x=(W-w)/2:y={y}[c2]")
        last = "c2"
    graph.append(f"[{last}]format=yuv420p,setsar=1[v]")
    return inputs, ";".join(graph)


def render_card_frame(spec: CardSpec, out_path: Path, workdir: Path, at: float | None = None) -> Path:
    """One still frame of the card (reveal complete by default) -- a cheap
    review preview; nothing else is rendered."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    inputs, graph = _build_cmd(spec, workdir)
    t = at if at is not None else min(spec.duration - 0.05, _ramp(spec) * 2 + 0.2)
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[v]",
           "-ss", f"{t:.3f}", "-frames:v", "1", "-q:v", "3", str(out_path)]
    result = run(cmd, timeout=120)
    if result.returncode != 0 or not out_path.exists():
        raise EndCardError(f"card frame failed:\n{result.stderr.strip()[-1500:]}")
    return out_path


def render_card_clip(spec: CardSpec, out_path: Path, workdir: Path) -> Path:
    """The full card clip (H.264 + silent AAC). Only used by the final render."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    inputs, graph = _build_cmd(spec, workdir)
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-f", "lavfi", "-t", f"{spec.duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo",
        "-filter_complex", graph, "-map", "[v]", "-map", f"{len(inputs) // 8}:a",
        "-r", f"{spec.fps:g}", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "160k", "-shortest", str(out_path),
    ]
    result = run(cmd, timeout=300)
    if result.returncode != 0 or not out_path.exists():
        raise EndCardError(f"card render failed:\n{result.stderr.strip()[-1500:]}")
    return out_path


def join_clips(first: Path, second: Path, out_path: Path, transition_s: float, fps: float) -> Path:
    """Appends `second` after `first`, dissolving over `transition_s`
    seconds (0 = hard cut). Both clips must share resolution."""
    a_fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    fps_s = f"{fps:g}"
    if transition_s > 0:
        offset = max(0.0, probe_duration(first) - transition_s)
        graph = (
            f"[0:v]fps={fps_s},setsar=1[v0];[1:v]fps={fps_s},setsar=1[v1];"
            f"[v0][v1]xfade=transition=fade:duration={transition_s:.3f}:offset={offset:.3f}[v];"
            f"[0:a]{a_fmt}[a0];[1:a]{a_fmt}[a1];[a0][a1]acrossfade=d={transition_s:.3f}[a]"
        )
    else:
        graph = (
            f"[0:v]fps={fps_s},setsar=1[v0];[1:v]fps={fps_s},setsar=1[v1];"
            f"[0:a]{a_fmt}[a0];[1:a]{a_fmt}[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )
    cmd = ["ffmpeg", "-y", "-i", str(first), "-i", str(second), "-filter_complex", graph,
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out_path)]
    result = run(cmd, timeout=3600)
    if result.returncode != 0:
        raise EndCardError(f"clip join failed:\n{result.stderr.strip()[-1500:]}")
    return out_path


def compose_with_cards(
    main_path: Path, out_path: Path, work_dir: Path, intro: CardSpec | None, end_card: CardSpec | None,
) -> Path:
    """Wraps the rendered content with the planned intro and/or end card."""
    current = main_path
    if intro is not None:
        clip = render_card_clip(intro, work_dir / "intro_card.mp4", work_dir)
        current = join_clips(clip, current, work_dir / "with_intro.mp4", intro.transition_s, intro.fps)
    if end_card is not None:
        clip = render_card_clip(end_card, work_dir / "end_card.mp4", work_dir)
        current = join_clips(current, clip, out_path, end_card.transition_s, end_card.fps)
    elif current != out_path:
        out_path.write_bytes(current.read_bytes())
    return out_path
