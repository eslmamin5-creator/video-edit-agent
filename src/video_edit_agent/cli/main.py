"""videoedit CLI entry point (spec sections 10, 27-29).

Commands:
  videoedit edit <video>            - run the full editing pipeline
  videoedit doctor                  - print the capability matrix
  videoedit setup                   - interactive first-run wizard
  videoedit config show|set         - inspect/edit layered config
  videoedit providers                - list transcription/motion providers and availability
  videoedit brand init|validate     - manage Brand Profiles
  videoedit project inspect         - print project memory for a project directory
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from video_edit_agent import __version__
from video_edit_agent.agents.assembler.pipeline import AssemblerError, run_assembler
from video_edit_agent.agents.creator.parser import ScriptParseError
from video_edit_agent.agents.creator.pipeline import run_creator
from video_edit_agent.brand.loader import BrandNotFoundError, init_brand, load_brand
from video_edit_agent.brand.validator import validate_brand
from video_edit_agent.cli import config as config_cli
from video_edit_agent.cli import setup as setup_cli
from video_edit_agent.bootstrap.setup import capability_matrix_for_project
from video_edit_agent.cli.doctor import print_doctor_report
from video_edit_agent.core.capability_router import full_capability_matrix
from video_edit_agent.core.config import AppConfig
from video_edit_agent.core.pipeline import run_pipeline
from video_edit_agent.core.project import ProjectPaths
from video_edit_agent.localization.interface import t

app = typer.Typer(help="videoedit — Arabic-first, model-agnostic AI video editing agent.")
app.add_typer(config_cli.app, name="config")
app.add_typer(setup_cli.app, name="setup")

brand_app = typer.Typer(help="Manage Brand Profiles.")
app.add_typer(brand_app, name="brand")

project_app = typer.Typer(help="Inspect project state.")
app.add_typer(project_app, name="project")

console = Console()


def _lang(cfg: AppConfig | None = None) -> str:
    cfg = cfg or AppConfig.load()
    lang = cfg.interface.language
    if lang != "auto":
        return lang
    import locale as _locale

    loc = _locale.getdefaultlocale()[0] or ""
    return "ar" if loc.lower().startswith("ar") else "en"


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show the videoedit version and exit."),
):
    if version:
        console.print(f"videoedit {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def edit(
    video: Path = typer.Argument(..., exists=True, help="Path to the source video file."),
    brand: Optional[str] = typer.Option(None, "--brand", help="Brand profile name under brands/."),
    preset: str = typer.Option("reel", "--preset", help="Export preset: reel|tiktok|shorts|square|landscape."),
    caption_style: str = typer.Option("word-highlight", "--caption-style"),
    offline: bool = typer.Option(False, "--offline", help="Block all cloud calls; local-only pipeline."),
    no_broll: bool = typer.Option(False, "--no-broll"),
    no_motion: bool = typer.Option(False, "--no-motion"),
):
    """Run the full editing pipeline on a single source video."""
    lang = _lang()
    console.print(t("analyzing_video", lang))

    matrix = full_capability_matrix()
    if not matrix["gemini_key"].available and not matrix["elevenlabs_key"].available:
        console.print(t("no_api_keys", lang))
    if offline:
        console.print(t("offline_mode", lang))

    def on_progress(stage: str) -> None:
        stage_messages = {
            "transcribe": "analyzing_video",
            "edl": "preparing_edl",
            "captions": "generating_captions",
            "motion": "generating_motion",
            "qa": "running_qa",
        }
        key = stage_messages.get(stage)
        if key:
            console.print(t(key, lang))

    result = run_pipeline(
        video,
        brand_name=brand,
        offline=offline,
        preset_name=preset,
        caption_style_name=caption_style,
        enable_broll=not no_broll,
        enable_motion=not no_motion,
        on_progress=on_progress,
    )

    if result.final_output is None:
        console.print(t("error_generic", lang, message="; ".join(result.warnings) or "pipeline did not complete"))
        raise typer.Exit(code=1)

    console.print(t("render_success", lang))
    console.print(f"Output: {result.final_output}")
    console.print(f"Project files: {result.project_dir}")
    if result.qa_report and result.qa_report.issues:
        console.print(f"QA issues: {len(result.qa_report.issues)} (see project.md for detail)")


@app.command()
def create(
    script: Path = typer.Argument(..., exists=True, help="Path to the source script (.txt/.md/.docx/.pdf)."),
    brand: Optional[str] = typer.Option(None, "--brand", help="Brand profile name under brands/."),
    style: str = typer.Option("mixed", "--style", help="Visual style: motion|cinematic|infographic|mixed."),
    preset: str = typer.Option("reel", "--preset", help="Export preset: reel|tiktok|shorts|square|landscape."),
    offline: bool = typer.Option(False, "--offline", help="Block all cloud calls; local-only Creator pipeline."),
):
    """Generate a video from a script: analysis -> scenes -> storyboard -> asset plan -> MasterTimeline -> render -> QA."""
    lang = _lang()
    if offline:
        console.print(t("offline_mode", lang))

    try:
        result = run_creator(script, brand_name=brand, style=style, offline=offline, preset_name=preset)
    except ScriptParseError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]Invalid --style: {exc}[/red]")
        raise typer.Exit(code=1)

    if result.final_output is None:
        console.print(t("error_generic", lang, message="; ".join(result.warnings) or "Creator did not complete"))
        raise typer.Exit(code=1)

    console.print(t("render_success", lang))
    console.print(f"Output: {result.final_output}")
    console.print(f"Project files: {result.project_dir}")
    console.print(f"Scenes: {len(result.scenes)}")
    if result.qa_report and result.qa_report.issues:
        console.print(f"QA issues: {len(result.qa_report.issues)} (see project.md for detail)")


@app.command()
def assemble(
    scenes: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory of existing scene video files."),
    rough: bool = typer.Option(False, "--rough", help="Produce a fast Rough Cut (edit/rough_cut.mp4)."),
    finish: bool = typer.Option(False, "--finish", help="Produce the finished film, reusing prior plan decisions."),
    preserve_order: bool = typer.Option(
        False, "--preserve-order", help="Guarantee the discovered scene order is never reordered."
    ),
    order: str = typer.Option(
        "preserve", "--order", help="Ordering policy: filename|script (ignored if --preserve-order is set)."
    ),
    script: Optional[Path] = typer.Option(None, "--script", help="Script file for --order script alignment."),
    brand: Optional[str] = typer.Option(None, "--brand", help="Brand profile name under brands/."),
    preset: str = typer.Option("reel", "--preset", help="Export preset: reel|tiktok|shorts|square|landscape."),
    offline: bool = typer.Option(False, "--offline", help="Block all cloud calls; local-only Assembler pipeline."),
):
    """Assemble multiple existing video scenes into a rough cut / finished film."""
    lang = _lang()
    if offline:
        console.print(t("offline_mode", lang))

    try:
        result = run_assembler(
            scenes,
            rough=rough,
            finish=finish,
            preserve_order=preserve_order,
            order=order,
            script_path=script,
            brand_name=brand,
            offline=offline,
            preset_name=preset,
        )
    except (AssemblerError, ScriptParseError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    if result.rough_cut_path is None and result.final_output is None:
        console.print(t("error_generic", lang, message="; ".join(result.warnings) or "Assembler did not complete"))
        raise typer.Exit(code=1)

    console.print(t("render_success", lang))
    if result.rough_cut_path:
        console.print(f"Rough Cut: {result.rough_cut_path}")
    if result.final_output:
        console.print(f"Final: {result.final_output}")
    console.print(f"Project files: {result.project_dir}")
    console.print(f"Scenes: {len(result.scene_inventory)} (order policy: {result.order_policy.value})")
    if result.qa_issues:
        console.print(f"QA issues: {len(result.qa_issues)} (see project.md for detail)")


@app.command()
def doctor():
    """Print the full capability matrix (ffmpeg, node, providers, engines)."""
    print_doctor_report(console)


@app.command()
def providers():
    """List transcription and motion providers with their availability."""
    matrix = capability_matrix_for_project(Path.cwd())
    console.print("[bold]Transcription providers[/bold]")
    for name in ("gemini_key", "elevenlabs_key", "faster-whisper", "openai-whisper", "whisper.cpp"):
        cap = matrix[name]
        console.print(f"  {cap.name}: {'available' if cap.available else 'unavailable'} ({cap.detail})")

    console.print("\n[bold]Motion engines[/bold]")
    for name in ("hyperframes", "remotion", "manim"):
        cap = matrix[name]
        console.print(f"  {cap.name}: {'available' if cap.available else 'unavailable'} ({cap.detail})")
    console.print("  simple: available (always on, guaranteed fallback)")


@brand_app.command("init")
def brand_init(name: str = typer.Argument(..., help="New brand name.")):
    """Create a new Brand Profile folder under brands/<name>/."""
    try:
        path = init_brand(name)
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"Created brand profile at {path}")


@brand_app.command("validate")
def brand_validate(name: str = typer.Argument(..., help="Brand name to validate.")):
    """Validate a Brand Profile's brand.yaml and referenced assets."""
    try:
        brand = load_brand(name)
    except BrandNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    result = validate_brand(brand)
    for warning in result.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    if result.ok:
        console.print(f"[green]Brand '{name}' is valid.[/green]")
    else:
        console.print(f"[red]Brand '{name}' has errors:[/red]")
        for error in result.errors:
            console.print(f"  - {error}")
        raise typer.Exit(code=1)


@project_app.command("inspect")
def project_inspect(project_dir: Path = typer.Argument(..., exists=True, help="Project source directory.")):
    """Print the human-readable project memory (project.md) for a project."""
    paths = ProjectPaths.for_source(project_dir)
    if not paths.project_md.exists():
        console.print(f"[yellow]No project memory found at {paths.project_md}[/yellow]")
        raise typer.Exit(code=1)
    console.print(paths.project_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
