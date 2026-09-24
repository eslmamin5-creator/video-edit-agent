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

import typer
from rich.console import Console

from video_edit_agent import __version__
from video_edit_agent.agents.assembler.pipeline import AssemblerError, run_assembler
from video_edit_agent.agents.creator.parser import ScriptParseError
from video_edit_agent.agents.creator.pipeline import run_creator
from video_edit_agent.bootstrap.setup import capability_matrix_for_project
from video_edit_agent.brand.loader import BrandNotFoundError, init_brand, load_brand
from video_edit_agent.brand.validator import validate_brand
from video_edit_agent.cli import config as config_cli
from video_edit_agent.cli import setup as setup_cli
from video_edit_agent.cli.doctor import print_doctor_report
from video_edit_agent.core.config import AppConfig
from video_edit_agent.core.pipeline import run_pipeline
from video_edit_agent.core.project import ProjectPaths
from video_edit_agent.localization.interface import t
from video_edit_agent.review import state as review_state

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


def _project_of(video_or_dir: Path) -> Path:
    """`videoedit approve talk.mp4` and `videoedit approve edit/` both work."""
    p = Path(video_or_dir)
    return p if p.is_dir() else ProjectPaths.for_source(p).edit_dir


def _show_board(edit_dir: Path) -> None:
    from video_edit_agent.review import treatment_board as board

    cards = board.items(edit_dir)
    if not cards:
        console.print("No major treatments to review in this video.")
        return
    for c in cards:
        console.print(f"[bold]{c.name}[/bold]  -  {c.status}")
        console.print(f"  {c.reason}")
        if c.preview is not None:
            console.print(f"  Preview: {c.preview}")
        console.print(f"  Choices: {' / '.join(c.actions)}   (name: {c.treatment_id})")


def _plan_again(video: Path, edit_dir: Path) -> None:
    """Renders the previews of anything that changed, reusing the saved setup answers and the cached transcript."""
    from video_edit_agent.review import setup_choices

    choices = setup_choices.resolve_choices(edit_dir / "review")
    run_pipeline(
        video, brand_name=choices.brand, offline=choices.offline, preset_name=choices.preset,
        caption_style_name=choices.caption_style, enable_broll=False, enable_motion=False, review=True,
        profile=choices.profile,
    )


@app.command()
def edit(
    video: Path = typer.Argument(..., exists=True, help="Path to the video."),
    profile: str | None = typer.Option(None, "--profile", help="How much editing: minimal | balanced | dynamic (default balanced)."),
    brand: str | None = typer.Option(None, "--brand", help="Your brand name. Without it a clean neutral look is used."),
    preset: str | None = typer.Option(None, "--preset", help="Format: reel | tiktok | shorts | square | landscape."),
    caption_style: str | None = typer.Option(None, "--caption-style", help="Caption look."),
    offline: bool = typer.Option(False, "--offline", help="Work fully on this computer; no cloud."),
    no_broll: bool = typer.Option(False, "--no-broll", hidden=True),
    no_motion: bool = typer.Option(False, "--no-motion", hidden=True),
    yes: bool = typer.Option(False, "--yes", "--no-review", help="Skip the review step and render straight away."),
    logo_mode: str | None = typer.Option(None, "--logo-mode", hidden=True),
):
    """Edit a video: analysis, a proposed edit, a short review, then the final MP4.

    The first run analyses the video and shows you the main treatments with short previews (nothing is rendered
    yet). Ask for changes with `videoedit revise`, approve with `videoedit approve`, then run this command again
    to get the final MP4."""
    from video_edit_agent.review import setup_choices

    lang = _lang()
    paths = ProjectPaths.for_source(video)
    try:
        choices = setup_choices.resolve_choices(
            paths.edit_dir / "review", profile=profile, brand=brand, preset=preset, caption_style=caption_style,
            offline=True if offline else None,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    if setup_choices.load_choices(paths.edit_dir / "review") is not None:
        console.print("Existing project found. Your earlier progress and approved edits are kept; continuing from where it stopped.")
    setup_choices.save_choices(paths.edit_dir / "review", choices)
    console.print(t("analyzing_video", lang))
    if choices.offline:
        console.print(t("offline_mode", lang))

    def on_progress(stage: str) -> None:
        key = {"transcribe": "analyzing_video", "edl": "preparing_edl", "captions": "generating_captions",
               "motion": "generating_motion", "qa": "running_qa"}.get(stage)
        if key:
            console.print(t(key, lang))

    result = run_pipeline(
        video,
        brand_name=choices.brand,
        offline=choices.offline,
        preset_name=choices.preset,
        caption_style_name=choices.caption_style,
        enable_broll=not no_broll,
        enable_motion=not no_motion,
        on_progress=on_progress,
        review=False if yes else None,
        logo_mode=logo_mode,
        profile=choices.profile,
    )

    if not result.ready_for_final_render and result.review_dir is not None:
        console.print(f"[bold]Ready for your review[/bold] ({choices.profile} style). Nothing has been rendered yet.")
        _show_board(paths.edit_dir)
        console.print(
            "Ask for a change:  [bold]videoedit revise <video> \"<what you want>\"[/bold]   "
            "Happy with it:  [bold]videoedit approve <video>[/bold]   then run [bold]videoedit edit <video>[/bold] again for the MP4."
        )
        if result.warnings:
            console.print(f"Notes: {'; '.join(result.warnings)}")
        return

    if result.final_output is None:
        console.print(t("error_generic", lang, message="; ".join(result.warnings) or "pipeline did not complete"))
        raise typer.Exit(code=1)

    console.print(t("render_success", lang))
    console.print(f"Your video: {result.final_output}")
    if result.qa_report and result.qa_report.issues:
        console.print(f"QA notes: {len(result.qa_report.issues)} (see project.md for detail)")


@app.command()
def review(video: Path = typer.Argument(..., exists=True, help="The video (or its edit folder).")):
    """Show the main treatments, each with its preview and what you can do with it."""
    _show_board(_project_of(video))


@app.command()
def revise(
    video: Path = typer.Argument(..., exists=True, help="The video (or its edit folder)."),
    request: str = typer.Argument(..., help="What to change, in your own words, e.g. خفف الزوم or كبر الهيدر."),
):
    """Ask for a change. The old approval no longer counts; a new preview is rendered for you to approve."""
    from video_edit_agent.review import treatment_board as board

    edit_dir = _project_of(video)
    try:
        board.revise(edit_dir, request)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if video.is_file():
        console.print("Updating the preview...")
        _plan_again(video, edit_dir)
    console.print("Changed. Please look at the new preview and approve it when you are happy.")
    _show_board(edit_dir)


@app.command()
def approve(video: Path = typer.Argument(..., exists=True, help="The video (or its edit folder).")):
    """Approve what the previews show. It is then locked: the final video uses exactly this."""
    from video_edit_agent.review import treatment_board as board

    edit_dir = _project_of(video)
    if video.is_file() and board.waiting_for_preview(edit_dir):
        console.print("Updating the preview first...")
        _plan_again(video, edit_dir)
    done = board.approve(edit_dir)
    try:
        review_state.approve(edit_dir / "review", note="approved via `videoedit approve`")
    except review_state.UnresolvedReviewItems as exc:
        console.print(f"[red]Not approved yet: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Approved and locked: {len(done)} change(s). Run [bold]videoedit edit {video}[/bold] to make the final MP4.")


@app.command()
def undo(video: Path = typer.Argument(..., exists=True, help="The video (or its edit folder).")):
    """Go back to the version before your last change."""
    from video_edit_agent.review import treatment_board as board

    try:
        board.undo(_project_of(video))
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=1) from exc
    console.print("Went back to the earlier version. Approve it again to lock it.")


@app.command()
def remove(
    video: Path = typer.Argument(..., exists=True, help="The video (or its edit folder)."),
    treatment: str = typer.Argument(..., help="Which treatment to remove (see `videoedit review`)."),
):
    """Remove a treatment from the edit."""
    from video_edit_agent.review import treatment_board as board

    board.remove(_project_of(video), treatment)
    console.print("Removed.")


@app.command(name="review-approve")
def review_approve(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory to approve."),
):
    """Flip the READY_FOR_FINAL_RENDER gate for a reviewed project (Review-
    First Editing Workflow spec section 8). Re-run `videoedit edit ... --yes`
    afterwards to actually render."""
    review_dir = project_dir / "review" if project_dir.name != "review" else project_dir
    try:
        state = review_state.approve(review_dir, note="approved via `videoedit review-approve`")
    except review_state.UnresolvedReviewItems as exc:
        console.print(f"[red]Not approved: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Approved. ready_for_final_render={state.ready_for_final_render}")


@app.command(name="review-flag")
def review_flag(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory."),
    segment_id: str = typer.Argument(..., help="Transcript segment id to keep open for the user."),
    reason: str = typer.Option("", "--reason", help="Why the segment is unresolved."),
    asr_text: str = typer.Option("", "--asr-text", help="The raw ASR text, for the reviewer's reference."),
    segment: int | None = typer.Option(None, "--segment", help="1-based segment number."),
):
    """Keep a transcript segment visibly unresolved; approval is refused
    until it is corrected (`review-correct`) or resolved (`review-resolve`)."""
    from video_edit_agent.review.schemas import UnresolvedTranscriptItem

    review_dir = project_dir / "review" if project_dir.name != "review" else project_dir
    state = review_state.flag_unresolved(
        review_dir,
        UnresolvedTranscriptItem(segment_id=segment_id, segment=segment, asr_text=asr_text, reason=reason),
    )
    console.print(f"{len(state.unresolved_transcript)} unresolved transcript segment(s).")


@app.command(name="review-resolve")
def review_resolve(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory."),
    segment_id: str = typer.Argument(..., help="Transcript segment id the user has confirmed as-is."),
):
    """Close an unresolved transcript item without changing its text."""
    review_dir = project_dir / "review" if project_dir.name != "review" else project_dir
    state = review_state.resolve_unresolved(review_dir, segment_id)
    console.print(f"{len(state.unresolved_transcript)} unresolved transcript segment(s) remain.")


@app.command(name="review-correct")
def review_correct(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory."),
    segment_id: str = typer.Argument(..., help="Transcript segment id (see transcript_review.json)."),
    text: str = typer.Argument(..., help="Corrected text for the segment (or for one word with --word)."),
    word: int | None = typer.Option(None, "--word", help="0-based word index: replace only that word."),
):
    """Persist a transcript correction in project state. Timing is preserved;
    re-run `videoedit edit` to regenerate the review artifacts."""
    from video_edit_agent.review.corrections import add_correction
    from video_edit_agent.review.schemas import TranscriptCorrection

    review_dir = project_dir / "review" if project_dir.name != "review" else project_dir
    saved = add_correction(review_dir, TranscriptCorrection(segment_id=segment_id, corrected_text=text, word_index=word))
    review_state.resolve_unresolved(review_dir, segment_id)  # a confirmed correction closes its flag
    console.print(f"Saved. {len(saved)} correction(s) stored in {review_dir}")


@app.command(name="review-chat")
def review_chat(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory."),
    message: str = typer.Argument("", help="The user's message, exactly as typed (Arabic or English). Empty shows the review."),
    review_lang: str = typer.Option("ar", "--lang", help="Language of the chat text: ar|en."),
    open_review: bool = typer.Option(False, "--open", help="Start the review: flag every low-confidence segment as unresolved."),
    source: Path | None = typer.Option(None, "--source", help="Audio/video to cut clips from (default: the project's own audio)."),
    page_size: int = typer.Option(8, "--page-size", help="Segments per page."),
):
    """Chat-first transcript review. Pass the user's message as-is; the reply
    (numbered sentences, statuses, confirmations) is meant to be shown in the
    conversation. Understands approve / whole-sentence / word corrections,
    audio on demand, and paging. Raw ASR is never modified."""
    from video_edit_agent.review.chat_session import TranscriptReviewChat, open_transcript_review
    from video_edit_agent.transcription.router import load_transcript

    edit_dir = project_dir.parent if project_dir.name == "review" else project_dir
    if open_review:
        flagged = open_transcript_review(edit_dir / "review", load_transcript(edit_dir / "transcript_unified.json"))
        console.print(f"{len(flagged)} segment(s) opened for review.")
    reply = TranscriptReviewChat(edit_dir, lang=review_lang, page_size=page_size, source_media=source).reply(message)
    print(reply.text)
    for clip in reply.audio_paths:
        print(f"AUDIO: {clip}")


@app.command(name="review-plan")
def review_plan(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory."),
    message: str = typer.Argument("", help="The user's message, exactly as typed (Arabic or English). Empty shows what is pending."),
    review_lang: str = typer.Option("ar", "--lang", help="Language of the chat text: ar|en."),
    open_plan: bool = typer.Option(False, "--open", help="Build the edit plan from the editorial decisions (kept if it already exists)."),
    rebuild: bool = typer.Option(False, "--rebuild", help="With --open: discard the saved plan and rebuild it."),
    settled: list[str] = typer.Option([], "--settled", help="TREATMENT=reason for treatments already approved elsewhere (repeatable)."),
):
    """Chat-first edit-plan / B-roll review. Pass the user's message as-is; the
    reply is meant to be shown in the conversation. Nothing is generated or
    rendered, and approving here never sets ready_for_final_render."""
    from video_edit_agent.review.edit_plan_chat import EditPlanChat, open_edit_plan

    edit_dir = project_dir.parent if project_dir.name == "review" else project_dir
    if open_plan:
        known = {}
        for item in settled:
            name, _, why = item.partition("=")
            known[name.strip()] = why.strip() or "approved earlier"
        plan = open_edit_plan(edit_dir, settled=known, rebuild=rebuild)
        console.print(f"{len(plan.reviewable())} decision(s) in the edit plan, {len(plan.pending())} pending.")
    print(EditPlanChat(edit_dir, lang=review_lang).reply(message).text)


@app.command(name="review-approve-visual")
def review_approve_visual(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory."),
    treatment: str = typer.Argument("hook_title", help="Text treatment whose look/animation is approved."),
):
    """Approve how a text treatment (e.g. the hook title) looks and moves,
    independently of its wording. The planned props are stored so a later copy
    change reuses them instead of redesigning the treatment."""
    import json

    review_dir = project_dir / "review" if project_dir.name != "review" else project_dir
    plan_path = review_dir.parent / "motion_plan.json"
    props = None
    if plan_path.exists():
        for item in json.loads(plan_path.read_text(encoding="utf-8")):
            spec = item.get("spec", item)
            if spec.get("kind") == treatment and spec.get("extra"):
                props = spec["extra"]
                break
    try:
        review_state.approve_visual(review_dir, treatment, props)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Visual treatment '{treatment}' approved (copy status unchanged).")


@app.command(name="review-copy")
def review_copy(
    project_dir: Path = typer.Argument(..., exists=True, help="The edit/ project directory."),
    text: str = typer.Argument(..., help="The approved on-screen copy, used exactly as given."),
    treatment: str = typer.Option("hook_title", "--treatment", help="Text treatment the copy is for."),
    rewrite: bool = typer.Option(False, "--rewrite", help="Mark the copy as an approved rewrite (default: user-supplied)."),
):
    """Approve on-screen copy for a text treatment. Replaces the REVIEW
    placeholder without resetting the treatment's visual approval; the
    transcript is never modified."""
    from video_edit_agent.review.schemas import CopySource

    review_dir = project_dir / "review" if project_dir.name != "review" else project_dir
    try:
        review_state.submit_copy(
            review_dir, treatment, text, CopySource.APPROVED_REWRITE if rewrite else CopySource.USER_SUPPLIED,
        )
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Copy for '{treatment}' approved. Re-run the review to regenerate its preview.")


@app.command()
def create(
    script: Path = typer.Argument(..., exists=True, help="Path to the source script (.txt/.md/.docx/.pdf)."),
    brand: str | None = typer.Option(None, "--brand", help="Brand profile name under brands/."),
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
    script: Path | None = typer.Option(None, "--script", help="Script file for --order script alignment."),
    brand: str | None = typer.Option(None, "--brand", help="Brand profile name under brands/."),
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
