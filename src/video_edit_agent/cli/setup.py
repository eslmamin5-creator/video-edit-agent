"""`videoedit setup` (v0.2.1 installation UX, spec section 6): the canonical
bootstrap entry point. Detects OS/Python, builds a private runtime under
`.runtime/`, installs the requested profile, checks ffmpeg/Node, and prints
a plain readiness report -- so a user (or a Claude Code Skill acting on
their behalf) never has to understand virtualenvs, pip extras, or provider
routing to get to a working install.

  videoedit setup                       # full bootstrap, profile=full-local
  videoedit setup --profile local       # bootstrap with a narrower profile
  videoedit setup --check               # read-only: report readiness, change nothing
  videoedit setup --repair              # rebuild a broken/missing private runtime
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console

from video_edit_agent.bootstrap.dependencies import DEFAULT_PROFILE, known_profiles
from video_edit_agent.bootstrap.setup import run_check, run_setup
from video_edit_agent.cli.doctor import print_doctor_report

console = Console()

app = typer.Typer(help="Bootstrap the private runtime and check environment readiness.")


@app.callback(invoke_without_command=True)
def main(
    profile: str = typer.Option(
        DEFAULT_PROFILE, "--profile",
        help=f"Install profile: one of {', '.join(known_profiles())}.",
    ),
    check: bool = typer.Option(False, "--check", help="Read-only: report readiness, make no changes."),
    repair: bool = typer.Option(False, "--repair", help="Rebuild a broken/missing private runtime."),
) -> None:
    project_root = Path.cwd()

    if check:
        console.print("[bold]videoedit setup --check[/bold] (read-only)\n")
        outcome = run_check(project_root)
    else:
        console.print(f"[bold]videoedit setup[/bold] — profile: {profile}\n")
        try:
            outcome = run_setup(project_root, profile=profile, repair=repair)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc

    for message in outcome.messages:
        console.print(message)

    console.print(
        f"\nffmpeg: {'[green]OK[/green]' if outcome.ffmpeg.healthy else '[red]missing[/red]'} "
        f"({outcome.ffmpeg.detail})"
    )
    console.print(
        f"ffprobe: {'[green]OK[/green]' if outcome.ffprobe.healthy else '[red]missing[/red]'} "
        f"({outcome.ffprobe.detail})"
    )
    console.print(f"node/npm: {outcome.node.detail}")

    console.print(
        "\nCloud providers (Gemini, ElevenLabs) are entirely optional and only used if you set "
        "GEMINI_API_KEY / ELEVENLABS_API_KEY as environment variables. Zero API keys is a fully "
        "supported setup."
    )

    console.print("\nFinal capability check:\n")
    runtime_python = getattr(outcome.runtime, "python_path", None)
    if runtime_python is not None:
        # Query the private runtime's own interpreter, not this process's --
        # package-presence checks (faster-whisper, mediapipe, ...) only see
        # what's installed in whichever interpreter actually runs them, and
        # right after building a fresh venv that is NOT this process.
        proc = subprocess.run(
            [str(runtime_python), "-m", "video_edit_agent.cli.main", "doctor"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            console.print(proc.stdout)
        else:
            console.print("[yellow]Could not query the private runtime for a doctor report; showing local view:[/yellow]")
            print_doctor_report(console)
    else:
        print_doctor_report(console)

    if outcome.state_path:
        console.print(f"\nSetup state saved: {outcome.state_path}")

    if outcome.ok:
        console.print("\n[green]Ready.[/green] Try: videoedit edit <video>")
    else:
        console.print("\n[yellow]Not fully ready yet[/yellow] -- see the messages above.")
        raise typer.Exit(code=1)
