"""`videoedit setup` wizard (spec section 27): a short, non-blocking
interactive flow that explains what's available, offers to install the local
transcription extra, and never demands an API key to proceed.
"""
from __future__ import annotations

import subprocess
import sys

import typer
from rich.console import Console
from rich.prompt import Confirm

from video_edit_agent.cli.doctor import print_doctor_report
from video_edit_agent.core.capability_router import full_capability_matrix

console = Console()


def run_setup_wizard() -> None:
    console.print("[bold]videoedit setup[/bold] — this will check your environment and offer optional installs.\n")

    matrix = full_capability_matrix()

    if not matrix["ffmpeg"].available or not matrix["ffprobe"].available:
        console.print(
            "[red]ffmpeg/ffprobe were not found on PATH.[/red] videoedit cannot render without them. "
            "Install ffmpeg (https://ffmpeg.org/download.html) and re-run `videoedit setup`."
        )

    if not matrix["faster-whisper"].available:
        console.print(
            "\nLocal transcription (Faster-Whisper) is not installed. This lets videoedit transcribe "
            "audio fully offline with NO API key required."
        )
        if Confirm.ask("Install the local transcription extra now?", default=True):
            _pip_install("video-edit-agent[local]")

    console.print(
        "\nCloud providers (Gemini, ElevenLabs) are entirely optional and only used if you set "
        "GEMINI_API_KEY / ELEVENLABS_API_KEY as environment variables. videoedit never asks you to "
        "paste a key into a prompt or a file."
    )

    console.print("\nFinal capability check:\n")
    print_doctor_report(console)
    console.print("\n[green]Setup complete.[/green] Run `videoedit edit <video>` to start editing.")


def _pip_install(package_spec: str) -> None:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package_spec], check=True)
    except subprocess.CalledProcessError as exc:
        console.print(f"[yellow]Install failed: {exc}. You can install it manually later.[/yellow]")


app = typer.Typer(help="Run the interactive setup wizard.")


@app.callback(invoke_without_command=True)
def main() -> None:
    run_setup_wizard()
