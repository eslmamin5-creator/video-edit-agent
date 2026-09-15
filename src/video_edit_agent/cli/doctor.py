"""`videoedit doctor` (spec section 28): prints the full capability matrix so
the user can see exactly what will run locally vs. via cloud providers, with
zero ambiguity about what "auto" mode will actually do.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from video_edit_agent.bootstrap.report import load_state
from video_edit_agent.core.capability_router import full_capability_matrix


def print_doctor_report(console: Console) -> None:
    matrix = full_capability_matrix()
    table = Table(title="videoedit doctor — capability matrix")
    table.add_column("Capability")
    table.add_column("Installed/Configured")
    table.add_column("Verified")
    table.add_column("Detail")

    for cap in matrix.values():
        status = "[green]yes[/green]" if cap.available else "[yellow]no[/yellow]"
        if cap.verified is None:
            verified = "n/a"
        elif cap.verified:
            verified = "[green]yes[/green]"
        else:
            verified = "[yellow]not yet[/yellow]"
        table.add_row(cap.name, status, verified, cap.detail)

    console.print(table)

    if not matrix["ffmpeg"].available or not matrix["ffprobe"].available:
        console.print("[red]ffmpeg/ffprobe are required for any rendering. Install ffmpeg and re-run.[/red]")

    if not matrix["faster-whisper"].available and not matrix["gemini_key"].available and not matrix["elevenlabs_key"].available:
        console.print(
            "[yellow]No transcription path is available yet. Run `videoedit setup --profile local` "
            "for offline transcription, or set GEMINI_API_KEY / ELEVENLABS_API_KEY.[/yellow]"
        )

    state = load_state(Path.cwd())
    if state is not None:
        console.print(
            f"\nLast `videoedit setup`: profile={state.profile}, python={state.python_version}, "
            f"verified={state.last_verified_at}"
        )
    else:
        console.print("\n[yellow]No bootstrap setup state found.[/yellow] Run `videoedit setup` first.")
