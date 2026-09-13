"""`videoedit doctor` (spec section 28): prints the full capability matrix so
the user can see exactly what will run locally vs. via cloud providers, with
zero ambiguity about what "auto" mode will actually do.
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from video_edit_agent.core.capability_router import full_capability_matrix


def print_doctor_report(console: Console) -> None:
    matrix = full_capability_matrix()
    table = Table(title="videoedit doctor — capability matrix")
    table.add_column("Capability")
    table.add_column("Status")
    table.add_column("Detail")

    for cap in matrix.values():
        status = "[green]OK[/green]" if cap.available else "[yellow]unavailable[/yellow]"
        table.add_row(cap.name, status, cap.detail)

    console.print(table)

    if not matrix["ffmpeg"].available or not matrix["ffprobe"].available:
        console.print("[red]ffmpeg/ffprobe are required for any rendering. Install ffmpeg and re-run.[/red]")

    if not matrix["faster-whisper"].available and not matrix["gemini_key"].available and not matrix["elevenlabs_key"].available:
        console.print(
            "[yellow]No transcription path is available yet. Run `pip install video-edit-agent[local]` "
            "for offline transcription, or set GEMINI_API_KEY / ELEVENLABS_API_KEY.[/yellow]"
        )
