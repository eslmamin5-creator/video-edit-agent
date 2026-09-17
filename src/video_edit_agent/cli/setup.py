"""`videoedit setup` (v0.2.1 installation UX, spec section 6): the canonical
bootstrap entry point. Detects OS/Python, builds a private runtime under
`.runtime/`, installs the requested profile, checks ffmpeg/Node, and prints
a plain readiness report -- so a user (or a Claude Code Skill acting on
their behalf) never has to understand virtualenvs, pip extras, or provider
routing to get to a working install.

  videoedit setup                       # full bootstrap, profile=full-local
  videoedit setup --profile local       # bootstrap with a narrower profile
  videoedit setup --check               # read-only: report readiness, change nothing
  videoedit setup --repair              # force-rebuild the private runtime, even if it looks healthy
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from video_edit_agent.bootstrap.dependencies import DEFAULT_PROFILE, install_extras, known_profiles
from video_edit_agent.bootstrap.setup import run_check, run_setup
from video_edit_agent.cli.doctor import print_doctor_report
from video_edit_agent.core.env_file import find_project_env_file, write_env_values

console = Console()

_PROVIDER_PROMPTS = (
    ("GEMINI_API_KEY", "Gemini API key", "gemini"),
    ("ELEVENLABS_API_KEY", "ElevenLabs API key", "elevenlabs"),
)


def _maybe_configure_cloud_providers(project_root: Path, runtime_python: str | None) -> None:
    """Optional interactive prompt (spec section 26 addendum): lets a normal
    user save GEMINI_API_KEY / ELEVENLABS_API_KEY to a local `.env` instead
    of learning OS environment variables. Skipped entirely outside a real
    terminal so it never blocks CI or a piped invocation. Entered values are
    never echoed back or logged.

    A key alone doesn't make a cloud provider usable: profiles like
    "full-local" deliberately don't install the "gemini"/"elevenlabs" SDKs
    (spec: offline-only), so entering a key here would otherwise leave the
    provider silently unusable despite `doctor` later reporting a key is
    set. When `runtime_python` (the private runtime's own interpreter) is
    available, this installs the extra for each key just entered, so the
    provider is truly ready immediately -- not a separate manual pip step.
    """
    if not sys.stdin.isatty():
        return

    if not typer.confirm("\nConfigure optional cloud providers?", default=False):
        return

    values: dict[str, str] = {}
    extras_needed: list[str] = []
    for env_name, label, extra in _PROVIDER_PROMPTS:
        entered = typer.prompt(f"{label} (blank to skip)", default="", show_default=False, hide_input=True)
        if entered.strip():
            values[env_name] = entered.strip()
            extras_needed.append(extra)

    if not values:
        console.print("No keys entered; nothing saved.")
        return

    env_path = find_project_env_file(project_root) or (project_root / ".env")
    write_env_values(env_path, values)
    console.print(f"Saved {len(values)} key(s) to {env_path} (never printed).")

    if runtime_python is None:
        console.print(
            "[yellow]Could not locate the private runtime to install the matching SDK(s) "
            "automatically; re-run `videoedit setup` to finish provider setup.[/yellow]"
        )
        return

    console.print(f"Installing SDK(s) for: {', '.join(extras_needed)} ...")
    result = install_extras(project_root, runtime_python, tuple(extras_needed))
    if result.ok:
        console.print("[green]SDK(s) installed.[/green]")
    else:
        console.print(f"[yellow]SDK install failed: {result.detail}[/yellow]")

app = typer.Typer(help="Bootstrap the private runtime and check environment readiness.")


@app.callback(invoke_without_command=True)
def main(
    profile: str = typer.Option(
        DEFAULT_PROFILE, "--profile",
        help=f"Install profile: one of {', '.join(known_profiles())}.",
    ),
    check: bool = typer.Option(False, "--check", help="Read-only: report readiness, make no changes."),
    repair: bool = typer.Option(
        False, "--repair",
        help="Force-rebuild the private runtime from scratch, even if it currently appears healthy.",
    ),
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
        "\nCloud providers (Gemini, ElevenLabs) are entirely optional. Zero API keys is a fully "
        "supported setup. Set GEMINI_API_KEY / ELEVENLABS_API_KEY as OS environment variables, or "
        "save them to a local .env (this setup can do that for you below)."
    )

    runtime_python = getattr(outcome.runtime, "python_path", None)

    if not check:
        _maybe_configure_cloud_providers(project_root, runtime_python)

    console.print("\nFinal capability check:\n")
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
