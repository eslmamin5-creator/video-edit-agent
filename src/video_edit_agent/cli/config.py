"""`videoedit config` command group (spec section 25): read/write layered
config without ever touching secrets (those stay in environment variables
only, per spec section 26).
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from video_edit_agent.core.config import AppConfig

app = typer.Typer(help="Inspect or edit videoedit configuration.")
console = Console()


@app.command("show")
def show(project: Path = typer.Option(None, "--project", help="Project directory to include project-level config.")):
    """Print the fully merged (defaults -> user -> project) configuration."""
    cfg = AppConfig.load(project_dir=project)
    console.print_json(cfg.model_dump_json())


@app.command("set")
def set_value(
    key: str = typer.Argument(..., help="Dotted config path, e.g. transcription.provider"),
    value: str = typer.Argument(...),
    project: Path = typer.Option(None, "--project", help="Save to project config instead of user config."),
):
    """Set a single config value and persist it (user-level by default)."""
    cfg = AppConfig.load(project_dir=project)
    data = cfg.model_dump(mode="json")
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = _coerce(value)

    updated = AppConfig.model_validate(data)
    if project:
        path = updated.save_project(project)
    else:
        path = updated.save_user()
    console.print(f"Saved {key} = {value} -> {path}")


def _coerce(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
