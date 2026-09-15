"""Bootstrap orchestration for `videoedit setup` (spec sections 6, 7, 17).

Pure logic, no Typer/Rich -- `cli/setup.py` is a thin presentation wrapper
around `run_setup`/`run_check` so this is directly unit-testable and reusable
from a Claude Code Skill driver script.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from video_edit_agent.bootstrap import capabilities as cap_mod
from video_edit_agent.bootstrap import ffmpeg as ffmpeg_mod
from video_edit_agent.bootstrap import node as node_mod
from video_edit_agent.bootstrap.dependencies import DEFAULT_PROFILE, InstallResult, install_profile, known_profiles
from video_edit_agent.bootstrap.detect import PythonSelection, detect_os, select_python
from video_edit_agent.bootstrap.report import SetupState, now_iso, platform_summary, save_state
from video_edit_agent.bootstrap.runtime import RuntimeStatus, check_runtime, ensure_runtime


@dataclass
class SetupOutcome:
    ok: bool
    os_kind: str
    python: PythonSelection | None
    runtime: RuntimeStatus | None
    install: InstallResult | None
    ffmpeg: ffmpeg_mod.ExecutableStatus
    ffprobe: ffmpeg_mod.ExecutableStatus
    node: node_mod.NodeStatus
    capability_states: list[cap_mod.CapabilityState] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    state_path: Path | None = None


def capability_states_in_runtime(python_executable: str, timeout: float = 30.0) -> list[cap_mod.CapabilityState] | None:
    """Runs the capability check INSIDE the given interpreter (the private
    runtime venv, not whatever process is currently running `videoedit
    setup`) so package-presence checks (faster-whisper, mediapipe, ...)
    reflect what was actually just installed there. Returns None if the
    subprocess check itself fails, so callers can fall back to an in-process
    check rather than crashing setup over a reporting step."""
    try:
        proc = subprocess.run(
            [python_executable, "-m", "video_edit_agent.bootstrap.capabilities"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return [cap_mod.CapabilityState(d["name"], d["state"], d["detail"]) for d in data]


def capability_matrix_in_runtime(python_executable: str, timeout: float = 30.0) -> dict[str, "cap_router.Capability"] | None:
    """Like `capability_states_in_runtime`, but returns the raw `Capability`
    objects (available/detail/verified) that `doctor`/`providers` render, by
    querying `core.capability_router` INSIDE the given interpreter. Returns
    None if the subprocess check itself fails, so callers can fall back to
    an in-process check rather than crashing over a reporting step."""
    from video_edit_agent.core import capability_router as cap_router

    try:
        proc = subprocess.run(
            [python_executable, "-m", "video_edit_agent.core.capability_router"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return {
        d["name"]: cap_router.Capability(d["name"], d["available"], d["detail"], d.get("verified"))
        for d in data
    }


def capability_matrix_for_project(project_root: Path) -> dict[str, "cap_router.Capability"]:
    """Best-effort accurate capability matrix for `doctor`/`providers`:
    prefers querying the project's private runtime venv (if healthy) so the
    report reflects what `videoedit setup` actually installed there, rather
    than whatever happens to be importable in the interpreter currently
    running the `videoedit` entry point. That entry point is often installed
    against the global/system Python (`pip install -e .` in the docs'
    Option B), which is NOT the private runtime and can under- or
    over-report capabilities depending on whatever else happens to be
    installed globally (see `capability_states_in_runtime`)."""
    from video_edit_agent.core.capability_router import full_capability_matrix

    status = check_runtime(project_root)
    if status.healthy:
        matrix = capability_matrix_in_runtime(str(status.python_path))
        if matrix is not None:
            return matrix
    return full_capability_matrix()


def run_check(project_root: Path) -> SetupOutcome:
    """`videoedit setup --check`: read-only. Makes no changes, creates
    nothing, installs nothing."""
    os_kind = detect_os()
    python = select_python(os_kind)
    ffmpeg_status = ffmpeg_mod.check_ffmpeg()
    ffprobe_status = ffmpeg_mod.check_ffprobe()
    node_status = node_mod.check_node()
    states = cap_mod.capability_states()

    messages = []
    if python is None:
        messages.append("No compatible Python (>=3.10) found on PATH.")
    if not ffmpeg_status.healthy:
        messages.append(ffmpeg_mod.install_guidance(os_kind))

    return SetupOutcome(
        ok=python is not None and ffmpeg_status.healthy and ffprobe_status.healthy,
        os_kind=os_kind, python=python, runtime=None, install=None,
        ffmpeg=ffmpeg_status, ffprobe=ffprobe_status, node=node_status,
        capability_states=states, messages=messages,
    )


def run_setup(
    project_root: Path,
    profile: str = DEFAULT_PROFILE,
    repair: bool = False,
) -> SetupOutcome:
    """Full bootstrap: detect OS/Python, ensure the private runtime, install
    the requested profile into it, check ffmpeg/Node, persist non-secret
    setup state. Idempotent -- re-running with a healthy runtime and the
    same profile is fast (pip no-ops on already-satisfied requirements) and
    makes no destructive changes."""
    if profile not in known_profiles():
        raise ValueError(f"unknown profile {profile!r}; choose one of {known_profiles()}")

    os_kind = detect_os()
    messages: list[str] = []

    python = select_python(os_kind)
    if python is None:
        return SetupOutcome(
            ok=False, os_kind=os_kind, python=None, runtime=None, install=None,
            ffmpeg=ffmpeg_mod.check_ffmpeg(), ffprobe=ffmpeg_mod.check_ffprobe(),
            node=node_mod.check_node(),
            messages=["No compatible Python (>=3.10) found. Install Python 3.11 and re-run `videoedit setup`."],
        )
    messages.append(python.explanation)

    runtime_status = ensure_runtime(project_root, python.executable, repair=repair)
    if not runtime_status.healthy:
        messages.append(f"Failed to prepare the private runtime: {runtime_status.reason}")
        return SetupOutcome(
            ok=False, os_kind=os_kind, python=python, runtime=runtime_status, install=None,
            ffmpeg=ffmpeg_mod.check_ffmpeg(), ffprobe=ffmpeg_mod.check_ffprobe(),
            node=node_mod.check_node(), messages=messages,
        )

    install_result = install_profile(project_root, str(runtime_status.python_path), profile=profile)
    if install_result.ok:
        messages.append(f"Installed profile '{profile}' into the private runtime.")
    else:
        messages.append(f"Dependency install for profile '{profile}' failed: {install_result.detail}")

    ffmpeg_status = ffmpeg_mod.check_ffmpeg()
    ffprobe_status = ffmpeg_mod.check_ffprobe()
    if not ffmpeg_status.healthy:
        messages.append(ffmpeg_mod.install_guidance(os_kind))
    node_status = node_mod.check_node()
    messages.append(node_status.detail)

    states = capability_states_in_runtime(str(runtime_status.python_path))
    if states is None:
        # Fall back to an in-process check (e.g. the subprocess probe itself
        # failed) -- still better than no report, but note it may reflect
        # the invoking interpreter rather than the private runtime.
        states = cap_mod.capability_states()
        messages.append(
            "Note: capability check ran in-process (could not query the private runtime directly); "
            "re-run `videoedit doctor` using the private runtime's videoedit for a fully accurate report."
        )

    state = SetupState(
        os=platform_summary(),
        python_executable=str(runtime_status.python_path),
        python_version=".".join(map(str, python.version)),
        profile=profile,
        ffmpeg_version=ffmpeg_status.version,
        node_version=node_status.node_version,
        npm_version=node_status.npm_version,
        capability_summary={s.name: s.state for s in states},
        last_verified_at=now_iso(),
    )
    saved_path = save_state(project_root, state)

    ok = install_result.ok and ffmpeg_status.healthy and ffprobe_status.healthy
    return SetupOutcome(
        ok=ok, os_kind=os_kind, python=python, runtime=runtime_status, install=install_result,
        ffmpeg=ffmpeg_status, ffprobe=ffprobe_status, node=node_status,
        capability_states=states, messages=messages, state_path=saved_path,
    )


def render_text_report(outcome: SetupOutcome) -> str:
    """Plain-text rendering usable both by the CLI (wrapped in Rich) and by
    a Claude Code Skill driver that just wants a string to show the user."""
    lines = [f"OS: {outcome.os_kind}"]
    if outcome.python:
        lines.append(outcome.python.explanation)
    for msg in outcome.messages:
        lines.append(msg)
    lines.append(f"ffmpeg: {'OK' if outcome.ffmpeg.healthy else 'missing'} ({outcome.ffmpeg.detail})")
    lines.append(f"ffprobe: {'OK' if outcome.ffprobe.healthy else 'missing'} ({outcome.ffprobe.detail})")
    lines.append(f"node/npm: {outcome.node.detail}")
    for state in outcome.capability_states:
        lines.append(f"  {state.name}: {state.state} -- {state.detail}")
    lines.append("READY" if outcome.ok else "NOT READY")
    return "\n".join(lines)
