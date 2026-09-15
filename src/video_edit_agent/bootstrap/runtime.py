"""Private per-project runtime (spec section 2): a venv the project owns,
never the global/system Python. Canonical location: `<project_root>/.runtime`.
"""
from __future__ import annotations

import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path

RUNTIME_DIRNAME = ".runtime"
VENV_DIRNAME = "venv"


def runtime_dir(project_root: Path) -> Path:
    return project_root / RUNTIME_DIRNAME


def venv_dir(project_root: Path) -> Path:
    return runtime_dir(project_root) / VENV_DIRNAME


def venv_python(project_root: Path) -> Path:
    """Path to the venv's own interpreter -- platform-specific layout."""
    vdir = venv_dir(project_root)
    if sys.platform.startswith("win"):
        return vdir / "Scripts" / "python.exe"
    return vdir / "bin" / "python"


@dataclass
class RuntimeStatus:
    exists: bool
    healthy: bool
    python_path: Path
    reason: str = ""


def check_runtime(project_root: Path) -> RuntimeStatus:
    """Checks whether the private runtime exists AND is actually usable
    (not just a directory left over from an interrupted setup)."""
    py = venv_python(project_root)
    if not py.exists():
        return RuntimeStatus(exists=False, healthy=False, python_path=py, reason="no venv at .runtime/venv")

    try:
        proc = subprocess.run(
            [str(py), "-c", "import sys; print(sys.version_info[:2])"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeStatus(exists=True, healthy=False, python_path=py, reason=f"venv python unusable: {exc}")

    if proc.returncode != 0:
        return RuntimeStatus(exists=True, healthy=False, python_path=py, reason="venv python failed to run")

    return RuntimeStatus(exists=True, healthy=True, python_path=py, reason="ok")


def create_runtime(project_root: Path, base_python: str) -> RuntimeStatus:
    """Creates the private venv using `base_python` as the base interpreter
    (via `venv` module semantics: spawns `base_python -m venv <dir>` so the
    venv is built by the selected interpreter, not whatever runs this code)."""
    vdir = venv_dir(project_root)
    vdir.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [base_python, "-m", "venv", str(vdir)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if proc.returncode != 0:
        return RuntimeStatus(
            exists=vdir.exists(), healthy=False, python_path=venv_python(project_root),
            reason=f"venv creation failed: {proc.stderr.strip()[:400]}",
        )
    return check_runtime(project_root)


def rebuild_runtime(project_root: Path, base_python: str) -> RuntimeStatus:
    """Repair path (spec `--repair`): unconditionally removes the existing
    runtime -- healthy or not -- and recreates it from scratch. Never touches
    anything outside `.runtime/` -- user media, caches, and project source
    are untouched."""
    import shutil

    rdir = runtime_dir(project_root)
    if rdir.exists():
        shutil.rmtree(rdir, ignore_errors=True)
    return create_runtime(project_root, base_python)


def ensure_runtime(project_root: Path, base_python: str, repair: bool = False) -> RuntimeStatus:
    """Idempotent entry point: reuses a healthy runtime, creates one if
    missing, and unconditionally force-rebuilds it whenever `repair=True` is
    passed -- even if the runtime currently appears healthy."""
    status = check_runtime(project_root)
    if status.healthy and not repair:
        return status
    if repair or not status.healthy:
        return rebuild_runtime(project_root, base_python)
    return create_runtime(project_root, base_python)
