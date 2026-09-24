"""Install profiles (spec section 5): named bundles of the pyproject
`[project.optional-dependencies]` extras, so `videoedit setup --profile X`
maps to a single, predictable `pip install` rather than users guessing which
extras they need.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

# Keep in sync with pyproject.toml [project.optional-dependencies].
# "core" = base install only, no extras.
PROFILE_EXTRAS: dict[str, tuple[str, ...]] = {
    "core": (),
    "local": ("local",),
    "subject": ("local", "subject"),
    "motion": ("local", "motion"),
    "full-local": ("full-local",),
}

DEFAULT_PROFILE = "full-local"


def known_profiles() -> list[str]:
    return sorted(PROFILE_EXTRAS)


def extras_for_profile(profile: str) -> tuple[str, ...]:
    if profile not in PROFILE_EXTRAS:
        raise ValueError(f"unknown install profile {profile!r}; choose one of {known_profiles()}")
    return PROFILE_EXTRAS[profile]


def pip_install_spec(project_root: Path, profile: str) -> str:
    """Builds the `pip install -e <path>[extra1,extra2]` target for a
    source install (spec section 13: source install is first-class, not a
    PyPI-only story)."""
    extras = extras_for_profile(profile)
    base = str(project_root)
    if not extras:
        return base
    return f"{base}[{','.join(extras)}]"


@dataclass
class InstallResult:
    ok: bool
    profile: str
    spec: str
    detail: str = ""


def install_extras(
    project_root: Path,
    python_executable: str,
    extras: tuple[str, ...],
    editable: bool = True,
    timeout: float = 900.0,
) -> InstallResult:
    """Installs specific pyproject extras (e.g. `("gemini",)`) into the given
    interpreter, independent of the named install profiles above.

    Needed because a profile's own extras (e.g. "full-local", which is
    deliberately offline-only and excludes "gemini"/"elevenlabs") may not
    cover a cloud provider the user configures an API key for *after* the
    profile was installed -- without this, a key saved via `videoedit
    setup`'s prompt would exist with no SDK installed, and the provider
    would silently never actually be usable despite `doctor` seeing a key.
    """
    if not extras:
        return InstallResult(True, "extras", str(project_root), "nothing to install")
    spec = f"{project_root}[{','.join(extras)}]"
    cmd = [python_executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(spec)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return InstallResult(False, "extras", spec, f"pip invocation failed: {exc}")

    if proc.returncode != 0:
        return InstallResult(False, "extras", spec, proc.stderr.strip()[-2000:])

    return InstallResult(True, "extras", spec, "installed")


def install_profile(
    project_root: Path,
    python_executable: str,
    profile: str = DEFAULT_PROFILE,
    editable: bool = True,
    timeout: float = 900.0,
) -> InstallResult:
    """Installs the requested profile into the given interpreter (expected
    to be the private runtime's own python -- this module never touches a
    global Python). Idempotent: pip itself no-ops on already-satisfied
    requirements, so re-running setup is safe and fast."""
    spec = pip_install_spec(project_root, profile)
    cmd = [python_executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(spec)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return InstallResult(False, profile, spec, f"pip invocation failed: {exc}")

    if proc.returncode != 0:
        return InstallResult(False, profile, spec, proc.stderr.strip()[-2000:])

    return InstallResult(True, profile, spec, "installed")
