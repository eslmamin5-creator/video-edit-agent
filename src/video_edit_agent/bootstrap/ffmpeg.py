"""ffmpeg/ffprobe detection and OS-specific install guidance (spec section 8).

Never installs system software silently -- only detects, reports a version
when possible, and prints exact next-step commands for the detected OS/package
manager. The user always runs the install themselves.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from video_edit_agent.bootstrap.detect import OSKind, detect_os


@dataclass
class ExecutableStatus:
    name: str
    found: bool
    path: str | None
    version: str | None
    healthy: bool
    detail: str = ""


def _probe_version(path: str, version_flag: str = "-version") -> tuple[bool, str | None]:
    try:
        proc = subprocess.run([path, version_flag], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False, None
    if proc.returncode != 0:
        return False, None
    first_line = (proc.stdout or proc.stderr or "").splitlines()[:1]
    match = re.search(r"version\s+(\S+)", first_line[0]) if first_line else None
    return True, (match.group(1) if match else (first_line[0].strip() if first_line else None))


def check_ffmpeg() -> ExecutableStatus:
    path = shutil.which("ffmpeg")
    if not path:
        return ExecutableStatus("ffmpeg", False, None, None, False, "not found on PATH")
    healthy, version = _probe_version(path)
    return ExecutableStatus("ffmpeg", True, path, version, healthy, "OK" if healthy else "found but did not run")


def check_ffprobe() -> ExecutableStatus:
    path = shutil.which("ffprobe")
    if not path:
        return ExecutableStatus("ffprobe", False, None, None, False, "not found on PATH")
    healthy, version = _probe_version(path)
    return ExecutableStatus("ffprobe", True, path, version, healthy, "OK" if healthy else "found but did not run")


def detect_windows_package_manager() -> str | None:
    for candidate in ("winget", "choco", "scoop"):
        if shutil.which(candidate):
            return candidate
    return None


def detect_linux_package_manager() -> str | None:
    for candidate in ("apt-get", "apt", "dnf", "yum", "pacman", "zypper", "apk"):
        if shutil.which(candidate):
            return candidate
    return None


def install_guidance(os_kind: str | None = None) -> str:
    """Exact, OS-specific commands to install ffmpeg -- text only, never
    executed automatically."""
    os_kind = os_kind or detect_os()

    if os_kind == OSKind.WINDOWS:
        pm = detect_windows_package_manager()
        lines = ["ffmpeg was not found. Install it, then re-run `videoedit setup`:"]
        if pm == "winget":
            lines.append("  winget install ffmpeg")
        elif pm == "choco":
            lines.append("  choco install ffmpeg")
        elif pm == "scoop":
            lines.append("  scoop install ffmpeg")
        else:
            lines.append("  winget install ffmpeg   (or: choco install ffmpeg / scoop install ffmpeg)")
            lines.append("  No package manager detected on PATH -- install winget/choco/scoop first, or")
            lines.append("  download a build from https://www.gyan.dev/ffmpeg/builds/ and add it to PATH.")
        return "\n".join(lines)

    if os_kind == OSKind.MACOS:
        if shutil.which("brew"):
            return "ffmpeg was not found. Install it, then re-run `videoedit setup`:\n  brew install ffmpeg"
        return (
            "ffmpeg was not found, and Homebrew isn't on PATH.\n"
            "Install Homebrew first (https://brew.sh), then:\n  brew install ffmpeg"
        )

    if os_kind == OSKind.LINUX:
        pm = detect_linux_package_manager()
        commands = {
            "apt-get": "sudo apt-get update && sudo apt-get install -y ffmpeg",
            "apt": "sudo apt update && sudo apt install -y ffmpeg",
            "dnf": "sudo dnf install -y ffmpeg",
            "yum": "sudo yum install -y ffmpeg",
            "pacman": "sudo pacman -S ffmpeg",
            "zypper": "sudo zypper install ffmpeg",
            "apk": "sudo apk add ffmpeg",
        }
        if pm and pm in commands:
            return f"ffmpeg was not found. Install it, then re-run `videoedit setup`:\n  {commands[pm]}"
        return (
            "ffmpeg was not found and no known package manager was detected.\n"
            "Install ffmpeg using your distribution's package manager, then re-run `videoedit setup`."
        )

    return "ffmpeg was not found. Install it for your OS from https://ffmpeg.org/download.html and re-run `videoedit setup`."
