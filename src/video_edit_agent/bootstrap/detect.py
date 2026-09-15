"""OS detection and Python interpreter selection (v0.2.1 installation UX).

Fresh-install testing on Windows showed `py`/`python` can resolve to a
bleeding-edge Python (e.g. 3.13+) even when a compatible 3.11 is installed
side-by-side, and some optional extras (mediapipe, faster-whisper wheels)
lag behind new CPython releases. So interpreter selection is a deliberate,
ordered preference -- never "whatever `python` happens to mean" -- and every
candidate is validated (importable, meets `requires-python`) before use.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass

MIN_PYTHON = (3, 10)
PREFERRED_PYTHON = (3, 11)
# Ordered: prefer 3.11, then other verified-compatible minors, oldest-first
# fallback to whatever satisfies requires-python. Bleeding-edge (>3.12) is
# never auto-preferred over these even if it's the only thing on PATH --
# `select_python` still accepts it as a last resort so setup doesn't just fail.
COMPATIBLE_MINORS: tuple[tuple[int, int], ...] = ((3, 11), (3, 12), (3, 10))


class OSKind:
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    OTHER = "other"


def detect_os() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return OSKind.WINDOWS
    if system == "darwin":
        return OSKind.MACOS
    if system == "linux":
        return OSKind.LINUX
    return OSKind.OTHER


@dataclass
class PythonCandidate:
    command: list[str]  # argv prefix, e.g. ["py", "-3.11"] or ["python3.11"]
    label: str


def candidate_commands(os_kind: str | None = None) -> list[PythonCandidate]:
    """Ordered list of interpreter invocations to try, most-preferred first.

    Windows uses the `py` launcher with explicit minor-version flags (the
    documented, reliable way to pick a specific Python on Windows -- a bare
    `python`/`python3` on PATH is unreliable there and listed only as a last
    resort). macOS/Linux use versioned binary names first, `python3` last.
    """
    os_kind = os_kind or detect_os()
    if os_kind == OSKind.WINDOWS:
        return [
            PythonCandidate(["py", "-3.11"], "py -3.11"),
            PythonCandidate(["py", "-3.12"], "py -3.12"),
            PythonCandidate(["py", "-3.10"], "py -3.10"),
            PythonCandidate(["python"], "python"),
        ]
    return [
        PythonCandidate(["python3.11"], "python3.11"),
        PythonCandidate(["python3.12"], "python3.12"),
        PythonCandidate(["python3.10"], "python3.10"),
        PythonCandidate(["python3"], "python3"),
    ]


@dataclass
class PythonProbeResult:
    candidate: PythonCandidate
    ok: bool
    version: tuple[int, int, int] | None
    executable: str | None
    reason: str = ""


def probe_python(candidate: PythonCandidate, timeout: float = 10.0) -> PythonProbeResult:
    """Actually invokes the candidate and asks it for its own version/path,
    rather than trusting the command name (a `python3.11` shim can point
    anywhere)."""
    exe = shutil.which(candidate.command[0])
    if exe is None:
        return PythonProbeResult(candidate, False, None, None, "not found on PATH")
    probe_code = "import sys; print('%d.%d.%d' % sys.version_info[:3]); print(sys.executable)"
    try:
        proc = subprocess.run(
            [*candidate.command, "-c", probe_code],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PythonProbeResult(candidate, False, None, None, f"failed to run: {exc}")

    if proc.returncode != 0:
        return PythonProbeResult(candidate, False, None, None, f"exited {proc.returncode}: {proc.stderr.strip()[:200]}")

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return PythonProbeResult(candidate, False, None, None, "unexpected probe output")

    try:
        major, minor, patch = (int(p) for p in lines[0].split("."))
    except ValueError:
        return PythonProbeResult(candidate, False, None, None, f"unparseable version: {lines[0]!r}")

    return PythonProbeResult(candidate, True, (major, minor, patch), lines[1])


@dataclass
class PythonSelection:
    executable: str
    version: tuple[int, int, int]
    command_label: str
    explanation: str


def _version_ok(version: tuple[int, int, int]) -> bool:
    return version[:2] >= MIN_PYTHON


def select_python(
    os_kind: str | None = None,
    prober=probe_python,
) -> PythonSelection | None:
    """Selects the best available interpreter: prefer 3.11, then the other
    verified-compatible minors, then anything meeting `requires-python`.
    Returns None if nothing on the system satisfies MIN_PYTHON. Always
    explains why the chosen interpreter was picked, per spec section 3."""
    os_kind = os_kind or detect_os()
    candidates = candidate_commands(os_kind)
    results = [prober(c) for c in candidates]
    valid = [r for r in results if r.ok and r.version and _version_ok(r.version)]

    if not valid:
        return None

    def rank(r: PythonProbeResult) -> int:
        v = r.version[:2]
        if v in COMPATIBLE_MINORS:
            return COMPATIBLE_MINORS.index(v)
        # Anything else meeting MIN_PYTHON (bleeding-edge or older-than-preferred)
        # ranks after all explicitly-compatible minors.
        return len(COMPATIBLE_MINORS)

    best = min(valid, key=rank)
    is_bleeding_edge = best.version[:2] not in COMPATIBLE_MINORS
    explanation = (
        f"selected {best.candidate.label} (Python {'.'.join(map(str, best.version))}) "
        f"at {best.executable}"
    )
    if is_bleeding_edge:
        explanation += (
            " -- no 3.11/3.12/3.10 found; falling back to this interpreter because it "
            "still satisfies requires-python>=3.10. Some optional extras (mediapipe, "
            "faster-whisper) may lag behind this version."
        )
    else:
        explanation += " (preferred/compatible minor version)."

    return PythonSelection(
        executable=best.executable,  # type: ignore[arg-type]
        version=best.version,  # type: ignore[arg-type]
        command_label=best.candidate.label,
        explanation=explanation,
    )


def current_interpreter_is_acceptable() -> bool:
    """True if the interpreter currently running videoedit itself already
    satisfies MIN_PYTHON -- lets `setup` skip creating a private runtime
    when the invoking interpreter is already fine (still isolated into
    .runtime by default, but avoids a pointless re-probe)."""
    return sys.version_info[:2] >= MIN_PYTHON
