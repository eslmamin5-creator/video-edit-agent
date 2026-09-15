"""Tests for bootstrap.detect: OS detection and Python interpreter selection."""
from __future__ import annotations

from video_edit_agent.bootstrap.detect import (
    COMPATIBLE_MINORS,
    MIN_PYTHON,
    OSKind,
    PythonCandidate,
    PythonProbeResult,
    candidate_commands,
    current_interpreter_is_acceptable,
    detect_os,
    select_python,
)


def test_detect_os_windows(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert detect_os() == OSKind.WINDOWS


def test_detect_os_macos(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert detect_os() == OSKind.MACOS


def test_detect_os_linux(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert detect_os() == OSKind.LINUX


def test_detect_os_other(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "SomeExoticOS")
    assert detect_os() == OSKind.OTHER


def test_windows_candidates_use_py_launcher_with_explicit_minors():
    candidates = candidate_commands(OSKind.WINDOWS)
    labels = [c.label for c in candidates]
    assert "py -3.11" in labels
    assert "py -3.12" in labels
    assert "py -3.10" in labels
    # bare `python` is present only as a last resort
    assert labels[-1] == "python"
    assert labels.index("py -3.11") < labels.index("python")


def test_unix_candidates_prefer_versioned_binaries():
    candidates = candidate_commands(OSKind.LINUX)
    labels = [c.label for c in candidates]
    assert labels[0] == "python3.11"
    assert labels[-1] == "python3"


def _fake_prober(version_by_label: dict[str, tuple[int, int, int] | None]):
    def prober(candidate: PythonCandidate) -> PythonProbeResult:
        version = version_by_label.get(candidate.label)
        if version is None:
            return PythonProbeResult(candidate, False, None, None, "not found")
        return PythonProbeResult(candidate, True, version, f"/usr/bin/{candidate.label.replace(' ', '_')}")
    return prober


def test_select_python_prefers_311_over_bleeding_edge():
    prober = _fake_prober({
        "python3.11": (3, 11, 9),
        "python3.12": None,
        "python3.10": None,
        "python3": (3, 13, 1),  # bleeding edge, also present
    })
    selection = select_python(OSKind.LINUX, prober=prober)
    assert selection is not None
    assert selection.version == (3, 11, 9)
    assert "preferred/compatible" in selection.explanation


def test_select_python_falls_back_to_bleeding_edge_when_nothing_else_available():
    prober = _fake_prober({
        "python3.11": None,
        "python3.12": None,
        "python3.10": None,
        "python3": (3, 13, 1),
    })
    selection = select_python(OSKind.LINUX, prober=prober)
    assert selection is not None
    assert selection.version == (3, 13, 1)
    assert "falling back" in selection.explanation


def test_select_python_returns_none_when_nothing_meets_minimum():
    prober = _fake_prober({
        "python3.11": None,
        "python3.12": None,
        "python3.10": None,
        "python3": (3, 8, 10),  # below MIN_PYTHON
    })
    assert select_python(OSKind.LINUX, prober=prober) is None


def test_select_python_explains_choice():
    prober = _fake_prober({"python3.11": (3, 11, 0), "python3.12": None, "python3.10": None, "python3": None})
    selection = select_python(OSKind.LINUX, prober=prober)
    assert selection is not None
    assert "3.11.0" in selection.explanation
    assert selection.command_label == "python3.11"


def test_compatible_minors_prefers_311_first():
    assert COMPATIBLE_MINORS[0] == (3, 11)


def test_min_python_matches_requires_python():
    assert MIN_PYTHON == (3, 10)


def test_current_interpreter_is_acceptable_reflects_running_interpreter():
    # This test suite itself only runs on Python >= 3.10 (pyproject requires it).
    assert current_interpreter_is_acceptable() is True
