"""Tests for bootstrap.setup: orchestration, --check purity, error paths.

Heavy real-subprocess paths (actual venv creation + pip install) are covered
by manual fresh-install acceptance, not here -- these tests mock the
sub-steps to verify orchestration logic and error handling quickly.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from video_edit_agent.bootstrap import setup as setup_mod
from video_edit_agent.bootstrap.dependencies import InstallResult
from video_edit_agent.bootstrap.detect import PythonSelection
from video_edit_agent.bootstrap.runtime import RuntimeStatus


def _fake_python_selection() -> PythonSelection:
    return PythonSelection(
        executable="/usr/bin/python3.11", version=(3, 11, 9),
        command_label="python3.11", explanation="selected python3.11 (Python 3.11.9)",
    )


def test_run_check_makes_no_filesystem_changes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(setup_mod, "select_python", lambda os_kind=None: _fake_python_selection())
    outcome = setup_mod.run_check(tmp_path)
    assert outcome.runtime is None
    assert outcome.install is None
    # No .runtime directory should have been created by a read-only check.
    assert not (tmp_path / ".runtime").exists()


def test_run_check_reports_not_ok_when_no_python_found(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(setup_mod, "select_python", lambda os_kind=None: None)
    outcome = setup_mod.run_check(tmp_path)
    assert outcome.ok is False
    assert any("No compatible Python" in m for m in outcome.messages)


def test_run_setup_rejects_unknown_profile(tmp_path: Path):
    try:
        setup_mod.run_setup(tmp_path, profile="not-a-real-profile")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown profile" in str(exc)


def test_run_setup_fails_cleanly_when_no_python(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(setup_mod, "select_python", lambda os_kind=None: None)
    outcome = setup_mod.run_setup(tmp_path, profile="core")
    assert outcome.ok is False
    assert outcome.runtime is None


def test_run_setup_fails_cleanly_when_runtime_creation_fails(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(setup_mod, "select_python", lambda os_kind=None: _fake_python_selection())
    monkeypatch.setattr(
        setup_mod, "ensure_runtime",
        lambda project_root, base_python, repair=False: RuntimeStatus(
            exists=False, healthy=False, python_path=tmp_path / ".runtime/venv/bin/python", reason="boom",
        ),
    )
    outcome = setup_mod.run_setup(tmp_path, profile="core")
    assert outcome.ok is False
    assert any("Failed to prepare the private runtime" in m for m in outcome.messages)


def test_run_setup_happy_path_persists_state(tmp_path: Path, monkeypatch):
    fake_python_path = tmp_path / ".runtime" / "venv" / "bin" / "python"
    fake_python_path.parent.mkdir(parents=True)
    fake_python_path.write_text("#!/bin/sh\n")

    monkeypatch.setattr(setup_mod, "select_python", lambda os_kind=None: _fake_python_selection())
    monkeypatch.setattr(
        setup_mod, "ensure_runtime",
        lambda project_root, base_python, repair=False: RuntimeStatus(
            exists=True, healthy=True, python_path=fake_python_path, reason="ok",
        ),
    )
    monkeypatch.setattr(
        setup_mod, "install_profile",
        lambda project_root, python_executable, profile="full-local": InstallResult(True, profile, "spec", "installed"),
    )

    class _FFStatus:
        def __init__(self, healthy, detail="OK", version="6.1.1"):
            self.healthy = healthy
            self.detail = detail
            self.version = version

    monkeypatch.setattr(setup_mod.ffmpeg_mod, "check_ffmpeg", lambda: _FFStatus(True))
    monkeypatch.setattr(setup_mod.ffmpeg_mod, "check_ffprobe", lambda: _FFStatus(True))

    class _NodeStatus:
        node_version = "v22.0.0"
        npm_version = "10.0.0"
        detail = "node/npm ok"

    monkeypatch.setattr(setup_mod.node_mod, "check_node", lambda: _NodeStatus())
    monkeypatch.setattr(setup_mod, "capability_states_in_runtime", lambda python_executable: [])

    outcome = setup_mod.run_setup(tmp_path, profile="core")
    assert outcome.ok is True
    assert outcome.state_path is not None
    assert outcome.state_path.exists()


def test_capability_states_in_runtime_returns_none_on_subprocess_failure(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no such interpreter")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert setup_mod.capability_states_in_runtime("/no/such/python") is None


def test_capability_states_in_runtime_parses_json_output(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = '[{"name": "ffmpeg", "state": "verified", "detail": "OK"}]'

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    states = setup_mod.capability_states_in_runtime("/fake/python")
    assert states is not None
    assert states[0].name == "ffmpeg"
    assert states[0].state == "verified"


def test_render_text_report_includes_ready_marker():
    from video_edit_agent.bootstrap import ffmpeg as ffmpeg_mod
    from video_edit_agent.bootstrap import node as node_mod

    fake_outcome = setup_mod.SetupOutcome(
        ok=True, os_kind="linux", python=_fake_python_selection(), runtime=None, install=None,
        ffmpeg=ffmpeg_mod.ExecutableStatus("ffmpeg", True, "/usr/bin/ffmpeg", "6.1.1", True, "OK"),
        ffprobe=ffmpeg_mod.ExecutableStatus("ffprobe", True, "/usr/bin/ffprobe", "6.1.1", True, "OK"),
        node=node_mod.NodeStatus(True, "v22.0.0", True, "10.0.0", True, "node ok"),
    )
    text = setup_mod.render_text_report(fake_outcome)
    assert "READY" in text
    assert "ffmpeg: OK" in text
