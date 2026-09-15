"""Tests for bootstrap.ffmpeg and bootstrap.node: detection + OS guidance."""
from __future__ import annotations

from video_edit_agent.bootstrap import ffmpeg as ffmpeg_mod
from video_edit_agent.bootstrap import node as node_mod
from video_edit_agent.bootstrap.detect import OSKind


def test_check_ffmpeg_reports_missing_when_not_on_path(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod.shutil, "which", lambda name: None)
    status = ffmpeg_mod.check_ffmpeg()
    assert status.found is False
    assert status.healthy is False


def test_check_ffmpeg_reports_healthy_when_present_and_runs(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg_mod, "_probe_version", lambda path, version_flag="-version": (True, "6.1.1"))
    status = ffmpeg_mod.check_ffmpeg()
    assert status.found is True
    assert status.healthy is True
    assert status.version == "6.1.1"


def test_windows_guidance_prefers_detected_package_manager(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod, "detect_windows_package_manager", lambda: "winget")
    guidance = ffmpeg_mod.install_guidance(OSKind.WINDOWS)
    assert "winget install ffmpeg" in guidance


def test_windows_guidance_falls_back_when_no_package_manager(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod, "detect_windows_package_manager", lambda: None)
    guidance = ffmpeg_mod.install_guidance(OSKind.WINDOWS)
    assert "winget" in guidance
    assert "No package manager detected" in guidance


def test_macos_guidance_uses_brew_when_available(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod.shutil, "which", lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    guidance = ffmpeg_mod.install_guidance(OSKind.MACOS)
    assert "brew install ffmpeg" in guidance


def test_macos_guidance_without_brew_explains_how_to_get_it(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod.shutil, "which", lambda name: None)
    guidance = ffmpeg_mod.install_guidance(OSKind.MACOS)
    assert "brew.sh" in guidance


def test_linux_guidance_detects_apt(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod, "detect_linux_package_manager", lambda: "apt-get")
    guidance = ffmpeg_mod.install_guidance(OSKind.LINUX)
    assert "apt-get install -y ffmpeg" in guidance


def test_linux_guidance_detects_dnf(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod, "detect_linux_package_manager", lambda: "dnf")
    guidance = ffmpeg_mod.install_guidance(OSKind.LINUX)
    assert "dnf install -y ffmpeg" in guidance


def test_linux_guidance_unknown_package_manager_still_actionable(monkeypatch):
    monkeypatch.setattr(ffmpeg_mod, "detect_linux_package_manager", lambda: None)
    guidance = ffmpeg_mod.install_guidance(OSKind.LINUX)
    assert "package manager" in guidance


def test_guidance_never_executes_anything_itself(monkeypatch):
    """install_guidance must be pure text generation -- it must never call
    subprocess.run to actually install something."""
    calls = []
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)))
    ffmpeg_mod.install_guidance(OSKind.LINUX)
    ffmpeg_mod.install_guidance(OSKind.WINDOWS)
    ffmpeg_mod.install_guidance(OSKind.MACOS)
    assert calls == []


def test_check_node_reports_not_found(monkeypatch):
    monkeypatch.setattr(node_mod.shutil, "which", lambda name: None)
    status = node_mod.check_node()
    assert status.node_found is False
    assert status.remotion_ready is False
    assert "optional" in status.detail


def test_check_node_reports_remotion_ready_when_both_present(monkeypatch):
    monkeypatch.setattr(node_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(node_mod, "_version", lambda cmd: "v22.0.0" if "node" in cmd else "10.0.0")
    status = node_mod.check_node()
    assert status.remotion_ready is True
    assert "Remotion" in status.detail
