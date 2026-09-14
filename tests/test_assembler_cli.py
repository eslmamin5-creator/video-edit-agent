"""CLI `videoedit assemble` command tests (Assembler spec section 34),
mirroring `test_creator_cli.py`'s pattern."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from video_edit_agent.agents.assembler.pipeline import AssemblerError, AssemblerResult
from video_edit_agent.agents.assembler.schemas import OrderPolicy
from video_edit_agent.cli.main import app

runner = CliRunner()


def _fake_result(tmp_path: Path, *, rough_cut: Path | None, final: Path | None) -> AssemblerResult:
    return AssemblerResult(
        project_dir=tmp_path / "edit",
        scene_inventory=[],
        rough_cut_path=rough_cut,
        final_output=final,
        order_policy=OrderPolicy.PRESERVE,
    )


def test_assemble_command_reports_error_for_missing_scenes(tmp_path: Path):
    with patch("video_edit_agent.cli.main.run_assembler", side_effect=AssemblerError("No supported video files")):
        result = runner.invoke(app, ["assemble", str(tmp_path), "--rough"])
    assert result.exit_code == 1
    assert "No supported video files" in result.output


def test_assemble_command_succeeds_and_prints_rough_cut_path(tmp_path: Path):
    rough_cut = tmp_path / "rough_cut.mp4"
    rough_cut.write_bytes(b"fake")
    fake_result = _fake_result(tmp_path, rough_cut=rough_cut, final=None)
    with patch("video_edit_agent.cli.main.run_assembler", return_value=fake_result) as mock_run:
        result = runner.invoke(app, ["assemble", str(tmp_path), "--rough", "--preserve-order", "--offline"])
    assert result.exit_code == 0
    assert rough_cut.name in result.output
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["offline"] is True
    assert kwargs["preserve_order"] is True
    assert kwargs["rough"] is True


def test_assemble_command_succeeds_and_prints_final_path(tmp_path: Path):
    final = tmp_path / "final.mp4"
    final.write_bytes(b"fake")
    fake_result = _fake_result(tmp_path, rough_cut=None, final=final)
    with patch("video_edit_agent.cli.main.run_assembler", return_value=fake_result):
        result = runner.invoke(app, ["assemble", str(tmp_path), "--finish"])
    assert result.exit_code == 0
    assert final.name in result.output


def test_assemble_command_exits_nonzero_when_no_output_produced(tmp_path: Path):
    fake_result = _fake_result(tmp_path, rough_cut=None, final=None)
    with patch("video_edit_agent.cli.main.run_assembler", return_value=fake_result):
        result = runner.invoke(app, ["assemble", str(tmp_path), "--rough"])
    assert result.exit_code == 1


def test_assemble_command_passes_order_and_script_options(tmp_path: Path):
    script = tmp_path / "script.txt"
    script.write_text("Hello world.", encoding="utf-8")
    rough_cut = tmp_path / "rough_cut.mp4"
    rough_cut.write_bytes(b"fake")
    fake_result = _fake_result(tmp_path, rough_cut=rough_cut, final=None)
    with patch("video_edit_agent.cli.main.run_assembler", return_value=fake_result) as mock_run:
        result = runner.invoke(
            app, ["assemble", str(tmp_path), "--rough", "--order", "script", "--script", str(script), "--brand", "acme_test"]
        )
    assert result.exit_code == 0
    _, kwargs = mock_run.call_args
    assert kwargs["order"] == "script"
    assert kwargs["script_path"] == script
    assert kwargs["brand_name"] == "acme_test"
