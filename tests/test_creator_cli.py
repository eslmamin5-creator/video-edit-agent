"""CLI `videoedit create` command tests (Creator spec section 27-29)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from video_edit_agent.agents.creator.parser import ScriptParseError
from video_edit_agent.agents.creator.pipeline import CreatorResult
from video_edit_agent.agents.creator.schemas import ScriptAnalysis
from video_edit_agent.cli.main import app

runner = CliRunner()


def _fake_result(tmp_path: Path, final_output: Path | None) -> CreatorResult:
    return CreatorResult(
        project_dir=tmp_path / "edit",
        final_output=final_output,
        script_analysis=ScriptAnalysis(),
        scenes=[],
    )


def test_create_command_reports_error_for_unsupported_script(tmp_path: Path):
    bad_script = tmp_path / "script.xyz"
    bad_script.write_text("data", encoding="utf-8")
    with patch("video_edit_agent.cli.main.run_creator", side_effect=ScriptParseError("bad format")):
        result = runner.invoke(app, ["create", str(bad_script)])
    assert result.exit_code == 1
    assert "bad format" in result.output


def test_create_command_reports_error_for_invalid_style(tmp_path: Path):
    script = tmp_path / "script.txt"
    script.write_text("Hello world.", encoding="utf-8")
    with patch("video_edit_agent.cli.main.run_creator", side_effect=ValueError("'bogus' is not a valid CreatorStyle")):
        result = runner.invoke(app, ["create", str(script), "--style", "bogus"])
    assert result.exit_code == 1
    assert "Invalid --style" in result.output


def test_create_command_succeeds_and_prints_output_path(tmp_path: Path):
    script = tmp_path / "script.txt"
    script.write_text("Hello world.", encoding="utf-8")
    final_output = tmp_path / "final.mp4"
    final_output.write_bytes(b"fake")
    fake_result = _fake_result(tmp_path, final_output)
    with patch("video_edit_agent.cli.main.run_creator", return_value=fake_result) as mock_run:
        result = runner.invoke(app, ["create", str(script), "--style", "mixed", "--offline"])
    assert result.exit_code == 0
    assert final_output.name in result.output
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["offline"] is True
    assert kwargs["style"] == "mixed"


def test_create_command_exits_nonzero_when_no_final_output(tmp_path: Path):
    script = tmp_path / "script.txt"
    script.write_text("Hello world.", encoding="utf-8")
    fake_result = _fake_result(tmp_path, None)
    with patch("video_edit_agent.cli.main.run_creator", return_value=fake_result):
        result = runner.invoke(app, ["create", str(script)])
    assert result.exit_code == 1
