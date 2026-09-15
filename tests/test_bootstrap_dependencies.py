"""Tests for bootstrap.dependencies: install profiles."""
from __future__ import annotations

from pathlib import Path

import pytest

from video_edit_agent.bootstrap.dependencies import (
    DEFAULT_PROFILE,
    PROFILE_EXTRAS,
    extras_for_profile,
    known_profiles,
    pip_install_spec,
)


def test_known_profiles_include_required_set():
    profiles = known_profiles()
    for expected in ("core", "local", "subject", "motion", "full-local"):
        assert expected in profiles


def test_default_profile_is_full_local():
    assert DEFAULT_PROFILE == "full-local"


def test_core_profile_has_no_extras():
    assert extras_for_profile("core") == ()


def test_full_local_profile_maps_to_full_local_extra():
    assert extras_for_profile("full-local") == ("full-local",)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        extras_for_profile("does-not-exist")


def test_pip_install_spec_core_has_no_brackets(tmp_path: Path):
    spec = pip_install_spec(tmp_path, "core")
    assert spec == str(tmp_path)
    assert "[" not in spec


def test_pip_install_spec_includes_extras_bracket(tmp_path: Path):
    spec = pip_install_spec(tmp_path, "subject")
    assert spec == f"{tmp_path}[local,subject]"


def test_every_profile_is_a_valid_pip_spec_shape(tmp_path: Path):
    for profile in known_profiles():
        spec = pip_install_spec(tmp_path, profile)
        assert spec.startswith(str(tmp_path))


def test_profile_extras_table_has_no_duplicated_unknown_extras():
    known_pyproject_extras = {"local", "gemini", "elevenlabs", "motion", "subject", "dev", "full-local", "all"}
    for extras in PROFILE_EXTRAS.values():
        for extra in extras:
            assert extra in known_pyproject_extras, f"profile references unknown extra {extra!r}"
