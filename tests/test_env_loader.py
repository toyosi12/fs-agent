"""Tests for .env loading helper."""

from __future__ import annotations

import os
from pathlib import Path

from fs_agent.env_loader import load_env_file


def test_load_env_file_sets_values(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "FOO=bar\n"
        "export BAZ=qux\n"
        "QUOTED=\"hello world\"\n"
    )
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    loaded_path = load_env_file(env_file)

    assert loaded_path == env_file
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"
    assert os.environ["QUOTED"] == "hello world"


def test_load_env_file_respects_existing(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\n")
    monkeypatch.setenv("FOO", "existing")

    load_env_file(env_file)

    assert os.environ["FOO"] == "existing"

    load_env_file(env_file, override=True)
    assert os.environ["FOO"] == "bar"
