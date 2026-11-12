"""Reading .env so commands work without `make`, and a clear message when the API key is missing."""

import pytest

from welldata import config
from welldata.llm import AnthropicProvider


def test_env_file_values_are_loaded_without_overriding_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nWELL_TEST_A=from_file\nWELL_TEST_B='quoted value'\nWELL_TEST_C=from_file\n")
    monkeypatch.delenv("WELL_TEST_A", raising=False)
    monkeypatch.delenv("WELL_TEST_B", raising=False)
    monkeypatch.setenv("WELL_TEST_C", "already_set")

    config.load_env_file(env)

    import os
    assert os.environ["WELL_TEST_A"] == "from_file"
    assert os.environ["WELL_TEST_B"] == "quoted value"
    assert os.environ["WELL_TEST_C"] == "already_set"


def test_missing_env_file_is_fine(tmp_path):
    config.load_env_file(tmp_path / "nope.env")


def test_missing_api_key_gives_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("claude-haiku-5-5")
