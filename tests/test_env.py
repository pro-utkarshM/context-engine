import os
from pathlib import Path

from context_engine.env import load_dotenv


def test_load_dotenv_sets_missing_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_MODEL=gpt-test\nOPENAI_BASE_URL=https://example.test/v1\n", encoding="utf-8")

    old_model = os.environ.pop("OPENAI_MODEL", None)
    old_base = os.environ.pop("OPENAI_BASE_URL", None)
    try:
        load_dotenv(env_file)
        assert os.environ["OPENAI_MODEL"] == "gpt-test"
        assert os.environ["OPENAI_BASE_URL"] == "https://example.test/v1"
    finally:
        if old_model is not None:
            os.environ["OPENAI_MODEL"] = old_model
        else:
            os.environ.pop("OPENAI_MODEL", None)
        if old_base is not None:
            os.environ["OPENAI_BASE_URL"] = old_base
        else:
            os.environ.pop("OPENAI_BASE_URL", None)
