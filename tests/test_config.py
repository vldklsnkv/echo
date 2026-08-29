from pathlib import Path

import pytest

from audio_transcriber.config import (
    ConfigOverrides,
    resolve_config,
    select_env_file,
)
from audio_transcriber.constants import (
    DEFAULT_GIGAAM_MAX_CHUNK_DURATION_S,
    DEFAULT_GIGAAM_MODEL,
    DEFAULT_ENGLISH_LOCAL_MODEL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MAX_CHUNK_DURATION_S,
)
from audio_transcriber.models import LocalEngine, Mode


def _overrides(**changes: object) -> ConfigOverrides:
    values: dict[str, object] = {"input_path": Path("recording.m4a")}
    values.update(changes)
    return ConfigOverrides(**values)  # type: ignore[arg-type]


def test_repository_env_prefers_dotenv_over_dotenv_prod(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=selected-key\n")
    (tmp_path / ".env.prod").write_text("OPENAI_API_KEY=fallback-key\n")

    config = resolve_config(_overrides(), tmp_path)

    assert config.credentials.openai_api_key == "selected-key"
    assert select_env_file(None, tmp_path) == tmp_path / ".env"


def test_explicit_env_file_bypasses_discovery_and_is_required(tmp_path: Path) -> None:
    explicit = tmp_path / "credentials.env"
    explicit.write_text("HF_TOKEN=hf-selected\n")
    (tmp_path / ".env").write_text("HF_TOKEN=ignored\n")

    config = resolve_config(_overrides(env_file=explicit), tmp_path)

    assert config.credentials.hf_token == "hf-selected"
    assert select_env_file(explicit, tmp_path) == explicit
    with pytest.raises(ValueError, match="does not exist"):
        select_env_file(tmp_path / "missing.env", tmp_path)


def test_only_allowlisted_dotenv_secrets_are_loaded_and_ambient_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=file-key\nHF_TOKEN=file-token\nUNRELATED_SECRET=must-not-load\n"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    monkeypatch.setenv("HF_TOKEN", "ambient-token")

    config = resolve_config(_overrides(), tmp_path)

    assert config.credentials.openai_api_key == "file-key"
    assert config.credentials.hf_token == "file-token"
    assert "file-key" not in repr(config.credentials)
    assert "file-token" not in repr(config)
    assert "credentials" not in config.to_dict()


def test_cli_overrides_safe_defaults() -> None:
    config = resolve_config(
        _overrides(mode=Mode.OPENAI, local_model="small", resume=False, render_srt=True),
        Path("/repository"),
    )

    assert config.mode is Mode.OPENAI
    assert config.local_model == "small"
    assert not config.resume
    assert config.render_srt


def test_local_engine_auto_routes_russian_to_gigaam_and_other_languages_to_whisper() -> None:
    russian = resolve_config(_overrides(mode=Mode.LOCAL, language="ru"), Path("/repository"))
    english = resolve_config(_overrides(mode=Mode.LOCAL, language="en"), Path("/repository"))
    unspecified = resolve_config(_overrides(mode=Mode.LOCAL), Path("/repository"))
    cloud = resolve_config(_overrides(mode=Mode.OPENAI, language="ru"), Path("/repository"))

    assert (russian.local_engine, russian.local_model, russian.max_chunk_duration_s) == (
        LocalEngine.GIGAAM,
        DEFAULT_GIGAAM_MODEL,
        DEFAULT_GIGAAM_MAX_CHUNK_DURATION_S,
    )
    assert (english.local_engine, english.local_model, english.max_chunk_duration_s) == (
        LocalEngine.WHISPER,
        DEFAULT_ENGLISH_LOCAL_MODEL,
        DEFAULT_MAX_CHUNK_DURATION_S,
    )
    assert unspecified.local_engine is LocalEngine.WHISPER
    assert cloud.max_chunk_duration_s == DEFAULT_MAX_CHUNK_DURATION_S


def test_gigaam_can_be_overridden_but_rejects_oversized_chunks() -> None:
    whisper = resolve_config(
        _overrides(mode=Mode.LOCAL, language="ru", local_engine=LocalEngine.WHISPER),
        Path("/repository"),
    )
    assert whisper.local_engine is LocalEngine.WHISPER
    assert whisper.local_model == DEFAULT_LOCAL_MODEL

    with pytest.raises(ValueError, match="must not exceed"):
        resolve_config(
            _overrides(
                mode=Mode.LOCAL,
                language="ru",
                local_engine=LocalEngine.GIGAAM,
                max_chunk_duration_s=25,
            ),
            Path("/repository"),
        )


def test_speaker_count_invariants_and_name_mapping() -> None:
    config = resolve_config(
        _overrides(
            speakers=3,
            speaker_names="SPEAKER_00=Alice,SPEAKER_01=Bob",
        ),
        Path("/repository"),
    )

    assert config.speaker_count.exact == 3
    assert config.speaker_names == (("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_config(_overrides(speakers=2, min_speakers=1), Path("/repository"))
    with pytest.raises(ValueError, match="SPEAKER_00=Alice"):
        resolve_config(_overrides(speaker_names="Alice"), Path("/repository"))


def test_control_characters_in_paths_are_rejected() -> None:
    with pytest.raises(ValueError, match="control characters"):
        resolve_config(_overrides(input_path=Path("bad\nname.m4a")), Path("/repository"))
