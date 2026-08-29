from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import cast

import pytest

from audio_transcriber.models import (
    ASRChunkResult,
    AudioChunk,
    AudioWindow,
    OutputPaths,
    TranscriptDocument,
)
import audio_transcriber.pipeline as pipeline_module
from audio_transcriber.pipeline import (
    RenderArtifact,
    publish_outputs,
    recover_incomplete_publish,
)
from audio_transcriber.state import RunStateStore


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rendered(run_root: Path) -> RenderArtifact:
    stage = run_root / "stages" / "06-render"
    stage.mkdir(parents=True)
    json_path = stage / "transcript.json"
    txt_path = stage / "transcript.txt"
    json_path.write_text('{"canonical": true}\n')
    txt_path.write_text("SPEAKER_00: hello\n")
    for path in (json_path, txt_path):
        os.chmod(path, 0o600)
    return RenderArtifact(
        document=cast(TranscriptDocument, None),
        staged_paths=OutputPaths(json_path, txt_path, None, None),
    )


def test_publish_refuses_overwrite_and_preserves_existing_targets(tmp_path: Path) -> None:
    store = RunStateStore(tmp_path / "runtime" / "runs" / "fingerprint", "fingerprint")
    rendered = _rendered(store.run_root)
    final = OutputPaths(tmp_path / "out.json", tmp_path / "out.txt", None, None)
    final.json.write_text("previous json")
    final.txt.write_text("previous txt")
    os.chmod(final.json, 0o640)

    with pytest.raises(FileExistsError):
        publish_outputs(rendered, final, overwrite=False, run_store=store)

    assert final.json.read_text() == "previous json"
    assert final.txt.read_text() == "previous txt"
    assert stat.S_IMODE(final.json.stat().st_mode) == 0o640


def test_publish_is_atomic_and_recovers_a_private_incomplete_journal(tmp_path: Path) -> None:
    store = RunStateStore(tmp_path / "runtime" / "runs" / "fingerprint", "fingerprint")
    rendered = _rendered(store.run_root)
    final = OutputPaths(tmp_path / "out.json", tmp_path / "out.txt", None, None)
    final.json.write_text("previous json")
    final.txt.write_text("previous txt")

    published = publish_outputs(rendered, final, overwrite=True, run_store=store)

    assert published == final
    assert final.json.read_bytes() == rendered.staged_paths.json.read_bytes()
    assert final.txt.read_bytes() == rendered.staged_paths.txt.read_bytes()
    assert stat.S_IMODE(final.json.stat().st_mode) == 0o600
    assert _sha256(final.json) == _sha256(rendered.staged_paths.json)
    recover_incomplete_publish(final, store)
    assert final.txt.read_bytes() == rendered.staged_paths.txt.read_bytes()


def test_publish_rejects_symlinked_targets(tmp_path: Path) -> None:
    store = RunStateStore(tmp_path / "runtime" / "runs" / "fingerprint", "fingerprint")
    rendered = _rendered(store.run_root)
    protected = tmp_path / "protected"
    protected.write_text("unchanged")
    final = OutputPaths(tmp_path / "out.json", tmp_path / "out.txt", None, None)
    final.json.symlink_to(protected)

    with pytest.raises(ValueError, match="unsafe"):
        publish_outputs(rendered, final, overwrite=True, run_store=store)

    assert protected.read_text() == "unchanged"


def test_asr_checkpoint_rejects_symlinked_directory(tmp_path: Path) -> None:
    store = RunStateStore(tmp_path / "runtime" / "runs" / "fingerprint", "fingerprint")
    checkpoint_parent = store.run_root / "checkpoints"
    checkpoint_parent.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    (checkpoint_parent / "asr").symlink_to(protected, target_is_directory=True)
    chunk = AudioChunk(
        AudioWindow("chunk-0000", 0.0, 1.0, 0.0, 0.0),
        tmp_path / "chunk.flac",
        1,
        "a" * 64,
    )
    result = ASRChunkResult(chunk, "fake", "model", "ru", ())

    with pytest.raises(ValueError, match="unsafe"):
        pipeline_module._write_asr_checkpoint(  # pyright: ignore[reportPrivateUsage]
            store, chunk, result
        )

    assert tuple(protected.iterdir()) == ()


def test_publish_failure_restores_prior_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStateStore(tmp_path / "runtime" / "runs" / "fingerprint", "fingerprint")
    rendered = _rendered(store.run_root)
    final = OutputPaths(tmp_path / "out.json", tmp_path / "out.txt", None, None)
    final.json.write_text("previous json")
    final.txt.write_text("previous txt")
    replace = pipeline_module.os.replace

    def fail_on_second_publish(source: Path | str, target: Path | str) -> None:
        if Path(target) == final.txt and Path(source).suffix == ".tmp":
            raise OSError("injected publish failure")
        replace(source, target)

    monkeypatch.setattr(pipeline_module.os, "replace", fail_on_second_publish)

    with pytest.raises(OSError, match="injected"):
        publish_outputs(rendered, final, overwrite=True, run_store=store)

    assert final.json.read_text() == "previous json"
    assert final.txt.read_text() == "previous txt"
    recover_incomplete_publish(final, store)


def test_completed_publish_recovery_removes_only_journaled_backups(tmp_path: Path) -> None:
    store = RunStateStore(tmp_path / "runtime" / "runs" / "fingerprint", "fingerprint")
    final = OutputPaths(tmp_path / "out.json", tmp_path / "out.txt", None, None)
    entries: list[dict[str, object]] = []
    for index, target in enumerate((final.json, final.txt)):
        target.write_text("published")
        backup = target.with_name(f".{target.stem}.fingerprint.{index}.deadbeef.backup")
        backup.write_text("previous")
        entries.append(
            {
                "source": str(target.absolute()),
                "target": str(target.absolute()),
                "temporary": str(target.with_suffix(".tmp").absolute()),
                "backup": str(backup.absolute()),
                "digest": "a" * 64,
                "had_predecessor": True,
            }
        )
    private_root = store.run_root.parent.parent
    digest_payload = "\x00".join(sorted(str(path.absolute()) for path in (final.json, final.txt)))
    digest = hashlib.sha256(digest_payload.encode()).hexdigest()
    journal_dir = private_root / "publishes"
    journal_dir.mkdir()
    journal = journal_dir / f"{digest}.json"
    journal.write_text(json.dumps({"complete": True, "entries": entries}))

    recover_incomplete_publish(final, store)

    assert final.json.read_text() == "published"
    assert final.txt.read_text() == "published"
    assert not journal.exists()
    assert not tuple(tmp_path.glob("*.backup"))
