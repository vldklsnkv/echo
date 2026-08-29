from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

import audio_transcriber.pipeline as pipeline_module
from audio_transcriber.config import ConfigOverrides, ResolvedConfig, resolve_config
from audio_transcriber.models import (
    ASRChunkResult,
    ASROptions,
    AsrSegment,
    AudioChunk,
    BackendFamily,
    ComputeDevice,
    DiarizationTurn,
    Mode,
    SpeakerCount,
    Word,
)
from audio_transcriber.pipeline import PipelineDependencies, PipelineRunner, publish_outputs
from audio_transcriber.bootstrap import RuntimeHandle


class _Runner:
    def __init__(self, duration_s: float = 2.0) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._duration_s = duration_s
        self._artifact_durations: dict[str, float] = {}

    def __call__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout_s
        command = tuple(argv)
        self.calls.append(command)
        if command[0] == "ffprobe":
            duration_s = self._artifact_durations.get(command[-1], self._duration_s)
            payload = {
                "format": {"duration": str(duration_s), "format_name": "wav", "size": "5"},
                "streams": [
                    {
                        "index": 0,
                        "codec_type": "audio",
                        "codec_name": "pcm_s16le",
                        "sample_rate": "16000",
                        "channels": 1,
                    }
                ],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[0] == "ffmpeg":
            if "volumedetect" in command:
                return subprocess.CompletedProcess(command, 0, "", "mean_volume: -20.0 dB")
            Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(command[-1]).write_bytes(b"audio")
            duration_s = (
                float(command[command.index("-t") + 1]) if "-t" in command else self._duration_s
            )
            self._artifact_durations[command[-1]] = duration_s
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 1, "", "unexpected command")


class _ASR:
    def __init__(self, events: list[str] | None = None) -> None:
        self.transcribe_calls = 0
        self.health_checks = 0
        self.close_calls = 0
        self.chunk_ids: list[str] = []
        self._events = events

    def health_check(self, model: str) -> None:
        assert model
        self.health_checks += 1
        if self._events is not None:
            self._events.append("asr:health")

    def transcribe(self, chunk: AudioChunk, options: ASROptions) -> ASRChunkResult:
        self.transcribe_calls += 1
        self.chunk_ids.append(chunk.window.id)
        words = (
            Word(0.0, 0.4, "Привет", 0.9, chunk.window.id),
            Word(0.5, 0.9, "world", 0.9, chunk.window.id),
        )
        return ASRChunkResult(
            chunk,
            "fake-local",
            options.model,
            "ru",
            (AsrSegment(0.0, 0.9, "Привет world", words, None, None),),
        )

    def close(self) -> None:
        self.close_calls += 1
        if self._events is not None:
            self._events.append("asr:close")


class _InterruptingASR(_ASR):
    def __init__(self, completed_before_interrupt: int) -> None:
        super().__init__()
        self._completed_before_interrupt = completed_before_interrupt

    def transcribe(self, chunk: AudioChunk, options: ASROptions) -> ASRChunkResult:
        if self.transcribe_calls >= self._completed_before_interrupt:
            raise RuntimeError("simulated interruption")
        return super().transcribe(chunk, options)


class _Diarizer:
    def __init__(self, duration_s: float = 2.0, events: list[str] | None = None) -> None:
        self.calls = 0
        self.health_checks = 0
        self.close_calls = 0
        self._duration_s = duration_s
        self._events = events

    def health_check(self) -> None:
        self.health_checks += 1
        if self._events is not None:
            self._events.append("diarizer:health")

    def close(self) -> None:
        self.close_calls += 1
        if self._events is not None:
            self._events.append("diarizer:close")

    def diarize(self, audio_path: Path, speakers: SpeakerCount) -> tuple[DiarizationTurn, ...]:
        assert audio_path.is_file()
        del speakers
        self.calls += 1
        return (DiarizationTurn(0.0, self._duration_s, "SPEAKER_00"),)


def _runtime(root: Path) -> RuntimeHandle:
    return RuntimeHandle(
        runtime_fingerprint="a" * 64,
        base_fingerprint="b" * 64,
        python=Path("python"),
        backend=BackendFamily.CPU,
        compute_device=ComputeDevice.CPU,
        state_path=root / "runtime.json",
        plugin_root=root,
        recovery_attempt=False,
    )


def test_local_pipeline_reuses_valid_stages_and_rebuilds_corrupt_descendants(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(b"input")
    (tmp_path / ".env").write_text("HF_TOKEN=hf-test\n")
    config = resolve_config(
        ConfigOverrides(input_path=source, mode=Mode.LOCAL, output_dir=tmp_path / "out"), tmp_path
    )
    runner = _Runner()
    events: list[str] = []
    asr = _ASR(events)
    diarizer = _Diarizer(events=events)
    dependencies = PipelineDependencies(
        command_runner=runner,
        which=lambda name: name,
        asr_factory=lambda _config, _runtime, _consent: asr,
        diarizer_factory=lambda _config, _runtime, _duration: diarizer,
        clock=lambda: datetime.now(UTC),
        publisher=publish_outputs,
        progress=events.append,
    )
    pipeline = PipelineRunner(_runtime(tmp_path), dependencies, state_root=tmp_path / "runtime")

    first = pipeline.run(config)
    assert first.json.is_file() and first.txt.is_file()
    assert asr.transcribe_calls == 1
    assert diarizer.calls == 1
    assert asr.close_calls == 1
    assert diarizer.close_calls == 1
    assert events.index("asr:close") < events.index("diarizer:health")
    assert any(event.startswith("asr | 1/1") for event in events)

    second = pipeline.run(config)
    assert second == first
    assert asr.transcribe_calls == 1
    assert diarizer.calls == 1

    state = next((tmp_path / "runtime" / "runs").iterdir())
    (state / "stages" / "04-diarize" / "diarization.json").write_text("{}")
    pipeline.run(config)

    assert asr.transcribe_calls == 1
    assert diarizer.calls == 2
    assert diarizer.close_calls == 2


def test_5400_second_asr_resume_reuses_every_completed_chunk(tmp_path: Path) -> None:
    source = tmp_path / "long-meeting.wav"
    source.write_bytes(b"input")
    (tmp_path / ".env").write_text("HF_TOKEN=hf-test\n")
    config = resolve_config(
        ConfigOverrides(
            input_path=source,
            mode=Mode.LOCAL,
            language="ru",
            output_dir=tmp_path / "out",
        ),
        tmp_path,
    )
    runner = _Runner(duration_s=5_400.0)
    active_asr: list[_InterruptingASR] = [_InterruptingASR(100)]
    dependencies = PipelineDependencies(
        command_runner=runner,
        which=lambda name: name,
        asr_factory=lambda _config, _runtime, _consent: active_asr[0],
        diarizer_factory=lambda _config, _runtime, duration: _Diarizer(duration),
        clock=lambda: datetime.now(UTC),
        publisher=publish_outputs,
    )
    pipeline = PipelineRunner(_runtime(tmp_path), dependencies, state_root=tmp_path / "runtime")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run(config)
    assert active_asr[0].chunk_ids == [f"chunk-{index:04d}" for index in range(100)]
    assert active_asr[0].close_calls == 1

    resumed = _InterruptingASR(1)
    active_asr[0] = resumed
    with pytest.raises(RuntimeError, match="simulated interruption"):
        pipeline.run(config)

    assert resumed.chunk_ids == ["chunk-0100"]
    assert resumed.close_calls == 1
    checkpoints = next((tmp_path / "runtime" / "runs").iterdir()) / "checkpoints" / "asr"
    assert len(tuple(checkpoints.glob("chunk-*.json"))) == 101


def test_asr_closes_when_checkpoint_cleanup_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"input")
    (tmp_path / ".env").write_text("HF_TOKEN=hf-test\n")
    config = resolve_config(
        ConfigOverrides(input_path=source, mode=Mode.LOCAL, output_dir=tmp_path / "out"),
        tmp_path,
    )
    asr = _ASR()
    diarizer = _Diarizer()
    pipeline = PipelineRunner(
        _runtime(tmp_path),
        PipelineDependencies(
            command_runner=_Runner(),
            which=lambda name: name,
            asr_factory=lambda _config, _runtime, _consent: asr,
            diarizer_factory=lambda _config, _runtime, _duration: diarizer,
            clock=lambda: datetime.now(UTC),
            publisher=publish_outputs,
        ),
        state_root=tmp_path / "runtime",
    )

    def interrupt_cleanup(_store: object) -> None:
        raise RuntimeError("checkpoint cleanup interrupted")

    monkeypatch.setattr(pipeline_module, "_clear_asr_checkpoints", interrupt_cleanup)

    with pytest.raises(RuntimeError, match="checkpoint cleanup interrupted"):
        pipeline.run(config)

    assert asr.close_calls == 1
    assert diarizer.health_checks == 0


def test_pipeline_refuses_to_publish_if_source_changes_mid_run(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"input")
    (tmp_path / ".env").write_text("HF_TOKEN=hf-test\n")
    config = resolve_config(
        ConfigOverrides(input_path=source, mode=Mode.LOCAL, output_dir=tmp_path / "out"),
        tmp_path,
    )

    class MutatingASR(_ASR):
        def transcribe(self, chunk: AudioChunk, options: ASROptions) -> ASRChunkResult:
            result = super().transcribe(chunk, options)
            source.write_bytes(b"changed recording")
            return result

    pipeline = PipelineRunner(
        _runtime(tmp_path),
        PipelineDependencies(
            command_runner=_Runner(),
            which=lambda name: name,
            asr_factory=lambda _config, _runtime, _consent: MutatingASR(),
            diarizer_factory=lambda _config, _runtime, duration: _Diarizer(duration),
            clock=lambda: datetime.now(UTC),
            publisher=publish_outputs,
        ),
        state_root=tmp_path / "runtime",
    )

    with pytest.raises(ValueError, match="changed after run state creation"):
        pipeline.run(config)

    assert not (tmp_path / "out" / "meeting.transcript.json").exists()


def test_fully_silent_pipeline_publishes_canonical_outputs(tmp_path: Path) -> None:
    source = tmp_path / "silence.wav"
    source.write_bytes(b"input")
    (tmp_path / ".env").write_text("HF_TOKEN=hf-test\n")
    config = resolve_config(
        ConfigOverrides(
            input_path=source,
            mode=Mode.LOCAL,
            output_dir=tmp_path / "out",
            render_srt=True,
            render_vtt=True,
        ),
        tmp_path,
    )

    class SilentASR(_ASR):
        def transcribe(self, chunk: AudioChunk, options: ASROptions) -> ASRChunkResult:
            self.transcribe_calls += 1
            self.chunk_ids.append(chunk.window.id)
            return ASRChunkResult(
                chunk,
                "fake-local",
                options.model,
                "ru",
                (),
                no_speech_probability=0.9,
            )

    class SilentDiarizer(_Diarizer):
        def diarize(self, audio_path: Path, speakers: SpeakerCount) -> tuple[DiarizationTurn, ...]:
            assert audio_path.is_file()
            del speakers
            self.calls += 1
            return ()

    pipeline = PipelineRunner(
        _runtime(tmp_path),
        PipelineDependencies(
            command_runner=_Runner(),
            which=lambda name: name,
            asr_factory=lambda _config, _runtime, _consent: SilentASR(),
            diarizer_factory=lambda _config, _runtime, duration: SilentDiarizer(duration),
            clock=lambda: datetime.now(UTC),
            publisher=publish_outputs,
        ),
        state_root=tmp_path / "runtime",
    )

    outputs = pipeline.run(config)

    payload = json.loads(outputs.json.read_text())
    assert payload["speakers"] == []
    assert payload["turns"] == []
    assert outputs.txt.read_text() == "\n"
    assert outputs.srt is not None and outputs.srt.read_text() == "\n"
    assert outputs.vtt is not None and outputs.vtt.read_text() == "WEBVTT\n\n\n"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are host prerequisites",
)
def test_local_generated_wav_pipeline_succeeds_offline(
    tmp_path: Path, generated_wav: tuple[Path, Path]
) -> None:
    source, _ = generated_wav
    (tmp_path / ".env").write_text("HF_TOKEN=hf-test\n")
    config = resolve_config(
        ConfigOverrides(input_path=source, mode=Mode.LOCAL, output_dir=tmp_path / "out"), tmp_path
    )
    asr = _ASR()
    diarizers: list[_Diarizer] = []

    def runner(
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )

    def diarizer_factory(
        _config: ResolvedConfig, _runtime: RuntimeHandle, duration_s: float
    ) -> _Diarizer:
        diarizer = _Diarizer(duration_s)
        diarizers.append(diarizer)
        return diarizer

    pipeline = PipelineRunner(
        _runtime(tmp_path),
        PipelineDependencies(
            command_runner=runner,
            which=shutil.which,
            asr_factory=lambda _config, _runtime, _consent: asr,
            diarizer_factory=diarizer_factory,
            clock=lambda: datetime.now(UTC),
            publisher=publish_outputs,
        ),
        state_root=tmp_path / "runtime",
    )

    outputs = pipeline.run(config)

    assert outputs.json.is_file() and outputs.txt.is_file()
    assert asr.transcribe_calls == 1
    assert diarizers[0].calls == 1
