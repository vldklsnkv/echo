from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from audio_transcriber.asr_local import FasterWhisperASR, GigaAMLocalASR, MlxASR
from audio_transcriber.errors import SafeError
from audio_transcriber.models import (
    ASROptions,
    AudioChunk,
    AudioWindow,
    BackendFamily,
    ComputeDevice,
)


def _chunk() -> AudioChunk:
    return AudioChunk(
        AudioWindow("chunk-0000", 0, 5, 0, 0), Path("/private/tmp/chunk.flac"), 1, "a" * 64
    )


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: tuple[FakeWord, ...]
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class FakeGigaWord:
    text: str
    start: float
    end: float


@dataclass
class FakeGigaResponse:
    text: str
    words: tuple[FakeGigaWord, ...] | None


def test_gigaam_falls_back_only_the_failed_chunk_then_returns_to_mps() -> None:
    loads: list[tuple[str, str]] = []
    calls: list[tuple[str, bool, str]] = []
    fallback_calls: list[dict[str, object]] = []
    mps_loads = 0

    class Model:
        def __init__(self, device: str) -> None:
            self.device = device

        def transcribe(self, path: str, *, word_timestamps: bool) -> FakeGigaResponse:
            calls.append((path, word_timestamps, self.device))
            if len(calls) == 1:
                raise RuntimeError("unsupported MPS operation")
            return FakeGigaResponse(
                "Привет, мир.", (FakeGigaWord("Привет,", 0.1, 0.5), FakeGigaWord("мир.", 0.6, 0.9))
            )

    def factory(model: str, device: str) -> Model:
        nonlocal mps_loads
        loads.append((model, device))
        if device == "mps":
            mps_loads += 1
        return Model(device)

    def fallback_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        fallback_calls.append({"path": path, **kwargs})
        return {
            "language": "ru",
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "fallback",
                    "words": [{"start": 0, "end": 1, "word": "fallback"}],
                }
            ],
        }

    adapter = GigaAMLocalASR(
        factory,
        ComputeDevice.MPS,
        fallback=MlxASR(lambda: fallback_transcribe),
        fallback_model="mlx-fallback",
    )
    fallback_result = adapter.transcribe(_chunk(), ASROptions("v3_e2e_rnnt", "ru"))
    primary_result = adapter.transcribe(_chunk(), ASROptions("v3_e2e_rnnt", "ru"))

    assert loads == [("v3_e2e_rnnt", "mps"), ("v3_e2e_rnnt", "mps")]
    assert mps_loads == 2
    assert all(device == "mps" for _, _, device in calls)
    assert fallback_calls[0]["path_or_hf_repo"] == "mlx-fallback"
    assert fallback_result.backend == "mlx"
    assert primary_result.backend == "gigaam-mps"
    assert [word.text for word in primary_result.segments[0].words] == ["Привет,", "мир."]


def test_gigaam_rejects_text_without_word_timestamps() -> None:
    class Model:
        def transcribe(self, path: str, *, word_timestamps: bool) -> FakeGigaResponse:
            del path, word_timestamps
            return FakeGigaResponse("текст", None)

    adapter = GigaAMLocalASR(lambda _model, _device: Model(), ComputeDevice.CPU)
    with pytest.raises(SafeError) as raised:
        adapter.transcribe(_chunk(), ASROptions("v3_e2e_rnnt", "ru"))
    assert "GigaAM" in raised.value.render(set())


def test_mlx_requests_independent_word_timestamps_and_is_lazy() -> None:
    calls: list[dict[str, object]] = []
    loads = 0

    def factory():
        nonlocal loads
        loads += 1

        def transcribe(path: str, **kwargs: object) -> dict[str, object]:
            calls.append({"path": path, **kwargs})
            return {
                "language": "ru",
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "привет",
                        "words": [{"start": 0, "end": 1, "word": "привет", "probability": 0.9}],
                    }
                ],
            }

        return transcribe

    adapter = MlxASR(factory)
    result = adapter.transcribe(_chunk(), ASROptions(model="mlx-model", language="ru", beam_size=5))
    adapter.transcribe(_chunk(), ASROptions(model="mlx-model", language="ru", beam_size=5))

    assert loads == 1
    assert result.language == "ru"
    assert calls[0]["word_timestamps"] is True
    assert calls[0]["condition_on_previous_text"] is False
    assert calls[0]["path_or_hf_repo"] == "mlx-model"
    assert "beam_size" not in calls[0]


def test_mlx_resolves_the_default_whisper_model_to_an_mlx_repository() -> None:
    calls: list[dict[str, object]] = []

    def transcribe(path: str, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, **kwargs})
        return {"language": "ru", "segments": []}

    result = MlxASR(lambda: transcribe).transcribe(
        _chunk(), ASROptions(model="large-v3-turbo", language="ru")
    )

    assert calls[0]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert result.model == "mlx-community/whisper-large-v3-turbo"


def test_mlx_resolves_the_fast_english_model_to_an_mlx_repository() -> None:
    calls: list[dict[str, object]] = []

    def transcribe(path: str, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, **kwargs})
        return {"language": "en", "segments": []}

    result = MlxASR(lambda: transcribe).transcribe(
        _chunk(), ASROptions(model="distil-large-v3", language="en")
    )

    assert calls[0]["path_or_hf_repo"] == "mlx-community/distil-whisper-large-v3"
    assert result.model == "mlx-community/distil-whisper-large-v3"


def test_mlx_health_check_and_transcribe_use_the_same_resolved_model() -> None:
    def transcribe(path: str, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        return {"language": "en", "segments": []}

    adapter = MlxASR(lambda: transcribe)
    adapter.health_check("distil-large-v3")

    result = adapter.transcribe(
        _chunk(), ASROptions(model="distil-large-v3", language="en")
    )

    assert result.model == "mlx-community/distil-whisper-large-v3"


def test_mlx_ignores_non_finite_optional_backend_metrics() -> None:
    def transcribe(path: str, **kwargs: object) -> dict[str, object]:
        del path, kwargs
        return {
            "language": "en",
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "hello",
                    "avg_logprob": float("-inf"),
                    "no_speech_prob": float("nan"),
                    "words": [
                        {
                            "start": 0,
                            "end": 1,
                            "word": "hello",
                            "probability": float("inf"),
                        }
                    ],
                }
            ],
        }

    result = MlxASR(lambda: transcribe).transcribe(
        _chunk(), ASROptions(model="distil-large-v3", language="en")
    )

    segment = result.segments[0]
    assert segment.average_log_probability is None
    assert segment.no_speech_probability is None
    assert segment.words[0].confidence is None


def test_faster_whisper_uses_approved_cuda_or_cpu_compute_and_exhausts_segments() -> None:
    loads: list[tuple[str, str, str]] = []
    calls: list[dict[str, object]] = []

    class Model:
        def transcribe(self, path: str, **kwargs: object):
            calls.append({"path": path, **kwargs})
            return iter((FakeSegment(0, 1, "hello", (FakeWord(0, 1, "hello", 0.8),)),)), type(
                "Info", (), {"language": "en"}
            )()

    def factory(model: str, device: str, compute_type: str) -> Model:
        loads.append((model, device, compute_type))
        return Model()

    cuda = FasterWhisperASR(factory, BackendFamily.CUDA, cuda_compute_type="float16")
    cpu = FasterWhisperASR(factory, BackendFamily.CPU, cuda_compute_type="float16")
    options = ASROptions(model="local", language="en", beam_size=5)

    assert cuda.transcribe(_chunk(), options).segments[0].words[0].text == "hello"
    cpu.transcribe(_chunk(), options)

    assert loads == [("local", "cuda", "float16"), ("local", "cpu", "int8")]
    assert calls[0]["word_timestamps"] is True
    assert calls[0]["condition_on_previous_text"] is False
    assert calls[0]["beam_size"] == 5
    assert calls[0]["language"] == "en"


def test_local_adapter_rejects_disabled_timestamp_contract_and_redacts_backend_errors() -> None:
    adapter = MlxASR(lambda: (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY=sk-test")))
    with pytest.raises(SafeError) as raised:
        adapter.transcribe(_chunk(), ASROptions(model="model", language=None))

    assert "sk-test" not in raised.value.render({"sk-test"})
    with pytest.raises(SafeError) as contract_error:
        adapter.transcribe(
            _chunk(),
            ASROptions(model="model", language=None, word_timestamps=False),
        )
    assert "timestamps" in contract_error.value.render(set())
