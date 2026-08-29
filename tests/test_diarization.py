from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from audio_transcriber.diarization import (
    FLUID_AUDIO_REVISION,
    FluidAudioDiarizer,
    validate_diarization,
)
from audio_transcriber.errors import SafeError
from audio_transcriber.models import DiarizationTurn, SpeakerCount


def test_validate_diarization_accepts_sorted_overlapping_turns_and_automatic_silence() -> None:
    turns = (
        DiarizationTurn(0.0, 2.0, "SPEAKER_00"),
        DiarizationTurn(1.0, 3.0, "SPEAKER_01"),
    )

    validate_diarization(turns, duration_s=3.0, speakers=SpeakerCount())
    validate_diarization((), duration_s=3.0, speakers=SpeakerCount())


def _unchecked_turn(start_s: float, end_s: float, raw_speaker: str) -> DiarizationTurn:
    turn = object.__new__(DiarizationTurn)
    object.__setattr__(turn, "start_s", start_s)
    object.__setattr__(turn, "end_s", end_s)
    object.__setattr__(turn, "raw_speaker", raw_speaker)
    return turn


@pytest.mark.parametrize(
    "turns",
    [
        (_unchecked_turn(-0.1, 1.0, "SPEAKER_00"),),
        (_unchecked_turn(2.0, 1.0, "SPEAKER_00"),),
        (_unchecked_turn(0.0, 3.1, "SPEAKER_00"),),
        (_unchecked_turn(0.0, 1.0, ""),),
        (_unchecked_turn(1.0, 2.0, "SPEAKER_01"), _unchecked_turn(0.0, 1.0, "SPEAKER_00")),
    ],
)
def test_validate_diarization_rejects_invalid_or_unsorted_turns(
    turns: tuple[DiarizationTurn, ...],
) -> None:
    with pytest.raises(ValueError):
        validate_diarization(turns, duration_s=3.0, speakers=SpeakerCount())


def test_validate_diarization_enforces_exact_and_range_counts_without_relabeling() -> None:
    turns = (
        DiarizationTurn(0.0, 1.0, "SPEAKER_00"),
        DiarizationTurn(1.0, 2.0, "SPEAKER_01"),
        DiarizationTurn(2.0, 3.0, "SPEAKER_02"),
    )

    validate_diarization(turns, duration_s=3.0, speakers=SpeakerCount(exact=3))
    validate_diarization(turns, duration_s=3.0, speakers=SpeakerCount(minimum=3))
    with pytest.raises(ValueError, match=r"requested 2.*observed 3"):
        validate_diarization(turns, duration_s=3.0, speakers=SpeakerCount(exact=2))
    with pytest.raises(ValueError, match="maximum"):
        validate_diarization(turns, duration_s=3.0, speakers=SpeakerCount(maximum=2))
    with pytest.raises(ValueError, match=r"requested 2.*observed 0"):
        validate_diarization((), duration_s=3.0, speakers=SpeakerCount(exact=2))


def test_fluid_audio_diarizer_maps_segments_and_speaker_constraints(tmp_path: Path) -> None:
    calls: list[tuple[Sequence[str], float]] = []

    def command_runner(
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        del env
        calls.append((argv, timeout_s))
        output = Path(argv[2])
        output.write_text(
            json.dumps(
                {
                    "engine": "FluidAudio/CoreML",
                    "sourceRevision": FLUID_AUDIO_REVISION,
                    "speakerCount": 2,
                    "segments": [
                        {"speaker": "speaker-b", "startSeconds": 1.0, "endSeconds": 2.0},
                        {"speaker": "speaker-a", "startSeconds": 0.0, "endSeconds": 1.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    diarizer = FluidAudioDiarizer(
        duration_s=2.0,
        command_runner=command_runner,
        runner_provider=lambda: Path("/private/tmp/meeting-diarizer"),
    )

    turns = diarizer.diarize(tmp_path / "audio.wav", SpeakerCount(minimum=1, maximum=2))

    assert turns == (
        DiarizationTurn(0.0, 1.0, "SPEAKER_00"),
        DiarizationTurn(1.0, 2.0, "SPEAKER_01"),
    )
    assert calls[0][0][-4:] == ["--min-speakers", "1", "--max-speakers", "2"]
    assert calls[0][1] == 1800.0


def test_fluid_audio_diarizer_rejects_unpinned_output(tmp_path: Path) -> None:
    def command_runner(
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout_s
        Path(argv[2]).write_text(
            json.dumps(
                {
                    "engine": "FluidAudio/CoreML",
                    "sourceRevision": "unexpected",
                    "speakerCount": 0,
                    "segments": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    diarizer = FluidAudioDiarizer(
        duration_s=2.0,
        command_runner=command_runner,
        runner_provider=lambda: Path("/private/tmp/meeting-diarizer"),
    )

    with pytest.raises(SafeError) as raised:
        diarizer.diarize(tmp_path / "audio.wav", SpeakerCount())
    assert "revision" in raised.value.render(set())
