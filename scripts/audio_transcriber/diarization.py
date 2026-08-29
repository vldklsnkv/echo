"""Local speaker diarization adapters with secret-safe validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from .errors import SafeError, redact_text
from .interfaces import CommandRunner
from .models import DiarizationTurn, SpeakerCount, StageName

FLUID_AUDIO_REVISION = "6428e29186573c6d33c598e25d460e6690bc0ee1"
_FLUID_AUDIO_REPOSITORY = "https://github.com/FluidInference/FluidAudio.git"
_NEMO_ARTIFACT_URL = (
    "https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.0/"
    "NemoTextProcessing.xcframework.zip"
)
_NEMO_ARTIFACT_SHA256 = "76d0ee9a32b1ee2193231299180ca9bc4fc7e98794e771b3d55d66498352d85f"
_NEMO_ARTIFACT_SIZE = 49_419_751
_REMOTE_NEMO_TARGET = """        .binaryTarget(
            name: \"NemoTextProcessing\",
            url:
                \"https://github.com/FluidInference/text-processing-rs/releases/download/v0.3.0/NemoTextProcessing.xcframework.zip\",
            checksum: \"76d0ee9a32b1ee2193231299180ca9bc4fc7e98794e771b3d55d66498352d85f\"
        ),"""
_LOCAL_NEMO_TARGET = """        .binaryTarget(
            name: \"NemoTextProcessing\",
            path: \"Vendor/NemoTextProcessing.xcframework\"
        ),"""


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_setup_command(argv: Sequence[str], *, timeout_s: float) -> str:
    completed = subprocess.run(
        argv,
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "setup command failed"
        raise RuntimeError(detail)
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _prepare_fluid_audio_vendor(package_root: Path) -> None:
    vendor = package_root / "vendor" / "FluidAudio"
    marker = vendor / ".meeting-transcriber-vendor"
    expected_marker = f"{FLUID_AUDIO_REVISION}\n{_NEMO_ARTIFACT_SHA256}\n"
    if vendor.exists() and (
        vendor.is_symlink() or vendor.resolve().parent != vendor.parent.resolve()
    ):
        raise RuntimeError("refusing to use an unsafe FluidAudio vendor path")
    if marker.is_file() and marker.read_text(encoding="utf-8") == expected_marker:
        return
    git = shutil.which("git")
    curl = shutil.which("curl")
    if git is None or curl is None:
        raise RuntimeError("git and curl are required to prepare FluidAudio")
    parent = vendor.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fluid-vendor-", dir=parent) as raw:
        temporary = Path(raw) / "FluidAudio"
        _run_setup_command(
            [
                git,
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                _FLUID_AUDIO_REPOSITORY,
                str(temporary),
            ],
            timeout_s=300,
        )
        _run_setup_command(
            [git, "-C", str(temporary), "checkout", "--detach", FLUID_AUDIO_REVISION],
            timeout_s=120,
        )
        head = _run_setup_command([git, "-C", str(temporary), "rev-parse", "HEAD"], timeout_s=30)
        if head != FLUID_AUDIO_REVISION:
            raise RuntimeError("FluidAudio checkout revision mismatch")

        archive = Path(raw) / "NemoTextProcessing.xcframework.zip"
        _run_setup_command(
            [curl, "-L", "--fail", "--retry", "3", "-o", str(archive), _NEMO_ARTIFACT_URL],
            timeout_s=300,
        )
        if archive.stat().st_size != _NEMO_ARTIFACT_SIZE:
            raise RuntimeError("FluidAudio binary artifact size mismatch")
        if _sha256(archive) != _NEMO_ARTIFACT_SHA256:
            raise RuntimeError("FluidAudio binary artifact checksum mismatch")
        target_parent = temporary / "Vendor"
        target_parent.mkdir(mode=0o700)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("FluidAudio binary artifact contains an unsafe path")
            bundle.extractall(target_parent)
        framework = target_parent / "NemoTextProcessing.xcframework"
        if not (framework / "Info.plist").is_file():
            raise RuntimeError("FluidAudio binary artifact is incomplete")

        manifest = temporary / "Package.swift"
        text = manifest.read_text(encoding="utf-8")
        if _REMOTE_NEMO_TARGET not in text:
            raise RuntimeError("FluidAudio package manifest does not match the pinned revision")
        manifest.write_text(text.replace(_REMOTE_NEMO_TARGET, _LOCAL_NEMO_TARGET), encoding="utf-8")
        (temporary / ".meeting-transcriber-vendor").write_text(expected_marker, encoding="utf-8")
        if vendor.exists():
            if vendor.is_symlink() or vendor.resolve().parent != parent.resolve():
                raise RuntimeError("refusing to replace an unsafe FluidAudio vendor path")
            shutil.rmtree(vendor)
        os.replace(temporary, vendor)


def _ensure_fluid_audio_runner() -> Path:
    package_root = _plugin_root() / "swift"
    runner = package_root / ".build" / "release" / "meeting-diarizer"
    if runner.is_file() and os.access(runner, os.X_OK):
        return runner
    _prepare_fluid_audio_vendor(package_root)
    swift = shutil.which("swift")
    if swift is None:
        raise RuntimeError("Swift 6 and macOS 14 or newer are required for FluidAudio")
    completed = subprocess.run(
        [
            swift,
            "build",
            "--package-path",
            str(package_root),
            "-c",
            "release",
            "--product",
            "meeting-diarizer",
        ],
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0 or not runner.is_file() or not os.access(runner, os.X_OK):
        detail = completed.stderr.strip() or completed.stdout.strip() or "Swift build failed"
        raise RuntimeError(detail)
    return runner


RunnerProvider = Callable[[], Path]


def _validate_duration(duration_s: float) -> None:
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("diarization duration must be a finite non-negative number")


def _validate_label(label: str) -> None:
    if not label.strip():
        raise ValueError("diarization raw speaker labels must not be empty")
    if any(character.isspace() and character not in {" "} for character in label):
        raise ValueError("diarization raw speaker labels must not contain control characters")


def validate_diarization(
    turns: Sequence[DiarizationTurn], *, duration_s: float, speakers: SpeakerCount
) -> None:
    """Validate raw local labels without conflating them with identities."""
    _validate_duration(duration_s)
    previous_key: tuple[float, float, str] | None = None
    labels: set[str] = set()
    for turn in turns:
        if not math.isfinite(turn.start_s) or not math.isfinite(turn.end_s):
            raise ValueError("diarization timestamps must be finite")
        if turn.start_s < 0 or turn.end_s < turn.start_s or turn.end_s > duration_s:
            raise ValueError("diarization turn lies outside the normalized audio duration")
        _validate_label(turn.raw_speaker)
        key = (turn.start_s, turn.end_s, turn.raw_speaker)
        if previous_key is not None and key < previous_key:
            raise ValueError("diarization turns must be sorted by start, end, and raw label")
        previous_key = key
        labels.add(turn.raw_speaker)

    observed = len(labels)
    if speakers.exact is not None and observed != speakers.exact:
        raise ValueError(
            "diarization exact speaker count mismatch: "
            f"requested {speakers.exact}, observed {observed}; "
            "check the requested count or retry with automatic speaker inference"
        )
    if speakers.minimum is not None and observed < speakers.minimum:
        raise ValueError(
            "diarization minimum speaker count mismatch: "
            f"minimum {speakers.minimum}, observed {observed}"
        )
    if speakers.maximum is not None and observed > speakers.maximum:
        raise ValueError(
            "diarization maximum speaker count mismatch: "
            f"maximum {speakers.maximum}, observed {observed}"
        )


class FluidAudioDiarizer:
    """Native CoreML adapter for FluidAudio's offline Community-1 pipeline."""

    def __init__(
        self,
        *,
        duration_s: float,
        command_runner: CommandRunner,
        runner_provider: RunnerProvider = _ensure_fluid_audio_runner,
    ) -> None:
        _validate_duration(duration_s)
        self._duration_s = duration_s
        self._command_runner = command_runner
        self._runner_provider = runner_provider
        self._runner: Path | None = None

    def _get_runner(self) -> Path:
        if self._runner is None:
            try:
                self._runner = self._runner_provider()
            except Exception as exc:
                raise SafeError(
                    stage=StageName.DIARIZE,
                    message=redact_text(str(exc), set()),
                    backend="fluidaudio-coreml",
                    recovery_hint="Confirm macOS 14+, Swift 6, and local FluidAudio build access.",
                ) from None
        return self._runner

    def health_check(self) -> None:
        self._get_runner()

    def close(self) -> None:
        self._runner = None

    @staticmethod
    def _speaker_arguments(speakers: SpeakerCount) -> list[str]:
        if speakers.exact is not None:
            return ["--speakers", str(speakers.exact)]
        arguments: list[str] = []
        if speakers.minimum is not None:
            arguments.extend(("--min-speakers", str(speakers.minimum)))
        if speakers.maximum is not None:
            arguments.extend(("--max-speakers", str(speakers.maximum)))
        return arguments

    def diarize(self, audio_path: Path, speakers: SpeakerCount) -> tuple[DiarizationTurn, ...]:
        try:
            with tempfile.TemporaryDirectory(
                prefix=".fluid-diarization-", dir=audio_path.parent
            ) as raw:
                directory = Path(raw)
                os.chmod(directory, 0o700)
                output_path = directory / "result.json"
                command = [
                    str(self._get_runner()),
                    str(audio_path),
                    str(output_path),
                    *self._speaker_arguments(speakers),
                ]
                completed = self._command_runner(
                    command,
                    timeout_s=max(1800.0, self._duration_s * 0.5),
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    raise RuntimeError(detail or "FluidAudio diarization failed")
                decoded = cast(object, json.loads(output_path.read_text(encoding="utf-8")))
            if not isinstance(decoded, dict):
                raise ValueError("FluidAudio output metadata is invalid")
            payload = cast(dict[object, object], decoded)
            if payload.get("engine") != "FluidAudio/CoreML":
                raise ValueError("FluidAudio output metadata is invalid")
            if payload.get("sourceRevision") != FLUID_AUDIO_REVISION:
                raise ValueError("FluidAudio output revision is invalid")
            raw_segments_value = payload.get("segments")
            if not isinstance(raw_segments_value, list):
                raise ValueError("FluidAudio output segments are invalid")
            raw_segments = cast(list[object], raw_segments_value)
            parsed_segments: list[tuple[str, float, float]] = []
            for raw_segment in raw_segments:
                if not isinstance(raw_segment, dict):
                    raise ValueError("FluidAudio segment is invalid")
                segment = cast(dict[object, object], raw_segment)
                label = segment.get("speaker")
                start = segment.get("startSeconds")
                end = segment.get("endSeconds")
                if (
                    not isinstance(label, str)
                    or not isinstance(start, (int, float))
                    or not isinstance(end, (int, float))
                ):
                    raise ValueError("FluidAudio segment fields are invalid")
                parsed_segments.append((label, float(start), float(end)))
            labels = sorted({label for label, _, _ in parsed_segments})
            label_map = {label: f"SPEAKER_{index:02d}" for index, label in enumerate(labels)}
            turns: list[DiarizationTurn] = []
            for label, start, end in parsed_segments:
                turns.append(
                    DiarizationTurn(
                        start_s=start,
                        end_s=min(end, self._duration_s),
                        raw_speaker=label_map[label],
                    )
                )
            result = tuple(
                sorted(turns, key=lambda turn: (turn.start_s, turn.end_s, turn.raw_speaker))
            )
            validate_diarization(result, duration_s=self._duration_s, speakers=speakers)
            return result
        except SafeError:
            raise
        except Exception as exc:
            raise SafeError(
                stage=StageName.DIARIZE,
                message=redact_text(str(exc), set()),
                backend="fluidaudio-coreml",
                recovery_hint="Check the local CoreML models and requested speaker-count constraints.",
            ) from None
