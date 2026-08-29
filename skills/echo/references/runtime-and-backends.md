# Runtime and backends

## Supported host and prerequisites

Version 2 supports Apple Silicon Macs running macOS 14 or newer. Russian local transcription uses official GigaAM v3 by default; English uses Distil-Whisper, while mixed speech and automatic language detection use Whisper Large v3 Turbo. `--local-engine` can force either backend, and `--local-model` accepts a supported model name or local checkpoint. GigaAM tries MPS first and retries only a failed chunk with Whisper instead of demoting the remaining recording to CPU. FluidAudio runs diarization through Core ML.

Require Python 3.13, `uv`, `ffmpeg`, `ffprobe`, Git, curl, Swift 6, and macOS 14 or newer. FluidAudio source and its checksum-pinned binary dependency are downloaded and built locally on first use; no Hugging Face token is required. OpenAI mode still uses the same local diarizer. Validate these requirements before normalization, ASR, or any upload.

## Reusable runtime and recovery

The launcher owns one plugin-local `.venv`. Its base fingerprint hashes runtime version `2`, the lockfile, operating-system version, architecture, and Python version. The runtime fingerprint adds the chosen ASR backend and compute capability. A verified cached fingerprint skips a full probe and sync; `--reprobe` refreshes the hardware decision without creating a second virtual environment.

Mutable coordination state is private to the resolved operating-system temporary directory under `meeting-transcriber-<uid>/`: the root, runtime children, locks, and run state are user-owned and mode `0700`. State schema version `1` and the runtime index identify the reusable cached runtime. Do not edit these files, follow symlinks, or move audio into this directory.

If a local ASR backend fails before audio work, it is recorded as unhealthy, excluded on the next probe, and the launcher re-executes once. FluidAudio build or model failures stop with a local recovery hint. Use `--reprobe` only after fixing the underlying host/model issue; it clears recorded ASR exclusions before re-probing.

ASR and diarization models are stage-scoped: ASR is closed and accelerator caches are cleared before FluidAudio is loaded. Completed ASR chunks are written atomically to private run-state checkpoints and reused after interruption. Checkpoints are removed only after the complete ASR stage is committed.

## Caches and safe recovery

Keep downloaded models in their providers' persistent user caches, never in `/tmp` or the repository. A safe first recovery is to check `uv`, `ffmpeg`, `ffprobe`, model access, and the source file, then run the resolved launcher with `--reprobe`. Do not delete the shared runtime directory or the plugin `.venv` while a run is active.

The runtime has an ordinary cheap index, a base fingerprint, and a backend/device runtime fingerprint. This preserves a stable cache while correctly rebuilding after a lockfile, host, Python, or capability change.
