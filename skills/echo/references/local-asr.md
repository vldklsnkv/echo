# Local ASR

## Engines, backends, and models

`--local-engine auto` selects GigaAM for explicit `--language ru` and Whisper otherwise. GigaAM defaults to `v3_e2e_rnnt`; the official GigaAM `0.2.0` source is pinned to commit `7447938d791c4f3e643386ee22c33777004293a5` because the older PyPI release lacks the required word timestamps. Explicit English uses the faster `distil-large-v3`; mixed speech and language auto-detection use `large-v3-turbo`. MLX aliases resolve to `mlx-community/distil-whisper-large-v3` and `mlx-community/whisper-large-v3-turbo` on Apple Silicon; CUDA/CPU use faster-whisper `>=1.2.1,<2`. Force an engine with `--local-engine gigaam` or `--local-engine whisper`; `--local-model` accepts that engine's model name or supported custom checkpoint path.

All adapters require word timestamps. Whisper passes `condition_on_previous_text=false`, making independently retried windows safe to merge and preventing prior chunk text from causing a continuation loop. GigaAM uses a maximum window of `24.0` seconds; Whisper and OpenAI use `600.0` seconds. The default overlap is `2.0` seconds. Language is auto-detected by Whisper when `--language` is absent; pass an ISO language code to pin it.

GigaAM first tries MPS. If one chunk fails, only that chunk is retried with Whisper and the next chunk returns to GigaAM/MPS; Echo never demotes the rest of the recording to CPU. Whisper CUDA uses `float16`; Whisper CPU uses `int8`. The ASR choice does not control the separate FluidAudio Core ML diarizer.

## Optional model smoke

Do not download a model merely to validate this skill. With an external, non-sensitive recording and an already-authorized local model, run the resolved launcher in an interactive terminal with `--mode local` and an exact `--speakers` count. Inspect canonical JSON provenance for the backend, model, device, and language before treating the smoke as successful.

Read the primary backend documentation when changing a model or its capabilities: [GigaAM](https://github.com/salute-developers/GigaAM), [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), and [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
