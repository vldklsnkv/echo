# Echo

Echo is a local-first Codex plugin for transcribing meetings, interviews, calls, voice memos, and other long recordings. It combines language-aware speech recognition with local speaker diarization and produces structured, resumable output with word timestamps.

Russian transcription uses GigaAM v3 by default. Explicit English uses Distil-Whisper, while mixed speech and automatic language detection use Whisper Large v3 Turbo. Speaker diarization runs locally through FluidAudio's Core ML pipeline.

## Highlights

- Local transcription for Russian, English, and mixed-language recordings.
- Automatic, ranged, or exact speaker-count policies.
- Word-level timestamps and aligned speaker labels.
- Optional human-readable speaker names applied after diarization.
- Resumable chunk processing for long recordings.
- Quality gates for missing coverage, repetition loops, and suspicious ASR output.
- Canonical JSON plus derived TXT, SRT, and VTT files.
- Optional OpenAI ASR behind an explicit upload-consent boundary.

## Requirements

Echo currently targets Apple Silicon Macs running macOS 14 or newer. The local runtime requires:

- Python 3.13;
- `uv`;
- `ffmpeg` and `ffprobe`;
- Git and curl;
- Swift 6 for the local FluidAudio build.

The first local run downloads speech models and builds the checksum-pinned diarization runtime. Those files are kept in local caches and are reused by later runs. No Hugging Face token is required.

## Using Echo

In Codex, invoke the plugin with a recording attached or available on disk:

```text
Use $echo:echo to transcribe this recording locally with speaker labels.
```

The command-line launcher is also available from the repository:

```sh
./scripts/transcribe_audio.py --help

./scripts/transcribe_audio.py /absolute/path/meeting.m4a \
  --mode local \
  --language ru \
  --speakers 2 \
  --speaker-names "SPEAKER_00=Alex,SPEAKER_01=Sam" \
  --output-dir /absolute/path/output \
  --resume --srt --vtt
```

Omit speaker flags for automatic diarization, use `--speakers N` for an exact count, or combine `--min-speakers` and `--max-speakers` for a range. `--resume` validates completed stages and continues from the first unfinished chunk.

## Output

The validated `.transcript.json` file is the source of truth. It records input identity, ASR and diarization provenance, speaker labels, turns, words, timestamps, retries, and quality results. Echo always derives a readable TXT transcript and can also render SRT and VTT subtitles.

Generic filenames such as `Recording.m4a` or `Voice Memo.m4a` are replaced with a short title derived locally from the transcript. Meaningful source names are preserved.

## Privacy and cloud mode

Local mode keeps recording contents on the machine. Audio, transcripts, checkpoints, model caches, and runtime state must not be committed to the repository.

OpenAI mode is optional and never inferred from the request. Before any upload, Echo explains that audio chunks will leave the machine and requires explicit consent. Non-interactive cloud runs additionally require `--allow-cloud-upload`.

Raw diarization labels identify speaker segments, not people. Display names are applied only when the user supplies an explicit mapping.

## Reliability and performance

Echo checkpoints every completed ASR chunk. An interrupted long meeting can resume without repeating finished work, and ASR memory is released before the Core ML diarizer is loaded.

On an M3 Air with warm caches, the regression targets for a 90-minute recording are up to 7 minutes for Russian and up to 10 minutes for English. They are engineering targets rather than guarantees; cold downloads, model availability, recording quality, and different hardware affect runtime.

If one Russian chunk fails in GigaAM, Echo retries that chunk with Whisper without forcing the rest of the recording onto the fallback backend.

## Development

Install and run the frozen test environment with `uv`:

```sh
uv run --python 3.13 --frozen pytest
```

The project is released under the MIT License. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for licensing and third-party attribution.
