# Echo

A local-first Codex plugin for long meeting recordings. Russian (`--language ru`)
uses GigaAM v3 by default; explicit English uses Distil-Whisper, while mixed
speech and language auto-detection use Whisper Large v3 Turbo. Fast speaker
diarization uses FluidAudio's native Core ML pipeline.
Resumable stages produce canonical JSON, TXT, SRT, and VTT.

The plugin is named `echo`: it keeps the full local-first transcription
architecture while presenting a short product name in Codex.

## Privacy

Local mode keeps recording contents on the machine. Model files and the pinned
FluidAudio runtime are downloaded once into local caches. No Hugging Face token
is required. OpenAI mode remains available only with explicit upload consent.

Long recordings are checkpointed after every ASR chunk. An interrupted 90-minute
meeting resumes from the first unfinished chunk, and ASR memory is released before
the Core ML speaker-diarization model is loaded.

On an M3 Air with warm model caches, the performance targets for a 90-minute
recording are at most 7 minutes for Russian and at most 10 minutes for English.
They are regression targets, not guarantees for cold downloads or other hardware.
A failed Russian chunk falls back to Whisper by itself; later chunks return to
GigaAM/MPS and never inherit a whole-recording CPU fallback.

## Attribution

See `NOTICE.md` and `LICENSE`.
