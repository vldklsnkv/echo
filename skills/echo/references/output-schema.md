# Output schema

Schema version `1` canonical JSON is the source of truth. It is UTF-8, validated after writing, and contains input identity, ASR/diarization provenance, stage/retry provenance, raw and display speaker labels, word timestamps, and the quality report. Confidence may be `null`.

```json
{
  "schema_version": 1,
  "input": {"canonical_path": "/private/recording.wav", "size": 1234, "mtime_ns": 1, "sample_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "duration_s": 1.2},
  "language": "en",
  "asr": {"mode": "local", "backend": "cpu", "model": "large-v3-turbo", "chunking": {"max_duration_s": 600.0, "overlap_s": 2.0}, "detected_language": "en"},
  "diarization": {"model": "FluidAudio/Community-1-CoreML", "revision": "6428e29186573c6d33c598e25d460e6690bc0ee1", "compute_device": "coreml", "speaker_count_parameters": {"exact": 1, "minimum": null, "maximum": null}},
  "provenance": {"runtime_version": "2", "run_fingerprint": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "completed_stages": ["configure", "inspect", "asr", "quality", "diarize", "align", "render"], "retries": []},
  "speakers": [{"raw_label": "SPEAKER_00", "display_name": "SPEAKER_00"}],
  "turns": [{"start_s": 0.0, "end_s": 0.4, "raw_speaker": "SPEAKER_00", "speaker": "SPEAKER_00", "text": "Hello.", "words": [{"word": {"start_s": 0.0, "end_s": 0.4, "text": "Hello.", "confidence": null, "source_chunk": "chunk-0000"}, "raw_speaker": "SPEAKER_00", "speaker": "SPEAKER_00"}]}],
  "quality": {"policy_version": "2", "status": "passed", "checks": [{"code": "coverage", "status": "passed", "detail": "passed"}], "warnings": [], "unresolved_errors": []}
}
```

For meaningful source filenames, the output stem stays unchanged. Generic recorder names such as `Recording`, `Recording-fixed`, `Voice Memo`, or `Запись` are automatically replaced after transcription with a short title derived locally from the first meaningful phrase. For example, `Recording.m4a` containing «План релиза в пятницу» produces `План релиза в пятницу.transcript.json` and matching TXT/SRT/VTT names. Every selected stem uses the suffixes `.transcript.json`, `.transcript.txt`, `.transcript.srt`, and `.transcript.vtt`. No transcript text leaves the machine for naming. The JSON and TXT files are always emitted; SRT and VTT need `--srt` and `--vtt`.

TXT uses one turn per line: `[00:00:00–00:00:01] SPEAKER_00: Hello.`. SRT uses one-based numbered cues and `HH:MM:SS,mmm` timestamps. VTT starts with `WEBVTT`, omits cue numbers, and uses `HH:MM:SS.mmm`. All derived formats use the display speaker and deterministic turn text; no renderer reads a provider-native response.
