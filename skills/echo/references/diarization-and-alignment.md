# Diarization and alignment

## Local FluidAudio contract

The local diarizer uses `FluidAudio/Community-1-CoreML` from FluidAudio revision `6428e29186573c6d33c598e25d460e6690bc0ee1`. It runs through Core ML, needs no account or token, and never receives an OpenAI credential. The first run downloads the pinned source and a checksum-verified binary dependency, builds the Swift runner locally, and then caches both the runner and diarization models.

`--speakers N` requests an exact count. A range forwards either or both `--min-speakers` and `--max-speakers`; no speaker flags requests automatic inference. Exact mode validates that the resulting distinct raw labels match the requested count.

Raw labels such as `SPEAKER_00` are opaque output labels. `--speaker-names` maps a returned raw label to a unique display name after diarization; it cannot create labels, repeat raw labels, or assign one display name to multiple raw labels. Do not create biometric speaker profiles or infer real-world identities.

## Word alignment

Each word is assigned to the overlapping diarization turn with the largest overlap. Ties resolve by earlier turn start then raw label. A word with no overlap uses the nearest turn, again resolving ties by start then raw label. Adjacent words become one transcript turn only when raw and display speaker labels agree and their gap is at most `1.5` seconds.

The canonical output retains both `raw_speaker` and `speaker`; renaming changes only the latter. This preserves reproducible alignment while presenting user-supplied display names in TXT, SRT, and VTT.
