# Private Actions pilot

The public `vldklsnkv/echo` repository is the plugin source. The pilot workflow
refuses to process recordings in public repositories. Run an exact copy of the
pilot commit in a separate private `echo-cloud` repository, with Actions enabled
and no untrusted contributors. Do not change the public repository's visibility.

The pilot keeps the existing GigaAM/Whisper and FluidAudio pipeline. `--mode local`
means inference on the runner, without sending audio to an ASR API. Compatibility,
runtime, available memory and quality on the hosted macOS VM are still unverified.

## One-recording acceptance procedure

1. Obtain the recording attached to the chat and compute its SHA-256. Upload only
   that recording as an asset of a private release in `echo-cloud`. Use a unique
   release tag and a simple ASCII asset filename; preserve the original elsewhere.
   Never place a recording in Git, a public release or workflow inputs.
2. Dispatch `cloud-transcribe.yml` from the exact pilot revision in the private
   repository, specifying the private release tag, asset filename, SHA-256 and
   language. The GITHUB_TOKEN is read-only and scoped to that repository.
3. Check that ASR, speaker diarization and quality gates actually complete.
   Download the private result artifact; check JSON provenance, time coverage,
   speaker turns and transcript text against representative parts of the audio.
   Record workflow URL, source commit, duration, runner time and observed issues.
   Passing unit tests or workflow dispatch alone is not acceptance.
4. Save the verified results in the user's Digital task folder, then delete the
   private input release and result artifact after successful retrieval. Failed
   runs retain the input release for a deliberate retry. Runner files are cleaned
   even on normal job failure; failure logs are returned only as a private artifact
   with one-day retention, never printed in the workflow log. Results expire
   after one day if not explicitly deleted sooner.
5. Only after acceptance, merge the reviewed changes into public `main`, update
   the private execution revision, and switch the installed Echo skill to cloud
   dispatch. The current skill and default local behavior remain intact meanwhile.

## Boundaries of this first pilot

One job runs at a time, with a two-hour ceiling. No schedules, model caches,
automatic uploads, speaker identity mappings or paid larger runners are configured.
Cross-run checkpoint recovery and hash-based duplicate prevention must be added
before switching new calls to cloud by default. `--resume` currently resumes only
within the same runner workspace. Interrupted jobs start over on a fresh runner.

Chat attachments still need an agent-accessible file or download URL. A GitHub
workflow cannot independently read this chat. The installed plugin will eventually
upload the attachment and dispatch the private job; processing must not silently
fall back to the user's Mac when the cloud job fails.
