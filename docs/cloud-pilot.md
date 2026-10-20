# Private cloud execution

The public `vldklsnkv/echo` repository contains the plugin source. Inference runs
from the same source revision in private `vldklsnkv/echo-cloud`. The workflow
refuses to process recordings in a public repository. Audio is never committed
to Git, and the original recording is preserved.

## Delivery from a chat

1. The chat agent obtains the attached recording and existing upload consent.
   `scripts/transcribe_cloud.py` checks repository privacy, computes SHA-256,
   uploads a private snapshot and dispatches `cloud-transcribe.yml`.
2. GigaAM or Whisper and FluidAudio run on a fresh macOS ARM runner. The computer
   only transfers files and validates results. No ASR API is used in this path.
3. The client retrieves temporary pipeline files. The chat agent writes an editable summary and
   complete transcript in Spaces Context under Calls. No file attachments or
   technical reports are added to the Page. A detailed, grounded call summary appears at the top,
   followed by the complete transcript grouped by topic. For substantive hour-long
   calls, the summary is typically 400-600 words with descriptive subsections;
   shorter calls scale to their content. Participant headings sit above each turn.
4. After complete Page readback, the agent invokes the client again with
   `--confirm-delivery` and the canonical Page URL. This removes temporary cloud
   input, result artifact and local staging transcripts. The recovery journal
   contains only the destination URL, checksums and job metadata.

Pages authorization stays in the chat agent; never put ChatGPT credentials on
Actions. If publication fails, report it and retain temporary recovery material.
Do not silently switch to permanent local storage or computer inference.

## Page delivery contract

The chat agent must read the live Context root, Calls and selected target native Agent Instructions before each Page write. See the Page write gate in `skills/echo/SKILL.md`. Only a permitted real new call creates one child under the existing Calls parent; retries/edits resume the canonical Page by recording/request identity and prior receipt. Incomplete reads, diagnostic events and pipeline failures never create replacement destinations. Fresh guarded writes and full transcript readback precede final delivery confirmation and cleanup.

The runner has no Pages credentials and cannot enforce parentage or verify that the agent read these instructions. A green run confirms processing/artifacts only. This source protocol does not prove adoption by an installed plugin or hosted task.

## Recovery and limits

Use the same staging directory and arguments to resume waiting or collect a
completed run. A journal and content-addressed workflow title prevent repeated
dispatch on ordinary retries. Recover an inspected dispatch with `--run-id`;
its title and branch must match the request. The agent also checks Context for
an existing transcript before creating a Page.

Failed jobs require inspection and deliberate retry. ASR checkpoints are not
restored onto fresh runners; failed runner inference starts over. Each job has
a two-hour ceiling. Distinct requests use distinct concurrency groups. No paid
larger runner, schedules, model caches or automatic recording intake are enabled.
Artifacts and private failure diagnostics expire after one day. Failed inputs
remain private until an inspected retry or explicit cleanup.

The client supports language and engine selection, automatic/exact/ranged speaker
counts and supplied display names. Raw labels do not identify people. Automated
quality checks do not establish word-for-word accuracy.

Transcript-derived output titles replace GitHub artifact-forbidden filename characters with spaces before publication. This changes only the output basename: original audio, transcript text, request/job identity and resume state remain intact.
