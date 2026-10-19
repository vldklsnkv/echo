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
3. The client retrieves temporary pipeline files. The chat agent writes only
   editable transcript text in Spaces Context under Calls. No file attachments,
   technical reports are added to the Page. A concise, grounded call summary precedes
   the complete transcript, grouped by topic with readable participant labels.
4. After complete Page readback, the agent invokes the client again with
   `--confirm-delivery` and the canonical Page URL. This removes temporary cloud
   input, result artifact and local staging transcripts. The recovery journal
   contains only the destination URL, checksums and job metadata.

Pages authorization stays in the chat agent; never put ChatGPT credentials on
Actions. If publication fails, report it and retain temporary recovery material.
Do not silently switch to permanent local storage or computer inference.

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
