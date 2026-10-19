#!/usr/bin/env python3
"""Dispatch or collect private Echo jobs without loading local speech models."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPOSITORY = "vldklsnkv/echo-cloud"
WORKFLOW = "cloud-transcribe.yml"


def gh(*arguments: str) -> str:
    result = subprocess.run(
        ["gh", *arguments], capture_output=True, text=True, timeout=600, check=False
    )
    if result.returncode:
        # Provider errors may include private filenames or diagnostics.
        raise RuntimeError(f"GitHub command failed ({arguments[0]}); inspect the private job.")
    return result.stdout


def assert_private() -> None:
    metadata = json.loads(gh("api", f"repos/{REPOSITORY}"))
    if metadata.get("private") is not True or metadata.get("full_name") != REPOSITORY:
        raise RuntimeError("Refusing to upload: the expected repository is not private.")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_journal(path: Path, payload: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".echo-journal-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def completed_outputs(journal: dict[str, Any], output: Path) -> bool:
    files = journal.get("files", {})
    return bool(files) and all(
        Path(name).name == name and (output / name).is_file() and digest(output / name) == expected
        for name, expected in files.items()
    )


def cleanup_input(journal: dict[str, Any], journal_path: Path) -> None:
    releases = json.loads(
        gh("release", "list", "--repo", REPOSITORY, "--limit", "1000", "--json", "tagName")
    )
    if any(release["tagName"] == journal["release_tag"] for release in releases):
        gh(
            "release",
            "delete",
            journal["release_tag"],
            "--repo",
            REPOSITORY,
            "--yes",
            "--cleanup-tag",
        )
    journal["input_deleted"] = True
    save_journal(journal_path, journal)


def collect(run_id: str, output: Path, journal: dict[str, Any], journal_path: Path) -> None:
    run = json.loads(gh("run", "view", run_id, "--repo", REPOSITORY, "--json", "conclusion,status"))
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise RuntimeError("The cloud run is not successful; no transcript has been published.")
    with tempfile.TemporaryDirectory(prefix=".echo-download-", dir=output) as raw:
        staging = Path(raw)
        gh(
            "run",
            "download",
            run_id,
            "--repo",
            REPOSITORY,
            "--name",
            f"echo-transcript-{run_id}",
            "--dir",
            str(staging),
        )
        files = sorted(staging.iterdir())
        if len(files) != 4 or any(not item.is_file() or item.is_symlink() for item in files):
            raise RuntimeError("Unexpected transcript artifact structure.")
        json_files = list(staging.glob("*.transcript.json"))
        if len(json_files) != 1:
            raise RuntimeError("Missing canonical transcript.")
        # This imports validators only; it never provisions or loads speech models.
        from audio_transcriber.renderers import load_json, validate_transcript

        document = load_json(json_files[0])
        validate_transcript(document)
        suffixes = {".json", ".txt", ".srt", ".vtt"}
        stem = json_files[0].name.removesuffix(".json")
        if {item.name for item in files} != {stem + suffix for suffix in suffixes}:
            raise RuntimeError("Transcript formats do not match the canonical output.")
        for item in files:
            destination = output / item.name
            if destination.exists() and digest(destination) != digest(item):
                raise RuntimeError("Refusing to replace an existing, different transcript.")
        for item in files:
            destination = output / item.name
            if not destination.exists():
                shutil.copyfile(item, destination)
                destination.chmod(0o600)
            if digest(destination) != digest(item):
                raise RuntimeError("Downloaded transcript verification failed.")
        journal["files"] = {item.name: digest(item) for item in files}
        journal["quality_warnings"] = [warning.to_dict() for warning in document.quality.warnings]
        save_journal(journal_path, journal)
    # Cloud cleanup waits for an independently verified Spaces delivery.


def confirm_delivery(
    journal: dict[str, Any], journal_path: Path, output: Path, page_url: str
) -> None:
    if not re.fullmatch(r"https://chatgpt\.com/space/page_[a-f0-9]{32}", page_url):
        raise RuntimeError("Expected a canonical Context Page URL after verified publication.")
    if not journal.get("page_url") and not completed_outputs(journal, output):
        raise RuntimeError(
            "Local staging files must still match before confirming Spaces delivery."
        )
    for name, checksum in journal["files"].items():
        path = output / name
        if path.exists() and digest(path) != checksum:
            raise RuntimeError("Refusing to remove a changed staging file.")
    journal["page_url"] = page_url
    save_journal(journal_path, journal)
    cleanup_input(journal, journal_path)
    run_id = journal["run_id"]
    artifacts = json.loads(gh("api", f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts"))
    for artifact in artifacts.get("artifacts", []):
        if artifact["name"] == f"echo-transcript-{run_id}":
            gh(
                "api",
                "--method",
                "DELETE",
                f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}",
            )
    for name in journal["files"]:
        (output / name).unlink(missing_ok=True)
    journal["delivered"] = True
    save_journal(journal_path, journal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", choices=("ru", "en", "auto"), default="auto")
    parser.add_argument("--local-engine", choices=("auto", "gigaam", "whisper"), default="auto")
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument("--speakers", type=int)
    counts.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--speaker-names", default="")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--allow-cloud-upload", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--confirm-delivery", metavar="CONTEXT_PAGE_URL")
    parser.add_argument("--run-id", help="Recover a verified existing dispatch for this request")
    args = parser.parse_args()
    from audio_transcriber.models import SpeakerCount

    counts = SpeakerCount(exact=args.speakers, minimum=args.min_speakers, maximum=args.max_speakers)
    args.speaker_policy = "auto"
    if counts.exact is not None:
        args.speaker_policy = str(counts.exact)
    elif counts.minimum is not None or counts.maximum is not None:
        args.speaker_policy = f"{counts.minimum or ''}-{counts.maximum or ''}"
    if not args.allow_cloud_upload:
        parser.error(
            "--allow-cloud-upload is required: audio is uploaded to private GitHub Actions"
        )
    source = args.input_path.resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {
        ".m4a",
        ".wav",
        ".mp3",
        ".aac",
        ".flac",
        ".ogg",
    }:
        parser.error("input must be a supported audio file")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    checksum = digest(source)
    settings = json.dumps(
        [
            checksum,
            args.language,
            args.local_engine,
            args.ref,
            args.speaker_policy,
            args.speaker_names,
        ]
    )
    key = hashlib.sha256(settings.encode()).hexdigest()[:24]
    descriptor = os.open(output / ".echo-cloud.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run_request(args, source, output, checksum, key)


def run_request(
    args: argparse.Namespace, source: Path, output: Path, checksum: str, key: str
) -> int:
    journal_path = output / f".echo-cloud-{key}.json"
    assert_private()
    journal = (
        json.loads(journal_path.read_text())
        if journal_path.exists()
        else {
            "sha256": checksum,
            "repository": REPOSITORY,
            "release_tag": f"input-{key}",
        }
    )
    if journal.get("sha256") != checksum or journal.get("repository") != REPOSITORY:
        raise RuntimeError("Cloud journal does not match the recording.")
    if journal.get("release_tag") != f"input-{key}":
        raise RuntimeError("Cloud journal contains an unexpected release tag.")
    if journal.get("delivered"):
        print(journal["page_url"])
        return 0
    if journal.get("page_url"):
        confirm_delivery(journal, journal_path, output, journal["page_url"])
        print(journal["page_url"])
        return 0
    if args.run_id:
        if not re.fullmatch(r"[0-9]+", args.run_id):
            raise RuntimeError("Invalid recovery run ID.")
        recovery = json.loads(
            gh(
                "run",
                "view",
                args.run_id,
                "--repo",
                REPOSITORY,
                "--json",
                "displayTitle,headBranch",
            )
        )
        if recovery["displayTitle"] != f"echo-{key}" or recovery["headBranch"] != args.ref:
            raise RuntimeError("Recovery run does not match this recording and settings.")
        journal["run_id"] = args.run_id
        save_journal(journal_path, journal)
    if completed_outputs(journal, output):
        if args.confirm_delivery or journal.get("page_url"):
            confirm_delivery(
                journal, journal_path, output, args.confirm_delivery or journal["page_url"]
            )
            print(journal["page_url"])
        else:
            print("Verified staging files ready for Spaces; no new upload or inference.")
        return 0
    if not journal.get("run_id"):
        # Upload an immutable snapshot; the original is never renamed or modified.
        with tempfile.TemporaryDirectory(prefix="echo-upload-") as raw:
            snapshot = Path(raw) / ("recording" + source.suffix.lower())
            shutil.copyfile(source, snapshot)
            snapshot.chmod(0o600)
            if digest(snapshot) != checksum:
                raise RuntimeError("Recording changed while preparing the upload.")
            tag = journal["release_tag"]
            existing = json.loads(gh("api", f"repos/{REPOSITORY}/releases?per_page=100"))
            release = next((item for item in existing if item["tag_name"] == tag), None)
            if release is None:
                gh(
                    "release",
                    "create",
                    tag,
                    str(snapshot),
                    "--repo",
                    REPOSITORY,
                    "--target",
                    args.ref,
                    "--title",
                    "Temporary Echo input",
                    "--notes",
                    "Private transcription input.",
                    "--prerelease",
                )
            elif not any(asset["name"] == snapshot.name for asset in release["assets"]):
                raise RuntimeError(
                    "Existing private input release is incomplete; inspect before retrying."
                )
        save_journal(journal_path, journal)
        # The workflow title is content-addressed, so interrupted dispatch can be recovered.
        title = f"echo-{key}"
        runs = json.loads(
            gh(
                "run",
                "list",
                "--repo",
                REPOSITORY,
                "--workflow",
                WORKFLOW,
                "--limit",
                "100",
                "--json",
                "databaseId,displayTitle",
            )
        )
        matching = [run for run in runs if run["displayTitle"] == title]
        if not matching:
            gh(
                "workflow",
                "run",
                WORKFLOW,
                "--repo",
                REPOSITORY,
                "--ref",
                args.ref,
                "-f",
                f"release_tag={journal['release_tag']}",
                "-f",
                f"asset_name={snapshot.name}",
                "-f",
                f"sha256={checksum}",
                "-f",
                f"language={args.language}",
                "-f",
                f"local_engine={args.local_engine}",
                "-f",
                f"request_id={key}",
                "-f",
                f"speaker_policy={args.speaker_policy}",
                "-f",
                f"speaker_names={args.speaker_names}",
            )
            for _ in range(12):
                runs = json.loads(
                    gh(
                        "run",
                        "list",
                        "--repo",
                        REPOSITORY,
                        "--workflow",
                        WORKFLOW,
                        "--limit",
                        "100",
                        "--json",
                        "databaseId,displayTitle",
                    )
                )
                matching = [run for run in runs if run["displayTitle"] == title]
                if matching:
                    break
                time.sleep(5)
        if not matching:
            raise RuntimeError("Dispatch outcome unknown; retry with the same output directory.")
        journal["run_id"] = str(matching[0]["databaseId"])
        save_journal(journal_path, journal)
    run_id = str(journal["run_id"])
    if not re.fullmatch(r"[0-9]+", run_id):
        raise RuntimeError("Invalid run ID in cloud journal.")
    print(f"https://github.com/{REPOSITORY}/actions/runs/{run_id}", flush=True)
    if args.no_wait:
        return 0
    for _ in range(300):
        run = json.loads(
            gh("run", "view", run_id, "--repo", REPOSITORY, "--json", "status,conclusion")
        )
        if run["status"] == "completed":
            collect(run_id, output, journal, journal_path)
            print(f"Verified temporary files ready for publication in Spaces: {output}")
            return 0
        print("Cloud job is still running.", flush=True)
        time.sleep(30)
    raise RuntimeError(
        "Cloud job still running; retry with the same output directory to collect it."
    )


if __name__ == "__main__":
    try:
        if sys.version_info < (3, 11):
            # Request only a Python interpreter, never the project's model dependencies.
            os.execvp(
                "uv", ["uv", "run", "--no-project", "--python", "3.13", __file__, *sys.argv[1:]]
            )
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Echo cloud: {error}", file=sys.stderr)
        raise SystemExit(1)
