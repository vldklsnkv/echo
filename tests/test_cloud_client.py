from __future__ import annotations

import json
from pathlib import Path

import pytest

import transcribe_cloud as cloud


@pytest.mark.parametrize("private,name", [(False, cloud.REPOSITORY), (True, "another/repo")])
def test_upload_requires_exact_private_repository(
    monkeypatch: pytest.MonkeyPatch, private: bool, name: str
) -> None:
    def metadata(*args: str) -> str:
        return json.dumps({"private": private, "full_name": name})

    monkeypatch.setattr(cloud, "gh", metadata)
    with pytest.raises(RuntimeError, match="Refusing to upload"):
        cloud.assert_private()


def test_missing_upload_consent_prevents_any_github_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["cloud", str(tmp_path / "input.m4a"), "--output-dir", str(tmp_path)]
    )

    def refuse(*args: str) -> str:
        pytest.fail("GitHub must not be called")

    monkeypatch.setattr(cloud, "gh", refuse)
    with pytest.raises(SystemExit, match="2"):
        cloud.main()


@pytest.mark.parametrize("filename", ["source with spaces.m4a", 'question? <release>|*.m4a'])
def test_dispatch_resume_does_not_upload_or_dispatch_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filename: str
) -> None:
    source = tmp_path / filename
    source.write_bytes(b"immutable audio fixture")
    output = tmp_path / "output"
    calls: list[tuple[str, ...]] = []
    request_id = ""

    def fake_gh(*args: str) -> str:
        nonlocal request_id
        calls.append(args)
        if args == ("api", f"repos/{cloud.REPOSITORY}"):
            return json.dumps({"private": True, "full_name": cloud.REPOSITORY})
        if args[0] == "api":
            return "[]"
        if args[:2] == ("release", "create"):
            snapshot = Path(args[3])
            assert snapshot.name == "recording.m4a"
            assert snapshot.read_bytes() == source.read_bytes()
            return ""
        if args[:2] == ("workflow", "run"):
            request_id = next(arg.split("=", 1)[1] for arg in args if arg.startswith("request_id="))
            return ""
        if args[:2] == ("run", "list"):
            return (
                json.dumps([{"databaseId": 123, "displayTitle": f"echo-{request_id}"}])
                if request_id
                else "[]"
            )
        pytest.fail(f"Unexpected call: {args}")

    monkeypatch.setattr(cloud, "gh", fake_gh)
    monkeypatch.setattr(
        "sys.argv",
        ["cloud", str(source), "--output-dir", str(output), "--allow-cloud-upload", "--no-wait"],
    )
    assert cloud.main() == 0
    first_request_id = request_id
    assert cloud.main() == 0
    assert request_id == first_request_id
    assert sum(call[:2] == ("release", "create") for call in calls) == 1
    assert sum(call[:2] == ("workflow", "run") for call in calls) == 1
    assert source.read_bytes() == b"immutable audio fixture"


def test_failed_job_cannot_publish_or_delete_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_gh(*args: str) -> str:
        calls.append(args)
        return json.dumps({"status": "completed", "conclusion": "failure"})

    monkeypatch.setattr(cloud, "gh", fake_gh)
    with pytest.raises(RuntimeError, match="not successful"):
        cloud.collect("123", tmp_path, {}, tmp_path / "journal.json")
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_changed_local_result_is_not_treated_as_verified(tmp_path: Path) -> None:
    result = tmp_path / "recording.transcript.txt"
    result.write_text("original")
    journal = {"files": {result.name: cloud.digest(result)}}
    assert cloud.completed_outputs(journal, tmp_path)
    result.write_text("modified")
    assert not cloud.completed_outputs(journal, tmp_path)
    assert not cloud.completed_outputs({"files": {"../outside": "abc"}}, tmp_path)


def test_delivery_cleanup_preserves_other_files_and_records_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcript = tmp_path / "recording.transcript.txt"
    transcript.write_text("spoken text")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")
    journal = {
        "release_tag": "input-test",
        "run_id": "123",
        "files": {transcript.name: cloud.digest(transcript)},
    }

    def fake_gh(*args: str) -> str:
        return "[]" if args[:2] == ("release", "list") else '{"artifacts": []}'

    monkeypatch.setattr(cloud, "gh", fake_gh)
    cloud.confirm_delivery(
        journal, tmp_path / "journal.json", tmp_path, "https://chatgpt.com/space/page_" + "a" * 32
    )
    assert journal["delivered"] is True
    assert not transcript.exists()
    assert unrelated.read_text() == "keep"


def test_delivery_cleanup_refuses_modified_staging_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcript = tmp_path / "recording.transcript.txt"
    transcript.write_text("original")
    journal = {"files": {transcript.name: cloud.digest(transcript)}}
    transcript.write_text("user edited")

    def refuse(*args: str) -> str:
        pytest.fail("Cloud cleanup must not begin")

    monkeypatch.setattr(cloud, "gh", refuse)
    with pytest.raises(RuntimeError, match="still match"):
        cloud.confirm_delivery(
            journal,
            tmp_path / "journal.json",
            tmp_path,
            "https://chatgpt.com/space/page_" + "a" * 32,
        )
    assert transcript.read_text() == "user edited"
