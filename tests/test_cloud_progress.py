from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path


def test_progress_filter_retains_private_log_without_printing_speech(tmp_path: Path) -> None:
    log = tmp_path / "pipeline.log"
    script = Path(__file__).resolve().parents[1] / "scripts" / "cloud_progress.py"
    payload = (
        b"private speech and diagnostics\n"
        b"asr | 1/100 | backend=gigaam-cpu\n"
        b"asr | 2/100 | backend=private speech\n"
        b"asr | 3/100 | checkpoint=reused\n"
        b"prefix asr | 4/100 | backend=gigaam-cpu\n"
    )
    result = subprocess.run(
        [sys.executable, str(script), str(log)], input=payload, capture_output=True, check=True
    )
    assert result.stdout == (
        b"asr | 1/100 | backend=gigaam-cpu\nasr | 3/100 | checkpoint=reused\n"
    )
    assert log.read_bytes() == payload
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_progress_filter_refuses_to_replace_existing_log(tmp_path: Path) -> None:
    log = tmp_path / "pipeline.log"
    log.write_text("original")
    script = Path(__file__).resolve().parents[1] / "scripts" / "cloud_progress.py"
    result = subprocess.run(
        [sys.executable, str(script), str(log)], input=b"replacement", capture_output=True
    )
    assert result.returncode != 0
    assert log.read_text() == "original"
