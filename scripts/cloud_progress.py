"""Keep private pipeline logs while publishing only numeric ASR progress."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_PROGRESS = re.compile(rb"asr \| [0-9]+/[0-9]+ \| (?:backend=[a-zA-Z0-9_-]+|checkpoint=reused)")


def main() -> None:
    path = Path(sys.argv[1])
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as log:
        for line in sys.stdin.buffer:
            log.write(line)
            log.flush()
            if _PROGRESS.fullmatch(line.rstrip(b"\r\n")):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
