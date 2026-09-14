#!/usr/bin/env python
"""Entry point: ``python tts.py --input input/script.txt``

The real code lives in ``src/local_tts``; this file only puts it on the path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from local_tts.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
