#!/usr/bin/env python3
"""Entry point for building a Markush IR corpus from a SMILES file.

Thin wrapper around ``markushscribe.markushgen.cli`` so the pipeline can be
invoked as ``python tools/build_markush_corpus.py --smi ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markushscribe.markushgen.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
