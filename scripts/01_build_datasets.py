#!/usr/bin/env python
"""Build processed datasets and deterministic splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism_tutor.data.build_dataset import main


if __name__ == "__main__":
    main()
