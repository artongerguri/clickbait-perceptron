"""Make the project root importable so ``import src.*`` works from anywhere.

Running ``pytest`` from the project root already puts the root on ``sys.path``,
but this guard keeps the suite runnable from any working directory and under any
IDE test runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
