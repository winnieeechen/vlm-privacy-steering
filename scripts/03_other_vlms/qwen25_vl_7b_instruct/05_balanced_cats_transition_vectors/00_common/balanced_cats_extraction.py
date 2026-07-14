#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts" / "10_balanced_cats_transition_vectors" / "00_common"))

from balanced_cats_extraction import run_extraction  # noqa: E402,F401
