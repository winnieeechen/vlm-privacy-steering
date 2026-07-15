#!/usr/bin/env python3
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / "00_common"
sys.path.insert(0, str(COMMON))
from method_entrypoints import extraction_entrypoint  # noqa: E402

if __name__ == "__main__":
    extraction_entrypoint("06_cats_pca_behavior_vectors", "natural_pc1", "under")
