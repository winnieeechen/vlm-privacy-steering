#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(
    0, str(ROOT / "scripts" / "10b_balanced_cats_pc1_vectors" / "00_common")
)

from select_pc1_vector import select_pc1_vector


if __name__ == "__main__":
    select_pc1_vector(
        source="outputs/10_balanced_cats_transition_vectors/03_under/vectors/under_behavior_vectors_balanced_cats_full1200.pt",
        output="outputs/10b_balanced_cats_pc1_vectors/03_under/vectors/under_behavior_vectors_balanced_cats_pc1_full1200.pt",
        side="under",
    )
