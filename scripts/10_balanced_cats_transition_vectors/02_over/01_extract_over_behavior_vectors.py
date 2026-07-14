#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "10_balanced_cats_transition_vectors" / "00_common"))

from balanced_cats_extraction import run_extraction


DEFAULTS = {
    "side": "over",
    "base_csv": "outputs/02_formal_full1200/00_base/base_qwen_vl_train_717.csv",
    "cache": "outputs/10_balanced_cats_transition_vectors/00_cache/all_train_answer_suffix_hidden_states.pt",
    "output": "outputs/10_balanced_cats_transition_vectors/02_over/vectors/behavior_vectors_balanced_cats_full1200.pt",
    "reference_vector": "outputs/04_low_rank_discriminant_vectors/02_over/vectors/behavior_vectors_full1200.pt",
    "reference_vector_key": "behavior_vector",
    "vector_key": "behavior_vector",
}


if __name__ == "__main__":
    run_extraction(DEFAULTS)
