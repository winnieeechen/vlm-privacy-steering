#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))

from cats_pca_behavior_extraction import run_extraction


DEFAULTS = {
    "side": "over",
    "base_csv": "outputs/02_formal_full1200/00_base/base_qwen_vl_train_717.csv",
    "cache": "outputs/06_cats_pca_behavior_vectors/02_over/cache/over_answer_suffix_hidden_states_train_layerall.pt",
    "output": "outputs/06_cats_pca_behavior_vectors/02_over/vectors/behavior_vectors_cats_pca_full1200.pt",
    "reference_vector": "outputs/04_low_rank_discriminant_vectors/02_over/vectors/behavior_vectors_full1200.pt",
    "reference_vector_key": "behavior_vector",
    "vector_key": "behavior_vector",
}


if __name__ == "__main__":
    run_extraction(DEFAULTS)
