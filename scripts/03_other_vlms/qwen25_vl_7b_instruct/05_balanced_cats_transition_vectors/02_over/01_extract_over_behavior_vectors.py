#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "scripts" / "10_balanced_cats_transition_vectors" / "00_common"))

from balanced_cats_extraction import run_extraction  # noqa: E402


DEFAULTS = {
    "side": "over",
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "base_csv": "outputs/03_other_vlms/qwen25_vl_7b_instruct/00_base/train/base_qwen25_vl_7b_train_717.csv",
    "cache": "outputs/03_other_vlms/qwen25_vl_7b_instruct/05_balanced_cats_transition_vectors/00_cache/all_train_answer_suffix_hidden_states.pt",
    "output": "outputs/03_other_vlms/qwen25_vl_7b_instruct/05_balanced_cats_transition_vectors/02_over/vectors/behavior_vectors_balanced_cats_train_717.pt",
    "reference_vector": "mean_delta",
    "reference_vector_key": "behavior_vector",
    "vector_key": "behavior_vector",
}


if __name__ == "__main__":
    run_extraction(DEFAULTS)
