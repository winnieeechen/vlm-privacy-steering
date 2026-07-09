#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))

from low_rank_vector_extraction import run_extraction


DEFAULTS = {
    "case_csv": "outputs/04_low_rank_discriminant_vectors/02_over/vector_cases/behavior_vector_cases_train_717.csv",
    "output": "outputs/04_low_rank_discriminant_vectors/02_over/vectors/behavior_vectors_full1200.pt",
    "activation_cache": "outputs/04_low_rank_discriminant_vectors/02_over/cache/behavior_train_last_token_activations.pt",
    "label_column": "behavior_label",
    "vector_key": "behavior_vector",
    "positive_definition": "A_to_A and B_to_B, correct privacy behavior",
    "negative_definition": "A_to_B, A_to_C, B_to_C, over-disclosure behavior",
}


if __name__ == "__main__":
    run_extraction(DEFAULTS)
