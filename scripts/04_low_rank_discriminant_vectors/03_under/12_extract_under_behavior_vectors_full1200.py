#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))

from low_rank_vector_extraction import run_extraction


DEFAULTS = {
    "case_csv": "outputs/04_low_rank_discriminant_vectors/03_under/vector_cases/under_behavior_vector_cases_train_717.csv",
    "output": "outputs/04_low_rank_discriminant_vectors/03_under/vectors/under_behavior_vectors_full1200.pt",
    "activation_cache": "outputs/04_low_rank_discriminant_vectors/03_under/cache/under_behavior_train_last_token_activations.pt",
    "label_column": "behavior_label",
    "vector_key": "behavior_vector",
    "positive_definition": "B_to_B and C_to_C, correct utility-preserving behavior",
    "negative_definition": "B_to_A, C_to_A, C_to_B, under-disclosure behavior",
}


if __name__ == "__main__":
    run_extraction(DEFAULTS)
