#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_common"))

from low_rank_vector_extraction import run_extraction


DEFAULTS = {
    "case_csv": "outputs/04_low_rank_discriminant_vectors/03_under/vector_cases/under_condition_vector_cases_train_717.csv",
    "output": "outputs/04_low_rank_discriminant_vectors/03_under/vectors/under_condition_vectors_full1200.pt",
    "activation_cache": "outputs/04_low_rank_discriminant_vectors/03_under/cache/under_condition_train_last_token_activations.pt",
    "label_column": "condition_label",
    "vector_key": "condition_vector",
    "positive_definition": "true_label B/C, utility-allowed input",
    "negative_definition": "true_label A, should refuse or avoid location identification",
}


if __name__ == "__main__":
    run_extraction(DEFAULTS)
