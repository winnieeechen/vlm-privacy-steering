#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "08_weighted_mean_behavior_vectors" / "00_common"))

from weighted_mean_extraction import run_extraction


DEFAULTS = {
    "case_csv": "outputs/04_low_rank_discriminant_vectors/03_under/vector_cases/under_behavior_vector_cases_train_717.csv",
    "activation_cache": "outputs/04_low_rank_discriminant_vectors/03_under/cache/under_behavior_train_last_token_activations.pt",
    "output": "outputs/09_severity_weighted_behavior_vectors/03_under/vectors/under_behavior_vectors_severity_weighted_full1200.pt",
    "label_column": "behavior_label",
    "vector_key": "behavior_vector",
}


if __name__ == "__main__":
    run_extraction(DEFAULTS)
