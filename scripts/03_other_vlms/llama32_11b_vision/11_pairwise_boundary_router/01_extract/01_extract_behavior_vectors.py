#!/usr/bin/env python3
import sys
from pathlib import Path


COMMON = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON))

from pairwise_llama32 import OUT_ROOT, extraction_parser, run_pair_extraction  # noqa: E402


def main():
    parser = extraction_parser(__doc__)
    parser.set_defaults(
        activation_cache=f"{OUT_ROOT}/cache/answer_mean_activations.pt",
        activation_kind="answer_mean",
        label_column="pred_label",
        output_dir=f"{OUT_ROOT}/vectors",
        vector_key="behavior_vector",
    )
    run_pair_extraction(parser.parse_args())


if __name__ == "__main__":
    main()

