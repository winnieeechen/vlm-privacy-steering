#!/usr/bin/env python3
from pathlib import Path

import plot_qwen_selected_test_table as table_plot


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "notes" / "presentation"

table_plot.TITLE = "Llama-3.2-11B-Vision: Selected Test Results"
table_plot.SUBTITLE = "Counts and percentages on the same 243 test examples"
table_plot.OUTPUT_CSV = OUTPUT_DIR / "llama_selected_test_results.csv"
table_plot.OUTPUT_PNG = OUTPUT_DIR / "llama_selected_test_results_table.png"
table_plot.EXPERIMENTS = [
    {
        "method": "Base",
        "setting": "No steering",
        "path": (
            "outputs/03_other_vlms/llama32_11b_vision/00_base/test/"
            "base_llama32_vision_test_243.csv"
        ),
        "case_key": "case_type",
    },
    {
        "method": "Mean Combined",
        "setting": "Conditional | over L24, under L32",
        "path": (
            "outputs/03_other_vlms/llama32_11b_vision/"
            "02_mean_behavior_vectors/04_combined_router/test/"
            "combined_test_over_first_over24_under32.csv"
        ),
        "case_key": "combined_case_type",
    },
    {
        "method": "Pairwise Router",
        "setting": "L28 | alpha_A=1.5, alpha_C=1.0",
        "path": (
            "outputs/03_other_vlms/llama32_11b_vision/"
            "11_pairwise_boundary_router/router/test/"
            "routed_llama32_vision_test_layer28_aa1.5_ac1.0_bisector.csv"
        ),
        "case_key": "steered_case_type",
    },
]


if __name__ == "__main__":
    table_plot.main()
