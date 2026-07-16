#!/usr/bin/env python3
import sys
from pathlib import Path


COMMON = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON))

from pairwise_llama32 import router_parser, run_router  # noqa: E402


if __name__ == "__main__":
    run_router(router_parser().parse_args())

