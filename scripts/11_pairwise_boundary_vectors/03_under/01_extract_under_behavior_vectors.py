#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    delegate = ROOT / "scripts" / "11_pairwise_boundary_vectors" / "02_over" / "01_extract_behavior_vectors.py"
    cmd = [sys.executable, str(delegate), "--compose-under"]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

