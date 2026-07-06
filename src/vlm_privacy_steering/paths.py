from pathlib import Path


def find_project_root() -> Path:
    """
    Find the root directory of the vlm-privacy-steering project.

    This avoids hard-coding ROOT = Path(__file__).resolve().parents[1],
    because scripts may be moved into deeper folders.
    """
    p = Path(__file__).resolve()

    for parent in p.parents:
        if (parent / "README.md").exists() and (parent / "external").exists():
            return parent

    raise RuntimeError("Cannot find project root. Expected README.md and external/.")


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
NOTES_DIR = PROJECT_ROOT / "notes"
LOGS_DIR = PROJECT_ROOT / "logs"

PILOT_DATA_DIR = DATA_DIR / "01_pilot_649"
FULL1200_DATA_DIR = DATA_DIR / "02_full1200"

PILOT_OUTPUTS_DIR = OUTPUTS_DIR / "01_pilot_small_scale"
FORMAL_OUTPUTS_DIR = OUTPUTS_DIR / "02_formal_full1200"

FULL1200_IMAGES_DIR = FULL1200_DATA_DIR / "images"
FULL1200_METADATA_DIR = FULL1200_DATA_DIR / "metadata"
FULL1200_PROCESSED_DIR = FULL1200_DATA_DIR / "processed_original_logic"

FULL1200_OVER_OUTPUTS_DIR = FORMAL_OUTPUTS_DIR / "02_over"
FULL1200_UNDER_OUTPUTS_DIR = FORMAL_OUTPUTS_DIR / "03_under"
FULL1200_COMBINED_OUTPUTS_DIR = FORMAL_OUTPUTS_DIR / "04_combined_router"
FULL1200_DUAL_OUTPUTS_DIR = FORMAL_OUTPUTS_DIR / "05_dual_additive"
