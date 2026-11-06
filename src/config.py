from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURE_DIR = RESULTS_DIR / "figures"

METRICS_DIR = RESULTS_DIR / "metrics"


# Dataset
DATASET_NAME = "Cora"

# Reproducibility
SEED = 42