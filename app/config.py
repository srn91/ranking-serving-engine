from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
DATASET_PATH = ARTIFACTS_DIR / "ranking_dataset.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_report.md"
TRAIN_QUERY_COUNT = 48
VALIDATION_QUERY_COUNT = 12
CANDIDATES_PER_QUERY = 10
