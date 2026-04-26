from __future__ import annotations

import json

import joblib
from sklearn.ensemble import GradientBoostingRegressor

from app.config import ARTIFACTS_DIR, DATASET_PATH, METADATA_PATH, MODEL_PATH
from app.dataset import CandidateRow, generate_dataset
from app.evaluation import grouped_predictions, map_at_k, ndcg_at_k


def _serialize_rows(rows: list[CandidateRow]) -> list[dict[str, object]]:
    return [row.to_dict() for row in rows]


def _train_rows(rows: list[CandidateRow]) -> tuple[list[list[float]], list[int]]:
    return [row.features() for row in rows], [row.relevance_label for row in rows]


def train_and_package(seed: int = 20260426) -> dict[str, object]:
    dataset = generate_dataset(seed=seed)
    train_rows = dataset["train"]
    validation_rows = dataset["validation"]

    train_features, train_labels = _train_rows(train_rows)
    validation_features, validation_labels = _train_rows(validation_rows)

    model = GradientBoostingRegressor(random_state=seed)
    model.fit(train_features, train_labels)
    predictions = model.predict(validation_features)
    grouped = grouped_predictions(predictions.tolist(), [row.query_id for row in validation_rows], validation_labels)

    metrics = {
        "queries_evaluated": len(grouped),
        "ndcg_at_5": round(ndcg_at_k(grouped, k=5), 4),
        "map_at_5": round(map_at_k(grouped, k=5), 4),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    DATASET_PATH.write_text(
        json.dumps(
            {
                "train": _serialize_rows(train_rows),
                "validation": _serialize_rows(validation_rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    METADATA_PATH.write_text(json.dumps({"seed": seed, "metrics": metrics}, indent=2), encoding="utf-8")

    return metrics
