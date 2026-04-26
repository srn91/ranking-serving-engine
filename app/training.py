from __future__ import annotations

import json

import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from app.config import ARTIFACTS_DIR, DATASET_PATH, METADATA_PATH, MODEL_PATH
from app.dataset import CandidateRow, generate_dataset
from app.evaluation import compare_rankers
from app.reporting import write_outputs


def _serialize_rows(rows: list[CandidateRow]) -> list[dict[str, object]]:
    return [row.to_dict() for row in rows]


def _train_rows(rows: list[CandidateRow]) -> tuple[list[list[float]], list[int]]:
    return [row.features() for row in rows], [row.relevance_label for row in rows]


def _build_models(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "gradient_boosting_baseline",
            GradientBoostingRegressor(random_state=seed, n_estimators=180, learning_rate=0.08),
        ),
        (
            "random_forest_challenger",
            RandomForestRegressor(
                random_state=seed + 11,
                n_estimators=220,
                max_depth=8,
                min_samples_leaf=2,
                n_jobs=-1,
            ),
        ),
    ]


def train_and_package(seed: int = 20260426) -> dict[str, object]:
    dataset = generate_dataset(seed=seed)
    train_rows = dataset["train"]
    validation_rows = dataset["validation"]

    train_features, train_labels = _train_rows(train_rows)

    ranked_models: list[tuple[str, object]] = []
    for model_name, model in _build_models(seed):
        model.fit(train_features, train_labels)
        ranked_models.append((model_name, model))

    experiment_report = compare_rankers(
        ranked_models,
        validation_rows,
        experiment_id="ranking-experiment-v1",
        selection_rule="higher ndcg_at_5, then map_at_5",
        seed=seed,
    )
    selected_model_name = str(experiment_report["selected_model"]["model_name"])
    selected_model = next(model for model_name, model in ranked_models if model_name == selected_model_name)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, MODEL_PATH)
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

    metadata = {
        "seed": seed,
        "experiment": experiment_report,
        "selected_model": selected_model_name,
        "model_path": str(MODEL_PATH),
        "dataset_path": str(DATASET_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_outputs(experiment_report)

    return {
        "experiment": experiment_report,
        "selected_model": selected_model_name,
    }
