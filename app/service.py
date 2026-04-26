from __future__ import annotations

import json

import joblib

from app.config import DATASET_PATH, METADATA_PATH, MODEL_PATH


def ensure_artifacts_exist() -> None:
    missing = [path for path in (MODEL_PATH, DATASET_PATH, METADATA_PATH) if not path.exists()]
    if missing:
        formatted = ", ".join(str(path.name) for path in missing)
        raise FileNotFoundError(f"missing ranking artifacts: {formatted}. Run `make train` first.")


def load_validation_rows() -> list[dict[str, object]]:
    ensure_artifacts_exist()
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return payload["validation"]


def load_metrics() -> dict[str, object]:
    ensure_artifacts_exist()
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))["metrics"]


def rank_query(query_id: str, k: int = 5) -> dict[str, object]:
    ensure_artifacts_exist()
    rows = [row for row in load_validation_rows() if row["query_id"] == query_id]
    if not rows:
        raise KeyError(query_id)

    model = joblib.load(MODEL_PATH)
    features = [
        [
            row["affinity_score"],
            row["freshness_score"],
            row["popularity_score"],
            row["price_fit_score"],
        ]
        for row in rows
    ]
    scores = model.predict(features)
    ranked_rows = sorted(
        (
            {
                "item_id": row["item_id"],
                "score": round(float(score), 6),
                "features": {
                    "affinity_score": row["affinity_score"],
                    "freshness_score": row["freshness_score"],
                    "popularity_score": row["popularity_score"],
                    "price_fit_score": row["price_fit_score"],
                },
                "relevance_label": row["relevance_label"],
            }
            for row, score in zip(rows, scores, strict=True)
        ),
        key=lambda row: row["score"],
        reverse=True,
    )

    return {
        "query_id": query_id,
        "results": ranked_rows[:k],
    }
