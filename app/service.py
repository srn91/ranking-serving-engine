from __future__ import annotations

import json

import joblib

from app.config import DATASET_PATH, METADATA_PATH, MODEL_PATH, REPORT_JSON_PATH, REPORT_MD_PATH

FRESHNESS_THRESHOLD = 0.72
MIN_FRESH_RESULTS = 2


def ensure_artifacts_exist() -> None:
    missing = [
        path
        for path in (MODEL_PATH, DATASET_PATH, METADATA_PATH, REPORT_JSON_PATH, REPORT_MD_PATH)
        if not path.exists()
    ]
    if missing:
        formatted = ", ".join(str(path.name) for path in missing)
        raise FileNotFoundError(f"missing ranking artifacts: {formatted}. Run `make train` first.")


def load_validation_rows() -> list[dict[str, object]]:
    ensure_artifacts_exist()
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return payload["validation"]


def load_metrics() -> dict[str, object]:
    ensure_artifacts_exist()
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def apply_freshness_guard(
    ranked_rows: list[dict[str, object]],
    k: int,
    freshness_threshold: float = FRESHNESS_THRESHOLD,
    min_fresh_results: int = MIN_FRESH_RESULTS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    top_k = [row.copy() for row in ranked_rows[:k]]
    eligible_fresh = [row.copy() for row in ranked_rows if float(row["features"]["freshness_score"]) >= freshness_threshold]
    required_fresh = min(k, min_fresh_results, len(eligible_fresh))
    fresh_in_window = [row for row in top_k if float(row["features"]["freshness_score"]) >= freshness_threshold]
    promotions = 0

    if required_fresh > len(fresh_in_window):
        remaining_fresh = [
            row.copy()
            for row in ranked_rows[k:]
            if float(row["features"]["freshness_score"]) >= freshness_threshold
        ]
        replaceable = sorted(
            [row for row in top_k if float(row["features"]["freshness_score"]) < freshness_threshold],
            key=lambda row: (float(row["score"]), float(row["features"]["freshness_score"])),
        )

        while remaining_fresh and replaceable and len(fresh_in_window) < required_fresh:
            promoted = remaining_fresh.pop(0)
            demoted = replaceable.pop(0)
            demoted_item_id = str(demoted["item_id"])
            top_k = [row for row in top_k if str(row["item_id"]) != demoted_item_id]
            promoted["served_score"] = round(float(promoted["score"]) + 0.05, 6)
            top_k.append(promoted)
            fresh_in_window.append(promoted)
            promotions += 1
            replaceable = sorted(
                [row for row in top_k if float(row["features"]["freshness_score"]) < freshness_threshold],
                key=lambda row: (float(row["score"]), float(row["features"]["freshness_score"])),
            )

    for row in top_k:
        row.setdefault("served_score", row["score"])

    ordered = sorted(top_k, key=lambda row: (float(row["served_score"]), float(row["score"])), reverse=True)
    return (
        ordered,
        {
            "policy": "freshness_guard_v1",
            "freshness_threshold": freshness_threshold,
            "required_fresh_results": required_fresh,
            "fresh_results_in_top_k": sum(
                int(float(row["features"]["freshness_score"]) >= freshness_threshold) for row in ordered
            ),
            "promotions_applied": promotions,
        },
    )


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
    constrained_rows, freshness_constraint = apply_freshness_guard(ranked_rows, k=k)

    return {
        "query_id": query_id,
        "selected_model": load_metrics()["selected_model"],
        "results": constrained_rows,
        "freshness_constraint": freshness_constraint,
    }
