from __future__ import annotations

from app.config import METADATA_PATH, REPORT_JSON_PATH, REPORT_MD_PATH
from app.dataset import generate_dataset
from app.evaluation import grouped_predictions, map_at_k, ndcg_at_k
from app.service import apply_freshness_guard
from app.training import train_and_package


def test_training_metrics_clear_quality_bar() -> None:
    results = train_and_package()

    assert results["experiment"]["experiment_id"] == "ranking-experiment-v1"
    assert results["experiment"]["selection_rule"] == "higher ndcg_at_5, then map_at_5"
    assert results["selected_model"] in {"gradient_boosting_baseline", "random_forest_challenger"}
    assert len(results["experiment"]["models"]) == 2
    assert results["experiment"]["selected_model"]["model_name"] == results["selected_model"]
    assert METADATA_PATH.exists()
    assert REPORT_JSON_PATH.exists()
    assert REPORT_MD_PATH.exists()
    assert results["experiment"]["selected_model"]["ndcg_at_5"] >= 0.93
    assert results["experiment"]["selected_model"]["map_at_5"] >= 0.88


def test_metric_helpers_rank_relevant_results_highly() -> None:
    dataset = generate_dataset()
    rows = dataset["validation"][:10]
    predictions = [row.affinity_score + row.price_fit_score for row in rows]
    labels = [row.relevance_label for row in rows]
    queries = [row.query_id for row in rows]
    grouped = grouped_predictions(predictions, queries, labels)

    assert ndcg_at_k(grouped, k=5) > 0.8
    assert map_at_k(grouped, k=5) > 0.6


def test_freshness_guard_promotes_fresh_candidates_into_top_k() -> None:
    ranked_rows = [
        {
            "item_id": "stale_a",
            "score": 0.98,
            "features": {"freshness_score": 0.35},
            "relevance_label": 3,
        },
        {
            "item_id": "stale_b",
            "score": 0.95,
            "features": {"freshness_score": 0.41},
            "relevance_label": 2,
        },
        {
            "item_id": "stale_c",
            "score": 0.92,
            "features": {"freshness_score": 0.38},
            "relevance_label": 2,
        },
        {
            "item_id": "fresh_a",
            "score": 0.88,
            "features": {"freshness_score": 0.91},
            "relevance_label": 1,
        },
        {
            "item_id": "fresh_b",
            "score": 0.86,
            "features": {"freshness_score": 0.83},
            "relevance_label": 1,
        },
    ]

    reranked, metadata = apply_freshness_guard(ranked_rows, k=3, freshness_threshold=0.72, min_fresh_results=2)

    assert metadata["promotions_applied"] == 2
    assert metadata["fresh_results_in_top_k"] >= 2
    assert {"fresh_a", "fresh_b"}.issubset({row["item_id"] for row in reranked})
