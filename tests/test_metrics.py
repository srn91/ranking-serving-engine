from __future__ import annotations

from app.dataset import generate_dataset
from app.evaluation import grouped_predictions, map_at_k, ndcg_at_k
from app.service import apply_freshness_guard
from app.training import train_and_package


def test_training_metrics_clear_quality_bar() -> None:
    metrics = train_and_package()

    assert metrics["queries_evaluated"] == 12
    assert metrics["ndcg_at_5"] >= 0.93
    assert metrics["map_at_5"] >= 0.88


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
