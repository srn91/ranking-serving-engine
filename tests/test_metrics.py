from __future__ import annotations

from app.dataset import generate_dataset
from app.evaluation import grouped_predictions, map_at_k, ndcg_at_k
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
