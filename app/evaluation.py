from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


def ndcg_at_k(grouped_rows: dict[str, list[tuple[float, int]]], k: int = 5) -> float:
    scores: list[float] = []
    for rows in grouped_rows.values():
        ranked = sorted(rows, key=lambda row: row[0], reverse=True)[:k]
        ideal = sorted(rows, key=lambda row: row[1], reverse=True)[:k]
        dcg = sum((2**label - 1) / math.log2(index + 2) for index, (_, label) in enumerate(ranked))
        idcg = sum((2**label - 1) / math.log2(index + 2) for index, (_, label) in enumerate(ideal))
        scores.append(dcg / idcg if idcg else 0.0)
    return sum(scores) / len(scores)


def map_at_k(grouped_rows: dict[str, list[tuple[float, int]]], k: int = 5) -> float:
    averages: list[float] = []
    for rows in grouped_rows.values():
        ranked = sorted(rows, key=lambda row: row[0], reverse=True)[:k]
        hits = 0
        precision_sum = 0.0
        relevant = sum(1 for _, label in rows if label > 0)
        normalization = min(relevant, k)
        for index, (_, label) in enumerate(ranked, start=1):
            if label > 0:
                hits += 1
                precision_sum += hits / index
        averages.append(precision_sum / normalization if normalization else 0.0)
    return sum(averages) / len(averages)


def grouped_predictions(predictions: list[float], query_ids: list[str], labels: list[int]) -> dict[str, list[tuple[float, int]]]:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for prediction, query_id, label in zip(predictions, query_ids, labels, strict=True):
        grouped[query_id].append((prediction, label))
    return dict(grouped)


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    ndcg_at_5: float
    map_at_5: float


def evaluate_ranker(model_name: str, model, validation_rows: list[Any]) -> dict[str, object]:
    features = [
        [
            _value(row, "affinity_score"),
            _value(row, "freshness_score"),
            _value(row, "popularity_score"),
            _value(row, "price_fit_score"),
        ]
        for row in validation_rows
    ]
    labels = [int(_value(row, "relevance_label")) for row in validation_rows]
    query_ids = [str(_value(row, "query_id")) for row in validation_rows]
    predictions = model.predict(features).tolist()
    grouped = grouped_predictions(predictions, query_ids, labels)
    ndcg = round(ndcg_at_k(grouped, k=5), 4)
    map_score = round(map_at_k(grouped, k=5), 4)
    return {
        "model_name": model_name,
        "ndcg_at_5": ndcg,
        "map_at_5": map_score,
        "queries_evaluated": len(grouped),
    }


def _value(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row[key]
    return getattr(row, key)


def compare_rankers(
    ranked_models: list[tuple[str, object]],
    validation_rows: list[Any],
    *,
    experiment_id: str,
    selection_rule: str,
    seed: int,
) -> dict[str, object]:
    evaluations = [evaluate_ranker(model_name, model, validation_rows) for model_name, model in ranked_models]
    selected = max(evaluations, key=lambda row: (row["ndcg_at_5"], row["map_at_5"]))
    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "selection_rule": selection_rule,
        "models": evaluations,
        "selected_model": selected,
    }
