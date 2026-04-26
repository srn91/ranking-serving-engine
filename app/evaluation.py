from __future__ import annotations

import math
from collections import defaultdict


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
