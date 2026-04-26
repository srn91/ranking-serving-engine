from __future__ import annotations

import math
import random
from dataclasses import dataclass

from app.config import CANDIDATES_PER_QUERY, TRAIN_QUERY_COUNT, VALIDATION_QUERY_COUNT


@dataclass(frozen=True)
class CandidateRow:
    query_id: str
    item_id: str
    affinity_score: float
    freshness_score: float
    popularity_score: float
    price_fit_score: float
    relevance_label: int

    def features(self) -> list[float]:
        return [
            self.affinity_score,
            self.freshness_score,
            self.popularity_score,
            self.price_fit_score,
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "item_id": self.item_id,
            "affinity_score": self.affinity_score,
            "freshness_score": self.freshness_score,
            "popularity_score": self.popularity_score,
            "price_fit_score": self.price_fit_score,
            "relevance_label": self.relevance_label,
        }


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _label_from_latent(latent_score: float) -> int:
    if latent_score >= 0.72:
        return 3
    if latent_score >= 0.64:
        return 2
    if latent_score >= 0.58:
        return 1
    return 0


def generate_dataset(seed: int = 20260426) -> dict[str, list[CandidateRow]]:
    rng = random.Random(seed)
    dataset: dict[str, list[CandidateRow]] = {"train": [], "validation": []}

    def _rows_for_split(split: str, count: int, split_offset: int) -> list[CandidateRow]:
        rows: list[CandidateRow] = []
        for query_index in range(count):
            query_id = f"query_{split_offset + query_index + 1:04d}"
            segment_bias = rng.uniform(-0.08, 0.08)
            for candidate_index in range(CANDIDATES_PER_QUERY):
                item_id = f"item_{query_index:03d}_{candidate_index:02d}"
                affinity = _bounded(0.35 + 0.07 * candidate_index + segment_bias + rng.uniform(-0.12, 0.12))
                freshness = _bounded(1.0 - candidate_index * 0.06 + rng.uniform(-0.08, 0.08))
                popularity = _bounded(0.25 + math.sin(candidate_index / 2.0) * 0.25 + rng.uniform(0.0, 0.25))
                price_fit = _bounded(0.9 - abs(candidate_index - 3) * 0.11 + rng.uniform(-0.07, 0.07))
                latent_score = (
                    0.42 * affinity
                    + 0.22 * freshness
                    + 0.16 * popularity
                    + 0.20 * price_fit
                )
                relevance_label = _label_from_latent(latent_score)
                rows.append(
                    CandidateRow(
                        query_id=query_id,
                        item_id=item_id,
                        affinity_score=round(affinity, 6),
                        freshness_score=round(freshness, 6),
                        popularity_score=round(popularity, 6),
                        price_fit_score=round(price_fit, 6),
                        relevance_label=relevance_label,
                    )
                )
        return rows

    dataset["train"] = _rows_for_split("train", TRAIN_QUERY_COUNT, 0)
    dataset["validation"] = _rows_for_split("validation", VALIDATION_QUERY_COUNT, TRAIN_QUERY_COUNT)
    return dataset
