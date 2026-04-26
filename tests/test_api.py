from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.service import FRESHNESS_THRESHOLD
from app.training import train_and_package


def test_rank_endpoint_returns_top_k_results() -> None:
    train_and_package()
    client = TestClient(app)

    response = client.get("/rank/query_0049?k=3")

    assert response.status_code == 200
    body = response.json()
    assert body["query_id"] == "query_0049"
    assert body["selected_model"] in {"gradient_boosting_baseline", "random_forest_challenger"}
    assert len(body["results"]) == 3
    assert body["freshness_constraint"]["policy"] == "freshness_guard_v1"
    assert all("served_score" in row for row in body["results"])


def test_rank_endpoint_enforces_freshness_quota_when_available() -> None:
    train_and_package()
    client = TestClient(app)

    response = client.get("/rank/query_0049?k=5")

    assert response.status_code == 200
    body = response.json()
    assert body["selected_model"] in {"gradient_boosting_baseline", "random_forest_challenger"}
    fresh_results = [
        row for row in body["results"] if row["features"]["freshness_score"] >= FRESHNESS_THRESHOLD
    ]
    assert len(fresh_results) >= body["freshness_constraint"]["required_fresh_results"]
