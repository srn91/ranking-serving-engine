from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.training import train_and_package


def test_rank_endpoint_returns_top_k_results() -> None:
    train_and_package()
    client = TestClient(app)

    response = client.get("/rank/query_0049?k=3")

    assert response.status_code == 200
    body = response.json()
    assert body["query_id"] == "query_0049"
    assert len(body["results"]) == 3
    assert body["results"][0]["score"] >= body["results"][1]["score"]
