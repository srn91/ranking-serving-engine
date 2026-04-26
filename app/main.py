from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.service import ensure_artifacts_exist, load_metrics, load_validation_rows, rank_query


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_artifacts_exist()
    yield


app = FastAPI(
    title="Ranking Serving Engine",
    description="A local-first ranking system with offline evaluation and artifact-backed top-k serving.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "metrics": load_metrics()}


@app.get("/queries")
def queries() -> dict[str, list[str]]:
    unique_queries = sorted({row["query_id"] for row in load_validation_rows()})
    return {"queries": unique_queries}


@app.get("/rank/{query_id}")
def rank(query_id: str, k: int = 5) -> dict[str, object]:
    try:
        return rank_query(query_id, k=k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown query_id: {query_id}") from exc
