from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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
    metadata = load_metrics()
    return {
        "status": "ok",
        "experiment": metadata["experiment"],
        "selected_model": metadata["selected_model"],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    metadata = load_metrics()
    selected_model = metadata["selected_model"]
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ranking Serving Engine</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:860px;margin:48px auto;padding:0 24px;line-height:1.5;color:#111}}a{{color:#0645ad}}</style></head>
<body>
<h1>Ranking Serving Engine</h1>
<p>Artifact-backed ranking service with held-out evaluation, model comparison, query-item features, and top-k serving.</p>
<ul><li>Selected model: {selected_model}</li></ul>
<h2>Open endpoints</h2>
<ul>
<li><a href="/rank/query_0049?k=5">Sample top-k ranking</a></li>
<li><a href="/queries">Validation query IDs</a></li>
<li><a href="/health">Health check</a></li>
<li><a href="/docs">API docs</a></li>
</ul>
</body></html>"""


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
