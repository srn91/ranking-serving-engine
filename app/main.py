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
<style>
body{{margin:0;background:#f8fafc;color:#0f172a;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;line-height:1.5}}
main{{max-width:1080px;margin:0 auto;padding:56px 24px}}.hero{{background:linear-gradient(135deg,#111827,#4338ca);color:white;border-radius:22px;padding:38px;box-shadow:0 24px 60px rgba(15,23,42,.18)}}
.eyebrow{{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#c7d2fe;font-weight:700}}h1{{font-size:42px;line-height:1.05;margin:10px 0 14px}}.hero p{{font-size:17px;color:#e0e7ff;max-width:780px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:22px 0}}.card{{background:white;border:1px solid #e2e8f0;border-radius:16px;padding:18px;box-shadow:0 10px 30px rgba(15,23,42,.06)}}
.metric{{font-size:25px;font-weight:800;color:#0f172a}}.label{{font-size:13px;color:#64748b;margin-top:3px}}.links{{display:flex;flex-wrap:wrap;gap:12px;margin-top:22px}}
a.button{{background:#0f172a;color:white;text-decoration:none;padding:11px 14px;border-radius:10px;font-weight:700}}a.secondary{{background:white;color:#0f172a;border:1px solid #cbd5e1}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}h1{{font-size:34px}}}}
</style></head>
<body><main>
<section class="hero"><div class="eyebrow">Ranking service</div><h1>Ranking Serving Engine</h1>
<p>Artifact-backed ranking service with held-out evaluation, model comparison, query-item features, and top-k serving.</p>
<div class="links"><a class="button" href="/rank/query_0049?k=5">Sample top-k ranking</a><a class="button secondary" href="/queries">Query IDs</a><a class="button secondary" href="/docs">API docs</a></div></section>
<section class="grid">
<div class="card"><div class="metric">{selected_model}</div><div class="label">selected model</div></div>
<div class="card"><div class="metric">12</div><div class="label">query groups</div></div>
<div class="card"><div class="metric">NDCG@5</div><div class="label">selection metric</div></div>
<div class="card"><div class="metric">top-k</div><div class="label">serving path</div></div>
</section>
<section class="card"><p>Use the sample ranking to inspect returned items, feature scores, and freshness constraints for a validation query.</p></section>
</main></body></html>"""


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
