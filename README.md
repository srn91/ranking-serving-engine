# ranking-serving-engine

A local-first ranking system that generates query-item candidates, compares two ranking models, records experiment metadata and a comparison report, and serves top-k results through a FastAPI endpoint.

## Problem

Recommendation and search systems are not just "train a model and hope." A credible ranking stack needs candidate features, a reproducible offline evaluation path, an artifact-backed serving layer, and a clean way to inspect what the top-k API is actually doing. This repo focuses on that end-to-end ranking workflow.

## Architecture

The current implementation keeps the stack laptop-runnable while still reflecting a real serving shape:

- deterministic synthetic query-item candidates simulate a personalization workload
- a feature pipeline computes affinity, freshness, price fit, and popularity signals per candidate
- two rankers are trained on the same query-grouped relevance labels and compared on the same validation set
- offline evaluation computes NDCG@5 and MAP@5 for each model
- a reporting layer writes experiment metadata, comparison JSON, and a Markdown summary
- a serving layer loads the selected artifact and returns top-k ranked items for a query

## Pipeline Walkthrough

The ranking path is split so the offline and online stories stay easy to reason about:

1. `app/dataset.py` generates query-grouped candidates and labels.
2. `app/training.py` fits the candidate rankers, selects the winner, and writes the artifact bundle.
3. `app/evaluation.py` computes grouped ranking metrics and the model comparison report.
4. `app/reporting.py` writes the JSON and Markdown experiment outputs.
5. `app/service.py` loads the selected model and ranks query items without retraining.
6. `app/main.py` serves the health, query index, and `/rank/{query_id}` endpoints.

The serving layer is intentionally artifact-backed and retraining-free. That keeps the request path predictable and makes it obvious where to add explicit latency instrumentation or caching later, rather than hiding those concerns inside the model code.

```mermaid
flowchart LR
    A["Synthetic query-item candidates"] --> B["Feature pipeline"]
    B --> C["Training dataset"]
    C --> D["Gradient-boosted baseline"]
    C --> E["Random forest challenger"]
    D --> F["Offline evaluation (NDCG / MAP)"]
    E --> F
    F --> G["Experiment comparison report"]
    G --> H["Artifact registry"]
    H --> I["FastAPI ranking service"]
    I --> J["/rank/{query_id} top-k response"]
```

## Tradeoffs

This implementation makes three deliberate tradeoffs:

1. The repo uses deterministic synthetic ranking data instead of a large behavioral log so the full workflow is reproducible locally.
2. The scorers are scikit-learn regressors rather than a heavier dedicated ranking library because local runnability matters more than squeezing a few extra points from the demo.
3. Serving uses artifact-backed in-memory ranking rather than Redis or a feature store so the repo stays focused on ranking logic and response shape before adding infrastructure depth.

## Repo Layout

```text
ranking-serving-engine/
├── app/
│   ├── cli.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── main.py
│   ├── reporting.py
│   ├── service.py
│   └── training.py
├── artifacts/
└── tests/
```

## Run Steps

### Install Dependencies

```bash
git clone https://github.com/srn91/ranking-serving-engine.git
cd ranking-serving-engine
python3 -m pip install -r requirements.txt
```

### Train the Rankers

```bash
make train
```

That produces:

- `artifacts/model.joblib`
- `artifacts/ranking_dataset.json`
- `artifacts/metadata.json`
- `artifacts/experiment_report.json`
- `artifacts/experiment_report.md`

### Evaluate Ranking Quality

```bash
make evaluate
```

`make evaluate` prints the selected model plus the experiment comparison report without retraining.

### Start the Ranking API

```bash
make serve
```

Useful endpoints:

- `http://127.0.0.1:8002/health`
- `http://127.0.0.1:8002/queries`
- `http://127.0.0.1:8002/rank/query_0049?k=5`

The served ranking response now includes a `freshness_constraint` block and a `served_score` per row so you can see when the request path promoted fresher candidates into the top-k window.

### Run the Full Quality Gate

```bash
make verify
```

## Hosted Deployment

- Live URL: [https://ranking-serving-engine.onrender.com](https://ranking-serving-engine.onrender.com)
- First path to click: `/queries`, then `/rank/query_0049?k=5`
- Browser smoke: passed on `/rank/query_0049?k=5`; direct HTTP to `/queries` and `/rank/query_0049?k=5` returned `200`
- Render config: Git-backed Python web service on `main`, `buildCommand=python3 -m pip install -r requirements.txt`, `startCommand=uvicorn app.main:app --host 0.0.0.0 --port $PORT`, `healthCheckPath=/health`, `plan=free`, `region=oregon`, auto-deploy enabled

## Validation

The repo currently verifies:

- deterministic generation of grouped ranking candidates
- artifact-backed training and serving with no hidden retraining in the API
- offline NDCG@5 and MAP@5 computation on held-out queries
- two ranking models compared on the same validation set with experiment metadata recorded to disk
- top-k serving for known queries using the stored artifact package
- a freshness-aware serving constraint can promote fresher candidates into top-k when the base scorer over-concentrates on stale results

The evaluation surface is intentionally inspectable:

- NDCG@5 shows whether the top of the list is ordered correctly.
- MAP@5 shows whether relevant items are surfaced early and consistently.
- The `/health` response includes the selected model and experiment comparison summary.
- The `/rank/{query_id}` response includes the selected model name, the item score, and feature context so the ranking decision can be inspected instead of treated as a black box.

Current expected evaluation snapshot:

- queries evaluated: `12`
- selected model is the one with the better NDCG@5, then MAP@5
- NDCG@5 and MAP@5 are recorded per model in the experiment report
- served query path returns ranked items with feature and score context
- served top-k can include freshness-based promotions while still exposing the underlying model score

Local quality gates:

- `make lint`
- `make test`
- `make train`
- `make evaluate`
- `make verify`

## Current Capabilities

The current implementation demonstrates:

- deterministic ranking candidate generation
- feature-based scoring with two compared ranking models
- grouped offline ranking metrics
- experiment metadata and report artifacts
- artifact-backed top-k serving through FastAPI
- explicit train/evaluate/serve separation so API behavior is reproducible
- freshness-aware reranking constraints at serve time without retraining the model
