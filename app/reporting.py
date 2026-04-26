from __future__ import annotations

import json
from pathlib import Path

from app.config import REPORT_JSON_PATH, REPORT_MD_PATH


def render_markdown(report: dict[str, object]) -> str:
    selected = report["selected_model"]
    lines = [
        "# Ranking Experiment Report",
        "",
        f"Experiment ID: `{report['experiment_id']}`",
        f"Selection rule: `{report['selection_rule']}`",
        f"Selected model: `{selected['model_name']}`",
        "",
        "## Model Comparison",
        "",
    ]
    for model in report["models"]:
        lines.append(
            f"- `{model['model_name']}` ndcg@5=`{model['ndcg_at_5']}` map@5=`{model['map_at_5']}` queries=`{model['queries_evaluated']}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: dict[str, object]) -> tuple[Path, Path]:
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD_PATH.write_text(render_markdown(report), encoding="utf-8")
    return REPORT_JSON_PATH, REPORT_MD_PATH


def load_report() -> dict[str, object]:
    return json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
