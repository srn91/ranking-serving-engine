from __future__ import annotations

import json

from app.service import load_metrics
from app.training import train_and_package


def train() -> None:
    metrics = train_and_package()
    print(json.dumps({"metrics": metrics}, indent=2))


def evaluate() -> None:
    print(json.dumps({"metrics": load_metrics()}, indent=2))


def main() -> None:
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in {"train", "evaluate"}:
        raise SystemExit("usage: python3 -m app.cli [train|evaluate]")

    command = sys.argv[1]
    if command == "train":
        train()
        return

    evaluate()


if __name__ == "__main__":
    main()
