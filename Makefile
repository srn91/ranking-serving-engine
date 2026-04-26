.PHONY: train evaluate serve test lint verify clean

train:
	python3 -m app.cli train

evaluate:
	python3 -m app.cli evaluate

serve:
	uvicorn app.main:app --host 127.0.0.1 --port 8002

test:
	pytest -q

lint:
	ruff check app tests

verify: lint test train evaluate

clean:
	rm -rf artifacts
