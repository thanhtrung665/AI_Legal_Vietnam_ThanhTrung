.PHONY: install lint test train serve docker

install:
	pip install -e ".[dev,extra]"

lint:
	ruff check src tests && ruff format --check src tests

test:
	pytest -q

train:
	python -m churn.models.train --config configs/train.yaml

serve:
	MODEL_PATH=models/model.joblib uvicorn churn.serving.api:app --host 0.0.0.0 --port 8000

docker:
	docker build -t churn-api:local .
