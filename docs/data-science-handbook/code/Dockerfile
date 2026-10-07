# syntax=docker/dockerfile:1
# Multi-stage build: dependencies are installed in a builder stage, the runtime image stays small.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim AS runtime
# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY models/model.joblib models/model.joblib
ENV MODEL_PATH=/app/models/model.joblib PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
