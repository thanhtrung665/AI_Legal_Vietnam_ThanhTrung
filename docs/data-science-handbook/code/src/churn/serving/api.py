"""Online inference service: uvicorn churn.serving.api:app --port 8000"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("churn.api")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.35"))
MAX_BATCH = 1000
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)  # load once per worker, not per request
    log.info("model loaded path=%s version=%s", MODEL_PATH, MODEL_VERSION)
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version=MODEL_VERSION, lifespan=lifespan)


class Customer(BaseModel):
    customer_id: str = Field(pattern=r"^C\d{7}$")
    signup_date: datetime
    age: float | None = Field(None, ge=18, le=100)
    tenure_months: int = Field(ge=0, le=600)
    monthly_charges: float = Field(gt=0, le=2000)
    total_charges: float = Field(ge=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    payment_method: str | None = None
    region: str
    support_calls: int = Field(ge=0)
    data_usage_gb: float | None = Field(None, ge=0)


class Prediction(BaseModel):
    customer_id: str
    churn_probability: float
    is_high_risk: bool
    model_version: str
    request_id: str


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the model is loaded and requests can be served."""
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=list[Prediction])
def predict(customers: list[Customer]) -> list[Prediction]:
    # Sync def: FastAPI runs it in a threadpool, so CPU-bound inference does not block the event loop.
    if not 1 <= len(customers) <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"batch size must be in [1, {MAX_BATCH}]")
    start = time.perf_counter()
    df = pd.DataFrame([c.model_dump() for c in customers])
    df["signup_date"] = pd.to_datetime(df["signup_date"], utc=True).dt.tz_localize(None)
    try:
        proba = state["model"].predict_proba(df)[:, 1]
    except Exception as exc:  # never leak stack traces to clients
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction error") from exc
    request_id = str(uuid.uuid4())
    log.info("request_id=%s n=%d latency_ms=%.1f mean_p=%.4f", request_id, len(df),
             1000 * (time.perf_counter() - start), float(proba.mean()))
    return [
        Prediction(customer_id=cid, churn_probability=round(float(p), 6), is_high_risk=bool(p >= THRESHOLD),
                   model_version=MODEL_VERSION, request_id=request_id)
        for cid, p in zip(df["customer_id"], proba, strict=True)
    ]
