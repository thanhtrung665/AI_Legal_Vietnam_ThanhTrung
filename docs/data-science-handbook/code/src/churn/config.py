"""Typed configuration: YAML for parameters, environment / .env for secrets."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SplitConfig(BaseModel):
    date_col: str = "signup_date"
    train_end: str = "2023-10-01"
    valid_end: str = "2024-01-01"
    gap_days: int = Field(0, ge=0)


class TrainConfig(BaseModel):
    target: str = "churn"
    random_state: int = 42
    data_path: str = "data/raw/churn.parquet"
    split: SplitConfig = SplitConfig()
    model_params: dict = {}
    quality_gate: dict[str, float] = {}


class Secrets(BaseSettings):
    """Never hard-code secrets; read them from the environment or a git-ignored .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # MLflow 3.x: the ./mlruns file store is in maintenance mode -> use a database backend
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    db_url: str = "sqlite:///local.db"


def load_config(path: str | Path) -> TrainConfig:
    with open(path, encoding="utf-8") as f:
        return TrainConfig(**(yaml.safe_load(f) or {}))
