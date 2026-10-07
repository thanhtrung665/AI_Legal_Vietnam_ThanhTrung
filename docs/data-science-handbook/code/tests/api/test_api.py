import joblib
import pytest
from fastapi.testclient import TestClient

PAYLOAD = [{"customer_id": "C0000001", "signup_date": "2024-01-01T00:00:00", "age": 30,
            "tenure_months": 3, "monthly_charges": 90.0, "total_charges": 270.0,
            "contract": "month-to-month", "payment_method": "cash", "region": "r01",
            "support_calls": 4, "data_usage_gb": 5.0}]


@pytest.fixture
def client(tmp_path, monkeypatch, trained_model):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_model, path)
    import churn.serving.api as api
    monkeypatch.setattr(api, "MODEL_PATH", str(path))
    with TestClient(api.app) as c:  # runs the lifespan (loads the model)
        yield c


def test_predict_ok(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    assert 0 <= r.json()[0]["churn_probability"] <= 1


def test_rejects_invalid_contract(client):
    assert client.post("/predict", json=[{**PAYLOAD[0], "contract": "lifetime"}]).status_code == 422


def test_ready(client):
    assert client.get("/ready").json()["status"] == "ready"
