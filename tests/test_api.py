import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # the "with" runs the startup that loads the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_positive(client):
    r = client.post("/predict", json={"text": "I love this, absolutely amazing!"})
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


def test_negative(client):
    r = client.post("/predict", json={"text": "This is terrible, I hate it so much"})
    assert r.json()["label"] == "negative"


def test_scores_sum_to_one(client):
    s = client.post("/predict", json={"text": "just got home"}).json()["scores"]
    assert abs(sum(s.values()) - 1.0) < 0.01


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["great", "awful", "ok"]})
    assert len(r.json()["results"]) == 3


def test_empty_text_rejected(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422