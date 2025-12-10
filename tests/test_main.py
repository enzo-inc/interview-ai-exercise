"""Tests for `api/main.py`."""
from fastapi.testclient import TestClient

from ai_exercise.main import app

client = TestClient(app)

def test_health_check_route() -> None:
    response = client.get("/health")
    assert response.status_code == 200

def test_chat_route() -> None:
    response = client.post("/chat", json={"query": "What is the capital of France?"})
    assert response.status_code == 200
