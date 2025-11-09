from main import app
from fastapi.testclient import TestClient


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.text == "true"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "API online",
        "description": "API for analyzing text sentiment",
        "endpoints": {
            "analyze": "POST /analyze - Text analysis",
            "health": "GET /health - API status check"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }
