from main import app
from fastapi.testclient import TestClient

from .utils import assert_json_structure

client = TestClient(app)


emotion_struct = {
    "label": str,
    "score": float,
    "percentage": float
}


def test_analysis():
    response = client.post("/analyze", json={"text": "sample text"})
    assert response.status_code == 200
    assert_json_structure(response.json(), {
        "top_emotion": emotion_struct,
        "emotions": [emotion_struct]
    })


def test_empty_text():
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Empty text"
    }


def test_empty_body():
    response = client.post("/analyze", json={})
    assert response.status_code == 422
