"""Route-level tests for strict /generate failure handling."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import DecodingStep, TopToken
from app.services.generation_validation import GenerationFailedError
from main import app


def _step() -> DecodingStep:
    return DecodingStep(
        step=1,
        context="",
        top_tokens=[TopToken(token="x", probability=0.9, token_id=1)],
        selected_token="x",
        selected_token_id=1,
        entropy_before=0.5,
    )


@pytest.fixture
def client():
    return TestClient(app)


def test_generate_returns_500_on_empty_steps(client):
    with patch(
        "app.api.routes.generate.llm_service.generate",
        new_callable=AsyncMock,
        return_value=("", []),
    ):
        r = client.post(
            "/generate",
            json={
                "prompt": "test",
                "use_syncode": False,
                "max_new_tokens": 4,
                "top_k": 5,
                "temperature": 1.0,
            },
        )
    assert r.status_code == 500, r.text
    detail = r.json()["detail"]
    assert "zero decoding steps" in str(detail) or "generation_failed" in str(detail)


def test_generate_returns_500_on_validation_error(client):
    with patch(
        "app.api.routes.generate.llm_service.generate",
        new_callable=AsyncMock,
        side_effect=GenerationFailedError(
            "test failure",
            reasons=["zero decoding steps"],
            step_count=0,
        ),
    ):
        r = client.post(
            "/generate",
            json={
                "prompt": "test",
                "use_syncode": False,
                "max_new_tokens": 4,
                "top_k": 5,
                "temperature": 1.0,
            },
        )
    assert r.status_code == 500
    assert r.json()["detail"]["error"] == "generation_failed"


def test_generate_returns_201_on_success(client):
    with patch(
        "app.api.routes.generate.llm_service.generate",
        new_callable=AsyncMock,
        return_value=("hello", [_step()]),
    ):
        r = client.post(
            "/generate",
            json={
                "prompt": "test",
                "use_syncode": False,
                "max_new_tokens": 4,
                "top_k": 5,
                "temperature": 1.0,
            },
        )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["total_steps"] == 1
    assert len(body["steps"]) == 1
    assert body["generated_text"] == "hello"
