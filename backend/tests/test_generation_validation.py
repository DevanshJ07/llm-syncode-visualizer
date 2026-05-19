"""Unit tests for generation_validation."""

import pytest

from app.models.schemas import DecodingStep, TopToken
from app.services.generation_validation import (
    GenerationFailedError,
    is_placeholder_output,
    validate_generation_result,
)


def _make_step(step: int = 1, token: str = "int", token_id: int = 100) -> DecodingStep:
    return DecodingStep(
        step=step,
        context="",
        top_tokens=[TopToken(token=token, probability=0.5, token_id=token_id)],
        selected_token=token,
        selected_token_id=token_id,
        entropy_before=1.0,
    )


def test_validate_ok():
    steps = [_make_step()]
    validate_generation_result("int foo", steps, [{"step": 1}])


def test_validate_zero_steps():
    with pytest.raises(GenerationFailedError) as exc:
        validate_generation_result("", [], [])
    assert "zero decoding steps" in str(exc.value)


def test_validate_empty_text():
    with pytest.raises(GenerationFailedError):
        validate_generation_result("  ", [_make_step()], [{"step": 1}])


def test_is_placeholder():
    assert is_placeholder_output("")
    assert is_placeholder_output("// (start)")
    assert not is_placeholder_output("int main()")
