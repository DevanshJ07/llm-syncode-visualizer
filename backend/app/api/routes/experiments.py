"""
GET /experiment/{id}
GET /experiment/{id}/steps/{step}
GET /experiments  (bonus list endpoint)

Read-only endpoints for retrieving stored experiment data.
All data is read from the JSON files written by the generate route.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import ExperimentResult, StepResponse
from app.services.experiment_store import store

router = APIRouter()


@router.get("/experiments", response_model=list[str])
async def list_experiments() -> list[str]:
    """Return all stored experiment IDs, newest first."""
    return store.list_ids()


@router.get("/experiment/{experiment_id}", response_model=ExperimentResult)
async def get_experiment(experiment_id: str) -> ExperimentResult:
    """Return the full experiment record including all decoding steps."""
    experiment = store.load(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )
    return experiment


@router.get("/experiment/{experiment_id}/steps/{step}", response_model=StepResponse)
async def get_step(experiment_id: str, step: int) -> StepResponse:
    """Return a single decoding step from an experiment.

    step is 1-indexed to match the JSON log format in PROJECT_SPEC.
    """
    experiment = store.load(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )

    if step < 1 or step > experiment.total_steps:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Step {step} out of range (1–{experiment.total_steps}).",
        )

    # Steps list is 0-indexed internally; step param is 1-indexed.
    return StepResponse(step=experiment.steps[step - 1], total_steps=experiment.total_steps)
