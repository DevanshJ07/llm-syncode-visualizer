"""
Experiment persistence layer.

Experiments are stored as individual JSON files under `logs/experiments/`.
This is intentionally simple for a research tool — swap to a database later
if experiment volume grows beyond a few thousand runs.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.models.schemas import ExperimentResult


class ExperimentStore:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.experiments_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _path(self, experiment_id: str) -> Path:
        return self.base_dir / f"{experiment_id}.json"

    # ------------------------------------------------------------------
    def new_id(self) -> str:
        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    def save(self, experiment: ExperimentResult) -> None:
        path = self._path(experiment.experiment_id)
        path.write_text(
            experiment.model_dump_json(indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    def load(self, experiment_id: str) -> ExperimentResult | None:
        path = self._path(experiment_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentResult(**data)

    # ------------------------------------------------------------------
    def list_ids(self) -> list[str]:
        return [p.stem for p in sorted(self.base_dir.glob("*.json"), reverse=True)]

    # ------------------------------------------------------------------
    def create_empty(self, prompt: str, mode: str, model_name: str) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=self.new_id(),
            prompt=prompt,
            mode=mode,
            model_name=model_name,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )


# Module-level singleton — import `store` in route handlers.
store = ExperimentStore()
