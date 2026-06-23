"""Load MBR-32 scenarios from bundled JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Scenario

DATA_DIR = Path(__file__).parent / "data"


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    data_path = path or (DATA_DIR / "samples.jsonl")
    scenarios: list[Scenario] = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(Scenario.model_validate_json(line))
    return scenarios
