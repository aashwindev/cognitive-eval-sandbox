"""
Gymnasium-compatible Agrarian ToM POMDP.

Each episode = one MBR question; agent selects MCQ after observing event stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evals.tom.monsoon_belief_revision.loader import load_scenarios
from evals.tom.monsoon_belief_revision.schema import Question, Scenario
from rl.features import FEATURE_DIM, extract_features


@dataclass
class Episode:
    scenario: Scenario
    question: Question


class AgrarianToMEnv:
    """Minimal POMDP env (gymnasium API subset — no hard gymnasium dep)."""

    metadata = {"name": "AgrarianToM-v0"}

    def __init__(self, scenarios: list[Scenario] | None = None, seed: int = 0):
        self.scenarios = scenarios or load_scenarios()
        self.rng = np.random.default_rng(seed)
        self._episodes: list[Episode] = []
        for sc in self.scenarios:
            for q in sc.questions:
                self._episodes.append(Episode(scenario=sc, question=q))
        self._idx = 0
        self._current: Episode | None = None
        self.action_space_n = 4
        self.observation_space_shape = (FEATURE_DIM,)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._idx = int(self.rng.integers(0, len(self._episodes)))
        self._current = self._episodes[self._idx]
        obs = extract_features(self._current.scenario, self._current.question)
        return obs, {"question_id": self._current.question.id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._current is None:
            raise RuntimeError("call reset() first")
        gold = self._current.question.answer_index
        reward = 1.0 if int(action) == gold else 0.0
        obs = extract_features(self._current.scenario, self._current.question)
        return obs, reward, True, False, {"correct": reward == 1.0, "gold": gold}

    def iter_all(self) -> list[Episode]:
        return list(self._episodes)
