"""Feature extraction φ(τ, q) for RL policies on ToM POMDP."""

from __future__ import annotations

import numpy as np

from evals.tom.monsoon_belief_revision.schema import AgentRole, Question, QuestionType, Scenario
from rl.belief_tracker import (
    ALL_AGENTS,
    BeliefTracker,
    Moisture,
    Pressure,
    SowWindow,
    parse_as_of_step,
    parse_perspective,
)

MOISTURE_IDX = {m: i for i, m in enumerate(Moisture)}
SOW_IDX = {s: i for i, s in enumerate(SowWindow)}
PRESSURE_IDX = {p: i for i, p in enumerate(Pressure)}
QTYPE_IDX = {q: i for i, q in enumerate(QuestionType)}
AGENT_IDX = {a.value: i for i, a in enumerate(ALL_AGENTS)}

N_MOISTURE = len(Moisture)
N_SOW = len(SowWindow)
N_PRESSURE = len(Pressure)
N_QTYPE = len(QuestionType)
N_AGENTS = len(ALL_AGENTS)
N_SIGNALS = 12  # fixed signal bucket count

SIGNAL_BUCKETS = [
    "signal_deficient",
    "signal_adequate",
    "signal_wet",
    "signal_instrument",
    "signal_officer_pressure",
    "signal_social_proof",
    "signal_harvest",
    "signal_delay_action",
    "signal_sow_action",
    "signal_irrigation",
    "signal_canopy_ok",
    "signal_root_deficit",
]

FEATURE_DIM = N_MOISTURE + N_SOW + N_PRESSURE + N_QTYPE + N_AGENTS + N_AGENTS * N_SIGNALS + 4


def extract_features(scenario: Scenario, question: Question) -> np.ndarray:
    step = parse_as_of_step(question.prompt, scenario)
    perspective = parse_perspective(question.prompt) or AgentRole.COOPERATIVE_LEAD.value
    tracker = BeliefTracker()
    state = tracker.snapshot_at_step(scenario, step)

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    offset = 0

    feat[offset + MOISTURE_IDX[state.moisture]] = 1.0
    offset += N_MOISTURE
    feat[offset + SOW_IDX[state.sow_window]] = 1.0
    offset += N_SOW
    feat[offset + PRESSURE_IDX[state.pressure]] = 1.0
    offset += N_PRESSURE
    feat[offset + QTYPE_IDX[question.question_type]] = 1.0
    offset += N_QTYPE
    if perspective in AGENT_IDX:
        feat[offset + AGENT_IDX[perspective]] = 1.0
    offset += N_AGENTS

    for ai, agent in enumerate(ALL_AGENTS):
        sigs = state.agent_signals.get(agent.value, set())
        for si, bucket in enumerate(SIGNAL_BUCKETS):
            if bucket in sigs:
                feat[offset + ai * N_SIGNALS + si] = 1.0
    offset += N_AGENTS * N_SIGNALS

    feat[offset] = step / max(1, len(scenario.events))
    feat[offset + 1] = state.last_public_step / max(1, len(scenario.events))
    feat[offset + 2] = len(question.choices) / 4.0
    feat[offset + 3] = 1.0  # bias

    return feat
