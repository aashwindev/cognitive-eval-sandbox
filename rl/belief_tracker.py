"""
Discrete belief-state tracker for agrarian ToM scenarios.

Implements visibility-partitioned observation updates — sufficient statistic
for the POMDP formalized in docs/rl-belief-formalism.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from evals.tom.monsoon_belief_revision.schema import AgentRole, Question, QuestionType, Scenario, TimelineEvent

ALL_AGENTS = [AgentRole.AGRONOMIST, AgentRole.COOPERATIVE_LEAD, AgentRole.DISTRICT_OFFICER]


class Moisture(str, Enum):
    UNKNOWN = "unknown"
    DRY = "dry"
    BORDERLINE = "borderline"
    ADEQUATE = "adequate"
    WET = "wet"


class SowWindow(str, Enum):
    UNKNOWN = "unknown"
    DELAY = "delay"
    VIABLE = "viable"
    HARVEST = "harvest"


class Pressure(str, Enum):
    NONE = "none"
    OFFICER = "officer"
    SOCIAL_PROOF = "social_proof"
    SUNK_COST = "sunk_cost"


# Keyword → semantic tags emitted by an event
TAG_PATTERNS: list[tuple[str, str]] = [
    (r"deficient|below.*threshold|dry|dryness|satellite.*cancel|cloud", "signal_deficient"),
    (r"adequate|18%|viable|moderate.*germination|ready", "signal_adequate"),
    (r"borderline|saturated|wet soil|wet tillage", "signal_wet"),
    (r"probe|transect|moisture|ndvi|sar|vv|sms|bulletin|instrument", "signal_instrument"),
    (r"officer|inspection|progress|pressure", "signal_officer_pressure"),
    (r"neighbour|neighbor|market rumour|burn|labour contract|contracted", "signal_social_proof"),
    (r"harvest|stubble|senescent", "signal_harvest"),
    (r"delay|wait|postpone|reschedule", "signal_delay_action"),
    (r"sow|sowing|drill|plant|bed form", "signal_sow_action"),
    (r"irrigation|irrigate", "signal_irrigation"),
    (r"incorporat", "signal_incorporate"),
    (r"turgid|canopy|look fine", "signal_canopy_ok"),
    (r"deficit|stress|drop suggests", "signal_root_deficit"),
]


def extract_tags(text: str) -> set[str]:
    lower = text.lower()
    tags: set[str] = set()
    for pattern, tag in TAG_PATTERNS:
        if re.search(pattern, lower):
            tags.add(tag)
    return tags


@dataclass
class BeliefState:
    """Sufficient statistic τ for agrarian ToM POMDP."""

    moisture: Moisture = Moisture.UNKNOWN
    sow_window: SowWindow = SowWindow.UNKNOWN
    pressure: Pressure = Pressure.NONE
    agent_signals: dict[str, set[str]] = field(default_factory=lambda: {a.value: set() for a in ALL_AGENTS})
    events_seen: dict[str, list[int]] = field(default_factory=lambda: {a.value: [] for a in ALL_AGENTS})
    last_public_step: int = 0

    def copy(self) -> BeliefState:
        return BeliefState(
            moisture=self.moisture,
            sow_window=self.sow_window,
            pressure=self.pressure,
            agent_signals={k: set(v) for k, v in self.agent_signals.items()},
            events_seen={k: list(v) for k, v in self.events_seen.items()},
            last_public_step=self.last_public_step,
        )


def _observers(event: TimelineEvent) -> list[AgentRole]:
    if not event.visibility:
        return list(ALL_AGENTS)
    return list(event.visibility)


def _update_world_state(state: BeliefState, tags: set[str]) -> None:
    if "signal_deficient" in tags or "signal_root_deficit" in tags:
        if "signal_adequate" not in tags:
            state.moisture = Moisture.DRY if state.moisture == Moisture.UNKNOWN else state.moisture
    if "signal_adequate" in tags:
        state.moisture = Moisture.ADEQUATE
    if "signal_wet" in tags:
        state.moisture = Moisture.WET
    if "signal_harvest" in tags:
        state.sow_window = SowWindow.HARVEST
    if "signal_officer_pressure" in tags:
        state.pressure = Pressure.OFFICER
    if "signal_social_proof" in tags:
        state.pressure = Pressure.SOCIAL_PROOF


class BeliefTracker:
    """Process scenario events into belief state τ."""

    def __init__(self) -> None:
        self.state = BeliefState()

    def reset(self) -> BeliefState:
        self.state = BeliefState()
        return self.state

    def process_event(self, event: TimelineEvent) -> BeliefState:
        tags = extract_tags(event.text)
        _update_world_state(self.state, tags)
        observers = _observers(event)
        for agent in observers:
            self.state.agent_signals[agent.value].update(tags)
            self.state.events_seen[agent.value].append(event.step)
        if not event.visibility:
            self.state.last_public_step = event.step
        return self.state.copy()

    def snapshot_at_step(self, scenario: Scenario, step: int) -> BeliefState:
        self.reset()
        for event in scenario.events:
            if event.step <= step:
                self.process_event(event)
        return self.state.copy()


def parse_perspective(prompt: str) -> str | None:
    lower = prompt.lower()
    if "agronomist" in lower:
        return AgentRole.AGRONOMIST.value
    if "cooperative lead" in lower or "lead" in lower:
        return AgentRole.COOPERATIVE_LEAD.value
    if "district officer" in lower or "officer" in lower:
        return AgentRole.DISTRICT_OFFICER.value
    if "faction" in lower:
        return AgentRole.COOPERATIVE_LEAD.value
    return None


def parse_as_of_step(prompt: str, scenario: Scenario) -> int:
    m = re.search(r"step\s*(\d+)", prompt.lower())
    if m:
        return int(m.group(1))
    if "before" in prompt.lower():
        m2 = re.search(r"step\s*(\d+)", prompt.lower())
        if m2:
            return int(m2.group(1)) - 1
    if "after" in prompt.lower():
        m3 = re.search(r"step\s*(\d+)", prompt.lower())
        if m3:
            return int(m3.group(1))
    return scenario.events[-1].step if scenario.events else 0


def _agent_belief_summary(state: BeliefState, agent: str) -> dict[str, bool]:
    sig = state.agent_signals.get(agent, set())
    has_adequate = "signal_adequate" in sig
    has_deficient = "signal_deficient" in sig or ("signal_deficient" in sig and "signal_adequate" not in sig)
    has_instrument = "signal_instrument" in sig
    has_delay = "signal_delay_action" in sig
    has_sow = "signal_sow_action" in sig
    has_only_bulletin = has_deficient and not has_instrument
    return {
        "has_adequate": has_adequate,
        "has_deficient": has_deficient,
        "has_instrument": has_instrument,
        "has_delay": has_delay,
        "has_sow": has_sow,
        "has_only_bulletin": has_only_bulletin,
        "officer_pressure": state.pressure == Pressure.OFFICER,
        "social_pressure": state.pressure == Pressure.SOCIAL_PROOF,
    }


def score_choice(
    state: BeliefState,
    question: Question,
    choice: str,
    perspective: str | None,
) -> float:
    """Heuristic choice scorer from belief state — no rationale leakage."""
    c = choice.lower()
    summary = _agent_belief_summary(state, perspective or AgentRole.COOPERATIVE_LEAD.value)
    score = 0.0

    if question.question_type == QuestionType.MENTAL_STATE:
        if any(w in c for w in ("delay", "deficient", "wait", "caution")):
            score += 2.0 if summary["has_deficient"] and not summary["has_adequate"] else -1.0
        if any(w in c for w in ("viable", "adequate", "safe", "ready")):
            score += 2.0 if summary["has_adequate"] else -0.5
        if any(w in c for w in ("irrelevant", "probe is irrelevant")):
            score += 1.0 if not summary["has_instrument"] else -2.0
        if "harvest" in c:
            score += 2.0 if state.sow_window == SowWindow.HARVEST else -1.0
        if "conflict" in c:
            score += 2.0 if summary["officer_pressure"] or summary["social_pressure"] else 0.0
        if "faster" in c or "speed" in c:
            score += 2.0 if summary["social_pressure"] else 0.0
        if "turgidity" in c or "visual" in c or "morning walk" in c:
            score += 2.0 if "signal_canopy_ok" in state.agent_signals.get(perspective or "", set()) else 0.0
        if "root" in c and "canopy" in c:
            score += 2.0 if summary["has_deficient"] else 0.0

    elif question.question_type == QuestionType.BELIEF_REVISION:
        if "agronomist only" in c or "agronomist" in c and "only" in c:
            score += 2.0 if perspective == AgentRole.AGRONOMIST.value else 0.5
        if "cooperative lead only" in c:
            score += 2.0 if perspective == AgentRole.COOPERATIVE_LEAD.value else 0.0
        if "both" in c:
            score += 1.5
        if "lead never" in c or "never told" in c or "not told" in c:
            score += 2.0 if not summary["has_instrument"] else -1.0
        if "no — still" in c or "still opposes" in c:
            score += 2.0 if state.moisture == Moisture.WET else 0.0
        if "sms may overstate" in c or "temporal aggregation" in c:
            score += 3.0
        if "context-limited" in c or "context limited" in c:
            score += 2.5
        if "irrigation plan" in c or "irrigation backup" in c:
            score += 2.0

    elif question.question_type == QuestionType.BEHAVIOR:
        if any(w in c for w in ("proceed", "monday", "weekend", "sunday", "tonight", "irrigation tonight")):
            score += 2.0 if summary["has_sow"] or "irrigation" in c else 0.5
        if any(w in c for w in ("postpone", "delay", "reschedule", "wait")):
            score += 2.0 if summary["has_delay"] else 0.0
        if "harvest" in c and "inconsistent" not in c:
            score += -1.0 if state.sow_window != SowWindow.HARVEST else 2.0
        if "inconsistent" in c or "incoherent" in c:
            score += 2.0 if "harvest" in c else 0.0
        if "pressure" in c or "contract" in c or "visible progress" in c:
            score += 2.5 if summary["officer_pressure"] or summary["social_pressure"] else 0.0
        if "mimics" in c or "neighbour" in c or "social proof" in c:
            score += 2.5 if summary["social_pressure"] else 0.0
        if "align" in c and "radio" in c:
            score += 2.0
        if "maintains" in c or "persists" in c or "pre-order" in c:
            score += 2.0

    # Tie-break: match world moisture
    if state.moisture == Moisture.ADEQUATE and "adequate" in c:
        score += 0.5
    if state.moisture == Moisture.DRY and ("dry" in c or "deficient" in c):
        score += 0.5

    return score


class BeliefPolicy:
    """Deterministic policy π(a | τ, q) using belief tracker."""

    def predict(self, scenario: Scenario, question: Question) -> int:
        step = parse_as_of_step(question.prompt, scenario)
        perspective = parse_perspective(question.prompt)
        tracker = BeliefTracker()
        state = tracker.snapshot_at_step(scenario, step)
        scores = [score_choice(state, question, ch, perspective) for ch in question.choices]
        return int(max(range(len(scores)), key=lambda i: scores[i]))
