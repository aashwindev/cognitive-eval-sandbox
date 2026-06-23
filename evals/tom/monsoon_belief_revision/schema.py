"""Pydantic schemas for MonsoonBelief-Rev (MBR-32) scenarios."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EventType(str, Enum):
    PERCEPTION = "perception"
    ACTION = "action"
    SPEECH = "speech"
    INSTRUMENT = "instrument"


class AgentRole(str, Enum):
    AGRONOMIST = "agronomist"
    COOPERATIVE_LEAD = "cooperative_lead"
    DISTRICT_OFFICER = "district_officer"


class QuestionType(str, Enum):
    MENTAL_STATE = "mental_state"
    BEHAVIOR = "behavior"
    BELIEF_REVISION = "belief_revision"


class TimelineEvent(BaseModel):
    step: int
    event_type: EventType
    actor: AgentRole | Literal["environment"]
    text: str
    visibility: list[AgentRole] = Field(
        default_factory=list,
        description="Agents who observe this event; empty means public.",
    )


class Question(BaseModel):
    id: str
    question_type: QuestionType
    prompt: str
    choices: list[str]
    answer_index: int
    rationale: str


class Scenario(BaseModel):
    id: str
    title: str
    region: str
    crop: str
    events: list[TimelineEvent]
    questions: list[Question]

    def narrative(self) -> str:
        lines = [f"# {self.title} ({self.region}, {self.crop})"]
        for e in self.events:
            vis = f" [visible: {', '.join(v.value for v in e.visibility)}]" if e.visibility else ""
            lines.append(f"Step {e.step} ({e.event_type.value}) — {e.actor}: {e.text}{vis}")
        return "\n".join(lines)
