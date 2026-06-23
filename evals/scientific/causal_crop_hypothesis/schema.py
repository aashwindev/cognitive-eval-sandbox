"""Causal crop hypothesis (CCH-24) — agronomic claim verification."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class CausalLabel(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNDERDETERMINED = "underdetermined"


class CausalItem(BaseModel):
    id: str
    region: str
    crop: str
    observations: list[str]
    intervention: str
    claimed_outcome: str
    label: CausalLabel
    falsifier: str
    distractor: str
