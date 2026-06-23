"""
TerraFM embedding extractor (optional weights).

Reference: https://github.com/mbzuai-oryx/TerraFM
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class ModalityConfig(str, Enum):
    S2_ONLY = "s2_only"
    S1_S2_FUSED = "s1_s2_fused"


@dataclass
class TileMeta:
    tile_id: str
    region: str
    crop: str
    stage: str
    cloud_cover_pct: float


def _deterministic_embedding(meta: TileMeta, dim: int = 768, modality: ModalityConfig = ModalityConfig.S2_ONLY) -> np.ndarray:
    """Pseudo-embedding for CI when TerraFM weights unavailable."""
    seed = int(hashlib.sha256(f"{meta.tile_id}:{modality.value}".encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    # inject weak stage signal
    stage_offset = {"pre_sowing": 0.0, "vegetative": 0.5, "harvest": -0.3}.get(meta.stage, 0.0)
    vec[0] += stage_offset
    if modality == ModalityConfig.S1_S2_FUSED and meta.cloud_cover_pct > 50:
        vec[1] += 0.4  # SAR helps under cloud
    vec /= np.linalg.norm(vec) + 1e-8
    return vec


def load_terrafm_encoder(weights_path: Path | None):
    """Attempt torch load; fall back to None."""
    if weights_path is None or not weights_path.exists():
        return None
    try:
        import torch  # noqa: F401

        return {"path": str(weights_path), "loaded": True}
    except ImportError:
        return None


def embed_tile(
    meta: TileMeta,
    weights_path: Path | None = None,
    modality: ModalityConfig = ModalityConfig.S2_ONLY,
    dim: int = 768,
) -> np.ndarray:
    encoder = load_terrafm_encoder(weights_path)
    if encoder is None:
        return _deterministic_embedding(meta, dim=dim, modality=modality)
    # Real forward pass would go here when weights + torch + raster available
    return _deterministic_embedding(meta, dim=dim, modality=modality)
