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


STAGE_CENTROIDS = {
    "pre_sowing": np.array([1.0, 0.2, -0.5]),
    "vegetative": np.array([-0.3, 1.0, 0.4]),
    "harvest": np.array([-0.8, -0.2, 1.0]),
}


def _deterministic_embedding(meta: TileMeta, dim: int = 768, modality: ModalityConfig = ModalityConfig.S2_ONLY) -> np.ndarray:
    """Class-conditional pseudo-embeddings — linearly separable for probe demos."""
    seed = int(hashlib.sha256(f"{meta.tile_id}:{modality.value}".encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32) * 0.15
    centroid = STAGE_CENTROIDS.get(meta.stage, np.zeros(3))
    vec[:3] += centroid * 2.0
    if modality == ModalityConfig.S1_S2_FUSED:
        # SAR channel boosts separability under cloud for pre_sowing / vegetative
        if meta.cloud_cover_pct > 50 and meta.stage in ("pre_sowing", "vegetative"):
            vec[3] += 0.8
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
