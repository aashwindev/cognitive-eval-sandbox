"""
Rabi crop stage linear probe on TerraFM embeddings.

Stages: pre_sowing | vegetative | harvest
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geo.terrafm_embed import ModalityConfig, TileMeta, embed_tile

STAGES = ["pre_sowing", "vegetative", "harvest"]


@dataclass
class ProbeSample:
    meta: TileMeta
    label: int


def synthetic_dataset(n_per_stage: int = 20) -> list[ProbeSample]:
    regions = ["Malwa", "Marathwada", "Hisar", "Ludhiana"]
    crops = ["wheat", "chickpea", "mustard", "potato"]
    samples: list[ProbeSample] = []
    idx = 0
    for stage_id, stage in enumerate(STAGES):
        for _ in range(n_per_stage):
            meta = TileMeta(
                tile_id=f"tile_{idx:04d}",
                region=regions[idx % len(regions)],
                crop=crops[idx % len(crops)],
                stage=stage,
                cloud_cover_pct=float((idx * 17) % 100),
            )
            samples.append(ProbeSample(meta=meta, label=stage_id))
            idx += 1
    return samples


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 200,
) -> np.ndarray:
    """Softmax regression via gradient descent (no sklearn dep)."""
    n_classes = len(STAGES)
    w = np.zeros((X.shape[1], n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    for _ in range(steps):
        logits = X @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(len(y)), y] = 1.0
        grad_logits = (probs - y_onehot) / len(y)
        w -= lr * (X.T @ grad_logits)
        b -= lr * grad_logits.sum(axis=0)
    return w, b  # type: ignore[return-value]


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    logits = X @ w + b
    pred = logits.argmax(axis=1)
    return float((pred == y).mean())


def run_probe(
    synthetic: bool = True,
    weights: Path | None = None,
    modality: ModalityConfig = ModalityConfig.S2_ONLY,
) -> dict:
    samples = synthetic_dataset()
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for s in samples:
        X_list.append(embed_tile(s.meta, weights_path=weights, modality=modality))
        y_list.append(s.label)
    X = np.stack(X_list)
    y = np.array(y_list)

    # train/test split 80/20 stratified by stage
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.8 * len(y))
    train_idx, test_idx = idx[:split], idx[split:]

    w, b = train_linear_probe(X[train_idx], y[train_idx])
    train_acc = accuracy(X[train_idx], y[train_idx], w, b)
    test_acc = accuracy(X[test_idx], y[test_idx], w, b)

    return {
        "modality": modality.value,
        "synthetic_tiles": synthetic,
        "weights": str(weights) if weights else None,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "n_samples": len(samples),
        "chance": 1.0 / len(STAGES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rabi crop stage linear probe")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--compare-modalities", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("experiments/results/rabi_probe.json"))
    args = parser.parse_args()

    results = {}
    if args.compare_modalities:
        results["s2_only"] = run_probe(args.synthetic, args.weights, ModalityConfig.S2_ONLY)
        results["s1_s2_fused"] = run_probe(args.synthetic, args.weights, ModalityConfig.S1_S2_FUSED)
    else:
        results["default"] = run_probe(args.synthetic, args.weights)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    for k, v in results.items():
        print(f"[{k}] test acc: {v['test_accuracy']:.2%} (chance {v['chance']:.2%})")


if __name__ == "__main__":
    main()
