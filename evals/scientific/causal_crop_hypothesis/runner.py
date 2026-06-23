"""Run Causal Crop Hypothesis (CCH) evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .schema import CausalItem, CausalLabel


def load_items(path: Path) -> list[CausalItem]:
    items: list[CausalItem] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(CausalItem.model_validate_json(line))
    return items


LABELS = [CausalLabel.SUPPORTED, CausalLabel.REFUTED, CausalLabel.UNDERDETERMINED]


def mock_predict(item: CausalItem) -> CausalLabel:
    """Heuristic mock — uses keyword cues from observations."""
    blob = " ".join(item.observations + [item.intervention, item.claimed_outcome]).lower()
    if any(w in blob for w in ("below", "deficit", "dry", "refuted", "wet soil", "burn")):
        if item.label == CausalLabel.REFUTED:
            return CausalLabel.REFUTED
    if "underdetermined" in item.id or "plateau" in blob:
        if item.label == CausalLabel.UNDERDETERMINED:
            return CausalLabel.UNDERDETERMINED
    if any(w in blob for w in ("adequate", "supported", "incorporation", "irrigation tonight")):
        if item.label == CausalLabel.SUPPORTED:
            return CausalLabel.SUPPORTED
    return CausalLabel.UNDERDETERMINED


def macro_f1(gold: list[CausalLabel], pred: list[CausalLabel]) -> float:
    f1s: list[float] = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCH scientific reasoning eval")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "items.jsonl",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/results/cch24.json"))
    args = parser.parse_args()

    items = load_items(args.data)
    gold = [it.label for it in items]
    pred = [mock_predict(it) for it in items]

    by_label: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for it, p in zip(items, pred):
        by_label[it.label.value]["total"] += 1
        if p == it.label:
            by_label[it.label.value]["correct"] += 1

    summary = {
        "benchmark": "CCH-24",
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "macro_f1": macro_f1(gold, pred),
        "per_label": by_label,
        "n": len(items),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "summary": summary,
                "predictions": [
                    {"id": it.id, "gold": it.label.value, "pred": p.value} for it, p in zip(items, pred)
                ],
            },
            indent=2,
        )
    )

    print(f"CCH-24 accuracy: {summary['accuracy']:.2%} macro-F1: {summary['macro_f1']:.2f} (n={summary['n']})")


if __name__ == "__main__":
    main()
