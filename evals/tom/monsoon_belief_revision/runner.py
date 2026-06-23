"""Run MonsoonBelief-Rev evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rl.belief_tracker import BeliefPolicy

from .loader import load_scenarios
from .schema import QuestionType, Scenario


@dataclass
class EvalResult:
    question_id: str
    question_type: QuestionType
    predicted: int
    gold: int
    correct: bool


def evaluate(
    scenarios: list[Scenario],
    policy: str = "belief",
) -> list[EvalResult]:
    belief = BeliefPolicy()
    results: list[EvalResult] = []

    for scenario in scenarios:
        for q in scenario.questions:
            if policy == "belief":
                pred = belief.predict(scenario, q)
            else:
                pred = 0
            results.append(
                EvalResult(
                    question_id=q.id,
                    question_type=q.question_type,
                    predicted=pred,
                    gold=q.answer_index,
                    correct=pred == q.answer_index,
                )
            )
    return results


def summarize(results: list[EvalResult]) -> dict:
    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_type[r.question_type.value].append(r.correct)
    return {
        "overall_accuracy": sum(r.correct for r in results) / len(results),
        "per_type": {k: sum(v) / len(v) for k, v in by_type.items()},
        "n": len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MonsoonBelief-Rev (MBR-32)")
    parser.add_argument("--policy", choices=["belief"], default="belief")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/mbr32.json"))
    args = parser.parse_args()

    scenarios = load_scenarios(args.data)
    results = evaluate(scenarios, policy=args.policy)
    summary = summarize(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "MBR-32",
        "policy": args.policy,
        "summary": summary,
        "details": [r.__dict__ for r in results],
    }
    for d in payload["details"]:
        d["question_type"] = d["question_type"].value
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"MBR-32 overall accuracy: {summary['overall_accuracy']:.2%} (n={summary['n']})")
    for k, v in summary["per_type"].items():
        print(f"  {k}: {v:.2%}")


if __name__ == "__main__":
    main()
