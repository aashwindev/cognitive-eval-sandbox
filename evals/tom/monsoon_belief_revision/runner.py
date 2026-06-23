"""Run MonsoonBelief-Rev evaluation."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .loader import load_scenarios
from .schema import QuestionType, Scenario


@dataclass
class EvalResult:
    question_id: str
    question_type: QuestionType
    predicted: int
    gold: int
    correct: bool


class MockLLM:
    """Keyword heuristic mock — biases toward applied ToM difficulty."""

    def complete(self, prompt: str, choices: list[str]) -> int:
        text = prompt.lower()
        # naive: pick first matching choice keyword in prompt
        for i, choice in enumerate(choices):
            tokens = choice.lower().split()[:3]
            if any(t in text for t in tokens if len(t) > 4):
                return i
        return 0


class HypothesisBankRunner:
    """Minimal Thought Tracing-style loop for MBR."""

    def __init__(self, llm: MockLLM, k: int = 4):
        self.llm = llm
        self.k = k

    def answer(self, scenario: Scenario, question: str, choices: list[str]) -> int:
        hypotheses = [f"Agent mental state hypothesis {i+1}" for i in range(self.k)]
        for event in scenario.events:
            obs = event.text.lower()
            # reweight: keep hypotheses whose index parity matches observation hash
            score = sum(ord(c) for c in obs) % self.k
            hypotheses = hypotheses[score:] + hypotheses[:score]
        enriched = f"{scenario.narrative()}\n\n{question}\nHypotheses: {hypotheses[0]}"
        return self.llm.complete(enriched, choices)


def parse_choice_index(response: str, n_choices: int) -> int:
    m = re.search(r"\b([0-9]+)\b", response)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_choices:
            return idx
        if 1 <= idx <= n_choices:
            return idx - 1
    return 0


def evaluate(
    scenarios: list[Scenario],
    use_tracing: bool = False,
    mock: bool = True,
) -> list[EvalResult]:
    llm = MockLLM()
    tracer = HypothesisBankRunner(llm) if use_tracing else None
    results: list[EvalResult] = []

    for scenario in scenarios:
        for q in scenario.questions:
            prompt = f"{scenario.narrative()}\n\nQuestion: {q.prompt}\nChoices:\n"
            for i, c in enumerate(q.choices):
                prompt += f"{i}. {c}\n"
            if tracer:
                pred = tracer.answer(scenario, q.prompt, q.choices)
            else:
                pred = llm.complete(prompt, q.choices)
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
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--use-tracing", action="store_true")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/mbr32.json"))
    args = parser.parse_args()

    scenarios = load_scenarios(args.data)
    results = evaluate(scenarios, use_tracing=args.use_tracing, mock=args.mock)
    summary = summarize(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "MBR-32",
        "use_tracing": args.use_tracing,
        "summary": summary,
        "details": [r.__dict__ for r in results],
    }
    # serialize enums
    for d in payload["details"]:
        d["question_type"] = d["question_type"].value
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"MBR-32 overall accuracy: {summary['overall_accuracy']:.2%} (n={summary['n']})")
    for k, v in summary["per_type"].items():
        print(f"  {k}: {v:.2%}")


if __name__ == "__main__":
    main()
