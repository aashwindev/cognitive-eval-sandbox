"""
Minimal Thought Tracing adapter (Kim et al., COLM 2025).

Implements hypothesis-bank propose → weight → resample without upstream deps.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from evals.tom.monsoon_belief_revision.loader import load_scenarios
from evals.tom.monsoon_belief_revision.runner import MockLLM, evaluate, summarize


@dataclass
class Hypothesis:
    text: str
    weight: float = 1.0


@dataclass
class HypothesisBank:
    """Particle filter over natural-language mental-state hypotheses."""

    k: int = 8
    llm: MockLLM = field(default_factory=MockLLM)
    particles: list[Hypothesis] = field(default_factory=list)

    def propose_initial(self, context: str) -> None:
        self.particles = [
            Hypothesis(text=f"Hypothesis {i+1}: agents coordinate on delayed sowing ({context[:40]}...)")
            for i in range(self.k)
        ]
        self._normalize()

    def observe(self, observation: str) -> None:
        """Weight particles by prompted likelihood of observation."""
        for p in self.particles:
            prompt = (
                f"Observation: {observation}\n"
                f"Mental state: {p.text}\n"
                "Rate consistency 0-10."
            )
            # mock likelihood from hash
            score = (sum(ord(c) for c in prompt) % 11) / 10.0
            p.weight *= max(score, 0.05)
        self._normalize()

    def resample(self) -> None:
        """Multinomial resample + light mutation."""
        if not self.particles:
            return
        cumulative: list[float] = []
        s = 0.0
        for p in self.particles:
            s += p.weight
            cumulative.append(s)
        import random

        rng = random.Random(42)
        new_particles: list[Hypothesis] = []
        for _ in range(self.k):
            r = rng.random() * s
            idx = next(i for i, c in enumerate(cumulative) if r <= c)
            parent = self.particles[idx]
            mutated = parent.text
            if rng.random() < 0.3:
                mutated += " [revised]"
            new_particles.append(Hypothesis(text=mutated, weight=1.0 / self.k))
        self.particles = new_particles

    def query(self, question: str, choices: list[str]) -> int:
        marginal = "\n".join(f"- {p.text} (w={p.weight:.3f})" for p in self.particles)
        prompt = f"Hypothesis bank:\n{marginal}\n\nQuestion: {question}"
        return self.llm.complete(prompt, choices)

    def _normalize(self) -> None:
        total = sum(p.weight for p in self.particles) or 1.0
        for p in self.particles:
            p.weight /= total


def try_import_upstream_tracer():
    root = os.environ.get("THOUGHT_TRACING_ROOT")
    if not root or not os.path.isdir(root):
        return None
    # upstream layout varies; we only check presence for integration docs
    return root


def compare_mbr() -> None:
    scenarios = load_scenarios()
    single = evaluate(scenarios, use_tracing=False)
    traced = evaluate(scenarios, use_tracing=True)
    s_single = summarize(single)
    s_traced = summarize(traced)
    print("=== MBR-32 Thought Tracing comparison (mock LLM) ===")
    print(f"Single-shot: {s_single['overall_accuracy']:.2%}")
    print(f"Tracing:     {s_traced['overall_accuracy']:.2%}")
    upstream = try_import_upstream_tracer()
    if upstream:
        print(f"Upstream thought-tracing found at: {upstream}")
    else:
        print("Set THOUGHT_TRACING_ROOT to clone for full ToMi parity.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-mbr", action="store_true")
    args = parser.parse_args()
    if args.compare_mbr:
        compare_mbr()


if __name__ == "__main__":
    main()
