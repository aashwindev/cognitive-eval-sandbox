"""
Thought Tracing adapter + belief-state SMC comparison.

Kim et al. (COLM 2025): NL hypothesis particles with LLM likelihoods.
This module: discrete belief particles via BeliefTracker (see rl/belief_tracker.py).
"""

from __future__ import annotations

import argparse
import os

from evals.tom.monsoon_belief_revision.loader import load_scenarios
from evals.tom.monsoon_belief_revision.runner import evaluate, summarize
from rl.belief_tracker import BeliefPolicy, BeliefTracker


def try_import_upstream_tracer() -> str | None:
    root = os.environ.get("THOUGHT_TRACING_ROOT")
    if root and os.path.isdir(root):
        return root
    return None


def run_discrete_smc_demo(scenario_id: str = "mbr_001") -> None:
    """Show belief-state evolution — working particle-style trace."""
    scenarios = {s.id: s for s in load_scenarios()}
    sc = scenarios[scenario_id]
    tracker = BeliefTracker()
    print(f"=== Discrete belief trace: {sc.title} ===")
    for event in sc.events:
        state = tracker.process_event(event)
        print(
            f"  step {event.step}: moisture={state.moisture.value} "
            f"sow={state.sow_window.value} pressure={state.pressure.value}"
        )


def compare_mbr() -> None:
    scenarios = load_scenarios()
    belief_results = evaluate(scenarios, policy="belief")
    summary = summarize(belief_results)
    print("=== MBR-32 Belief-State Policy (POMDP sufficient statistic) ===")
    print(f"Accuracy: {summary['overall_accuracy']:.2%} (n={summary['n']})")
    for k, v in summary["per_type"].items():
        print(f"  {k}: {v:.2%}")
    upstream = try_import_upstream_tracer()
    if upstream:
        print(f"Upstream thought-tracing clone: {upstream}")
    else:
        print("Set THOUGHT_TRACING_ROOT for NL particle filter parity runs.")
    run_discrete_smc_demo()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-mbr", action="store_true")
    parser.add_argument("--trace", type=str, default=None, help="scenario id for SMC demo")
    args = parser.parse_args()
    if args.trace:
        run_discrete_smc_demo(args.trace)
    if args.compare_mbr:
        compare_mbr()


if __name__ == "__main__":
    main()
