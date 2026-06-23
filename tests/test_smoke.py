"""Smoke tests for cognitive-eval-sandbox."""

from pathlib import Path

from evals.scientific.causal_crop_hypothesis.runner import load_items, mock_predict
from evals.tom.monsoon_belief_revision.loader import load_scenarios
from evals.tom.monsoon_belief_revision.runner import evaluate, summarize
from geo.tasks.rabi_crop_probe import run_probe


def test_mbr_loads_eight_scenarios():
    scenarios = load_scenarios()
    assert len(scenarios) == 8
    assert sum(len(s.questions) for s in scenarios) == 32


def test_mbr_eval_runs():
    scenarios = load_scenarios()
    results = evaluate(scenarios)
    summary = summarize(results)
    assert summary["n"] == 32
    assert 0.0 <= summary["overall_accuracy"] <= 1.0


def test_cch_loads_items():
    path = Path(__file__).parent.parent / "evals/scientific/causal_crop_hypothesis/data/items.jsonl"
    items = load_items(path)
    assert len(items) >= 12
    for it in items:
        assert mock_predict(it) is not None


def test_rabi_probe_runs():
    out = run_probe()
    assert out["test_accuracy"] >= out["chance"]
