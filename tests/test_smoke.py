"""Smoke and integration tests."""

from pathlib import Path

from evals.scientific.causal_crop_hypothesis.runner import load_items, predict
from evals.tom.monsoon_belief_revision.loader import load_scenarios
from evals.tom.monsoon_belief_revision.runner import evaluate, summarize
from geo.tasks.rabi_crop_probe import run_probe
from rl.agrarian_tom_env import AgrarianToMEnv
from rl.belief_tracker import BeliefPolicy
from rl.train import evaluate_policy, train_reinforce, train_supervised


def test_mbr_loads_eight_scenarios():
    scenarios = load_scenarios()
    assert len(scenarios) == 8
    assert sum(len(s.questions) for s in scenarios) == 32


def test_belief_policy_beats_chance():
    scenarios = load_scenarios()
    results = evaluate(scenarios, policy="belief")
    summary = summarize(results)
    assert summary["overall_accuracy"] >= 0.65, summary


def test_supervised_rl_high_accuracy():
    env = AgrarianToMEnv()
    policy = train_supervised(env, epochs=500)
    acc = evaluate_policy(policy, env)["accuracy"]
    assert acc >= 0.85, acc


def test_reinforce_beats_random():
    env = AgrarianToMEnv()
    policy = train_reinforce(env, episodes=1500)
    acc = evaluate_policy(policy, env)["accuracy"]
    assert acc >= 0.35, acc  # above 25% chance


def test_cch_rule_engine():
    path = Path(__file__).parent.parent / "evals/scientific/causal_crop_hypothesis/data/items.jsonl"
    items = load_items(path)
    correct = sum(1 for it in items if predict(it) == it.label)
    assert correct / len(items) >= 0.75


def test_rabi_probe_learns():
    out = run_probe()
    assert out["test_accuracy"] >= 0.55
