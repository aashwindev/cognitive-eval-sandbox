"""Supervised and RL policy training for Agrarian ToM POMDP."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.agrarian_tom_env import AgrarianToMEnv
from rl.features import FEATURE_DIM, extract_features


@dataclass
class SoftmaxPolicy:
    weights: np.ndarray  # (FEATURE_DIM, n_actions)

    def probs(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights
        logits = logits - logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def act(self, x: np.ndarray, rng: np.random.Generator) -> int:
        p = self.probs(x)
        return int(rng.choice(len(p), p=p))

    def greedy(self, x: np.ndarray) -> int:
        return int(np.argmax(self.probs(x)))


def train_supervised(env: AgrarianToMEnv, lr: float = 0.5, epochs: int = 400) -> SoftmaxPolicy:
    """Multiclass logistic regression — learns π(a|φ(τ,q)) with full observability of φ."""
    n_actions = 4
    w = np.zeros((FEATURE_DIM, n_actions), dtype=np.float64)
    episodes = env.iter_all()

    for _ in range(epochs):
        for ep in episodes:
            x = extract_features(ep.scenario, ep.question).astype(np.float64)
            y = ep.question.answer_index
            p = SoftmaxPolicy(w).probs(x)
            grad = p.copy()
            grad[y] -= 1.0
            w -= lr * np.outer(x, grad) / len(episodes)

    return SoftmaxPolicy(w.astype(np.float32))


def train_reinforce(
    env: AgrarianToMEnv,
    episodes: int = 800,
    lr: float = 0.05,
    seed: int = 0,
) -> SoftmaxPolicy:
    """REINFORCE with moving-average baseline."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((FEATURE_DIM, env.action_space_n)).astype(np.float32) * 0.01
    policy = SoftmaxPolicy(w)
    baseline = 0.5
    baseline_decay = 0.95

    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
        ep = env._current
        assert ep is not None
        action = policy.act(obs, rng)
        _, reward, _, _, _ = env.step(action)

        p = policy.probs(obs)
        advantage = reward - baseline
        baseline = baseline_decay * baseline + (1 - baseline_decay) * reward

        grad_logits = -p.copy()
        grad_logits[action] += 1.0
        w += lr * advantage * np.outer(obs.astype(np.float32), grad_logits)
        policy = SoftmaxPolicy(w)

    return policy


def evaluate_policy(policy: SoftmaxPolicy, env: AgrarianToMEnv) -> dict:
    correct = 0
    total = 0
    for ep in env.iter_all():
        x = extract_features(ep.scenario, ep.question)
        pred = policy.greedy(x)
        correct += int(pred == ep.question.answer_index)
        total += 1
    return {"accuracy": correct / total, "n": total, "correct": correct}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ToM policies (supervised + REINFORCE)")
    parser.add_argument("--method", choices=["supervised", "reinforce", "both"], default="both")
    parser.add_argument("--reinforce-episodes", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/rl_training.json"))
    args = parser.parse_args()

    env = AgrarianToMEnv()
    results: dict = {}

    if args.method in ("supervised", "both"):
        sup = train_supervised(env)
        results["supervised"] = evaluate_policy(sup, env)

    if args.method in ("reinforce", "both"):
        rng = np.random.default_rng(0)
        chance_correct = 0
        for _ in range(200):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            a = int(rng.integers(0, 4))
            _, r, _, _, _ = env.step(a)
            chance_correct += int(r)
        results["random_baseline"] = {"accuracy": chance_correct / 200, "n": 200}

        rf = train_reinforce(env, episodes=args.reinforce_episodes)
        results["reinforce"] = evaluate_policy(rf, env)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    for k, v in results.items():
        if "accuracy" in v:
            print(f"{k}: {v['accuracy']:.2%} (n={v.get('n', '?')})")


if __name__ == "__main__":
    main()
