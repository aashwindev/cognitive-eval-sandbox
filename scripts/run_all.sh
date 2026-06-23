#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== RL training (supervised + REINFORCE) ==="
python -m rl.train --method both

echo "=== MonsoonBelief-Rev (belief-state policy) ==="
python -m evals.tom.monsoon_belief_revision.runner

echo "=== Belief trace demo ==="
python -m evals.tom.thought_tracing_adapter --compare-mbr

echo "=== Causal Crop Hypothesis (rule engine) ==="
python -m evals.scientific.causal_crop_hypothesis.runner

echo "=== Rabi crop probe ==="
python -m geo.tasks.rabi_crop_probe --synthetic --compare-modalities

echo "Done. Results in experiments/results/"
