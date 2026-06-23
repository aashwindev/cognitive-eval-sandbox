#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== MonsoonBelief-Rev ==="
python -m evals.tom.monsoon_belief_revision.runner --mock

echo "=== Thought Tracing comparison ==="
python -m evals.tom.thought_tracing_adapter --compare-mbr

echo "=== Causal Crop Hypothesis ==="
python -m evals.scientific.causal_crop_hypothesis.runner --mock

echo "=== Rabi crop probe ==="
python -m geo.tasks.rabi_crop_probe --synthetic --compare-modalities

echo "Done. Results in experiments/results/"
