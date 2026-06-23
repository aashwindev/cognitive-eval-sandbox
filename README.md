# cognitive-eval-sandbox

**Public research sandbox: theory-of-mind as RL under partial observability + scientific reasoning evals.**

[![Public](https://img.shields.io/badge/repo-public-green)](https://github.com/aashwindev/cognitive-eval-sandbox)

This repo treats **social cognition as a POMDP**: hidden agronomic world state, visibility-partitioned observations, and policies that map **belief states** to mental-state / behavior predictions. It ships **working code** — not mock stubs — with measured accuracies on custom benchmarks.

**Live results** (`bash scripts/run_all.sh`):

| Component | Method | Accuracy |
|-----------|--------|----------|
| MBR-32 ToM | Belief-state policy \(\pi(a \mid \tau)\) | **84%** |
| MBR-32 ToM | Supervised RL on \(\phi(\tau)\) | **100%** |
| MBR-32 ToM | REINFORCE policy gradient | **91%** |
| CCH-24 science | Causal rule engine | **100%** |
| Rabi probe | Linear classifier on embeddings | **100%** test |

## Core thesis

LLM ToM benchmarks often collapse to pattern matching. We instead:

1. **Formalize** agrarian coordination as a POMDP with belief-state reduction ([`docs/rl-belief-formalism.md`](docs/rl-belief-formalism.md))
2. **Implement** a discrete belief tracker (visibility masks + emission tags) — structurally analogous to Thought Tracing's SMC particles (Kim et al., COLM 2025) but **trainable**
3. **Train** softmax policies with supervised learning and **REINFORCE** on `AgrarianToMEnv`
4. **Evaluate** on MonsoonBelief-Rev (MBR-32), a custom benchmark targeting the SimpleToM gap (explicit mental state ≠ applied behavior)

Parallel track: **model-based scientific reasoning** (causal Horn clauses over agronomic claims) and **representation learning** (TerraFM-style embeddings × Bhoonidhi ISRO data plane).

## Repository map

```
cognitive-eval-sandbox/
├── rl/                          # ★ POMDP + belief tracker + REINFORCE
│   ├── belief_tracker.py        #   sufficient statistic τ, visibility updates
│   ├── agrarian_tom_env.py      #   Gym-style ToM POMDP
│   ├── features.py              #   φ(τ, q) feature map
│   └── train.py                 #   supervised + REINFORCE
├── evals/tom/monsoon_belief_revision/   # MBR-32 custom ToM benchmark
├── evals/scientific/causal_crop_hypothesis/  # CCH-24 + rule engine
├── geo/                         # TerraFM embed + Bhoonidhi STAC
└── docs/                        # RL formalism + research landscape
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Full pipeline (RL train + all evals)
bash scripts/run_all.sh

# Individual runs
python -m rl.train --method both                    # supervised 100%, REINFORCE ~91%
python -m evals.tom.monsoon_belief_revision.runner  # belief policy 84%
python -m evals.tom.thought_tracing_adapter --compare-mbr --trace mbr_001
python -m evals.scientific.causal_crop_hypothesis.runner
python -m geo.tasks.rabi_crop_probe --compare-modalities
```

## RL / AI research depth

### POMDP formulation

Hidden state \(s\): soil moisture, sow viability, agent latent beliefs.  
Observations \(o_t\): instruments, speech, actions — **partitioned by visibility** (agronomist sees probe; lead may not).  
Evaluator policy \(\pi(a \mid \tau_t)\) acts on belief statistic \(\tau_t\) extracted by `BeliefTracker`.

See [`docs/rl-belief-formalism.md`](docs/rl-belief-formalism.md) for:

- Belief-state MDP reduction (Kaelbling et al., 1998)
- Connection to Thought Tracing as approximate Bayes without weight learning
- Dec-POMDP / multi-agent RL outlook
- Causal scientific reasoning as model-based RL

### Three policy classes

| Policy | Training | Role |
|--------|----------|------|
| `BeliefPolicy` | Hand-specified scoring over \(\tau\) | Interpretable upper bound |
| Supervised softmax | Logistic regression on \(\phi(\tau)\) | **Learns optimal MCQ mapper** |
| `REINFORCEPolicy` | Policy gradient, sparse \(\{0,1\}\) reward | **True RL** — 91% vs 21% random |

### Thought Tracing relationship

We do **not** fork [thought-tracing](https://github.com/skywalker023/thought-tracing). Our discrete particle filter over \(\tau\) is the auditable analogue of their NL hypothesis bank. Set `THOUGHT_TRACING_ROOT` to run upstream ToMi harnesses side-by-side.

## Benchmarks

### MonsoonBelief-Rev (MBR-32)

8 agrarian scenarios × 4 questions — belief revision under partitioned instruments (district bulletin vs field probe vs SAR). Targets **applied ToM** per [SimpleToM](https://arxiv.org/html/2410.13648v1).

### Causal Crop Hypothesis (CCH-24)

24 agronomic \((obs, intervention, claim)\) triples with `{supported, refuted, underdetermined}` labels. `rule_engine.py` implements falsification-aware Horn clauses.

### Geo: TerraFM × Bhoonidhi

ISRO [Bhoonidhi](https://bhoonidhi.nrsc.gov.in/) regional Sentinel hub + TerraFM embedding ablation for rabi crop-stage classification.

## Research docs

- [`docs/rl-belief-formalism.md`](docs/rl-belief-formalism.md) — **POMDP / belief-state RL / Thought Tracing bridge**
- [`docs/research-landscape.md`](docs/research-landscape.md) — ToM + scientific reasoning survey 2024–2026
- [`docs/research-agenda.md`](docs/research-agenda.md) — open problems (Dec-POMDP, offline RL on advisories)
- [`docs/thought-tracing-integration.md`](docs/thought-tracing-integration.md)
- [`docs/terra-isro-bridge.md`](docs/terra-isro-bridge.md)

## Citation

```bibtex
@misc{cognitive_eval_sandbox2025,
  title  = {cognitive-eval-sandbox: ToM as Belief-State RL + Scientific Reasoning Evals},
  author = {Aashwin},
  year   = {2025},
  url    = {https://github.com/aashwindev/cognitive-eval-sandbox}
}
```

## License

MIT
