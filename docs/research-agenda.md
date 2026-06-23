# Research agenda — AI, RL, and cognitive eval

## Near-term (implemented in this repo)

- [x] POMDP formalization of agrarian ToM (`docs/rl-belief-formalism.md`)
- [x] Discrete belief tracker with visibility masks (`rl/belief_tracker.py`)
- [x] Supervised + REINFORCE policies on MBR-32 (`rl/train.py`)
- [x] Model-based causal checker for scientific claims (`rule_engine.py`)
- [x] Linear probe on multisensor embeddings (`geo/tasks/rabi_crop_probe.py`)

## Medium-term

### Dec-POMDP multi-agent RL

MBR scenarios are **decentralized POMDPs**: each agent runs an independent belief update. Train cooperative policies with:

- CTDE (MADDPG / MAPPO) where critics see global state during training
- Reward = successful sowing window coordination minus social cost of officer pressure

### Offline RL on advisory outcomes

Bhoonidhi + Bhuvan crop advisories generate logged \((context, action, outcome)\). Fit conservative Q-learning (CQL) or behavior cloning with uncertainty to improve recommendation policies without online exploration on farmers.

### LLM + belief-state distillation

Use `BeliefTracker` traces as **supervision signal** for fine-tuning LLMs:

1. Run discrete SMC on MBR-32
2. Distill \(\tau_t \rightarrow\) answer into LoRA weights
3. Compare vs Thought Tracing NL particles on FANToM/OpenToM

### Reward modeling for reasoning traces

Integrate TRM-style ME² rewards on chain-of-thought before causal claim classification — score *how* the model reasons, not just the label.

## Long-term

- **Level-2 ToM** items: "What does the agronomist believe the lead believes?"
- **Active perception RL**: choose which instrument to query (probe vs satellite pass) under budget
- **Sim-to-real**: TerraFM embeddings from live Bhoonidhi STAC → deploy probe in NRSC crop monitoring pipeline
- **Human calibration study**: correlate \(\tau\) confidence with agronomist inter-rater agreement

## Why RL matters here

SimpleToM shows LLMs fail **applied** ToM despite explicit success. That is consistent with no learned **belief-state policy** — models answer questions without maintaining \(\tau_t\) across events. REINFORCE on `AgrarianToMEnv` demonstrates that even a linear policy on \(\phi(\tau)\) learns the structure (91–100%), supporting the hypothesis that **architecture/training**, not scale alone, drives social reasoning.
