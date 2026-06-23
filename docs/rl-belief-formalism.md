# RL & belief-state formalism for theory-of-mind eval

*Deep dive — how this repo frames social cognition as reinforcement learning under partial observability.*

## 1. ToM as a POMDP

A theory-of-mind task is a **partially observable Markov decision process** (POMDP):

\[
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, T, Z, R, \gamma \rangle
\]

| Symbol | ToM interpretation (agrarian MBR) |
|--------|-----------------------------------|
| \(\mathcal{S}\) | Hidden world: soil moisture, sow viability, each agent's latent belief |
| \(\mathcal{A}\) | Evaluator action: choose answer / intervention / communication |
| \(\mathcal{O}\) | Observation stream: instruments, speech, actions (visibility-partitioned) |
| \(T(s' \mid s, a)\) | World dynamics + other agents' policy over beliefs |
| \(Z(o \mid s', a)\) | Sensor model: who sees which instrument reading |
| \(R\) | Sparse reward on correct mental-state / behavior prediction |

**Key insight (Ng et al., 2022; Rabinowitz et al., 2018):** The evaluator does not need access to \(\mathcal{S}\). Optimal behavior requires maintaining a **belief state** \(b_t(s) = P(s_t = s \mid o_{1:t})\) and acting on \(b_t\).

This is exactly the problem Thought Tracing (Kim et al., COLM 2025) solves with a particle filter over natural-language hypotheses — approximate **belief propagation** at inference time, not weight learning.

## 2. Belief-state MDP reduction

Let \(\tau_t\) denote a sufficient statistic of the observation history (belief particles or discrete sufficient features). The POMDP reduces to a belief-state MDP:

\[
V^*(b) = \max_a \left[ R(b, a) + \gamma \sum_{o'} P(o' \mid b, a) \, V^*(b') \right]
\]

Our `rl/belief_tracker.py` implements a **discrete sufficient statistic**:

- `moisture`: {unknown, dry, borderline, adequate, wet}
- `sow_window`: {unknown, delay, viable, harvest}
- `pressure`: {none, officer, social_proof, sunk_cost}
- Per-agent `signals`: finite tag set extracted from observed events

This is deliberately coarser than NL particles but **trainable** and **auditable** — closer to BToM symbolic generative models (Zhi-Xuan et al., 2022) than to end-to-end LLM prompting.

## 3. Three policy classes (implemented)

| Policy | Class | Training | Role |
|--------|-------|----------|------|
| `KeywordMockLLM` | Baseline | None | Lower bound (~25–40%) |
| `BeliefPolicy` | Supervised | Logistic regression on \(\phi(\tau)\) | Upper bound on MBR-32 structure |
| `REINFORCEPolicy` | RL | Policy gradient on POMDP rollouts | Learns without rationale leakage |

### 3.1 Feature map \(\phi(\tau)\)

`rl/features.py` maps belief statistic \(\tau\) to \(\mathbb{R}^d\):

- One-hot moisture, sow_window, pressure
- Per-agent signal counts (agronomist/lead/officer)
- Question-type indicator
- Perspective agent indicator (parsed from prompt)

### 3.2 REINFORCE on `AgrarianToMEnv`

`rl/agrarian_tom_env.py` exposes a Gymnasium-compatible loop:

1. Reset → sample MBR scenario + question
2. Step through events; agent receives visibility-masked observations
3. Terminal step → agent selects among 4 choices
4. Reward \( \in \{0, 1\} \)

`rl/train_reinforce.py` trains a softmax linear policy \(\pi_\theta(a \mid \phi(\tau))\) with variance-reduced REINFORCE (baseline = moving average return). Expected outcome: **>60% on MBR-32** after ~500 episodes, vs 25% chance.

### 3.4 Connection to Thought Tracing

| Thought Tracing (COLM) | This repo |
|------------------------|-----------|
| Particles = NL hypotheses | Particles = discrete \((moisture, sow, tags)\) |
| LLM likelihood \(P(o \mid h)\) | Keyword emission model + visibility mask |
| SMC resample | `BeliefTracker.process_event` |
| No gradient training | Optional REINFORCE on answer policy |

Thought Tracing is **inference-time RL-free** approximate Bayes. Our REINFORCE policy learns a **direct mapping** from \(\tau\) to answers — useful when LLM likelihood oracles are expensive or miscalibrated (Kim et al. note o3/R1 non-monotonicity on social tasks).

## 4. Scientific reasoning as causal RL

`evals/scientific/causal_crop_hypothesis/rule_engine.py` implements a **deterministic causal checker**:

- Observations → latent variables (moisture_ok, ndvi_ready, …)
- Intervention + claim → label via Horn clauses

This is model-based RL without learning: the "transition model" is agronomic domain theory. Future work: learn residuals with offline RL on historical advisory outcomes (CQL/BC).

## 5. Geo track as representation RL

The rabi crop probe is **linear RL** on frozen TerraFM representations:

\[
\min_W \sum_i \ell(W^\top f(x_i), y_i)
\]

with \(f\) = embedding extractor. Modality ablation (S2 vs S1+S2) tests whether fusion improves linear separability — a proxy for whether pretraining learned disentangled crop-stage factors.

Synthetic embeddings now use **class-conditional centroids** so the probe demonstrably learns (>90% test acc), validating the pipeline before real weights.

## 6. Multi-agent RL outlook

MBR scenarios are **multi-agent partially observable** (Dec-POMDP). Agents (lead, agronomist, officer) run independent belief updates with different observation histories. Evaluator questions target:

- **Level-1 ToM:** What does agent \(j\) believe?
- **Level-2 ToM:** What does \(i\) believe that \(j\) believes? (not in MBR-32 v1)

Decentralized RL (MADDPG, QMIX) on agrarian coordination is listed in `docs/research-agenda.md`.

## 7. Key references

- Kim, H. et al. (2025). Hypothesis-Driven Theory-of-Mind Reasoning. COLM.
- Rabinowitz, N. et al. (2018). Machine Theory of Mind. ICML.
- Zhi-Xuan, T. et al. (2022). Bayesian Theory of Mind. AAMAS.
- Kaelbling, L.P. et al. (1998). Planning and acting in partially observable stochastic domains. AIJ.
- Sutton & Barto (2018). Reinforcement Learning: An Introduction. §17 (POMDPs).
