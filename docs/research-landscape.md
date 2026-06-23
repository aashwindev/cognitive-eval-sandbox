# Research landscape: Theory of Mind & scientific reasoning (2024–2026)

*Memo for cognitive-eval-sandbox — last updated June 2025.*

## 1. The fragmentation problem

Cognitive evaluation for LLMs has forked into:

- **Social cognition benchmarks** (ToMi, FANToM, BigToM, OpenToM, ToMBench, CogToM, MOMENTS)
- **Scientific reasoning suites** (ARC, ScienceQA, GPQA, SciBench, LabBench)

These communities rarely cross-pollinate, yet real scientific practice *is* a theory-of-mind problem: researchers hold beliefs about hidden mechanisms, update on noisy instruments, and coordinate with peers who see different evidence.

This sandbox exists at that intersection.

## 2. Theory of Mind: what changed since 2023

### 2.1 From false-belief tests to applied ToM

Classic Sally–Anne style tasks saturated quickly. **SimpleToM** (Gandhi et al., 2024) decomposed performance into:

| Subskill | Example question | GPT-4o accuracy (paper) |
|----------|------------------|-------------------------|
| Mental state (explicit) | "What does Alice believe?" | 95.6% |
| Behavior (applied) | "What will Alice do next?" | 49.5% |
| Social judgment (applied) | "Was Bob's action reasonable?" | 15.3% |

**Takeaway:** High explicit ToM scores are insufficient for deployment in social/scientific settings. MonsoonBelief-Rev weights *behavior* and *belief-revision* items 2:1.

### 2.2 Benchmark contamination & scale

**ToMBench** (Ma et al., ACL 2024) built 2,860 bilingual MCQs from scratch to avoid leakage. **CogToM** (ACL 2026) expands to 46 paradigms / 8k instances grounded in human cognitive science.

Our approach: *small-N, high-density* scenarios (32 items) with full provenance JSON — easier to audit than scaling MCQ factories.

### 2.3 Inference-time mental-state tracking

**Thought Tracing** (Kim et al., COLM 2025) treats ToM as sequential hypothesis inference:

1. Maintain a particle bank of natural-language mental-state hypotheses \(h_i\)
2. Weight by observation likelihood \(w_i \propto P(o \mid h_i)\) using the LLM as likelihood oracle
3. Resample / mutate hypotheses as new dialogue observations arrive
4. Answer downstream questions by marginalizing over the bank

This is structurally identical to Bayesian Theory of Mind (BToM; Zhi-Xuan et al., 2022) but swaps symbolic generative models for LLM proposals.

We implement a **minimal tracer** (`thought_tracing_adapter.py`) to compare single-shot vs hypothesis-bank accuracy on MBR-32 without requiring GPU flash-attn stacks from upstream.

### 2.4 Reasoning models behave differently on social tasks

Kim et al. report that o3/R1-class models show *non-monotonic* gains on ToM vs math — sometimes worse than GPT-4o on belief-tracking despite superior chain-of-thought on GSM8K. This supports evaluating **domain-specific** inference algorithms rather than assuming one CoT template generalizes.

## 3. Scientific reasoning: beyond answer correctness

### 3.1 TRM and "good ways to think"

The **Thinking Reward Model** (TRM; 2025) scores reasoning *traces* along ME² axes:

- Macro-Efficiency / Macro-Effectiveness (global structure)
- Micro-Efficiency / Micro-Effectiveness (local steps)

Scientific hypothesis evaluation needs analogous structure: a model can state the correct crop stage while citing irrelevant NDVI trivia (high answer, low reasoning quality).

### 3.2 Causal crop hypothesis eval (CCH-24)

Our `causal_crop_hypothesis` benchmark presents triples:

```
(premise observations, intervention, claimed outcome)
```

Models must label: `{supported, refuted, underdetermined}` and cite which observation *would* falsify the claim. Items are grounded in rabi-season agronomy (wheat/chickpea) consistent with ISRO crop monitoring products on Bhuvan.

This mirrors the **hypothesis-driven** framing of Thought Tracing but for physical causation instead of mental states.

## 4. Geospatial thread: TerraFM + ISRO data plane

### 4.1 TerraFM (ICLR 2026)

Key design choices relevant to our probe task:

| Mechanism | Role |
|-----------|------|
| Modality-specific patch embeddings | Handles 12-channel S2 + 2-channel S1 without RGB collapse |
| Cross-attention fusion | Spatially selective sensor weighting |
| Dual-centering (WorldCover priors) | Mitigates long-tail land-cover bias |

Pretraining: 18.7M tiles from Major-TOM; benchmarks on GEO-Bench / Copernicus-Bench.

### 4.2 ISRO-adjacent data (Bhoonidhi)

NRSC operates a **regional Sentinel hub** for India:

- Sentinel-1 & 2 archived from 2019 with ~30 min latency vs SciHub
- STAC-compliant search API ([Bhoonidhi API spec](https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/))
- Open direct-download collections for research users

**Why not train TerraFM here?** Weekend scope. We instead:

1. Document the STAC integration path
2. Run a **linear probe** on frozen embeddings for `{pre-sowing, vegetative, harvest}`
3. Compare S2-only vs S1+S2 fused embeddings (scientific claim: SAR helps under cloud)

### 4.3 Connection to ToM track

MonsoonBelief-Rev scenarios reference the *same* observables (district drought bulletin, field moisture probe, delayed satellite pass) that the geo track manipulates. A future unified agent would:

- maintain mental-state hypotheses about cooperative sowing decisions, and
- ground those hypotheses in embedder confidence from TerraFM tiles.

## 5. Evaluation protocol in this repo

| Track | Metric | Baselines |
|-------|--------|-----------|
| MBR-32 | accuracy on behavior + revision items | single-shot, CoT, tracer (K=8) |
| CCH-24 | macro-F1 on causal labels | single-shot, CoT |
| Rabi probe | top-1 stage accuracy | random, S2-only, S1+S2 |

All runners support `--mock` for CI-friendly deterministic baselines.

## 6. Open questions (research backlog)

1. **Calibration:** Do hypothesis weights from Thought Tracing correlate with human confidence on MBR-32?
2. **Modality ablation:** Does SAR fusion improve both EO probe *and* LLM scientific reasoning when SAR availability is explicit in the prompt?
3. **Cross-cultural ToM:** Agrarian scenarios introduce caste/cooperative power asymmetries — do models default to stereotyped "rational actor" updates?
4. **Instrument latency:** Belief-revision items with delayed satellite passes test whether models understand *information arrival* vs *world state change*.

## References

- Kim, H. et al. (2025). Hypothesis-Driven Theory-of-Mind Reasoning for LLMs. COLM.
- Gandhi, S. et al. (2024). SimpleToM. arXiv:2410.13648.
- Xu, H. et al. (2024). OpenToM. ACL 2024.
- Ma, Z. et al. (2024). ToMBench. ACL 2024.
- Danish, M.S. et al. (2025). TerraFM. ICLR 2026.
- Tong, H. et al. (2026). CogToM. ACL 2026.
- Zhi-Xuan, T. et al. (2022). Bayesian Theory of Mind. AAMAS.
