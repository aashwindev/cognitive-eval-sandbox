# cognitive-eval-sandbox

**Evals for theory-of-mind + scientific reasoning — experiments.**

A weekend-scale research sandbox that connects two threads I care about:

1. **Social cognition** — hypothesis-driven mental-state tracking (inspired by [Thought Tracing](https://github.com/skywalker023/thought-tracing), COLM 2025), extended with a custom benchmark for *belief revision under partial observability*.
2. **Geospatial scientific reasoning** — TerraFM-style multisensor embeddings on ISRO-adjacent Sentinel data (Bhoonidhi regional hub), with a minimal downstream crop-stage probe.

This is not a fork pin. It is an original integration layer: adapters, one custom ToM eval, one causal-science eval, and a reproducible EO pipeline sketch grounded in public ISRO/NRSC infrastructure.

## Why these two tracks together?

Theory-of-mind and scientific reasoning are usually evaluated in isolation. Recent work shows why that split is misleading:

| Finding | Source | Implication for this repo |
|--------|--------|---------------------------|
| LLMs ace explicit mental-state questions but fail behavior prediction | [SimpleToM](https://arxiv.org/html/2410.13648v1) (2024) | Our custom eval tests *applied* ToM, not questionnaire ToM |
| Inference-time hypothesis propagation beats single-shot CoT on ToMi/FANToM/BigToM | [Thought Tracing](https://arxiv.org/html/2502.11881) (COLM 2025) | We ship a lightweight tracer adapter, not a full reimplementation |
| Frontier models track physical-world beliefs better than psychological states | [OpenToM](https://aclanthology.org/2024.acl-long.466/) (ACL 2024) | MonsoonBelief-Rev targets *epistemic* vs *instrumental* belief updates |
| Multisensor EO FMs generalize when modalities are treated as views | [TerraFM](https://arxiv.org/html/2506.06281) (ICLR 2026) | Our geo track uses S1+S2 fusion semantics, Bhoonidhi STAC for India tiles |

## Repository map

```
cognitive-eval-sandbox/
├── docs/                    # research memos (landscape, integration notes)
├── evals/tom/               # theory-of-mind benchmarks + ThoughtTracing adapter
│   └── monsoon_belief_revision/   # ★ custom eval (MBR-32)
├── evals/scientific/        # causal / hypothesis-testing evals
│   └── causal_crop_hypothesis/    # ★ scientific reasoning over agronomic claims
├── geo/                     # TerraFM embedding + Bhoonidhi STAC bridge
│   └── tasks/rabi_crop_probe.py
└── scripts/                 # one-command runners
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Custom ToM eval (mock LLM — no API key required)
python -m evals.tom.monsoon_belief_revision.runner --mock

# Scientific reasoning eval
python -m evals.scientific.causal_crop_hypothesis.runner --mock

# Geo downstream probe (synthetic tiles if TerraFM weights absent)
python -m geo.tasks.rabi_crop_probe --synthetic
```

## Custom eval: MonsoonBelief-Rev (MBR-32)

**Problem class:** An agronomist and a village cooperative agent observe monsoon rainfall with *different information partitions* (drought bulletin vs on-field moisture probe). Questions require:

- tracking each agent's belief about sowing window,
- predicting *whose action changes* when a new observation arrives,
- distinguishing epistemic updates from preference-driven decisions.

This directly targets the SimpleToM gap (mental state ≠ behavior) in a domain where scientific instrumentation and social coordination interact — common in ISRO/NRSC crop advisory workflows ([Bhuvan crop services](https://bhuvan.nrsc.gov.in/)).

See [`evals/tom/monsoon_belief_revision/README.md`](evals/tom/monsoon_belief_revision/README.md).

## Geo track: TerraFM × Bhoonidhi

ISRO's [Bhoonidhi](https://bhoonidhi.nrsc.gov.in/) hosts a **regional Sentinel-1/2 hub** (30-minute latency vs Copernicus SciHub for the Indian subcontinent). We provide:

- `geo/bhoonidhi_stac.py` — STAC search scaffold (collection-aware, auth-ready)
- `geo/terrafm_embed.py` — TerraFM-B embedding extractor (optional weights)
- `geo/tasks/rabi_crop_probe.py` — linear probe: embedding → {pre-sowing, vegetative, harvest}

This is a *downstream scientific task* parallel to the LLM evals: can multisensor representations support falsifiable agronomic stage classification on Indian rabi-season tiles?

## Relationship to thought-tracing

We **do not vendor** the upstream repo. Instead:

- `evals/tom/thought_tracing_adapter.py` implements the SMC-style hypothesis bank interface Kim et al. describe (generate → weight → resample → answer).
- Point `THOUGHT_TRACING_ROOT` at a local clone to run their ToMi harness side-by-side.

```bash
git clone https://github.com/skywalker023/thought-tracing.git ../thought-tracing
export THOUGHT_TRACING_ROOT=../thought-tracing
python -m evals.tom.thought_tracing_adapter --compare-mbr
```

## Research docs

- [`docs/research-landscape.md`](docs/research-landscape.md) — ToM + scientific reasoning benchmark survey (2024–2026)
- [`docs/thought-tracing-integration.md`](docs/thought-tracing-integration.md) — adapter design vs COLM algorithm
- [`docs/terra-isro-bridge.md`](docs/terra-isro-bridge.md) — TerraFM modalities, Bhoonidhi STAC, India-specific EO context

## Citation

If you use MonsoonBelief-Rev or the integration notes, cite this sandbox and the upstream works it builds on:

```bibtex
@misc{cognitive_eval_sandbox2025,
  title  = {cognitive-eval-sandbox: Theory-of-Mind and Scientific Reasoning Experiments},
  author = {Aashwin},
  year   = {2025},
  url    = {https://github.com/aashw/cognitive-eval-sandbox}
}
```

## License

MIT — see [LICENSE](LICENSE).
