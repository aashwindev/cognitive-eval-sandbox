# Thought Tracing integration notes

This document explains how `evals/tom/thought_tracing_adapter.py` relates to the official [thought-tracing](https://github.com/skywalker023/thought-tracing) repository without forking it.

## Algorithm sketch (Kim et al., COLM 2025)

Thought Tracing approximates:

\[
P(h_t \mid o_{1:t}) \propto P(h_t \mid h_{t-1}, a_{t-1}) \cdot P(o_t \mid h_t)
\]

where:

- \(h_t\) = natural-language hypothesis about agent mental state at step \(t\)
- \(o_t\) = observation (utterance, action, perception)
- LLM proposes \(h_t\) and scores \(P(o_t \mid h_t)\) via prompting

Implementation uses a particle filter with \(K\) hypotheses (default 8).

## Our minimal adapter

| Upstream component | Our equivalent | Notes |
|--------------------|----------------|-------|
| `Tracer` class | `HypothesisBank` | Same API surface: `propose`, `weight`, `resample` |
| Benchmark harnesses | MBR-32 runner | Custom agrarian scenarios |
| `flash-attn` local models | Optional OpenAI API | `--mock` for offline |

### Interface

```python
bank = HypothesisBank(k=8, model=llm)
for event in scenario.events:
    bank.observe(event.text)
    bank.resample()
answer = bank.query(scenario.question)
```

### Likelihood estimation modes

1. **`prompting`** — ask LLM "How likely is this observation if hypothesis H is true?" (0–10 scale)
2. **`contrast`** — compare relative likelihood vs a neutral hypothesis (cheaper, noisier)

Upstream also supports learned likelihood models; we defer that to keep dependencies light.

## Side-by-side evaluation

```bash
export THOUGHT_TRACING_ROOT=/path/to/thought-tracing  # optional
python -m evals.tom.thought_tracing_adapter --compare-mbr
```

When `THOUGHT_TRACING_ROOT` is set, the adapter attempts to import upstream `Tracer` for parity checks on shared ToMi-style items. MBR-32 items are exclusive to this repo.

## Design choice: why not fork?

1. **Upstream env is heavy** (`flash-attn`, multi-benchmark subtrees).
2. **Original contribution is MBR-32 + CCH-24**, not a tracer reimplementation.
3. **Adapter pattern** documents integration competence for profile/README purposes without maintenance burden of 4 benchmark forks.

## Extending

To add a new ToM eval compatible with Thought Tracing:

1. Define scenarios as `list[Observation]` with typed events (`perception`, `action`, `speech`).
2. Register questions with `requires_tracing: bool` flag.
3. Run `thought_tracing_adapter.run_eval(dataset, tracer_type="tracer")`.

See `monsoon_belief_revision/schema.py` for the canonical scenario format.
