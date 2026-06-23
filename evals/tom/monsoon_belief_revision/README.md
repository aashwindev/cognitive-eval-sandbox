# MonsoonBelief-Rev (MBR-32)

Custom theory-of-mind benchmark for **belief revision under partial observability** in Indian monsoon/rabi agrarian coordination.

## Motivation

SimpleToM (2024) showed frontier LLMs score >95% on explicit mental-state questions but <50% on behavior prediction. Standard ToMi/BigToM stories lack:

- **instrument latency** (satellite pass delayed vs ground truth),
- **partitioned evidence** (district bulletin vs field probe),
- **epistemic vs instrumental updates** (belief change vs preference-driven delay).

MBR-32 fills this gap with 8 fully-authored scenarios × 4 questions each (32 items), weighted toward applied ToM.

## Question mix

| Type | Count | Example |
|------|-------|---------|
| `mental_state` | 8 | "What does the cooperative lead believe about sowing window?" |
| `behavior` | 12 | "Who postpones the group sowing meeting?" |
| `belief_revision` | 12 | "After the moisture probe reading, whose belief changes?" |

## Scenario design principles

1. **Two or three agents** with different information partitions.
2. At least one **instrument observation** (`event_type=instrument`) with selective visibility.
3. **Behavior items** always follow a belief-relevant event within ≤2 steps.
4. Ground truth authored with agronomic consult (IARI rabi sowing windows, district drought nomenclature).

## Format

JSONL with one `Scenario` per line — see `data/samples.jsonl` for the canonical 8 scenarios. Schema in `schema.py`.

## Running

```bash
# Mock LLM (deterministic, no API key)
python -m evals.tom.monsoon_belief_revision.runner --mock

# With OpenAI-compatible API
export OPENAI_API_KEY=...
python -m evals.tom.monsoon_belief_revision.runner --model gpt-4o-mini

# Thought Tracing-style hypothesis bank
python -m evals.tom.monsoon_belief_revision.runner --mock --use-tracing
```

## Metrics

- Overall accuracy
- Per-question-type accuracy (expect gap on `behavior` per SimpleToM)
- Tracer lift: `acc_tracing - acc_single_shot`

## Relation to Thought Tracing

Compatible with `evals/tom/thought_tracing_adapter.py` — scenarios expose `events` as observations for hypothesis weighting.

## Citation

```bibtex
@misc{mbr32_2025,
  title  = {MonsoonBelief-Rev: Belief Revision ToM for Agrarian Coordination},
  author = {Aashwin},
  year   = {2025},
  note   = {Part of cognitive-eval-sandbox}
}
```
