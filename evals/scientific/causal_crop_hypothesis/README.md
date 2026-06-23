# Causal Crop Hypothesis (CCH-24)

Scientific reasoning eval: classify agronomic causal claims given observations + intervention.

## Labels

| Label | Meaning |
|-------|---------|
| `supported` | Claim follows from observations under standard agronomy |
| `refuted` | Claim contradicted by at least one observation |
| `underdetermined` | Insufficient evidence; additional measurement needed |

Each item includes a **falsifier** (what observation would settle the claim) to score explanation quality in future work.

## Items

24 hand-authored items in `data/items.jsonl` covering:

- NDVI trends vs sowing decisions
- SAR backscatter vs soil moisture inference
- Irrigation interventions under heat stress
- Stubble management vs potato emergence

Grounded in rabi/kharif-tail contexts aligned with ISRO crop monitoring use cases.

## Run

```bash
python -m evals.scientific.causal_crop_hypothesis.runner --mock
```
