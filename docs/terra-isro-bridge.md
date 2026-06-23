# TerraFM × Bhoonidhi bridge

Geospatial scientific reasoning track — how frozen EO embeddings connect to ISRO's data plane.

## Data sources

### Bhoonidhi (NRSC/ISRO)

[Bhoonidhi](https://bhoonidhi.nrsc.gov.in/) is ISRO's open/priced EO data hub:

- **47 satellite sensors** in catalogue (IRS family + foreign missions)
- **Regional Sentinel-1/2 distribution** for Indian subcontinent
- **STAC API** for programmatic search ([spec](https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/))

Relevant collections for TerraFM-compatible tiles:

| Collection | Modality | Use in sandbox |
|------------|----------|----------------|
| `sentinel2_l2a` | Optical 13-band | Primary vegetation signal |
| `sentinel1_rtc` | SAR backscatter | Cloud-penetrating structure |

Authentication: `POST /auth/token` → Bearer header on search/download endpoints.

Our `geo/bhoonidhi_stac.py` implements search scaffolding; live downloads require registered credentials (not bundled).

### Major-TOM / TerraFM weights

[TerraFM](https://github.com/mbzuai-oryx/TerraFM) (MBZUAI ORYX, ICLR 2026) pretrained on Major-TOM tiles:

- ViT-Base/Large backbones
- Inputs: S1 RTC + S2 L1C + S2 L2A (12 effective channels at 224×224)
- Weights: `TerraFM-B.pth` from HuggingFace `MBZUAI/TerraFM`

## Downstream task: Rabi crop stage probe

**Scientific question:** Given a frozen TerraFM embedding of a winter (rabi) field tile in semi-arid India, can a linear classifier recover growth stage?

**Stages:**

1. `pre_sowing` — bare/prepped soil, low NDVI
2. `vegetative` — active canopy development
3. `harvest` — senescent / stubble / bare post-harvest

**Why rabi?** ISRO's crop monitoring services emphasize rabi wheat and chickpea in central India ([Bhuvan crop portal](https://bhuvan.nrsc.gov.in/)). Aligns eval narrative with national EO application context.

## Expected embedding ablation

| Configuration | Hypothesis |
|---------------|------------|
| S2-only | Strong vegetative vs harvest separation via NDVI-like channels |
| S1+S2 fused | Gains on pre-sowing under cloud (monsoon tail / winter haze) |
| Random init ViT | Baseline — should not beat chance |

Run:

```bash
python -m geo.tasks.rabi_crop_probe --synthetic  # no weights
python -m geo.tasks.rabi_crop_probe --weights weights/TerraFM-B.pth
```

## Synthetic fallback

Without GPU weights, `terrafm_embed.py` emits deterministic pseudo-embeddings seeded by tile metadata so the probe pipeline remains testable in CI.

## Linking to LLM scientific eval

`causal_crop_hypothesis` items reference observables that map to STAC metadata fields:

- `cloud_cover_pct` → S2 usability
- `vv_backscatter_delta` → soil moisture proxy from S1
- `ndvi_14d_trend` → crop vigor

A unified agent could condition LLM hypotheses on embedder confidence — left as future work (see `docs/research-landscape.md` §5).

## India-specific EO context

ISRO operates **Resourcesat / RISAT** alongside foreign Sentinels. Bhoonidhi federates both. TerraFM was trained globally (Major-TOM), not India-only — the probe tests **transfer** to subcontinental rabi agriculture, which is the realistic deployment scenario for NRSC analytics pipelines.

## References

- Danish, M.S. et al. (2025). TerraFM. arXiv:2506.06281.
- ISRO NRSC. Bhoonidhi API Specification v2.
- ISRO. Space Based Earth Observation Applications. isro.gov.in.
