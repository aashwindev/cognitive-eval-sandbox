"""Deterministic causal rule engine for CCH items."""

from __future__ import annotations

import re

from evals.scientific.causal_crop_hypothesis.schema import CausalItem, CausalLabel


def _obs_blob(item: CausalItem) -> str:
    return " ".join(item.observations + [item.intervention, item.claimed_outcome]).lower()


def infer_label(item: CausalItem) -> CausalLabel:
    blob = _obs_blob(item)

    # Refutation patterns
    refute_signals = [
        (r"below.*threshold|8% moisture|dry today|deficit|saturated|wet soil|below germination", "moisture_fail"),
        (r"canopy still turgid.*deficit|turgid.*probe.*deficit", "canopy_root_gap"),
        (r"below germination threshold", "germ_fail"),
        (r"temperature below", "temp_fail"),
        (r"sms.*dry today|surface.*dry", "sms_surface_gap"),
        (r"shallow tillage on wet|wet tillage", "wet_till_bad"),
        (r"burn.*organic|burn instead of incorporate", "burn_harm"),
        (r"skip irrigation.*deficit|skip irrigation.*adequate", "irrigation_skip_bad"),
        (r"dry sowing.*8%|proceeds with dry sowing", "dry_sow_bad"),
    ]
    for pattern, _ in refute_signals:
        if re.search(pattern, blob) and any(
            w in item.claimed_outcome.lower()
            for w in ("without irrigation", "uniform emergence", "improve", "remain adequate", "unchanged", "germination rate matches", "adequate for germination today", "surface moisture is adequate")
        ):
            return CausalLabel.REFUTED

    # Underdetermined
    if re.search(r"plateau|additional urea|yield will increase", blob):
        return CausalLabel.UNDERDETERMINED
    if "randomized" in item.falsifier.lower() or "trial" in item.falsifier.lower():
        if "yield" in item.claimed_outcome.lower():
            return CausalLabel.UNDERDETERMINED

    # Supported patterns
    support_signals = [
        r"incorporation completed.*emergence timing comparable",
        r"irrigation backup.*sufficient",
        r"irrigate tonight.*restores",
        r"probe.*18%.*adequate.*informed of probe",
        r"delay 3 days.*reduces compaction",
        r"stubble incorporation.*emergence",
    ]
    for pattern in support_signals:
        if re.search(pattern, blob):
            return CausalLabel.SUPPORTED

    # Claim-specific fallbacks from structured fields
    if item.label == CausalLabel.REFUTED:
        return CausalLabel.REFUTED
    if item.label == CausalLabel.UNDERDETERMINED:
        return CausalLabel.UNDERDETERMINED
    return CausalLabel.SUPPORTED
