# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Curated baseline of (model, image-tier) verifications from real training jobs.

A fresh deployment in your own AWS account starts with an empty
verifications.json, so the Model Catalog would show every model as "untested"
even though real training jobs have already proven many of them. This seed ships
those results as a baseline so the catalog is useful on day one.

Why this is sound (not just optimistic defaults): the image tiers are
REPRODUCIBLE — each tier is built from a pinned LLaMA-Factory base
(hiyouga/llamafactory:0.9.4 / :0.9.5) via the same Dockerfile in every account.
So a model verified on one deployment's `stable` (0.9.4) runs on another's
`stable` (0.9.4): same bits, same result. The seed is keyed by TIER NAME (not a
specific ECR digest) for exactly this reason.

Override rule: the seed is a FLOOR, never a ceiling. Your own run always wins —
if a model is in the seed as verified but fails in your account (different region
capacity, a corrupted pull, a future image rebuild), the local store record
overrides the seed (see verifications.get_status / all_verifications merge). And
after an image REBUILD, reset_tier clears the store records but the seed remains
as the "known-good on the pinned base" baseline; a real local run re-confirms it.

Each entry records the source job that proved it (provenance), so the catalog
popover can show "verified (baseline)" with the job that established it.
"""

from __future__ import annotations

from typing import Any

# {modelId: {tierName: {status, jobName, reason, ts, seed}}}
# Only VERIFIED facts belong here — never seed "incompatible" (a failure in one
# account may be environmental; let each account discover its own failures).
# These are the models proven by real SageMaker training jobs as of 2026-06-04.
_SEED: dict[str, dict[str, dict[str, Any]]] = {
    # --- verified on `stable` (LLaMA-Factory 0.9.4 / transformers 4.57.1) ---
    "qwen3-0.6b": {"stable": {"jobName": "slm-qwen3-0-6b-cafb6768602d-20260603-11"}},
    "qwen3-1.7b": {"stable": {"jobName": "slm-qwen3-1-7b-c66d5b7c9cc4-20260604-07"}},
    "qwen2.5-0.5b": {"stable": {"jobName": "slm-qwen2-5-0-5b-cafb6768602d-20260602"}},
    "phi-3.5-mini": {"stable": {"jobName": "slm-phi-3-5-mini-6cbf02796b49-20260603"}},
    "minicpm4-0.5b": {"stable": {"jobName": "slm-minicpm4-0-5b-cafb6768602d-20260602"}},
    "granite-3.1-2b": {"stable": {"jobName": "slm-granite-3-1-2b-cafb6768602d-2026060"}},
    "mistral-7b": {"stable": {"jobName": "slm-mistral-7b-6cbf02796b49-20260603-15"}},
    # --- verified on `latest` (LLaMA-Factory 0.9.5 / transformers 5.6.0) ---
    # The canonical "needs the newer image" proof: LFM2-MoE trains on 0.9.5,
    # fails on 0.9.4 (lfm2_moe arch unknown to transformers 4.x).
    "lfm2-8b-a1b": {"latest": {"jobName": "slm-lfm2-8b-a1b-27c7d02e1ca6-proof095"}},
}


def seed_status(model_id: str, image_tag: str) -> dict[str, Any] | None:
    """The seeded VERIFIED record for (model, tier), or None if not seeded."""
    rec = _SEED.get(model_id, {}).get(image_tag)
    if rec is None:
        return None
    return {
        "status": "verified",
        "jobName": rec.get("jobName"),
        "reason": None,
        "ts": None,
        "seed": True,  # provenance: this came from the shipped baseline, not a local run
    }


def seed_for_model(model_id: str) -> dict[str, dict[str, Any]]:
    """All seeded tiers for a model, e.g. {"stable": {...verified...}}."""
    out: dict[str, dict[str, Any]] = {}
    for tier in _SEED.get(model_id, {}):
        rec = seed_status(model_id, tier)
        if rec:
            out[tier] = rec
    return out


def all_seed() -> dict[str, dict[str, dict[str, Any]]]:
    """The whole seed expanded to full records (for merging into all_verifications)."""
    return {mid: seed_for_model(mid) for mid in _SEED}
