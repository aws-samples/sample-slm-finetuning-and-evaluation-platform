# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Model-card (HF config.json) architecture facts for hyperparameter defaults.

The recommender (recommend.py) seeds the values a model card can LEGITIMATELY
constrain — context length and the freeze-trainable-layer cap — from the card's
architecture, falling back to its size×dataset heuristics for everything the card
doesn't specify (LR, epochs, rank, save_steps).

What a card actually exposes (verified against real repos): `config.json` carries
ARCHITECTURE facts — `num_hidden_layers`, `max_position_embeddings`, `model_type`,
`torch_dtype` — NOT training hyperparameters. There is no standard field for a
recommended fine-tuning LR/epochs/rank, so we only read what's really there and
cite it; the deterministic engine supplies the training knobs.

Best-effort + cached: a failed/absent/gated config degrades to {} so the
recommender always works (it just falls back to the size-based heuristic). Reuses
onboard's validated fetch (the repo id is allowlisted before it hits a URL).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=256)
def fetch_arch(hf_model_id: str) -> dict[str, Any]:
    """Architecture facts from the model's HF config.json, or {} on any failure.

    Returns a dict with (when present): maxPositionEmbeddings, numHiddenLayers,
    modelType, torchDtype. Cached per repo for the process lifetime — the card is
    immutable enough for a recommendation, and this avoids re-fetching on every
    'Suggest defaults' click. NEVER raises: an offline/gated/private/garbage repo
    yields {} so the recommender just uses its deterministic fallback."""
    # Imported lazily so a unit test can monkeypatch onboard's fetch without this
    # module pulling the network at import time.
    from .onboard import HF_CONFIG, _http_json, _validate_repo
    from .secrets import get_hf_token

    try:
        repo = _validate_repo(hf_model_id)  # allowlist before the value hits a URL
    except Exception:  # noqa: BLE001 — a non-repo id (shouldn't happen) → no card
        return {}
    try:
        token = get_hf_token()
        config = _http_json(HF_CONFIG.format(repo=repo), token)
    except Exception:  # noqa: BLE001 — offline / gated / private / 404 → fall back
        return {}
    if not isinstance(config, dict):
        return {}

    out: dict[str, Any] = {}
    mpe = config.get("max_position_embeddings")
    if isinstance(mpe, int) and mpe > 0:
        out["maxPositionEmbeddings"] = mpe
    nhl = config.get("num_hidden_layers")
    if isinstance(nhl, int) and nhl > 0:
        out["numHiddenLayers"] = nhl
    if config.get("model_type"):
        out["modelType"] = str(config["model_type"])
    if config.get("torch_dtype"):
        out["torchDtype"] = str(config["torch_dtype"])
    return out
