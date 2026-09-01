# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Serverless (SageMaker Public Hub) model discovery + runtime tag overlay.

The serverless engine is offered for a catalog model only when it carries a
`serverless_model_id` — the SageMaker Public Hub id whose V3 recipe trains it
(catalog.py:93). Those ids were hand-curated on ~11 rows, which drifts: AWS adds
new customizable models, and a curated row can be missed or go stale.

This module is the auto-discovery analogue of releases.discover_new_models (the
"Find new models" image-diff flow), applied to the serverless catalog instead of
image tiers:

1. DISCOVER — list SageMakerPublicHub models, keep the ones whose keywords mark
   them serverless-customizable (`@capability:customization` + a
   `@recipe:finetuning_*` recipe), and CLASSIFY them against our catalog:
     * untaggedMatches — a catalog row exists but has no serverless tag yet
       (we CAN light up the serverless engine for it)
     * staleTags       — a catalog row is tagged but the hub no longer lists that
       id as customizable (the tag may be dead)
     * newCandidates   — customizable on the hub but no catalog row at all
       (a future "Add from Hugging Face" pick)
   Suggestions ONLY — never auto-tags (verify-before-trust, same as image discovery).

2. APPLY — register a discovered (catalogId → hubId) tag as a RUNTIME overlay in
   config.json (mirrors aws_config.register_image_tier), so the serverless engine
   becomes available for that model WITHOUT a redeploy. The static
   CATALOG.serverless_model_id is the floor; the overlay only FILLS empties (a
   hand-curated tag is never silently overridden). catalog.list_models/get_model
   apply the overlay so every launch/verification gate (which reads
   serverless_model_id off the ModelSpec) honors it uniformly.

Deterministic core; the only network call is the SageMaker control plane
(list_hub_contents, read-only). Degrades gracefully (returns a note + empty
lists) when the hub is unreachable, like releases.py.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .aws_config import _invalidate_config_cache, _saved, load_aws_config
from .store import get_store

# The public model hub JumpStart/serverless content lives in. Region-specific —
# always queried in cfg.region (us-east-1 for this deployment).
_HUB_NAME = "SageMakerPublicHub"

# config.json key holding the runtime tag overlay: {catalogId: hubId}.
_OVERLAY_KEY = "serverlessModelIds"

# A SageMaker Public Hub content name (e.g. "huggingface-reasoning-qwen3-06b",
# "meta-textgeneration-llama-3-2-3b-instruct"). Strict allowlist BEFORE the value
# is persisted or echoed into a launch spec — closes the tool-misuse surface
# (OWASP-Agentic ASI-02 / CWE-918) the same way releases._validate_tag does.
_HUB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]{0,127}$")

# Keyword markers (verified live on the hub content metadata).
_CAP_CUSTOMIZATION = "@capability:customization"
_RECIPE_PREFIX = "@recipe:finetuning_"
_HF_ID_PREFIX = "@huggingface-id:"


def _validate_hub_id(hub_id: str) -> str:
    """Return a clean hub model id or raise ValueError. Trust boundary for every
    hub id that flows into config.json or a serverless launch spec."""
    if not isinstance(hub_id, str):
        raise ValueError("hub model id must be a string")
    hub_id = hub_id.strip()
    if not hub_id or not _HUB_ID_RE.match(hub_id):
        raise ValueError(
            f"invalid hub model id '{hub_id}': expected a SageMaker Public Hub "
            "content name (lowercase letters, digits, '.' or '-')"
        )
    return hub_id


# --- runtime overlay (config.json) -------------------------------------------

def serverless_overlay() -> dict[str, str]:
    """The runtime (catalogId → hubId) serverless-tag overlay from config.json.
    Empty when none registered. Read-cached (5s) like every other config read."""
    saved = _saved().get(_OVERLAY_KEY)
    if not isinstance(saved, dict):
        return {}
    return {str(k): str(v) for k, v in saved.items() if v}


def register_serverless_model_id(catalog_id: str, hub_id: str) -> dict[str, str]:
    """Persist a (catalogId → hubId) serverless tag into config.json at RUNTIME, so
    the serverless engine becomes available for that model with NO redeploy.

    Mirrors aws_config.register_image_tier: read-modify-write of the fresh config
    doc, then invalidate the cache. Validates the hub id (trust boundary) and that
    the catalog id resolves to a known model. The model is still UNVERIFIED on the
    serverless surface until a smoke test proves it (verify-before-trust)."""
    from .catalog import get_model

    hub_id = _validate_hub_id(hub_id)
    if not get_model(catalog_id):
        raise ValueError(f"unknown model id '{catalog_id}'")
    saved = _saved(use_cache=False)  # read-modify-write → fresh, never the cache
    overlay = dict(saved.get(_OVERLAY_KEY) or {})
    overlay[str(catalog_id)] = hub_id
    saved[_OVERLAY_KEY] = overlay
    get_store().write_root_json("config.json", saved)
    _invalidate_config_cache()
    return serverless_overlay()


def unregister_serverless_model_id(catalog_id: str) -> dict[str, str]:
    """Remove a runtime serverless tag overlay entry (does NOT touch a static
    CATALOG tag — those are code). No-op if absent."""
    saved = _saved(use_cache=False)
    overlay = dict(saved.get(_OVERLAY_KEY) or {})
    if str(catalog_id) in overlay:
        del overlay[str(catalog_id)]
        saved[_OVERLAY_KEY] = overlay
        get_store().write_root_json("config.json", saved)
        _invalidate_config_cache()
    return serverless_overlay()


# --- live hub query ----------------------------------------------------------

def _parse_keywords(kws: list[str]) -> dict[str, Any] | None:
    """If these hub-content keywords mark a serverless-CUSTOMIZABLE model, return
    {hf, recipes:[...]}; else None. A model is customizable iff it carries the
    customization capability AND at least one finetuning recipe."""
    if _CAP_CUSTOMIZATION not in kws:
        return None
    recipes = sorted({
        k[len(_RECIPE_PREFIX):] for k in kws if k.startswith(_RECIPE_PREFIX)
    })
    if not recipes:
        return None
    hf = ""
    for k in kws:
        if k.startswith(_HF_ID_PREFIX):
            hf = k[len(_HF_ID_PREFIX):].strip()
            break
    return {"hf": hf, "recipes": recipes}


# Backstop so a runaway NextToken loop can't spin forever (696 models / 100 ≈ 7
# pages today; 50 is generous headroom).
_MAX_HUB_PAGES = 50


def list_customizable_hub_models() -> list[dict[str, Any]]:
    """Every SageMakerPublicHub model marked serverless-customizable, as
    [{name, hf, recipes}]. Returns [] on any control-plane error (the UI then
    shows 'could not check'). list_hub_contents summaries already carry the
    keywords, so no per-item describe is needed.

    NB: list_hub_contents is NOT a paginable boto operation (no get_paginator), so
    we page manually with NextToken — the same way the AWS CLI does."""
    from .orchestrate import _session

    cfg = load_aws_config()
    try:
        _, boto_sess = _session(cfg)
        sm = boto_sess.client("sagemaker", region_name=cfg.region)
        out: list[dict[str, Any]] = []
        token: str | None = None
        for _ in range(_MAX_HUB_PAGES):
            kwargs: dict[str, Any] = {
                "HubName": _HUB_NAME, "HubContentType": "Model", "MaxResults": 100,
            }
            if token:
                kwargs["NextToken"] = token
            resp = sm.list_hub_contents(**kwargs)
            for m in resp.get("HubContentSummaries", []):
                parsed = _parse_keywords(m.get("HubContentSearchKeywords", []) or [])
                if parsed is None:
                    continue
                out.append({"name": m["HubContentName"], **parsed})
            token = resp.get("NextToken")
            if not token:
                break
        return out
    except Exception:  # noqa: BLE001 — degrade like releases.list_upstream_releases
        return []


def describe_hub_hf(hub_name: str) -> dict[str, Any]:
    """Fill the gaps the list-summary keyword leaves: the HF repo id (from the
    content document's `Url`, e.g. https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
    and the input/output modalities (authoritative VLM detection). The list
    summary's `@huggingface-id` keyword is EMPTY for several models (Qwen2.5-32b/72b,
    qwen3-32b…), which would mislabel them 'no HF repo'. describe_hub_content carries
    the real id — but it's an N+1 call, so only call it for the few candidates that
    need it. Returns {hf, modalities:[...]}; degrades to {} on any error."""
    from .orchestrate import _session

    cfg = load_aws_config()
    try:
        _, boto_sess = _session(cfg)
        sm = boto_sess.client("sagemaker", region_name=cfg.region)
        d = sm.describe_hub_content(
            HubName=_HUB_NAME, HubContentType="Model", HubContentName=hub_name)
        doc = json.loads(d.get("HubContentDocument") or "{}")
        hf = ""
        url = doc.get("Url") or ""
        m = re.search(r"huggingface\.co/([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*)", url)
        if m:
            hf = m.group(1)
        modalities = [
            *(doc.get("InputModalities") or []),
            *(doc.get("OutputModalities") or []),
        ]
        return {"hf": hf, "modalities": [str(x).lower() for x in modalities]}
    except Exception:  # noqa: BLE001
        return {}


def _match_hub_to_catalog(hub: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index hub models by the GUESSED catalog hub-content name(s) they could tag.
    Returns {hubName: hubRecord}. Used to match a catalog row (by its existing
    serverless_model_id OR a name guessed from its hf id) to a live hub entry."""
    return {h["name"]: h for h in hub}


def _hub_name_guesses(hf_model_id: str) -> list[str]:
    """Candidate Public Hub content names for a catalog model's HF id, for rows
    whose hub entry carries an empty @huggingface-id keyword (so an hf-id match
    can't find them). Mirrors the observed naming, e.g.
    'meta-llama/Llama-3.2-1B-Instruct' → 'meta-textgeneration-llama-3-2-1b-instruct'."""
    slug = hf_model_id.split("/")[-1].lower().replace(".", "-")
    return [
        f"huggingface-reasoning-{slug}",
        f"huggingface-llm-{slug}",
        f"meta-textgeneration-{slug}",
        f"deepseek-llm-{slug}",
        f"openai-reasoning-{slug}",
    ]


def discover_serverless_models() -> dict[str, Any]:
    """Classify the live serverless-customizable hub models against our catalog.

    Returns a flat `allModels:[{hubId, displayName, id, hfModelId, recipes, gated,
    state, verified, reason}]` (EVERY customizable hub model, for a browsable
    pick-and-add table; `state` ∈ enabled|addable|onboardable|unavailable) PLUS the
    diff buckets `untaggedMatches`/`staleTags`/`newCandidates` (back-compat) +
    `customizableCount`/`note`. Suggestions only — applying a tag or onboarding a
    candidate is an explicit action. Empty (with a note) when the hub is unreachable."""
    from .catalog import CATALOG, list_models, model_id_for_hf

    hub = list_customizable_hub_models()
    if not hub:
        return {
            "customizableCount": 0, "allModels": [], "untaggedMatches": [],
            "staleTags": [], "newCandidates": [],
            "note": "could not list SageMaker Public Hub models (no access or none found).",
        }

    by_name = _match_hub_to_catalog(hub)
    by_hf = {h["hf"].lower(): h for h in hub if h["hf"]}
    live_names = set(by_name)

    # The SERVED catalog (built-ins + runtime overlay), so a tag already applied
    # via the overlay isn't re-suggested as "untagged".
    served = list_models()
    tagged_now = {m["id"]: m.get("serverlessModelId", "") for m in served}

    untagged: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    enabled: list[dict[str, Any]] = []  # catalog rows already serverless-tagged AND live
    matched_catalog_ids: set[str] = set()

    # Walk the served catalog: classify each row's serverless status vs the hub.
    for m in served:
        cid = m["id"]
        cur = tagged_now.get(cid, "")
        hf = (m.get("hfModelId") or "")
        # Find a live hub entry for this row: by current tag, by hf id, or by guess.
        hub_hit = None
        if cur and cur in by_name:
            hub_hit = by_name[cur]
        elif hf.lower() in by_hf:
            hub_hit = by_hf[hf.lower()]
        else:
            for guess in _hub_name_guesses(hf):
                if guess in by_name:
                    hub_hit = by_name[guess]
                    break
        if hub_hit:
            matched_catalog_ids.add(hub_hit["name"])
            if not cur:
                untagged.append({
                    "id": cid, "displayName": m.get("displayName", cid),
                    "hfModelId": hf, "hubId": hub_hit["name"],
                    "recipes": hub_hit["recipes"], "gated": bool(m.get("gated")),
                })
            else:
                # Already serverless-tagged AND the hub still lists it → enabled.
                enabled.append({
                    "id": cid, "displayName": m.get("displayName", cid),
                    "hfModelId": hf, "hubId": hub_hit["name"],
                    "recipes": hub_hit["recipes"], "gated": bool(m.get("gated")),
                    "verified": (m.get("verifications") or {}).get("serverless", {}).get("status") == "verified",
                })
        elif cur and cur not in live_names:
            # Tagged, but the hub no longer lists that id as customizable.
            stale.append({"id": cid, "displayName": m.get("displayName", cid),
                          "hubId": cur})

    # New candidates: customizable hub models that didn't match ANY catalog row.
    # Each is flagged `onboardable` = can we add it end-to-end with one click?
    # Yes only when it has a public HF repo id (so probe_model can read its
    # config/template) AND is not vision-language (our text training/eval path
    # can't serve a VLM). Nova has no HF repo; *vlm* models are multimodal — both
    # are listed for awareness but not one-click.
    new_candidates: list[dict[str, Any]] = []
    for h in hub:
        if h["name"] in matched_catalog_ids:
            continue
        hf = h["hf"]
        modalities: list[str] = []
        # The list summary's @huggingface-id is empty for several models (e.g.
        # Qwen2.5-32b/72b, qwen3-32b) — they DO have a public repo. Fall back to
        # describe_hub_content (per-item, only for the few that need it) to recover
        # the real HF id + authoritative modalities, so we don't mislabel them.
        if not hf:
            extra = describe_hub_hf(h["name"])
            hf = extra.get("hf", "") or ""
            modalities = extra.get("modalities", [])
        # Skip if its (possibly just-recovered) hf id already maps to a catalog row.
        if hf and model_id_for_hf_safe(hf):
            continue
        # VLM detection: prefer modalities (authoritative), else the name marker.
        is_vlm = ("image" in modalities or "vision" in modalities
                  or "vlm" in h["name"] or "vlm" in hf.lower())
        onboardable = bool(hf) and not is_vlm
        reason = ""
        if not hf:
            reason = "no public Hugging Face repo (e.g. Amazon Nova) — not onboardable here"
        elif is_vlm:
            reason = "vision-language model — the text training/eval path can't serve it"
        new_candidates.append({
            "hubId": h["name"], "hf": hf, "recipes": h["recipes"],
            "onboardable": onboardable, "reason": reason,
        })

    note = ""
    if not untagged and not new_candidates and not stale:
        note = "every serverless-customizable hub model is already reflected in the catalog."

    # ONE flat, browsable list of EVERY serverless-customizable hub model with its
    # current state, so the UI can show a single pick-and-add table (not just the
    # diff buckets). `state` drives the per-row action + whether it's selectable:
    #   enabled      — already in the catalog + serverless-tagged (nothing to do)
    #   addable      — in the catalog, untagged → one-click "Enable serverless"
    #   onboardable  — not in the catalog, has an HF repo → "Add + enable serverless"
    #   unavailable  — no HF repo / VLM → awareness only (reason explains)
    all_models: list[dict[str, Any]] = []
    for e in enabled:
        all_models.append({
            "hubId": e["hubId"], "displayName": e["displayName"], "id": e["id"],
            "hfModelId": e["hfModelId"], "recipes": e["recipes"], "gated": e["gated"],
            "state": "enabled", "verified": e["verified"], "reason": "",
        })
    for u in untagged:
        all_models.append({
            "hubId": u["hubId"], "displayName": u["displayName"], "id": u["id"],
            "hfModelId": u["hfModelId"], "recipes": u["recipes"], "gated": u["gated"],
            "state": "addable", "verified": False, "reason": "",
        })
    for c in new_candidates:
        all_models.append({
            "hubId": c["hubId"], "displayName": c["hubId"], "id": "",
            "hfModelId": c["hf"], "recipes": c["recipes"], "gated": False,
            "state": "onboardable" if c["onboardable"] else "unavailable",
            "verified": False, "reason": c["reason"],
        })

    return {
        "customizableCount": len(hub),
        "allModels": sorted(all_models, key=lambda x: x["displayName"].lower()),
        # Kept for back-compat with any caller of the diff buckets.
        "untaggedMatches": sorted(untagged, key=lambda x: x["id"]),
        "staleTags": sorted(stale, key=lambda x: x["id"]),
        "newCandidates": sorted(new_candidates, key=lambda x: x["hubId"]),
        "note": note,
    }


def model_id_for_hf_safe(hf_model_id: str) -> str | None:
    """catalog.model_id_for_hf, import-safe (avoid a hard import cycle at module load)."""
    try:
        from .catalog import model_id_for_hf

        return model_id_for_hf(hf_model_id)
    except Exception:  # noqa: BLE001
        return None
