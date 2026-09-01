# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Per-(model, image-tier) verification status — what makes the catalog trustworthy.

The multi-image design hinges on one fact: a model that works on image 0.9.4 is
NOT automatically known to work on 0.9.5. Verification is therefore tied to the
IMAGE TAG, not just the model. An image bump resets everyone on that tier to
"untested" until a real run (or smoke test) re-proves them. This is what keeps
the Model Catalog honest instead of stale.

Status values:
  verified     — a real training job for (model, tag) reached Completed. Proof.
  incompatible — a run for (model, tag) failed in a way that means the stack
                 can't load this model (e.g. ImportError needing a newer
                 transformers). The signal to move it to a newer tier.
  untested     — no run on this (model, tag) yet. The default for everything.

Storage mirrors the custom_models.json root-doc pattern (onboard.py): a single
JSON document {modelId: {imageTag: {status, jobName, reason, ts}}}, so it's
visible to every Lambda and survives restarts. Auto-promotion writes here from
the race reconcile loop on any TRAINING→Completed transition; the smoke-test and
the (future) agentic self-healing write here too.

Deterministic + no AWS: this module only reads/writes the state store. Callers
pass the `ts` (no time in library code, matching the rest of the platform).
"""

from __future__ import annotations

from typing import Any

from .aws_config import DEFAULT_IMAGE_TIER, image_tiers
from .store import get_store

# Single root document: {modelId: {imageTag: record}}.
_VERIF_FILE = "verifications.json"

VERIFIED = "verified"
INCOMPATIBLE = "incompatible"
UNTESTED = "untested"
PENDING = "pending"  # a smoke-test job is in flight for this (model, tier)
# A gated model whose HF token lacks Meta/Google license approval — the job
# failed on a 403/GatedRepoError, NOT because the stack can't run the model. This
# is FIXABLE by accepting the license on HF, so it's distinct from incompatible
# (which means the image genuinely can't load the architecture).
ACCESS_DENIED = "access_denied"


_ACCESS_DENIED_REASON = (
    "Access denied: this gated model's HF token lacks license approval. Accept "
    "the license on the model's Hugging Face page, then re-test."
)


def _is_access_error(reason: str | None) -> bool:
    """Does a job's failure reason look like a gated-repo / auth denial (vs a real
    incompatibility)? Matches the HF GatedRepoError / 403 signatures."""
    r = (reason or "").lower()
    return any(
        s in r
        for s in ("gatedrepoerror", "gated repo", "403", "is restricted",
                  "authorized list", "access to model", "request access")
    )


def classify_failure(job_name: str, failure_reason: str | None) -> tuple[str, str]:
    """A Failed/Stopped smoke-test → (status, reason). Returns ACCESS_DENIED with a
    fix-it message when the failure is a gated-repo 403 (the signature is usually
    only in the LOGS, not the bare FailureReason), else INCOMPATIBLE with the
    original reason."""
    base = failure_reason or "training failed"
    if _is_access_error(failure_reason):
        return ACCESS_DENIED, _ACCESS_DENIED_REASON
    from .orchestrate import job_log_tail

    if _is_access_error(job_log_tail(job_name)):
        return ACCESS_DENIED, _ACCESS_DENIED_REASON
    return INCOMPATIBLE, base

# A run reaching Completed always wins (it's proof); incompatible is a strong
# negative but a later success overrides it. PENDING ranks LOWEST so a stored
# verified/incompatible is never overwritten by a new pending — but pending DOES
# replace untested (and set_pending uses force so a re-test from a known state
# shows progress). Higher rank = more authoritative.
# ACCESS_DENIED ranks WITH incompatible (a negative result a later success
# overrides) — but it means "fixable on HF", not "the stack can't run it".
_RANK = {UNTESTED: 0, PENDING: 0, INCOMPATIBLE: 1, ACCESS_DENIED: 1, VERIFIED: 2}


# Verification is keyed (model, image_tag, METHOD). The parameterization matters:
# a model proven on plain LoRA is NOT automatically known to train under QLoRA
# (4-bit bitsandbytes load can fail for an arch that LoRA handles fine), so each
# method must prove itself — verify-before-trust, exactly like the per-image rule.
#
# Storage stays BACKWARD-COMPATIBLE: LoRA (the only method that existed) keeps
# living under the bare `image_tag` key, so every record written before this
# change AND the shipped seed remain valid untouched. A non-LoRA method gets a
# namespaced sibling key `f"{image_tag}::{method}"`. So one model's record dict
# might be {"stable": <lora>, "stable::qlora": <qlora>}.
DEFAULT_METHOD = "lora"

# Verification is ALSO an engine axis (model, ENGINE, image_tag, method). The
# default engine (LLaMA-Factory) trains on an ECR image, so its verification
# SURFACE is the image tier (stable/latest). A non-default engine has no image of
# ours — e.g. the SageMaker serverless engine trains on a managed recipe — so its
# surface is a fixed token naming the engine ("serverless"), not an image tier.
# This keeps engine a FIRST-CLASS axis (a serverless proof is distinct from an
# LLaMA-Factory proof for the same model) while reusing the exact (surface, method)
# key scheme — so existing data is byte-identical and there's no migration:
#   * llama_factory → surface = the image_tag (stable/latest) — unchanged keys.
#   * sagemaker_serverless → surface = "serverless" (the same token the prior
#     race autopromote already wrote, so existing serverless proofs still read).
DEFAULT_ENGINE = "llama_factory"
# engine name → its verification surface token (when the engine has no image tier).
# Engines absent here use the image_tag as their surface (the default behavior).
_ENGINE_SURFACE = {"sagemaker_serverless": "serverless"}

# LoRA adapter VARIANT is the newest verification axis (model, surface, method,
# VARIANT). DoRA/rsLoRA/PiSSA/LoRA+ are MODIFIERS that ride the lora/qlora method
# but change training (and, for DoRA/PiSSA, the merge) enough that a plain-LoRA
# proof does NOT cover them — DoRA was the recommended variant yet silently
# inherited plain LoRA's badge (a completed DoRA run autopromoted to the BARE LoRA
# key). Verify-before-trust must therefore be per-variant.
#
# Plain "lora" is the default and adds NO token, so every pre-variant record stays
# byte-identical (no migration). A non-plain variant appends `::<variant>` LAST,
# AFTER an explicit method token — so DoRA is `surface::lora::dora`, never a bare
# `surface::dora` that _split_key would misread as a method. Variants only ride the
# adapter methods (lora/qlora); full/freeze have no adapter, so they never carry a
# variant token even if one is passed (normalized to "lora", as Hyperparams does).
DEFAULT_VARIANT = "lora"
_KNOWN_VARIANTS = ("lora", "dora", "rslora", "pissa", "loraplus")
_FULL_WEIGHT_METHODS = ("full", "freeze")


def _norm_variant(method: str, lora_variant: str | None) -> str:
    """The effective variant for a (method, variant) pair. Full/freeze have no
    adapter so they always normalize to plain "lora" (mirrors Hyperparams.__post_init__);
    an unknown variant also falls back to plain so a bad value can't forge a key."""
    v = lora_variant or DEFAULT_VARIANT
    if method in _FULL_WEIGHT_METHODS or v not in _KNOWN_VARIANTS:
        return DEFAULT_VARIANT
    return v


def _surface(image_tag: str, engine: str = DEFAULT_ENGINE) -> str:
    """The verification surface for (image_tag, engine): the image tier for the
    default engine, or the engine's fixed token (e.g. "serverless") for an engine
    that has no image of ours. This is what the (surface, method) key is built on."""
    return _ENGINE_SURFACE.get(engine, image_tag)


def _key(image_tag: str, method: str, engine: str = DEFAULT_ENGINE,
         lora_variant: str = DEFAULT_VARIANT) -> str:
    """Composite storage key. Plain LoRA → the bare surface (back-compat); a
    non-default method → surface::method; a non-plain LoRA variant →
    surface::method::variant (the method token is ALWAYS present alongside a
    variant, so a variant can never be confused with a method on parse). The
    surface is the image tier (default engine) or the engine's token (e.g.
    "serverless"). For llama_factory + plain LoRA this is the bare image_tag,
    byte-identical to every pre-engine record + the shipped seed."""
    surface = _surface(image_tag, engine)
    variant = _norm_variant(method, lora_variant)
    if method == DEFAULT_METHOD and variant == DEFAULT_VARIANT:
        return surface  # bare — unchanged for plain llama_factory LoRA
    if variant == DEFAULT_VARIANT:
        return f"{surface}::{method}"  # non-default method, plain variant (e.g. qlora)
    # A non-plain variant always carries an explicit method token (DEFAULT_METHOD
    # for the plain method) so `surface::lora::dora` is unambiguous to _split_key.
    return f"{surface}::{method}::{variant}"


# Short-lived read cache for the verifications doc. `model_status_map` reads it
# once PER TIER per model and `/api/models` enriches every model, so a single
# request fanned out to 100+ identical S3 GETs of this one file (~13s observed).
# The doc only changes on a write (training completion / smoke test / heal), which
# we invalidate explicitly; the 5s TTL is just a cross-invocation freshness
# backstop. monotonic() is a cache clock, not a persisted record timestamp, so it
# doesn't break the module's "no time in records" rule.
_CACHE_TTL_S = 5.0
_all_cache: dict[str, Any] | None = None
_all_cache_at: float = 0.0


def _all(use_cache: bool = True) -> dict[str, Any]:
    """The verifications doc. `use_cache=True` (read-only callers) may return the
    cached copy; read-modify-write callers MUST pass use_cache=False so they mutate
    a FRESH object (no aliasing of the shared cache) before writing it back."""
    global _all_cache, _all_cache_at
    import time

    if use_cache and _all_cache is not None and (time.monotonic() - _all_cache_at) < _CACHE_TTL_S:
        return _all_cache
    doc = get_store().read_root_json(_VERIF_FILE)
    if use_cache:
        _all_cache = doc
        _all_cache_at = time.monotonic()
    return doc


def _invalidate_all_cache() -> None:
    """Drop the cached doc so the next read reflects a just-written change."""
    global _all_cache
    _all_cache = None


def _write(doc: dict[str, Any]) -> None:
    get_store().write_root_json(_VERIF_FILE, doc)
    _invalidate_all_cache()


def get_status(model_id: str, image_tag: str, method: str = DEFAULT_METHOD,
               engine: str = DEFAULT_ENGINE,
               lora_variant: str = DEFAULT_VARIANT) -> dict[str, Any]:
    """The verification record for (model, engine, tag, method, variant). The store
    (this account's own runs) ALWAYS wins; if it has no record, fall back to the
    shipped baseline seed — but ONLY for the default engine + plain LoRA, so a
    non-LoRA method, a non-default engine (e.g. serverless) OR a non-plain LoRA
    variant (DoRA/PiSSA/…) never inherits a plain LLaMA-Factory LoRA proof;
    otherwise UNTESTED. Never None — callers can read .['status']."""
    variant = _norm_variant(method, lora_variant)
    rec = _all().get(model_id, {}).get(_key(image_tag, method, engine, variant))
    if rec is not None:
        return rec
    if method == DEFAULT_METHOD and engine == DEFAULT_ENGINE and variant == DEFAULT_VARIANT:
        from .verification_seed import seed_status

        seeded = seed_status(model_id, image_tag)
        if seeded is not None:
            return seeded
    return {"status": UNTESTED, "jobName": None, "reason": None, "ts": None}


def set_status(
    model_id: str,
    image_tag: str,
    status: str,
    *,
    method: str = DEFAULT_METHOD,
    engine: str = DEFAULT_ENGINE,
    lora_variant: str = DEFAULT_VARIANT,
    job_name: str | None = None,
    reason: str | None = None,
    ts: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Record a (model, engine, tag, method, variant) verification result.

    By default this only "upgrades" the record by authority (UNTESTED <
    INCOMPATIBLE < VERIFIED) so a transient failure can't erase a prior proven
    success — verification is informational, not a hard block, and a flaky run
    (spot reclaim, download hiccup) shouldn't permanently mark a good model bad.
    Pass force=True for an explicit operator re-test that should overwrite
    regardless (e.g. the catalog "re-test" button after an image rebuild).
    Returns the stored record.
    """
    if status not in (VERIFIED, INCOMPATIBLE, ACCESS_DENIED, UNTESTED, PENDING):
        raise ValueError(f"unknown verification status: {status}")
    doc = _all(use_cache=False)  # read-modify-write → fresh, never the shared cache
    per_model = doc.setdefault(model_id, {})
    key = _key(image_tag, method, engine, lora_variant)
    existing = per_model.get(key)
    if existing and not force and _RANK[status] < _RANK.get(existing.get("status"), 0):
        return existing  # keep the more authoritative existing record
    rec = {"status": status, "jobName": job_name, "reason": reason, "ts": ts}
    per_model[key] = rec
    _write(doc)
    return rec


def set_pending(model_id: str, image_tag: str, job_name: str, ts: str | None = None,
                method: str = DEFAULT_METHOD, engine: str = DEFAULT_ENGINE,
                lora_variant: str = DEFAULT_VARIANT) -> dict[str, Any]:
    """Record that a smoke-test job is IN FLIGHT for (model, engine, tier, method,
    variant). Persisted (with force) so the catalog shows 'verifying…' even after
    the user navigates away — the reconcile loop (resolve_pending_verifications)
    later advances it to the real result when the job finishes. Stores jobName to
    poll AND lora_variant so resolve can write back to the SAME per-variant key
    (else a DoRA smoke-test would resolve onto the plain-LoRA key)."""
    return set_status(model_id, image_tag, PENDING, method=method, engine=engine,
                      lora_variant=lora_variant, job_name=job_name, ts=ts, force=True)


# Reverse map: a surface token → the engine it belongs to (for engines with no
# image tier). Inverse of _ENGINE_SURFACE; used by _split_key to recover the axis.
_SURFACE_ENGINE = {v: k for k, v in _ENGINE_SURFACE.items()}


def _split_key(key: str) -> tuple[str, str, str, str]:
    """Inverse of _key: a stored key → (surface, method, engine, lora_variant). A
    bare key is plain LoRA; `surface::method` carries the method; `surface::method::
    variant` also carries the non-plain LoRA variant. The surface maps back to its
    engine when it's an engine token (e.g. "serverless"→sagemaker_serverless), else
    it's an image tier on the default engine. Returns the SURFACE as the first
    element (for the default engine that's the image_tag; for an engineless engine
    it's the token, which is what reset_tier compares against)."""
    parts = key.split("::")
    surface = parts[0]
    method = parts[1] if len(parts) >= 2 else DEFAULT_METHOD
    variant = parts[2] if len(parts) >= 3 else DEFAULT_VARIANT
    engine = _SURFACE_ENGINE.get(surface, DEFAULT_ENGINE)
    return surface, method, engine, variant


def list_pending() -> list[dict[str, Any]]:
    """All (model, tier, method, variant) records currently PENDING — each with its
    jobName. Used by the reconcile loop to advance in-flight verifications headlessly.
    Carries loraVariant so resolve writes back to the SAME per-variant key."""
    out: list[dict[str, Any]] = []
    for model_id, per in _all().items():
        for key, rec in per.items():
            if rec.get("status") == PENDING and rec.get("jobName"):
                surface, method, engine, variant = _split_key(key)
                out.append({"modelId": model_id, "imageTag": surface, "method": method,
                            "engine": engine, "loraVariant": variant,
                            "jobName": rec["jobName"]})
    return out


def resolve_pending_verifications() -> dict[str, Any]:
    """Advance every PENDING (model, tier) by polling its smoke-test job: Completed
    → verified, Failed/Stopped → incompatible, still running → left pending.

    Headless + best-effort (called from the reconcile Lambda), so a verification
    resolves to a trustworthy result even if no UI is open. Returns a small summary
    {checked, resolved}. Never raises — a bad job is skipped, not fatal."""
    from .orchestrate import describe_job

    pending = list_pending()
    resolved = 0
    for p in pending:
        try:
            d = describe_job(p["jobName"])
        except Exception:  # noqa: BLE001 — job not describable yet; leave pending
            continue
        st = d.get("status")
        method = p.get("method", DEFAULT_METHOD)
        engine = p.get("engine", DEFAULT_ENGINE)
        variant = p.get("loraVariant", DEFAULT_VARIANT)
        # For an engineless engine (e.g. serverless) the stored surface IS the
        # engine token; _surface(engine) ignores image_tag, so passing the surface
        # as image_tag with the engine reproduces the same key.
        if st == "Completed":
            set_status(p["modelId"], p["imageTag"], VERIFIED, method=method,
                       engine=engine, lora_variant=variant, job_name=p["jobName"],
                       ts=d.get("trainingEndTime"))
            resolved += 1
        elif st in ("Failed", "Stopped"):
            status, reason = classify_failure(
                p["jobName"], d.get("failureReason") or f"training {str(st).lower()}"
            )
            set_status(p["modelId"], p["imageTag"], status, method=method,
                       engine=engine, lora_variant=variant, job_name=p["jobName"],
                       reason=reason, ts=d.get("trainingEndTime"), force=True)
            resolved += 1
        # else still InProgress → leave PENDING
    return {"checked": len(pending), "resolved": resolved}


def reset_tier(image_tag: str) -> int:
    """Clear verification records for one image tier — call this after the image
    for that tier is rebuilt, so every model must re-prove itself on the new bits.
    Clears EVERY method on that tier (LoRA's bare key AND any `tag::method`
    siblings). Scoped to the DEFAULT (LLaMA-Factory) engine only: an ECR image
    rebuild is irrelevant to the serverless engine's managed recipe, so serverless
    proofs (surface "serverless") survive a tier reset. Returns records cleared."""
    doc = _all(use_cache=False)  # read-modify-write → fresh, never the shared cache
    cleared = 0
    for model_id in list(doc.keys()):
        for key in list(doc[model_id].keys()):
            surface, _method, engine, _variant = _split_key(key)
            if engine == DEFAULT_ENGINE and surface == image_tag:
                del doc[model_id][key]
                cleared += 1
        if not doc[model_id]:
            del doc[model_id]
    if cleared:
        _write(doc)
    return cleared


def model_status_map(model_id: str) -> dict[str, dict[str, Any]]:
    """All known (tier, method) records for a model, e.g.
    {"stable": {...}, "stable::qlora": {...}, "latest": {...}}. Always includes the
    model's known tiers on LoRA (from image_tiers) so the catalog UI can show the
    full grid; QLoRA (and any future method) keys appear only once recorded.
    Resolution per key: store record > shipped seed (LoRA only) > UNTESTED stub."""
    recorded = _all().get(model_id, {})
    out: dict[str, dict[str, Any]] = {}
    for tier in image_tiers():
        out[tier] = get_status(model_id, tier)
    # Serverless-mapped models get a "serverless" verification surface too — a REAL
    # engine axis (model, sagemaker_serverless, …), distinct from the LLaMA-Factory
    # image tiers. A completed serverless race autopromotes it (race._autopromote),
    # so this shows verified/untested for real (no longer a can-never-promote stub).
    # Only for models that actually offer the serverless engine.
    try:
        from .catalog import get_model

        spec = get_model(model_id)
        if spec is not None and getattr(spec, "serverless_model_id", ""):
            # engine="sagemaker_serverless" → surface "serverless"; same key the
            # recorded-keys loop below would yield, so setdefault won't duplicate.
            out["serverless"] = get_status(model_id, "serverless",
                                           engine="sagemaker_serverless")
    except Exception:  # noqa: BLE001 — never let catalog lookup break the map
        pass
    # Include any recorded keys not already present (QLoRA siblings, retired tiers,
    # serverless::* if ever added).
    for key, rec in recorded.items():
        out.setdefault(key, rec)
    return out


def is_verified(model_id: str, image_tag: str | None, method: str = DEFAULT_METHOD,
                engine: str = DEFAULT_ENGINE, lora_variant: str = DEFAULT_VARIANT) -> bool:
    """Convenience for the race-picker default filter."""
    return get_status(model_id, image_tag or DEFAULT_IMAGE_TIER, method,
                      engine, lora_variant)["status"] == VERIFIED


def all_verifications() -> dict[str, Any]:
    """The whole verification map (for the catalog page + image-status counts).
    Shipped seed baseline first, then this account's own store records layered ON
    TOP (store wins per (model, tier)) — so a fresh deployment sees the shipped
    baseline plus whatever it has run or overridden locally."""
    from .verification_seed import all_seed

    merged: dict[str, Any] = {}
    for model_id, tiers in all_seed().items():
        merged[model_id] = dict(tiers)
    for model_id, tiers in _all().items():
        merged.setdefault(model_id, {})
        merged[model_id].update(tiers)  # store overrides the seed
    return merged


def backfill_from_races() -> dict[str, Any]:
    """Seed verification from existing race history — the catalog self-heals from
    what's already happened. Any race entry whose training reached DONE/EVAL
    stages (i.e. its train job completed) marks (model, model's tier) verified.

    Deterministic + read-only against AWS (uses persisted race state, not live
    SageMaker). Idempotent thanks to set_status's authority rank. Returns a small
    summary {scanned, promoted}."""
    from .catalog import get_model
    from .race import DONE, EVALUATING, EVAL_PENDING, _load
    from .store import get_store

    # States that imply the TRAINING job completed (so the model loaded + trained
    # on its image — proof for that tier).
    trained_states = {EVAL_PENDING, EVALUATING, DONE}
    promoted = 0
    scanned = 0
    for race_id in get_store().list_keys("races"):
        race = _load(race_id)
        if race is None:
            continue
        for entry in race.entries:
            scanned += 1
            if entry.state not in trained_states or not entry.train_job:
                continue
            model = get_model(entry.model_id)
            tag = getattr(model, "image_tag", "stable") if model else "stable"
            # A completed run proves the EXACT (engine, method, variant) it used — a
            # QLoRA success proves QLoRA not LoRA; a DoRA success proves DoRA not
            # plain LoRA; a serverless success proves the serverless surface, not the
            # LLaMA-Factory image tier. Read all from the entry's persisted hp
            # (defaults for pre-engine/pre-method/pre-variant races → plain LoRA).
            method = (entry.hp or {}).get("finetuning_type", DEFAULT_METHOD)
            engine = (entry.hp or {}).get("engine", DEFAULT_ENGINE)
            variant = (entry.hp or {}).get("lora_variant", DEFAULT_VARIANT)
            before = get_status(entry.model_id, tag, method, engine, variant)["status"]
            set_status(entry.model_id, tag, VERIFIED, method=method, engine=engine,
                       lora_variant=variant, job_name=entry.train_job)
            if before != VERIFIED:
                promoted += 1
    return {"scanned": scanned, "promoted": promoted}
