# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Fine-tuning race — the platform's core value-add.

Train N models in parallel on the SAME train/eval split, auto-eval each on the
SAME held-out set as it finishes, and rank them by eval metric to pick a winner.
This is the orchestration + comparison layer LLaMA-Factory itself doesn't give
you: parallel headless fan-out, auto instance pick per model, one shared eval
harness, deterministic.

Each race entry is a small state machine advanced by `reconcile_race` (called by
a background loop and on status reads):

  PENDING ──launch train──▶ TRAINING ──train done──▶ EVAL_PENDING
     │                                                    │
     │                                              launch eval
     ▼                                                    ▼
  FAILED ◀── any job Failed ──────────────────────▶ EVALUATING ──eval done──▶ DONE(metrics)

State persists to data/races/<race_id>/race.json so the race survives backend
restarts and truly runs headless (no UI required).

Costs money: launches one training job + one eval job per model. Triggered only
by the explicit /api/race action.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .catalog import DecodingParams, Hyperparams, _instance_for, get_model
from .dispatch import dispatch_worker
from .limits import check_race_launch, max_global_concurrent_races
from .obs import log_event
from .orchestrate import (
    describe_job,
    fetch_metrics,
    launch_base_eval_job,
    launch_eval_job,
    launch_training_job,
    snapshot_curves_if_terminal,
    stop_training_job,
)
from .reward_curve import fetch_reward_curve
from .storage import split_dir
from .store import get_store

# Collection name within the state store (was the data/races directory).
RACES = "races"
RACE_FILE = "race.json"

# Entry lifecycle states.
PENDING = "pending"
LAUNCHING = "launching"  # serverless entry: launch dispatched to the worker, no job yet
TRAINING = "training"
EVAL_PENDING = "eval_pending"
EVALUATING = "evaluating"
DONE = "done"
FAILED = "failed"

TERMINAL = {DONE, FAILED}

# Completion-notification recipients: keep it small + sane so a typo or a paste of
# junk can't bloat the race doc or spam SES. Light syntactic check only (real
# deliverability is enforced by SES verification at submission time).
_MAX_NOTIFY_EMAILS = 5
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalize_notify_emails(emails: list[str] | None) -> list[str]:
    """Clean a list of notification emails: strip/lowercase, drop blanks + anything
    that isn't a plausible address, de-dupe (order-preserving), cap the count. Pure +
    total (never raises) so a bad value degrades to 'no notification', never a launch
    failure."""
    if not emails:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in emails:
        if not isinstance(raw, str):
            continue
        e = raw.strip().lower()
        # Length-cap BEFORE the regex. _EMAIL_RE's three [^@\s]+ runs are ambiguous
        # (the last two both match dots), so a long dotless string backtracks
        # quadratically. 254 is the RFC 5321 maximum for a forward path, so nothing
        # legitimate is lost and the worst case becomes trivial.
        if not e or len(e) > 254 or e in seen or not _EMAIL_RE.match(e):
            continue
        seen.add(e)
        out.append(e)
        if len(out) >= _MAX_NOTIFY_EMAILS:
            break
    return out


def entry_key_for(model_id: str, hp: dict[str, Any] | None) -> str:
    """Stable per-entry key. Bare `model_id` for the default (llama_factory +
    plain LoRA) — back-compat with races persisted before methods/engines
    existed. Otherwise the non-default axes are appended as `::<token>` so the
    SAME model can run with different methods (LoRA vs QLoRA) AND different
    engines (llama_factory vs serverless) in ONE race as distinct entries.

    Token order is method first (pre-existing scheme, unchanged), then engine,
    then stage — but stage is appended ONLY for "rlvr", so every key created
    before RLVR existed is byte-identical (sft/dpo/kto are inferred from the
    dataset shape, which is shared across a race, so they never needed a token):
      (llama_factory, lora)      -> "m"            (unchanged)
      (llama_factory, qlora)     -> "m::qlora"     (unchanged)
      (sagemaker_serverless, lora) -> "m::sagemaker_serverless"
      (sagemaker_serverless, sft) -> "m::sagemaker_serverless" (stage not in key)
      (sagemaker_serverless, rlvr) -> "m::sagemaker_serverless::rlvr"
      (sagemaker_serverless, rlaif) -> "m::sagemaker_serverless::rlaif"
    RLVR/RLAIF need their own token because the serverless engine can train
    different objectives on the same (model, method, engine) — so each must be
    able to race as a distinct leaderboard row. Only rlvr/rlaif append a stage
    token, so every pre-RLVR key stays byte-identical.

    LoRA VARIANT is a later axis: DoRA/rsLoRA/PiSSA/LoRA+ ride the same
    (model, method) but are DISTINCT runs, so the SAME model can race plain LoRA
    vs DoRA vs rsLoRA in ONE race. A non-plain variant appends `::<variant>`
    (after method/engine/stage). Plain "lora" appends nothing, so every key
    created before variants existed is byte-identical — and a `qlora` DoRA run is
    `m::qlora::dora`. Without this token two variants of the same (model, method)
    would collide on one key and silently overwrite each other.

    PREFERENCE-LOSS is the newest axis: ORPO and SimPO are stage=dpo with a
    different pref_loss (same chosen/rejected dataset, different algorithm), so
    DPO vs ORPO vs SimPO of the SAME (model, method) are DISTINCT runs. A non-default
    pref_loss (orpo/simpo) appends `::pref<loss>` ABSOLUTELY last; plain DPO
    (pref_loss=sigmoid) appends nothing, so every pre-pref_loss key stays
    byte-identical — e.g. (lora, dpo, simpo) -> "m::prefsimpo".

    KTO LOSS WEIGHTS (λ_D / λ_U) are the last axis: for a KTO run, two settings of
    kto_chosen_weight / kto_rejected_weight are DISTINCT runs (a sweep of the
    class-balance weighting), so they must be DISTINCT entries in one race. A
    non-default pair appends `::kw<chosen>x<rejected>` LAST (after the variant). The
    default 1.0/1.0 appends nothing, so every pre-existing key — and every non-KTO
    run — stays byte-identical. Without this token two weightings of the same KTO
    (model, method) collide on the race's per-entry state map ({entry_key: state})
    and one silently overwrites the other. (Like the lora_variant axis, this
    distinguishes the runs WITHIN a race; the cross-model leaderboard still dedups
    on the training job name, which doesn't carry the weights — a pre-existing
    platform limit shared by every per-entry axis, not specific to KTO.)
    """
    h = hp or {}
    method = h.get("finetuning_type", "lora")
    engine = h.get("engine", "llama_factory")
    stage = h.get("stage", "sft")
    variant = h.get("lora_variant", "lora")
    parts = [model_id]
    if method != "lora":
        parts.append(method)
    if engine != "llama_factory":
        parts.append(engine)
    if stage in ("rlvr", "rlaif"):
        parts.append(stage)
    # Variant LAST (before KTO weights) so the token order (method, engine, stage)
    # above is unchanged; only emitted for a non-plain variant (full/freeze
    # normalize to "lora", so they never add a token). Back-compat: plain-LoRA
    # keys stay bare.
    if variant and variant != "lora":
        parts.append(variant)
    # KTO loss-weight token, ABSOLUTELY last — only for a KTO run with a non-default
    # weighting, so every non-KTO key and every default-weight KTO key is unchanged.
    # `_g` strips a trailing ".0" so 3.0 → "3" (stable, no float noise in the key).
    if stage == "kto":
        cw = h.get("kto_chosen_weight", 1.0)
        rw = h.get("kto_rejected_weight", 1.0)
        if (cw, rw) != (1.0, 1.0):
            def _g(x: float) -> str:
                return f"{float(x):g}"
            parts.append(f"kw{_g(cw)}x{_g(rw)}")
    # PREFERENCE-LOSS token: ORPO and SimPO are stage=dpo with a different pref_loss
    # (the same chosen/rejected dataset, a different algorithm), so DPO vs ORPO vs
    # SimPO of the SAME (model, method) are DISTINCT runs and must be DISTINCT entries
    # in one race. Append `::pref<loss>` ONLY for stage=dpo with a non-default loss
    # (orpo/simpo). Plain DPO (pref_loss=sigmoid) appends nothing, so every pre-pref_loss
    # key — and every non-DPO run — stays byte-identical. Without this, racing DPO vs
    # ORPO vs SimPO collides them on the per-entry state map and all but one are lost.
    if stage == "dpo":
        pref_loss = h.get("pref_loss", "sigmoid")
        if pref_loss and pref_loss != "sigmoid":
            parts.append(f"pref{pref_loss}")
    return "::".join(parts)


@dataclass
class RaceEntry:
    model_id: str
    model_display: str
    instance_type: str
    hp: dict[str, Any] = field(default_factory=dict)  # this model's OWN hyperparams
    state: str = PENDING
    train_job: str | None = None
    eval_job: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    # The S3 prefix SageMaker syncs this entry's training checkpoints to (set on
    # launch for the LLaMA-Factory engine). Recorded so a "Resume from checkpoint"
    # retry can re-point a fresh job at it and pick up from the last step instead
    # of retraining from scratch. None for serverless (it manages its own state).
    checkpoint_s3: str | None = None
    # Base-model control: eval of the UNTRAINED model on the same test set, so the
    # UI can show base → fine-tuned lift (how much fine-tuning helped). Runs in
    # PARALLEL with training (no dependency on the train job). Additive + advisory
    # — never affects winner ranking (which uses the fine-tuned metrics only).
    base_eval_job: str | None = None
    base_metrics: dict[str, Any] | None = None
    # Set True once this entry has been auto-converted from spot → on-demand by the
    # capacity fallback, so the reconcile loop relaunches it AT MOST once (the new
    # on-demand job restarts the capacity-wait clock, but this flag is the explicit
    # idempotency guard). Surfaced in the UI so the cost/behavior change is visible.
    spot_fell_back: bool = False
    # UTC ISO timestamp set when a serverless entry is dispatched to the worker and
    # left LAUNCHING. Reconcile uses it to time a stuck LAUNCHING entry out to FAILED
    # if the worker never filled in train_job (a lost/early worker read, or the
    # worker can't launch) — so it self-heals instead of hanging "running" forever.
    launching_at: str | None = None

    @property
    def entry_key(self) -> str:
        return entry_key_for(self.model_id, self.hp)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entryKey"] = self.entry_key  # stable id for UI keying + per-entry ops
        return d


@dataclass
class Race:
    race_id: str
    split_id: str
    stamp: str  # caller-supplied (no time in lib code)
    decoding: dict[str, Any]  # serialized DecodingParams — SHARED (fair eval)
    entries: list[RaceEntry] = field(default_factory=list)
    name: str = ""  # human-friendly label; race_id stays the stable id
    archived: bool = False  # soft-hide from the default Races list (restorable)
    # Spot is a RACE-LEVEL cost choice (not per-model): all training jobs in the
    # race run on managed spot w/ checkpoint+resume. Changes nothing about the
    # model produced — purely cheaper-but-interruptible capacity.
    use_spot: bool = False
    # Max wall-clock per training job (persisted so retries reuse it). Default 5h.
    max_run_seconds: int = 18000
    # Spot→on-demand fallback: if set (minutes) AND use_spot, the reconcile loop
    # auto-converts a training job stuck WAITING for spot capacity past this many
    # minutes to on-demand (stop + relaunch reusing the checkpoint). None/0 = off
    # (the default — opt-in, since on-demand costs ~3x spot).
    spot_fallback_minutes: int | None = None
    # Email addresses to notify once the WHOLE race reaches a terminal state (all
    # entries done/failed). Captured at launch from the user (prefilled with their
    # Cognito email). Empty = no notification. `notified` is the exactly-once guard:
    # the reconcile loop runs repeatedly, so we send only on the FIRST tick that
    # observes a fully-finished race, then flip this True (see _maybe_notify_complete).
    notify_emails: list[str] = field(default_factory=list)
    notified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raceId": self.race_id,
            "name": self.name,
            "archived": self.archived,
            "useSpot": self.use_spot,
            "maxRunSeconds": self.max_run_seconds,
            "spotFallbackMinutes": self.spot_fallback_minutes,
            "notifyEmails": self.notify_emails,
            "notified": self.notified,
            "splitId": self.split_id,
            "stamp": self.stamp,
            "decoding": self.decoding,
            "entries": [e.to_dict() for e in self.entries],
        }


def _save(race: Race) -> None:
    store = get_store()
    wd = store.workdir(RACES, race.race_id)
    (wd / RACE_FILE).write_text(json.dumps(race.to_dict(), indent=2), encoding="utf-8")
    store.commit(RACES, race.race_id)


def _load(race_id: str) -> Race | None:
    raw = get_store().read_file(RACES, race_id, RACE_FILE)
    if raw is None:
        return None
    d = json.loads(raw)
    return Race(
        race_id=d["raceId"],
        name=d.get("name", ""),  # tolerate races persisted before naming existed
        archived=d.get("archived", False),  # tolerate pre-archive races
        use_spot=d.get("useSpot", False),  # tolerate pre-spot races
        max_run_seconds=d.get("maxRunSeconds", 18000),  # tolerate pre-maxrun races
        spot_fallback_minutes=d.get("spotFallbackMinutes"),  # tolerate pre-fallback races
        notify_emails=d.get("notifyEmails", []),  # tolerate pre-notify races
        notified=d.get("notified", False),
        split_id=d["splitId"],
        stamp=d["stamp"],
        decoding=d["decoding"],
        # Strip computed/display-only keys (e.g. entryKey) that to_dict adds but
        # aren't constructor fields, so the round-trip through storage is clean.
        entries=[RaceEntry(**{k: v for k, v in e.items() if k != "entryKey"})
                 for e in d["entries"]],
    )


def list_races() -> list[dict[str, Any]]:
    out = []
    for key in get_store().list_keys(RACES):
        race = _load(key)
        if race:
            out.append(
                {
                    "raceId": race.race_id,
                    "name": race.name,
                    "archived": race.archived,
                    "useSpot": race.use_spot,
                    "splitId": race.split_id,
                    "stamp": race.stamp,
                    "models": [e.model_id for e in race.entries],
                    # Keyed by entry_key (model_id, or model_id::method) so a
                    # same-model LoRA+QLoRA race has two DISTINCT state entries.
                    "states": {e.entry_key: e.state for e in race.entries},
                }
            )
    return sorted(out, key=lambda r: r["stamp"], reverse=True)


def _count_active_races(summaries: list[dict[str, Any]] | None = None) -> int:
    """Number of races with at least one non-terminal entry, in the CURRENT tenant
    scope (list_races is store-partitioned per tenant). Pass pre-fetched summaries
    to avoid a redundant store scan."""
    rows = summaries if summaries is not None else list_races()
    return sum(
        1 for r in rows
        if any(s not in TERMINAL for s in r.get("states", {}).values())
    )


def count_global_active_races() -> int:
    """Active races across EVERY tenant (cross-tenant total) — the input to the
    optional platform-wide concurrency cap. Mirrors the reconcile loop's tenant
    walk: the default (un-prefixed) tenant plus every per-user tenant, skipping the
    read-only samples namespace. Best-effort: a per-tenant read error is swallowed
    so the count degrades gracefully rather than blocking a launch.

    Only called when the global cap is enabled (max_global_concurrent_races() > 0),
    so the cross-tenant scan cost is never paid on a default single-tenant or
    cap-disabled deployment."""
    from .samples import SAMPLES_TENANT
    from .store import list_tenants
    from .tenancy import DEFAULT_TENANT, tenant_scope

    total = 0
    for tenant in [DEFAULT_TENANT, *list_tenants()]:
        if tenant == SAMPLES_TENANT:
            continue
        try:
            with tenant_scope(tenant):
                total += _count_active_races()
        except Exception:  # noqa: BLE001 — never let one tenant's read error block a launch
            log_event("limits.global_count.tenant_read_failed", level="WARNING", tenant=tenant)
    return total


def set_archived(race_id: str, archived: bool) -> Race | None:
    """Archive (hide) or restore a race. Pure display state — never touches the
    race's jobs or metrics. Returns the updated race, or None if not found."""
    race = _load(race_id)
    if race is None:
        return None
    race.archived = archived
    _save(race)
    return race


@dataclass
class RaceModel:
    """One model + its OWN hyperparams + optional instance override."""
    model_id: str
    hp: Hyperparams
    instance_type: str | None = None  # None → auto-pick from catalog


def start_race(
    split_id: str,
    models: list[RaceModel],
    decoding: DecodingParams,
    stamp: str,
    name: str = "",
    use_spot: bool = False,
    max_run_seconds: int = 18000,
    spot_fallback_minutes: int | None = None,
    notify_emails: list[str] | None = None,
) -> Race:
    """Launch a training job per model in parallel on the shared split.

    Each model carries its OWN hyperparameters (different families may need
    different settings). A single model is just a race of one. The chat template
    + LoRA targets always come per-model from the catalog manifest. `name` is an
    optional human-friendly label (the race_id stays the stable identifier).
    `use_spot` (race-level) runs all training jobs on managed spot w/ resume.
    """
    if split_dir(split_id) is None:
        raise ValueError(f"split {split_id} not found")
    if len(models) < 1:
        raise ValueError("need at least 1 model")

    race_id = f"race-{split_id}-{stamp}"
    entries: list[RaceEntry] = []
    for rm in models:
        model = get_model(rm.model_id)
        if model is None:
            raise ValueError(f"unknown model id: {rm.model_id}")
        method = (rm.hp.finetuning_type if rm.hp else "lora")
        # Server-side size gate (the UI Select also blocks this, but a direct API /
        # clone / programmatic call must NOT bypass it): full/freeze are offered only
        # for models whose catalog allowed_methods includes them (≤2B). Without this,
        # a full run of an oversize model would launch a billable g6e job that OOMs.
        if method in ("full", "freeze") and method not in (model.allowed_methods or ()):
            raise ValueError(
                f"{model.display_name} ({model.params_b}B) does not support "
                f"'{method}' fine-tuning — full/freeze are offered only for smaller "
                f"models that fit one GPU (allowed: {model.allowed_methods})"
            )
        # Auto instance pick. model.suggested_instance is sized for fp16 LoRA (the
        # default), but FULL-WEIGHT methods (full/freeze) need far more GPU memory,
        # so re-resolve the instance method-aware when the run isn't LoRA/QLoRA —
        # otherwise a 1.7B full-FT lands on a 24GB g5 and OOMs. An explicit
        # rm.instance_type override still wins.
        instance = rm.instance_type or (
            _instance_for(model.params_b, method)
            if method in ("full", "freeze")
            else model.suggested_instance
        )
        entries.append(
            RaceEntry(model_id=rm.model_id, model_display=model.display_name,
                      instance_type=instance, hp=asdict(rm.hp))
        )

    # Cost guardrails (raises LimitExceeded → API 400) before any billable launch.
    # active = THIS tenant's running races (store is per-tenant); global_active =
    # the cross-tenant total, computed ONLY when the optional platform-wide cap is
    # enabled (else None → the scan is skipped and the global check is a no-op).
    active = _count_active_races()
    global_active = count_global_active_races() if max_global_concurrent_races() > 0 else None
    check_race_launch(
        num_models=len(entries),
        instance_types=[e.instance_type for e in entries],
        active_race_count=active,
        global_active_race_count=global_active,
    )

    race = Race(race_id=race_id, name=name.strip(), split_id=split_id, stamp=stamp,
                decoding=asdict(decoding), entries=entries, use_spot=use_spot,
                max_run_seconds=max_run_seconds, spot_fallback_minutes=spot_fallback_minutes,
                notify_emails=_normalize_notify_emails(notify_emails))

    # Persist the race BEFORE the fan-out so a serverless worker (dispatched
    # async, InvocationType=Event) is GUARANTEED to read a race.json that already
    # contains every entry. Previously the race was only saved AFTER the inline
    # loop finished launching the other entries, so a fast worker could load a
    # race that didn't yet have its entry, hit the `entry is None` guard, return
    # in ~60ms without launching, and leave the entry stuck in LAUNCHING forever
    # (then the post-loop _save froze that stale state). See launch_serverless_entry.
    _save(race)

    # Fan out: launch all training jobs (wait=False), each with its own hp.
    #
    # LLaMA-Factory launches are fast (est.fit(wait=False) returns immediately), so
    # they run inline here. SERVERLESS launches are SLOW (a subprocess spins up the
    # V3 SDK + fetches the hub recipe + CreateTrainingJob — ~10-30s) and would blow
    # the 29s API-Gateway timeout AND block the other entries. So a serverless entry
    # is DISPATCHED to the worker Lambda (15-min budget) and left LAUNCHING; the
    # worker calls launch_serverless_entry() which fills in its train_job. Falls
    # back to inline when no worker is configured (local dev).
    for i, (rm, entry) in enumerate(zip(models, race.entries)):
        is_serverless = (rm.hp.engine == "sagemaker_serverless"
                         if hasattr(rm.hp, "engine") else
                         (entry.hp or {}).get("engine") == "sagemaker_serverless")
        if is_serverless and dispatch_worker({
            "task": "serverless_launch", "raceId": race.race_id,
            "entryKey": entry.entry_key, "stamp": f"{stamp}-{i}",
        }):
            # Dispatched off the request path; the worker will set train_job.
            from datetime import datetime, timezone

            entry.state = LAUNCHING
            entry.launching_at = datetime.now(timezone.utc).isoformat()
        else:
            try:
                res = launch_training_job(
                    model_id=entry.model_id,
                    split_id=split_id,
                    hp=rm.hp,
                    instance_type=entry.instance_type,
                    stamp=f"{stamp}-{i}",
                    use_spot=use_spot,
                    max_run_seconds=max_run_seconds,
                )
                entry.train_job = res["jobName"]
                entry.checkpoint_s3 = res.get("checkpointS3")
                entry.state = TRAINING
            except Exception as e:  # noqa: BLE001
                entry.state = FAILED
                entry.error = f"train launch failed: {e}"

        # Also launch the BASE-model eval in parallel (no dependency on training)
        # so the leaderboard can show base → fine-tuned lift. Best-effort: a base
        # launch failure must NOT fail the race entry (the fine-tuned path is what
        # matters); we just leave base_metrics empty.
        #
        # RLAIF is the exception: its held-out set is PROMPT-ONLY (no gold answer),
        # so the reference-overlap eval can't score it (and would crash on a row
        # with no assistant turn). There's also nothing to compute lift against —
        # so an RLAIF entry skips base eval entirely (it's ranked by judge reward).
        if not _entry_is_rlaif(entry):
            try:
                bres = launch_base_eval_job(
                    model_id=entry.model_id,
                    split_id=split_id,
                    decoding=decoding,
                    stamp=f"{stamp}-{i}-base",
                    instance_type=entry.instance_type,
                    max_run_seconds=max_run_seconds,
                )
                entry.base_eval_job = bres["jobName"]
            except Exception as e:  # noqa: BLE001
                log_event("race.base_eval.launch_failed", raceId=race.race_id,
                          model=entry.model_id, error=str(e))

    _save(race)
    log_event(
        "race.launch",
        raceId=race.race_id,
        name=race.name,
        splitId=split_id,
        models=[e.model_id for e in race.entries],
        launched=sum(1 for e in race.entries if e.state == TRAINING),
        failed=sum(1 for e in race.entries if e.state == FAILED),
    )
    return race


def launch_serverless_entry(race_id: str, entry_key: str, stamp: str) -> Race | None:
    """Run a serverless entry's (slow) training launch OFF the request path — called
    by the worker Lambda after start_race dispatched it. Loads the race, launches
    the job via launch_training_job (which routes to the serverless engine), and
    records train_job + TRAINING (or FAILED). Idempotent-ish: only acts on an entry
    still in LAUNCHING/PENDING so a duplicate dispatch can't double-launch."""
    race = _load(race_id)
    if race is None:
        return None
    entry = _find_entry(race, entry_key)
    if entry is None or entry.state not in (LAUNCHING, PENDING):
        return race  # already launched/failed — don't double-launch

    # Compute the launch OUTCOME into locals first; persist it in a re-load-merge
    # below. Don't save the `race` we loaded at the top: in a multi-serverless
    # race the entries launch on SEPARATE worker invocations CONCURRENTLY, each
    # doing load→mutate→save of the WHOLE race.json — so saving our stale copy
    # would clobber a SIBLING entry's just-written train_job (a lost update; the
    # observed serverless-run-1 bug where a Completed entry showed as failed).
    new_state, new_train_job, new_error = entry.state, None, None
    try:
        res = launch_training_job(
            model_id=entry.model_id,
            split_id=race.split_id,
            hp=_hp_from_dict(entry.hp),
            instance_type=entry.instance_type,
            stamp=stamp,
            use_spot=race.use_spot,
            max_run_seconds=race.max_run_seconds,
        )
        new_train_job = res["jobName"]
        new_state = TRAINING
        log_event("race.serverless.launched", raceId=race_id, model=entry.model_id,
                  trainJob=new_train_job)
    except Exception as e:  # noqa: BLE001
        new_state = FAILED
        new_error = f"train launch failed: {e}"
        log_event("race.serverless.launch_failed", level="WARNING", raceId=race_id,
                  model=entry.model_id, error=str(e))

    # Re-load the LATEST race and write ONLY this entry's fields, so a concurrent
    # sibling launch can't be clobbered. Best-effort retry to ride out a save that
    # interleaves with another worker's write.
    for _ in range(5):
        fresh = _load(race_id)
        if fresh is None:
            return race
        fe = _find_entry(fresh, entry_key)
        if fe is None:
            return fresh
        fe.state = new_state
        fe.train_job = new_train_job
        fe.error = new_error
        _save(fresh)
        # Verify our write survived (didn't lose to a racing sibling save); if it
        # did, retry the merge. Cheap re-read — the store is the source of truth.
        check = _find_entry(_load(race_id) or fresh, entry_key)
        if check is not None and check.train_job == new_train_job and check.state == new_state:
            return fresh
    return _load(race_id) or race


def _entry_is_rlaif(entry: "RaceEntry") -> bool:
    """True for an RLAIF entry (RL from AI Feedback). RLAIF's held-out set is
    prompt-only (the AI judge scores responses reference-free), so it skips the
    gold-overlap eval + base-lift path and is ranked by its judge reward instead."""
    return (entry.hp or {}).get("stage") == "rlaif"


def _entry_is_dpo_kto(entry: "RaceEntry") -> bool:
    """True for a preference entry (DPO/KTO). Its held-out gold is ONE acceptable
    answer (the chosen/desirable response), so gold-overlap metrics undercount a
    different-but-good answer — the LLM judge is the meaningful signal. Such a race
    defaults to ranking by the judge + auto-judges every done entry (not just the
    winner)."""
    return (entry.hp or {}).get("stage") in ("dpo", "kto")


def _rlaif_reward_metrics(train_job: str) -> dict[str, Any]:
    """Build a metrics dict for an RLAIF entry straight from its training
    `metrics.jsonl` (no eval job runs — there's no reference answer to score).

    The leaderboard ranks RLAIF by `reward_mean` = the FINAL judge reward. We
    prefer the held-out reward (val-core/customized/reward/mean@1, logged at eval
    steps) when present, falling back to the last training-reward step. Returns a
    metrics dict shaped like an eval result (count + the rank field) so it slots
    into the existing leaderboard/rank_entries path unchanged. Empty dict if the
    curve has no data yet (caller leaves the entry un-DONE and retries)."""
    curve = fetch_reward_curve(train_job)
    if not curve.get("hasData"):
        return {}
    val = curve.get("valReward") or []
    final_reward: float | None = None
    if val:
        final_reward = val[-1].get("value")
    if final_reward is None:
        means = [m for m in (curve.get("rewardMean") or []) if isinstance(m, (int, float))]
        final_reward = means[-1] if means else None
    if final_reward is None:
        return {}
    steps = curve.get("steps") or []
    return {
        "count": len(steps),
        "reward_mean": float(final_reward),
        # Source so the UI can label it "held-out" vs "training" reward.
        "reward_source": "held_out" if val else "training",
    }


def _hp_from_dict(d: dict[str, Any]) -> Hyperparams:
    return Hyperparams(**d)


def _decoding_from_dict(d: dict[str, Any]) -> DecodingParams:
    return DecodingParams(**d)


def _find_entry(race: "Race", entry_key: str) -> "RaceEntry | None":
    """Locate an entry by its stable entry_key. Falls back to matching model_id so
    a caller passing a bare model_id still finds the LoRA entry (back-compat)."""
    e = next((e for e in race.entries if e.entry_key == entry_key), None)
    if e is None:
        e = next((e for e in race.entries if e.model_id == entry_key), None)
    return e


def can_resume_entry(entry: "RaceEntry") -> bool:
    """True when a failed TRAINING entry can resume from a checkpoint rather than
    retrain from scratch: it must be a LLaMA-Factory training failure (not an eval
    failure) that recorded a checkpoint S3 prefix. Serverless entries (which manage
    their own state and have no checkpoint_s3) are never resumable here."""
    return bool(
        entry.state == FAILED
        and not entry.eval_job        # training failed (eval never started)
        and entry.checkpoint_s3       # a checkpoint prefix was recorded at launch
        and (entry.hp or {}).get("engine", "llama_factory") == "llama_factory"
    )


def retry_entry(race_id: str, entry_key: str, stamp: str, resume: bool = False) -> Race | None:
    """Re-launch a FAILED race entry from the stage that failed.

    - training failed (no eval job yet) → relaunch training; reconcile auto-evals.
      When `resume` is set AND the entry is resumable (LLaMA-Factory, a checkpoint
      prefix was recorded), the new job re-uses that checkpoint S3 prefix so
      SageMaker restores it and training continues from the last step instead of
      from scratch (the container entrypoint auto-detects the restored checkpoint).
    - eval failed (training had completed)→ relaunch only the eval (training output
      already in S3), saving a full re-train.
    `entry_key` is the stable per-entry id (model_id, or model_id::method).
    Returns the updated race, or None if the race/entry isn't found.
    """
    race = _load(race_id)
    if race is None:
        return None
    entry = _find_entry(race, entry_key)
    if entry is None or entry.state != FAILED:
        return race  # nothing to do

    # Filesystem-/job-name-safe suffix (entry_key may contain "::").
    safe = entry.entry_key.replace("::", "-")
    if entry.eval_job:
        # Training had produced a model; only the eval failed → re-eval.
        res = launch_eval_job(
            source_job_name=entry.train_job,
            split_id=race.split_id,
            decoding=_decoding_from_dict(race.decoding),
            stamp=f"{stamp}-retry-{safe}",
            instance_type=entry.instance_type,
            engine=(entry.hp or {}).get("engine", "llama_factory"),
        )
        entry.eval_job = res["jobName"]
        entry.state = EVALUATING
    else:
        # Training itself failed. Resume from the last checkpoint when asked AND
        # possible; otherwise relaunch from scratch (the existing behaviour).
        resume_ckpt = entry.checkpoint_s3 if (resume and can_resume_entry(entry)) else None
        res = launch_training_job(
            model_id=entry.model_id,
            split_id=race.split_id,
            hp=_hp_from_dict(entry.hp),
            instance_type=entry.instance_type,
            stamp=f"{stamp}-retry-{safe}",
            use_spot=race.use_spot,
            max_run_seconds=race.max_run_seconds,
            resume_checkpoint_s3=resume_ckpt,
        )
        entry.train_job = res["jobName"]
        # A resume keeps reading/writing the SAME checkpoint prefix; a fresh retry
        # records the new job's own prefix.
        entry.checkpoint_s3 = resume_ckpt or res.get("checkpointS3")
        entry.state = TRAINING
        log_event("race.entry.retry", raceId=race_id, model=entry.model_id,
                  resumed=bool(resume_ckpt), trainJob=entry.train_job)
    entry.error = None
    entry.metrics = None
    _save(race)
    return race


def reconcile_race(race_id: str) -> Race | None:
    """Advance every entry's state machine one step. Idempotent + safe to call
    repeatedly (background loop + on read)."""
    race = _load(race_id)
    if race is None:
        return None

    changed = False
    for entry in race.entries:
        # Base-model eval runs in PARALLEL with the train→eval pipeline (it has no
        # dependency on training), so reconcile it independently of entry.state —
        # it can complete while the entry is still TRAINING. Additive only.
        try:
            changed |= _advance_base_eval(entry)
        except Exception as e:  # noqa: BLE001 — base eval is best-effort, never fatal
            log_event("race.base_eval.error", level="WARNING", raceId=race.race_id,
                      model=entry.model_id, error=str(e))

        # Back-fill the async LLM-judge score onto DONE entries (it's written by the
        # worker after the entry finishes), so a DPO/KTO race can rank by it. Runs
        # regardless of state (a DONE entry is skipped by _advance below).
        try:
            changed |= _advance_judge(entry)
        except Exception as e:  # noqa: BLE001 — judge back-fill is best-effort
            log_event("race.judge.error", level="WARNING", raceId=race.race_id,
                      model=entry.model_id, error=str(e))

        if entry.state in TERMINAL:
            continue
        try:
            changed |= _advance(race, entry)
        except Exception as e:  # noqa: BLE001 — never let one entry break the loop
            entry.state = FAILED
            entry.error = str(e)
            changed = True

    if changed:
        _save(race)
        _maybe_autojudge_winner(race)
    # Completion email — runs every tick but is self-guarded (all-terminal + the
    # `notified` flag), so it fires exactly once on the first fully-finished tick,
    # independent of whether `changed` was set this tick. Persists its own flag.
    _maybe_notify_complete(race)
    return race


def _maybe_notify_complete(race: Race) -> None:
    """Send the one-time completion email when a race is FULLY terminal and has
    notification recipients. Exactly-once via race.notified (reconcile is called
    repeatedly). Best-effort: a notify failure must never break reconcile, and a
    SEND failure does NOT set the flag (so a transient SES error can be retried next
    tick), while a clean skip (no recipients) DOES set it so we stop checking."""
    if race.notified:
        return
    if not race.entries or any(e.state not in TERMINAL for e in race.entries):
        return  # not finished yet
    if not race.notify_emails:
        # Nothing to send — mark notified so we don't re-check this race every tick.
        race.notified = True
        _save(race)
        return
    try:
        from .notify import send_race_complete_email

        ranked = rank_entries(race)
        sent = send_race_complete_email(race, ranked)
    except Exception as e:  # noqa: BLE001 — notify is best-effort, never break reconcile
        log_event("notify.complete_failed", level="WARNING",
                  raceId=race.race_id, error=str(e))
        sent = False
    if sent:
        # Flip the flag ONLY on a successful send, so a transient failure retries.
        race.notified = True
        _save(race)
        log_event("notify.complete_marked", raceId=race.race_id)


def _advance_base_eval(entry: RaceEntry) -> bool:
    """Fetch base-model eval metrics once its job completes. Independent of the
    entry's train/eval state (base eval runs in parallel). Returns True if it
    changed anything. Never raises into the main loop (caller guards too)."""
    if not entry.base_eval_job or entry.base_metrics is not None:
        return False
    st = describe_job(entry.base_eval_job)["status"]
    if st == "Completed":
        entry.base_metrics = fetch_metrics(entry.base_eval_job)
        log_event("race.entry.base_done", model=entry.model_id)
        return True
    if st in ("Failed", "Stopped"):
        # Mark it resolved (empty) so we stop polling — base eval is advisory.
        entry.base_metrics = {}
        log_event("race.entry.base_failed", level="WARNING",
                  model=entry.model_id, status=st)
        return True
    return False


def _advance_judge(entry: RaceEntry) -> bool:
    """Back-fill the LLM-judge score onto a DONE entry's metrics so the leaderboard
    can RANK by it (the judge runs async on the worker and writes its own store
    minutes after the entry reaches DONE; this picks it up on a later reconcile).

    Stores entry.metrics['llm_judge'] as the judge's mean score NORMALIZED to 0-1
    (judgeScore is 1-5), matching every other rank metric's 0-1 scale + pct(). Only
    for DONE entries with an eval job, written once. Returns True if it set it."""
    if entry.state != DONE or not entry.eval_job or not entry.metrics:
        return False
    if entry.metrics.get("llm_judge") is not None:
        return False  # already back-filled
    try:
        from .judge import load_judge

        res = load_judge(entry.eval_job)
        score = res.get("judgeScore") if res else None
        if score is None:
            return False  # not judged yet (or empty eval) — retry next tick
        entry.metrics["llm_judge"] = float(score) / 5.0  # 1-5 → 0-1
        log_event("race.entry.judge_backfill", model=entry.model_id,
                  evalJob=entry.eval_job, judge=score)
        return True
    except Exception as e:  # noqa: BLE001 — judge back-fill is best-effort
        log_event("race.entry.judge_backfill_failed", level="WARNING",
                  model=entry.model_id, error=str(e))
        return False


def _maybe_autojudge_winner(race: Race) -> None:
    """When a race is fully finished, auto-dispatch an LLM-as-judge pass.

    SFT/RLVR/RLAIF: judge only the WINNER (the others on demand) — the judge is a
    real per-row Bedrock cost, so we don't fan it out by default.
    DPO/KTO: judge EVERY done entry — the judge is their RANK metric (gold-overlap
    scores against one acceptable answer), so all racers need a score to be ranked.
    Best-effort, idempotent (skips already-judged/in-flight). Never breaks reconcile."""
    if not race.entries or any(e.state not in TERMINAL for e in race.entries):
        return  # not finished yet
    ranked = rank_entries(race)
    judge_all = any(_entry_is_dpo_kto(e) for e in race.entries)
    if judge_all:
        targets = [r for r in ranked if r["state"] == DONE and r.get("eval_job")]
    else:
        winner = next((r for r in ranked if r.get("isWinner")), None)
        targets = [winner] if winner and winner.get("eval_job") else []
    try:
        from .judge import judge_status
        from .dispatch import dispatch_worker

        for t in targets:
            # Idempotent: skip if already judged or in flight.
            if judge_status(t["eval_job"])["status"] in ("none", "failed"):
                dispatch_worker({"task": "judge", "evalJob": t["eval_job"]})
                log_event("judge.autodispatch", raceId=race.race_id,
                          model=t["model_id"], evalJob=t["eval_job"])
    except Exception as e:  # noqa: BLE001 — auto-judge is best-effort
        log_event("judge.autodispatch_failed", level="WARNING",
                  raceId=race.race_id, error=str(e))


def _autopromote(race: Race, entry: RaceEntry) -> None:
    """Mark (model, image_tag) VERIFIED when its training job completes.

    This is the self-healing heart of the multi-image design: the catalog learns
    which models work on which image purely from real run history — no separate
    bookkeeping. Any successful training run on a tier proves that model on that
    tier (auto-promote on ANY success, per the design). Best-effort: never breaks
    reconcile. The (model→tier) comes from the catalog manifest; the timestamp is
    the job's end time (an I/O fact, not library-generated time)."""
    try:
        from .verifications import VERIFIED, set_status

        model = get_model(entry.model_id)
        # Verification is keyed on ENGINE as a first-class axis. set_status resolves
        # the surface: for LLaMA-Factory that's the ECR image tier (stable/latest);
        # for the serverless engine (no image of ours) it's the fixed "serverless"
        # surface. Passing the engine explicitly (instead of overloading image_tag)
        # keeps a serverless proof DISTINCT from an LLaMA-Factory proof and means an
        # image-tier reset never wipes serverless proofs.
        engine = (entry.hp or {}).get("engine", "llama_factory")
        image_tag = getattr(model, "image_tag", "stable") if model else "stable"
        # Promote the EXACT method this entry trained with — a QLoRA success proves
        # QLoRA, not plain LoRA (different load path). Falls back to lora for
        # entries persisted before the method field existed.
        method = (entry.hp or {}).get("finetuning_type", "lora")
        # Likewise promote the EXACT LoRA variant — a DoRA success proves DoRA, not
        # plain LoRA (DoRA/PiSSA change training + the merge). Without this, a
        # completed DoRA run autopromoted to the BARE LoRA key, so DoRA silently
        # inherited (and polluted) plain LoRA's badge. Defaults to "lora" for
        # pre-variant entries; full/freeze normalize to "lora" inside set_status.
        variant = (entry.hp or {}).get("lora_variant", "lora")
        ts = None
        try:
            # Module-level describe_job (already imported) so it's the same call
            # reconcile uses — and is stubbed together in tests.
            ts = describe_job(entry.train_job).get("trainingEndTime")
        except Exception:  # noqa: BLE001 — ts is cosmetic; don't fail promotion
            pass
        set_status(entry.model_id, image_tag, VERIFIED, method=method, engine=engine,
                   lora_variant=variant, job_name=entry.train_job, ts=ts)
        log_event("verify.autopromote", raceId=race.race_id, model=entry.model_id,
                  imageTag=image_tag, engine=engine, method=method, loraVariant=variant,
                  trainJob=entry.train_job)
    except Exception as e:  # noqa: BLE001 — auto-promote is best-effort
        log_event("verify.autopromote_failed", level="WARNING", raceId=race.race_id,
                  model=entry.model_id, error=str(e))


def _snapshot_curve(race: Race, entry: RaceEntry) -> None:
    """Persist the entry's training curve durably (best-effort) when training
    ends — so it survives CloudWatch down-sampling/expiry. Never breaks reconcile."""
    if not entry.train_job:
        return
    try:
        if snapshot_curves_if_terminal(entry.train_job):
            log_event("curve.snapshot", raceId=race.race_id, model=entry.model_id,
                      trainJob=entry.train_job)
    except Exception as e:  # noqa: BLE001 — snapshot is best-effort
        log_event("curve.snapshot_failed", level="WARNING", raceId=race.race_id,
                  model=entry.model_id, error=str(e))


def _maybe_spot_fallback(race: Race, entry: RaceEntry, info: dict[str, Any]) -> bool:
    """If a spot training job is stuck WAITING for capacity past the race's
    spot_fallback_minutes, stop it and relaunch on-demand reusing the checkpoint.

    Returns True if it converted the entry (state changed), else False. Best-effort:
    any failure is swallowed (the job keeps waiting — never breaks reconcile).

    Mis-fire guards (all must hold), tuned to NEVER kill a job that's actually
    making progress:
      * the race opted in (spot_fallback_minutes set) AND is a spot race;
      * the entry is LLaMA-Factory with a recorded checkpoint prefix (serverless
        manages its own capacity/state — excluded, like can_resume_entry);
      * not already fallen back (idempotency — at most one conversion per entry);
      * the container has NOT started yet (trainingStartTime is None) — a job that
        reached the training loop is making progress, never convert it;
      * the secondary status / failure reason looks like a capacity stall (reuse
        selfheal's capacity signature — NOT a generic slow image pull);
      * elapsed since the job was CREATED exceeds the threshold.
    """
    minutes = race.spot_fallback_minutes
    if not minutes or not race.use_spot:
        return False
    if entry.spot_fell_back:
        return False
    if (entry.hp or {}).get("engine", "llama_factory") != "llama_factory":
        return False
    if not entry.checkpoint_s3:
        return False
    # Only act while still PROVISIONING (container not started). A job that has
    # begun training has trainingStartTime set — leave it alone.
    if info.get("trainingStartTime"):
        return False
    # Capacity-stall signature (shared with selfheal so the two agree on what
    # "waiting for spot capacity" looks like).
    from .selfheal import is_capacity_stall

    sig = f"{info.get('secondaryStatus') or ''} {info.get('failureReason') or ''}"
    if not is_capacity_stall(sig):
        return False
    # Elapsed since the job was created (the capacity-wait clock).
    created = info.get("creationTime")
    if not created:
        return False
    try:
        from datetime import datetime, timezone

        # CreationTime stringifies as an ISO-8601/aware datetime; parse defensively.
        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00").split("+")[0] + "+00:00") \
            if "+" not in created and "Z" not in created else datetime.fromisoformat(created.replace("Z", "+00:00"))
        elapsed_min = (datetime.now(timezone.utc) - created_dt).total_seconds() / 60.0
    except Exception:  # noqa: BLE001 — unparseable timestamp → don't act
        return False
    if elapsed_min < minutes:
        return False

    # Convert: stop the stuck spot job, relaunch on-demand reusing the checkpoint.
    try:
        stop_training_job(entry.train_job)
        # A job stopped while still WAITING for capacity has no checkpoint yet, so
        # resume just starts fresh — acceptable (nothing was lost). If it HAD a
        # checkpoint, resume picks up from it.
        safe = entry.entry_key.replace("::", "-")
        res = launch_training_job(
            model_id=entry.model_id,
            split_id=race.split_id,
            hp=_hp_from_dict(entry.hp),
            instance_type=entry.instance_type,
            stamp=f"{race.stamp}-{safe}-ondemand",
            use_spot=False,  # the whole point: on-demand capacity
            max_run_seconds=race.max_run_seconds,
            resume_checkpoint_s3=entry.checkpoint_s3,
        )
        entry.train_job = res["jobName"]
        entry.checkpoint_s3 = entry.checkpoint_s3 or res.get("checkpointS3")
        entry.spot_fell_back = True
        entry.state = TRAINING  # stays training, now on-demand
        log_event("race.entry.spot_fallback", raceId=race.race_id, model=entry.model_id,
                  trainJob=entry.train_job, waitedMinutes=round(elapsed_min, 1))
        return True
    except Exception as e:  # noqa: BLE001 — never break reconcile on a fallback hiccup
        log_event("race.entry.spot_fallback_failed", level="WARNING",
                  raceId=race.race_id, model=entry.model_id, error=str(e))
        return False


# How long a serverless entry may sit in LAUNCHING before reconcile gives up on
# it. The worker normally fills in train_job within seconds; 20 min is generous
# headroom for a cold start, and well short of any real training run, so a stuck
# entry (lost/early worker read, or the worker can't launch) self-heals to FAILED
# instead of hanging "running" forever.
_LAUNCHING_TIMEOUT_MIN = 20.0


def _advance(race: Race, entry: RaceEntry) -> bool:
    """One transition for a single entry. Returns True if state changed."""
    # A serverless entry dispatched to the worker sits in LAUNCHING until the worker
    # sets its train_job. If the worker already did (state just lagged), pick it up;
    # otherwise time it out to FAILED so it can't hang forever (the bug behind a
    # "stuck running" run with no underlying job).
    if entry.state == LAUNCHING:
        if entry.train_job:
            entry.state = TRAINING  # worker landed the job; resume normal flow
            log_event("race.launching.recovered", raceId=race.race_id,
                      model=entry.model_id, trainJob=entry.train_job)
            return True
        from datetime import datetime, timezone

        try:
            since = datetime.fromisoformat(entry.launching_at) if entry.launching_at else None
        except (TypeError, ValueError):
            since = None
        elapsed_min = ((datetime.now(timezone.utc) - since).total_seconds() / 60.0
                       if since else _LAUNCHING_TIMEOUT_MIN + 1)  # no stamp → eligible to fail
        if elapsed_min >= _LAUNCHING_TIMEOUT_MIN:
            entry.state = FAILED
            entry.error = ("serverless launch never started (the worker didn't record a "
                           "training job within {:.0f} min) — retry the entry").format(_LAUNCHING_TIMEOUT_MIN)
            log_event("race.launching.timeout", level="WARNING", raceId=race.race_id,
                      model=entry.model_id, elapsedMin=round(elapsed_min, 1))
            return True
        return False  # still within grace — wait for the worker

    if entry.state == TRAINING:
        info = describe_job(entry.train_job)
        st = info["status"]
        if st == "Completed":
            # Train done → snapshot the (now final) reward/loss curve durably and
            # record the (model, image) as verified (auto-promote).
            _snapshot_curve(race, entry)
            _autopromote(race, entry)
            # RLAIF has NO reference eval: its held-out set is prompt-only and the
            # reward IS the AI judge's score. So instead of launching a gold-overlap
            # eval (which would crash on a prompt-only row), read the final judge
            # reward from the training metrics.jsonl and go straight to DONE, ranked
            # by reward_mean. The reward file lags the job's terminal status by a
            # moment, so an empty result just leaves the entry TRAINING for the next
            # reconcile tick (no state change) rather than producing a 0-metric DONE.
            if _entry_is_rlaif(entry):
                metrics = _rlaif_reward_metrics(entry.train_job)
                if not metrics:
                    return False  # reward curve not flushed yet — retry next tick
                entry.metrics = metrics
                entry.state = DONE
                log_event("race.entry.done", raceId=race.race_id, model=entry.model_id,
                          objective="rlaif", rewardMean=metrics.get("reward_mean"))
                return True
            res = launch_eval_job(
                source_job_name=entry.train_job,
                split_id=race.split_id,
                decoding=_decoding_from_dict(race.decoding),
                # entry_key (model_id or model_id::method) → unique per entry so a
                # same-model LoRA+QLoRA race doesn't collide eval-job stamps.
                stamp=f"{race.stamp}-{entry.entry_key.replace('::', '-')}",
                instance_type=entry.instance_type,
                # Engine-aware eval: serverless artifacts live in a different
                # subdir + need OUR eval image. Default None → unchanged LF path.
                engine=(entry.hp or {}).get("engine", "llama_factory"),
            )
            entry.eval_job = res["jobName"]
            entry.state = EVALUATING
            log_event("race.entry.train_done", raceId=race.race_id,
                      model=entry.model_id, evalJob=entry.eval_job)
            return True
        if st in ("Failed", "Stopped"):
            # Snapshot whatever curve was logged before the failure (useful for
            # diagnosing why it failed), then mark failed.
            _snapshot_curve(race, entry)
            entry.state = FAILED
            entry.error = f"training {st.lower()}"
            log_event("race.entry.failed", level="WARNING", raceId=race.race_id,
                      model=entry.model_id, stage="training", status=st)
            return True
        # Still InProgress → check the spot→on-demand capacity fallback.
        return _maybe_spot_fallback(race, entry, info)

    if entry.state == EVALUATING:
        st = describe_job(entry.eval_job)["status"]
        if st == "Completed":
            entry.metrics = fetch_metrics(entry.eval_job)
            entry.state = DONE
            log_event("race.entry.done", raceId=race.race_id, model=entry.model_id)
            return True
        if st in ("Failed", "Stopped"):
            entry.state = FAILED
            entry.error = f"eval {st.lower()}"
            log_event("race.entry.failed", level="WARNING", raceId=race.race_id,
                      model=entry.model_id, stage="eval", status=st)
            return True
        return False

    return False


# Deterministic metrics available to rank a race by (higher is better). The UI
# lets the user pick which one decides the winner. `token_f1` is the default —
# it's the most informative for loosely-formatted answers (exact/normalized are
# often 0 when models wrap the answer in prose). `reward_mean` is the RLAIF judge
# reward (reference-free) — the ONLY meaningful metric for an RLAIF race, since
# there's no gold answer for the overlap metrics to score against.
RANK_METRICS = [
    "token_f1",
    "rouge_l",
    "char_f1",
    "contains_gold",
    "normalized_match",
    "exact_match",
    "reward_mean",
    "llm_judge",
]
DEFAULT_RANK_METRIC = "token_f1"
# The rank metric for an RLAIF race (judge reward). RLAIF races are single-
# objective — the dataset shape gates the stage — so a race is either all-RLAIF
# (ranked by reward) or has none (ranked by a gold-overlap metric).
RLAIF_RANK_METRIC = "reward_mean"
# The rank metric for a preference race (DPO/KTO): the LLM-judge score (0-1,
# back-filled into entry.metrics['llm_judge'] from the judge store). Gold-overlap
# scores against ONE acceptable answer, so the judge is the meaningful default.
DPO_KTO_RANK_METRIC = "llm_judge"


def effective_rank_metric(race: Race, requested: str = "") -> str:
    """Resolve the metric a race is actually ranked by:
      * RLAIF race  → always reward_mean (prompt-only; no gold-overlap exists).
      * DPO/KTO race → defaults to llm_judge (overlap is vs one acceptable answer),
        but honours an explicit gold-overlap request so a user can still inspect it.
      * any other   → the requested metric, clamped to RANK_METRICS (default token_f1).
    `requested` empty/unset means "no explicit user choice" → apply the per-objective
    default. reward_mean is RLAIF-only; rejected elsewhere to avoid an all-blank board."""
    if any(_entry_is_rlaif(e) for e in race.entries):
        return RLAIF_RANK_METRIC
    explicit = bool(requested) and requested in RANK_METRICS and requested != RLAIF_RANK_METRIC
    if explicit:
        return requested
    # No (valid) explicit choice → preference races default to the judge.
    if any(_entry_is_dpo_kto(e) for e in race.entries):
        return DPO_KTO_RANK_METRIC
    return DEFAULT_RANK_METRIC


def rank_entries(race: Race, rank_metric: str = "") -> list[dict[str, Any]]:
    """Return entries enriched with rankScore for the chosen metric, best-first
    among DONE entries. Empty rank_metric → the per-objective default."""
    rank_metric = effective_rank_metric(race, rank_metric)
    rows = []
    for e in race.entries:
        score = e.metrics.get(rank_metric) if e.metrics else None
        # canResume drives the UI's "Resume from checkpoint" affordance: a failed
        # LLaMA-Factory TRAINING run that recorded a checkpoint prefix can continue
        # from its last step rather than retrain from scratch.
        rows.append({**e.to_dict(), "rankScore": score, "rankMetric": rank_metric,
                     "canResume": can_resume_entry(e)})
    done = [r for r in rows if r["state"] == DONE and r["rankScore"] is not None]
    rest = [r for r in rows if not (r["state"] == DONE and r["rankScore"] is not None)]
    done.sort(key=lambda r: r["rankScore"], reverse=True)
    for i, r in enumerate(done):
        r["isWinner"] = i == 0
    return done + rest
