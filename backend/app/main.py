# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""FastAPI app for the SLM fine-tune/eval platform — local test UI.

Covers dataset upload and chat-format validation, the held-out train/eval split
and its disjointness (eval ∩ train = ∅) assertion, manifest→YAML render,
SageMaker orchestration, vLLM eval, and the leaderboard. Keep endpoints thin; domain logic lives in dedicated modules
(e.g. `validation`, `split`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Literal

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from .aws_config import (
    AwsAccountUnresolvedError,
    load_aws_config,
    resolve_profile,
    resolve_region,
    save_config,
)
from .baseline import baseline_status, load_baseline, run_sonnet_baseline, set_baseline_status
from .catalog import DecodingParams, Hyperparams, get_model, list_models
from .dispatch import dispatch_worker
from .judge import judge_status, load_judge, run_judge, set_judge_status
from .leaderboard import build_leaderboard, list_eval_splits
from .limits import LimitExceeded, limits_summary
from .notify import ensure_notify_recipients_verified
from .orchestrate import (
    describe_job,
    fetch_metrics,
    fetch_training_curves,
    launch_eval_job,
    launch_training_job,
    list_completed_jobs,
    preflight,
)
from .race import (
    RANK_METRICS,
    TERMINAL,
    RaceModel,
    effective_rank_metric,
    list_races,
    rank_entries,
    reconcile_race,
    retry_entry,
    set_archived,
    start_race,
)
from .render import render_all
from .secrets import hf_token_is_set, set_hf_token
from .split import SplitReport, assert_disjoint, auto_split
from .storage import list_datasets, persist_eval_only, persist_split, set_dataset_archived, split_dir, split_meta
from .validation import validate_jsonl

RECONCILE_INTERVAL_SECONDS = 20


async def _reconcile_one_tenant() -> None:
    """Advance every non-terminal race in the CURRENT tenant scope."""
    for summary in list_races():
        states = summary.get("states", {})
        if any(s not in TERMINAL for s in states.values()):
            # reconcile_race does AWS calls; run off the event loop.
            await asyncio.to_thread(reconcile_race, summary["raceId"])


async def _race_reconcile_loop() -> None:
    """Background loop: advance every non-terminal race so they progress headless
    (train→eval→done) even with no UI open. Best-effort; errors are swallowed.

    Runs with no request context, so it must reconcile each tenant explicitly:
    the default (un-prefixed) tenant always, plus every per-user tenant once
    multi-tenancy is enabled. With the flag off, list_tenants() is empty and only
    the default tenant is reconciled — identical to the original behaviour."""
    from .samples import SAMPLES_TENANT
    from .store import list_tenants
    from .tenancy import DEFAULT_TENANT, tenant_scope

    while True:
        await asyncio.sleep(RECONCILE_INTERVAL_SECONDS)
        try:
            # The sample namespace is a curated, completed showcase — NOT a real
            # user. Never reconcile it (its races are terminal + shared; advancing
            # them would mutate everyone's view).
            for tenant in [DEFAULT_TENANT, *list_tenants()]:
                if tenant == SAMPLES_TENANT:
                    continue
                with tenant_scope(tenant):
                    await _reconcile_one_tenant()
            # Verifications are global (root doc) — resolve once, tenant-agnostic.
            from .verifications import resolve_pending_verifications

            await asyncio.to_thread(resolve_pending_verifications)
        except Exception:  # noqa: BLE001 — never kill the loop
            pass


# In Lambda there is no always-on process to host a background loop — a
# scheduled reconcile Lambda (EventBridge) advances races instead. Only run the
# in-process loop in a long-lived server (the local dev box / a container).
_RUN_RECONCILE_LOOP = not os.environ.get("AWS_LAMBDA_FUNCTION_NAME")


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_race_reconcile_loop()) if _RUN_RECONCILE_LOOP else None
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


app = FastAPI(title="SLM Platform API", version="0.1.0", lifespan=_lifespan)

# The Vite dev server proxies /api, so CORS isn't strictly needed in dev, but
# allow localhost origins so the UI also works if pointed directly at the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Resolve the calling tenant once per request and bind it to the contextvar that
# TenantStore reads, so every store access in the request is scoped to that user.
# Inert until SLM_MULTI_TENANT is on: resolve_tenant_from_request returns the
# default tenant (state stays un-prefixed) regardless of the JWT until then.
@app.middleware("http")
async def _bind_tenant(request, call_next):
    from .tenancy import resolve_tenant_from_request, set_tenant

    set_tenant(resolve_tenant_from_request(request))
    return await call_next(request)

# Cap on the DECOMPRESSED dataset size (guards against OOM + zip bombs). Uploads
# are gzip-compressed in the browser to slip under API Gateway's hard 10 MB
# request limit (JSONL compresses ~10×), then transparently decompressed here —
# so this ceiling applies to the real content, not the wire payload.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# gzip streams start with these magic bytes (RFC 1952). We detect them so the
# browser can send either gzipped or plain content and we Just Work.
_GZIP_MAGIC = b"\x1f\x8b"


def _maybe_gunzip(raw: bytes) -> bytes:
    """Decompress if the payload is gzip, else return as-is. Enforces the
    decompressed-size ceiling incrementally so a small gzip 'bomb' can't expand
    into an OOM."""
    if raw[:2] != _GZIP_MAGIC:
        return raw
    import zlib

    # 16 + MAX_WBITS tells zlib to expect a gzip (not zlib) header.
    decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = bytearray()
    for chunk_start in range(0, len(raw), 1 << 20):  # 1 MB input chunks
        out += decomp.decompress(raw[chunk_start : chunk_start + (1 << 20)], MAX_UPLOAD_BYTES + 1 - len(out))
        if len(out) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"decompressed file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )
    out += decomp.flush()
    return bytes(out)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "slm-platform-backend"}


@app.post("/api/datasets/validate")
async def validate_dataset(file: UploadFile = File(...)) -> dict:
    """Validate an uploaded chat-template JSONL file.

    Returns a structured report (row counts, errors, preview) — never 500s on
    malformed dataset content; bad rows surface as errors in the body.
    """
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        raw = _maybe_gunzip(raw)  # browser may gzip to fit API Gateway's 10 MB cap
    except HTTPException:
        return {
            "valid": False,
            "totalLines": 0,
            "validRows": 0,
            "invalidRows": 0,
            "errors": [{"line": 0, "message": f"decompressed file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"}],
            "preview": [],
            "roleCounts": {},
        }
    if len(raw) > MAX_UPLOAD_BYTES:
        return {
            "valid": False,
            "totalLines": 0,
            "validRows": 0,
            "invalidRows": 0,
            "errors": [
                {
                    "line": 0,
                    "message": f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
                }
            ],
            "preview": [],
            "roleCounts": {},
        }

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "valid": False,
            "totalLines": 0,
            "validRows": 0,
            "invalidRows": 0,
            "errors": [{"line": 0, "message": "file is not valid UTF-8 text"}],
            "preview": [],
            "roleCounts": {},
        }

    report = validate_jsonl(text)
    result = report.to_dict()
    result["filename"] = file.filename
    return result


def _decode_text(raw: bytes, label: str) -> str:
    """gunzip if needed, enforce the size cap, decode UTF-8. Raises HTTP errors."""
    raw = _maybe_gunzip(raw)  # may be gzipped (browser compresses for transfer)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{label}' exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail=f"'{label}' is not valid UTF-8 text")


async def _read_text(file: UploadFile) -> str:
    """Read a multipart-uploaded file as UTF-8 text, enforcing the size cap."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    return _decode_text(raw, file.filename or "file")


def _read_upload_text(upload_id: str) -> str:
    """Read a previously presigned-uploaded S3 object as UTF-8 text. This is the
    large-file path: the browser PUT the file straight to S3 (no API Gateway 10 MB
    limit), so we fetch it here and decode it the same way."""
    from .uploads import fetch_upload_bytes

    try:
        raw = fetch_upload_bytes(upload_id, MAX_UPLOAD_BYTES + 1)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 — surface S3/fetch errors as 502
        raise HTTPException(status_code=502, detail=f"could not read upload {upload_id}: {e}")
    return _decode_text(raw, upload_id)


async def _resolve_text(file: UploadFile | None, upload_id: str | None, what: str) -> str:
    """Resolve dataset text from EITHER a multipart file (small, < ~10 MB) OR a
    presigned-S3 upload_id (large). Exactly one must be provided."""
    if upload_id:
        return _read_upload_text(upload_id)
    if file is not None:
        return await _read_text(file)
    raise HTTPException(status_code=400, detail=f"no {what} provided (file or upload id)")


class UploadUrlRequest(BaseModel):
    filename: str


@app.post("/api/datasets/upload-url")
def dataset_upload_url(req: UploadUrlRequest) -> dict:
    """Presigned S3 PUT URL for a DIRECT browser→S3 upload (bypasses API
    Gateway's 10 MB body cap, so datasets can be arbitrarily large). The browser
    PUTs the file to the returned URL, then passes the returned uploadId to a
    split/eval endpoint, which reads the object from S3 server-side."""
    from .uploads import make_upload_url
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    try:
        return make_upload_url(req.filename, stamp)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not create upload URL: {e}")


@app.get("/api/datasets")
def datasets(include_archived: bool = False) -> dict:
    """The dataset library. By default EXCLUDES archived datasets, so the
    Fine-tune + Eval pickers only ever see available ones. The Datasets master
    page passes include_archived=true to manage the full set.

    When the user has sample runs enabled, the sample datasets are unioned in
    (tagged isSample=true) so the showcase run's split is visible — which is also
    what makes its split-keyed leaderboard populate for the user."""
    from .samples import overlay_datasets

    return {"datasets": overlay_datasets(list_datasets(include_archived=include_archived))}


@app.post("/api/datasets/{split_id}/archive")
def dataset_archive(split_id: str, archived: bool = True) -> dict:
    """Archive (hide) or restore a dataset. Display-only — never deletes data
    (datasets back races + the leaderboard)."""
    if not set_dataset_archived(split_id, archived):
        raise HTTPException(status_code=404, detail=f"unknown dataset: {split_id}")
    return {"splitId": split_id, "archived": archived}


@app.get("/api/datasets/{split_id}/profile")
def dataset_profile(split_id: str, cutoff_len: int | None = None) -> dict:
    """Deterministic dataset investigation: structure, task type, output/label/
    JSON profile, quality smells, train↔eval leakage, and a RECOMMENDED eval
    strategy (which metric to rank on + why). Advisory — reads the split's files,
    never mutates. `cutoff_len` (optional) enables truncation-risk estimation.
    Backs the 'Investigate dataset' wizard."""
    from .profiler import profile_dataset

    if split_dir(split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset: {split_id}")
    try:
        return profile_dataset(split_id, cutoff_len=cutoff_len)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"profile failed: {e}")


@app.post("/api/datasets/{split_id}/investigate/questions")
def investigate_questions(split_id: str, cutoff_len: int | None = None) -> dict:
    """Agentic investigation, step 1: profile the dataset (deterministic) then ask
    the Strands agent on Bedrock AgentCore Runtime for facet-gated follow-up
    questions about the business context the data can't reveal. Returns
    {questions, summary, profile}; persisted per split. Advisory."""
    from .investigator import start_questions

    if split_dir(split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset: {split_id}")
    try:
        return start_questions(split_id, cutoff_len=cutoff_len)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"investigate questions failed: {e}")


@app.post("/api/datasets/{split_id}/investigate/proposal")
def investigate_proposal(split_id: str, body: dict = Body(...)) -> dict:
    """Agentic investigation, step 2: the user's answers to the follow-up questions
    are folded into a confirmed eval-config proposal (task type, ranking metric,
    also-watch, cutoff guidance, flagged issues) that pre-fills Fine-tune and locks
    the metric. Body: {"answers": {"q1": "...", ...}}. Persisted per split."""
    from .investigator import start_proposal

    if split_dir(split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset: {split_id}")
    answers = body.get("answers") or {}
    # An EMPTY answers object is valid: the agent can return zero follow-up
    # questions ("nothing the data can't reveal"), in which case the user proceeds
    # straight to a recommendation with no answers. Only a non-dict is rejected.
    if not isinstance(answers, dict):
        raise HTTPException(status_code=400, detail="'answers' must be an object")
    try:
        return start_proposal(split_id, answers)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"investigate proposal failed: {e}")


@app.post("/api/race/{race_id}/triage")
def triage_race_entry(race_id: str, model_id: str, body: dict = Body(default={})) -> dict:
    """Diagnose a FAILED race entry: the agent reads the job's failure reason, log
    tail, and config, and returns a plain-language diagnosis + a concrete advisory
    fix (the user decides whether to retry). `model_id` is the entry_key (model_id,
    or model_id::method) picking which entry."""
    from .investigator import start_triage
    from .race import reconcile_race, _find_entry

    race = reconcile_race(race_id)  # also refreshes state so failure info is current
    if race is None:
        raise HTTPException(status_code=404, detail=f"unknown race: {race_id}")
    entry = _find_entry(race, model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"entry {model_id} not in race")
    job = entry.train_job or entry.eval_job
    if not job:
        raise HTTPException(status_code=400, detail="entry has no job to diagnose yet")
    try:
        # Async: triage gathers logs + calls the agent (can exceed API GW's 29s),
        # so dispatch to the worker + poll, per AWS best practice. Local runs inline.
        return start_triage(race_id, entry.entry_key, job, entry.error)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"triage failed: {e}")


@app.get("/api/race/{race_id}/triage/{model_id}")
def triage_race_status(race_id: str, model_id: str) -> dict:
    """Poll a triage run: {status, result?}."""
    from .investigator import load_triage, triage_status

    return {**triage_status(race_id, model_id), "result": load_triage(race_id, model_id)}


# Presets retired from the catalog → the surviving preset to substitute when
# cloning a historical RLVR run, so the seeded form is valid + launchable. Only
# 'prime_code' was dropped (no pure-python test-passing reward); prime_math is the
# closest surviving verifiable reward. A still-valid preset passes through unchanged.
_RETIRED_PRESET_REWARDS = {"prime_code": "prime_math"}


def _clone_preset_reward(preset: str) -> str:
    """Map a cloned run's stored preset reward to a currently-valid one (retired
    presets → their substitute), else pass it through. Pure; clone-time only."""
    return _RETIRED_PRESET_REWARDS.get(preset, preset)


@app.get("/api/race/{race_id}/clone-config")
def clone_race_config(race_id: str) -> dict:
    """Reconstruct a run's launch config so the UI can pre-fill the Fine-Tune
    builder with the SAME dataset + models/hyperparameters — then the user edits
    (add/remove models, swap dataset, tweak hp) and submits as a NEW run.

    We deliberately do NOT mutate the original race (its entries are tied to real
    SageMaker jobs + leaderboard rows). Clone is pure read → seed the builder; the
    new run goes through the normal /api/race path with all its guards. Each
    entry's full hp (engine/stage/method/lora/…) is replayed into a
    RaceModelConfig-shaped dict (camelCase aliases the UI already speaks)."""
    from .race import _load

    race = _load(race_id)
    if race is None:
        raise HTTPException(status_code=404, detail=f"unknown race: {race_id}")

    models = []
    for e in race.entries:
        hp = e.hp or {}
        models.append({
            "modelId": e.model_id,
            "instanceType": e.instance_type,
            "engine": hp.get("engine", "llama_factory"),
            "stage": hp.get("stage", "sft"),
            "prefBeta": hp.get("pref_beta", 0.1),
            # Preference-loss family + SimPO margin, so cloning an ORPO/SimPO run
            # keeps the loss instead of silently downgrading to plain DPO (sigmoid).
            "prefLoss": hp.get("pref_loss", "sigmoid"),
            "simpoGamma": hp.get("simpo_gamma", 0.5),
            # KTO per-class weights, so cloning a re-weighted KTO run keeps them.
            "ktoChosenWeight": hp.get("kto_chosen_weight", 1.0),
            "ktoRejectedWeight": hp.get("kto_rejected_weight", 1.0),
            # Efficiency knobs, so a cloned run keeps NEFTune/Liger/packing.
            "neftuneNoiseAlpha": hp.get("neftune_noise_alpha", 0.0),
            "enableLigerKernel": hp.get("enable_liger_kernel", False),
            "packing": hp.get("packing", False),
            # Map a dropped preset (the retired 'prime_code') to the nearest surviving
            # one so cloning a historical RLVR run lands on a VALID, launchable form
            # value instead of a stale option the picker can't show / the launch
            # rejects. The user sees the substitute selected and can change it before
            # launching (it's a clone-time form default, not a silent retrain).
            "presetRewardFunction": _clone_preset_reward(hp.get("preset_reward_function", "")),
            # Without this a cloned RLVR run that used a CUSTOM reward silently
            # downgrades to the gsm8k preset (the FE reads rewardFunctionId).
            "rewardFunctionId": hp.get("reward_function_id", ""),
            "finetuningType": hp.get("finetuning_type", "lora"),
            "loraRank": hp.get("lora_rank", 8),
            "loraAlpha": hp.get("lora_alpha"),
            # LoRA variant + LoRA+ ratio, so cloning a DoRA/rsLoRA/PiSSA/LoRA+ run
            # keeps the variant instead of silently downgrading to plain LoRA.
            "loraVariant": hp.get("lora_variant", "lora"),
            "loraplusLrRatio": hp.get("loraplus_lr_ratio", 16.0),
            "learningRate": hp.get("learning_rate", 1.0e-4),
            "numTrainEpochs": hp.get("num_train_epochs", 3.0),
            "perDeviceTrainBatchSize": hp.get("per_device_train_batch_size", 1),
            "gradientAccumulationSteps": hp.get("gradient_accumulation_steps", 8),
            "cutoffLen": hp.get("cutoff_len"),
            "saveSteps": hp.get("save_steps", 500),
            "maxSamples": hp.get("max_samples"),
            "earlyStoppingEnabled": hp.get("early_stopping_enabled", False),
            "earlyStoppingPatience": hp.get("early_stopping_patience", 3),
        })
    dec = race.decoding or {}
    return {
        "splitId": race.split_id,
        "name": (race.name or race.race_id) + " (clone)",
        "useSpot": race.use_spot,
        "maxRunSeconds": race.max_run_seconds,
        "spotFallbackMinutes": race.spot_fallback_minutes,
        "evalMaxNewTokens": dec.get("max_new_tokens", 256),
        "evalTemperature": dec.get("temperature", 0.0),
        "models": models,
    }


@app.get("/api/race/{race_id}/export/{model_id}")
def export_model_info(race_id: str, model_id: str, license_accepted: bool = False) -> dict:
    """Export manifest + a presigned weights URL for a race entry's fine-tune, so
    the user can deploy it in their own AWS account. License-driven: gated adapter
    bases return adapter-only, ungated return the merged standalone model. A gated
    FULL/FREEZE fine-tune ships merged weights of the gated base, so its download is
    withheld until `license_accepted=true` (the user accepts the base license)."""
    from .export import ExportError, export_info

    try:
        return export_info(race_id, model_id, license_accepted=license_accepted)
    except ExportError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"export failed: {e}")


@app.get("/api/race/{race_id}/export/{model_id}/bundle")
def export_model_bundle(race_id: str, model_id: str, license_accepted: bool = False):
    """Download the deploy bundle: a small zip (deploy.sh, inference.py,
    Dockerfile, README, manifest.json). Weights are NOT in the zip — deploy.sh
    pulls them via the presigned URL in manifest.json.

    A gated full/freeze fine-tune is license-gated: without license_accepted=true
    the manifest would carry no weights URL (deploy.sh would have nothing to fetch),
    so the bundle is refused with a 403 until the license is accepted."""
    from fastapi import Response

    from .bundle import build_bundle
    from .export import ExportError, export_info

    try:
        # Cheap manifest-only check first: if this artifact is license-gated and the
        # user hasn't accepted, refuse rather than hand back a download-less bundle.
        info = export_info(race_id, model_id, license_accepted=license_accepted)
        if info.get("licenseRequired"):
            raise HTTPException(
                status_code=403,
                detail=(f"This is a gated full fine-tune of {info.get('licenseModel')}. "
                        "Accept the base model's license to download the deploy bundle."),
            )
        filename, data = build_bundle(race_id, model_id, license_accepted=license_accepted)
    except HTTPException:
        raise
    except ExportError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"bundle build failed: {e}")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/leaderboard/interpret")
def interpret_leaderboard(body: dict = Body(...)) -> dict:
    """Results-interpreter agent: given a split's leaderboard + the user's stated
    priorities, recommend which model to ship, in plain language. Body:
    {"splitId": "...", "priorities": "optional free text"}. Async (worker + poll)."""
    from .investigator import start_interpret

    split_id = body.get("splitId")
    if not split_id or split_dir(split_id) is None:
        raise HTTPException(status_code=404, detail="unknown or missing splitId")
    try:
        return start_interpret(split_id, body.get("priorities", ""))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"interpret failed: {e}")


@app.get("/api/leaderboard/interpret/{split_id}")
def interpret_leaderboard_status(split_id: str) -> dict:
    """Poll an interpret run: {status, result?}."""
    from .investigator import interpret_status, load_interpret

    return {**interpret_status(split_id), "result": load_interpret(split_id)}


@app.get("/api/datasets/{split_id}/investigate")
def investigate_get(split_id: str) -> dict:
    """Fetch any persisted investigation (questions + proposal + status) for a
    split, so the wizard survives refresh/restart."""
    from .investigator import (
        investigation_status,
        load_answers,
        load_proposal,
        load_questions,
    )

    if split_dir(split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset: {split_id}")
    return {
        "status": investigation_status(split_id),
        "questions": load_questions(split_id),
        "proposal": load_proposal(split_id),
        "answers": load_answers(split_id),
    }


@app.post("/api/datasets/eval-only")
async def upload_eval_dataset(
    file: UploadFile | None = File(None),
    upload_id: str | None = Form(None),
    name: str = Form(""),
) -> dict:
    """Persist an EVAL-ONLY dataset for standalone evaluation.

    Unlike the split endpoints (which produce a train+eval split for fine-tuning),
    this takes a single JSONL of held-out rows to score a fine-tuned model on —
    no training half. Blocks if any row is invalid (same integrity bar as splits).
    Accepts a multipart `file` (small) OR a presigned-S3 `upload_id` (large)."""
    text = await _resolve_text(file, upload_id, "eval file")
    report = validate_jsonl(text)
    if report.invalid_rows > 0 or report.valid_rows == 0:
        return {
            "ok": False,
            "evalRows": report.valid_rows,
            "invalidRows": report.invalid_rows,
            "errors": [{"line": e.line, "message": e.message} for e in report.errors],
            "name": name.strip(),
        }
    # Re-parse the valid rows (validator counts; we need the objects).
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    split_id, _ = persist_eval_only(
        rows, {"name": name.strip(), "source": "eval-upload"}
    )
    _cleanup_upload(upload_id)
    return {
        "ok": True,
        "splitId": split_id,
        "evalRows": len(rows),
        "invalidRows": 0,
        "name": name.strip(),
    }


@app.post("/api/datasets/split/assert")
async def split_assert(
    train: UploadFile | None = File(None),
    eval: UploadFile | None = File(None),
    val: UploadFile | None = File(None),
    train_upload_id: str | None = Form(None),
    eval_upload_id: str | None = Form(None),
    val_upload_id: str | None = Form(None),
    val_ratio: float | None = Form(None),
    name: str = Form(""),
) -> dict:
    """Two- (or three-) file mode: assert pairwise disjointness.

    Always asserts eval ∩ train = ∅. Optionally adds a validation set used for
    in-training eval / early stopping (the held-out eval set is never used for
    stopping): either upload a `val` file (3-file mode) OR set `val_ratio` to
    auto-carve that fraction of train. `name` is a human-friendly label.
    Each file may be a multipart upload (small) OR a presigned-S3 upload id (large).
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    train_text = await _resolve_text(train, train_upload_id, "train file")
    eval_text = await _resolve_text(eval, eval_upload_id, "eval file")
    val_text = None
    if val is not None or val_upload_id:
        val_text = await _resolve_text(val, val_upload_id, "val file")
    if val_ratio is not None and not (0.0 < val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")
    report = assert_disjoint(train_text, eval_text, val_text=val_text, val_ratio=val_ratio)
    _finalize_split(report, {"name": name.strip(), "source": "assert"})
    for uid in (train_upload_id, eval_upload_id, val_upload_id):
        _cleanup_upload(uid)
    result = report.to_dict()
    result["name"] = name.strip()
    return result


@app.post("/api/datasets/split/auto")
async def split_auto(
    file: UploadFile | None = File(None),
    upload_id: str | None = Form(None),
    eval_ratio: float = Form(0.2),
    seed: int = Form(42),
    val_ratio: float | None = Form(None),
    stratify: bool = Form(False),
    name: str = Form(""),
) -> dict:
    """One-file mode: deterministically split into train/eval (disjoint by construction).

    Optionally carve `val_ratio` of the post-eval train rows as a validation set.
    `stratify` keeps each class proportionally represented across splits (label
    tasks only; falls back to random otherwise). Accepts a multipart `file`
    (small) OR a presigned-S3 `upload_id` (large).
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < eval_ratio < 1.0):
        raise HTTPException(status_code=400, detail="eval_ratio must be between 0 and 1 (exclusive)")
    if val_ratio is not None and not (0.0 < val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")
    text = await _resolve_text(file, upload_id, "dataset file")
    report = auto_split(text, eval_ratio=eval_ratio, seed=seed, val_ratio=val_ratio, stratify=stratify)
    _finalize_split(
        report,
        {"name": name.strip(), "source": "auto", "evalRatio": eval_ratio, "seed": seed,
         "stratified": stratify},
    )
    _cleanup_upload(upload_id)
    result = report.to_dict()
    result["name"] = name.strip()
    return result


@app.post("/api/datasets/preference")
async def create_preference_dataset(
    file: UploadFile | None = File(None),
    upload_id: str | None = Form(None),
    test_ratio: float = Form(0.1),
    val_ratio: float | None = Form(0.1),
    seed: int = Form(42),
    name: str = Form(""),
) -> dict:
    """Create a PREFERENCE dataset (for DPO) from a JSONL of chosen/rejected pairs.

    Each row is {messages|prompt, chosen, rejected} (flexible shapes — see
    validation.parse_preference_jsonl). The one uploaded file is split 3 ways: a
    held-out TEST set (its `chosen` responses become the generation-eval gold for
    the leaderboard — genuinely unseen in training), a VAL set (in-training DPO
    eval → the val-loss curve), and TRAIN. Defaults: 10% test / 10% val / 80% train."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if val_ratio is not None and not (0.0 < val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")
    from .validation import parse_preference_jsonl
    from .storage import persist_preference_split, preference_eval_rows

    text = await _resolve_text(file, upload_id, "preference dataset file")
    rows, report = parse_preference_jsonl(text)
    if not report.valid:
        # Surface the first few row errors so the user can fix the file.
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{report.invalid_rows} of {report.total_lines} rows invalid",
                "errors": [{"line": e.line, "message": e.message} for e in report.errors[:10]],
            },
        )
    # Unified 3-way split: hold out a genuine TEST portion, a VAL portion (val-loss
    # signal), keep the rest for TRAIN.
    train_rows, val_rows, test_rows = _three_way_split(rows, test_ratio, val_ratio or 0.0, seed)
    # Held-out TEST gold = the `chosen` responses of the test portion (messages-
    # shaped). Falls back to deriving from train only if the split yielded no test.
    eval_rows = preference_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_preference_split(
        train_rows,
        {"name": name.strip(), "source": "preference", "trainRows": len(train_rows),
         "valRows": len(val_rows) if val_rows else 0, "hasVal": bool(val_rows),
         "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    _cleanup_upload(upload_id)
    return {
        "splitId": split_id, "name": name.strip(), "shape": "preference",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalPairs": len(rows), "preview": report.preview,
    }


def _hash_order(n: int, seed: int, salt: str = "") -> list[int]:
    """Deterministic shuffle of indices 0..n-1 by a seed-salted hash (no RNG)."""
    import hashlib

    return sorted(range(n),
                  key=lambda i: hashlib.sha256(f"{seed}:{salt}:{i}".encode()).hexdigest())


def _deterministic_val_carve(rows: list[dict], val_ratio: float | None, seed: int):
    """Split rows into (train, val) by a seed-salted hash order (no RNG state).
    Returns (train_rows, val_rows|None). val_ratio falsy → (rows, None)."""
    if not val_ratio or len(rows) <= 1:
        return rows, None
    order = _hash_order(len(rows), seed)
    n_val = max(1, int(len(rows) * val_ratio))
    val_idx = set(order[:n_val])
    val_rows = [rows[i] for i in range(len(rows)) if i in val_idx] or None
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
    return train_rows, val_rows


def _three_way_split(rows: list[dict], test_ratio: float, val_ratio: float, seed: int):
    """One list → (train, val, test) by a seed-salted hash order (no RNG).

    The UNIFIED one-file split for DPO/KTO (and any future objective): hold out a
    genuine TEST portion (scored on the leaderboard), a VAL portion (early-stopping
    signal → the val-loss curve), and keep the rest for TRAIN. Disjoint by
    construction. Degrades gracefully on tiny inputs: guarantees ≥1 train row, and
    drops test/val to empty rather than starving train. test/val ratios are
    fractions of the whole. Returns (train, val|None, test|None)."""
    n = len(rows)
    if n <= 1:
        return rows, None, None
    order = _hash_order(n, seed)
    shuffled = [rows[i] for i in order]
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    # A POSITIVE ratio that rounds to 0 on a small dataset would otherwise yield no
    # held-out test → the persist layer silently derives eval gold from TRAIN (eval
    # overlaps train, inflating leaderboard metrics). Force ≥1 when there's room.
    if test_ratio > 0 and n_test == 0 and n >= 2:
        n_test = 1
    if val_ratio > 0 and n_val == 0 and n - n_test >= 2:
        n_val = 1
    # Never let test+val consume everything — always leave at least 1 train row.
    while n_test + n_val >= n and (n_test + n_val) > 0:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, (val or None), (test or None)


def _kto_three_way_split(rows: list[dict], test_ratio: float, val_ratio: float, seed: int):
    """Stratified 3-way split for KTO: carve test/val from EACH class (good/bad)
    independently so all three splits keep both classes (KTO's loss needs both to
    contrast). If a class is too small to split, that class stays entirely in
    train. Returns (train, val|None, test|None)."""
    good = [r for r in rows if r.get("kto_tag")]
    bad = [r for r in rows if not r.get("kto_tag")]
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for cls in (good, bad):
        if len(cls) < 3:  # too few to give up rows to test+val and keep a train row
            train += cls
            continue
        tr, v, te = _three_way_split(cls, test_ratio, val_ratio, seed)
        train += tr
        val += v or []
        test += te or []
    return train, (val or None), (test or None)


@app.post("/api/datasets/kto")
async def create_kto_dataset(
    file: UploadFile | None = File(None),
    upload_id: str | None = Form(None),
    test_ratio: float = Form(0.1),
    val_ratio: float | None = Form(0.1),
    seed: int = Form(42),
    name: str = Form(""),
) -> dict:
    """Create a KTO dataset from a JSONL of binary-labelled completions.

    Each row is {messages|prompt+completion, kto_tag|label} (flexible — see
    validation.parse_kto_jsonl). The one uploaded file is split 3 ways
    (stratified by good/bad so all splits keep both classes): a held-out TEST set
    (its DESIRABLE completions become the generation-eval gold for the
    leaderboard), a VAL set (in-training KTO eval → the val-loss curve), and TRAIN.
    Defaults: 10% test / 10% val / 80% train."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if val_ratio is not None and not (0.0 < val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")
    from .validation import parse_kto_jsonl
    from .storage import persist_kto_split, kto_eval_rows

    text = await _resolve_text(file, upload_id, "KTO dataset file")
    rows, report = parse_kto_jsonl(text)
    if not report.valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{report.invalid_rows} of {report.total_lines} rows invalid",
                "errors": [{"line": e.line, "message": e.message} for e in report.errors[:10]],
            },
        )
    # Guard: KTO needs BOTH classes present, else the loss has nothing to contrast.
    n_good = sum(1 for r in rows if r.get("kto_tag"))
    if n_good == 0 or n_good == len(rows):
        raise HTTPException(
            status_code=400,
            detail=("KTO needs both desirable AND undesirable examples; this file has only "
                    f"{'desirable' if n_good else 'undesirable'} ones."),
        )
    # Stratified 3-way split (both classes in every split).
    train_rows, val_rows, test_rows = _kto_three_way_split(rows, test_ratio, val_ratio or 0.0, seed)
    # Held-out TEST gold = the DESIRABLE completions of the test portion (messages-
    # shaped). Falls back to deriving from train only if the split produced no test.
    eval_rows = kto_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_kto_split(
        train_rows,
        {"name": name.strip(), "source": "kto", "trainRows": len(train_rows),
         "valRows": len(val_rows) if val_rows else 0, "hasVal": bool(val_rows),
         "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    _cleanup_upload(upload_id)
    return {
        "splitId": split_id, "name": name.strip(), "shape": "kto",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalRows": len(rows), "desirable": n_good, "preview": report.preview,
    }


@app.post("/api/datasets/rlvr")
async def create_rlvr_dataset(
    file: UploadFile | None = File(None),
    upload_id: str | None = Form(None),
    test_ratio: float = Form(0.1),
    val_ratio: float | None = Form(0.1),
    seed: int = Form(42),
    name: str = Form(""),
) -> dict:
    """Create an RLVR dataset from a JSONL of prompt + verifiable ground_truth.

    Each row is {messages:[...prompt turns], ground_truth:"..."} (a bare `prompt`
    string + `ground_truth`/`answer` is also accepted — see
    validation.parse_rlvr_jsonl). RLVR rewards VERIFIABLY-correct answers, so the
    target lives in its own `ground_truth` field (NOT a worked solution to imitate).
    The file is split 3 ways: a held-out TEST set (its ground_truth answers become
    the generation-eval gold for the leaderboard), a VAL set, and TRAIN. The chosen
    reward function (a gsm8k/prime_math preset or a custom reward) is picked at
    LAUNCH time, so the dataset itself stays reward-agnostic. Defaults: 10% test / 10% val."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if val_ratio is not None and not (0.0 < val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")
    from .validation import parse_rlvr_jsonl
    from .storage import persist_rlvr_split, rlvr_eval_rows

    text = await _resolve_text(file, upload_id, "RLVR dataset file")
    rows, report = parse_rlvr_jsonl(text)
    if not report.valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{report.invalid_rows} of {report.total_lines} rows invalid",
                "errors": [{"line": e.line, "message": e.message} for e in report.errors[:10]],
            },
        )
    train_rows, val_rows, test_rows = _three_way_split(rows, test_ratio, val_ratio or 0.0, seed)
    # Held-out TEST gold = the ground_truth answers of the test portion (messages-
    # shaped). Falls back to deriving from train only if the split produced no test.
    eval_rows = rlvr_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_rlvr_split(
        train_rows,
        {"name": name.strip(), "source": "rlvr", "trainRows": len(train_rows),
         "valRows": len(val_rows) if val_rows else 0, "hasVal": bool(val_rows),
         "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    _cleanup_upload(upload_id)
    return {
        "splitId": split_id, "name": name.strip(), "shape": "rlvr",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalRows": len(rows), "preview": report.preview,
    }


@app.post("/api/datasets/rlaif")
async def create_rlaif_dataset(
    file: UploadFile | None = File(None),
    upload_id: str | None = Form(None),
    test_ratio: float = Form(0.1),
    val_ratio: float | None = Form(0.1),
    seed: int = Form(42),
    name: str = Form(""),
) -> dict:
    """Create an RLAIF dataset from a JSONL of PROMPTS (no ground_truth).

    Each row is {messages:[...prompt turns]} (a bare `prompt` string is also
    accepted — see validation.parse_rlaif_jsonl). RLAIF trains the model to maximize
    an AI JUDGE's score (a reward PROMPT picked at launch), so there's no verifiable
    target — the data is just the prompts the model generates from. Split 3 ways
    (held-out TEST prompts for the leaderboard's judge-based eval, VAL, TRAIN).
    Defaults: 10% test / 10% val."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if val_ratio is not None and not (0.0 < val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")
    from .validation import parse_rlaif_jsonl
    from .storage import persist_rlaif_split, rlaif_eval_rows

    text = await _resolve_text(file, upload_id, "RLAIF dataset file")
    rows, report = parse_rlaif_jsonl(text)
    if not report.valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{report.invalid_rows} of {report.total_lines} rows invalid",
                "errors": [{"line": e.line, "message": e.message} for e in report.errors[:10]],
            },
        )
    train_rows, val_rows, test_rows = _three_way_split(rows, test_ratio, val_ratio or 0.0, seed)
    # Held-out TEST = the prompt-only rows of the test portion (no gold — the judge
    # scores generated responses). Falls back to deriving from train if no test.
    eval_rows = rlaif_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_rlaif_split(
        train_rows,
        {"name": name.strip(), "source": "rlaif", "trainRows": len(train_rows),
         "valRows": len(val_rows) if val_rows else 0, "hasVal": bool(val_rows),
         "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    _cleanup_upload(upload_id)
    return {
        "splitId": split_id, "name": name.strip(), "shape": "rlaif",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalRows": len(rows), "preview": report.preview,
    }


# --- Load a SAMPLE of a public Hugging Face dataset as training data ---
# Lets us prove the leaderboard discriminates models on REAL task data without a
# customer dataset. Fetches via the HF datasets-server HTTP API, converts the
# native columns to the platform's `messages` format, then feeds the EXISTING
# auto_split → _finalize_split path (nothing downstream changes).


class HFPreviewRequest(BaseModel):
    dataset: str
    config: str | None = None
    split: str | None = None


@app.post("/api/datasets/hf/preview")
def hf_preview(req: HFPreviewRequest) -> dict:
    """Inspect a HF dataset before importing: list its splits, columns, a
    suggested column→messages mapping, and a few raw + converted preview rows.
    Uses the stored HF token (if set) so gated datasets work."""
    from .hf_ingest import HFIngestError, list_splits, preview
    from .secrets import get_hf_token

    token = get_hf_token()
    try:
        prev = preview(req.dataset, config=req.config, split=req.split, token=token)
        return {
            **prev.to_dict(),
            "splits": list_splits(req.dataset, token),
        }
    except HFIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HF preview failed: {e}")


class HFImportRequest(BaseModel):
    dataset: str
    config: str
    split: str
    name: str
    # column mapping
    user_field: str = Field(alias="userField")
    target_field: str = Field(alias="targetField")
    system_field: str | None = Field(default=None, alias="systemField")
    context_field: str | None = Field(default=None, alias="contextField")
    instruction: str = ""
    # sampling + split
    max_rows: int = Field(default=500, alias="maxRows")
    seed: int = 42
    eval_ratio: float = Field(default=0.2, alias="evalRatio")
    val_ratio: float | None = Field(default=None, alias="valRatio")
    stratify: bool = False

    model_config = {"populate_by_name": True}


@app.post("/api/datasets/hf/import")
def hf_import(req: HFImportRequest) -> dict:
    """Fetch a deterministic SAMPLE of a HF dataset, convert it to chat-template
    rows, then run it through the standard auto-split → persist path. The result
    is an ordinary dataset usable everywhere (fine-tune, race, eval, investigate)."""
    from .hf_ingest import ColumnMapping, HFIngestError, sample_to_jsonl
    from .secrets import get_hf_token

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < req.eval_ratio < 1.0):
        raise HTTPException(status_code=400, detail="eval_ratio must be between 0 and 1 (exclusive)")
    if req.val_ratio is not None and not (0.0 < req.val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")

    mapping = ColumnMapping(
        user_field=req.user_field,
        target_field=req.target_field,
        system_field=req.system_field,
        context_field=req.context_field,
        instruction=req.instruction,
    )
    token = get_hf_token()
    try:
        jsonl, stats = sample_to_jsonl(
            req.dataset,
            mapping,
            config=req.config,
            split=req.split,
            max_rows=req.max_rows,
            seed=req.seed,
            token=token,
        )
    except HFIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HF import failed: {e}")

    if not jsonl.strip():
        raise HTTPException(
            status_code=400,
            detail="no rows could be converted — check the column mapping",
        )

    report = auto_split(
        jsonl, eval_ratio=req.eval_ratio, seed=req.seed, val_ratio=req.val_ratio,
        stratify=req.stratify,
    )
    _finalize_split(
        report,
        {
            "name": req.name.strip(),
            "source": "huggingface",
            "hfDataset": req.dataset,
            "hfConfig": req.config,
            "hfSplit": req.split,
            "hfSampleSeed": req.seed,
            "evalRatio": req.eval_ratio,
            "stratified": req.stratify,
        },
    )
    result = report.to_dict()
    result["name"] = req.name.strip()
    result["hfStats"] = stats
    return result


class HFPreferenceImportRequest(BaseModel):
    dataset: str
    config: str
    split: str
    name: str
    # preference column mapping
    chosen_field: str = Field(alias="chosenField")
    rejected_field: str = Field(alias="rejectedField")
    prompt_field: str | None = Field(default=None, alias="promptField")
    system_field: str | None = Field(default=None, alias="systemField")
    instruction: str = ""
    # sampling + 3-way split (held-out test + val), matching the upload path
    max_rows: int = Field(default=500, alias="maxRows")
    seed: int = 42
    test_ratio: float = Field(default=0.1, alias="testRatio")
    val_ratio: float | None = Field(default=0.1, alias="valRatio")

    model_config = {"populate_by_name": True}


@app.post("/api/datasets/hf/import-preference")
def hf_import_preference(req: HFPreferenceImportRequest) -> dict:
    """Fetch a deterministic SAMPLE of a HF PREFERENCE dataset (chosen/rejected),
    convert it to canonical ranking rows, then persist via the preference path —
    yielding a DPO-ready dataset whose held-out eval is derived from the chosen
    responses (so the shared leaderboard works unchanged)."""
    from .hf_ingest import HFIngestError, PreferenceMapping, sample_preference_to_jsonl
    from .secrets import get_hf_token
    from .storage import persist_preference_split, preference_eval_rows

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < req.test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if req.val_ratio is not None and not (0.0 < req.val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")

    mapping = PreferenceMapping(
        chosen_field=req.chosen_field,
        rejected_field=req.rejected_field,
        prompt_field=req.prompt_field,
        system_field=req.system_field,
        instruction=req.instruction,
    )
    token = get_hf_token()
    try:
        rows, stats = sample_preference_to_jsonl(
            req.dataset, mapping, config=req.config, split=req.split,
            max_rows=req.max_rows, seed=req.seed, token=token,
        )
    except HFIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HF preference import failed: {e}")

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="no preference pairs could be converted — check the chosen/rejected mapping",
        )

    # Unified 3-way split (held-out test + val), same as the upload path.
    train_rows, val_rows, test_rows = _three_way_split(rows, req.test_ratio, req.val_ratio or 0.0, req.seed)
    eval_rows = preference_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_preference_split(
        train_rows,
        {"name": req.name.strip(), "source": "huggingface", "hfDataset": req.dataset,
         "hfConfig": req.config, "hfSplit": req.split, "hfSampleSeed": req.seed,
         "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
         "hasVal": bool(val_rows), "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    return {
        "splitId": split_id, "name": req.name.strip(), "shape": "preference",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalPairs": len(rows), "hfStats": stats,
    }


class HFKtoImportRequest(BaseModel):
    dataset: str
    config: str
    split: str
    name: str
    # KTO column mapping
    completion_field: str = Field(alias="completionField")
    label_field: str = Field(alias="labelField")
    prompt_field: str | None = Field(default=None, alias="promptField")
    system_field: str | None = Field(default=None, alias="systemField")
    instruction: str = ""
    max_rows: int = Field(default=500, alias="maxRows")
    seed: int = 42
    test_ratio: float = Field(default=0.1, alias="testRatio")
    val_ratio: float | None = Field(default=0.1, alias="valRatio")

    model_config = {"populate_by_name": True}


@app.post("/api/datasets/hf/import-kto")
def hf_import_kto(req: HFKtoImportRequest) -> dict:
    """Fetch a deterministic SAMPLE of a HF dataset with a completion + a binary
    good/bad label, convert it to canonical KTO rows, then persist via the KTO
    3-way split (held-out test from the desirable completions)."""
    from .hf_ingest import HFIngestError, KtoMapping, sample_kto_to_jsonl
    from .secrets import get_hf_token
    from .storage import persist_kto_split, kto_eval_rows

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < req.test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if req.val_ratio is not None and not (0.0 < req.val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")

    mapping = KtoMapping(
        completion_field=req.completion_field,
        label_field=req.label_field,
        prompt_field=req.prompt_field,
        system_field=req.system_field,
        instruction=req.instruction,
    )
    token = get_hf_token()
    try:
        rows, stats = sample_kto_to_jsonl(
            req.dataset, mapping, config=req.config, split=req.split,
            max_rows=req.max_rows, seed=req.seed, token=token,
        )
    except HFIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HF KTO import failed: {e}")

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="no KTO rows could be converted — check the completion/label mapping",
        )
    n_good = sum(1 for r in rows if r.get("kto_tag"))
    if n_good == 0 or n_good == len(rows):
        raise HTTPException(
            status_code=400,
            detail=("KTO needs both desirable AND undesirable examples; the sampled rows have only "
                    f"{'desirable' if n_good else 'undesirable'} ones."),
        )

    # Stratified 3-way split (both classes in every split), same as the upload path.
    train_rows, val_rows, test_rows = _kto_three_way_split(rows, req.test_ratio, req.val_ratio or 0.0, req.seed)
    eval_rows = kto_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_kto_split(
        train_rows,
        {"name": req.name.strip(), "source": "huggingface", "hfDataset": req.dataset,
         "hfConfig": req.config, "hfSplit": req.split, "hfSampleSeed": req.seed,
         "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
         "hasVal": bool(val_rows), "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    return {
        "splitId": split_id, "name": req.name.strip(), "shape": "kto",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalRows": len(rows), "desirable": n_good, "hfStats": stats,
    }


class HFRlvrImportRequest(BaseModel):
    dataset: str
    config: str
    split: str
    name: str
    # RLVR column mapping: a prompt + a verifiable ground-truth answer.
    prompt_field: str = Field(alias="promptField")
    ground_truth_field: str = Field(alias="groundTruthField")
    system_field: str | None = Field(default=None, alias="systemField")
    instruction: str = ""
    max_rows: int = Field(default=500, alias="maxRows")
    seed: int = 42
    test_ratio: float = Field(default=0.1, alias="testRatio")
    val_ratio: float | None = Field(default=0.1, alias="valRatio")

    model_config = {"populate_by_name": True}


@app.post("/api/datasets/hf/import-rlvr")
def hf_import_rlvr(req: HFRlvrImportRequest) -> dict:
    """Fetch a deterministic SAMPLE of a HF dataset with a prompt + a verifiable
    answer (e.g. gsm8k's question + answer), convert it to canonical RLVR rows
    ({messages, ground_truth}), then persist via the RLVR 3-way split. The preset
    reward function is chosen at launch time, so the dataset stays reward-agnostic."""
    from .hf_ingest import HFIngestError, RlvrMapping, sample_rlvr_to_jsonl
    from .secrets import get_hf_token
    from .storage import persist_rlvr_split, rlvr_eval_rows

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < req.test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if req.val_ratio is not None and not (0.0 < req.val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")

    mapping = RlvrMapping(
        prompt_field=req.prompt_field,
        ground_truth_field=req.ground_truth_field,
        system_field=req.system_field,
        instruction=req.instruction,
    )
    token = get_hf_token()
    try:
        rows, stats = sample_rlvr_to_jsonl(
            req.dataset, mapping, config=req.config, split=req.split,
            max_rows=req.max_rows, seed=req.seed, token=token,
        )
    except HFIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HF RLVR import failed: {e}")

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="no RLVR rows could be converted — check the prompt/ground_truth mapping",
        )

    train_rows, val_rows, test_rows = _three_way_split(rows, req.test_ratio, req.val_ratio or 0.0, req.seed)
    eval_rows = rlvr_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_rlvr_split(
        train_rows,
        {"name": req.name.strip(), "source": "huggingface", "hfDataset": req.dataset,
         "hfConfig": req.config, "hfSplit": req.split, "hfSampleSeed": req.seed,
         "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
         "hasVal": bool(val_rows), "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    return {
        "splitId": split_id, "name": req.name.strip(), "shape": "rlvr",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalRows": len(rows), "hfStats": stats,
    }


class HFRlaifImportRequest(BaseModel):
    dataset: str
    config: str
    split: str
    name: str
    # RLAIF column mapping: ONLY a prompt is needed (prompt-only; the AI judge
    # reward-prompt scores the response at training time — no answer column).
    prompt_field: str = Field(alias="promptField")
    system_field: str | None = Field(default=None, alias="systemField")
    instruction: str = ""
    max_rows: int = Field(default=500, alias="maxRows")
    seed: int = 42
    test_ratio: float = Field(default=0.1, alias="testRatio")
    val_ratio: float | None = Field(default=0.1, alias="valRatio")

    model_config = {"populate_by_name": True}


@app.post("/api/datasets/hf/import-rlaif")
def hf_import_rlaif(req: HFRlaifImportRequest) -> dict:
    """Fetch a deterministic SAMPLE of a HF dataset's PROMPT column, convert it to
    canonical RLAIF rows ({messages} prompt-only), then persist via the RLAIF 3-way
    split. There's no answer/ground_truth — the AI-judge reward prompt (chosen at
    launch) scores a freshly generated response, so the dataset stays judge-agnostic."""
    from .hf_ingest import HFIngestError, RlaifMapping, sample_rlaif_to_jsonl
    from .secrets import get_hf_token
    from .storage import persist_rlaif_split, rlaif_eval_rows

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="dataset name is required")
    if not (0.0 < req.test_ratio < 1.0):
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1 (exclusive)")
    if req.val_ratio is not None and not (0.0 < req.val_ratio < 1.0):
        raise HTTPException(status_code=400, detail="val_ratio must be between 0 and 1 (exclusive)")

    mapping = RlaifMapping(
        prompt_field=req.prompt_field,
        system_field=req.system_field,
        instruction=req.instruction,
    )
    token = get_hf_token()
    try:
        rows, stats = sample_rlaif_to_jsonl(
            req.dataset, mapping, config=req.config, split=req.split,
            max_rows=req.max_rows, seed=req.seed, token=token,
        )
    except HFIngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"HF RLAIF import failed: {e}")

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="no RLAIF rows could be converted — check the prompt mapping",
        )

    train_rows, val_rows, test_rows = _three_way_split(rows, req.test_ratio, req.val_ratio or 0.0, req.seed)
    eval_rows = rlaif_eval_rows(test_rows) if test_rows else None
    split_id, _ = persist_rlaif_split(
        train_rows,
        {"name": req.name.strip(), "source": "huggingface", "hfDataset": req.dataset,
         "hfConfig": req.config, "hfSplit": req.split, "hfSampleSeed": req.seed,
         "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
         "hasVal": bool(val_rows), "testHeldOut": bool(test_rows)},
        val=val_rows,
        eval_=eval_rows,
    )
    return {
        "splitId": split_id, "name": req.name.strip(), "shape": "rlaif",
        "trainRows": len(train_rows), "valRows": len(val_rows) if val_rows else 0,
        "testRows": len(test_rows) if test_rows else 0,
        "totalRows": len(rows), "hfStats": stats,
    }


def _cleanup_upload(upload_id: str | None) -> None:
    """Delete a consumed presigned-upload staging object (best-effort)."""
    if not upload_id:
        return
    from .uploads import delete_upload

    delete_upload(upload_id)


def _finalize_split(report: SplitReport, meta: dict) -> None:
    """Persist a successful split to disk and stamp its id onto the report.

    Only disjoint splits with both halves non-empty are persisted — a failed
    disjointness check shouldn't produce a referenceable training artifact. The
    optional validation set (if any) is persisted alongside, and its presence +
    mode recorded in meta so early stopping can be gated on it.
    """
    if not report.ok:
        return
    split_id, _ = persist_split(
        report.train_rows_full,
        report.eval_rows_full,
        {
            "mode": report.mode,
            "trainRows": report.train_rows,
            "evalRows": report.eval_rows,
            "hasVal": report.has_val,
            "valMode": report.val_mode,
            "valRows": report.val_rows,
            "valRatio": report.val_ratio,
            **meta,
        },
        val=report.val_rows_full or None,
    )
    report.split_id = split_id


# --- Model catalog + LLaMA-Factory YAML render ---


@app.get("/api/models")
def get_models() -> dict:
    """List the engine-neutral model catalog + hyperparameter bounds/defaults.

    Each model is enriched with its per-image-tier verification status (so the
    UI can show which image each model is proven on, and filter the race picker
    to verified models by default)."""
    from .aws_config import image_tiers
    from .engines.base import engine_enabled
    from .verifications import model_status_map

    # The serverless engine is gated by the saved enableSagemakerServerless flag
    # (Settings toggle, default ON). When it's OFF, hide it from the catalog the
    # UI builds the engine picker from (so a model that maps to serverless looks
    # exactly like an unmapped one), AND the launch endpoint independently rejects
    # serverless requests — defense in depth: one flag drives picker and guard.
    serverless_on = engine_enabled("sagemaker_serverless")
    models = list_models()
    for m in models:
        m["verifications"] = model_status_map(m["id"])
        if not serverless_on:
            m["engines"] = [e for e in m.get("engines", ["llama_factory"])
                            if e != "sagemaker_serverless"]
    return {
        "models": models,
        "bounds": Hyperparams.bounds(),
        "imageTiers": image_tiers(),
    }


@app.get("/api/verifications")
def get_verifications() -> dict:
    """The full (model, image-tier) → status map + the tier→tag table. Backs the
    Model Catalog page."""
    from .aws_config import image_tiers
    from .verifications import all_verifications

    return {"verifications": all_verifications(), "imageTiers": image_tiers()}


@app.post("/api/verifications/backfill")
def backfill_verifications() -> dict:
    """Seed verification from existing race history (any model whose training
    completed is marked verified on its tier). Idempotent."""
    from .verifications import backfill_from_races

    return backfill_from_races()


@app.post("/api/models/{model_id}/diagnose")
def diagnose_model_failure(model_id: str, reason: str | None = None) -> dict:
    """Self-healing triage: classify a model's training failure and, if it means
    the model needs a newer software stack, recommend the next image tier (and
    whether that image is already built). Read-only — recommends, doesn't act;
    the operator (or the agent loop) then smoke-tests on the recommended tier via
    /api/models/{id}/smoke-test?image_tag=<tier>.

    `reason` (query) is the failure text; when omitted we look up the model's
    most recent failed run from race history."""
    from .selfheal import diagnose

    model = get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model {model_id}")

    failure_reason = reason
    if not failure_reason:
        # Best-effort: find the latest failed race entry for this model + its tier
        # verification reason, so the UI can call diagnose with just a model id.
        from .verifications import get_status

        rec = get_status(model_id, getattr(model, "image_tag", "stable"))
        failure_reason = rec.get("reason")
    try:
        return diagnose(model, failure_reason)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"diagnose failed: {e}")


@app.get("/api/images")
def list_images() -> dict:
    """Per-tier ECR image status for the Images management page: each tier's tag,
    whether it's in ECR (+ pushedAt/size), its CodeBuild project + last build
    status, and how many models are verified on it. Read-only."""
    from .selfheal import image_tier_status

    try:
        return {"tiers": image_tier_status()}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not list images: {e}")


@app.post("/api/images/{image_tag}/build")
def build_image(image_tag: str) -> dict:
    """Trigger the CodeBuild project that (re)builds a tier's training image.
    Returns the build id to poll. Used by the Images page Rebuild action and the
    self-healing flow when a tier's image isn't in ECR yet."""
    from .selfheal import trigger_image_build

    try:
        return trigger_image_build(image_tag)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not start image build: {e}")


@app.post("/api/images/{image_tag}/reset-verifications")
def reset_image_verifications(image_tag: str) -> dict:
    """Clear all (model, tier) verification records for one tier — call after the
    tier's image is rebuilt so every model must re-prove itself on the new bits
    (verification is tied to the image, not just the model). Returns the count
    cleared. `image_tag` here is the TIER NAME (e.g. 'latest')."""
    from .verifications import reset_tier

    cleared = reset_tier(image_tag)
    return {"tier": image_tag, "cleared": cleared}


@app.get("/api/images/check-updates")
def check_image_updates() -> dict:
    """Check Docker Hub for newer LLaMA-Factory releases than the tags we build.
    Read-only. Powers the Images page 'Check for updates' button."""
    from .releases import check_for_updates

    try:
        return check_for_updates()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"update check failed: {e}")


class BuildReleaseRequest(BaseModel):
    lf_tag: str = Field(..., alias="lfTag")  # e.g. "0.9.6"
    vllm_version: str = Field("0.8.5.post1", alias="vllmVersion")

    model_config = {"populate_by_name": True}


@app.post("/api/images/build-release")
def build_release_endpoint(req: BuildReleaseRequest) -> dict:
    """Build a specific LLaMA-Factory release via the adhoc CodeBuild project and
    register it as a new image tier at runtime (no cdk deploy). The new tier's
    models all start 'untested'; existing models stay on their proven tier."""
    from .releases import build_release

    try:
        return build_release(req.lf_tag.strip(), req.vllm_version.strip())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not start release build: {e}")


@app.get("/api/images/{image_tag}/new-models")
def new_models_for_image(image_tag: str, base_tag: str | None = None) -> dict:
    """Architectures the `image_tag` image newly supports vs an older image (its
    capability manifest minus the baseline's), mapped to suggested HF models to
    probe + smoke-test. `image_tag` is the ECR TAG (e.g. '0.9.5'). Powers the
    Model Catalog 'Find new models' button."""
    from .releases import discover_new_models

    try:
        return discover_new_models(image_tag, base_tag)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"new-model discovery failed: {e}")


@app.get("/api/models/serverless-candidates")
def serverless_candidates() -> dict:
    """Find which models are serverless-customizable on the LIVE SageMaker Public
    Hub and classify them against our catalog: untaggedMatches (a catalog row we
    can light up the serverless engine for), staleTags (a tag the hub no longer
    lists), newCandidates (customizable but not in the catalog). Suggestions only —
    applying a tag is the explicit POST below. The serverless analogue of the
    Images-page 'Find new models'. Never raises (degrades to a note)."""
    from .serverless_catalog import discover_serverless_models

    try:
        return discover_serverless_models()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"serverless discovery failed: {e}")


class ServerlessTagRequest(BaseModel):
    """Apply (or clear) a runtime serverless-tag overlay for a catalog model: the
    SageMaker Public Hub id whose V3 recipe trains it. Blank hubId clears it."""
    hub_id: str = Field("", alias="hubId")

    model_config = {"populate_by_name": True}


@app.post("/api/models/{model_id}/serverless-tag")
def set_serverless_tag(model_id: str, req: ServerlessTagRequest) -> dict:
    """Apply a discovered serverless tag (catalogId → Public Hub id) as a RUNTIME
    overlay — the serverless engine becomes available for the model with NO
    redeploy. A blank hubId clears a previously-applied overlay tag (a static
    CATALOG tag is code and cannot be cleared here). The model stays UNVERIFIED on
    the serverless surface until a smoke test proves it (verify-before-trust)."""
    from .serverless_catalog import register_serverless_model_id, unregister_serverless_model_id

    try:
        if req.hub_id.strip():
            overlay = register_serverless_model_id(model_id, req.hub_id)
        else:
            overlay = unregister_serverless_model_id(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "id": model_id, "hubId": req.hub_id.strip(), "overlay": overlay}


# --- Feedback board (issues / ideas / praise, with screenshot attachments) ---
# A shared, global board: everyone sees all entries; the submitter's stable
# identity is stamped as author. Attachments reuse the dataset presigned-upload
# flow (POST /api/datasets/upload-url → uploadId → here, which copies the image
# into a permanent per-entry prefix). See feedback.py.


@app.get("/api/feedback")
def list_feedback_ep() -> dict:
    """All feedback entries (newest-first) + the allowed type/status values, so the
    UI can render the board and its pickers. Each attachment carries a fresh
    presigned GET url."""
    from .feedback import FEEDBACK_STATUSES, FEEDBACK_TYPES, MAX_ATTACHMENTS, list_feedback

    return {
        "feedback": list_feedback(),
        "types": list(FEEDBACK_TYPES),
        "statuses": list(FEEDBACK_STATUSES),
        "maxAttachments": MAX_ATTACHMENTS,
        # The current user (so the UI can show Delete only on the viewer's own rows).
        "me": _current_author(),
    }


def _current_author() -> str:
    """The submitter's stable identity for attribution (alias/sub via the tenancy
    seam), or 'anonymous' when no auth context is configured."""
    try:
        from .tenancy import current_tenant

        return current_tenant() or "anonymous"
    except Exception:  # noqa: BLE001
        return "anonymous"


class FeedbackRequest(BaseModel):
    """Submit feedback. `attachmentUploadIds` are presigned-upload ids (from
    /api/datasets/upload-url) for image screenshots — copied into the entry."""
    type: str  # issue | idea | praise
    title: str
    body: str = ""
    attachment_upload_ids: list[str] = Field(default_factory=list, alias="attachmentUploadIds")

    model_config = {"populate_by_name": True}


@app.post("/api/feedback")
def create_feedback_ep(req: FeedbackRequest) -> dict:
    """Persist a feedback entry (author = the current user), copying any image
    attachments from the uploads prefix into the entry. 400 on a bad type/title or
    a non-image/oversized/too-many attachment."""
    from datetime import datetime, timezone

    from .feedback import FeedbackError, create_feedback

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    try:
        return create_feedback(
            type=req.type, title=req.title, body=req.body, author=_current_author(),
            attachment_upload_ids=req.attachment_upload_ids, stamp=stamp,
        )
    except FeedbackError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not save feedback: {e}")


class FeedbackStatusRequest(BaseModel):
    status: str  # open | planned | done | wont_do


@app.post("/api/feedback/{feedback_id}/status")
def set_feedback_status_ep(feedback_id: str, req: FeedbackStatusRequest) -> dict:
    """Triage an entry's lifecycle status (anyone can). 404/400 on unknown id/status."""
    from .feedback import FeedbackError, set_feedback_status

    try:
        return set_feedback_status(feedback_id, req.status)
    except FeedbackError as e:
        # "not found" → 404, otherwise a validation 400.
        code = 404 if "not found" in str(e) else 400
        raise HTTPException(status_code=code, detail=str(e))


@app.delete("/api/feedback/{feedback_id}")
def delete_feedback_ep(feedback_id: str) -> dict:
    """Delete an entry — only the author may (delete-own). 404 if missing, 403 if
    the current user isn't the author."""
    from .feedback import FeedbackError, delete_feedback

    try:
        ok = delete_feedback(feedback_id, _current_author())
    except FeedbackError as e:
        raise HTTPException(status_code=403, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"feedback not found: {feedback_id}")
    return {"ok": True, "id": feedback_id, "deleted": True}


# --- Auto-onboard a model from a Hugging Face id (Tier 1) ---


class ProbeRequest(BaseModel):
    repo: str  # e.g. "Qwen/Qwen3-4B-Instruct-2507"


@app.post("/api/models/probe")
def probe_model_endpoint(req: ProbeRequest) -> dict:
    """Fetch HF metadata/config for a repo and derive a draft model spec
    (template/cutoff/instance/params/gated). Read-only — nothing is saved; the
    user reviews + (optionally) verifies before adding it to the catalog."""
    from .onboard import probe_model

    try:
        return probe_model(req.repo.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"probe failed: {e}")


class SaveModelRequest(BaseModel):
    id: str
    display_name: str = Field(..., alias="displayName")
    hf_model_id: str = Field(..., alias="hfModelId")
    template: str
    family: str = "Custom"
    params_b: float = Field(..., alias="paramsB")
    default_cutoff_len: int = Field(..., alias="defaultCutoffLen")
    suggested_instance: str = Field(..., alias="suggestedInstance")
    gated: bool = False
    smoke_tested: bool = Field(False, alias="smokeTested")
    image_tag: str = Field("stable", alias="imageTag")  # pin to an image tier
    # Optional SageMaker Public Hub id, set when onboarding a serverless-
    # customizable model so the serverless engine is available immediately
    # (validated at the trust boundary in save_custom_model).
    serverless_model_id: str = Field("", alias="serverlessModelId")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


@app.post("/api/models/custom")
def add_custom_model(req: SaveModelRequest) -> dict:
    """Persist an onboarded model into the catalog. Rejects an unknown template
    (it would fail the engine at parse time) — deterministic guard, no job."""
    from .onboard import is_known_template, save_custom_model

    if not is_known_template(req.template):
        raise HTTPException(
            status_code=400,
            detail=f"template '{req.template}' is not a known LLaMA-Factory template",
        )
    try:
        save_custom_model(req.model_dump(by_alias=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "id": req.id}


@app.get("/api/models/custom")
def list_custom() -> dict:
    """List only the auto-onboarded (custom) models."""
    from .onboard import list_custom_models

    return {"models": list_custom_models()}


@app.delete("/api/models/custom/{model_id}")
def remove_custom_model(model_id: str) -> dict:
    from .onboard import delete_custom_model

    if not delete_custom_model(model_id):
        raise HTTPException(status_code=404, detail=f"no custom model {model_id}")
    return {"ok": True, "id": model_id}


@app.post("/api/models/{model_id}/smoke-test")
def smoke_test_model(model_id: str, image_tag: str | None = None,
                     method: Literal["lora", "qlora", "full", "freeze"] = "lora",
                     engine: Literal["llama_factory", "sagemaker_serverless"] = "llama_factory",
                     lora_variant: Literal["lora", "dora", "rslora", "pissa", "loraplus"] = "lora") -> dict:
    """Launch a real but tiny (capped, 1-epoch) training job to confirm the
    engine accepts this model + template ON A SPECIFIC IMAGE with a specific
    METHOD (+ LoRA VARIANT). Authoritative verification — costs a few cents.
    Returns the job name to poll via /api/verify/{job} (which records the
    verified/incompatible result).

    `image_tag` (query) verifies the model on that image tier instead of its
    catalog default — this is how a model is proven on `latest` (0.9.5) before
    being pinned there. `method` (query) verifies a parameterization other than
    plain LoRA (e.g. qlora) — QLoRA's 4-bit load can fail where LoRA works, so it
    needs its own proof. `lora_variant` (query) verifies a non-plain adapter
    variant (DoRA/rsLoRA/PiSSA/LoRA+) — DoRA/PiSSA change training (and the merge)
    enough that a plain-LoRA proof doesn't cover them. `engine` (query) verifies a
    non-default training engine (e.g. sagemaker_serverless) — a serverless run
    proves the serverless surface, distinct from the LLaMA-Factory image tiers.
    Requires a small dataset to exist (any persisted split); the smoke test caps
    rows so it's cheap."""
    from datetime import datetime, timezone

    model = get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model {model_id}")
    # Serverless verification requires the engine enabled + a Public-Hub mapping
    # (same guards the race launch enforces), surfaced as a clear 400 here.
    if engine == "sagemaker_serverless":
        from .engines.base import engine_enabled

        if not engine_enabled("sagemaker_serverless"):
            raise HTTPException(status_code=400,
                                detail="the SageMaker Serverless engine is not enabled")
        if not getattr(model, "serverless_model_id", ""):
            raise HTTPException(status_code=400,
                                detail=f"{model.display_name} has no SageMaker serverless equivalent")
        # The serverless engine shells out to a separate V3-SDK interpreter
        # (SLM_SERVERLESS_PYTHON). The deployed image sets it; a local dev backend
        # usually doesn't. Preflight it here so the user gets a friendly "not
        # available in this environment" 400 instead of the raw SLM_SERVERLESS_PYTHON
        # internals leaking through the generic 502 after they click verify.
        if not os.environ.get("SLM_SERVERLESS_PYTHON", "").strip():
            raise HTTPException(
                status_code=400,
                detail=("Serverless verification isn't available in this environment. "
                        "The serverless engine needs its separate V3 SageMaker SDK runtime, "
                        "which the deployed app has but this local backend doesn't. "
                        "Use the deployed app to verify on serverless, or set "
                        "SLM_SERVERLESS_PYTHON locally."),
            )
    datasets_list = list_datasets()
    # The smoke test always trains SFT (stage defaults to sft), so it needs a
    # MESSAGES-shaped dataset — a preference (ranking) split would fail the engine.
    split = next(
        (d for d in datasets_list
         if (d.get("trainRows") or 0) > 0 and d.get("shape", "sft") != "preference"),
        None,
    )
    if split is None:
        raise HTTPException(
            status_code=400,
            detail="smoke test needs a messages-shaped dataset with training rows — create one first",
        )
    # Route to the requested engine. Serverless ignores image_tag (managed recipe),
    # and only supports lora — force it so a qlora pick can't 400 the smoke test.
    hp_method = "lora" if engine == "sagemaker_serverless" else method
    # Variants are an LLaMA-Factory-only axis (only render.py emits use_dora/
    # pissa_init/… — the serverless managed recipe never reads lora_variant). They
    # also ride the adapter methods only (lora/qlora); full/freeze have no adapter.
    # Normalize a stray variant to plain "lora" for any non-LF engine OR full/freeze
    # so a serverless+dora request can't mint a false `serverless::lora::dora` proof
    # for a run that actually trained plain LoRA (mirrors the hp_method=lora forcing
    # above + the verification-key normalization).
    hp_variant = (
        lora_variant
        if engine == "llama_factory" and hp_method in ("lora", "qlora")
        else "lora"
    )
    # Size gate: full/freeze are offered only for models whose allowed_methods
    # includes them (≤2B). Reject before a billable launch that would OOM.
    if hp_method in ("full", "freeze") and hp_method not in (model.allowed_methods or ()):
        raise HTTPException(
            status_code=400,
            detail=(f"{model.display_name} ({model.params_b}B) does not support "
                    f"'{hp_method}' fine-tuning (allowed: {model.allowed_methods})"),
        )
    # Hyperparams.__post_init__ hard-rejects the invalid QLoRA × {DoRA, PiSSA}
    # pairing (a 4-bit base lacks the full-precision weights they need). Surface it
    # as a clean 400 BEFORE a billable launch rather than letting the ValueError
    # become a generic 500.
    try:
        hp = Hyperparams(finetuning_type=hp_method, engine=engine, lora_variant=hp_variant,
                         num_train_epochs=1.0, max_samples=16, save_steps=100000)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Instance must be method-aware (mirrors race.py): full/freeze need the bigger
    # g6e (model.suggested_instance is LoRA-sized — a g5 that would OOM a full-weight
    # run and falsely mark the model full-incompatible).
    from .catalog import _instance_for

    instance_type = (
        _instance_for(model.params_b, hp_method)
        if hp_method in ("full", "freeze")
        else model.suggested_instance
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        res = launch_training_job(
            model_id=model_id,
            split_id=split["splitId"],
            hp=hp,
            instance_type=instance_type,
            stamp=f"smoke-{stamp}",
            max_run_seconds=1800,
            image_tag=image_tag,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"smoke test launch failed: {e}")
    # Persist PENDING so the catalog shows "verifying…" even after the user
    # navigates away; the reconcile loop resolves it to verified/incompatible.
    # For serverless, the surface is the "serverless" token (set_pending resolves
    # it from engine), not the image tier — res["imageTag"] is ignored there.
    from .verifications import set_pending

    # The serverless engine is a MANAGED recipe with no ECR image, so its launch
    # result has no imageTag/imageUri — and set_pending derives the surface from
    # `engine` ("serverless"), ignoring image_tag entirely. Read them defensively
    # so a serverless smoke test can't KeyError after the job already launched
    # (which left the launch un-recorded and 500'd the UI mid-spin).
    res_image_tag = res.get("imageTag", "")
    set_pending(model_id, res_image_tag, res["jobName"], method=hp_method,
                engine=engine, lora_variant=hp_variant,
                ts=datetime.now(timezone.utc).isoformat())
    return {
        "jobName": res["jobName"],
        "modelId": model_id,
        "splitId": split["splitId"],
        "imageTag": res_image_tag,
        "method": hp_method,
        "engine": engine,
        "loraVariant": hp_variant,
        "imageUri": res.get("imageUri", ""),
    }


@app.get("/api/verify/{model_id}/{image_tag}/{job_name}")
def verify_smoke_result(model_id: str, image_tag: str, job_name: str,
                        method: Literal["lora", "qlora", "full", "freeze"] = "lora",
                        engine: Literal["llama_factory", "sagemaker_serverless"] = "llama_factory",
                        lora_variant: Literal["lora", "dora", "rslora", "pissa", "loraplus"] = "lora") -> dict:
    """Poll a smoke-test/verification job and, when it reaches a terminal state,
    record the (model, engine, image_tag, method, variant) verification result.

    Completed → verified. Failed/Stopped → incompatible, tagged with the failure
    reason (which the agentic self-healing classifier later reads to decide
    whether the model needs a newer image). Returns the live status + the stored
    record. Verification is informational (not a hard block), so a Failed result
    won't erase a prior proven success unless this is an explicit re-test. The
    `lora_variant` (query) must match the smoke-test's variant so the result lands
    on the SAME per-variant key (else a DoRA result resolves onto plain LoRA)."""
    from .verifications import VERIFIED, classify_failure, get_status, set_status

    try:
        d = describe_job(job_name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not describe job: {e}")
    status = d["status"]
    if status == "Completed":
        rec = set_status(model_id, image_tag, VERIFIED, method=method, engine=engine,
                         lora_variant=lora_variant, job_name=job_name,
                         ts=d.get("trainingEndTime"))
    elif status in ("Failed", "Stopped"):
        # Tell gated-access denial (fixable on HF) apart from real incompatibility.
        vstatus, reason = classify_failure(job_name, d.get("failureReason") or f"training {status.lower()}")
        rec = set_status(model_id, image_tag, vstatus, method=method, engine=engine,
                         lora_variant=lora_variant, job_name=job_name, reason=reason,
                         ts=d.get("trainingEndTime"))
    else:
        rec = get_status(model_id, image_tag, method, engine, lora_variant)  # still running — no change
    return {"jobName": job_name, "jobStatus": status, "modelId": model_id,
            "imageTag": image_tag, "method": method, "engine": engine,
            "loraVariant": lora_variant, "verification": rec}


class RenderRequest(BaseModel):
    model_id: str = Field(..., alias="modelId")
    split_id: str = Field(..., alias="splitId")
    # Hyperparameter overrides; any omitted field uses the Hyperparams default.
    # Objective: "sft" (default, messages data) or "dpo" (preference/ranking data).
    stage: Literal["sft", "dpo", "kto"] = Field("sft", alias="stage")
    pref_beta: float = Field(0.1, alias="prefBeta")
    # DPO preference-loss family: sigmoid (standard DPO), orpo, or simpo (both
    # reference-free). Ignored unless stage=dpo. simpo_gamma is SimPO's margin.
    pref_loss: Literal["sigmoid", "orpo", "simpo"] = Field("sigmoid", alias="prefLoss")
    simpo_gamma: float = Field(0.5, alias="simpoGamma")
    # KTO-only per-class loss weights (1.0/1.0 = neutral). Ignored unless stage=kto.
    kto_chosen_weight: float = Field(1.0, alias="ktoChosenWeight")
    kto_rejected_weight: float = Field(1.0, alias="ktoRejectedWeight")
    # Efficiency knobs (llama_factory only; no-op defaults keep configs byte-identical).
    neftune_noise_alpha: float = Field(0.0, alias="neftuneNoiseAlpha")
    enable_liger_kernel: bool = Field(False, alias="enableLigerKernel")
    packing: bool = Field(False, alias="packing")
    # Parameterization: "lora"/"qlora" (adapter) or "full"/"freeze" (full-weight,
    # llama_factory + SFT only; gated to small models — see catalog.allowed_methods).
    finetuning_type: Literal["lora", "qlora", "full", "freeze"] = Field("lora", alias="finetuningType")
    quantization_bit: int | None = Field(None, alias="quantizationBit")
    lora_rank: int = Field(8, alias="loraRank")
    lora_alpha: int | None = Field(None, alias="loraAlpha")
    # LoRA VARIANT — a modifier on finetuning_type=lora/qlora (NOT a new method):
    # lora|dora|rslora|pissa|loraplus. Ignored for full/freeze. loraplus carries
    # the B/A learning-rate ratio. See Hyperparams.lora_variant + render.py.
    lora_variant: Literal["lora", "dora", "rslora", "pissa", "loraplus"] = Field("lora", alias="loraVariant")
    loraplus_lr_ratio: float = Field(16.0, alias="loraplusLrRatio")
    # freeze-only: number of top transformer layers to train. Ignored unless freeze.
    freeze_trainable_layers: int = Field(2, alias="freezeTrainableLayers")
    learning_rate: float = Field(1.0e-4, alias="learningRate")
    num_train_epochs: float = Field(3.0, alias="numTrainEpochs")
    per_device_train_batch_size: int = Field(1, alias="perDeviceTrainBatchSize")
    gradient_accumulation_steps: int = Field(8, alias="gradientAccumulationSteps")
    cutoff_len: int | None = Field(None, alias="cutoffLen")
    save_steps: int = Field(500, alias="saveSteps")
    max_samples: int | None = Field(None, alias="maxSamples")
    early_stopping_enabled: bool = Field(False, alias="earlyStoppingEnabled")
    early_stopping_patience: int = Field(3, alias="earlyStoppingPatience")

    # populate_by_name lets the API accept camelCase aliases; protected_namespaces
    # disabled so the `model_id` field doesn't collide with pydantic's `model_` guard.
    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class SuggestRequest(BaseModel):
    model_id: str = Field(..., alias="modelId")
    split_id: str | None = Field(None, alias="splitId")
    # Optional explicit override of dataset size (e.g. before a split exists);
    # otherwise we read trainRows + hasVal from the split's meta.
    train_rows: int | None = Field(None, alias="trainRows")
    has_val: bool | None = Field(None, alias="hasVal")
    # The method the user has picked — so the recommendation is method-aware
    # (full/freeze need a ~1e-5 LR + no LoRA rank, not the LoRA-scale default).
    finetuning_type: Literal["lora", "qlora", "full", "freeze"] = Field("lora", alias="finetuningType")
    # The objective (LLaMA-Factory stage) — so the LoRA LR is objective-aware
    # (SFT ~2e-4 vs DPO/KTO ~5e-6; the SFT LR diverges on a preference loss).
    objective: Literal["sft", "dpo", "kto", "rlvr", "rlaif"] = Field("sft", alias="objective")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


@app.post("/api/recommend")
def recommend_config(req: SuggestRequest) -> dict:
    """Deterministic starting-point hyperparameters for a model + dataset, with a
    per-field rationale. A sensible default the user can edit — the race remains
    the source of truth. Reads dataset size/has-val from the split's meta when a
    split_id is given (overridable for previewing before a split exists)."""
    from .recommend import suggest_config

    model = get_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model id: {req.model_id}")

    train_rows = req.train_rows
    has_val = req.has_val
    if (train_rows is None or has_val is None) and req.split_id:
        meta = split_meta(req.split_id)
        if train_rows is None:
            train_rows = meta.get("trainRows")
        if has_val is None:
            has_val = bool(meta.get("hasVal"))
    if train_rows is None:
        train_rows = 1000  # neutral assumption when size is unknown
    if has_val is None:
        has_val = False

    # Tier 1: seed card-constrained values (cutoff ceiling, freeze-layer cap) from
    # the model card's architecture. Best-effort + cached — a failed fetch yields
    # {} and the recommender falls back to its size×dataset heuristics.
    from .model_card import fetch_arch

    arch = fetch_arch(model.hf_model_id)
    rec = suggest_config(model, int(train_rows), bool(has_val),
                         finetuning_type=req.finetuning_type, arch=arch,
                         objective=req.objective)
    return {"modelId": req.model_id, "splitId": req.split_id, **rec.to_dict()}


@app.post("/api/advise")
def advise_config(req: SuggestRequest) -> dict:
    """LLM advisor (Tier 3): propose a small SWEEP of configs to RACE, grounded
    in the deterministic baseline + model facts, bounds-validated, with a
    deterministic fallback if the LLM is unavailable. The race picks the winner."""
    from .advisor import advise_sweep

    model = get_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model id: {req.model_id}")
    train_rows, has_val = req.train_rows, req.has_val
    if (train_rows is None or has_val is None) and req.split_id:
        meta = split_meta(req.split_id)
        train_rows = train_rows if train_rows is not None else meta.get("trainRows")
        has_val = has_val if has_val is not None else bool(meta.get("hasVal"))
    rec = advise_sweep(model, int(train_rows or 1000), bool(has_val),
                       finetuning_type=req.finetuning_type)
    return {"modelId": req.model_id, **rec}


class AutofillRequest(BaseModel):
    split_id: str = Field(..., alias="splitId")
    # The job CEILING ("up to N") the user picked — the planner fills only as many arms
    # as add signal (≤ this), never padding. Mapped to the effort ceilings 4/8/16.
    ceiling: int = Field(8, ge=1, le=64, alias="ceiling")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


def _hp_to_card(hp) -> dict:
    """Map a planned RaceModel's Hyperparams to the camelCase fields the manual
    Step-2 card stages, so an auto-filled arm carries the SAME research-backed config
    (objective-aware LR, DoRA/full method, rank, etc.) the guided planner would use."""
    return {
        "engine": hp.engine,
        "stage": hp.stage,
        "finetuningType": hp.finetuning_type,
        "loraRank": str(hp.lora_rank),
        "loraAlpha": "" if hp.lora_alpha is None else str(hp.lora_alpha),
        "loraVariant": hp.lora_variant,
        "loraplusLrRatio": str(hp.loraplus_lr_ratio),
        "freezeTrainableLayers": str(hp.freeze_trainable_layers),
        "learningRate": repr(hp.learning_rate),
        "numTrainEpochs": str(hp.num_train_epochs),
        "perDeviceTrainBatchSize": str(hp.per_device_train_batch_size),
        "gradientAccumulationSteps": str(hp.gradient_accumulation_steps),
        "cutoffLen": "" if hp.cutoff_len is None else str(hp.cutoff_len),
        "saveSteps": str(hp.save_steps),
        "maxSamples": "" if hp.max_samples is None else str(hp.max_samples),
        "earlyStoppingEnabled": hp.early_stopping_enabled,
        "earlyStoppingPatience": str(hp.early_stopping_patience),
        "prefLoss": hp.pref_loss,
    }


@app.post("/api/autofill-race")
def autofill_race(req: AutofillRequest) -> dict:
    """Auto-assemble the best RACE PORTFOLIO for a dataset — the SAME research-backed
    planner the Guided flow uses (race_planner.plan_race), exposed for the manual
    Fine-tune Step-2 "auto-fill" button. Returns one arm per planned model with the
    full staged-card hyperparameters, plus the ceiling/meaningfulCount/capped
    transparency so the UI can say "filled M of up-to-N — more wouldn't add signal"
    instead of padding. The promise is a portfolio to RACE (the leaderboard picks the
    winner), NOT a single 'best' config."""
    from .profiler import profile_dataset
    from .race_planner import plan_race
    from .secrets import get_hf_token

    if split_dir(req.split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown split: {req.split_id}")

    prof = profile_dataset(req.split_id)
    # max_jobs honors the EXACT ceiling the user picked (4/6/8/…/16), not a snapped tier.
    plan = plan_race(prof, "balanced", hf_token_ok=bool(get_hf_token()), max_jobs=req.ceiling)
    if not plan.supported:
        return {"supported": False, "reason": plan.reason, "models": [],
                "ceiling": req.ceiling, "meaningfulCount": 0, "capped": False,
                "objective": plan.objective}

    models = []
    for p in plan.planned:
        m = get_model(p.race_model.model_id)
        models.append({
            "modelId": p.race_model.model_id,
            "displayName": p.display_name,
            "label": p.display_label(),
            "family": m.family if m else "",
            "gated": bool(m.gated) if m else False,
            "paramsB": p.params_b,
            "role": p.role,
            "hp": _hp_to_card(p.race_model.hp),
        })
    return {
        "supported": True,
        "objective": plan.objective,
        "rankMetric": plan.rank_metric,
        "detectedTask": plan.detected_task,
        "models": models,
        "meaningfulCount": len(plan.planned),
        "ceiling": plan.job_budget,
        "capped": plan.capped,
        "gatesApplied": plan.gates_applied,
    }


@app.post("/api/render")
def render_config(req: RenderRequest) -> dict:
    """Render LLaMA-Factory train + export YAML for a model + split + hyperparams."""
    model = get_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model id: {req.model_id}")
    if split_dir(req.split_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown split id: {req.split_id} — run a split first",
        )

    # Surface an invalid hyperparameter combo (e.g. packing on a non-SFT stage) as a
    # clean 400 with the guard message, not an opaque 500 — mirrors race_launch +
    # smoke_test. _hyperparams_from → Hyperparams.__post_init__ raises ValueError.
    try:
        hp = _hyperparams_from(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    yamls = render_all(model, hp, req.split_id)
    return {"model": model.to_dict(), "splitId": req.split_id, **yamls}


def _hyperparams_from(req: "RenderRequest | TrainRequest") -> Hyperparams:
    return Hyperparams(
        stage=req.stage,
        pref_beta=req.pref_beta,
        pref_loss=req.pref_loss,
        simpo_gamma=req.simpo_gamma,
        kto_chosen_weight=req.kto_chosen_weight,
        kto_rejected_weight=req.kto_rejected_weight,
        neftune_noise_alpha=req.neftune_noise_alpha,
        enable_liger_kernel=req.enable_liger_kernel,
        packing=req.packing,
        finetuning_type=req.finetuning_type,
        quantization_bit=req.quantization_bit,
        lora_rank=req.lora_rank,
        lora_alpha=req.lora_alpha,
        lora_variant=req.lora_variant,
        loraplus_lr_ratio=req.loraplus_lr_ratio,
        freeze_trainable_layers=req.freeze_trainable_layers,
        learning_rate=req.learning_rate,
        num_train_epochs=req.num_train_epochs,
        per_device_train_batch_size=req.per_device_train_batch_size,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
        cutoff_len=req.cutoff_len,
        save_steps=req.save_steps,
        max_samples=req.max_samples,
        early_stopping_enabled=req.early_stopping_enabled,
        early_stopping_patience=req.early_stopping_patience,
    )


# --- SageMaker training launch + status (COSTS MONEY) ---


class TrainRequest(RenderRequest):
    """Render fields + launch knobs. Inherits all hyperparameter aliases."""

    instance_type: str = Field("ml.g5.2xlarge", alias="instanceType")
    max_run_seconds: int = Field(3600, alias="maxRunSeconds")


@app.post("/api/train")
def train_launch(req: TrainRequest) -> dict:
    """Launch a SageMaker training job (wait=False). This creates a billable job.

    Only invoked from the explicit UI launch action. Validates model + split,
    then renders/uploads/launches via the deterministic orchestrator.
    """
    model = get_model(req.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model id: {req.model_id}")
    if split_dir(req.split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown split id: {req.split_id}")

    # Invalid hyperparameter combo (e.g. packing on a non-SFT stage) → clean 400 with
    # the guard message BEFORE the billable launch, not an opaque 500. Mirrors
    # race_launch (m.to_hp()) + smoke_test. Separate from the launch_training_job
    # try/except below (which surfaces AWS/SDK errors as 502).
    try:
        hp = _hyperparams_from(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Timestamp stamp lives in the endpoint (orchestrator stays time/RNG-free).
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        result = launch_training_job(
            model_id=req.model_id,
            split_id=req.split_id,
            hp=hp,
            instance_type=req.instance_type,
            stamp=stamp,
            max_run_seconds=req.max_run_seconds,
        )
    except Exception as e:  # surface AWS/SDK errors to the UI rather than 500-ing opaquely
        raise HTTPException(status_code=502, detail=f"launch failed: {e}")
    return result


@app.get("/api/train/{job_name}")
def train_status(job_name: str) -> dict:
    """Poll a training job's status (read-only)."""
    try:
        return describe_job(job_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"could not describe job {job_name}: {e}")


@app.get("/api/train/{job_name}/curves")
def train_curves(job_name: str) -> dict:
    """Loss/lr/epoch time series for a training job (from CloudWatch).

    Powers the live training curve on the Races detail page. Series are empty
    (not an error) for jobs too new to have logged yet or launched before metric
    scraping was added — the UI shows a 'no data yet' state in that case.
    """
    try:
        return fetch_training_curves(job_name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not fetch curves for {job_name}: {e}")


@app.get("/api/train/{job_name}/reward-curve")
def reward_curve(job_name: str) -> dict:
    """RLVR reward trajectory (GRPO reward over training steps) for a serverless
    RLVR job, parsed from its metrics.jsonl. Empty (hasData=False) — not an error
    — for non-RLVR jobs or jobs too new to have logged; the UI shows 'no data yet'.
    """
    from .reward_curve import fetch_reward_curve

    try:
        return fetch_reward_curve(job_name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not fetch reward curve for {job_name}: {e}")


# --- RLVR custom reward functions ------------------------------------------
# A reward function = user-authored `reward(response, ground_truth) -> float`
# (or a generated 'reward = leaderboard metric' snippet), packaged into a Lambda
# and registered as a SageMaker Evaluator, then picked at RLVR launch. See
# reward_functions.py. Deploy (Lambda + Evaluator create) is slow → dispatched to
# the worker; the record carries a status the UI polls.


@app.get("/api/reward-functions")
def list_reward_fns() -> dict:
    """The current tenant's reward functions + the metrics a 'metric' reward can
    mirror + the judge models an RLAIF reward prompt can use (so the UI can offer a
    metric picker, a code editor, and a judge-model dropdown)."""
    from .reward_functions import ALLOWED_JUDGE_MODELS, list_reward_functions
    from .reward_templates import scoring

    return {
        "rewardFunctions": list_reward_functions(),
        "metrics": list(scoring.METRIC_NAMES),
        "judgeModels": list(ALLOWED_JUDGE_MODELS),
    }


class RewardFunctionRequest(BaseModel):
    name: str
    # Exactly one source: `snippet` (user code, verifiable RLVR reward), `metric`
    # (generate a verifiable snippet from a leaderboard metric), or `prompt` (an
    # RLAIF AI-judge reward prompt — kind 'reward_prompt', no Lambda).
    snippet: str | None = None
    metric: str | None = None
    prompt: str | None = None
    # RLAIF only: the judge model the recipe scores with ("" = recipe default).
    reward_model_id: str = Field("", alias="rewardModelId")

    model_config = {"populate_by_name": True}


@app.post("/api/reward-functions")
def create_reward_fn(req: RewardFunctionRequest) -> dict:
    """Validate + persist a reward function, then dispatch its deploy to the worker.
    Returns the record with status=deploying. A verifiable reward (snippet|metric)
    builds a Lambda + Evaluator; an RLAIF reward `prompt` needs no AWS resources
    (it's passed inline to RLAIFTrainer) so it's usable immediately. Validation
    errors (bad snippet/prompt, wrong number of sources) are 400."""
    from datetime import datetime, timezone

    from .reward_functions import (
        RewardError, make_reward_function, make_reward_prompt, save_reward_function,
    )

    if not req.name.strip():
        raise HTTPException(status_code=400, detail="reward function name is required")
    sources = [s for s in (req.snippet, req.metric, req.prompt) if s]
    if len(sources) != 1:
        raise HTTPException(
            status_code=400,
            detail="provide exactly one of `snippet`, `metric`, or `prompt`",
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        if req.prompt:
            rf = make_reward_prompt(req.name, req.prompt,
                                    reward_model_id=req.reward_model_id, stamp=stamp)
        else:
            rf = make_reward_function(req.name, snippet=req.snippet, metric=req.metric, stamp=stamp)
    except RewardError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rf.status = "deploying"
    save_reward_function(rf)
    payload = {"task": "deploy_reward", "rewardId": rf.id}
    if not dispatch_worker(payload):
        # Local dev (no worker Lambda): deploy inline so it's usable immediately.
        from .reward_functions import run_deploy_reward_task

        try:
            run_deploy_reward_task(rf.id)
        except Exception as e:  # noqa: BLE001 — surfaced via the record's status
            raise HTTPException(status_code=502, detail=f"reward deploy failed: {e}")
    from .reward_functions import get_reward_function

    return get_reward_function(rf.id) or rf.to_dict()


@app.get("/api/reward-functions/{reward_id}")
def get_reward_fn(reward_id: str) -> dict:
    from .reward_functions import get_reward_function

    rec = get_reward_function(reward_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"reward function not found: {reward_id}")
    return rec


@app.delete("/api/reward-functions/{reward_id}")
def delete_reward_fn(reward_id: str) -> dict:
    """Remove a reward function from the registry. (Best-effort: leaves the AWS
    Lambda/Evaluator in place — they're cheap, idempotently reused by hash, and a
    running RLVR job may still reference the Evaluator.)"""
    from .reward_functions import delete_reward_function

    if not delete_reward_function(reward_id):
        raise HTTPException(status_code=404, detail=f"reward function not found: {reward_id}")
    return {"rewardId": reward_id, "deleted": True}


class RewardValidateRequest(BaseModel):
    # Validate EITHER a verifiable reward snippet OR an RLAIF reward prompt.
    snippet: str | None = None
    prompt: str | None = None


@app.post("/api/reward-functions/validate")
def validate_reward_fn(req: RewardValidateRequest) -> dict:
    """Parse-validate a reward snippet OR a reward prompt WITHOUT saving — powers
    inline editor feedback. Returns {ok: true} or {ok: false, error: ...}."""
    from .reward_functions import RewardError, validate_reward_prompt, validate_snippet

    try:
        if req.prompt is not None:
            validate_reward_prompt(req.prompt)
        else:
            validate_snippet(req.snippet or "")
        return {"ok": True}
    except RewardError as e:
        return {"ok": False, "error": str(e)}


class RewardTryRequest(BaseModel):
    """A sample rollout to score. Exactly one source: `rewardId` (a SAVED reward),
    `snippet` (unsaved editor code), or `metric` (generate the metric snippet)."""
    response: str
    ground_truth: str = Field("", alias="groundTruth")
    reward_id: str | None = Field(None, alias="rewardId")
    snippet: str | None = None
    metric: str | None = None

    model_config = {"populate_by_name": True}


class RewardPromptTryRequest(BaseModel):
    """Dry-run an RLAIF judge RUBRIC on a few (prompt, response) candidates so the
    user can SEE the score spread before deploying. `samples` are good/bad candidate
    responses to score; `rewardModelId` is the judge (blank = preview with the
    platform default). Server caps the candidate count + per-string length so a
    user can't spam billable Converse calls."""
    prompt: str  # the rubric text (must contain {{prompt}} + {{response}})
    samples: list[dict] = Field(default_factory=list)  # [{prompt, response, intendedLabel?}]
    reward_model_id: str = Field("", alias="rewardModelId")

    model_config = {"populate_by_name": True}


class RewardDomainCheckRequest(BaseModel):
    """Pre-launch advisory: does a reward source match a split's ground_truth?"""
    split_id: str = Field(..., alias="splitId")
    preset_reward_function: str = Field("", alias="presetRewardFunction")
    reward_function_id: str = Field("", alias="rewardFunctionId")

    model_config = {"populate_by_name": True}


@app.post("/api/reward-functions/domain-check")
def reward_domain_check(req: RewardDomainCheckRequest) -> dict:
    """Advisory: would the chosen RLVR reward grade this split's ground_truth, or
    score ~0 every step (a wasted billable run)? Returns {warning: str|null} +
    the detected groundTruthTask. Surfaced inline on the FineTune RLVR step so the
    user sees the mismatch BEFORE launching — the same check race_launch does, but
    pre-submit. Never raises on a bad profile (advisory)."""
    from .reward_functions import get_reward_function, reward_domain_warning

    gt_task = None
    try:
        from .profiler import profile_dataset

        prof = profile_dataset(req.split_id)
        gt_task = (prof.get("rlvr") or {}).get("groundTruthTask")
    except Exception:  # noqa: BLE001 — advisory; never fail on the profiler
        gt_task = None

    metric = ""
    if req.reward_function_id:
        rf = get_reward_function(req.reward_function_id) or {}
        metric = rf.get("metric") or ""  # only metric-kind rewards are checkable
    warning = reward_domain_warning(
        preset=req.preset_reward_function, metric=metric, ground_truth_task=gt_task)
    return {"warning": warning, "groundTruthTask": gt_task}


@app.post("/api/reward-functions/try")
def try_reward_fn(req: RewardTryRequest) -> dict:
    """Dry-run a reward against ONE sample (response, ground_truth) — IN-PROCESS,
    no AWS, no billable run — so the user sees the score the deployed Lambda would
    return BEFORE launching GRPO. Source = `rewardId` (saved), `snippet` (editor
    code), or `metric` (the generated metric snippet). Returns {score} or 400."""
    from .reward_functions import (
        RewardError,
        get_reward_function,
        metric_reward_snippet,
        try_reward,
    )

    snippet = req.snippet
    try:
        if req.reward_id:
            rec = get_reward_function(req.reward_id)
            if rec is None:
                raise HTTPException(status_code=404, detail=f"reward function not found: {req.reward_id}")
            snippet = rec["snippet"]
        elif req.metric:
            snippet = metric_reward_snippet(req.metric)
        if not snippet:
            raise HTTPException(status_code=400, detail="provide a rewardId, snippet, or metric to try")
        score = try_reward(snippet, req.response, req.ground_truth)
    except RewardError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"score": score}


# Caps so a user can't spam billable Converse calls through the dry-run.
_MAX_DRYRUN_SAMPLES = 12
_MAX_DRYRUN_STR_LEN = 4000


@app.post("/api/reward-functions/try-prompt")
def try_reward_prompt_fn(req: RewardPromptTryRequest) -> dict:
    """Dry-run an RLAIF judge RUBRIC on candidate responses — scores each via the
    model-agnostic Converse path, returns per-sample scores + a good/bad spread, so
    the user can confirm the rubric DISCRIMINATES before a billable GRPO run. The
    rubric's chosen judge is used; blank previews with the platform default judge
    (indicative — the deployed reward uses the recipe judge)."""
    from .reward_functions import RewardError, try_reward_prompt

    samples = (req.samples or [])[:_MAX_DRYRUN_SAMPLES]
    if not samples:
        raise HTTPException(status_code=400, detail="provide at least one sample to score")
    scored: list[dict] = []
    try:
        for s in samples:
            resp = str(s.get("response", ""))[:_MAX_DRYRUN_STR_LEN]
            pr = str(s.get("prompt", ""))[:_MAX_DRYRUN_STR_LEN]
            r = try_reward_prompt(req.prompt, pr, resp, req.reward_model_id)
            scored.append({
                "prompt": pr, "response": resp,
                "intendedLabel": s.get("intendedLabel"),
                "score": r["score"], "reasoning": r["reasoning"], "error": r["error"],
            })
    except RewardError as e:  # placeholder/judge-id validation — actionable 400
        raise HTTPException(status_code=400, detail=str(e))

    good = [s["score"] for s in scored if s.get("intendedLabel") == "good"]
    bad = [s["score"] for s in scored if s.get("intendedLabel") == "bad"]
    spread = None
    if good and bad:
        gm, bm = sum(good) / len(good), sum(bad) / len(bad)
        spread = {"goodMean": round(gm, 4), "badMean": round(bm, 4),
                  "separation": round(gm - bm, 4), "discriminates": (gm - bm) >= 0.3}
    return {"samples": scored, "scoreSpread": spread, "indicative": True}


class RewardAuthorRequest(BaseModel):
    """Ask the reward-author agent to draft + calibrate an RLAIF judge rubric for a
    plain-English goal, grounded in a dataset's prompt-only profile. `priorResult`
    (optional) feeds a previous draft + feedback back in for regenerate-with-feedback."""
    split_id: str = Field(..., alias="splitId")
    goal: str
    prior_result: dict | None = Field(None, alias="priorResult")

    model_config = {"populate_by_name": True}


@app.post("/api/reward-functions/author")
def author_reward_prompt_ep(req: RewardAuthorRequest) -> dict:
    """Draft + calibrate an RLAIF judge rubric with the reward-author agent: it
    writes a {{prompt}}/{{response}} rubric for the goal, scores fabricated good/bad
    candidates with a real judge, and iterates until they separate — proving the
    rubric discriminates BEFORE a billable GRPO run. The agent makes several judge
    calls + an AgentCore round-trip, which can exceed API GW's 29s, so this
    dispatches to the worker and the client polls; local dev runs inline. ADVISORY —
    the returned draft pre-fills the authoring form; deploy stays an explicit click."""
    from .investigator import start_reward_author

    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="a training goal is required")
    try:
        return start_reward_author(req.split_id, req.goal.strip(), prior_result=req.prior_result)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"reward author failed: {e}")


@app.get("/api/reward-functions/author/{split_id}")
def author_reward_prompt_status(split_id: str) -> dict:
    """Poll a reward-author run: {status, result?}. `result` is the draft rubric +
    score spread once done; null while running."""
    from .investigator import load_reward_author, reward_author_status

    return {**reward_author_status(split_id), "result": load_reward_author(split_id)}


# --- Offline batch eval ---


@app.get("/api/jobs/completed")
def completed_jobs() -> dict:
    """List completed training jobs (eval candidates) with their model artifacts."""
    try:
        return {"jobs": list_completed_jobs()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not list jobs: {e}")


class EvalRequest(BaseModel):
    source_job_name: str = Field(..., alias="sourceJobName")
    split_id: str = Field(..., alias="splitId")
    backend: str = Field("vllm")
    temperature: float = Field(0.0)
    top_p: float = Field(1.0, alias="topP")
    max_new_tokens: int = Field(256, alias="maxNewTokens")
    seed: int = Field(42)
    instance_type: str = Field("ml.g5.2xlarge", alias="instanceType")
    max_run_seconds: int = Field(3600, alias="maxRunSeconds")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


@app.post("/api/eval")
def eval_launch(req: EvalRequest) -> dict:
    """Launch an eval job for a completed model on the held-out split (billable)."""
    if split_dir(req.split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown split id: {req.split_id}")

    decoding = DecodingParams(
        backend=req.backend,
        temperature=req.temperature,
        top_p=req.top_p,
        max_new_tokens=req.max_new_tokens,
        seed=req.seed,
    )
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        return launch_eval_job(
            source_job_name=req.source_job_name,
            split_id=req.split_id,
            decoding=decoding,
            stamp=stamp,
            instance_type=req.instance_type,
            max_run_seconds=req.max_run_seconds,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"eval launch failed: {e}")


@app.get("/api/eval/{job_name}")
def eval_status(job_name: str) -> dict:
    """Poll an eval job; include metrics.json once the job is Completed."""
    try:
        status = describe_job(job_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"could not describe job {job_name}: {e}")
    metrics = None
    if status["status"] == "Completed":
        try:
            metrics = fetch_metrics(job_name)
        except Exception as e:  # metrics fetch failure shouldn't hide job status
            # Do NOT return str(e): the S3/tarfile error text carries bucket names and
            # key paths. Log the detail, hand the caller a stable marker.
            from .obs import log_event

            log_event("eval.metrics.fetch_failed", level="WARNING",
                      job=job_name, error=f"{type(e).__name__}: {e}")
            status["metricsError"] = "metrics could not be read for this job"
    status["metrics"] = metrics
    return status


# --- Leaderboard + Sonnet baseline ---


@app.get("/api/leaderboard/splits")
def leaderboard_splits() -> dict:
    """List split ids that have completed eval jobs (for the leaderboard selector)."""
    try:
        return {"splits": list_eval_splits()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not list eval splits: {e}")


@app.get("/api/leaderboard")
def leaderboard(split_id: str) -> dict:
    """Comparison table for ONE split: latest eval per model (fair, same eval set).

    Includes any cached frontier-model baselines for this split (one per Claude
    model that's been run — Haiku/Sonnet/Opus), persisted on disk so they survive
    split-switching and backend restarts. `baseline` (singular) is kept for
    back-compat = the Sonnet 4.5 baseline if present.
    """
    from .baseline import load_all_baselines

    try:
        return {
            "splitId": split_id,
            "rows": build_leaderboard(split_id),
            "baseline": load_baseline(split_id),  # back-compat (Sonnet 4.5)
            "baselines": load_all_baselines(split_id),  # all frontier baselines
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not build leaderboard: {e}")


class SonnetBaselineRequest(BaseModel):
    split_id: str = Field(..., alias="splitId")
    max_new_tokens: int = Field(256, alias="maxNewTokens")
    temperature: float = Field(0.0)
    baseline_key: str = Field("sonnet-4-5", alias="baselineKey")  # which Claude model

    model_config = {"populate_by_name": True}


@app.post("/api/baseline/sonnet")
def sonnet_baseline(req: SonnetBaselineRequest) -> dict:
    """Start the Sonnet 4.5 baseline over the held-out eval rows.

    The baseline makes one Bedrock call per eval row, which can exceed API
    Gateway's 29s limit on large eval sets. So when a worker Lambda is configured
    (hosted), we dispatch it ASYNCHRONOUSLY and return {status: running}; the UI
    polls GET /api/baseline/sonnet/{splitId}. With no worker (local dev) we run
    inline and return the metrics directly.
    """
    if split_dir(req.split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown split id: {req.split_id}")

    from .baseline import BASELINE_MODELS

    if req.baseline_key not in BASELINE_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown baseline model: {req.baseline_key}")

    payload = {
        "task": "baseline",
        "splitId": req.split_id,
        "maxNewTokens": req.max_new_tokens,
        "temperature": req.temperature,
        "baselineKey": req.baseline_key,
    }
    if dispatch_worker(payload):
        set_baseline_status(req.split_id, "running", key=req.baseline_key)
        return {"splitId": req.split_id, "status": "running", "baselineKey": req.baseline_key}

    # Local dev (no worker Lambda): run inline.
    try:
        metrics = run_sonnet_baseline(
            req.split_id, max_new_tokens=req.max_new_tokens,
            temperature=req.temperature, baseline_key=req.baseline_key,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"baseline failed: {e}")
    return {"splitId": req.split_id, "status": "done", "metrics": metrics,
            "baselineKey": req.baseline_key}


@app.get("/api/baseline/models")
def baseline_models() -> dict:
    """The selectable Claude baseline models (id, label, price) for the UI."""
    from .baseline import BASELINE_MODELS, DEFAULT_BASELINE

    return {
        "default": DEFAULT_BASELINE,
        "models": [
            {"key": k, "provider": v.get("provider", "Other"), "label": v["label"],
             "modelId": v["modelId"], "inPer1k": v["inPer1k"], "outPer1k": v["outPer1k"]}
            for k, v in BASELINE_MODELS.items()
        ],
    }


@app.get("/api/baseline/sonnet/{split_id}")
def sonnet_baseline_status(split_id: str, baseline_key: str = "sonnet-4-5") -> dict:
    """Poll a baseline run for a given model: {status, metrics?}."""
    st = baseline_status(split_id, key=baseline_key)
    return {"splitId": split_id, "baselineKey": baseline_key, **st,
            "metrics": load_baseline(split_id, key=baseline_key)}


# --- Fine-tuning race: train N models in parallel, auto-eval, pick winner ---


class RaceModelConfig(BaseModel):
    """One model + its OWN hyperparameters (different families need different
    settings). Chat template + LoRA targets come per-model from the catalog."""
    model_id: str = Field(..., alias="modelId")
    instance_type: str | None = Field(None, alias="instanceType")
    # Training engine: "llama_factory" (default, our frozen-image path) or
    # "sagemaker_serverless" (managed serverless SFT/DPO). Per-run so the same
    # model can race on both engines as distinct leaderboard rows.
    engine: Literal["llama_factory", "sagemaker_serverless"] = Field("llama_factory", alias="engine")
    # Objective: "sft" (default), "dpo" (preference data), "kto", "rlvr" (GRPO
    # against a verifiable reward), or "rlaif" (GRPO from AI feedback — an AI judge
    # scores a reward prompt). rlvr + rlaif are serverless-engine only.
    stage: Literal["sft", "dpo", "kto", "rlvr", "rlaif"] = Field("sft", alias="stage")
    pref_beta: float = Field(0.1, alias="prefBeta")
    # DPO preference-loss FAMILY: sigmoid (standard DPO, default), orpo (ORPO), or
    # simpo (SimPO). ORPO/SimPO are reference-free preference losses on the SAME
    # chosen/rejected dataset — only the algorithm differs. Ignored unless stage=dpo.
    pref_loss: Literal["sigmoid", "orpo", "simpo"] = Field("sigmoid", alias="prefLoss")
    # SimPO-only: target reward margin γ (LF simpo_gamma). Ignored unless pref_loss=simpo.
    simpo_gamma: float = Field(0.5, alias="simpoGamma")
    # KTO-only per-class loss weights (LF kto_chosen_weight/kto_rejected_weight =
    # the KTO paper's λ_D/λ_U). 1.0/1.0 = neutral default; raised on the minority
    # class to fix label imbalance. Ignored unless stage=kto.
    kto_chosen_weight: float = Field(1.0, alias="ktoChosenWeight")
    kto_rejected_weight: float = Field(1.0, alias="ktoRejectedWeight")
    # Efficiency knobs (LLaMA-Factory engine only; orthogonal to objective/method).
    # Defaults are no-ops so an unchanged run is byte-identical. neftune adds
    # embedding noise (quality), liger fuses kernels (speed/memory), packing
    # concatenates short SFT samples (throughput; SFT-only — backend rejects others).
    neftune_noise_alpha: float = Field(0.0, alias="neftuneNoiseAlpha")
    enable_liger_kernel: bool = Field(False, alias="enableLigerKernel")
    packing: bool = Field(False, alias="packing")
    # RLVR-only reward: EITHER a preset (gsm8k|prime_math) OR a custom
    # reward_function_id (a deployed reward function). Exactly one when stage=rlvr
    # (enforced by Hyperparams); both ignored otherwise. A preset is resolved to an
    # auto-provisioned built-in reward function's Evaluator ARN at launch.
    preset_reward_function: str = Field("", alias="presetRewardFunction")
    reward_function_id: str = Field("", alias="rewardFunctionId")
    # RLAIF-only: the judge model the recipe scores rollouts with. "" = recipe
    # default. (The reward PROMPT comes from reward_function_id → a reward_prompt
    # record.) Ignored unless stage=rlaif.
    reward_model_id: str = Field("", alias="rewardModelId")
    # Parameterization: "lora"/"qlora" (adapter) or "full"/"freeze" (full-weight,
    # llama_factory + SFT only; small models only — see catalog.allowed_methods).
    finetuning_type: Literal["lora", "qlora", "full", "freeze"] = Field("lora", alias="finetuningType")
    quantization_bit: int | None = Field(None, alias="quantizationBit")
    lora_rank: int = Field(8, alias="loraRank")
    lora_alpha: int | None = Field(None, alias="loraAlpha")
    # LoRA variant modifier (rides finetuning_type=lora/qlora; ignored for
    # full/freeze): lora|dora|rslora|pissa|loraplus. loraplus carries the B/A LR
    # ratio. So the same model can race plain LoRA vs DoRA as distinct rows.
    lora_variant: Literal["lora", "dora", "rslora", "pissa", "loraplus"] = Field("lora", alias="loraVariant")
    loraplus_lr_ratio: float = Field(16.0, alias="loraplusLrRatio")
    freeze_trainable_layers: int = Field(2, alias="freezeTrainableLayers")
    learning_rate: float = Field(1.0e-4, alias="learningRate")
    num_train_epochs: float = Field(3.0, alias="numTrainEpochs")
    per_device_train_batch_size: int = Field(1, alias="perDeviceTrainBatchSize")
    gradient_accumulation_steps: int = Field(8, alias="gradientAccumulationSteps")
    cutoff_len: int | None = Field(None, alias="cutoffLen")
    save_steps: int = Field(500, alias="saveSteps")
    max_samples: int | None = Field(None, alias="maxSamples")
    early_stopping_enabled: bool = Field(False, alias="earlyStoppingEnabled")
    early_stopping_patience: int = Field(3, alias="earlyStoppingPatience")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}

    def to_hp(self) -> Hyperparams:
        return Hyperparams(
            engine=self.engine,
            stage=self.stage,
            pref_beta=self.pref_beta,
            pref_loss=self.pref_loss,
            simpo_gamma=self.simpo_gamma,
            kto_chosen_weight=self.kto_chosen_weight,
            kto_rejected_weight=self.kto_rejected_weight,
            neftune_noise_alpha=self.neftune_noise_alpha,
            enable_liger_kernel=self.enable_liger_kernel,
            packing=self.packing,
            preset_reward_function=self.preset_reward_function,
            reward_function_id=self.reward_function_id,
            reward_model_id=self.reward_model_id,
            finetuning_type=self.finetuning_type,
            quantization_bit=self.quantization_bit,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_variant=self.lora_variant,
            loraplus_lr_ratio=self.loraplus_lr_ratio,
            freeze_trainable_layers=self.freeze_trainable_layers,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            cutoff_len=self.cutoff_len,
            save_steps=self.save_steps,
            max_samples=self.max_samples,
            early_stopping_enabled=self.early_stopping_enabled,
            early_stopping_patience=self.early_stopping_patience,
        )


class RaceRequest(BaseModel):
    split_id: str = Field(..., alias="splitId")
    models: list[RaceModelConfig] = Field(...)  # 1 = single job, 2+ = a race
    name: str = Field("", alias="name")  # optional human-friendly label
    # Spot is a race-level cost toggle (not per-model): cheaper, interruptible,
    # checkpoint+resume. Changes nothing about the model produced.
    use_spot: bool = Field(False, alias="useSpot")
    # Max wall-clock per training job before SageMaker stops it. Default 5h so a
    # normal run isn't guillotined mid-training (the old 1h default killed
    # multi-epoch jobs as MaxRuntimeExceeded). Early stopping usually ends sooner.
    max_run_seconds: int = Field(18000, alias="maxRunSeconds")
    # Spot→on-demand fallback (opt-in, minutes): when a spot job can't get capacity
    # within this many minutes, auto-convert it to on-demand (stop + relaunch from
    # checkpoint). None/0 = off (on-demand costs ~3x spot). Only meaningful with
    # useSpot. Must be < maxRunSeconds/60 so we intervene before SageMaker gives up.
    spot_fallback_minutes: int | None = Field(None, alias="spotFallbackMinutes")
    # Shared decoding for the eval stage (identical across racers = fair).
    eval_max_new_tokens: int = Field(256, alias="evalMaxNewTokens")
    eval_temperature: float = Field(0.0, alias="evalTemperature")
    # Email addresses to notify when the whole run finishes (all entries terminal).
    # Prefilled in the UI with the user's Cognito email; they can edit/add. Empty =
    # no notification. Normalized + capped + SES-verification-kicked-off at launch.
    notify_emails: list[str] = Field(default_factory=list, alias="notifyEmails")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


def _prelaunch_quality_gate(split_id: str, race_models: list) -> list[str]:
    """Pre-launch dataset quality gate — runs SYNCHRONOUSLY on the request, BEFORE
    any (billable) job is dispatched, so the user gets an immediate 400 / warnings
    instead of a race that flips to FAILED in the worker or silently mis-trains.

    Two tiers:
      * HARD BLOCK (raises HTTPException 400): conditions that WOULD fail the job or
        waste a billable run —
          - the GRPO (rlvr/rlaif) effective-train-row floor of 128 (the same check
            map_hyperparameters makes in the worker; lifted here so it's synchronous);
          - a serverless data-conversion dry-run (convert_file) that the worker would
            otherwise hit only after launch — catches format issues (no assistant
            turn, empty chosen/rejected, empty ground_truth, 0 usable rows);
          - any `error`-severity profiler finding (malformed rows, leakage, mostly
            identical DPO pairs).
      * WARN (returned, non-blocking): `warning`/`info` profiler findings (truncation,
        low output diversity, KTO imbalance, DPO length bias) — surfaced so the user
        can decide, the same way reward-domain warnings are returned.

    Returns the list of non-blocking warning strings. Reuses the REAL engine/profiler
    logic (map_hyperparameters, serverless_data.convert_file, profiler.warnings_from)
    so the gate can't drift from what the worker actually does."""
    from .engines.sagemaker_serverless import (
        _RECIPE_TRAIN_SPLIT, map_hyperparameters)
    from . import profiler as _profiler

    # split_meta / split_dir are referenced via the module globals (imported at
    # top of main.py) so tests that monkeypatch them are honored and the gate stays
    # consistent with the rest of the launch path.
    warnings: list[str] = []
    meta = split_meta(split_id) or {}
    has_val = bool(meta.get("valRows"))
    run_dir = split_dir(split_id)

    # 1) Profiler-based quality findings (errors block, warnings/info are returned).
    #    Advisory by design — never let a profiler hiccup fail a launch.
    try:
        cutoffs = [getattr(m.hp, "cutoff_len", None) for m in race_models]
        cutoff = next((c for c in cutoffs if c), None)
        prof = _profiler.profile_dataset(split_id, cutoff_len=cutoff)
        for w in _profiler.warnings_from(prof):
            msg = w.get("message", "")
            if not msg:
                continue
            if w.get("severity") == "error":
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset quality check failed: {msg} Fix the dataset before launching.")
            warnings.append(msg)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — profiler is advisory; never crash a launch on it
        pass

    # 2) Per-objective HARD checks that mirror what the worker does at launch.
    for m in race_models:
        stage = getattr(m.hp, "stage", "sft")
        engine = getattr(m.hp, "engine", "llama_factory")
        # GRPO row floor (128 effective). The engine raises this in the worker; do it
        # synchronously here so the user can't dispatch a doomed RLVR/RLAIF run. Use
        # the SAME effective-row estimate the engine uses (0.9 carve when no val set).
        # Prefer meta.trainRows; fall back to counting the on-disk train file (some
        # persist paths don't stamp trainRows) so the floor check is always honest.
        if stage in ("rlvr", "rlaif"):
            raw = meta.get("trainRows")
            if not (isinstance(raw, int) and raw > 0) and run_dir is not None:
                from pathlib import Path as _Path
                train_f = _Path(run_dir) / "train.jsonl"
                if train_f.exists():
                    raw = sum(1 for ln in train_f.read_text(encoding="utf-8").splitlines() if ln.strip())
            if isinstance(raw, int) and raw > 0:
                effective = raw if has_val else int(raw * _RECIPE_TRAIN_SPLIT)
                try:
                    map_hyperparameters(m.hp, train_rows=effective, raw_train_rows=raw)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
        # Serverless conversion dry-run: the worker reshapes the split to the recipe
        # format and FAILS the launch on a bad row. Do that dry-run here (to a temp
        # file) so format issues surface as a synchronous 400, not a FAILED entry.
        # Guard on a real on-disk train.jsonl (run_dir may be None, or a stub in tests).
        if engine == "sagemaker_serverless":
            import tempfile
            from pathlib import Path as _Path
            from .engines import serverless_data as _sd

            train_src = _Path(run_dir) / "train.jsonl" if run_dir else None
            if train_src is not None and train_src.exists():
                with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=True) as tmp:
                    try:
                        _sd.convert_file(train_src, _Path(tmp.name), stage)
                    except _sd.DataConversionError as e:
                        raise HTTPException(
                            status_code=400,
                            detail=(f"This dataset can't be converted for {stage.upper()} on the "
                                    f"serverless engine: {e}. Fix the dataset format before launching."))
    return warnings


@app.post("/api/race")
def race_launch(req: RaceRequest) -> dict:
    """Launch one fine-tuning job per model in parallel on the shared split.

    Handles BOTH the single-model case (one job) and the multi-model race. Each
    model carries its own hyperparameters; eval decoding is shared for fairness.
    """
    if split_dir(req.split_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown split id: {req.split_id}")
    if len(req.models) < 1:
        raise HTTPException(status_code=400, detail="need at least 1 model")

    # Objective ↔ dataset-shape guard: each objective needs its matching data
    # shape — sft↔messages, dpo↔preference (ranking), kto↔kto (labelled). The split
    # records its shape in meta.json (sft splits have no/"sft" shape). A mismatch
    # fails the engine cryptically mid-job, so reject early with a clear message
    # rather than burning a billable launch.
    shape = (split_meta(req.split_id) or {}).get("shape", "sft")
    # Which dataset shape each objective requires. RLVR has its OWN shape (prompt +
    # verifiable ground_truth) — distinct from SFT so a plain SFT dataset can't be
    # silently misused for reward-based training.
    REQUIRED_SHAPE = {"sft": "sft", "dpo": "preference", "kto": "kto",
                      "rlvr": "rlvr", "rlaif": "rlaif"}
    SHAPE_DESC = {
        "sft": "messages (prompt → response)",
        "preference": "preference (chosen/rejected pairs)",
        "kto": "KTO (completions labelled good/bad)",
        "rlvr": "RLVR (prompt + verifiable ground_truth)",
        "rlaif": "RLAIF (prompt-only; AI judge)",
    }
    for m in req.models:
        need = REQUIRED_SHAPE.get(m.stage, "sft")
        if need != shape:
            raise HTTPException(
                status_code=400,
                detail=(f"{m.stage.upper()} needs a {SHAPE_DESC.get(need, need)} dataset, but this split is "
                        f"{SHAPE_DESC.get(shape, shape)}-shaped. Use a matching dataset or objective."),
            )

    # Serverless-engine guards (clear 400s before a billable launch):
    #   - the engine must be enabled (dark-launch flag), and
    #   - the model must have a SageMaker Public Hub mapping.
    # The (stage, method) capability gate lives in Hyperparams (raised by to_hp()).
    from .engines.base import engine_enabled

    for m in req.models:
        if m.engine == "llama_factory":
            continue
        if not engine_enabled(m.engine):
            raise HTTPException(
                status_code=400,
                detail=(f"The {m.engine} engine is not enabled on this deployment. "
                        "Use the LLaMA-Factory engine, or enable it in Settings."),
            )
        spec = get_model(m.model_id)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"unknown model id: {m.model_id}")
        if not getattr(spec, "serverless_model_id", ""):
            raise HTTPException(
                status_code=400,
                detail=(f"{spec.display_name} has no SageMaker serverless equivalent — "
                        "it can only be fine-tuned with the LLaMA-Factory engine."),
            )

    try:
        race_models = [
            RaceModel(model_id=m.model_id, hp=m.to_hp(), instance_type=m.instance_type)
            for m in req.models
        ]
    except ValueError as e:
        # Hyperparams gate (e.g. an engine that can't do the requested stage/method).
        raise HTTPException(status_code=400, detail=str(e))

    # Pre-launch dataset quality gate (synchronous): hard-blocks issues that would
    # fail the job or waste a billable run (GRPO row floor, bad serverless conversion,
    # error-severity profiler findings) and collects non-blocking quality warnings.
    quality_warnings = _prelaunch_quality_gate(req.split_id, race_models)

    # RLVR reward-domain check (advisory, non-blocking): does the chosen reward
    # match the dataset's ground_truth shape? A gsm8k/prime_math preset on prose,
    # or a numeric_match custom reward on text, scores ~0 every step — a wasted
    # (billable) GRPO run. We profile the ground_truth task once and warn; we do
    # NOT block (the user may have a delimiter the reward extracts).
    reward_warnings: list[str] = []
    if any(m.stage == "rlvr" for m in req.models):
        from .reward_functions import get_reward_function, reward_domain_warning

        gt_task = None
        try:
            from .profiler import profile_dataset

            prof = profile_dataset(req.split_id)
            gt_task = (prof.get("rlvr") or {}).get("groundTruthTask")
        except Exception:  # noqa: BLE001 — advisory; never fail a launch on the profiler
            gt_task = None
        if gt_task:
            for m in req.models:
                if m.stage != "rlvr":
                    continue
                metric = ""
                if m.reward_function_id:
                    rf = get_reward_function(m.reward_function_id) or {}
                    metric = rf.get("metric") or ""  # only metric-kind rewards are checkable
                w = reward_domain_warning(
                    preset=m.preset_reward_function, metric=metric, ground_truth_task=gt_task)
                if w and w not in reward_warnings:
                    reward_warnings.append(w)

    # Reasoning-aware eval budget: if any racer is a reasoning family (Qwen3, R1,
    # GLM-Z1, gpt-oss), raise the SHARED eval max_new_tokens to the reasoning floor
    # so its <think> CoT closes — an unclosed block gets stripped to "" and scores
    # 0 (see catalog.reasoning_eval_floor). Only ever raises a too-low value; the
    # eval stays identical across racers (fairness). Logged so the bump is visible.
    from .catalog import reasoning_eval_floor
    from .obs import log_event

    eff_max_new = reasoning_eval_floor([m.model_id for m in req.models], req.eval_max_new_tokens)
    if eff_max_new != req.eval_max_new_tokens:
        log_event("race.eval.reasoning_token_bump", requested=req.eval_max_new_tokens,
                  effective=eff_max_new, models=[m.model_id for m in req.models])
    decoding = DecodingParams(
        max_new_tokens=eff_max_new, temperature=req.eval_temperature
    )
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        race = start_race(req.split_id, race_models, decoding, stamp, name=req.name,
                          use_spot=req.use_spot, max_run_seconds=req.max_run_seconds,
                          spot_fallback_minutes=req.spot_fallback_minutes,
                          notify_emails=req.notify_emails)
    except LimitExceeded as e:
        # Cost guardrail — a client error (the request asked for too much), 400.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"race launch failed: {e}")
    # Kick off SES verification for any not-yet-verified notification recipient so
    # the AWS verification email goes out immediately (best-effort, never fails the
    # launch). The UI uses the per-address status to tell the user to click the link.
    notify_status = ensure_notify_recipients_verified(race.notify_emails)
    return {"raceId": race.race_id, "name": race.name, "entries": rank_entries(race),
            # rewardWarnings: reward↔dataset-domain mismatches (RLVR). qualityWarnings:
            # non-blocking dataset-quality flags (truncation, low diversity, KTO skew,
            # DPO length bias) from the synchronous pre-launch gate.
            "rewardWarnings": reward_warnings,
            "qualityWarnings": quality_warnings,
            # Per-address SES verification status (verified | pending) so the UI can
            # tell the user which addresses still need to click the AWS link.
            "notify": notify_status}


@app.get("/api/races")
def races() -> dict:
    """List all races (summary). Includes archived ones; the UI filters by default.
    When the user has enabled sample runs, the curated showcase races are unioned
    in (tagged isSample=true) so a new user has something to explore."""
    from .samples import overlay_races

    return {"races": overlay_races(list_races())}


@app.get("/api/samples/status")
def samples_status() -> dict:
    """Whether the current user has sample runs enabled, and how many exist to
    show. Drives the Settings 'Import sample run' button's state."""
    from .samples import sample_race_ids, samples_enabled

    return {"enabled": samples_enabled(), "sampleCount": len(sample_race_ids())}


class SamplesUpdate(BaseModel):
    enabled: bool = True


@app.post("/api/samples/import")
def samples_import(update: SamplesUpdate) -> dict:
    """Enable (or disable) the curated sample runs for the current user. This does
    NOT copy data — it flips a per-user flag so the list/detail/dataset read paths
    union in the shared sample namespace (read-only). New users go from a blank app
    to a real dataset → fine-tune → leaderboard to explore."""
    from .samples import sample_race_ids, set_samples_enabled

    set_samples_enabled(update.enabled)
    return {"enabled": update.enabled, "sampleCount": len(sample_race_ids())}


@app.get("/api/limits")
def get_limits() -> dict:
    """Current cost guardrails (shown on Settings; enforced at race launch)."""
    return limits_summary()


# --- LLM-as-judge: quality scoring of an eval job's predictions (Bedrock) ---


@app.post("/api/judge/{eval_job}")
def start_judge(eval_job: str, judge_key: str | None = None) -> dict:
    """Start an LLM-as-judge pass over a completed eval job's predictions.

    judge_key picks the judge model (any baseline key — Claude or Nova, via the
    Converse API); defaults to Sonnet 4.5. Like the baseline, this can exceed API
    Gateway's 29s limit (one Bedrock call per eval row), so hosted it dispatches
    the worker Lambda and returns {status: running}; poll GET /api/judge/{eval_job}.
    Local dev runs inline. Idempotent: if already judged, returns the cached result.
    """
    cached = load_judge(eval_job)
    if cached is not None:
        return {"evalJob": eval_job, "status": "done", "result": cached}

    if dispatch_worker({"task": "judge", "evalJob": eval_job, "judgeKey": judge_key}):
        set_judge_status(eval_job, "running")
        return {"evalJob": eval_job, "status": "running"}

    try:
        result = run_judge(eval_job, judge_key=judge_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"judge failed: {e}")
    return {"evalJob": eval_job, "status": "done", "result": result}


@app.get("/api/judge/{eval_job}")
def get_judge(eval_job: str) -> dict:
    """Poll a judge run: {status: none|running|done|failed, result?}."""
    st = judge_status(eval_job)
    return {"evalJob": eval_job, **st, "result": load_judge(eval_job)}


@app.post("/api/race/{race_id}/archive")
def race_archive(race_id: str, archived: bool = True) -> dict:
    """Archive (hide) or restore a race. Display-only — never touches its jobs.

    `archived=true` hides it from the default Races list; `archived=false`
    restores it. Returns the updated race summary fields.
    """
    race = set_archived(race_id, archived)
    if race is None:
        raise HTTPException(status_code=404, detail=f"unknown race: {race_id}")
    return {"raceId": race.race_id, "name": race.name, "archived": race.archived}


@app.post("/api/race/{race_id}/retry")
def race_retry(race_id: str, model_id: str, resume: bool = False, rank_metric: str = "") -> dict:
    """Retry a FAILED entry from the stage that failed (train or eval).

    `resume=true` resumes a failed TRAINING run from its last checkpoint (only
    honoured when the entry is resumable — LLaMA-Factory with a recorded checkpoint
    prefix; ignored otherwise, falling back to a fresh retrain)."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    try:
        race = retry_entry(race_id, model_id, stamp, resume=resume)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"retry failed: {e}")
    if race is None:
        raise HTTPException(status_code=404, detail=f"unknown race: {race_id}")
    # Return the FULL race shape (same as race_status) — the UI replaces its detail
    # state with this response, so omitting splitId/useSpot/rankMetric would blank
    # the dataset link + spot badge until the next poll.
    eff = effective_rank_metric(race, rank_metric)
    return {
        "raceId": race.race_id,
        "name": race.name,
        "splitId": race.split_id,
        "useSpot": race.use_spot,
        "rankMetric": eff,
        "rankMetrics": RANK_METRICS,
        "entries": rank_entries(race, eff),
    }


@app.get("/api/race/{race_id}")
def race_status(race_id: str, rank_metric: str = "") -> dict:
    """Reconcile + return a race with ranked entries (winner first).

    `rank_metric` (query param) chooses which eval metric decides the winner.

    A SAMPLE race (curated showcase) is read from the shared sample namespace and
    NEVER reconciled — it's completed + shared, so mutating it would affect every
    user. We just load + rank it (read-only) and tag the response isSample=true.
    """
    from .samples import SAMPLES_TENANT, is_sample_race
    from .tenancy import tenant_scope

    if is_sample_race(race_id):
        from .race import _load

        with tenant_scope(SAMPLES_TENANT):
            race = _load(race_id)
            if race is None:
                raise HTTPException(status_code=404, detail=f"unknown race: {race_id}")
            eff = effective_rank_metric(race, rank_metric)
            entries = rank_entries(race, eff)
        return {
            "raceId": race.race_id, "name": race.name, "splitId": race.split_id,
            "useSpot": race.use_spot, "rankMetric": eff, "rankMetrics": RANK_METRICS,
            "entries": entries, "isSample": True,
        }

    race = reconcile_race(race_id)
    if race is None:
        raise HTTPException(status_code=404, detail=f"unknown race: {race_id}")
    # An RLAIF race is always ranked by judge reward (no gold to score against), so
    # report the EFFECTIVE metric — not the requested one — so the UI's "Rank by"
    # control reflects what actually decided the winner.
    eff = effective_rank_metric(race, rank_metric)
    return {
        "raceId": race.race_id,
        "name": race.name,
        "splitId": race.split_id,
        "useSpot": race.use_spot,
        "rankMetric": eff,
        "rankMetrics": RANK_METRICS,
        "entries": rank_entries(race, eff),
    }


# --- HuggingFace token (Secrets Manager) — unlocks gated models ---


class HfTokenUpdate(BaseModel):
    token: str


@app.get("/api/hf-token")
def hf_token_status() -> dict:
    """Whether an HF token is stored (never returns the value).

    isSet = this user's OWN token; usingSharedFallback = they have none but a
    shared fallback token is covering for them (the UI nags them to set their
    own — borrowed access runs under the owner's HF account + licenses)."""
    from .secrets import get_hf_token

    own = hf_token_is_set()
    return {"isSet": own, "usingSharedFallback": (not own) and get_hf_token() is not None}


@app.put("/api/hf-token")
def set_hf_token_endpoint(update: HfTokenUpdate) -> dict:
    """Store the HF token in Secrets Manager (write-only; enables gated models)."""
    if not update.token.strip():
        raise HTTPException(status_code=400, detail="token must not be empty")
    try:
        set_hf_token(update.token)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"could not store token: {e}")
    return {"isSet": hf_token_is_set()}


# --- Settings: AWS environment config + preflight (for portability) ---


class ConfigUpdate(BaseModel):
    region: str | None = None
    account: str | None = None
    bucket: str | None = None
    role_arn: str | None = Field(None, alias="roleArn")
    image_uri: str | None = Field(None, alias="imageUri")
    profile: str | None = None
    # Runtime toggle for the SageMaker serverless engine. Persisted to config.json
    # as `enableSagemakerServerless`, which engines.base.engine_enabled() reads as
    # the SINGLE source of truth — so flipping this in Settings turns the engine
    # on/off WITHOUT a redeploy. It defaults ON; there is no env override.
    enable_sagemaker_serverless: bool | None = Field(None, alias="enableSagemakerServerless")

    model_config = {"populate_by_name": True}


@app.get("/api/config")
def get_config() -> dict:
    """Return the effective AWS config (saved > env > default). No secrets here —
    these are account ids / ARNs / bucket names, safe to display."""
    from .engines.base import engine_enabled

    serverless_enabled = engine_enabled("sagemaker_serverless")
    try:
        cfg = load_aws_config()
    except AwsAccountUnresolvedError as e:
        # First run with no credentials and no SLM_AWS_ACCOUNT. This endpoint backs
        # the Settings page, which is where the account is entered, so it must NOT
        # fail — a 500 here would leave the user no way to fix the thing it is
        # complaining about. Return the fields that ARE known and blank the ones
        # derived from the account, with the reason attached.
        return {
            "region": resolve_region(),
            "account": "",
            "bucket": "",
            "roleArn": "",
            "imageUri": "",
            "profile": resolve_profile(),
            "enableSagemakerServerless": serverless_enabled,
            "configError": str(e),
        }
    return {
        "region": cfg.region,
        "account": cfg.account_id,
        "bucket": cfg.bucket,
        "roleArn": cfg.role_arn,
        "imageUri": cfg.image_uri,
        "profile": cfg.profile,
        # Effective state of the serverless engine (saved config, default ON). The
        # Settings toggle reflects + sets this; the catalog/picker honor it.
        "enableSagemakerServerless": serverless_enabled,
    }


@app.put("/api/config")
def put_config(update: ConfigUpdate) -> dict:
    """Persist config overrides (Settings page). Only non-None fields saved.

    by_alias=True so keys persist in the SAME camelCase form the readers use
    (load_aws_config reads "roleArn"/"imageUri"; engine_enabled reads
    "enableSagemakerServerless" as the single source of truth). Booleans (the
    serverless toggle) persist as-is — save_config only skips None / empty
    strings, so toggling OFF (False) is saved and sticks."""
    save_config(update.model_dump(by_alias=True, exclude_none=True))
    return get_config()


@app.post("/api/config/check")
def check_config() -> dict:
    """Run the read-only preflight: creds, account match, bucket, role, image."""
    try:
        return preflight()
    except Exception as e:
        # The botocore text here names the execution-role ARN on an AccessDenied and
        # the bucket/account on others. Log it; return a fixed message.
        from .obs import log_event

        log_event("config.preflight.failed", level="WARNING",
                  error=f"{type(e).__name__}: {e}")
        raise HTTPException(
            status_code=502,
            detail="preflight failed; see the API function's CloudWatch logs for detail",
        ) from e


@app.post("/api/config/reset-cutoff")
def set_reset_cutoff() -> dict:
    """Set the reset cutoff to NOW: hides all existing SageMaker jobs from the UI
    (their records can't be deleted). Local state + S3 are cleared separately."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    save_config({"resetCutoff": now})
    return {"resetCutoff": now}


# --- Settings: per-agent model selection -------------------------------------


class AgentModelsUpdate(BaseModel):
    """{roleKey: baselineModelKey} overrides. A value equal to the role default
    (or empty) clears that role's override (tracks the default)."""
    overrides: dict[str, str] = Field(default_factory=dict)


@app.get("/api/agent-models")
def get_agent_models() -> dict:
    """Which Bedrock model each AI agent uses, the selectable models, and which
    roles are deploy-time (their override applies only after the agent redeploys)."""
    from .agent_models import settings_view

    return settings_view()


@app.put("/api/agent-models")
def put_agent_models(update: AgentModelsUpdate) -> dict:
    """Persist per-agent model overrides (Settings page). Validates each role +
    model key; returns the fresh view. In-process agents (advisor/self-heal/judge)
    honor the change on the next call; deploy-time agents need a redeploy."""
    from .agent_models import set_overrides

    try:
        return set_overrides(update.overrides)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Guided Fine-tuning agent ("Pit Crew") — a conversational front door for non-ML
# users. Thin HTTP wrappers over the pitcrew.py state machine; all judgement lives
# there. The agent PROPOSES a race; only the explicit approve action launches it.
# =============================================================================
def _utc_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _new_session_id() -> str:
    """A filesystem-/S3-key-safe session id (stamp + short random suffix)."""
    import uuid
    from datetime import datetime, timezone

    return (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            + "-" + uuid.uuid4().hex[:8])


class PitcrewTurn(BaseModel):
    action: str
    payload: dict = Field(default_factory=dict)
    # Optimistic-concurrency guard: the version the client last saw. A stale value
    # (a second tab advanced the session) yields a 409 so the UI can reload.
    expected_version: int | None = Field(None, alias="expectedVersion")

    model_config = {"populate_by_name": True}


class DetectShapeRequest(BaseModel):
    # Either reference a presigned upload (large file) or send raw text (small).
    upload_id: str | None = Field(None, alias="uploadId")
    text: str | None = None

    model_config = {"populate_by_name": True}


@app.post("/api/datasets/detect-shape")
async def datasets_detect_shape(req: DetectShapeRequest) -> dict:
    """Classify raw JSONL into a dataset SHAPE (sft/preference/kto/rlvr/rlaif) so the
    guided agent can pick the right import path + describe the data in plain language,
    without the user choosing a type. Reads either an uploaded object or inline text."""
    from .validation import detect_shape

    if req.upload_id:
        text = _read_upload_text(req.upload_id)
    elif req.text is not None:
        text = req.text
    else:
        raise HTTPException(status_code=400, detail="provide uploadId or text")
    return detect_shape(text)


@app.get("/api/pitcrew/sessions")
def pitcrew_sessions() -> dict:
    """List the user's guided sessions (history sidebar), newest first."""
    from .pitcrew import list_sessions

    return {"sessions": list_sessions()}


@app.post("/api/pitcrew/sessions")
def pitcrew_new_session() -> dict:
    """Start a new guided session and return it (with the greeting message)."""
    from .pitcrew import start_session

    return start_session(_new_session_id(), _utc_stamp())


@app.get("/api/pitcrew/sessions/{session_id}")
def pitcrew_get_session(session_id: str) -> dict:
    """Load a guided session (reconciling its race if one launched)."""
    from .pitcrew import get_session

    session = get_session(session_id, _utc_stamp())
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return session


class PitcrewRename(BaseModel):
    title: str


@app.put("/api/pitcrew/sessions/{session_id}/title")
def pitcrew_rename_session(session_id: str, req: PitcrewRename) -> dict:
    """Rename a guided session (the sidebar label). User-chosen titles stick — the
    goal step won't overwrite them."""
    from .pitcrew import rename_session

    session = rename_session(session_id, req.title)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return session


@app.post("/api/pitcrew/sessions/{session_id}/archive")
def pitcrew_archive_session(session_id: str, archived: bool = True) -> dict:
    """Hide (or restore) a guided session from the sidebar. Soft-delete — never
    orphans a launched race."""
    from .pitcrew import archive_session

    if not archive_session(session_id, archived):
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return {"sessionId": session_id, "archived": archived}


@app.post("/api/pitcrew/sessions/{session_id}/advance")
def pitcrew_advance(session_id: str, turn: PitcrewTurn) -> dict:
    """Advance a guided session one user turn (the state machine decides what's next).
    The only billable action is approve, which launches the race via start_race."""
    from .pitcrew import advance

    try:
        return advance(session_id, turn.action, turn.payload, _utc_stamp(),
                       expected_version=turn.expected_version)
    except ValueError as e:
        msg = str(e)
        # A stale-version conflict is a 409 (reload), an unknown session a 404.
        if "another tab" in msg:
            raise HTTPException(status_code=409, detail=msg)
        if "unknown session" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


class PitcrewEdit(BaseModel):
    message_index: int = Field(..., alias="messageIndex")
    text: str
    expected_version: int | None = Field(None, alias="expectedVersion")

    model_config = {"populate_by_name": True}


@app.post("/api/pitcrew/sessions/{session_id}/edit")
def pitcrew_edit_message(session_id: str, req: PitcrewEdit) -> dict:
    """Edit/rewind an earlier free-text user message (the goal, or a task correction).
    Truncates the conversation there, unlinks any downstream dataset (kept on disk),
    and replays. Blocked once a race has launched. A stale expectedVersion → 409."""
    from .pitcrew import edit_message

    try:
        session = edit_message(session_id, req.message_index, req.text, _utc_stamp(),
                               expected_version=req.expected_version)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return session
