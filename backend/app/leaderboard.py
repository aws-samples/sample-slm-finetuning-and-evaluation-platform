# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Leaderboard aggregation — the comparison the project exists to produce.

A leaderboard is only meaningful WITHIN one eval set (same held-out rows + same
decoding for every candidate — the methodology rule). So it is SPLIT-SCOPED:
given a split id, it finds completed eval jobs on THAT split, collapses to the
LATEST eval per model (re-runs don't duplicate), and assembles one row per model
with quality + cost + latency. Comparing across different splits (e.g. a 3-row
vs a 72-row eval) is invalid and intentionally not offered.

Cost columns (honest measured-vs-derived labeling, per methodology):
  - train_cost_usd            MEASURED: train BillableTimeInSeconds × instance $/hr
  - eval_tokens_per_sec       MEASURED: from eval timing
  - p50_latency_ms            MEASURED: from eval timing
  - projected_self_host_cost_per_1k  DERIVED: (instance $/hr ÷ 3600) ÷ tok/s × 1000

The Sonnet baseline row is produced separately (baseline.py) on the SAME split,
with ACTUAL api_cost_per_1k, and merged in by the endpoint.
"""

from __future__ import annotations

import re
from typing import Any

from .aws_config import instance_hourly, load_aws_config, spot_discount_factor
from .orchestrate import _session, fetch_metrics, job_after_cutoff
from .storage import is_dataset_archived, split_dir, split_meta, split_name

# Source training job name is embedded in the eval job's `model` channel S3 URI:
#   .../jobs/<train-job-name>/output/<train-job-name>/output/model.tar.gz
_SRC_JOB_RE = re.compile(r"/jobs/(?P<job>slm-[^/]+?)/output/")


def _source_train_job(model_s3_uri: str) -> str | None:
    m = _SRC_JOB_RE.search(model_s3_uri)
    return m.group("job") if m else None


def _resolve_train_job(sm, base_or_full: str) -> str | None:
    """Resolve a training job name from the base name embedded in the S3 URI.

    SageMaker truncates long base_job_names and appends its own timestamp, so the
    real job name is neither equal to nor a prefix of the embedded base. Match on
    a stable prefix: 'slm-<modelid>-<splitid>' (everything through the 12-hex id).
    """
    try:
        sm.describe_training_job(TrainingJobName=base_or_full)
        return base_or_full
    except Exception:
        pass
    m = re.match(r"(slm-.+?-[0-9a-f]{12})", base_or_full)
    prefix = m.group(1) if m else base_or_full
    try:
        resp = sm.list_training_jobs(
            NameContains=prefix, MaxResults=10, SortBy="CreationTime", SortOrder="Descending"
        )
        for s in resp.get("TrainingJobSummaries", []):
            name = s["TrainingJobName"]
            if name.startswith(prefix) and not name.startswith("slm-eval-"):
                return name
    except Exception:
        return None
    return None


def _train_cost(sm, train_job_name: str) -> dict[str, Any]:
    """Measured train cost for a training job (billable seconds × instance rate).

    SPOT FIX: a spot job's BillableTimeInSeconds is billed at the spot rate, so
    multiplying by the on-demand rate overstates cost ~3×. When the job ran on
    managed spot we apply the configurable spot discount factor and flag the
    figure as an estimate (trainCostIsEstimate=True)."""
    resolved = _resolve_train_job(sm, train_job_name)
    if not resolved:
        return {"trainBillableSeconds": None, "trainInstance": None, "trainCostUsd": None,
                "trainSpot": False, "trainCostIsEstimate": False}
    try:
        d = sm.describe_training_job(TrainingJobName=resolved)
    except Exception:
        return {"trainBillableSeconds": None, "trainInstance": None, "trainCostUsd": None,
                "trainSpot": False, "trainCostIsEstimate": False}
    # Serverless model-customization jobs have NO ResourceConfig/instance and bill
    # per-token (SFT/DPO → BillableTimeInSeconds=0) or by duration (RFT). The
    # instance-hour formula doesn't apply, so report a serverless basis instead of
    # crashing on the missing ResourceConfig.
    if d.get("ServerlessJobConfig") is not None or "ResourceConfig" not in d:
        billable = d.get("BillableTimeInSeconds")
        secs = d.get("TrainingTimeInSeconds") or billable
        return {
            "trainBillableSeconds": billable,
            "trainInstance": "serverless",
            "trainCostUsd": None,  # token/duration-priced out-of-band; not derivable here
            "trainSpot": False,
            "trainCostIsEstimate": False,
            "trainServerless": True,
            "trainDurationSeconds": secs,
        }
    billable = d.get("BillableTimeInSeconds")
    instance = (d.get("ResourceConfig") or {}).get("InstanceType")
    is_spot = bool(d.get("EnableManagedSpotTraining"))
    rate = instance_hourly(instance)
    cost = None
    if billable and rate:
        cost = billable / 3600 * rate
        if is_spot:
            cost *= spot_discount_factor()  # spot billed below on-demand
        cost = round(cost, 4)
    return {"trainBillableSeconds": billable, "trainInstance": instance, "trainCostUsd": cost,
            "trainSpot": is_spot, "trainCostIsEstimate": is_spot}


def _projected_serve_cost_per_1k(instance: str | None, tokens_per_sec: float | None) -> float | None:
    rate = instance_hourly(instance) if instance else None
    if not rate or not tokens_per_sec:
        return None
    return round((rate / 3600) / tokens_per_sec * 1000, 4)


# Eval job name encodes the split: slm-eval-<12hex splitId>-<stamp...>
_EVAL_SPLIT_RE = re.compile(r"^slm-eval-([0-9a-f]{12})-")


def _eval_split_id(eval_job_name: str) -> str | None:
    m = _EVAL_SPLIT_RE.match(eval_job_name)
    return m.group(1) if m else None


def list_eval_splits(max_jobs: int = 60) -> list[dict[str, Any]]:
    """List split ids that have at least one completed eval job (newest first),
    with how many eval jobs each has — used to populate the split selector.

    SageMaker jobs are ACCOUNT-GLOBAL (not tenant-scoped — a job name can't carry
    a user prefix), but the dataset files a baseline/leaderboard needs live in the
    tenant-scoped `runs/` store. So we list a split ONLY if its dataset is actually
    present for the CURRENT tenant (split_dir not None) — otherwise the dropdown
    would offer phantom splits (another user's, or since-deleted) that fail with
    'unknown split id' when you try to baseline them."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    resp = sm.list_training_jobs(
        StatusEquals="Completed", SortBy="CreationTime", SortOrder="Descending",
        MaxResults=max_jobs,
    )
    seen: dict[str, dict[str, Any]] = {}
    for s in resp.get("TrainingJobSummaries", []):
        name = s["TrainingJobName"]
        sid = _eval_split_id(name)
        if not sid or not job_after_cutoff(s):
            continue
        if sid not in seen and split_dir(sid) is None:
            continue  # dataset not available for this tenant — don't offer it
        if is_dataset_archived(sid):
            continue  # archived datasets are hidden from the leaderboard too
        if sid not in seen:
            seen[sid] = {
                "splitId": sid,
                "name": split_name(sid),  # human dataset name, if given
                "evalJobs": 0,
                "latest": str(s.get("CreationTime")),
                # If dataset investigation recommended a ranking metric, surface
                # it so the leaderboard can default 'Rank by' to it.
                "recommendedRankMetric": split_meta(sid).get("recommendedRankMetric"),
            }
        seen[sid]["evalJobs"] += 1
    return list(seen.values())


def build_leaderboard(split_id: str, max_jobs: int = 60) -> list[dict[str, Any]]:
    """One row per MODEL for eval jobs on `split_id`, collapsed to the latest
    eval per model (re-runs don't duplicate). Comparison is valid because every
    row shares the same held-out eval set."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")

    resp = sm.list_training_jobs(
        StatusEquals="Completed", SortBy="CreationTime", SortOrder="Descending",
        MaxResults=max_jobs,
    )
    # Jobs come newest-first; first one seen per model = latest. Dedup on model.
    by_model: dict[str, dict[str, Any]] = {}
    for s in resp.get("TrainingJobSummaries", []):
        name = s["TrainingJobName"]
        if not name.startswith("slm-eval-") or _eval_split_id(name) != split_id:
            continue
        if not job_after_cutoff(s):
            continue
        d = sm.describe_training_job(TrainingJobName=name)
        # Base-model eval jobs are named slm-eval-<split>-base-<stamp> and have NO
        # model channel (eval.py loads the untrained model straight from HF). They
        # are the "did fine-tuning help?" control — surface them as their own
        # '<model> (base)' rows so base → fine-tuned → frontier all show here.
        is_base = "-base-" in name
        if is_base:
            base_model_id = (d.get("Environment", {}) or {}).get("SLM_EVAL_BASE_MODEL", "")
            model_label = f"{_hf_to_label(base_model_id)} (base)"
            src_job = None
        else:
            model_uri = next(
                (c["DataSource"]["S3DataSource"]["S3Uri"]
                 for c in d.get("InputDataConfig", []) if c["ChannelName"] == "model"),
                "",
            )
            src_job = _source_train_job(model_uri)
            model_label = _model_label(src_job)
        if model_label in by_model:
            continue  # already have this model's latest eval

        metrics = fetch_metrics(name)
        if not metrics:
            continue
        train = _train_cost(sm, src_job) if src_job else {
            "trainBillableSeconds": None, "trainInstance": None, "trainCostUsd": None,
            "trainSpot": False, "trainCostIsEstimate": False,
        }
        timing = metrics.get("timing", {}) or {}
        tps = timing.get("tokens_per_sec")
        eval_instance = d["ResourceConfig"]["InstanceType"]

        by_model[model_label] = {
            "evalJob": name,
            "sourceJob": src_job,
            "model": model_label,
            "splitId": split_id,
            "count": metrics.get("count"),
            "exactMatch": metrics.get("exact_match"),
            "normalizedMatch": metrics.get("normalized_match"),
            "containsGold": metrics.get("contains_gold"),
            "tokenF1": metrics.get("token_f1"),
            "rougeL": metrics.get("rouge_l"),
            "charF1": metrics.get("char_f1"),
            "lengthRatio": metrics.get("length_ratio"),
            "jsonStructural": metrics.get("json_structural"),
            # RLAIF judge reward (reference-free); only RLAIF eval rows carry it, so
            # it's None for everyone else + dropped here before. Threaded through so
            # an RLAIF winner shows a real number on the cross-model leaderboard.
            "rewardMean": metrics.get("reward_mean"),
            "backend": metrics.get("decoding", {}).get("backend"),
            "tokensPerSec": tps,
            "p50LatencyMs": timing.get("p50_latency_ms"),
            # Tail latency — p90/p99 expose the slow requests p50 hides (None for
            # older eval runs that didn't record them; the UI degrades gracefully).
            "p90LatencyMs": timing.get("p90_latency_ms"),
            "p99LatencyMs": timing.get("p99_latency_ms"),
            "trainCostUsd": train["trainCostUsd"],
            "trainInstance": train["trainInstance"],
            "trainSpot": train["trainSpot"],
            "trainCostIsEstimate": train["trainCostIsEstimate"],
            "projectedServeCostPer1k": _projected_serve_cost_per_1k(eval_instance, tps),
            "evalInstance": eval_instance,
            # A base-model control row is a kind of baseline (untrained reference),
            # so flag it as such; `kind` lets the UI distinguish base-SLM from a
            # frontier (API) baseline if it wants to.
            "isBaseline": is_base,
            "kind": "base" if is_base else "finetuned",
            # Fine-tuning METHOD parsed from the job-name tag (orchestrate._job_name
            # appends -<method> for non-lora). Lets the results-interpreter agent
            # weigh a full/freeze candidate's higher cost + catastrophic-forgetting
            # risk (vs LoRA) when recommending what to ship. "lora" when untagged.
            "method": _method_from_label(model_label),
            "creationTime": str(s.get("CreationTime")),
        }
    return list(by_model.values())


def _method_from_label(model_label: str) -> str:
    """Recover the fine-tuning method from a model label whose job-name tag encodes
    it as a `-<method>` suffix (e.g. 'qwen3-1-7b-full' -> 'full'). Defaults to
    'lora' (the untagged case)."""
    for m in ("qlora", "freeze", "full"):
        if model_label.endswith(f"-{m}"):
            return m
    return "lora"


def _hf_to_label(hf_model_id: str) -> str:
    """Label a base-eval row by the SAME identity as its fine-tuned sibling: the
    catalog id (e.g. 'qwen2.5-0.5b'), so the base and fine-tuned rows line up
    instead of showing two different-looking names (catalog id vs HF repo name).
    Falls back to the HF repo name, then a generic label."""
    if not hf_model_id:
        return "base model"
    from .catalog import model_id_for_hf

    cat_id = model_id_for_hf(hf_model_id)
    if cat_id:
        return cat_id
    return hf_model_id.split("/", 1)[-1]


def _model_label(src_job: str | None) -> str:
    """slm-qwen3-4b-<split>-<stamp...> -> 'qwen3-4b' (best effort)."""
    if not src_job:
        return "unknown"
    rest = src_job[len("slm-"):] if src_job.startswith("slm-") else src_job
    # strip the trailing split id + timestamps; keep the model-id head.
    parts = rest.split("-")
    # model ids in the catalog look like qwen3-4b, llama-3.1-8b, gpt-oss-20b…
    # heuristic: take parts until we hit the 12-hex split id.
    keep: list[str] = []
    for p in parts:
        if re.fullmatch(r"[0-9a-f]{12}", p):
            break
        keep.append(p)
    return "-".join(keep) or rest
