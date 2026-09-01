# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Rough cost + time estimate for a race — the number a non-ML user approves on.

The Guided Fine-tuning agent (pitcrew.py) gates the ONLY billable action behind a
human approval screen, so that screen needs an honest "what will this cost / how
long?" figure. The platform has never had a cost model (only hardcoded $/hr
strings in the catalog UI), so this is it: a deterministic, no-AWS-call estimate
built from a static per-instance price table × a coarse runtime heuristic.

DESIGN STANCE — deliberately a RANGE, clearly labelled an estimate:
  * Real training time depends on tokens/row, GPU contention, spot interruptions,
    and early stopping (which usually ends sooner), none of which we know up front.
    So every figure is a lo–hi RANGE (±40%) and the caller shows it with a
    disclaimer, never a false-precision single number.
  * Deterministic + offline: same inputs → same estimate (no Pricing API call,
    no latency/IAM/failure modes in a synchronous wizard step). Prices are a
    maintained constant, mirroring how the catalog UI already quotes them.

WHAT A RACE ACTUALLY BILLS — start_race launches up to THREE SageMaker jobs per
model (race.py): the training job, an eval job on the held-out set, and a
base-model eval (skipped for RLAIF). All three bill instance-time, so the cost
sums them; the jobs run in PARALLEL across models, so wall-clock is the SLOWEST
single model's chain, not the sum.
"""

from __future__ import annotations

import math
from typing import Any

# --- Static price table (US-East-1 SageMaker *training* on-demand $/hr) --------
# Maintained-by-hand constants (the platform has no live pricing call in the
# synchronous wizard). VERIFIED 2026-07-02 against the AWS Pricing API
# (AmazonSageMaker, regionCode us-east-1, Training usagetype). g6e.2xlarge is pinned
# to the same ~$2.80/hr the catalog UI quotes, so the two surfaces never disagree.
# Covers every instance _instance_for can assign PLUS the g6e.4/8xlarge that
# limits.allowed_instance_types permits via a manual override — so a valid launch
# never silently falls back to _FALLBACK_HOURLY_USD (which would misprice g6e.8xlarge
# by ~2×). Update alongside any instance the catalog/limits add.
INSTANCE_HOURLY_USD: dict[str, float] = {
    "ml.g5.2xlarge": 1.52,    # Pricing API: 1.515
    "ml.g5.4xlarge": 2.03,
    "ml.g5.8xlarge": 3.06,    # was 2.72 (wrong; under-quoted 7-8B LoRA ~11%) — API: 3.06
    "ml.g5.12xlarge": 7.09,
    "ml.g6e.2xlarge": 2.80,   # matches CatalogPage's "~$2.80/hr" full-weight quote
    "ml.g6e.4xlarge": 3.76,   # allowed via manual override (not auto-assigned)
    "ml.g6e.8xlarge": 5.66,   # allowed via manual override (not auto-assigned)
    "ml.g6e.12xlarge": 13.10,  # API: 13.12
}
# Fallback $/hr for an instance not in the table (rather than crash a wizard step).
_FALLBACK_HOURLY_USD = 3.00

# Managed spot saves ~65% vs on-demand (interruptible + checkpoint/resume). Race
# spot is a launch-level toggle; when on, the whole race bills at this fraction.
SPOT_DISCOUNT = 0.35  # i.e. spot ≈ 35% of on-demand

# --- Runtime heuristic ---------------------------------------------------------
# Effective optimizer batch = per_device_batch(1) × grad_accum(8), matching
# recommend.py / render.py — so train STEPS = rows × epochs / 8.
_EFFECTIVE_BATCH = 8

# Seconds per optimizer step, banded by model size. LoRA on the catalog's g5/g6e
# cards; full-weight is heavier (more to update + checkpoint), so a multiplier
# applies below. Coarse anchors from observed runs, not a benchmark — the ±range
# absorbs the slop.
def _seconds_per_step(params_b: float) -> float:
    if params_b <= 2:
        return 0.5
    if params_b <= 4:
        return 1.0
    if params_b <= 8:
        return 1.8
    if params_b <= 14:
        return 3.0
    return 4.5


# Full-weight methods touch every parameter (optimizer state + checkpoint I/O for
# the whole model), so they run materially slower per step than a LoRA adapter.
_FULL_WEIGHT_STEP_MULTIPLIER = 2.5

# Fixed per-JOB overhead (minutes): instance provisioning + container image pull
# before the training loop starts. Each of the 3 jobs/model pays it.
_JOB_OVERHEAD_MIN = 8.0

# An eval job is an inference pass over the held-out set — far cheaper than
# training. Estimated as eval_rows × a per-row generation cost, banded by size.
def _eval_seconds(eval_rows: int, params_b: float) -> float:
    per_row = 0.4 if params_b <= 4 else 0.9 if params_b <= 8 else 1.6
    return eval_rows * per_row


# Training compute scales with TOKENS, not rows: a dataset of long documents trains
# far longer than the same row-count of short ones. The per-step anchors above are
# calibrated for a ~512-token reference example; this scales seconds-per-step by how
# long the actual examples are (p95 tokens ÷ 512), clamped to a sane band so a
# pathological outlier can't 100× the quote. `seq_tokens=0`/None → factor 1.0
# (backward-compatible with callers that don't pass a length signal).
_SEQ_REFERENCE_TOKENS = 512.0


def _seq_factor(seq_tokens: float | None) -> float:
    if not seq_tokens or seq_tokens <= 0:
        return 1.0
    return max(0.5, min(4.0, seq_tokens / _SEQ_REFERENCE_TOKENS))


# Uncertainty band applied to every figure (±). Training time is genuinely noisy;
# a single number would be false precision for a non-expert's spend decision.
_RANGE_FRAC = 0.40


def _round_money(x: float) -> float:
    """Round to cents for small figures, to the dollar once it's into the tens."""
    return round(x, 2) if x < 10 else round(x)


def estimate_job_minutes(
    *,
    train_rows: int,
    eval_rows: int,
    epochs: float,
    params_b: float,
    method: str,
    base_eval: bool,
    seq_tokens: float | None = None,
) -> float:
    """Estimated WALL-CLOCK minutes for ONE model's full chain (train → eval, plus
    a parallel base eval). The eval(s) overlap training only partly, so we sum
    train + eval; base eval runs alongside training so it never extends the chain
    beyond the train+eval total — it adds COST (its own job-time) but not time.
    `seq_tokens` (p95 tokens/example) scales the per-step time — long-context data
    trains multiples longer than the same row-count of short examples."""
    steps = math.ceil(max(1, train_rows) * max(0.1, epochs) / _EFFECTIVE_BATCH)
    sps = _seconds_per_step(params_b) * _seq_factor(seq_tokens)
    if method in ("full", "freeze"):
        sps *= _FULL_WEIGHT_STEP_MULTIPLIER
    train_min = _JOB_OVERHEAD_MIN + (steps * sps) / 60.0
    eval_min = _JOB_OVERHEAD_MIN + _eval_seconds(eval_rows, params_b) / 60.0
    return train_min + eval_min


def estimate_job_cost_usd(
    *,
    instance_type: str,
    train_rows: int,
    eval_rows: int,
    epochs: float,
    params_b: float,
    method: str,
    base_eval: bool,
    use_spot: bool,
    seq_tokens: float | None = None,
) -> float:
    """Estimated $ billed for ONE model = (train + eval + optional base-eval) job
    minutes × the instance rate. All three jobs run on the SAME instance type.
    `seq_tokens` scales training time with example length (see estimate_job_minutes)."""
    rate = INSTANCE_HOURLY_USD.get(instance_type, _FALLBACK_HOURLY_USD)
    if use_spot:
        rate *= SPOT_DISCOUNT
    steps = math.ceil(max(1, train_rows) * max(0.1, epochs) / _EFFECTIVE_BATCH)
    sps = _seconds_per_step(params_b) * _seq_factor(seq_tokens)
    if method in ("full", "freeze"):
        sps *= _FULL_WEIGHT_STEP_MULTIPLIER
    train_min = _JOB_OVERHEAD_MIN + (steps * sps) / 60.0
    eval_min = _JOB_OVERHEAD_MIN + _eval_seconds(eval_rows, params_b) / 60.0
    billed_min = train_min + eval_min
    if base_eval:
        billed_min += eval_min  # base eval is a second, parallel eval-shaped job
    return (billed_min / 60.0) * rate


def estimate_race(
    entries: list[dict[str, Any]],
    *,
    train_rows: int,
    eval_rows: int,
    use_spot: bool = False,
    seq_tokens: float | None = None,
) -> dict[str, Any]:
    """Estimate cost + wall-clock for a whole race.

    `entries` = one dict per model, each with keys:
        instanceType, paramsB, epochs, method (finetuning_type), baseEval (bool).
    Cost SUMS every model's jobs (each bills its own instance-time); wall-clock is
    the SLOWEST single model's chain (the race fans out in parallel). `seq_tokens`
    (p95 tokens/example, from the profile) scales training time with example length
    so long-context data isn't badly under-quoted. Every figure is a lo–hi RANGE.
    """
    if not entries:
        return {"totalUsd": {"lo": 0.0, "hi": 0.0}, "wallClockMin": {"lo": 0, "hi": 0},
                "jobs": 0, "useSpot": use_spot, "perModel": []}

    per_model: list[dict[str, Any]] = []
    total_cost = 0.0
    max_minutes = 0.0
    for e in entries:
        params_b = float(e.get("paramsB", 1.0))
        epochs = float(e.get("epochs", 3.0))
        method = e.get("method", "lora")
        instance = e.get("instanceType", "ml.g5.2xlarge")
        base_eval = bool(e.get("baseEval", True))
        cost = estimate_job_cost_usd(
            instance_type=instance, train_rows=train_rows, eval_rows=eval_rows,
            epochs=epochs, params_b=params_b, method=method, base_eval=base_eval,
            use_spot=use_spot, seq_tokens=seq_tokens,
        )
        minutes = estimate_job_minutes(
            train_rows=train_rows, eval_rows=eval_rows, epochs=epochs,
            params_b=params_b, method=method, base_eval=base_eval, seq_tokens=seq_tokens,
        )
        total_cost += cost
        max_minutes = max(max_minutes, minutes)
        per_model.append({
            "instanceType": instance,
            "costUsd": _round_money(cost),
            "minutes": round(minutes),
        })

    return {
        "totalUsd": {
            "lo": _round_money(total_cost * (1 - _RANGE_FRAC)),
            "hi": _round_money(total_cost * (1 + _RANGE_FRAC)),
        },
        "wallClockMin": {
            "lo": round(max_minutes * (1 - _RANGE_FRAC)),
            "hi": round(max_minutes * (1 + _RANGE_FRAC)),
        },
        # jobs = up to 3 per model (train + eval + base-eval). Surfaced so the UI
        # can explain why the cost is more than "one job per model".
        "jobs": sum((2 + (1 if e.get("baseEval", True) else 0)) for e in entries),
        "useSpot": use_spot,
        "perModel": per_model,
        "disclaimer": (
            "Rough estimate — actual cost depends on data length, GPU availability, "
            "and early stopping (which often finishes sooner). Spot pricing and runtimes vary."
        ),
    }
