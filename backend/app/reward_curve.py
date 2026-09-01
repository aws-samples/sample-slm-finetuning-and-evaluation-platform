# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""RLVR reward curve: the GRPO reward trajectory over training.

A serverless RLVR job writes a `metrics.jsonl` (VERL trainer output) to its
output prefix — one row per training step with the reward stats. We surface the
reward-over-steps curve so a user can SEE the model improving (analogous to the
loss curve for SFT/DPO). Keys observed on a real completed RLVR job:
    step / training/global_step      -> x axis
    critic/rewards/mean|max|min      -> training reward per step
    val-core/customized/reward/mean@1 -> held-out reward at eval steps

The parse is a pure function over the jsonl text (unit-testable); fetching the
file from S3 is a thin separate step.
"""
from __future__ import annotations

import json
from typing import Any

# Candidate keys (first present wins) — VERL/recipe versions vary the exact name.
_STEP_KEYS = ("step", "training/global_step", "global_step")
_REWARD_MEAN_KEYS = ("critic/rewards/mean", "critic/score/mean", "reward/mean")
_REWARD_MAX_KEYS = ("critic/rewards/max", "critic/score/max")
_REWARD_MIN_KEYS = ("critic/rewards/min", "critic/score/min")
_VAL_REWARD_KEYS = ("val-core/customized/reward/mean@1", "val/reward/mean", "val-core/reward/mean@1")


def _pick(row: dict, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k])
    return None


def parse_reward_curve(text: str) -> dict[str, Any]:
    """Parse a metrics.jsonl into reward curve series. Returns:
        {"steps": [...], "rewardMean": [...], "rewardMax": [...],
         "rewardMin": [...], "valReward": [{"step","value"}...]}
    Skips malformed lines + rows with no resolvable step. valReward is sparse
    (only the steps where a held-out reward was logged)."""
    steps: list[int] = []
    mean: list[float | None] = []
    rmax: list[float | None] = []
    rmin: list[float | None] = []
    val: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        step = _pick(row, _STEP_KEYS)
        if step is None:
            continue
        s = int(step)
        steps.append(s)
        mean.append(_pick(row, _REWARD_MEAN_KEYS))
        rmax.append(_pick(row, _REWARD_MAX_KEYS))
        rmin.append(_pick(row, _REWARD_MIN_KEYS))
        v = _pick(row, _VAL_REWARD_KEYS)
        if v is not None:
            val.append({"step": s, "value": v})
    return {
        "steps": steps,
        "rewardMean": mean,
        "rewardMax": rmax,
        "rewardMin": rmin,
        "valReward": val,
        "hasData": bool(steps),
    }


def fetch_reward_curve(train_job: str) -> dict[str, Any]:
    """Fetch + parse a serverless RLVR job's metrics.jsonl from its output prefix.
    Returns the parsed curve (empty/hasData=False if the file isn't there yet —
    not an error, so the UI shows a 'no data yet' state while the job warms up)."""
    from .aws_config import load_aws_config
    from .orchestrate import _session

    cfg = load_aws_config()
    sm_sess, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    try:
        d = sm.describe_training_job(TrainingJobName=train_job)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"could not describe {train_job}: {e}") from e
    art = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not art:
        return parse_reward_curve("")  # no artifact yet → empty curve
    # metrics.jsonl sits at the output/model root of the (uncompressed) prefix.
    base = art.rstrip("/")
    key_uri = f"{base}/metrics.jsonl"
    bucket, _, key = key_uri[len("s3://"):].partition("/")
    s3 = boto_sess.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        text = obj["Body"].read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — not written yet (job still warming up)
        return parse_reward_curve("")
    return parse_reward_curve(text)
