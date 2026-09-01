# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""RLVR custom-reward Lambda handler.

SageMaker's serverless RLVR loop invokes this Lambda once per rollout batch to
score each model response against its verifiable target. Contract (authoritative,
from model-customize-evaluation-preset-custom-scorers.html):

  input : {"batch": [{"id": str, "messages": [...], "reference_answer": {...}}]}
          - `messages` is the rollout conversation (the model's response is the
            last assistant turn).
          - `reference_answer` is the dataset row's reward_model.ground_truth
            (our VERL ground_truth arrives here).
  output: {"statusCode": 200, "body": json([
             {"id": str, "aggregate_reward_score": float,
              "metrics_list": [{"name": str, "value": float, "type": "reward"}]}
          ])}

The user's reward logic is in user_reward.py as `reward(response, ground_truth)
-> float` (generated from the snippet authored in the UI). `scoring` (eval.py's
extraction + metrics) is importable by that snippet. Every per-row call is wrapped
in try/except → a bad/raising reward scores 0.0 and is logged, so one bad row can
never crash the GRPO loop (which would fail the whole — billable — training job).
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# The user's reward(response, ground_truth) -> float. Imported at module load so
# an import-time error in the snippet surfaces immediately (fail fast on deploy),
# not silently per-row.
from user_reward import reward  # noqa: E402


def _response_text(messages) -> str:
    """The model's response = the content of the last assistant turn (VERL puts
    the rollout completion there). Falls back to the last turn's content."""
    if not isinstance(messages, list) or not messages:
        return ""
    for turn in reversed(messages):
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            return str(turn.get("content", "") or "")
    last = messages[-1]
    return str(last.get("content", "") or "") if isinstance(last, dict) else str(last)


def _ground_truth(reference_answer) -> str:
    """Our VERL ground_truth comes through `reference_answer`. It's usually a bare
    string, but tolerate {"ground_truth": ...} / {"answer": ...} wrappers."""
    if isinstance(reference_answer, dict):
        for k in ("ground_truth", "answer", "value", "target"):
            if k in reference_answer:
                return str(reference_answer[k] or "")
        return json.dumps(reference_answer)
    return str(reference_answer if reference_answer is not None else "")


def _score_one(item: dict) -> dict:
    rid = item.get("id")
    response = _response_text(item.get("messages"))
    gt = _ground_truth(item.get("reference_answer"))
    try:
        val = float(reward(response, gt))
        if val != val or val in (float("inf"), float("-inf")):  # NaN/inf guard
            val = 0.0
    except Exception as e:  # noqa: BLE001 — a bad row scores 0, never kills the run
        logger.warning("reward() raised for id=%s: %s", rid, e)
        val = 0.0
    val = max(0.0, min(1.0, val))
    return {
        "id": rid,
        "aggregate_reward_score": val,
        "metrics_list": [{"name": "custom_reward", "value": val, "type": "reward"}],
    }


def _extract_batch(event):
    """Normalise the envelope shapes the RLVR loop may send into a flat list of
    per-row score items.

    Observed on a real GRPO job (qwen3-1.7b): the loop invokes the Lambda with the
    batch as a BARE JSON LIST (event == [{id, messages, reference_answer}, …]), NOT
    the {"batch": [...]} envelope the preset-scorer docs describe. The deploy-time
    validation invoke used the {"batch": …} shape, so this mismatch only surfaced
    once GRPO actually called it — and it crashed every reward call with
    'list' object has no attribute 'get'. Accept both (plus a few common wrappers)
    so an envelope quirk can never crash the — billable — training loop."""
    if event is None:
        return []
    if isinstance(event, list):
        return event
    if isinstance(event, dict):
        for key in ("batch", "records", "instances", "inputs", "items"):
            v = event.get(key)
            if isinstance(v, list):
                return v
        # A single un-wrapped row → score it alone.
        if "messages" in event or "id" in event:
            return [event]
    return []


def handler(event, context):  # noqa: ANN001 — Lambda signature
    batch = _extract_batch(event)
    results = [_score_one(it) for it in batch if isinstance(it, dict)]
    return {"statusCode": 200, "body": json.dumps(results)}
