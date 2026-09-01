# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LLM-as-judge evaluation — quality scoring beyond string-overlap metrics.

The deterministic metrics (exact/F1/ROUGE/contains-gold) reward surface overlap;
they undercount paraphrased-but-correct answers. This judges each eval
prediction with a strong model (Sonnet, via the same Bedrock path baseline.py
uses): a 1–5 quality score + short reason, scored against the gold answer.

Runs as a WORKER pass over an eval job's saved predictions.jsonl (no SageMaker
image rebuild). Results persist per EVAL JOB (the stable, reusable key) under the
"judge" store collection, so a model judged once stays judged. The race winner
is judged automatically; other entries are judged on demand.
"""
from __future__ import annotations

import json
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .aws_config import load_aws_config
from .baseline import BASELINE_MODELS, baseline_model
from .obs import log_event
from .orchestrate import _session
from .store import get_store

JUDGE = "judge"  # store collection, keyed by eval job name
_JUDGE_FILE = "judge.json"
_STATUS_FILE = "judge_status.json"

# Which model judges, by default. Any BASELINE_MODELS key works now that the
# judge calls Converse (Claude OR Nova) — Sonnet 4.5 stays the default for
# back-compatible scores. Override per-run via run_judge(judge_key=...).
DEFAULT_JUDGE_KEY = "sonnet-4-5"

# Task-aware multi-dimension rubric. The judge sees the ORIGINAL PROMPT (so it can
# check faithfulness/grounding — the thing lexical metrics + gold-matching can't),
# the gold answer, and the model's answer. It scores several 1-5 dimensions; which
# dimensions apply is chosen per task type. `overall` is the holistic score (also
# the headline number, back-compatible with the old single-score UI).
#
# Dimensions:
#   correctness  — is the answer factually right vs the gold / the prompt's data?
#   faithfulness — every claim grounded in the prompt; NO invented facts (hallucination)
#   format       — obeys required output format (JSON shape, label set, structure)
#   completeness — covers what was asked; nothing important missing
#   conciseness  — no padding/rambling; within any length expectation
_DIMENSIONS_BY_TASK: dict[str, list[str]] = {
    "json": ["correctness", "faithfulness", "format", "completeness", "conciseness"],
    "label": ["correctness", "format"],
    "numeric": ["correctness"],
    "text": ["correctness", "faithfulness", "completeness", "conciseness"],
}
_ALL_DIMENSIONS = ["correctness", "faithfulness", "format", "completeness", "conciseness"]

_JUDGE_SYSTEM = (
    "You are a strict, fair evaluation judge for fine-tuned language-model outputs. "
    "You are given the ORIGINAL PROMPT the model received, a reference (gold) answer, "
    "and the MODEL ANSWER. Rate the model answer on each requested dimension from 1 "
    "(very poor) to 5 (excellent). Judge MEANING, not exact wording — a correct "
    "paraphrase scores high. Be especially strict on FAITHFULNESS: if the model "
    "states any fact (name, number, price, entity) that is NOT supported by the "
    "prompt, that is a hallucination and faithfulness must score low, even if it "
    "reads well. FORMAT scores whether the answer obeys the format the prompt asked "
    "for (e.g. valid JSON with the required keys, or exactly one allowed label). "
    "Reply with ONLY a JSON object mapping each requested dimension to an integer "
    '1-5, plus an integer "overall" and a one-sentence "reason". Example: '
    '{"correctness":5,"faithfulness":4,"format":5,"overall":4,"reason":"..."}.'
)


def _detect_task(gold: str) -> str:
    """Lightweight task type from the gold answer (mirrors eval.detect_task):
    json | numeric | label | text. Chooses the rubric dimensions."""
    g = (gold or "").strip()
    if g[:1] in "{[":
        try:
            json.loads(g)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass
    if re.match(r"^[+-]?\$?\s*\d[\d,]*\.?\d*\s*%?$", g):
        return "numeric"
    if len(g.split()) <= 4 and not re.search(r"[.!?]", g):
        return "label"
    return "text"


def _judge_one(
    client, prompt: str, gold: str, pred: str, model_id: str
) -> tuple[dict[str, int], int, str, int, int]:
    """Return (per_dimension scores, overall 1-5, reason, in_tokens, out_tokens)."""
    task = _detect_task(gold)
    dims = _DIMENSIONS_BY_TASK.get(task, _DIMENSIONS_BY_TASK["text"])
    user = (
        f"ORIGINAL PROMPT (the input the model had to answer):\n{prompt}\n\n"
        f"GOLD (reference answer):\n{gold}\n\n"
        f"MODEL ANSWER:\n{pred}\n\n"
        f"Score these dimensions: {', '.join(dims)}. "
        "Return the JSON object now."
    )
    # Converse API — model-AGNOSTIC: this same call works for Anthropic Claude
    # AND Amazon Nova (and others), with uniform system/message + usage shapes.
    # (The old invoke_model used the Anthropic-only schema, which Nova rejects,
    # so the judge was Claude-locked.) Mirrors baseline.py's Converse path.
    resp = client.converse(
        modelId=model_id,
        system=[{"text": _JUDGE_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0.0},
    )
    out_msg = resp.get("output", {}).get("message", {})
    text = "".join(b.get("text", "") for b in out_msg.get("content", []))
    usage = resp.get("usage", {})
    per_dim, overall, reason = _parse_judge(text, dims)
    return per_dim, overall, reason, usage.get("inputTokens", 0), usage.get("outputTokens", 0)


def _parse_judge(text: str, dims: list[str]) -> tuple[dict[str, int], int, str]:
    """Pull the per-dimension scores + overall + reason out of the judge reply.
    Tolerant of stray prose. Falls back to a single digit when JSON is malformed."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))

            def clamp(v: Any) -> int | None:
                try:
                    return max(1, min(5, int(v)))
                except (ValueError, TypeError):
                    return None

            per_dim = {d: clamp(obj.get(d)) for d in dims if clamp(obj.get(d)) is not None}
            overall = clamp(obj.get("overall"))
            if overall is None and per_dim:
                overall = round(sum(per_dim.values()) / len(per_dim))
            reason = str(obj.get("reason", ""))[:300]
            if overall is not None:
                return per_dim, overall, reason
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    d = re.search(r"[1-5]", text)
    sc = int(d.group(0)) if d else 0
    return {}, sc, text.strip()[:300]


def _load_predictions(eval_job_name: str) -> list[dict[str, str]]:
    """Download predictions.jsonl from a completed eval job's model.tar.gz."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    d = sm.describe_training_job(TrainingJobName=eval_job_name)
    if d["TrainingJobStatus"] != "Completed":
        raise ValueError(f"eval job {eval_job_name} is not Completed")
    artifact = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not artifact:
        raise ValueError(f"eval job {eval_job_name} has no artifact")

    s3 = boto_sess.client("s3")
    _, _, rest = artifact.partition("s3://")
    bucket, _, key = rest.partition("/")
    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "model.tar.gz"
        s3.download_file(bucket, key, str(local))
        with tarfile.open(local) as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith("predictions.jsonl")), None)
            if member is None:
                raise ValueError("predictions.jsonl not found in eval artifact")
            f = tar.extractfile(member)
            for line in f.read().decode("utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


# --- persistence (keyed by eval job) -----------------------------------------

def set_judge_status(eval_job: str, status: str, detail: str = "") -> None:
    store = get_store()
    wd = store.workdir(JUDGE, eval_job)
    (wd / _STATUS_FILE).write_text(json.dumps({"status": status, "detail": detail}), encoding="utf-8")
    store.commit(JUDGE, eval_job)


def load_judge(eval_job: str) -> dict[str, Any] | None:
    raw = get_store().read_file(JUDGE, eval_job, _JUDGE_FILE)
    return json.loads(raw) if raw else None


def judge_status(eval_job: str) -> dict[str, Any]:
    if load_judge(eval_job) is not None:
        return {"status": "done", "detail": ""}
    raw = get_store().read_file(JUDGE, eval_job, _STATUS_FILE)
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"status": "none", "detail": ""}


def _save_judge(eval_job: str, result: dict[str, Any]) -> None:
    store = get_store()
    wd = store.workdir(JUDGE, eval_job)
    (wd / _JUDGE_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    store.commit(JUDGE, eval_job)


def run_judge(eval_job: str, judge_key: str | None = None) -> dict[str, Any]:
    """Judge every prediction of a completed eval job; persist + return the result.

    judge_key selects the judge model (any BASELINE_MODELS key — Claude or Nova,
    via Converse); defaults to Sonnet 4.5. Result shape: {judgeScore (mean 1-5),
    judgedRows, distribution{1..5}, apiCostUsd, model, judgeKey,
    samples[{gold,pred,score,reason}], status}.
    """
    # No per-run judge picked → use the Settings default for the 'judge' role
    # (falls back to DEFAULT_JUDGE_KEY when unset). A per-run judge_key still wins.
    from .agent_models import resolve_baseline_key

    effective_key = judge_key or resolve_baseline_key("judge") or DEFAULT_JUDGE_KEY
    spec = baseline_model(effective_key)
    model_id = spec["modelId"]
    in_per_1k, out_per_1k = spec["inPer1k"], spec["outPer1k"]

    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    client = boto_sess.client("bedrock-runtime", region_name=cfg.region)

    rows = _load_predictions(eval_job)
    # Empty eval → don't emit a misleading judgeScore=0.0 over 0 rows. Surface an
    # explicit status the UI can show instead (an empty held-out set means the
    # split produced no scoreable eval — a data problem, not a judge verdict).
    if not rows:
        result = {
            "judgeScore": None, "dimensions": {}, "judgedRows": 0,
            "distribution": {str(i): 0 for i in range(1, 6)},
            "model": model_id, "judgeKey": effective_key,
            "label": spec["label"], "apiCostUsd": 0.0, "samples": [],
            "status": "no_eval_rows",
            "statusDetail": "the eval set has no rows to judge (empty held-out split)",
        }
        _save_judge(eval_job, result)
        return result
    scores: list[int] = []  # overall scores
    dim_sums: dict[str, float] = {}
    dim_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    in_tok = out_tok = 0
    for r in rows:
        gold = r.get("gold", "")
        # Judge the EXTRACTED answer when present (reasoning scaffolding removed),
        # else the raw prediction (older eval jobs predate the 'extracted' field).
        pred = r.get("extracted") or r.get("prediction", "")
        prompt = r.get("prompt", "")  # absent on pre-rubric eval jobs → judge degrades
        per_dim, overall, reason, it, ot = _judge_one(client, prompt, gold, pred, model_id)
        scores.append(overall)
        for d, v in per_dim.items():
            dim_sums[d] = dim_sums.get(d, 0.0) + v
            dim_counts[d] = dim_counts.get(d, 0) + 1
        in_tok += it
        out_tok += ot
        if len(samples) < 10:  # keep a few examples for the UI
            samples.append({"gold": gold, "pred": pred, "score": overall,
                            "dimensions": per_dim, "reason": reason})

    n = len(scores) or 1
    dist = {str(i): scores.count(i) for i in range(1, 6)}
    # Mean per-dimension score across rows where that dimension applied.
    dimensions = {
        d: round(dim_sums[d] / dim_counts[d], 3) for d in sorted(dim_sums) if dim_counts[d]
    }
    api_cost = round(in_tok / 1000 * in_per_1k + out_tok / 1000 * out_per_1k, 6)
    result = {
        "judgeScore": round(sum(scores) / n, 3),  # overall mean (back-compatible headline)
        "dimensions": dimensions,                  # per-dimension means (faithfulness/format/…)
        "judgedRows": len(scores),
        "distribution": dist,
        "model": model_id,
        "judgeKey": effective_key,
        "label": spec["label"],
        "apiCostUsd": api_cost,
        "samples": samples,
    }
    _save_judge(eval_job, result)
    return result


def run_judge_task(eval_job: str, judge_key: str | None = None) -> None:
    """Worker entrypoint: judge with running/failed status tracking for polling."""
    set_judge_status(eval_job, "running")
    log_event("judge.start", evalJob=eval_job, judgeKey=judge_key or DEFAULT_JUDGE_KEY)
    try:
        res = run_judge(eval_job, judge_key=judge_key)
        set_judge_status(eval_job, "done")
        log_event("judge.done", evalJob=eval_job, judgeScore=res["judgeScore"], rows=res["judgedRows"])
    except Exception as e:  # noqa: BLE001
        set_judge_status(eval_job, "failed", str(e))
        log_event("judge.error", level="ERROR", evalJob=eval_job, error=str(e))
        raise
