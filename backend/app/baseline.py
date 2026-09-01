# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Sonnet 4.5 baseline via Bedrock — the reference row on the leaderboard.

Runs the SAME held-out eval rows through Sonnet 4.5 with deterministic decoding,
computes the SAME deterministic metrics as the SLM eval, and reports ACTUAL API
cost (api_cost_per_1k), so the leaderboard can compare a fine-tuned SLM's
projected self-host cost against Sonnet's real API cost at comparable quality.

Runs locally in the backend (a few API calls) — NOT a SageMaker job. Reuses the
metric functions baked into the container's eval.py by reimplementing the same
small set here (kept identical on purpose; deterministic + dependency-free).
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .aws_config import load_aws_config
from .orchestrate import _session
from .storage import RUNS, split_dir
from .store import get_store

# Baseline model registry — the frontier Claude models a user can score the
# held-out eval set against, as reference rows on the leaderboard. Each carries
# its cross-region inference-profile id (verified available in-account) + ACTUAL
# list price (USD per 1K tokens) so the leaderboard compares a fine-tuned SLM's
# projected self-host cost against each frontier model's real API cost.
# id → {provider, label, modelId, inPer1k, outPer1k}.
# ALL invoked via Bedrock's Converse API — which is multi-provider, so adding a
# model from any Bedrock provider is just a dict entry (no new code). The UI
# groups the picker by `provider`. Prices are Bedrock on-demand $/1K tokens
# (us-east-1 list) and drift — treat as estimates for the cost column. Each model
# needs per-account Bedrock model access enabled; the runner surfaces an
# access-denied hint pointing at the Bedrock console when it isn't.
BASELINE_MODELS: dict[str, dict[str, Any]] = {
    # --- Anthropic ---
    "haiku-4-5": {
        "provider": "Anthropic", "label": "Claude Haiku 4.5",
        "modelId": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "inPer1k": 0.001, "outPer1k": 0.005,
    },
    "sonnet-4-5": {
        "provider": "Anthropic", "label": "Claude Sonnet 4.5",
        "modelId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "inPer1k": 0.003, "outPer1k": 0.015,
    },
    "sonnet-4-6": {
        "provider": "Anthropic", "label": "Claude Sonnet 4.6",
        "modelId": "us.anthropic.claude-sonnet-4-6",
        "inPer1k": 0.003, "outPer1k": 0.015,
    },
    "opus-4-8": {
        "provider": "Anthropic", "label": "Claude Opus 4.8",
        "modelId": "us.anthropic.claude-opus-4-8",
        "inPer1k": 0.015, "outPer1k": 0.075,
    },
    # --- Amazon Nova (low-cost frontier — a strong "buy vs fine-tune" anchor) ---
    "nova-micro": {
        "provider": "Amazon", "label": "Amazon Nova Micro",
        "modelId": "us.amazon.nova-micro-v1:0",
        "inPer1k": 0.000035, "outPer1k": 0.00014,
    },
    "nova-lite": {
        "provider": "Amazon", "label": "Amazon Nova Lite",
        "modelId": "us.amazon.nova-lite-v1:0",
        "inPer1k": 0.00006, "outPer1k": 0.00024,
    },
    "nova-pro": {
        "provider": "Amazon", "label": "Amazon Nova Pro",
        "modelId": "us.amazon.nova-pro-v1:0",
        "inPer1k": 0.0008, "outPer1k": 0.0032,
    },
    # --- Meta Llama (Bedrock inference profiles → us. prefix) ---
    "llama-3-3-70b": {
        "provider": "Meta", "label": "Llama 3.3 70B Instruct",
        "modelId": "us.meta.llama3-3-70b-instruct-v1:0",
        "inPer1k": 0.00072, "outPer1k": 0.00072,
    },
    "llama-3-1-8b": {
        "provider": "Meta", "label": "Llama 3.1 8B Instruct",
        "modelId": "us.meta.llama3-1-8b-instruct-v1:0",
        "inPer1k": 0.00022, "outPer1k": 0.00022,
    },
    "llama-3-2-3b": {
        "provider": "Meta", "label": "Llama 3.2 3B Instruct",
        "modelId": "us.meta.llama3-2-3b-instruct-v1:0",
        "inPer1k": 0.00015, "outPer1k": 0.00015,
    },
    # --- Mistral ---
    "mistral-large-2": {
        "provider": "Mistral", "label": "Mistral Large 2",
        "modelId": "mistral.mistral-large-2402-v1:0",
        "inPer1k": 0.004, "outPer1k": 0.012,
    },
    "mistral-small": {
        "provider": "Mistral", "label": "Mistral Small",
        "modelId": "mistral.mistral-small-2402-v1:0",
        "inPer1k": 0.001, "outPer1k": 0.003,
    },
    # --- Cohere ---
    "command-r-plus": {
        "provider": "Cohere", "label": "Command R+",
        "modelId": "cohere.command-r-plus-v1:0",
        "inPer1k": 0.0025, "outPer1k": 0.01,
    },
    "command-r": {
        "provider": "Cohere", "label": "Command R",
        "modelId": "cohere.command-r-v1:0",
        "inPer1k": 0.00015, "outPer1k": 0.0006,
    },
}
DEFAULT_BASELINE = "sonnet-4-5"


def baseline_model(key: str | None) -> dict[str, Any]:
    """Resolve a baseline model spec by key, defaulting to Sonnet 4.5."""
    return BASELINE_MODELS.get(key or DEFAULT_BASELINE, BASELINE_MODELS[DEFAULT_BASELINE])


# Back-compat: judge.py + selfheal.py import these Sonnet-4.5 constants.
SONNET_MODEL_ID = BASELINE_MODELS["sonnet-4-5"]["modelId"]
SONNET_INPUT_PER_1K = BASELINE_MODELS["sonnet-4-5"]["inPer1k"]
SONNET_OUTPUT_PER_1K = BASELINE_MODELS["sonnet-4-5"]["outPer1k"]


# --- metrics (identical to container/eval.py; kept in sync deliberately) ---

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _rouge_l(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    lcs = _lcs_len(p, g)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def _char_f1(pred: str, gold: str) -> float:
    p, g = Counter(_normalize(pred)), Counter(_normalize(gold))
    if not p and not g:
        return 1.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def _contains_gold(pred: str, gold: str) -> float:
    np_, ng = _normalize(pred), _normalize(gold)
    return 1.0 if ng and ng in np_ else 0.0


def _json_structural(pred: str, gold: str) -> float | None:
    try:
        gold_obj = json.loads(gold)
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        pred_obj = json.loads(pred)
    except (json.JSONDecodeError, ValueError):
        return 0.0

    def shape(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: shape(v) for k, v in sorted(o.items())}
        if isinstance(o, list):
            return ["list"]
        return type(o).__name__

    return 1.0 if shape(pred_obj) == shape(gold_obj) else 0.0


def _compute_metrics(preds: list[str], golds: list[str]) -> dict[str, Any]:
    n = len(golds)
    exact = norm = contains = 0
    f1_sum = rouge_sum = charf1_sum = lenratio_sum = 0.0
    json_scores: list[float] = []
    per_total: dict[str, int] = defaultdict(int)
    per_correct: dict[str, int] = defaultdict(int)
    for pred, gold in zip(preds, golds):
        exact += int(pred.strip() == gold.strip())
        nm = _normalize(pred) == _normalize(gold)
        norm += int(nm)
        f1_sum += _token_f1(pred, gold)
        rouge_sum += _rouge_l(pred, gold)
        charf1_sum += _char_f1(pred, gold)
        contains += int(_contains_gold(pred, gold) > 0)
        gl = max(1, len(_normalize(gold).split()))
        lenratio_sum += len(_normalize(pred).split()) / gl
        js = _json_structural(pred, gold)
        if js is not None:
            json_scores.append(js)
        cls = _normalize(gold)
        per_total[cls] += 1
        per_correct[cls] += int(nm)

    def avg(x: float) -> float:
        return round(x / n, 4) if n else 0.0

    return {
        "count": n,
        "exact_match": avg(exact),
        "normalized_match": avg(norm),
        "contains_gold": avg(contains),
        "token_f1": avg(f1_sum),
        "rouge_l": avg(rouge_sum),
        "char_f1": avg(charf1_sum),
        "length_ratio": avg(lenratio_sum),
        "json_structural": round(sum(json_scores) / len(json_scores), 4) if json_scores else None,
        "per_class_accuracy": {
            c: {"total": per_total[c], "correct": per_correct[c],
                "accuracy": round(per_correct[c] / per_total[c], 4)}
            for c in sorted(per_total)
        },
    }


def _load_eval_rows(split_id: str) -> tuple[list[list[dict]], list[str]]:
    run_dir = split_dir(split_id)
    if run_dir is None:
        raise ValueError(f"split {split_id} not found")
    prompts: list[list[dict]] = []
    golds: list[str] = []
    for line in (run_dir / "eval.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        prompts.append(msgs[:last])
        golds.append(msgs[last]["content"])
    return prompts, golds


def _to_converse_messages(turns: list[dict]) -> tuple[str | None, list[dict]]:
    """Split chat turns into (system_prompt, messages) for the Bedrock CONVERSE
    API. Converse content blocks are {"text": ...} (no "type" key) and the system
    prompt is passed separately. Model-agnostic — same shape for Claude + Nova."""
    system = None
    messages = []
    for t in turns:
        if t["role"] == "system":
            system = t["content"]
        else:
            messages.append({"role": t["role"], "content": [{"text": t["content"]}]})
    return system, messages


def run_sonnet_baseline(
    split_id: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    baseline_key: str | None = None,
) -> dict[str, Any]:
    """Score a baseline Claude model on the held-out eval rows; return metrics +
    actual cost. `baseline_key` selects the model (default Sonnet 4.5). The
    function name is kept for back-compat (callers/tests import it)."""
    spec = baseline_model(baseline_key)
    model_id = spec["modelId"]
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    client = boto_sess.client("bedrock-runtime", region_name=cfg.region)

    prompts, golds = _load_eval_rows(split_id)
    preds: list[str] = []
    in_tokens = out_tokens = 0

    for turns in prompts:
        system, messages = _to_converse_messages(turns)
        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_new_tokens, "temperature": temperature},
        }
        if system:
            kwargs["system"] = [{"text": system}]
        # Converse API is model-AGNOSTIC — one code path works for Anthropic Claude
        # AND Amazon Nova (and others), with uniform message + usage shapes. (The
        # old invoke_model used the Anthropic-only schema, which Nova rejects.)
        try:
            resp = client.converse(**kwargs)
        except client.exceptions.AccessDeniedException as e:
            # Bedrock model access is per-account opt-in — translate the raw 403
            # into an actionable hint instead of a cryptic stack trace.
            raise RuntimeError(
                f"Bedrock access denied for {model_id}. Enable access to this model "
                f"in the Bedrock console (Model access) for region {cfg.region}, then retry."
            ) from e
        out_msg = resp.get("output", {}).get("message", {})
        text = "".join(b.get("text", "") for b in out_msg.get("content", []))
        preds.append(text.strip())
        usage = resp.get("usage", {})
        in_tokens += usage.get("inputTokens", 0)
        out_tokens += usage.get("outputTokens", 0)

    metrics = _compute_metrics(preds, golds)
    api_cost = round(
        in_tokens / 1000 * spec["inPer1k"] + out_tokens / 1000 * spec["outPer1k"], 6
    )
    cost_per_1k_rows = round(api_cost / len(golds) * 1000, 4) if golds else None
    metrics["baseline"] = {
        "key": baseline_key or DEFAULT_BASELINE,
        "label": spec["label"],
        "model": model_id,
        "inputTokens": in_tokens,
        "outputTokens": out_tokens,
        "apiCostUsd": api_cost,
        "apiCostPer1kRows": cost_per_1k_rows,
    }
    _save_baseline(split_id, metrics, baseline_key or DEFAULT_BASELINE)
    return metrics


# Sonnet 4.5 keeps the legacy filename so already-cached baselines still load;
# every other model gets baseline_<key>.json. Same for the status marker.
def _baseline_file(key: str) -> str:
    return "sonnet_baseline.json" if key == DEFAULT_BASELINE else f"baseline_{key}.json"


def _status_file(key: str) -> str:
    return (
        "sonnet_baseline_status.json"
        if key == DEFAULT_BASELINE
        else f"baseline_{key}_status.json"
    )


def _save_baseline(split_id: str, metrics: dict[str, Any], key: str = DEFAULT_BASELINE) -> None:
    store = get_store()
    if not store.file_exists(RUNS, split_id, "dataset_info.json"):
        return  # no such split; nothing to attach the baseline to
    wd = store.workdir(RUNS, split_id)
    (wd / _baseline_file(key)).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    store.commit(RUNS, split_id)


def load_baseline(split_id: str, key: str = DEFAULT_BASELINE) -> dict[str, Any] | None:
    """Return a previously-computed baseline for this split + model, if cached."""
    raw = get_store().read_file(RUNS, split_id, _baseline_file(key))
    return json.loads(raw) if raw else None


def load_all_baselines(split_id: str) -> list[dict[str, Any]]:
    """Every cached baseline for this split (one per model that's been run)."""
    out: list[dict[str, Any]] = []
    for key in BASELINE_MODELS:
        m = load_baseline(split_id, key)
        if m is not None:
            out.append(m)
    return out


# --- async status tracking (the baseline can exceed API Gateway's 29s limit on
# large eval sets, so it runs in a background worker; the UI polls this) -------

def set_baseline_status(
    split_id: str, status: str, detail: str = "", key: str = DEFAULT_BASELINE
) -> None:
    """Persist a baseline run status: 'running' | 'done' | 'failed'."""
    store = get_store()
    if not store.file_exists(RUNS, split_id, "dataset_info.json"):
        return
    wd = store.workdir(RUNS, split_id)
    (wd / _status_file(key)).write_text(
        json.dumps({"status": status, "detail": detail}), encoding="utf-8"
    )
    store.commit(RUNS, split_id)


def baseline_status(split_id: str, key: str = DEFAULT_BASELINE) -> dict[str, Any]:
    """Return {status, detail}. 'done' if a baseline result exists; else the
    last recorded status, or 'none' if never run."""
    if load_baseline(split_id, key) is not None:
        return {"status": "done", "detail": ""}
    raw = get_store().read_file(RUNS, split_id, _status_file(key))
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"status": "none", "detail": ""}


def run_baseline_task(
    split_id: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    baseline_key: str | None = None,
) -> None:
    """Worker entrypoint: run a baseline model, tracking status for the poller.

    Wraps run_sonnet_baseline (which persists the result on success) with
    running/failed status markers so the UI can poll without holding an HTTP
    connection open past the API Gateway timeout.
    """
    key = baseline_key or DEFAULT_BASELINE
    set_baseline_status(split_id, "running", key=key)
    try:
        run_sonnet_baseline(split_id, max_new_tokens=max_new_tokens,
                            temperature=temperature, baseline_key=key)
        set_baseline_status(split_id, "done", key=key)
    except Exception as e:  # noqa: BLE001 — record failure for the poller
        set_baseline_status(split_id, "failed", str(e), key=key)
        raise
