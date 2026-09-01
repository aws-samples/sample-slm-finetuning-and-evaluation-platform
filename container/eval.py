# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Offline batch evaluation for the SLM platform.

Loads a merged model, generates over the held-out eval prompts with DETERMINISTIC
decoding (temp=0, fixed seed), and computes a panel of deterministic metrics so a
user can compare candidates on whichever metric matters. Writes metrics.json +
predictions.jsonl to the SageMaker output dir.

Methodology rules honored:
  - same eval rows + same decoding params across all candidates (params come from
    the rendered eval config, identical for every model in a comparison).
  - eval set is the held-out split (already asserted disjoint from train).
  - deterministic metrics only in v1 (exact / normalized / token-F1 /
    JSON-structural / per-class). LLM-as-judge deferred.

Inference backend is swappable behind `generate_batch()`. Default: vLLM. Falls
back to HF transformers .generate() if vLLM is unavailable (env SLM_EVAL_BACKEND=hf).
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# --- IO paths (SageMaker conventions; overridable via env for local test) ---
MODEL_DIR = os.environ.get("SLM_EVAL_MODEL_DIR", "/opt/ml/input/data/model")
DATASET_DIR = os.environ.get("SLM_EVAL_DATASET_DIR", "/opt/ml/input/data/dataset")
OUTPUT_DIR = os.environ.get("SLM_EVAL_OUTPUT_DIR", "/opt/ml/model")
EVAL_FILE = os.environ.get("SLM_EVAL_FILE", "eval.jsonl")

# --- decoding defaults (deterministic); overridable via env from the config ---
MAX_NEW_TOKENS = int(os.environ.get("SLM_EVAL_MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("SLM_EVAL_TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("SLM_EVAL_TOP_P", "1.0"))
SEED = int(os.environ.get("SLM_EVAL_SEED", "42"))


# ----------------------------- metrics ------------------------------------- #

def _normalize(s: str) -> str:
    """Lowercase, strip punctuation + articles + extra whitespace (SQuAD-style)."""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- answer extraction ----------------------------------------------------- #
# Modern instruct/reasoning models wrap the real answer in scaffolding the task
# never asked for: a <think>...</think> chain-of-thought block (Qwen3, R1, etc.),
# a gpt-oss "harmony" <|channel|>analysis…<|channel|>final<|message|> structure,
# <thinking>/<reasoning>/<scratchpad> blocks, ```json fences, or a sentence of
# preamble. Scoring the RAW string then unfairly zeroes a perfect answer — this
# was the real cause of a 0% JSON score on a run where 99% of outputs were valid
# JSON behind a <think></think> prefix. We strip that scaffolding once and score
# the EXTRACTED answer. Kept in sync with profiler.detect_scaffold (which flags
# the same families on the gold answers as an early heads-up).

# Paired reasoning blocks across the common families. Each is removed wholesale.
_REASONING_BLOCK_RES = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reason>.*?</reason>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<scratchpad>.*?</scratchpad>", re.DOTALL | re.IGNORECASE),
]
_OPEN_TAGS = ("<think>", "<thinking>", "<reasoning>", "<reason>", "<scratchpad>")
_CLOSE_TAGS = ("</think>", "</thinking>", "</reasoning>", "</reason>", "</scratchpad>")
_FENCE_RE = re.compile(r"```(?:json|python|[a-zA-Z0-9_+-]*)?\s*(.*?)```", re.DOTALL)
# gpt-oss harmony: <|channel|>final<|message|>ANSWER(<|end|>/<|return|>). The real
# answer lives in the `final` channel; everything before it is analysis/preamble.
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|channel\|>|$)",
    re.DOTALL | re.IGNORECASE,
)
# Any leftover harmony control tokens to scrub if the structure was partial.
_HARMONY_TOKEN_RE = re.compile(r"<\|/?(?:channel|message|start|end|return|constrain)\|>", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _strip_think(s: str) -> str:
    """Remove paired reasoning blocks (<think>/<thinking>/<reasoning>/… ) and a
    lone unmatched opener if the model never closed it (truncated generation)."""
    for rx in _REASONING_BLOCK_RES:
        s = rx.sub("", s)
    # Unclosed reasoning block: drop everything from the opener (keep any tail
    # after a stray closer, in case the open/close got mismatched).
    for open_t, close_t in zip(_OPEN_TAGS, _CLOSE_TAGS):
        if open_t in s:
            s = s.split(open_t)[0] + (s.split(close_t)[-1] if close_t in s else "")
    return s.strip()


def extract_answer(pred: str) -> str:
    """The model's actual answer with reasoning/markdown scaffolding removed.

    Order: 1. gpt-oss harmony `final` channel (if present, that IS the answer);
    2. strip <think>/<thinking>/<reasoning>/<scratchpad> blocks + leftover harmony
    tokens; 3. unwrap <answer>…</answer> if present; 4. if a ```code/json``` fence
    exists, take its contents; else return the de-scaffolded text. Cheap +
    deterministic; every 'extracted_*' metric runs on this so reasoning models
    (Qwen3, R1, gpt-oss, …) aren't unfairly scored."""
    s = pred
    # gpt-oss harmony: prefer the explicit `final` channel payload.
    finals = _HARMONY_FINAL_RE.findall(s)
    if finals:
        s = finals[-1]
    s = _strip_think(s)
    s = _HARMONY_TOKEN_RE.sub("", s)  # scrub any leftover harmony control tokens
    ans = _ANSWER_TAG_RE.findall(s)
    if ans:
        s = max(ans, key=len)
    fences = _FENCE_RE.findall(s)
    if fences:
        # Prefer the longest fenced block (usually the answer payload).
        s = max(fences, key=len)
    return s.strip()


def _first_json(s: str) -> Any | None:
    """Parse the first balanced {...} or [...] object found in s, else None.
    Tolerant of leading/trailing prose around an otherwise-valid JSON answer."""
    s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
    # Find the first { or [ and scan to its matching close.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(s)):
            if s[i] == open_ch:
                depth += 1
            elif s[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


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
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (for ROUGE-L)."""
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
    """ROUGE-L F-measure on normalized tokens (longest common subsequence)."""
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
    """Character-level F1 (bag of chars) — forgiving of token/format differences."""
    p, g = Counter(_normalize(pred)), Counter(_normalize(gold))
    if not p and not g:
        return 1.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def _contains_gold(pred: str, gold: str) -> float:
    """1.0 if the (normalized) gold answer appears within the (normalized) pred.

    Useful when models emit the right answer wrapped in extra prose — credits a
    correct answer that exact/normalized match would score 0.
    """
    np_, ng = _normalize(pred), _normalize(gold)
    if not ng:
        return 0.0
    return 1.0 if ng in np_ else 0.0


def _shape(o: Any) -> Any:
    """Structural fingerprint of a JSON value (keys + value types, lists collapsed)."""
    if isinstance(o, dict):
        return {k: _shape(v) for k, v in sorted(o.items())}
    if isinstance(o, list):
        return ["list"]
    return type(o).__name__


def _json_structural(pred_obj: Any, gold_obj: Any) -> float:
    """1.0 if pred/gold JSON objects share the same key set + value-type shape."""
    return 1.0 if _shape(pred_obj) == _shape(gold_obj) else 0.0


def _json_key_recall(pred_obj: Any, gold_obj: Any) -> float | None:
    """Fraction of the gold dict's top-level keys present in the pred dict.
    None when gold isn't a dict (key recall undefined)."""
    if not isinstance(gold_obj, dict):
        return None
    if not isinstance(pred_obj, dict) or not gold_obj:
        return 0.0
    present = sum(1 for k in gold_obj if k in pred_obj)
    return present / len(gold_obj)


# --- task-type detection (per row, from the GOLD answer) ------------------- #
# The harness runs metrics appropriate to each row's task, auto-detected from the
# gold — so one eval works across classification, JSON-generation, numeric
# extraction, and free-form generation tasks with zero per-dataset config.

_NUM_RE = re.compile(r"^[+-]?\$?\s*\d[\d,]*\.?\d*\s*%?$")


def _as_number(s: str) -> float | None:
    """Parse a bare numeric answer (handles $, commas, %, sign), else None."""
    t = s.strip()
    if not _NUM_RE.match(t):
        return None
    t = t.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def detect_task(gold: str) -> str:
    """Classify a row by its gold answer: 'json' | 'numeric' | 'label' | 'text'.
    Drives which task-specific metrics are reported for that row."""
    if _first_json(gold) is not None and gold.strip()[:1] in "{[":
        return "json"
    if _as_number(gold) is not None:
        return "numeric"
    # A short answer (≤4 words, no sentence punctuation) reads as a class label.
    g = gold.strip()
    if len(g.split()) <= 4 and not re.search(r"[.!?]", g):
        return "label"
    return "text"


def compute_metrics(preds: list[str], golds: list[str],
                    rejecteds: list[str | None] | None = None) -> dict[str, Any]:
    n = len(golds)
    # Preference-native (DPO) win-rate: of the rows that carry a `rejected` ref,
    # the fraction where the model's answer is closer (token-F1) to the chosen gold
    # than to the rejected response — i.e. it reproduces the stored preference.
    # None for SFT/KTO/RLVR (no rejected ref). A tie counts as NOT a win (strict >).
    rej_list = rejecteds if rejecteds is not None else [None] * n
    wr_rows = wr_win = 0
    # Raw-string strict match (kept for transparency) AND extracted-answer strict
    # match (fair to reasoning models that prepend <think> or fence the answer).
    exact = exact_ex = norm = contains = contains_ex = 0
    f1_sum = rouge_sum = charf1_sum = lenratio_sum = 0.0
    had_scaffold = 0  # rows where extraction changed the string (think/fence/prose)
    # task-aware accumulators
    json_rows = json_valid = 0
    json_struct_sum = json_keyrec_sum = 0.0
    json_keyrec_rows = 0
    num_rows = num_match = 0
    label_rows = label_correct = 0
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    task_counts: dict[str, int] = defaultdict(int)

    for pred, gold, rej in zip(preds, golds, rej_list):
        ans = extract_answer(pred)  # answer with <think>/```fences/preamble removed
        if ans != pred.strip():
            had_scaffold += 1
        task = detect_task(gold)
        task_counts[task] += 1

        # --- preference win-rate (DPO): only rows that carry a rejected ref ---
        if rej is not None:
            wr_rows += 1
            wr_win += int(_token_f1(ans, gold) > _token_f1(ans, rej))

        # --- overlap metrics: scored on the EXTRACTED answer (the fair target) ---
        exact += int(pred.strip() == gold.strip())          # raw, for transparency
        exact_ex += int(ans.strip() == gold.strip())        # extracted (fair)
        nm = _normalize(ans) == _normalize(gold)
        norm += int(nm)
        f1_sum += _token_f1(ans, gold)
        rouge_sum += _rouge_l(ans, gold)
        charf1_sum += _char_f1(ans, gold)
        contains += int(_contains_gold(pred, gold) > 0)
        contains_ex += int(_contains_gold(ans, gold) > 0)
        gl = max(1, len(_normalize(gold).split()))
        lenratio_sum += len(_normalize(ans).split()) / gl

        # --- task-aware metrics, auto-selected by the gold's shape ---
        if task == "json":
            json_rows += 1
            gold_obj = _first_json(gold)
            pred_obj = _first_json(ans)  # tolerant parse of the extracted answer
            if pred_obj is not None:
                json_valid += 1
                json_struct_sum += _json_structural(pred_obj, gold_obj)
                kr = _json_key_recall(pred_obj, gold_obj)
                if kr is not None:
                    json_keyrec_sum += kr
                    json_keyrec_rows += 1
            else:
                kr = _json_key_recall({}, gold_obj)
                if kr is not None:
                    json_keyrec_rows += 1  # counts as 0 recall
        elif task == "numeric":
            num_rows += 1
            gn, pn = _as_number(gold), _as_number(ans)
            if pn is not None and gn is not None and abs(pn - gn) < 1e-6:
                num_match += 1
        elif task == "label":
            label_rows += 1
            label_correct += int(nm)

        cls = _normalize(gold)
        per_class_total[cls] += 1
        per_class_correct[cls] += int(nm)

    per_class = {
        cls: {
            "total": per_class_total[cls],
            "correct": per_class_correct[cls],
            "accuracy": round(per_class_correct[cls] / per_class_total[cls], 4),
        }
        for cls in sorted(per_class_total)
    }

    def avg(x: float) -> float:
        return round(x / n, 4) if n else 0.0

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "count": n,
        # strict match — both raw and extracted (extracted is the fair number)
        "exact_match": avg(exact),
        "exact_match_extracted": avg(exact_ex),
        "normalized_match": avg(norm),
        "contains_gold": avg(contains),
        "contains_gold_extracted": avg(contains_ex),
        # overlap (scored on the extracted answer)
        "token_f1": avg(f1_sum),
        "rouge_l": avg(rouge_sum),
        "char_f1": avg(charf1_sum),
        "length_ratio": avg(lenratio_sum),
        # extraction diagnostic: how often the model wrapped its answer in
        # <think>/fences/prose (high → strict raw metrics understate quality)
        "scaffold_rate": avg(had_scaffold),
        # --- task-aware (None when no row of that task type) ---
        # JSON: valid = parses at all (strict format gate); structural = same
        # key/type shape; key_recall = fraction of gold keys present.
        "json_valid": rate(json_valid, json_rows),
        "json_structural": round(json_struct_sum / json_rows, 4) if json_rows else None,
        "json_key_recall": round(json_keyrec_sum / json_keyrec_rows, 4) if json_keyrec_rows else None,
        "json_rows": json_rows,
        # numeric extraction tasks
        "numeric_match": rate(num_match, num_rows),
        "numeric_rows": num_rows,
        # classification tasks (short-label gold)
        "label_accuracy": rate(label_correct, label_rows),
        "label_rows": label_rows,
        # preference-native (DPO): fraction of rows where the answer is closer to
        # chosen than rejected. None for SFT/KTO/RLVR (no rejected ref). The metric
        # that actually measures what DPO optimizes — vs gold-overlap, which scores
        # against one acceptable answer.
        "chosen_win_rate": rate(wr_win, wr_rows),
        "chosen_win_rate_rows": wr_rows,
        # how the eval set broke down by auto-detected task type
        "task_mix": dict(task_counts),
        "per_class_accuracy": per_class,
    }


# ----------------------------- data ---------------------------------------- #

def load_eval_rows(path: Path) -> tuple[list[list[dict]], list[str], list[str | None]]:
    """Return (prompt_messages, gold_answers, rejected_refs).

    prompt_messages = all turns up to (but excluding) the final assistant turn.
    gold = the content of the final assistant turn.
    rejected_ref = the DPO 'rejected' response for this prompt when present (extra
    top-level key on a preference eval row), else None. Used ONLY for the
    preference-native chosen_win_rate metric; absent (None) for SFT/KTO/RLVR.
    """
    prompts: list[list[dict]] = []
    golds: list[str] = []
    rejecteds: list[str | None] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row["messages"]
        # final assistant turn is the target; everything before it is the prompt.
        # A PROMPT-ONLY row (no assistant turn) has no reference answer — skip it
        # rather than crash (`max()` over an empty sequence raises). This shouldn't
        # happen for the gold-overlap objectives (SFT/DPO/KTO/RLVR all carry a gold
        # answer); RLAIF is prompt-only by design and is ranked by judge reward
        # WITHOUT this eval, so any such row here is defensive belt-and-suspenders.
        assistant_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if not assistant_idxs:
            continue
        last_assistant = max(assistant_idxs)
        prompts.append(msgs[:last_assistant])
        golds.append(msgs[last_assistant]["content"])
        rej = row.get("rejected_ref")
        rejecteds.append(str(rej) if rej else None)
    return prompts, golds, rejecteds


# --------------------------- inference backends ---------------------------- #

# Cap the vLLM context. Models like Qwen3 advertise a 256K max_position_embeddings,
# which makes vLLM try to reserve a KV cache far larger than a small GPU's memory.
# Our eval prompts are short; this ceiling is plenty and keeps the cache small.
VLLM_MAX_MODEL_LEN = int(os.environ.get("SLM_EVAL_VLLM_MAX_MODEL_LEN", "8192"))
VLLM_GPU_UTIL = float(os.environ.get("SLM_EVAL_VLLM_GPU_UTIL", "0.90"))
# Whether transformers/vLLM may execute modeling code shipped inside the model
# repo (`auto_map` in its config.json). That code runs unsandboxed in this job,
# under the job's execution role, so it is OFF unless the launcher explicitly
# turns it on for a model whose architecture requires it (ModelSpec.
# trust_remote_code -> launch_eval_job -> here).
TRUST_REMOTE_CODE = os.environ.get("SLM_EVAL_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes", "on")


def _percentile(values: list[float], p: float) -> float:
    # Callers MUST guard empty input (see _timing, which emits None for no data) —
    # the 0.0 here is only a safe fallback so a stray direct call can't ZeroDivision/
    # IndexError; it is never the reported metric.
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _timing(total_output_tokens: int, gen_seconds: float, latencies_ms: list[float]) -> dict[str, Any]:
    # p50 is the typical request; p90/p99 expose the TAIL (the slow requests a median
    # hides) — production latency budgets are set on the tail, not the median. All
    # three come from the same per-request latency list. None when no latencies.
    return {
        "gen_seconds": round(gen_seconds, 3),
        "output_tokens": total_output_tokens,
        "tokens_per_sec": round(total_output_tokens / gen_seconds, 2) if gen_seconds > 0 else None,
        "p50_latency_ms": round(_percentile(latencies_ms, 0.5), 1) if latencies_ms else None,
        "p90_latency_ms": round(_percentile(latencies_ms, 0.9), 1) if latencies_ms else None,
        "p99_latency_ms": round(_percentile(latencies_ms, 0.99), 1) if latencies_ms else None,
    }


def generate_batch_vllm(prompt_messages: list[list[dict]]) -> tuple[list[str], dict[str, Any]]:
    import time

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_DIR,
        seed=SEED,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_UTIL,
        enforce_eager=True,  # skip CUDA graph capture — faster startup for short eval runs
    )
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]
    params = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS, seed=SEED
    )
    t0 = time.perf_counter()
    outs = llm.generate(texts, params)
    gen_seconds = time.perf_counter() - t0
    preds = [o.outputs[0].text.strip() for o in outs]
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    # vLLM batches internally; approximate per-request latency as even share.
    per_req = (gen_seconds * 1000) / len(outs) if outs else 0.0
    return preds, _timing(total_tokens, gen_seconds, [per_req] * len(outs))


def generate_batch_hf(prompt_messages: list[list[dict]]) -> tuple[list[str], dict[str, Any]]:
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=TRUST_REMOTE_CODE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    torch.manual_seed(SEED)
    preds: list[str] = []
    latencies_ms: list[float] = []
    total_tokens = 0
    t_all = time.perf_counter()
    for m in prompt_messages:
        text = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            top_p=TOP_P,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        gen = out[0][inputs["input_ids"].shape[1]:]
        total_tokens += int(gen.shape[0])
        preds.append(tok.decode(gen, skip_special_tokens=True).strip())
    gen_seconds = time.perf_counter() - t_all
    return preds, _timing(total_tokens, gen_seconds, latencies_ms)


def generate_batch(prompt_messages: list[list[dict]]) -> tuple[list[str], str, dict[str, Any]]:
    backend = os.environ.get("SLM_EVAL_BACKEND", "vllm").lower()
    if backend == "hf":
        preds, timing = generate_batch_hf(prompt_messages)
        return preds, "hf", timing
    try:
        preds, timing = generate_batch_vllm(prompt_messages)
        return preds, "vllm", timing
    except Exception as e:  # noqa: BLE001 — fall back rather than fail the eval
        print(f"[eval] vLLM unavailable ({e}); falling back to HF generate")
        preds, timing = generate_batch_hf(prompt_messages)
        return preds, "hf", timing


# ------------------------------- main -------------------------------------- #

def main() -> None:
    eval_path = Path(DATASET_DIR) / EVAL_FILE
    print(f"[eval] loading eval rows from {eval_path}")
    prompts, golds, rejecteds = load_eval_rows(eval_path)
    print(f"[eval] {len(golds)} eval rows; model dir {MODEL_DIR}")

    preds, backend, timing = generate_batch(prompts)
    metrics = compute_metrics(preds, golds, rejecteds)
    metrics["decoding"] = {
        "backend": backend,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
    }
    metrics["timing"] = timing

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for prompt_msgs, p, g in zip(prompts, preds, golds):
            # Store the raw prediction, the extracted answer (scaffolding removed),
            # the gold, AND a flattened prompt — so the LLM judge can check
            # faithfulness/grounding against the actual input, not just gold-match.
            prompt_text = "\n\n".join(
                f"[{m.get('role','user')}] {m.get('content','')}" for m in prompt_msgs
            )
            f.write(
                json.dumps(
                    {"prompt": prompt_text, "prediction": p, "extracted": extract_answer(p), "gold": g},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("[eval] metrics:")
    print(json.dumps(metrics, indent=2))
    print("[eval] done")


if __name__ == "__main__":
    main()
