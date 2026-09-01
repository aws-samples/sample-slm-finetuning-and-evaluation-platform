# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Torch-free answer extraction + metric scoring — the SAME logic the eval
harness (container/eval.py) ranks models on, packaged so a reward Lambda can
score RLVR rollouts against the verifiable ground_truth WITHOUT importing torch
or any of the eval container's heavy deps.

This is a faithful copy of eval.py's extraction + metric functions (stdlib only:
json/re/string/collections). Kept deliberately standalone so it can be dropped
into a Lambda deployment package verbatim — a Lambda can't import the backend.
If eval.py's scoring changes, mirror it here (and vice-versa); the metric NAMES
are the contract shared with race.RANK_METRICS so "reward = the metric we rank
on" stays literally true.

Public surface a reward function (preset or user snippet) uses:
    extract_answer(text) -> str          # strip <think>/fences/harmony scaffolding
    score(metric, response, ground_truth) -> float   # 0.0..1.0 for one rollout
    METRIC_NAMES                          # the scoreable metric keys
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any

# --- normalization --------------------------------------------------------- #


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation + articles + extra whitespace (SQuAD-style)."""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --- answer extraction (mirrors eval.py) ----------------------------------- #

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
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|channel\|>|$)",
    re.DOTALL | re.IGNORECASE,
)
_HARMONY_TOKEN_RE = re.compile(r"<\|/?(?:channel|message|start|end|return|constrain)\|>", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _strip_think(s: str) -> str:
    for rx in _REASONING_BLOCK_RES:
        s = rx.sub("", s)
    for open_t, close_t in zip(_OPEN_TAGS, _CLOSE_TAGS):
        if open_t in s:
            s = s.split(open_t)[0] + (s.split(close_t)[-1] if close_t in s else "")
    return s.strip()


def extract_answer(pred: str) -> str:
    """The model's actual answer with reasoning/markdown scaffolding removed
    (gpt-oss harmony final channel, <think>/<reasoning> blocks, <answer> tags,
    ```fences```). Same order as eval.py.extract_answer."""
    s = pred
    finals = _HARMONY_FINAL_RE.findall(s)
    if finals:
        s = finals[-1]
    s = _strip_think(s)
    s = _HARMONY_TOKEN_RE.sub("", s)
    ans = _ANSWER_TAG_RE.findall(s)
    if ans:
        s = max(ans, key=len)
    fences = _FENCE_RE.findall(s)
    if fences:
        s = max(fences, key=len)
    return s.strip()


# --- JSON / numeric helpers ------------------------------------------------ #


def _first_json(s: str) -> Any | None:
    s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass
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


_NUM_RE = re.compile(r"^[+-]?\$?\s*\d[\d,]*\.?\d*\s*%?$")


def _as_number(s: str) -> float | None:
    t = s.strip()
    if not _NUM_RE.match(t):
        # tolerate a number embedded in a short answer (e.g. "#### 72")
        m = re.search(r"[+-]?\d[\d,]*\.?\d*", t.replace("$", "").replace(",", ""))
        if not m:
            return None
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            return None
    t = t.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


# --- metric scorers (each → 0.0..1.0 for ONE (response, gold) pair) -------- #


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
    if not ng:
        return 0.0
    return 1.0 if ng in np_ else 0.0


def _normalized_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def _numeric_match(pred: str, gold: str) -> float:
    gn, pn = _as_number(gold), _as_number(pred)
    if pn is None or gn is None:
        return 0.0
    return 1.0 if abs(pn - gn) < 1e-6 else 0.0


def _json_valid(pred: str, gold: str) -> float:
    return 1.0 if _first_json(pred) is not None else 0.0


# Metric key → scorer. Keys mirror race.RANK_METRICS + the task-aware metrics so
# a reward can be "the exact number the leaderboard ranks on".
_SCORERS = {
    "token_f1": _token_f1,
    "rouge_l": _rouge_l,
    "char_f1": _char_f1,
    "contains_gold": _contains_gold,
    "normalized_match": _normalized_match,
    "exact_match": _exact_match,
    "numeric_match": _numeric_match,
    "label_accuracy": _normalized_match,  # label = normalized exact on a short gold
    "json_valid": _json_valid,
}
METRIC_NAMES = tuple(_SCORERS.keys())


def score(metric: str, response: str, ground_truth: str) -> float:
    """Reward for ONE rollout: extract the answer from `response`, then score it
    against `ground_truth` with the named metric. Always returns a float in
    0.0..1.0; an unknown metric falls back to token_f1 (never raises, so a reward
    call can't crash the GRPO loop)."""
    scorer = _SCORERS.get(metric, _token_f1)
    try:
        ans = extract_answer(response or "")
        val = float(scorer(ans, ground_truth or ""))
    except Exception:  # noqa: BLE001 — a bad row must score 0, never kill the run
        return 0.0
    # clamp to [0, 1]
    return max(0.0, min(1.0, val))
