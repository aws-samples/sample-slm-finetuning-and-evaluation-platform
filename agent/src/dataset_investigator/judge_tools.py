# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""VENDORED pure judge-scoring helpers for the reward-author agent.

These are deliberate, line-for-line MIRRORS of the same-named helpers in
``backend/app/reward_functions.py`` (validate_reward_prompt / _fill_reward_prompt /
_parse_judge_score / try_reward_prompt + the ALLOWED_JUDGE_MODELS and
_DRY_RUN_FALLBACK_JUDGE constants).

WHY DUPLICATED: the AgentCore Runtime container packages ONLY the agent tree
(``agent.py`` + ``src/dataset_investigator/``); it does NOT import the backend
``app`` package. The reward-author agent calls the judge from INSIDE the runtime
(the runtime's own execution role makes the Bedrock Converse call), so the two
pure functions are vendored here to keep the agent self-contained.

DRIFT GUARD: ``backend/tests/test_reward_author_parity.py`` imports BOTH this
module and ``app.reward_functions`` and asserts they agree on a fixed fixture set
(placeholder validation, prompt filling, judge-reply parsing, the allowlist, and
a stubbed Converse score). ANY edit here MUST be mirrored in reward_functions.py
(and vice-versa) or that test fails. Follow-up: consolidate into one importable
module shared by backend + agent.

Kept import-light at module load (only json/re/math) — boto3 is imported lazily
inside try_reward_prompt — so the backend test venv can load this file by path
WITHOUT pulling in strands.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

# Judge models the RLAIF recipe accepts (mirror of reward_functions.ALLOWED_JUDGE_MODELS,
# filtered to us-east-1). "" = the recipe's default judge.
ALLOWED_JUDGE_MODELS = (
    "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-120b-1:0",
    "qwen.qwen3-32b-v1:0",
    "qwen.qwen3-coder-30b-a3b-v1:0",
)

# The dry-run judge used when the rubric's chosen judge is blank/recipe-default:
# the recipe default isn't Converse-invokable inline, so PREVIEW with the
# platform's default judge (Sonnet 4.5). Mirror of _DRY_RUN_FALLBACK_JUDGE.
_DRY_RUN_FALLBACK_JUDGE = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# The two placeholders the RLAIF judge fills with the rollout's prompt + the
# model's response. Without them the judge can't see what it's scoring.
_REWARD_PROMPT_PLACEHOLDERS = ("{{prompt}}", "{{response}}")


class RewardPromptError(ValueError):
    """A reward (judge) prompt is invalid — missing placeholders or a bad judge id.
    The agent's own error type (the backend uses RewardError); the drift test
    compares behavior, not the exception class."""


def validate_reward_prompt(prompt: str) -> None:
    """Validate an RLAIF reward (judge) prompt — mirror of
    reward_functions.validate_reward_prompt. Must be non-empty and contain BOTH the
    {{prompt}} and {{response}} placeholders (whitespace inside the braces is
    tolerated). Raises RewardPromptError with an actionable message."""
    if not prompt or not prompt.strip():
        raise RewardPromptError("reward prompt is empty")
    compact = re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{{\1}}", prompt)
    missing = [p for p in _REWARD_PROMPT_PLACEHOLDERS if p not in compact]
    if missing:
        raise RewardPromptError(
            "reward prompt must include the "
            + " and ".join(missing)
            + " placeholder(s) so the AI judge can see the rollout's prompt and the "
            "model's response. Example: 'Rate the response to {{prompt}}: {{response}} "
            "— reply with JSON {\"score\": 0..1, \"reasoning\": \"...\"}'."
        )


def _fill_reward_prompt(prompt_text: str, prompt: str, response: str) -> str:
    """Substitute {{prompt}} / {{response}} (whitespace-tolerant). Mirror of
    reward_functions._fill_reward_prompt."""
    out = re.sub(r"\{\{\s*prompt\s*\}\}", lambda _m: prompt or "", prompt_text)
    out = re.sub(r"\{\{\s*response\s*\}\}", lambda _m: response or "", out)
    return out


def _parse_judge_score(text: str) -> tuple[float, str, str | None]:
    """Pull {"score":0..1,"reasoning":...} out of a judge reply. Mirror of
    reward_functions._parse_judge_score. Never raises — a malformed reply yields
    (0.0, "", "<reason>") so a calibration loop survives one bad row."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return 0.0, "", "judge reply had no JSON object"
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return 0.0, "", "judge reply was not valid JSON"
    raw = obj.get("score")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0, str(obj.get("reasoning", "")), "judge reply had no numeric score"
    if math.isnan(val) or math.isinf(val):  # NaN/inf guard (mirrors try_reward)
        val = 0.0
    return max(0.0, min(1.0, val)), str(obj.get("reasoning", "")), None


def try_reward_prompt(
    prompt_text: str,
    prompt: str,
    response: str,
    judge_model_id: str = "",
    *,
    region: str = "us-east-1",
    _client: Any = None,
) -> dict[str, Any]:
    """Dry-run an RLAIF judge RUBRIC on one (prompt, response) pair — the vendored
    judge analogue of try_reward, called by the reward-author agent's score tool.
    Mirror of reward_functions.try_reward_prompt (the agent builds its own boto3
    bedrock-runtime client from the runtime's ambient credentials rather than the
    backend's boto session).

    Validation that DOES raise (caller-actionable, before any billable Converse):
      * the rubric is missing a placeholder (validate_reward_prompt), or
      * judge_model_id is a non-empty id NOT in ALLOWED_JUDGE_MODELS.
    Returns {"score": float in [0,1], "reasoning": str, "error": str|None};
    NEVER raises on a malformed/erroring judge reply (returns score 0.0 + error).
    """
    validate_reward_prompt(prompt_text)  # raises if placeholders missing
    jm = (judge_model_id or "").strip()
    if jm and jm not in ALLOWED_JUDGE_MODELS:
        raise RewardPromptError(
            f"unknown judge model {jm!r}; choose one of {list(ALLOWED_JUDGE_MODELS)} "
            "(or leave blank for the recipe default)"
        )
    model_id = jm or _DRY_RUN_FALLBACK_JUDGE
    filled = _fill_reward_prompt(prompt_text, prompt, response)

    client = _client
    if client is None:
        from .aws_user_agent import get_client

        client = get_client("bedrock-runtime", region_name=region)

    try:
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": filled}]}],
            inferenceConfig={"maxTokens": 300, "temperature": 0.0},
        )
        out_msg = resp.get("output", {}).get("message", {})
        text = "".join(b.get("text", "") for b in out_msg.get("content", []))
    except Exception as e:  # noqa: BLE001 — a judge/Bedrock error must not kill the loop
        return {"score": 0.0, "reasoning": "", "error": f"judge call failed: {e}"}

    score, reasoning, err = _parse_judge_score(text)
    return {"score": score, "reasoning": reasoning, "error": err}
