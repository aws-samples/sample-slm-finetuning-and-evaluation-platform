# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Agentic dataset-investigation core — Strands agent logic.

The deterministic profiler (backend/app/profiler.py) already answers the
DATA-FACET half of dataset quality: structure, task type, label balance, JSON
validity, duplicates, truncation risk, train↔eval leakage. What it CANNOT know
is the TASK and HUMAN facets — the business context behind the numbers:
  * which output contract the model must hit,
  * which failure mode actually matters,
  * whether an observed pattern (e.g. a 2% minority class) is intended or a bug.

This agent layers on top. Grounded in "The Five Facets of Data Quality"
(arXiv 2403.00526): a quality dimension is assessed by some mix of the
data/source/system/task/human facets. We use that as the question-selection gate:

    Ask ONLY about dimensions where the TASK or HUMAN facet is the assessor
    AND the deterministic profile could not resolve it. Never ask what the
    profile already answers (that would be data-facet — derivable, not asked).

Two actions (stateless; the backend passes the precomputed profile in):
  * "questions" — profile in → 3–6 targeted, facet-tagged questions out.
  * "proposal"  — profile + the user's answers in → a confirmed ConfigProposal
                  (task_type, rank_metric, alsoWatch, cutoff guidance, flagged
                  issues, per-answer rationale) that pre-fills Fine-tune and
                  locks the eval metric. Advisory; the user can override.

Deterministic core, LLM only for fuzzy judgment (which questions matter, how to
phrase them, how to fold answers into a recommendation) — never for anything the
profiler already computed.
"""
from __future__ import annotations

import json
import re
from typing import Any

from strands import Agent, tool
from strands.models import BedrockModel

from .judge_tools import (
    ALLOWED_JUDGE_MODELS,
    RewardPromptError,
    try_reward_prompt as _try_reward_prompt_pure,
    validate_reward_prompt as _validate_reward_prompt_pure,
)

# Sonnet 4.5 cross-region inference profile — same id the platform's judge/baseline use.
SONNET_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

MAX_QUESTIONS = 6
MIN_QUESTIONS = 3

# Valid rank metrics the platform actually supports (keep in sync with
# backend/app/profiler.recommend_eval_strategy + container/eval.py). The proposal
# must lock onto one of these — never invent a metric name.
VALID_RANK_METRICS = [
    "label_accuracy",
    "json_structural",
    "json_valid",
    "json_key_recall",
    "numeric_match",
    "token_f1",
    "rouge_l",
    "char_f1",
    "exact_match",
    "normalized_match",
    "contains_gold",
    "llm_judge:overall",
    "llm_judge:faithfulness",
]

# ---------------------------------------------------------------------------
# Facet gate — the heart of "ask only what you can't derive".
#
# From arXiv 2403.00526, the dimensions whose PRIMARY assessor is the TASK or
# HUMAN facet (++): relevancy, added-value, appropriate-amount, believability,
# timeliness, concise-representation, understandability — plus the INTENT behind
# any ambiguous data-facet finding. Those are exactly what the agent may ask.
# Everything the profiler computes (accuracy/completeness/consistency/uniqueness/
# balance/diversity/representativity) is DATA-facet → derivable → never asked.
# ---------------------------------------------------------------------------

_SYSTEM_QUESTIONS = (
    "You are a data-investigation agent for an LLM fine-tuning platform. A "
    "deterministic profiler has ALREADY measured everything computable from the "
    "dataset files: task type, label balance, JSON validity, output lengths, "
    "duplicate/empty/malformed rows, truncation risk, and train/eval leakage. You "
    "will be given that profile as JSON.\n\n"
    "Your job: produce a SHORT list of follow-up questions for the user — but ONLY "
    "about things the profile CANNOT reveal, because they depend on business "
    "context (the 'task' and 'human' facets of data quality). Examples of allowed "
    "questions: what the model's output must look like in production (the output "
    "contract), which kind of mistake is most costly, whether an observed pattern "
    "(e.g. a rare class, a mix of task types, duplicates) is intentional or a data "
    "bug, how fresh/representative the data must be, and what 'good' means for this "
    "task.\n\n"
    "HARD RULES:\n"
    "1. NEVER ask anything the profile already answers (counts, balance, validity, "
    "lengths, task type, leakage). Asking a derivable question is a failure.\n"
    "1a. For label/classification tasks, duplicate OUTPUTS are EXPECTED and normal "
    "(a small fixed set of labels is shared across many rows). NEVER ask whether "
    "output duplication is a data bug for a label task — that is derivable and "
    "wrong to ask. (Duplicate INPUTS/prompts would be worth flagging; duplicate "
    "labels are not.)\n"
    f"2. Ask between {MIN_QUESTIONS} and {MAX_QUESTIONS} questions. Fewer is better "
    "if the dataset is clean and unambiguous.\n"
    "3. Anchor every question in a SPECIFIC observation from the profile (cite the "
    "number) so the user sees why it matters.\n"
    "4. Each question must change a downstream decision (the eval metric to rank "
    "on, the cutoff length, whether to clean/rebalance, or how to interpret a "
    "warning). If a question wouldn't change anything, drop it.\n\n"
    "OBJECTIVE-SPECIFIC FACETS (when the profile has a non-SFT `shape`):\n"
    "- PREFERENCE / DPO data ('shape':'preference', `preference` stats present): the "
    "profiler already measured pair count, identical pairs, and chosen-vs-rejected "
    "lengths — do NOT re-ask those. DO ask the human-context questions: how the "
    "preferences were judged (human raters / a reward model / a heuristic) and how "
    "strong/consistent they are; whether `chosen` and `rejected` were generated by a "
    "model similar to the one being tuned (on-policy) or pasted from elsewhere; and, "
    "IF the profiler flagged a chosen-longer length bias, whether longer really means "
    "better here or is an artifact (the classic DPO verbosity trap).\n"
    "- KTO data ('shape':'kto', `kto` stats present): the profiler already measured "
    "the desirable:undesirable balance — don't re-ask the ratio. DO ask: whether the "
    "observed balance is representative of production; whether avoiding bad outputs "
    "(downside) or producing good ones (upside) matters more (this sets the KTO loss "
    "weighting); and how noisy the good/bad labels are.\n"
    "- RLVR data ('shape':'rlvr', `rlvr` stats present): RLVR trains with GRPO to "
    "maximize a VERIFIABLE reward — the model is rewarded only when a reward function "
    "can CHECK its answer against `ground_truth`. The profiler reports "
    "`groundTruthTask` (numeric/json/label/text), `numericGroundTruthRate`, and "
    "`emptyGroundTruth`. DO ask: (1) is `ground_truth` a single machine-checkable "
    "target (a final number, a canonical string, code that passes tests) rather than "
    "a free-text worked solution? — RLVR rewards a verifiable target, not imitation; "
    "(2) which reward domain matches — gsm8k/prime_math expect a numeric/extractable "
    "answer, prime_code expects code, otherwise a CUSTOM reward is needed — and does "
    "the detected `groundTruthTask` match it? (3) does `ground_truth` use a delimiter "
    "the reward can extract (e.g. the gsm8k `####` convention)? Flag if "
    "`groundTruthTask` is 'text' with long length (looks like prose, not a checkable "
    "answer) or if `emptyGroundTruth` > 0.\n\n"
    "Reply with ONLY a JSON object: "
    '{"questions": [{"id": "q1", "facet": "task|human", "question": "...", '
    '"why": "one line citing the profile observation that prompts this", '
    '"affects": "what downstream decision this informs"}], '
    '"summary": "one sentence on the dataset\'s apparent purpose from the profile"}'
)

_SYSTEM_PROPOSAL = (
    "You are a data-investigation agent for an LLM fine-tuning platform. You are "
    "given (a) the deterministic dataset profile, (b) the profiler's own "
    "recommended eval strategy, and (c) the user's answers to your follow-up "
    "questions. Produce a final, confirmed configuration proposal that will "
    "pre-fill the fine-tuning form and LOCK the evaluation metric.\n\n"
    "Combine the deterministic recommendation with the business context the user "
    "just supplied. The user's stated priority (e.g. 'format correctness matters "
    "most', 'minority class is the whole point') should OVERRIDE the default metric "
    "choice when they conflict — explain why when you do.\n\n"
    f"The rankMetric MUST be one of exactly these strings: {', '.join(VALID_RANK_METRICS)}. "
    "Never invent a metric.\n\n"
    "Reply with ONLY a JSON object: "
    '{"taskType": "json|label|numeric|text", '
    '"rankMetric": "<one of the allowed metrics>", '
    '"alsoWatch": ["<metric>", ...], '
    '"cutoffGuidance": "short note on sequence length given the truncation profile", '
    '"flaggedIssues": ["actionable issue the user should fix before training, if any"], '
    '"rationale": ["bullet explaining each key choice, referencing the profile AND the user answers"], '
    '"appliedAnswers": {"q1": "how this answer shaped the proposal", ...}}'
)


# Per-invocation override of the reasoning model. The platform threads the
# Settings-selected model id into the invoke payload as "modelId"; the entrypoint
# (agent.py) calls set_reasoning_model() with it before dispatching. Defaults to
# Sonnet 4.5. Reset each request so one invocation's choice can't leak to the next.
_REASONING_MODEL_ID = SONNET_MODEL_ID


def set_reasoning_model(model_id: str | None) -> None:
    """Set the Bedrock model the dataset/triage/interpret agents reason with for
    this invocation. None/empty restores the Sonnet 4.5 default."""
    global _REASONING_MODEL_ID
    _REASONING_MODEL_ID = (model_id or "").strip() or SONNET_MODEL_ID


def _bedrock_agent(system_prompt: str, region: str) -> Agent:
    """A Strands Agent backed by the selected Bedrock model, deterministic (temp 0).
    Model defaults to Sonnet 4.5; overridable per request via set_reasoning_model."""
    model = BedrockModel(model_id=_REASONING_MODEL_ID, region_name=region, temperature=0.0)
    return Agent(model=model, system_prompt=system_prompt)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object from the model reply, tolerant of stray prose."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in agent reply: {text[:200]}")
    return json.loads(m.group(0))


def _slim_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Trim the profile to what the agent needs — drops verbose distributions the
    model doesn't reason over, keeping the prompt cheap and focused."""
    def slim_file(f: dict[str, Any] | None) -> dict[str, Any] | None:
        if not f:
            return None
        keep = {
            k: f.get(k)
            for k in (
                "rows", "parsed", "malformed", "emptyOutputs", "duplicateOutputs",
                "taskMix", "dominantTask", "taskConsistency", "scaffold",
                "json", "labels", "truncation",
            )
            if f.get(k) is not None
        }
        # For classification, duplicate OUTPUTS are inherent (a small fixed label
        # set shared across many rows) — the profiler suppresses that warning for
        # label tasks. Drop the raw count so the agent doesn't re-raise it as a
        # false "is this a data bug?" question; the label distribution already
        # conveys output diversity.
        if keep.get("dominantTask") == "label":
            keep.pop("duplicateOutputs", None)
        return keep

    return {
        "name": profile.get("name"),
        "hasVal": profile.get("hasVal"),
        "evalOnly": profile.get("evalOnly"),
        "structure": profile.get("structure"),
        "train": slim_file(profile.get("train")),
        "eval": slim_file(profile.get("eval")),
        "leakage": profile.get("leakage"),
        "recommendation": profile.get("recommendation"),
        "warnings": profile.get("warnings"),
        # Objective + preference/KTO/RLVR stats so the agent can reason about
        # non-SFT data. Without these it's blind to chosen/rejected length bias,
        # identical pairs, KTO class balance, and RLVR ground_truth verifiability.
        # Only present for the matching shape.
        "shape": profile.get("shape"),
        "objective": profile.get("objective"),
        "preference": profile.get("preference"),
        "kto": profile.get("kto"),
        "rlvr": profile.get("rlvr"),
        # RLAIF prompt-only stats (rows, promptTask, promptTaskMix, promptWordLen) so
        # the reward-author agent can ground its candidate good/bad responses in the
        # dataset's actual prompt mix instead of guessing.
        "rlaif": profile.get("rlaif"),
    }


def generate_questions(profile: dict[str, Any], region: str = "us-east-1") -> dict[str, Any]:
    """Profile in → facet-gated follow-up questions out. Clamps to MIN..MAX."""
    agent = _bedrock_agent(_SYSTEM_QUESTIONS, region)
    slim = _slim_profile(profile)
    prompt = (
        "Here is the deterministic dataset profile:\n\n"
        f"{json.dumps(slim, indent=2)}\n\n"
        "Generate the follow-up questions now (JSON only)."
    )
    reply = str(agent(prompt))
    obj = _extract_json(reply)
    questions = obj.get("questions", [])[:MAX_QUESTIONS]
    # Ensure stable ids the frontend/proposal step can key on.
    for i, q in enumerate(questions, 1):
        q.setdefault("id", f"q{i}")
    return {"questions": questions, "summary": obj.get("summary", "")}


def synthesize_proposal(
    profile: dict[str, Any],
    answers: dict[str, str],
    region: str = "us-east-1",
) -> dict[str, Any]:
    """Profile + user answers → confirmed ConfigProposal. rankMetric is validated
    against VALID_RANK_METRICS; on a bad value we fall back to the profiler's own
    deterministic recommendation (never ship an invented metric)."""
    agent = _bedrock_agent(_SYSTEM_PROPOSAL, region)
    slim = _slim_profile(profile)
    rec = profile.get("recommendation", {})
    prompt = (
        "DETERMINISTIC PROFILE:\n"
        f"{json.dumps(slim, indent=2)}\n\n"
        "PROFILER'S RECOMMENDED EVAL STRATEGY:\n"
        f"{json.dumps(rec, indent=2)}\n\n"
        "USER'S ANSWERS TO THE FOLLOW-UP QUESTIONS:\n"
        f"{json.dumps(answers, indent=2)}\n\n"
        "Produce the final configuration proposal now (JSON only)."
    )
    reply = str(agent(prompt))
    obj = _extract_json(reply)

    # Guardrail: lock rankMetric to a real metric, else use the deterministic one.
    rank = obj.get("rankMetric")
    if rank not in VALID_RANK_METRICS:
        fallback = rec.get("rankMetric", "token_f1")
        obj["rankMetric"] = fallback
        obj.setdefault("flaggedIssues", []).append(
            f"agent proposed unsupported metric '{rank}'; fell back to '{fallback}'"
        )
    # Keep only supported alsoWatch metrics.
    obj["alsoWatch"] = [m for m in obj.get("alsoWatch", []) if m in VALID_RANK_METRICS]
    return obj


# ===========================================================================
# Failure-triage agent — "why did my run break, and how do I fix it?"
#
# Reads a failed training/eval job's failure reason + log tail + its config, and
# returns a plain-language diagnosis + a CONCRETE, advisory fix. Deterministic
# core elsewhere (selfheal.classify_failure runs first in the backend and is
# passed in as a hint); the agent's value is interpreting the fuzzy log text and
# turning it into an actionable recommendation a non-expert can follow.
# ADVISORY ONLY — it recommends a retry/config change; the user clicks retry.
# ===========================================================================

_SYSTEM_TRIAGE = (
    "You are a debugging assistant for an LLM fine-tuning platform built on "
    "Amazon SageMaker + LLaMA-Factory. A training or evaluation job FAILED. You "
    "are given the failure reason, a tail of the job log, the model, and the job "
    "config (instance type, hyperparameters). A deterministic classifier has "
    "already tagged the failure category — use it as a strong hint but rely on "
    "the log for specifics.\n\n"
    "Explain, in plain language a non-expert can act on: what went wrong, the "
    "most likely root cause, and the SINGLE most effective fix to try next. Be "
    "concrete — name the exact knob (e.g. 'lower micro_batch_size from 8 to 2', "
    "'switch instance from ml.g5.2xlarge to ml.g5.4xlarge for more GPU memory', "
    "'this chat template isn't in the current image — use the latest image "
    "tier', 'raise cutoff_len'). Distinguish a RETRYABLE transient issue (spot "
    "capacity, throttling) from a real config/resource problem. Never invent log "
    "lines.\n\n"
    "METHOD MATTERS — check the config's finetuning_type:\n"
    "  • lora / qlora (adapter): the LoRA-centric fixes above apply (rank, "
    "micro_batch, instance size).\n"
    "  • full / freeze (FULL-WEIGHT, no adapter): these are far heavier and behave "
    "differently. For an OUT-OF-MEMORY failure the fix is usually NOT just a bigger "
    "instance (full fine-tuning needs ~16-20 bytes/param of GPU memory and is capped "
    "at small models on one GPU) — prefer: switch from 'full' to 'freeze' (train only "
    "the top N layers), lower freeze_trainable_layers, enable gradient checkpointing, "
    "lower per_device_train_batch_size, or pick a SMALLER model. For DIVERGENCE / NaN "
    "loss / garbage output on full/freeze, the usual cause is the learning rate being "
    "too high — full-weight needs ~1e-5, NOT the LoRA-scale 1e-4; recommend lowering "
    "learning_rate to ~1e-5. Do NOT suggest lora_rank for a full/freeze run (it has "
    "no adapter).\n\n"
    "Reply with ONLY a JSON object: "
    '{"summary": "one-sentence what-happened", '
    '"rootCause": "the most likely cause, grounded in the log", '
    '"fix": "the single most effective next action, concrete and specific", '
    '"retryable": true|false, '
    '"configChanges": {"<field>": "<suggested value>", ...}, '
    '"confidence": "high|medium|low"}'
)


def triage_failure(context: dict[str, Any], region: str = "us-east-1") -> dict[str, Any]:
    """Diagnose a failed job. `context` = {model, failureReason, logTail, config,
    classification}. Returns a plain-language diagnosis + concrete advisory fix."""
    agent = _bedrock_agent(_SYSTEM_TRIAGE, region)
    prompt = (
        "FAILED JOB CONTEXT:\n"
        f"{json.dumps(context, indent=2, default=str)[:8000]}\n\n"
        "Diagnose the failure and recommend the single best fix now (JSON only)."
    )
    reply = str(agent(prompt))
    return _extract_json(reply)


# ===========================================================================
# Reward-prompt authoring agent — "write me a calibrated RLAIF judge rubric."
#
# This is the FIRST tool-using agent on the runtime (the other four call
# Agent(prompt) once, no tools). It must ACT, not just opine: draft a rubric,
# fabricate good/bad candidate responses, score them with a REAL judge LLM, read
# the spread, and revise until good clearly beats bad — a closed feedback loop
# deterministic code can't do. ADVISORY: it returns a draft for the user to
# review + deploy through the unchanged path; it never persists or launches.
#
# Hard caps (enforced in CODE, never just the prompt): a per-session judge-call
# counter raises once it crosses the cap, so a runaway tool loop can't burn
# billable Converse calls; the round budget is folded into that same cap.
# ===========================================================================

# Bound the loop in CODE, not just the prompt: ~3 rounds × ~8 candidates.
_REWARD_AUTHOR_MAX_JUDGE_CALLS = 24
_REWARD_AUTHOR_MAX_ROUNDS = 3
# Discrimination threshold (good mean − bad mean) the rubric must clear to be
# called calibrated — mirrors the backend dry-run spread's `discriminates`.
_REWARD_AUTHOR_SEPARATION = 0.3

_SYSTEM_REWARD_AUTHOR = (
    "You are a reward-rubric author for RLAIF (GRPO-from-AI-feedback) fine-tuning. "
    "Given a plain-English training goal and a profile of the PROMPT-ONLY dataset, "
    "write a precise AI-judge rubric that scores a model response from 0.0 to 1.0.\n\n"
    "The rubric MUST contain the literal {{prompt}} and {{response}} placeholders "
    "(the judge fills them with the rollout's prompt + the model's response) and "
    "MUST end by instructing the judge to reply with ONLY JSON "
    '{"score": <0..1>, "reasoning": "<one sentence>"}.\n\n'
    "Make the criteria concrete and goal-specific. For 'reward concise, friendly "
    "tone': reward brevity + warmth, penalize verbosity + curtness. VAGUE rubrics "
    "that score everything ~0.5 are the failure mode you exist to prevent.\n\n"
    "To PROVE the rubric works you MUST use your tools, in this order:\n"
    "1. Call `set_rubric` with your draft. It validates the placeholders AND makes "
    "the draft the CURRENT rubric that scoring uses; it returns an error if a "
    "placeholder is missing (fix and call again). (`validate_reward_prompt` is also "
    "available to dry-check a draft without making it current.)\n"
    "2. Draft 3-4 deliberately GOOD and 3-4 deliberately BAD candidate responses to "
    "sample dataset prompts (draw on the profile's promptTask/promptTaskMix so they "
    "are REALISTIC for this dataset), then call `score_candidate` on EACH (pass its "
    "intended_label 'good' or 'bad') to get its 0..1 judge score. Realistic bad "
    "examples matter — fabricated-obvious 'bad' answers give a false-positive spread.\n"
    "3. Inspect the spread. If (good mean − bad mean) < ~0.3 the rubric does not "
    "discriminate: REVISE it, call `set_rubric` with the new text, and re-score. You "
    "get at most 3 rounds and a hard cap on total score_candidate calls — if you "
    "exceed it the tools stop responding, so spend calls deliberately.\n\n"
    "Pick ONE judge model id from this allowed list (or the empty string '' for the "
    f"recipe default): {list(ALLOWED_JUDGE_MODELS)}. Never invent a judge id.\n\n"
    "You NEVER deploy or launch — you only produce a calibrated draft for the user "
    "to review.\n\n"
    "When done, emit your final result as a SINGLE fenced ```json block (and nothing "
    "after it) with EXACTLY this schema:\n"
    "```json\n"
    "{\n"
    '  "draftPrompt": "<full rubric text with {{prompt}} and {{response}}>",\n'
    '  "rewardModelId": "<one of the allowed ids or empty string>",\n'
    '  "samples": [{"prompt": "...", "response": "...", "intendedLabel": "good|bad", '
    '"score": 0.0, "reasoning": "..."}],\n'
    '  "scoreSpread": {"goodMean": 0.0, "badMean": 0.0, "separation": 0.0, '
    '"discriminates": true},\n'
    '  "rationale": ["why this rubric matches the goal", "what each criterion rewards"],\n'
    '  "iterations": 1,\n'
    '  "warnings": ["any caveat: tiny sample / weak separation / judge unavailable"]\n'
    "}\n"
    "```"
)


def _extract_last_json(text: str) -> dict[str, Any]:
    """Extract the agent's final result object from a chatty, multi-turn tool-loop
    reply. The greedy `_extract_json` (`\\{.*\\}`) grabs from the FIRST brace to the
    LAST, which spans tool-call noise and breaks here — so we prefer a fenced
    ```json block, then fall back to the LAST balanced top-level object."""
    # 1) A fenced ```json block (what the system prompt asks for) — take the last one.
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    for block in reversed(fences):
        try:
            return json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue
    # 2) Fall back: scan for balanced top-level {...} objects, return the last that
    #    parses (the final answer is the last object in the reply).
    objs: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objs.append(text[start : i + 1])
    for obj in reversed(objs):
        try:
            return json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"no JSON result object in agent reply: {(text or '')[:200]}")


def _coerce_spread(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the good/bad score spread from the scored samples — authoritative,
    NOT trusted from the model (it could miscompute its own arithmetic). Mirrors the
    backend dry-run spread shape."""
    good = [float(s.get("score", 0.0)) for s in samples if s.get("intendedLabel") == "good"]
    bad = [float(s.get("score", 0.0)) for s in samples if s.get("intendedLabel") == "bad"]
    if not good or not bad:
        return {"goodMean": None, "badMean": None, "separation": None, "discriminates": False}
    gm, bm = sum(good) / len(good), sum(bad) / len(bad)
    return {
        "goodMean": round(gm, 4), "badMean": round(bm, 4),
        "separation": round(gm - bm, 4), "discriminates": (gm - bm) >= _REWARD_AUTHOR_SEPARATION,
    }


def author_reward_prompt(
    goal: str,
    profile: dict[str, Any],
    prior_result: dict[str, Any] | None = None,
    region: str = "us-east-1",
) -> dict[str, Any]:
    """Draft + calibrate an RLAIF judge rubric for `goal`, grounded in the prompt-only
    dataset `profile`. Uses Strands tools to score candidate responses with a real
    judge and iterate until good clearly beats bad (or a cap is hit). Returns the
    output schema in _SYSTEM_REWARD_AUTHOR. Advisory — never deploys.

    `prior_result` (optional): a previous draft + the user's feedback, so the agent
    can regenerate-with-feedback."""
    # Per-session judge-call budget, enforced in CODE. The closure counter is the
    # real backstop — the prompt's "max 3 rounds" is advisory; this is not.
    state = {"judge_calls": 0, "scored": []}

    @tool
    def validate_reward_prompt(prompt: str) -> str:
        """Validate a draft RLAIF judge rubric. Returns 'ok' if it contains both the
        {{prompt}} and {{response}} placeholders, else an actionable error string.

        Args:
            prompt: the draft rubric text to validate.
        """
        try:
            _validate_reward_prompt_pure(prompt)
            return "ok — the rubric has both required placeholders."
        except RewardPromptError as e:
            return f"INVALID: {e}"

    @tool
    def score_candidate(
        prompt: str, response: str, intended_label: str, judge_model_id: str = ""
    ) -> str:
        """Score ONE candidate response against the CURRENT rubric using a real AI
        judge, to measure whether the rubric separates good from bad. Returns the
        0..1 score + the judge's reasoning (or an error string if the cap is hit or
        the judge fails).

        Args:
            prompt: the dataset prompt the response is answering.
            response: the candidate model response to score.
            intended_label: 'good' or 'bad' — what you INTEND this candidate to be,
                so the spread can be computed.
            judge_model_id: the judge model id (one of the allowed list, or '' for
                the recipe-default preview judge).
        """
        if state["judge_calls"] >= _REWARD_AUTHOR_MAX_JUDGE_CALLS:
            return (
                "STOP: judge-call budget exhausted "
                f"({_REWARD_AUTHOR_MAX_JUDGE_CALLS} calls). Do not call score_candidate "
                "again — emit your final JSON result now with the scores you already have."
            )
        rubric = state.get("current_rubric") or ""
        if not rubric:
            return "ERROR: set a rubric with validate_reward_prompt first, then score against it."
        state["judge_calls"] += 1
        try:
            out = _try_reward_prompt_pure(rubric, prompt, response, judge_model_id, region=region)
        except RewardPromptError as e:
            # A bad judge id / missing placeholder is actionable — surface it, don't crash.
            return f"ERROR: {e}"
        label = "good" if str(intended_label).lower().startswith("g") else "bad"
        rec = {
            "prompt": prompt, "response": response, "intendedLabel": label,
            "score": out["score"], "reasoning": out["reasoning"], "error": out["error"],
        }
        state["scored"].append(rec)
        if out["error"]:
            return f"judge error (scored 0.0): {out['error']}"
        return f"score={out['score']:.3f} ({label}) — {out['reasoning']}"

    @tool
    def set_rubric(prompt: str) -> str:
        """Set the CURRENT rubric that subsequent score_candidate calls will use.
        Call this whenever you draft or revise the rubric, BEFORE scoring against it.
        Returns 'ok' or a validation error.

        Args:
            prompt: the rubric text to make current.
        """
        try:
            _validate_reward_prompt_pure(prompt)
        except RewardPromptError as e:
            return f"INVALID (rubric not set): {e}"
        state["current_rubric"] = prompt
        return "ok — this rubric is now current; score candidates against it."

    model = BedrockModel(model_id=SONNET_MODEL_ID, region_name=region, temperature=0.0)
    agent = Agent(
        model=model,
        system_prompt=_SYSTEM_REWARD_AUTHOR,
        tools=[set_rubric, validate_reward_prompt, score_candidate],
    )

    slim = _slim_profile(profile)
    parts = [
        f"TRAINING GOAL (what the reward should encourage):\n{goal}\n",
        "PROMPT-ONLY DATASET PROFILE (RLAIF rows have no ground_truth; the judge IS "
        f"the reward):\n{json.dumps(slim, indent=2)}\n",
        f"ALLOWED JUDGE MODELS: {list(ALLOWED_JUDGE_MODELS)} (or '' for recipe default).\n",
        "Author and CALIBRATE the rubric now. Use set_rubric, then score_candidate on "
        "good and bad examples, revise if separation < 0.3, then emit the final JSON.",
    ]
    if prior_result:
        parts.append(
            "\nA PRIOR draft + the user's feedback to improve on:\n"
            f"{json.dumps(prior_result, indent=2, default=str)[:4000]}"
        )
    reply = str(agent("\n".join(parts)))

    obj = _extract_last_json(reply)

    # ---- Authoritative server-side reconciliation (never trust the model's math) ----
    # Use the rubric we actually validated + scored against, and the scores we
    # actually captured from the judge tool, not whatever the model echoed back.
    final_rubric = state.get("current_rubric") or obj.get("draftPrompt") or ""
    scored = state["scored"]
    warnings: list[str] = list(obj.get("warnings") or [])

    # The draft must pass validation (a missing placeholder can never reach deploy).
    try:
        _validate_reward_prompt_pure(final_rubric)
    except RewardPromptError as e:
        warnings.append(f"draft rubric failed validation: {e}")
        # Fall back to whatever the model emitted as draftPrompt if that validates.
        cand = obj.get("draftPrompt") or ""
        try:
            _validate_reward_prompt_pure(cand)
            final_rubric = cand
            warnings.pop()  # the fallback is valid
        except RewardPromptError:
            pass

    # Lock the judge id to the allowlist (or '').
    reward_model_id = str(obj.get("rewardModelId") or "").strip()
    if reward_model_id and reward_model_id not in ALLOWED_JUDGE_MODELS:
        warnings.append(
            f"agent proposed unsupported judge model '{reward_model_id}'; "
            "fell back to the recipe default"
        )
        reward_model_id = ""

    # Spread is computed from the REAL captured scores, not the model's claim.
    spread = _coerce_spread(scored) if scored else obj.get("scoreSpread")
    if not scored:
        warnings.append(
            "the agent emitted no judge-scored candidates — the spread is the model's "
            "own (unverified) estimate; re-run to calibrate against a real judge."
        )
    elif not (spread and spread.get("discriminates")):
        warnings.append(
            "the rubric did not clearly separate good from bad (separation < 0.3) — "
            "tighten the criteria before deploying, or treat the reward as weak."
        )
    if state["judge_calls"] >= _REWARD_AUTHOR_MAX_JUDGE_CALLS:
        warnings.append(
            f"hit the {_REWARD_AUTHOR_MAX_JUDGE_CALLS}-judge-call cap; the draft is the "
            "best seen within budget."
        )

    return {
        "draftPrompt": final_rubric,
        "rewardModelId": reward_model_id,
        "samples": scored or (obj.get("samples") or []),
        "scoreSpread": spread,
        "rationale": list(obj.get("rationale") or []),
        "iterations": int(obj.get("iterations") or 1),
        "judgeCalls": state["judge_calls"],
        "warnings": warnings,
    }


# ===========================================================================
# Results-interpreter agent — "which model should I actually ship?"
#
# Reads the leaderboard (every candidate's quality/cost/latency + any frontier
# baselines) and the user's stated constraints, and recommends which model to
# ship, in plain language, with the tradeoff reasoning. ADVISORY.
# ===========================================================================

_SYSTEM_INTERPRET = (
    "You are an ML advisor helping a user choose which fine-tuned model to ship "
    "from a leaderboard. Each row has quality metrics (token_f1, json_valid, "
    "label_accuracy, llm_judge scores, etc.), a measured training cost, a "
    "projected self-host inference cost per 1k tokens, and latency. Some rows are "
    "frontier-model baselines (Claude Haiku/Sonnet/Opus) with their ACTUAL API "
    "cost — these are the 'buy instead of fine-tune' alternatives.\n\n"
    "Given the user's priorities (which may emphasize cost, latency, or a "
    "specific quality dimension), recommend ONE model to ship and explain the "
    "tradeoff in plain language. Quantify the comparison (e.g. '92% of the F1 at "
    "40% the cost'). If a fine-tuned SLM beats or matches a frontier baseline at "
    "lower cost, say so explicitly — that's the platform's core value. If the "
    "metrics are within noise, say it's a toss-up and recommend the cheaper/"
    "faster one. Be honest about any caveat (tiny eval set, missing metric).\n\n"
    "Each row also has a `method` (lora | qlora | full | freeze). When a candidate "
    "was trained with full or freeze (full-weight) fine-tuning, factor in its "
    "tradeoff: it can reach higher quality (all weights updated) but costs more to "
    "train and carries a higher catastrophic-forgetting risk than a LoRA adapter, "
    "especially on small datasets. If a full/freeze row only marginally beats a "
    "LoRA row, prefer the LoRA one and say why; if it wins clearly, note the "
    "higher train cost as the price of that quality. Mention the method in your "
    "reasoning when it affects the recommendation.\n\n"
    "Reply with ONLY a JSON object: "
    '{"recommendation": "the model id/name to ship", '
    '"reasoning": "2-3 sentences quantifying the tradeoff in plain language", '
    '"runnerUp": "the next-best option and when you\'d pick it instead", '
    '"vsBaseline": "how the pick compares to the frontier baseline(s), if any", '
    '"caveats": ["any caveat the user should know"]}'
)


def interpret_results(
    leaderboard: dict[str, Any],
    priorities: str = "",
    region: str = "us-east-1",
) -> dict[str, Any]:
    """Recommend which model to ship. `leaderboard` = {rows, baselines}.
    `priorities` = the user's free-text constraints (optional)."""
    agent = _bedrock_agent(_SYSTEM_INTERPRET, region)
    prompt = (
        "LEADERBOARD:\n"
        f"{json.dumps(leaderboard, indent=2, default=str)[:9000]}\n\n"
        f"USER PRIORITIES: {priorities or '(none stated — balance quality, cost, latency)'}\n\n"
        "Recommend which model to ship now (JSON only)."
    )
    reply = str(agent(prompt))
    return _extract_json(reply)
