# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Agentic dataset-investigation — backend client for the AgentCore Runtime.

The deterministic profiler (profiler.py) answers the data-facet half of dataset
quality. This module drives the AGENT half: it invokes the Strands agent hosted
on Bedrock AgentCore Runtime (deployed separately from agent/) to (1) generate
facet-gated follow-up questions about the business context the data can't reveal,
and (2) fold the user's answers into a confirmed eval-config proposal that
pre-fills Fine-tune and locks the ranking metric.

The agent is stateless: we compute the profile here and pass it in the payload,
so the runtime only needs Bedrock access, not the dataset. Results persist per
split under the "investigations" store collection so a finished investigation
survives refresh/restart (mirrors judge.py's per-key persistence).

Advisory only — never mutates the dataset; the user can override the proposal.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any

from .aws_config import load_aws_config
from .obs import log_event
from .orchestrate import _session
from .profiler import profile_dataset
from .store import get_store

INVESTIGATIONS = "investigations"  # store collection, keyed by split_id
_QUESTIONS_FILE = "questions.json"
_PROPOSAL_FILE = "proposal.json"
_STATUS_FILE = "status.json"

# The AgentCore Runtime is deployed by the `agentcore` CLI and gets a RANDOM
# suffix on its id (e.g. dataset_investigator-7y41YU9maE), so the ARN differs per
# account/deploy and must NOT be hardcoded. Only the runtime NAME is stable, so we
# resolve the ARN by name at runtime (control plane) and cache it. Resolution
# order: explicit override (config/env) → look up by name. This keeps a fresh
# deploy working with zero config.
_RUNTIME_NAME = os.environ.get("SLM_AGENT_RUNTIME_NAME", "dataset_investigator")
_arn_cache: dict[str, str] = {}


def _resolve_arn_by_name(name: str) -> str:
    """Find the AgentCore runtime ARN by its (stable) name via the control plane.
    Prefers a READY runtime; raises if none match."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    ctl = boto_sess.client("bedrock-agentcore-control", region_name=cfg.region)
    matches = []
    paginator = None
    try:
        paginator = ctl.get_paginator("list_agent_runtimes")
    except Exception:  # noqa: BLE001 — fall back to a single call if no paginator
        pass
    pages = paginator.paginate() if paginator else [ctl.list_agent_runtimes()]
    for page in pages:
        for rt in page.get("agentRuntimes", []):
            if rt.get("agentRuntimeName") == name:
                matches.append(rt)
    if not matches:
        raise RuntimeError(
            f"no AgentCore runtime named '{name}' found in {cfg.region}; "
            "deploy it from agent/ by running `agentcore configure --entrypoint "
            "agent.py` and then `agentcore launch`"
        )
    ready = next((m for m in matches if m.get("status") == "READY"), matches[0])
    return ready["agentRuntimeArn"]


def _runtime_arn() -> str:
    # 1) explicit override wins (config.json or env) — full ARN.
    cfg = load_aws_config()
    override = getattr(cfg, "agent_runtime_arn", None) or os.environ.get("SLM_AGENT_RUNTIME_ARN")
    if override:
        return override
    # 2) resolve by stable name, cached for the process lifetime.
    if _RUNTIME_NAME not in _arn_cache:
        _arn_cache[_RUNTIME_NAME] = _resolve_arn_by_name(_RUNTIME_NAME)
    return _arn_cache[_RUNTIME_NAME]


def _invoke_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    """Call the AgentCore Runtime with a JSON payload, return the parsed reply.

    Stamps the Settings-resolved model id onto the payload (reward_author uses the
    'reward_author' role; the other actions use 'dataset_agents'). These are
    DEPLOY-TIME agents: the model is baked into the deployed runtime, so the id is
    only honored once the agent in agent/ is redeployed to read payload["modelId"].
    Until then it's a no-op — but threading it now means a redeploy is all it takes.
    """
    from .agent_models import resolve_model_id

    if isinstance(payload, dict) and "modelId" not in payload:
        role = "reward_author" if payload.get("action") == "reward_author" else "dataset_agents"
        payload = {**payload, "modelId": resolve_model_id(role)}

    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    client = boto_sess.client("bedrock-agentcore", region_name=cfg.region)
    # runtimeSessionId must be >= 33 chars; a uuid4 hex (32) + prefix satisfies it.
    session_id = "investigate-" + uuid.uuid4().hex + uuid.uuid4().hex[:8]
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=_runtime_arn(),
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode("utf-8"),
    )
    body = resp["response"].read()
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    out = json.loads(body)
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(f"agent error: {out['error']}")
    return out


# --- persistence (keyed by split) --------------------------------------------

def _set_status(split_id: str, phase: str, status: str, detail: str = "") -> None:
    store = get_store()
    wd = store.workdir(INVESTIGATIONS, split_id)
    (wd / _STATUS_FILE).write_text(
        json.dumps({"phase": phase, "status": status, "detail": detail}), encoding="utf-8"
    )
    store.commit(INVESTIGATIONS, split_id)


def investigation_status(split_id: str) -> dict[str, Any]:
    raw = get_store().read_file(INVESTIGATIONS, split_id, _STATUS_FILE)
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"phase": "none", "status": "none", "detail": ""}


def load_questions(split_id: str) -> dict[str, Any] | None:
    raw = get_store().read_file(INVESTIGATIONS, split_id, _QUESTIONS_FILE)
    return json.loads(raw) if raw else None


def load_proposal(split_id: str) -> dict[str, Any] | None:
    raw = get_store().read_file(INVESTIGATIONS, split_id, _PROPOSAL_FILE)
    return json.loads(raw) if raw else None


def load_answers(split_id: str) -> dict[str, str]:
    """The user's previously-typed answers (persisted at proposal time), so the
    wizard can repopulate the textboxes on reload — letting the user see, edit,
    and regenerate a new recommendation from their prior answers."""
    raw = get_store().read_file(INVESTIGATIONS, split_id, "answers.json")
    if not raw:
        return {}
    try:
        return json.loads(raw).get("answers", {}) or {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _save(split_id: str, file_name: str, obj: dict[str, Any]) -> None:
    store = get_store()
    wd = store.workdir(INVESTIGATIONS, split_id)
    (wd / file_name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    store.commit(INVESTIGATIONS, split_id)


# --- public actions -----------------------------------------------------------

def generate_questions(split_id: str, cutoff_len: int | None = None) -> dict[str, Any]:
    """Profile the split (deterministic) → ask the agent for facet-gated questions.
    Persists + returns {questions, summary, profile}. The profile is returned too
    so the frontend can show it and pass it back unchanged to the proposal step."""
    _set_status(split_id, "questions", "running")
    log_event("investigate.questions.start", splitId=split_id)
    try:
        profile = profile_dataset(split_id, cutoff_len=cutoff_len)
        result = _invoke_runtime({"action": "questions", "profile": profile})
        payload = {
            "questions": result.get("questions", []),
            "summary": result.get("summary", ""),
            "profile": profile,
        }
        _save(split_id, _QUESTIONS_FILE, payload)
        _set_status(split_id, "questions", "done")
        log_event("investigate.questions.done", splitId=split_id,
                  n=len(payload["questions"]))
        return payload
    except Exception as e:  # noqa: BLE001
        _set_status(split_id, "questions", "failed", str(e))
        log_event("investigate.questions.error", level="ERROR", splitId=split_id, error=str(e))
        raise


def synthesize_proposal(split_id: str, answers: dict[str, str]) -> dict[str, Any]:
    """User's answers + the stored profile → confirmed config proposal (persisted).
    Reuses the profile captured at question time so the two agent calls agree."""
    _set_status(split_id, "proposal", "running")
    log_event("investigate.proposal.start", splitId=split_id)
    try:
        stored = load_questions(split_id)
        if not stored or "profile" not in stored:
            # Recompute if the questions step wasn't persisted (e.g. direct call).
            profile = profile_dataset(split_id)
        else:
            profile = stored["profile"]
        result = _invoke_runtime(
            {"action": "proposal", "profile": profile, "answers": answers}
        )
        result["splitId"] = split_id
        # Close the reward↔leaderboard-metric loop: map the recommended RANK metric
        # → the verifiable RLVR REWARD metric that mirrors it (None when the rank
        # metric isn't a per-row verifiable check, e.g. llm_judge:*). Deterministic
        # + authoritative (NOT asked of the LLM); surfaced on the FineTune RLVR step
        # so the user can reward on the same metric they're ranked on.
        rank = result.get("rankMetric")
        reward_metric = None
        try:
            from .reward_functions import reward_metric_for_rank

            reward_metric = reward_metric_for_rank(rank)
        except Exception:  # noqa: BLE001 — advisory; never fail the proposal
            reward_metric = None
        result["recommendedRewardMetric"] = reward_metric
        _save(split_id, _PROPOSAL_FILE, result)
        # Record the recommended metric(s) on the dataset so the leaderboard can
        # default its 'Rank by' to it AND the FineTune RLVR step can pre-offer the
        # matching reward (the user can still override both).
        if rank:
            try:
                from .storage import set_recommended_metric

                set_recommended_metric(
                    split_id, rank, result.get("alsoWatch", []) or [],
                    reward_metric=reward_metric,
                )
            except Exception:  # noqa: BLE001 — advisory; never fail the proposal
                pass
        # KTO analog of the reward recommendation: if the dataset is KTO-shaped and
        # class-imbalanced, persist the profiler's λ_D / λ_U so the FineTune KTO step
        # can one-click pre-fill them. Skip a balanced set (weightsBalanced) so we
        # never store a no-op 1.0/1.0. Read from the profile captured above.
        try:
            kto = (profile or {}).get("kto") or {}
            if kto and not kto.get("weightsBalanced", True):
                cw = kto.get("recommendedChosenWeight")
                rw = kto.get("recommendedRejectedWeight")
                if cw is not None and rw is not None:
                    from .storage import set_recommended_kto_weights

                    set_recommended_kto_weights(split_id, float(cw), float(rw))
        except Exception:  # noqa: BLE001 — advisory; never fail the proposal
            pass
        _set_status(split_id, "proposal", "done")
        log_event("investigate.proposal.done", splitId=split_id,
                  rankMetric=result.get("rankMetric"))
        return result
    except Exception as e:  # noqa: BLE001
        _set_status(split_id, "proposal", "failed", str(e))
        log_event("investigate.proposal.error", level="ERROR", splitId=split_id, error=str(e))
        raise


# --- async dispatch (API GW 29s timeout) -------------------------------------
# Both agent calls take ~20-28s (AgentCore + Bedrock), too close to the 29s API
# Gateway limit on the hosted path. So the endpoints dispatch to the worker
# Lambda (like baseline/judge) and the frontend polls GET /investigate. Locally
# (no worker configured) they run inline.

def start_questions(split_id: str, cutoff_len: int | None = None) -> dict[str, Any]:
    """Kick off question generation. Hosted → dispatch to worker, return running.
    Local → run inline, return the questions."""
    from .dispatch import dispatch_worker

    _set_status(split_id, "questions", "running")
    if dispatch_worker({"task": "investigate_questions", "splitId": split_id,
                        "cutoffLen": cutoff_len}):
        return {"status": "running"}
    return {"status": "done", **generate_questions(split_id, cutoff_len=cutoff_len)}


def start_proposal(split_id: str, answers: dict[str, str]) -> dict[str, Any]:
    """Kick off proposal synthesis. Hosted → dispatch; local → run inline.
    Answers are persisted first so the worker can pick them up."""
    from .dispatch import dispatch_worker

    _set_status(split_id, "proposal", "running")
    # Persist the answers so the worker (separate process) has them.
    _save(split_id, "answers.json", {"answers": answers})
    if dispatch_worker({"task": "investigate_proposal", "splitId": split_id}):
        return {"status": "running"}
    return {"status": "done", **synthesize_proposal(split_id, answers)}


def run_questions_task(split_id: str, cutoff_len: int | None = None) -> None:
    """Worker entrypoint for question generation."""
    generate_questions(split_id, cutoff_len=cutoff_len)


def run_proposal_task(split_id: str) -> None:
    """Worker entrypoint for proposal synthesis; reads persisted answers."""
    raw = get_store().read_file(INVESTIGATIONS, split_id, "answers.json")
    answers = (json.loads(raw).get("answers") if raw else {}) or {}
    synthesize_proposal(split_id, answers)


# ===========================================================================
# Failure-triage agent — diagnose a failed training/eval job + propose a fix.
# The deterministic selfheal.classify_failure runs first (cheap, reproducible)
# and is passed to the agent as a hint; the agent interprets the fuzzy log.
# ===========================================================================

def _gather_failure_context(job_name: str, model_id: str, failure_reason: str | None) -> dict[str, Any]:
    """Collect what the triage agent needs: the failure reason, a CloudWatch log
    tail, the job's instance/hyperparams, and the deterministic classification."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    context: dict[str, Any] = {"model": model_id, "failureReason": failure_reason or ""}

    # Job config (instance + hyperparameters) from the SageMaker training job.
    try:
        sm = boto_sess.client("sagemaker")
        d = sm.describe_training_job(TrainingJobName=job_name)
        context["config"] = {
            "instance": d.get("ResourceConfig", {}).get("InstanceType"),
            "hyperParameters": d.get("HyperParameters", {}),
        }
        if not failure_reason:
            context["failureReason"] = d.get("FailureReason", "") or ""
        context["failureReason"] = context["failureReason"] or d.get("FailureReason", "")
    except Exception as e:  # noqa: BLE001
        context["config"] = {"error": f"could not describe job: {e}"}

    # Log tail from CloudWatch (/aws/sagemaker/TrainingJobs, stream <job>/algo-1-*).
    try:
        logs = boto_sess.client("logs")
        grp = "/aws/sagemaker/TrainingJobs"
        streams = logs.describe_log_streams(
            logGroupName=grp, logStreamNamePrefix=job_name,
            orderBy="LastEventTime", descending=True, limit=1,
        ).get("logStreams", [])
        if streams:
            ev = logs.get_log_events(
                logGroupName=grp, logStreamName=streams[0]["logStreamName"],
                startFromHead=False, limit=40,
            ).get("events", [])
            context["logTail"] = "\n".join(e["message"].rstrip() for e in ev)[-6000:]
    except Exception as e:  # noqa: BLE001
        context["logTail"] = f"(log unavailable: {e})"

    # Deterministic classification hint (reuses the selfheal signature table).
    try:
        from .selfheal import classify_failure

        context["classification"] = classify_failure(context.get("failureReason"))
    except Exception:  # noqa: BLE001
        context["classification"] = {}
    return context


def triage_failure(job_name: str, model_id: str, failure_reason: str | None = None) -> dict[str, Any]:
    """Diagnose a failed job via the agent runtime. Returns the agent's diagnosis
    (summary/rootCause/fix/retryable/configChanges/confidence) + the gathered
    context for transparency. Advisory."""
    log_event("triage.start", job=job_name, model=model_id)
    context = _gather_failure_context(job_name, model_id, failure_reason)
    result = _invoke_runtime({"action": "triage", "context": context})
    result["_context"] = {"classification": context.get("classification"),
                          "instance": context.get("config", {}).get("instance")}
    log_event("triage.done", job=job_name, retryable=result.get("retryable"))
    return result


# --- async triage (AWS best practice for >29s on API GW: job + poll) ----------
# Triage adds describe_training_job + CloudWatch log fetch BEFORE the agent call,
# and the runtime can cold-start, so the total can exceed API Gateway's 29s. We
# run it on the worker Lambda and poll, like baseline/judge/investigation.
# Keyed by race_id + model_id (a triage belongs to one failed entry).

def _triage_key(race_id: str, model_id: str) -> str:
    return f"triage-{race_id}-{model_id}"


def start_triage(race_id: str, model_id: str, job_name: str,
                 failure_reason: str | None = None) -> dict[str, Any]:
    """Kick off triage. Hosted → worker + return running; local → inline."""
    from .dispatch import dispatch_worker

    key = _triage_key(race_id, model_id)
    _set_status(key, "triage", "running")
    # Persist the inputs so the worker (separate process) can run it.
    _save(key, "triage_input.json", {"job": job_name, "model": model_id, "reason": failure_reason})
    if dispatch_worker({"task": "investigate_triage", "raceId": race_id, "modelId": model_id}):
        return {"status": "running"}
    return {"status": "done", "result": run_triage_task(race_id, model_id)}


def run_triage_task(race_id: str, model_id: str) -> dict[str, Any]:
    """Worker entrypoint for triage; reads persisted inputs, persists the result."""
    key = _triage_key(race_id, model_id)
    raw = get_store().read_file(INVESTIGATIONS, key, "triage_input.json")
    inp = json.loads(raw) if raw else {}
    try:
        result = triage_failure(inp.get("job", ""), model_id, inp.get("reason"))
        _save(key, "triage.json", result)
        _set_status(key, "triage", "done")
        return result
    except Exception as e:  # noqa: BLE001
        _set_status(key, "triage", "failed", str(e))
        raise


def load_triage(race_id: str, model_id: str) -> dict[str, Any] | None:
    raw = get_store().read_file(INVESTIGATIONS, _triage_key(race_id, model_id), "triage.json")
    return json.loads(raw) if raw else None


def triage_status(race_id: str, model_id: str) -> dict[str, Any]:
    return investigation_status(_triage_key(race_id, model_id))


# ===========================================================================
# Results-interpreter agent — recommend which model to ship from a leaderboard.
# ===========================================================================

def interpret_results(split_id: str, priorities: str = "") -> dict[str, Any]:
    """Build the leaderboard for a split and ask the agent which model to ship."""
    from .baseline import load_all_baselines
    from .leaderboard import build_leaderboard

    log_event("interpret.start", splitId=split_id)
    leaderboard = {
        "rows": build_leaderboard(split_id),
        "baselines": load_all_baselines(split_id),
    }
    if not leaderboard["rows"]:
        return {"error": "no completed evaluations on this dataset yet"}
    result = _invoke_runtime(
        {"action": "interpret", "leaderboard": leaderboard, "priorities": priorities}
    )
    log_event("interpret.done", splitId=split_id, recommendation=result.get("recommendation"))
    return result


# --- async interpret (job + poll; same rationale as triage) -------------------
# Keyed by split_id. Priorities are persisted so the worker can read them.

def _interpret_key(split_id: str) -> str:
    return f"interpret-{split_id}"


def start_interpret(split_id: str, priorities: str = "") -> dict[str, Any]:
    """Kick off interpret. Hosted → worker + return running; local → inline."""
    from .dispatch import dispatch_worker

    key = _interpret_key(split_id)
    _set_status(key, "interpret", "running")
    _save(key, "interpret_input.json", {"priorities": priorities})
    if dispatch_worker({"task": "investigate_interpret", "splitId": split_id}):
        return {"status": "running"}
    return {"status": "done", "result": run_interpret_task(split_id)}


def run_interpret_task(split_id: str) -> dict[str, Any]:
    """Worker entrypoint for interpret; reads persisted priorities, persists result.

    The persisted result is stamped with `ranAt` (UTC ISO) + the `priorities` it was
    run with, so the UI can show "last recommendation" (when + on what priorities)
    when the user reopens the leaderboard — the recommendation survives reloads."""
    from datetime import datetime, timezone

    key = _interpret_key(split_id)
    raw = get_store().read_file(INVESTIGATIONS, key, "interpret_input.json")
    priorities = (json.loads(raw).get("priorities") if raw else "") or ""
    try:
        result = interpret_results(split_id, priorities)
        # Persisted-result metadata (advisory; the agent's own fields are untouched).
        result = {**result, "ranAt": datetime.now(timezone.utc).isoformat(),
                  "priorities": priorities}
        _save(key, "interpret.json", result)
        _set_status(key, "interpret", "done")
        return result
    except Exception as e:  # noqa: BLE001
        _set_status(key, "interpret", "failed", str(e))
        raise


def load_interpret(split_id: str) -> dict[str, Any] | None:
    raw = get_store().read_file(INVESTIGATIONS, _interpret_key(split_id), "interpret.json")
    return json.loads(raw) if raw else None


def interpret_status(split_id: str) -> dict[str, Any]:
    return investigation_status(_interpret_key(split_id))


# ===========================================================================
# Reward-prompt authoring agent — draft + calibrate an RLAIF judge rubric.
#
# ENTRY-slot agent: before an RLAIF launch, translate a plain-English goal into a
# discriminating {{prompt}}/{{response}} judge rubric and PROVE it separates good
# from bad (a real judge LLM scores fabricated candidates) before any billable
# GRPO run. The deterministic profiler computes the prompt-only profile here; the
# agent (the only tool-using one on the runtime) does the calibration loop. The
# returned draft pre-fills the authoring form — deploy stays the user's explicit,
# unchanged click. Async (multiple judge calls + AgentCore round-trip >> 29s).
# Keyed by split_id (one in-flight draft per dataset); the goal + optional prior
# result are persisted so the worker can read them.
# ===========================================================================

_REWARD_AUTHOR_FILE = "reward_author.json"


def _reward_author_key(split_id: str) -> str:
    return f"reward-author-{split_id}"


def author_reward_prompt(split_id: str, goal: str,
                         prior_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Profile the split (deterministic) → ask the agent to draft + calibrate an
    RLAIF judge rubric for `goal`. Returns the agent's draft (draftPrompt,
    rewardModelId, samples, scoreSpread, rationale, warnings, ...). Advisory — the
    user reviews + deploys through the unchanged path; nothing is launched here."""
    log_event("reward_author.start", splitId=split_id)
    profile = profile_dataset(split_id)
    payload: dict[str, Any] = {"action": "reward_author", "goal": goal, "profile": profile}
    if prior_result:
        payload["priorResult"] = prior_result
    result = _invoke_runtime(payload)
    result["splitId"] = split_id
    log_event("reward_author.done", splitId=split_id,
              discriminates=(result.get("scoreSpread") or {}).get("discriminates"),
              judgeCalls=result.get("judgeCalls"))
    return result


def start_reward_author(split_id: str, goal: str,
                        prior_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Kick off rubric authoring. Hosted → worker + return running; local → inline.
    The goal (+ optional prior result) is persisted first so the worker can read it."""
    from .dispatch import dispatch_worker

    key = _reward_author_key(split_id)
    _set_status(key, "reward_author", "running")
    _save(key, "reward_author_input.json", {"goal": goal, "priorResult": prior_result})
    if dispatch_worker({"task": "reward_author", "splitId": split_id}):
        return {"status": "running"}
    return {"status": "done", "result": run_reward_author_task(split_id)}


def run_reward_author_task(split_id: str) -> dict[str, Any]:
    """Worker entrypoint: read the persisted goal, author the rubric, persist it."""
    key = _reward_author_key(split_id)
    raw = get_store().read_file(INVESTIGATIONS, key, "reward_author_input.json")
    inp = json.loads(raw) if raw else {}
    try:
        result = author_reward_prompt(
            split_id, inp.get("goal", ""), prior_result=inp.get("priorResult"))
        _save(key, _REWARD_AUTHOR_FILE, result)
        _set_status(key, "reward_author", "done")
        return result
    except Exception as e:  # noqa: BLE001
        _set_status(key, "reward_author", "failed", str(e))
        log_event("reward_author.error", level="ERROR", splitId=split_id, error=str(e))
        raise


def load_reward_author(split_id: str) -> dict[str, Any] | None:
    raw = get_store().read_file(INVESTIGATIONS, _reward_author_key(split_id), _REWARD_AUTHOR_FILE)
    return json.loads(raw) if raw else None


def reward_author_status(split_id: str) -> dict[str, Any]:
    return investigation_status(_reward_author_key(split_id))
