# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""AWS Lambda entrypoints for the hosted deployment.

`handler`          — the FastAPI app behind API Gateway, via Mangum (the ASGI↔
                     Lambda adapter). Set as the API Lambda's image CMD.
`reconcile_handler`— invoked on a schedule (EventBridge) to advance every
                     non-terminal race one step, replacing the in-process loop
                     that a long-lived server would run. Stateless + idempotent.
`worker_handler`   — invoked ASYNCHRONOUSLY (InvocationType=Event) by the API
                     for long jobs that would exceed API Gateway's 29s timeout
                     (e.g. the Sonnet baseline over many eval rows). Dispatches
                     on the event's `task` field.

All rely on the CloudStore backend (SLM_STORAGE_BACKEND=cloud) and the Lambda
execution role for AWS access — no profiles, no local disk.
"""

from __future__ import annotations

from typing import Any

from mangum import Mangum

from .baseline import run_baseline_task
from .judge import run_judge_task
from .main import app
from .obs import log_event
from .race import TERMINAL, list_races, reconcile_race

# Mangum translates API Gateway (HTTP API v2) events to ASGI and back.
handler = Mangum(app, lifespan="off")


def reconcile_handler(event: Any = None, context: Any = None) -> dict[str, Any]:
    """Advance all non-terminal races + resolve in-flight verifications.
    Wired to EventBridge Scheduler.

    MUST walk EVERY tenant: with multi-tenancy on, each user's races live under
    users/<tenant>/races, so a tenant-agnostic list_races() (default tenant only)
    would never advance a per-user race — it would hang in TRAINING forever and the
    guided flow's completion email would never fire. Mirrors the in-process loop's
    tenant walk (main._race_reconcile_loop). With the flag off, list_tenants() is
    empty, so this reduces to the original single-tenant behaviour."""
    from .samples import SAMPLES_TENANT
    from .store import list_tenants
    from .tenancy import DEFAULT_TENANT, tenant_scope

    advanced: list[str] = []
    for tenant in [DEFAULT_TENANT, *list_tenants()]:
        if tenant == SAMPLES_TENANT:
            continue  # curated showcase — terminal + shared, never reconcile
        try:
            with tenant_scope(tenant):
                for summary in list_races():
                    states = summary.get("states", {})
                    if any(s not in TERMINAL for s in states.values()):
                        reconcile_race(summary["raceId"])
                        advanced.append(summary["raceId"])
        except Exception as e:  # noqa: BLE001 — one tenant's error mustn't stall the rest
            log_event("reconcile.tenant_failed", level="WARNING", tenant=tenant, error=str(e))
    # Advance any pending smoke-test verifications (model-catalog "verify"
    # button) to their real result — headless, so progress survives navigation.
    # Verifications are a GLOBAL root doc (tenant-agnostic), so resolve once.
    verif = {"checked": 0, "resolved": 0}
    try:
        from .verifications import resolve_pending_verifications

        verif = resolve_pending_verifications()
    except Exception as e:  # noqa: BLE001 — never let it break the race reconcile
        log_event("reconcile.verify_failed", level="WARNING", error=str(e))
    if advanced or verif["resolved"]:
        log_event("reconcile.tick", reconciled=advanced, count=len(advanced),
                  verificationsResolved=verif["resolved"])
    return {"reconciled": advanced, "count": len(advanced), "verifications": verif}


def worker_handler(event: Any = None, context: Any = None) -> dict[str, Any]:
    """Run a long task off the request path. Event shape: {"task": ..., ...}.

    Binds the tenant the dispatcher stamped into the payload so the worker reads
    the right user's state (datasets/races under users/<sub>/) and HF token — it
    has no request context of its own."""
    event = event or {}
    from .tenancy import set_tenant

    set_tenant(event.get("tenant") or None)  # None → DEFAULT_TENANT
    task = event.get("task")
    log_event("worker.start", task=task, splitId=event.get("splitId"))
    try:
        if task == "baseline":
            run_baseline_task(
                event["splitId"],
                max_new_tokens=event.get("maxNewTokens", 256),
                temperature=event.get("temperature", 0.0),
                baseline_key=event.get("baselineKey"),
            )
            log_event("worker.done", task=task, splitId=event["splitId"],
                      baselineKey=event.get("baselineKey"))
            return {"task": task, "splitId": event["splitId"], "status": "done"}
        if task == "judge":
            run_judge_task(event["evalJob"], judge_key=event.get("judgeKey"))
            log_event("worker.done", task=task, evalJob=event["evalJob"])
            return {"task": task, "evalJob": event["evalJob"], "status": "done"}
        if task == "pitcrew_launch":
            # Guided-agent race launch off the 29s API path: many models = many
            # synchronous SageMaker CreateTrainingJob calls that would time out the
            # request (and, before the raceId was persisted, made a retry launch a
            # DUPLICATE race). The worker (15-min budget) runs the launch + saves.
            from .pitcrew import run_pitcrew_launch

            run_pitcrew_launch(event["sessionId"], event["stamp"])
            log_event("worker.done", task=task, sessionId=event["sessionId"])
            return {"task": task, "sessionId": event["sessionId"], "status": "done"}
        if task == "serverless_launch":
            # Slow serverless training launch, off the 29s API path (subprocess +
            # V3 SDK + hub recipe fetch + CreateTrainingJob). Fills in the entry's
            # train_job + flips it to TRAINING (or FAILED).
            from .race import launch_serverless_entry

            launch_serverless_entry(event["raceId"], event["entryKey"], event["stamp"])
            log_event("worker.done", task=task, raceId=event["raceId"],
                      entryKey=event["entryKey"])
            return {"task": task, "raceId": event["raceId"], "status": "done"}
        if task == "deploy_reward":
            # Slow RLVR reward deploy off the 29s API path: create the reward Lambda
            # (boto3) + register the Evaluator (V3 subprocess). Flips the registry
            # entry to deployed (or records the error).
            from .reward_functions import run_deploy_reward_task

            run_deploy_reward_task(event["rewardId"])
            log_event("worker.done", task=task, rewardId=event["rewardId"])
            return {"task": task, "rewardId": event["rewardId"], "status": "done"}
        if task == "investigate_questions":
            from .investigator import run_questions_task

            run_questions_task(event["splitId"], cutoff_len=event.get("cutoffLen"))
            log_event("worker.done", task=task, splitId=event["splitId"])
            return {"task": task, "splitId": event["splitId"], "status": "done"}
        if task == "investigate_proposal":
            from .investigator import run_proposal_task

            run_proposal_task(event["splitId"])
            log_event("worker.done", task=task, splitId=event["splitId"])
            return {"task": task, "splitId": event["splitId"], "status": "done"}
        if task == "investigate_triage":
            from .investigator import run_triage_task

            run_triage_task(event["raceId"], event["modelId"])
            log_event("worker.done", task=task, raceId=event["raceId"], modelId=event["modelId"])
            return {"task": task, "status": "done"}
        if task == "investigate_interpret":
            from .investigator import run_interpret_task

            run_interpret_task(event["splitId"])
            log_event("worker.done", task=task, splitId=event["splitId"])
            return {"task": task, "splitId": event["splitId"], "status": "done"}
        if task == "reward_author":
            # RLAIF reward-prompt authoring agent: tool-using, makes several judge
            # Converse calls + an AgentCore round-trip → well over API GW's 29s.
            from .investigator import run_reward_author_task

            run_reward_author_task(event["splitId"])
            log_event("worker.done", task=task, splitId=event["splitId"])
            return {"task": task, "splitId": event["splitId"], "status": "done"}
    except Exception as e:  # noqa: BLE001 — log + re-raise so Lambda marks it failed
        log_event("worker.error", level="ERROR", task=task, error=str(e))
        raise
    log_event("worker.unknown_task", level="WARNING", task=task)
    return {"error": f"unknown task: {task}"}
