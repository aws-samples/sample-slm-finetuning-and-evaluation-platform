# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for the agentic dataset-investigation wiring (investigator.py + routes).

The actual agent reasoning runs in the AgentCore Runtime (Strands + Bedrock), so
these cover the BACKEND contract around it: routes registered, persistence,
status tracking, and the async-dispatch fallback — with the runtime call mocked.
"""
from __future__ import annotations

import json


def test_investigate_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in [
        "/api/datasets/{split_id}/investigate/questions",
        "/api/datasets/{split_id}/investigate/proposal",
        "/api/datasets/{split_id}/investigate",
        "/api/reward-functions/author",
        "/api/reward-functions/author/{split_id}",
    ]:
        assert p in paths, f"missing route {p}"


def test_questions_persist_and_status(temp_store, monkeypatch):
    """generate_questions invokes the runtime, persists, and flips status to done."""
    from app import investigator

    monkeypatch.setattr(investigator, "profile_dataset", lambda sid, cutoff_len=None: {"name": sid})
    monkeypatch.setattr(
        investigator, "_invoke_runtime",
        lambda payload: {"questions": [{"id": "q1", "question": "?"}], "summary": "s"},
    )

    out = investigator.generate_questions("split-x")
    assert out["questions"][0]["id"] == "q1"
    assert out["profile"] == {"name": "split-x"}

    # persisted + retrievable
    assert investigator.load_questions("split-x")["summary"] == "s"
    assert investigator.investigation_status("split-x") == {
        "phase": "questions", "status": "done", "detail": ""
    }


def test_proposal_reuses_stored_profile(temp_store, monkeypatch):
    """synthesize_proposal feeds the profile captured at question time + answers."""
    from app import investigator

    # Seed a questions result with a profile.
    investigator._save("split-y", investigator._QUESTIONS_FILE,
                       {"questions": [], "summary": "", "profile": {"name": "split-y", "p": 1}})

    captured = {}

    def fake_invoke(payload):
        captured.update(payload)
        return {"taskType": "label", "rankMetric": "label_accuracy", "alsoWatch": []}

    monkeypatch.setattr(investigator, "_invoke_runtime", fake_invoke)
    out = investigator.synthesize_proposal("split-y", {"q1": "ans"})

    assert captured["action"] == "proposal"
    assert captured["profile"] == {"name": "split-y", "p": 1}  # reused, not recomputed
    assert captured["answers"] == {"q1": "ans"}
    assert out["rankMetric"] == "label_accuracy"
    assert out["splitId"] == "split-y"
    # Reward↔metric loop: label_accuracy IS a verifiable per-row scorer → mirrored.
    assert out["recommendedRewardMetric"] == "label_accuracy"
    assert investigator.load_proposal("split-y")["taskType"] == "label"


def test_proposal_threads_reward_metric_and_none_case(temp_store, monkeypatch):
    """The proposal carries recommendedRewardMetric (the verifiable reward that
    mirrors the rank metric), persisted on the dataset; None when the rank metric
    isn't a per-row verifiable check (e.g. an llm_judge metric)."""
    from app import investigator, storage

    # A verifiable rank metric → mirrored reward, persisted on the dataset meta.
    storage.persist_split  # touch attr exists (import sanity)
    investigator._save("ds-num", investigator._QUESTIONS_FILE,
                       {"questions": [], "summary": "", "profile": {"name": "ds-num"}})
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: {"taskType": "numeric", "rankMetric": "numeric_match", "alsoWatch": []})
    # set_recommended_metric needs the dataset to exist; stub it to capture the call.
    captured = {}
    monkeypatch.setattr(storage, "set_recommended_metric",
                        lambda sid, rank, watch, reward_metric=None: captured.update(
                            {"sid": sid, "rank": rank, "reward": reward_metric}) or True)
    out = investigator.synthesize_proposal("ds-num", {})
    assert out["recommendedRewardMetric"] == "numeric_match"
    assert captured == {"sid": "ds-num", "rank": "numeric_match", "reward": "numeric_match"}

    # A non-verifiable rank metric (llm_judge) → no reward equivalent (None).
    investigator._save("ds-judge", investigator._QUESTIONS_FILE,
                       {"questions": [], "summary": "", "profile": {"name": "ds-judge"}})
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: {"taskType": "text", "rankMetric": "llm_judge:overall", "alsoWatch": []})
    out2 = investigator.synthesize_proposal("ds-judge", {})
    assert out2["recommendedRewardMetric"] is None
    assert captured["reward"] is None  # persisted as None too


def test_runtime_error_surfaces(temp_store, monkeypatch):
    from app import investigator

    monkeypatch.setattr(investigator, "profile_dataset", lambda sid, cutoff_len=None: {})
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        investigator.generate_questions("split-z")
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert investigator.investigation_status("split-z")["status"] == "failed"


def test_runtime_arn_resolves_by_name_not_hardcoded(temp_store, monkeypatch):
    """The ARN must be resolved by the runtime's stable NAME (suffix is random per
    deploy), with an explicit override winning. No hardcoded suffix."""
    from app import investigator

    investigator._arn_cache.clear()
    # No override → resolve by name via the (mocked) control plane.
    monkeypatch.delenv("SLM_AGENT_RUNTIME_ARN", raising=False)
    monkeypatch.setattr(
        investigator, "_resolve_arn_by_name",
        lambda name: f"arn:aws:bedrock-agentcore:us-east-1:111:runtime/{name}-ABC123",
    )

    class _Cfg:
        agent_runtime_arn = None

    monkeypatch.setattr(investigator, "load_aws_config", lambda: _Cfg())
    arn = investigator._runtime_arn()
    assert arn.endswith("dataset_investigator-ABC123")  # resolved, not hardcoded
    # cached on second call (no second resolve needed)
    assert investigator._arn_cache["dataset_investigator"] == arn

    # Explicit env override wins.
    investigator._arn_cache.clear()
    monkeypatch.setenv("SLM_AGENT_RUNTIME_ARN", "arn:aws:bedrock-agentcore:us-east-1:222:runtime/custom-XYZ")
    assert investigator._runtime_arn().endswith("custom-XYZ")


def test_start_runs_inline_without_worker(temp_store, monkeypatch):
    """With no worker configured (local), start_* run inline and return results."""
    from app import investigator

    monkeypatch.delenv("SLM_WORKER_FUNCTION", raising=False)
    monkeypatch.setattr(investigator, "profile_dataset", lambda sid, cutoff_len=None: {"name": sid})
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: {"questions": [], "summary": "ok"})

    out = investigator.start_questions("split-inline")
    assert out["status"] == "done"
    assert out["summary"] == "ok"


def test_interpret_result_persists_with_ranat_and_priorities(temp_store, monkeypatch):
    """The interpret recommendation is persisted AND stamped with ranAt + the
    priorities it was run with, so the leaderboard can show 'what you ran last
    time' on reload (load_interpret returns the same stamped result)."""
    from app import investigator

    import app.leaderboard as lb
    import app.baseline as bl

    monkeypatch.delenv("SLM_WORKER_FUNCTION", raising=False)
    # interpret_results imports these locally from their source modules.
    monkeypatch.setattr(lb, "build_leaderboard", lambda sid: [{"model": "m"}])
    monkeypatch.setattr(bl, "load_all_baselines", lambda sid: [])
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: {"recommendation": "qwen3-1.7b", "reasoning": "best F1/cost"})

    out = investigator.start_interpret("split-z", "cost matters most")
    res = out["result"]
    assert res["recommendation"] == "qwen3-1.7b"
    # stamped metadata for "last run" display
    assert res["priorities"] == "cost matters most"
    assert "ranAt" in res and res["ranAt"]  # UTC ISO timestamp
    # survives a reload via the persisted store
    loaded = investigator.load_interpret("split-z")
    assert loaded["recommendation"] == "qwen3-1.7b"
    assert loaded["priorities"] == "cost matters most"
    assert loaded["ranAt"] == res["ranAt"]
    assert investigator.interpret_status("split-z")["status"] == "done"


# --- reward-prompt authoring agent wiring ------------------------

def test_reward_author_invokes_runtime_with_goal_and_profile(temp_store, monkeypatch):
    """author_reward_prompt profiles the split + invokes the runtime with the
    reward_author action, the goal, and the prompt-only profile."""
    from app import investigator

    monkeypatch.setattr(investigator, "profile_dataset",
                        lambda sid, cutoff_len=None: {"name": sid, "shape": "rlaif"})
    captured = {}

    def fake_invoke(payload):
        captured.update(payload)
        return {"draftPrompt": "Rate {{prompt}} {{response}}", "rewardModelId": "qwen.qwen3-32b-v1:0",
                "scoreSpread": {"goodMean": 0.9, "badMean": 0.2, "separation": 0.7, "discriminates": True},
                "samples": [], "rationale": [], "warnings": [], "iterations": 2}

    monkeypatch.setattr(investigator, "_invoke_runtime", fake_invoke)
    out = investigator.author_reward_prompt("ds-rlaif", "reward concise tone")

    assert captured["action"] == "reward_author"
    assert captured["goal"] == "reward concise tone"
    assert captured["profile"] == {"name": "ds-rlaif", "shape": "rlaif"}
    assert "priorResult" not in captured  # none passed
    assert out["splitId"] == "ds-rlaif"
    assert out["scoreSpread"]["discriminates"] is True


def test_reward_author_threads_prior_result(temp_store, monkeypatch):
    """A prior_result (regenerate-with-feedback) is forwarded to the runtime."""
    from app import investigator

    monkeypatch.setattr(investigator, "profile_dataset", lambda sid, cutoff_len=None: {"name": sid})
    captured = {}
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: captured.update(p) or {"draftPrompt": "x"})
    investigator.author_reward_prompt("ds", "goal", prior_result={"draftPrompt": "old"})
    assert captured["priorResult"] == {"draftPrompt": "old"}


def test_start_reward_author_inline_persists_and_status(temp_store, monkeypatch):
    """With no worker (local), start_reward_author runs inline, persists the draft,
    and flips status to done; load_reward_author returns it; status reports done."""
    from app import investigator

    monkeypatch.delenv("SLM_WORKER_FUNCTION", raising=False)
    monkeypatch.setattr(investigator, "profile_dataset", lambda sid, cutoff_len=None: {"name": sid})
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: {"draftPrompt": "Rate {{prompt}} {{response}}",
                                   "rewardModelId": "", "scoreSpread": None,
                                   "samples": [], "rationale": [], "warnings": []})

    out = investigator.start_reward_author("ds-1", "reward friendly tone")
    assert out["status"] == "done"
    assert out["result"]["draftPrompt"] == "Rate {{prompt}} {{response}}"
    # The goal was persisted for the worker before running.
    raw = __import__("json").loads(
        investigator.get_store().read_file(
            investigator.INVESTIGATIONS, investigator._reward_author_key("ds-1"),
            "reward_author_input.json"))
    assert raw["goal"] == "reward friendly tone"
    # Result persisted + retrievable + status done.
    assert investigator.load_reward_author("ds-1")["draftPrompt"] == "Rate {{prompt}} {{response}}"
    assert investigator.reward_author_status("ds-1")["status"] == "done"


def test_start_reward_author_dispatches_to_worker(temp_store, monkeypatch):
    """With a worker configured, start dispatches (running) instead of inline, and
    run_reward_author_task reads the persisted goal to do the work."""
    from app import dispatch, investigator

    monkeypatch.setenv("SLM_WORKER_FUNCTION", "worker-fn")
    dispatched = {}
    monkeypatch.setattr(dispatch, "dispatch_worker",
                        lambda payload: dispatched.update(payload) or True)
    out = investigator.start_reward_author("ds-2", "reward terse answers")
    assert out == {"status": "running"}
    assert dispatched == {"task": "reward_author", "splitId": "ds-2"}

    # The worker task reads the persisted goal + invokes the runtime.
    monkeypatch.setattr(investigator, "profile_dataset", lambda sid, cutoff_len=None: {"name": sid})
    captured = {}
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: captured.update(p) or {"draftPrompt": "Rate {{prompt}} {{response}}"})
    res = investigator.run_reward_author_task("ds-2")
    assert captured["goal"] == "reward terse answers"
    assert res["draftPrompt"] == "Rate {{prompt}} {{response}}"


def test_reward_author_failure_sets_failed_status(temp_store, monkeypatch):
    """A runtime error during the worker task flips status to failed (so the poll
    surfaces it rather than hanging)."""
    from app import investigator

    monkeypatch.setattr(investigator, "profile_dataset", lambda sid, cutoff_len=None: {"name": sid})
    investigator._save(investigator._reward_author_key("ds-bad"),
                       "reward_author_input.json", {"goal": "g", "priorResult": None})
    monkeypatch.setattr(investigator, "_invoke_runtime",
                        lambda p: (_ for _ in ()).throw(RuntimeError("agent boom")))
    try:
        investigator.run_reward_author_task("ds-bad")
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert investigator.reward_author_status("ds-bad")["status"] == "failed"
