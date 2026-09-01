# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Bedrock AgentCore Runtime entrypoint for the dataset-investigation agent.

AgentCore Runtime POSTs a JSON payload to the @app.entrypoint function and
returns its JSON result. This agent is STATELESS: the platform backend computes
the deterministic profile (profiler.py) and passes it in, so the runtime needs no
data access of its own — only Bedrock InvokeModel (Sonnet) for the reasoning.

Payload contract:
  {"action": "questions", "profile": {...}}
      -> {"questions": [...], "summary": "..."}
  {"action": "proposal",  "profile": {...}, "answers": {"q1": "...", ...}}
      -> {"taskType","rankMetric","alsoWatch","cutoffGuidance",
          "flaggedIssues","rationale","appliedAnswers"}
  {"action": "triage",    "context": {...}}      -> diagnosis + fix
  {"action": "interpret", "leaderboard": {...}}  -> which model to ship
  {"action": "reward_author", "goal": "...", "profile": {...}, "priorResult"?: {...}}
      -> {"draftPrompt","rewardModelId","samples","scoreSpread","rationale",
          "iterations","judgeCalls","warnings"}   (tool-using; scores candidates
          with a real judge LLM and iterates — the only multi-turn agent)

Local dev:   uv run agent.py   then POST to http://localhost:8080/invocations
Deploy:      agentcore configure --entrypoint agent.py  &&  agentcore launch

This runtime is deployed on its own, not by the CDK app, so the launch has to
pass USER_AGENT_STRING (the solution's user-agent suffix, built from the fields
in infra/project_config.json) for its Bedrock calls to be attributed to the
solution. Without it the agent works exactly as before, just unmeasured.
"""
from __future__ import annotations

import os
import sys

# Make the src-layout package importable whether the container runs this file
# directly or as a module (AgentCore CodeBuild copies the project as-is).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from dataset_investigator.core import (
    author_reward_prompt,
    generate_questions,
    interpret_results,
    set_reasoning_model,
    synthesize_proposal,
    triage_failure,
)

app = BedrockAgentCoreApp()

# Region for Bedrock calls — AgentCore sets AWS_REGION in the runtime; default to
# us-east-1 (where the platform + Sonnet inference profile live).
REGION = os.environ.get("AWS_REGION", "us-east-1")


@app.entrypoint
def invoke(payload: dict) -> dict:
    payload = payload or {}
    action = payload.get("action")
    # Honor the platform-selected reasoning model (Settings → AI agents). Set per
    # request and reset to the default when absent, so a choice never leaks across
    # invocations. Only affects the Strands reasoning agents (not the RLAIF judge,
    # which validates its own open-weight model set in judge_tools).
    set_reasoning_model(payload.get("modelId"))

    # Dataset-investigation actions (need a profile).
    if action in ("questions", "proposal"):
        profile = payload.get("profile")
        if not isinstance(profile, dict):
            return {"error": "payload must include a 'profile' object from the profiler"}
        if action == "questions":
            return generate_questions(profile, region=REGION)
        answers = payload.get("answers") or {}
        if not isinstance(answers, dict):
            return {"error": "'proposal' action requires an 'answers' object"}
        return synthesize_proposal(profile, answers, region=REGION)

    # Failure-triage action — diagnose a failed job + propose a fix.
    if action == "triage":
        context = payload.get("context")
        if not isinstance(context, dict):
            return {"error": "'triage' action requires a 'context' object"}
        return triage_failure(context, region=REGION)

    # Results-interpreter action — which model to ship.
    if action == "interpret":
        leaderboard = payload.get("leaderboard")
        if not isinstance(leaderboard, dict):
            return {"error": "'interpret' action requires a 'leaderboard' object"}
        return interpret_results(leaderboard, payload.get("priorities", ""), region=REGION)

    # Reward-prompt authoring action — draft + calibrate an RLAIF judge rubric.
    # The FIRST tool-using agent (it scores candidates with a real judge LLM and
    # iterates); needs a free-text `goal` + a prompt-only dataset `profile`.
    if action == "reward_author":
        goal = payload.get("goal")
        profile = payload.get("profile")
        if not isinstance(goal, str) or not goal.strip():
            return {"error": "'reward_author' action requires a non-empty 'goal' string"}
        if not isinstance(profile, dict):
            return {"error": "'reward_author' action requires a 'profile' object from the profiler"}
        return author_reward_prompt(
            goal, profile, prior_result=payload.get("priorResult"), region=REGION
        )

    return {
        "error": f"unknown action '{action}'; expected "
        "questions|proposal|triage|interpret|reward_author"
    }


if __name__ == "__main__":
    # Local dev server on :8080 (same contract AgentCore Runtime uses).
    app.run()
