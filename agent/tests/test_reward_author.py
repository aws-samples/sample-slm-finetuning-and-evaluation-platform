# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for the reward-prompt authoring agent's load-bearing logic — the parts
that are OURS, not the LLM's: result extraction, the hard judge-call cap, and the
server-side reconciliation (the spread is recomputed from the REAL captured scores,
not trusted from the model).

The Strands Agent is mocked so no Bedrock is touched: a fake Agent drives the tool
loop (set_rubric → score_candidate ×N) the way a real agent would, then returns a
final fenced-JSON reply. We monkeypatch the per-call judge so try_reward_prompt
never hits AWS.

Run: AWS_PROFILE=… uv run -m pytest tests/ -q   (or just `uv run -m pytest`)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataset_investigator.core as core  # noqa: E402
from dataset_investigator import judge_tools  # noqa: E402

RUBRIC = (
    'Rate {{prompt}} and {{response}} for concision + friendliness — reply ONLY '
    'JSON {"score": 0..1, "reasoning": "..."}'
)


def _final(**fields) -> str:
    """Build the agent's final fenced-JSON reply the way a real LLM would — with
    json.dumps so the rubric's own quotes/braces are properly escaped (a hand-
    concatenated string would be invalid JSON)."""
    return "```json\n" + json.dumps(fields) + "\n```"


def _stub_judge(score: float):
    """A stub Converse client returning a fixed score, so score_candidate doesn't
    hit AWS."""

    class _C:
        def converse(self, **_kw):
            return {"output": {"message": {"content": [
                {"text": f'{{"score": {score}, "reasoning": "stub"}}'}]}}}

    return _C()


class _FakeAgent:
    """Stand-in for a Strands Agent that USES its tools like a real one would, then
    emits a final fenced-JSON result. The `script` callback receives the tool map
    and drives the loop; `final` is the reply string returned from __call__."""

    def __init__(self, *, model, system_prompt, tools):
        # Strands DecoratedFunctionTool exposes the python name via __wrapped__;
        # key the tool map by that so the script can call tools["set_rubric"] etc.
        self.tools = {getattr(getattr(t, "__wrapped__", t), "__name__", str(i)): t
                      for i, t in enumerate(tools)}

    def __call__(self, _prompt):
        return type(self)._script(self.tools)


def _install_fake_agent(monkeypatch, judge_score, script):
    """Point core.Agent at a fake whose tool-loop is `script` (a fn taking the tool
    map and returning the final reply string). All judge calls return judge_score.

    core's score tool calls the import-time binding `core._try_reward_prompt_pure`,
    so we patch THAT to route through the real vendored fn with a stub Converse
    client (exercises the real fill/parse/clamp path, no AWS)."""
    monkeypatch.setattr(core, "BedrockModel", lambda **kw: object())
    monkeypatch.setattr(
        core, "_try_reward_prompt_pure",
        lambda rubric, p, r, jm, region="us-east-1": judge_tools.try_reward_prompt(
            rubric, p, r, jm, region=region, _client=_stub_judge(judge_score)),
    )
    _FakeAgent._script = staticmethod(script)
    monkeypatch.setattr(core, "Agent", _FakeAgent)


PROFILE = {"name": "rlaif-demo", "shape": "rlaif",
           "rlaif": {"rows": 100, "promptTask": "text", "promptTaskMix": {"text": 100}}}


def test_spread_recomputed_from_real_scores_not_model_claim(monkeypatch):
    """The agent claims a bogus spread; the server reconciles it from the ACTUAL
    judge scores captured by the score_candidate tool."""

    def script(tools):
        tools["set_rubric"](RUBRIC)
        # good candidates score 0.9 (stub), bad also 0.9 here (no real separation),
        # but the model LIES in its final JSON that separation is huge.
        tools["score_candidate"]("p1", "good answer", "good", "qwen.qwen3-32b-v1:0")
        tools["score_candidate"]("p2", "bad answer", "bad", "qwen.qwen3-32b-v1:0")
        return _final(
            draftPrompt=RUBRIC, rewardModelId="qwen.qwen3-32b-v1:0",
            scoreSpread={"goodMean": 0.99, "badMean": 0.01, "separation": 0.98,
                         "discriminates": True},
            samples=[], rationale=["x"], iterations=1, warnings=[])

    _install_fake_agent(monkeypatch, judge_score=0.9, script=script)
    out = core.author_reward_prompt("reward concise tone", PROFILE)

    # Recomputed from the real (equal) scores → separation 0.0, NOT the model's 0.98.
    assert out["scoreSpread"]["separation"] == 0.0
    assert out["scoreSpread"]["discriminates"] is False
    assert out["judgeCalls"] == 2
    # The samples returned are the REAL captured ones, not the model's empty list.
    assert len(out["samples"]) == 2
    # A weak-separation warning is surfaced.
    assert any("separate" in w or "separation" in w for w in out["warnings"])


def test_real_separation_is_reported(monkeypatch):
    """When good genuinely beats bad, discriminates is True (spread from real scores)."""
    calls = {"n": 0}

    def stub_factory(rubric, p, r, jm, region="us-east-1"):
        # good labels score high, bad low — driven by the candidate text marker.
        calls["n"] += 1
        score = 0.9 if "GOOD" in r else 0.1
        return judge_tools.try_reward_prompt(rubric, p, r, jm, region=region,
                                             _client=_stub_judge(score))

    monkeypatch.setattr(core, "BedrockModel", lambda **kw: object())
    monkeypatch.setattr(core, "_try_reward_prompt_pure", stub_factory)

    def script(tools):
        tools["set_rubric"](RUBRIC)
        tools["score_candidate"]("p", "GOOD reply", "good", "qwen.qwen3-32b-v1:0")
        tools["score_candidate"]("p", "GOOD reply 2", "good", "qwen.qwen3-32b-v1:0")
        tools["score_candidate"]("p", "bad reply", "bad", "qwen.qwen3-32b-v1:0")
        return _final(draftPrompt=RUBRIC, rewardModelId="qwen.qwen3-32b-v1:0", iterations=2)

    _FakeAgent._script = staticmethod(script)
    monkeypatch.setattr(core, "Agent", _FakeAgent)

    out = core.author_reward_prompt("reward concise tone", PROFILE)
    assert out["scoreSpread"]["goodMean"] == 0.9
    assert out["scoreSpread"]["badMean"] == 0.1
    assert out["scoreSpread"]["discriminates"] is True
    assert out["iterations"] == 2


def test_judge_call_cap_is_enforced_in_code(monkeypatch):
    """Beyond the hard cap, score_candidate refuses (returns STOP) and does not call
    the judge — so a runaway loop can't burn billable Converse calls."""

    def script(tools):
        tools["set_rubric"](RUBRIC)
        results = []
        # Try to score far more than the cap.
        for i in range(core._REWARD_AUTHOR_MAX_JUDGE_CALLS + 10):
            label = "good" if i % 2 == 0 else "bad"
            results.append(tools["score_candidate"](f"p{i}", f"r{i}", label, "qwen.qwen3-32b-v1:0"))
        # The calls past the cap must be refused.
        assert any("STOP" in r for r in results), "cap not enforced"
        return _final(draftPrompt=RUBRIC)

    _install_fake_agent(monkeypatch, judge_score=0.5, script=script)
    out = core.author_reward_prompt("goal", PROFILE)
    # judge_calls never exceeds the cap.
    assert out["judgeCalls"] == core._REWARD_AUTHOR_MAX_JUDGE_CALLS
    assert any("cap" in w for w in out["warnings"])


def test_invalid_judge_id_in_final_json_falls_back(monkeypatch):
    """A hallucinated judge id in the final result is rejected → '' (recipe default)
    with a warning; a valid id passes through."""

    def script(tools):
        tools["set_rubric"](RUBRIC)
        tools["score_candidate"]("p", "g", "good", "")
        tools["score_candidate"]("p", "b", "bad", "")
        return _final(draftPrompt=RUBRIC, rewardModelId="us.amazon.nova-pro-v1:0")

    _install_fake_agent(monkeypatch, judge_score=0.5, script=script)
    out = core.author_reward_prompt("goal", PROFILE)
    assert out["rewardModelId"] == ""  # bad id dropped
    assert any("unsupported judge model" in w for w in out["warnings"])


def test_missing_placeholder_draft_is_flagged(monkeypatch):
    """If the agent never set a valid rubric and its final draftPrompt lacks
    placeholders, the result carries a validation warning (can't reach deploy)."""

    def script(tools):
        # never call set_rubric with a valid rubric; emit a placeholder-less draft.
        return _final(draftPrompt="rate it somehow", rewardModelId="")

    _install_fake_agent(monkeypatch, judge_score=0.5, script=script)
    out = core.author_reward_prompt("goal", PROFILE)
    assert any("validation" in w or "placeholder" in w for w in out["warnings"])
    # No scored candidates → also warns it wasn't calibrated.
    assert any("no judge-scored" in w for w in out["warnings"])
