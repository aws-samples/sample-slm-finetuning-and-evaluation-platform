# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Drift guard for the VENDORED judge helpers.

The reward-author agent (agent/src/dataset_investigator/judge_tools.py) carries
its OWN copy of validate_reward_prompt / _fill_reward_prompt / _parse_judge_score
/ try_reward_prompt + the ALLOWED_JUDGE_MODELS and _DRY_RUN_FALLBACK_JUDGE
constants, because the AgentCore Runtime container does NOT import the backend
`app` package (see judge_tools.py module docstring).

This test imports BOTH copies and asserts they agree on a fixed fixture set, so
the two can never silently diverge — an edit to one without the other fails here.
The agent module is loaded by PATH (it only imports json/re/math at module load;
boto3 is lazy), so the backend test venv can load it without `strands`.

Follow-up: consolidate the two into one importable module shared by backend + agent.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

# Load the vendored agent copy by file path (no strands import at module load).
_VENDORED_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "agent" / "src" / "dataset_investigator" / "judge_tools.py"
)


def _load_vendored():
    spec = importlib.util.spec_from_file_location("vendored_judge_tools", _VENDORED_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUBRIC = (
    'Rate the response to {{prompt}}: {{response}} — reply with JSON '
    '{"score": 0..1, "reasoning": "..."}'
)


def test_allowlist_and_fallback_constants_match():
    """The judge allowlist + the dry-run fallback judge must be byte-identical."""
    from app import reward_functions as backend

    vend = _load_vendored()
    assert tuple(vend.ALLOWED_JUDGE_MODELS) == tuple(backend.ALLOWED_JUDGE_MODELS)
    assert vend._DRY_RUN_FALLBACK_JUDGE == backend._DRY_RUN_FALLBACK_JUDGE


@pytest.mark.parametrize(
    "prompt_text",
    [
        RUBRIC,
        "no placeholders at all",
        "only {{prompt}} present",
        "only {{response}} present",
        "   ",  # blank
        "whitespace tolerant {{ prompt }} and {{  response  }}",
    ],
)
def test_validate_agrees(prompt_text):
    """Both copies accept/reject the SAME rubrics (raising vs not raising)."""
    from app import reward_functions as backend

    vend = _load_vendored()
    backend_ok = _raises(lambda: backend.validate_reward_prompt(prompt_text))
    vend_ok = _raises(lambda: vend.validate_reward_prompt(prompt_text))
    assert backend_ok == vend_ok, f"validate disagreement on {prompt_text!r}"


@pytest.mark.parametrize(
    "tmpl,pr,resp",
    [
        ("P={{prompt}} R={{response}}", "hello", "world"),
        ("P={{ prompt }} R={{response}}", "a", "b"),
        ("no placeholders", "x", "y"),
        ("{{prompt}}{{prompt}}{{response}}", "p", "r"),
    ],
)
def test_fill_agrees(tmpl, pr, resp):
    """{{prompt}}/{{response}} substitution is identical."""
    from app import reward_functions as backend

    vend = _load_vendored()
    assert backend._fill_reward_prompt(tmpl, pr, resp) == vend._fill_reward_prompt(tmpl, pr, resp)


@pytest.mark.parametrize(
    "reply",
    [
        '{"score": 0.7, "reasoning": "ok"}',
        'prose then {"score": 1.5, "reasoning": "clamp"} trailing',  # clamp >1
        '{"score": -3, "reasoning": "clamp low"}',  # clamp <0
        '{"score": "NaN", "reasoning": "bad number"}',
        "no json at all",
        '{"reasoning": "no score key"}',
        '{"score": 0.42}',  # missing reasoning
        "not json {oops",
    ],
)
def test_parse_judge_score_agrees(reply):
    """The judge-reply parser yields the SAME (score, reasoning, error) tuple."""
    from app import reward_functions as backend

    vend = _load_vendored()
    assert backend._parse_judge_score(reply) == vend._parse_judge_score(reply)


class _StubConverse:
    """A stub bedrock-runtime client returning a fixed judge reply."""

    def __init__(self, text: str):
        self._text = text

    def converse(self, **_kw):
        return {"output": {"message": {"content": [{"text": self._text}]}}}


def test_try_reward_prompt_agrees_with_stub_judge():
    """End-to-end with a stubbed Converse client: both copies fill the rubric, parse
    the reply, and return the same score/reasoning/error. (The backend takes the
    client via `_client`; the vendored copy via `_client`.)"""
    from app import reward_functions as backend

    vend = _load_vendored()
    reply = '{"score": 0.83, "reasoning": "concise + friendly"}'
    b = backend.try_reward_prompt(RUBRIC, "How do I reset?", "Sure! Click reset.",
                                  "qwen.qwen3-32b-v1:0", _client=_StubConverse(reply))
    v = vend.try_reward_prompt(RUBRIC, "How do I reset?", "Sure! Click reset.",
                               "qwen.qwen3-32b-v1:0", _client=_StubConverse(reply))
    assert b == v == {"score": 0.83, "reasoning": "concise + friendly", "error": None}


def test_try_reward_prompt_rejects_bad_judge_id_in_both():
    """A non-allowlisted judge id raises in BOTH copies BEFORE any Converse call."""
    from app import reward_functions as backend

    vend = _load_vendored()
    with pytest.raises(backend.RewardError):
        backend.try_reward_prompt(RUBRIC, "p", "r", "us.amazon.nova-pro-v1:0",
                                  _client=_StubConverse("{}"))
    with pytest.raises(vend.RewardPromptError):
        vend.try_reward_prompt(RUBRIC, "p", "r", "us.amazon.nova-pro-v1:0",
                               _client=_StubConverse("{}"))


def test_try_reward_prompt_never_raises_on_judge_failure_in_both():
    """A judge that raises yields a graceful error result (score 0.0), never an
    uncaught exception — in both copies."""
    from app import reward_functions as backend

    vend = _load_vendored()

    class _Boom:
        def converse(self, **_kw):
            raise RuntimeError("throttled")

    b = backend.try_reward_prompt(RUBRIC, "p", "r", "qwen.qwen3-32b-v1:0", _client=_Boom())
    v = vend.try_reward_prompt(RUBRIC, "p", "r", "qwen.qwen3-32b-v1:0", _client=_Boom())
    assert b["score"] == v["score"] == 0.0
    assert b["error"] and v["error"]


def _raises(fn) -> bool:
    """True if calling fn() does NOT raise (i.e. it validated ok)."""
    try:
        fn()
        return True
    except Exception:
        return False
