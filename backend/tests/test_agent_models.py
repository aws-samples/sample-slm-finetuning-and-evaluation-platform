# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Per-agent Bedrock model selection — resolver, override persistence, validation.

The store is pointed at a temp dir by the `temp_store` fixture, so set_overrides
writes config.json under tmp and never touches real state.
"""
import pytest

from app import agent_models
from app.baseline import DEFAULT_BASELINE


# --- defaults / resolver -----------------------------------------------------

def test_unset_role_resolves_to_its_default(temp_store):
    # No override saved → each role tracks its declared default.
    assert agent_models.resolve_baseline_key("advisor") == "sonnet-4-5"
    assert agent_models.resolve_baseline_key("judge") == DEFAULT_BASELINE


def test_unknown_role_falls_back_to_default_baseline(temp_store):
    assert agent_models.resolve_baseline_key("does-not-exist") == DEFAULT_BASELINE


def test_resolve_model_id_returns_concrete_bedrock_id(temp_store):
    mid = agent_models.resolve_model_id("advisor")
    assert mid == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


# --- override round-trip -----------------------------------------------------

def test_override_persists_and_resolves(temp_store):
    agent_models.set_overrides({"advisor": "opus-4-8"})
    assert agent_models.resolve_baseline_key("advisor") == "opus-4-8"
    assert agent_models.resolve_model_id("advisor") == "us.anthropic.claude-opus-4-8"
    # Other roles are untouched by an advisor override.
    assert agent_models.resolve_baseline_key("judge") == DEFAULT_BASELINE


def test_setting_back_to_default_clears_the_override(temp_store):
    agent_models.set_overrides({"advisor": "opus-4-8"})
    assert agent_models.resolve_baseline_key("advisor") == "opus-4-8"
    # Picking the role default again clears the stored override (tracks default).
    agent_models.set_overrides({"advisor": "sonnet-4-5"})
    assert agent_models._overrides() == {}
    assert agent_models.resolve_baseline_key("advisor") == "sonnet-4-5"


def test_empty_value_clears_the_override(temp_store):
    agent_models.set_overrides({"selfheal": "haiku-4-5"})
    assert agent_models.resolve_baseline_key("selfheal") == "haiku-4-5"
    agent_models.set_overrides({"selfheal": ""})
    assert "selfheal" not in agent_models._overrides()


def test_multiple_roles_independent(temp_store):
    agent_models.set_overrides({"advisor": "opus-4-8", "judge": "nova-pro"})
    assert agent_models.resolve_baseline_key("advisor") == "opus-4-8"
    assert agent_models.resolve_baseline_key("judge") == "nova-pro"
    assert agent_models.resolve_baseline_key("selfheal") == "sonnet-4-5"


# --- validation --------------------------------------------------------------

def test_unknown_role_rejected(temp_store):
    with pytest.raises(ValueError, match="unknown agent role"):
        agent_models.set_overrides({"not-a-role": "opus-4-8"})


def test_unknown_model_rejected(temp_store):
    with pytest.raises(ValueError, match="unknown model"):
        agent_models.set_overrides({"advisor": "gpt-9-ultra"})


# --- settings view -----------------------------------------------------------

def test_settings_view_shape(temp_store):
    v = agent_models.settings_view()
    assert {r["key"] for r in v["roles"]} == {
        "advisor", "selfheal", "judge", "dataset_agents", "reward_author", "pitcrew"
    }
    # Every role exposes a resolved selection + the deploy-time flag.
    for r in v["roles"]:
        assert r["selectedKey"] and r["selectedModelId"] and r["selectedLabel"]
        assert isinstance(r["deployTime"], bool)
    # The dataset/reward-author agents are deploy-time; the in-process ones aren't.
    by_key = {r["key"]: r for r in v["roles"]}
    assert by_key["dataset_agents"]["deployTime"] is True
    assert by_key["reward_author"]["deployTime"] is True
    assert by_key["advisor"]["deployTime"] is False
    assert by_key["judge"]["deployTime"] is False
    # The model registry is offered as choices.
    assert any(m["key"] == "opus-4-8" for m in v["models"])
    assert v["default"] == DEFAULT_BASELINE


def test_settings_view_reflects_override(temp_store):
    agent_models.set_overrides({"judge": "opus-4-8"})
    v = agent_models.settings_view()
    judge = next(r for r in v["roles"] if r["key"] == "judge")
    assert judge["selectedKey"] == "opus-4-8"
    assert judge["selectedLabel"] == "Claude Opus 4.8"
