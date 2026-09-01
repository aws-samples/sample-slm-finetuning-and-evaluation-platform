# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Per-agent Bedrock model selection — the SINGLE source of truth for which model
each of the app's AI agents uses, and the resolver that lets the Settings page
override any of them at runtime (persisted to config.json, no redeploy).

Why this exists
---------------
The app has several AI "agents" (LLM-backed helpers). Each used to hardcode its
own model id in its own module, so there was no way to swap, say, the run-config
advisor onto Opus without editing code. This module collects those roles, gives
each a stable KEY, a human label, a default model, and the set of models it can
validly use — and a resolver that layers a saved override on top.

Two classes of agent, with different reach
------------------------------------------
  * IN-PROCESS agents call Bedrock directly from the backend (advisor, self-heal
    classifier, eval judge, RLAIF reward judge). An override here takes effect on
    the NEXT call — no redeploy. These honor the resolver live.
  * DEPLOYED agents run on the AgentCore runtime (the dataset investigate / triage
    / results-interpret / reward-author actions in agent/). Their model is baked
    into the deployed runtime image, and the invoke payload carries no model. We
    thread the resolved id into the payload so a redeployed runtime CAN honor it,
    but until the agent is redeployed to read it, the override is advisory. The
    Settings UI labels these "applies after the agent is redeployed".

Model choices
-------------
The valid models reuse baseline.BASELINE_MODELS (the Converse-invokable registry
already used by the judge + baselines), so adding a model in one place lights it
up everywhere. The RLAIF reward judge is the exception: its valid set is the SDK's
ALLOWED_REWARD_MODEL_IDS (open-weight judges only), not the Claude/Nova registry.
"""

from __future__ import annotations

from typing import Any

from .aws_config import _saved
from .baseline import BASELINE_MODELS, DEFAULT_BASELINE

# config.json key under which the {roleKey: baselineKey} override map is stored.
_CONFIG_KEY = "agentModels"


# Each agent role the user can configure. `key` is the stable id (config + API),
# `default` is a BASELINE_MODELS key (or "" for the reward judge = recipe default),
# `choices` names where the valid options come from, and `deployTime` flags the
# AgentCore-runtime agents whose override only lands after a redeploy.
AGENT_ROLES: list[dict[str, Any]] = [
    {
        "key": "advisor",
        "label": "Run-config advisor",
        "description": "Proposes hyperparameter sweeps to race (Fine-Tune → 'Suggest configs').",
        "default": "sonnet-4-5",
        "choices": "baseline",
        "deployTime": False,
    },
    {
        "key": "selfheal",
        "label": "Self-heal classifier",
        "description": "Classifies why a model failed verification ('Why did this fail?').",
        "default": "sonnet-4-5",
        "choices": "baseline",
        "deployTime": False,
    },
    {
        "key": "judge",
        "label": "Evaluation judge (default)",
        "description": "LLM-as-judge for eval scoring. Can still be overridden per eval run.",
        "default": DEFAULT_BASELINE,
        "choices": "baseline",
        "deployTime": False,
    },
    {
        "key": "dataset_agents",
        "label": "Dataset agents (investigate / triage / interpret)",
        "description": "AgentCore reasoning agents at the dataset + results boundaries.",
        "default": "sonnet-4-5",
        "choices": "baseline",
        "deployTime": True,
    },
    {
        "key": "reward_author",
        "label": "Reward-prompt author",
        "description": "Drafts + calibrates an RLAIF judge rubric ('Draft with AI').",
        "default": "sonnet-4-5",
        "choices": "baseline",
        "deployTime": True,
    },
    {
        "key": "pitcrew",
        "label": "Guided Fine-tuning narrator",
        "description": "Writes the plain-language guidance in the Guided Fine-tuning chat. "
                       "Narration only — never decides configs (the plan is rule-based).",
        "default": "sonnet-4-5",
        "choices": "baseline",
        "deployTime": False,
    },
]

_ROLE_BY_KEY = {r["key"]: r for r in AGENT_ROLES}


def _overrides() -> dict[str, str]:
    """The saved {roleKey: baselineKey} override map (empty if none set)."""
    raw = _saved().get(_CONFIG_KEY)
    return raw if isinstance(raw, dict) else {}


def resolve_baseline_key(role_key: str) -> str:
    """The BASELINE_MODELS key chosen for a role: saved override (if it names a
    valid baseline) → the role's default. Unknown role → DEFAULT_BASELINE."""
    role = _ROLE_BY_KEY.get(role_key)
    default = role["default"] if role else DEFAULT_BASELINE
    chosen = _overrides().get(role_key)
    if chosen and chosen in BASELINE_MODELS:
        return chosen
    return default


def resolve_model_id(role_key: str) -> str:
    """The concrete Bedrock model id for a role (resolved key → BASELINE_MODELS).
    Falls back to the default baseline's id if the chosen key somehow isn't in the
    registry, so a caller always gets an invokable id."""
    key = resolve_baseline_key(role_key)
    spec = BASELINE_MODELS.get(key) or BASELINE_MODELS[DEFAULT_BASELINE]
    return spec["modelId"]


def settings_view() -> dict[str, Any]:
    """Payload for the Settings page: each role with its label/description, the
    resolved selection (key + id + label), whether it's a deploy-time agent, and
    the list of selectable models. The model list is the shared baseline registry."""
    models = [
        {"key": k, "label": v["label"], "provider": v["provider"], "modelId": v["modelId"]}
        for k, v in BASELINE_MODELS.items()
    ]
    roles = []
    for r in AGENT_ROLES:
        sel_key = resolve_baseline_key(r["key"])
        sel = BASELINE_MODELS.get(sel_key, BASELINE_MODELS[DEFAULT_BASELINE])
        roles.append({
            "key": r["key"],
            "label": r["label"],
            "description": r["description"],
            "deployTime": r["deployTime"],
            "selectedKey": sel_key,
            "selectedLabel": sel["label"],
            "selectedModelId": sel["modelId"],
        })
    return {"roles": roles, "models": models, "default": DEFAULT_BASELINE}


def set_overrides(updates: dict[str, str]) -> dict[str, Any]:
    """Persist agent-model overrides. Each value must be a known BASELINE_MODELS
    key for a known role; a value equal to the role default (or empty) CLEARS that
    role's override so it tracks the default again. Returns the fresh settings view.

    Raises ValueError on an unknown role or an invalid model key so a bad pick
    can't be silently swallowed (surfaced as a 400 by the endpoint)."""
    from .aws_config import _invalidate_config_cache
    from .store import get_store

    current = dict(_overrides())
    for role_key, model_key in (updates or {}).items():
        if role_key not in _ROLE_BY_KEY:
            raise ValueError(f"unknown agent role '{role_key}'")
        # Clear (track default) when the pick is empty or equals the role default.
        if not model_key or model_key == _ROLE_BY_KEY[role_key]["default"]:
            current.pop(role_key, None)
            continue
        if model_key not in BASELINE_MODELS:
            raise ValueError(f"unknown model '{model_key}' for role '{role_key}'")
        current[role_key] = model_key

    # Write the whole map under the single config key. save_config skips None/empty
    # values but a dict is preserved; we write the document directly to also allow
    # CLEARING back to {} (save_config's merge would never remove keys).
    doc = _saved(use_cache=False)
    if current:
        doc[_CONFIG_KEY] = current
    else:
        doc.pop(_CONFIG_KEY, None)
    get_store().write_root_json("config.json", doc)
    _invalidate_config_cache()
    return settings_view()
