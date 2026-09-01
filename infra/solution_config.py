# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Single source of truth for the solution's identity and deployment properties.

Everything metrics-relevant about this solution — its ID, its published name, its
version — is declared ONCE in `project_config.json` and read through here. Two
completely independent measurement pipelines consume those three fields, and
neither one can be fixed after a release goes out:

  1. DEPLOYMENT COUNTING keys off the CloudFormation stack Description and
     nothing else. The collector matches descriptions containing "(SO", so the
     ID must appear PARENTHESISED — see `stack_description`. No resource tag,
     no stack name, and no metadata block substitutes for it.
  2. API-USAGE ATTRIBUTION keys off a user-agent suffix on every AWS SDK call —
     see `user_agent_string`. That value is threaded to the compute that makes
     the calls as the USER_AGENT_STRING environment variable.

Scattering the ID or the version as literals is what makes the two signals drift
apart at release time, so both are derived here and nowhere else.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "project_config.json"

# CloudFormation's hard limit on a stack Description. Exceeding it fails the
# deploy at CreateStack — long after synth — so `stack_description` asserts it
# here instead, where the failure is a one-line message at synth time.
_MAX_DESCRIPTION_BYTES = 1024


@lru_cache(maxsize=1)
def _config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _solution() -> dict:
    return _config().get("solution", {}) or {}


def solution_id() -> str:
    """The assigned solution ID, e.g. "SO0363"."""
    return str(_solution().get("id", "")).strip()


def solution_name() -> str:
    """The published solution name, as a customer sees it."""
    return str(_solution().get("name", "")).strip()


def solution_version() -> str:
    """The release version, e.g. "v1.0.0". Rendered into BOTH signals."""
    return str(_solution().get("version", "")).strip()


def stack_description(component: str = "") -> str:
    """Build a metrics-countable CloudFormation stack Description.

    The main stack carries the BARE id:

        (SO0363) - SLM Finetuning and Evaluation Platform. Version v1.0.0

    A supporting stack in the same install carries a SUFFIXED id, so that one
    install of an N-stack app counts as one deployment rather than N:

        (SO0363-core) - SLM Finetuning and Evaluation Platform - core. Version v1.0.0

    Pass `component` only for a supporting stack. Exactly ONE stack per install
    may take the bare form, and it must be the stack at the TOP of the
    dependency graph — a foundation stack can be left behind by an abandoned
    deploy, and counting that as a complete install overstates adoption.

    A standalone template that is NOT part of an install (anything created at
    runtime by the application rather than by this app) must not carry the ID at
    all, so it must not call this function.
    """
    component = (component or "").strip()
    ident = f"{solution_id()}-{component}" if component else solution_id()
    suffix = f" - {component}" if component else ""
    description = (
        f"({ident}) - {solution_name()}{suffix}. Version {solution_version()}"
    )

    # Fail loudly at synth rather than shipping a description the collector
    # cannot match or CloudFormation will not accept.
    if "(SO" not in description:
        raise ValueError(
            f"solution id {solution_id()!r} does not produce a countable "
            f"description; expected an 'SO'-prefixed id in project_config.json"
        )
    encoded = len(description.encode("utf-8"))
    if encoded > _MAX_DESCRIPTION_BYTES:
        raise ValueError(
            f"stack description is {encoded} bytes, over CloudFormation's "
            f"{_MAX_DESCRIPTION_BYTES}-byte limit: {description!r}"
        )
    return description


def user_agent_string() -> str:
    """The user-agent suffix that attributes an AWS API call to this solution.

    Format is fixed by the collector: "AWSSOLUTION/<id>/<version>". The slashes
    are load-bearing — a mangled separator makes the call unmatchable — which is
    why this is appended as a raw user-agent extra rather than through any SDK
    "app id" mechanism, all of which sanitise the value.
    """
    return f"AWSSOLUTION/{solution_id()}/{solution_version()}"


def _notifications() -> dict:
    return _config().get("notifications", {}) or {}


def notify_from_email() -> str:
    """Verified SES sender for race-completion email. Empty = sending skipped."""
    return str(_notifications().get("from_email", "")).strip()


def notify_admin_email() -> str:
    """Address CC'd on race-completion email. Empty = no admin copy."""
    return str(_notifications().get("admin_email", "")).strip()
