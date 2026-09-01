# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Cost guardrails for a multi-user hosted deployment.

Each race launches one GPU training job (and later one eval job) PER model, so
the spend risk is: too many models in one race, too many races running at once,
or an oversized instance. These limits cap that. They're advisory defaults tuned
for a dev/prototyping account and overridable by env var so a deployment can
loosen or tighten them without code changes.

Enforced at launch (race.start_race); a violation raises LimitExceeded, which
the API surfaces as a 400 with a clear message — never a silent, expensive
surprise.
"""
from __future__ import annotations

import os


class LimitExceeded(ValueError):
    """A launch was rejected by a cost guardrail (surfaced as HTTP 400)."""


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Max models in a single race (= number of parallel GPU training jobs). The guided
# "Thorough" ceiling is up to 16, so this must be ≥16 or the planner's ceiling would be
# silently clamped below what the UI offers. This bounds ONE race; the per-tenant and
# global concurrent-race caps below are what protect account-wide GPU spend, so raising
# this does not weaken cross-tenant isolation.
def max_models_per_race() -> int:
    return _int_env("SLM_MAX_MODELS_PER_RACE", 16)


# Max races with at least one non-terminal entry at once (each holds GPU jobs).
# Enforced PER TENANT: the state store partitions races under users/<tenant>/, so
# the active-race count start_race passes is already scoped to the current user.
def max_concurrent_races() -> int:
    return _int_env("SLM_MAX_CONCURRENT_RACES", 5)


# OPTIONAL global (cross-tenant) ceiling on simultaneously-active races, an account
# blast-radius backstop for a multi-user deployment: the per-tenant cap above bounds
# ONE user, but N users each at their cap can still flood the account's GPU quota.
# 0 (the default) = DISABLED: no global cap, and start_race does NOT perform the
# cross-tenant scan at all (zero added cost/latency). Set a positive value to cap the
# total number of active races across every tenant.
def max_global_concurrent_races() -> int:
    return _int_env("SLM_MAX_GLOBAL_CONCURRENT_RACES", 0)


# Instance types allowed for training/eval jobs (denylist everything pricier).
# Comma-separated env override; defaults to the g5 + g6e families the catalog uses.
def allowed_instance_types() -> set[str]:
    raw = os.environ.get(
        "SLM_ALLOWED_INSTANCE_TYPES",
        # g5 (A10G 24 GB) for ≤8B + g6e (L40S 48 GB) for 9B+ — the size-aware set
        # catalog._instance_for assigns. Keep in sync: a model whose suggested
        # instance isn't allowed here would be blocked at launch.
        "ml.g5.2xlarge,ml.g5.4xlarge,ml.g5.8xlarge,ml.g5.12xlarge,"
        "ml.g6e.2xlarge,ml.g6e.4xlarge,ml.g6e.8xlarge,ml.g6e.12xlarge",
    )
    return {s.strip() for s in raw.split(",") if s.strip()}


def check_race_launch(num_models: int, instance_types: list[str], active_race_count: int,
                      global_active_race_count: int | None = None) -> None:
    """Raise LimitExceeded if a proposed race would breach a guardrail.

    `active_race_count` is the CURRENT TENANT's active races (the store is
    tenant-partitioned, so this is naturally per-user). `global_active_race_count`
    is the OPTIONAL cross-tenant total: pass it only when the global cap is enabled
    (max_global_concurrent_races() > 0); None skips the global check entirely."""
    mmax = max_models_per_race()
    if num_models > mmax:
        raise LimitExceeded(
            f"race has {num_models} models; the limit is {mmax} per race "
            f"(set SLM_MAX_MODELS_PER_RACE to change)"
        )

    allowed = allowed_instance_types()
    bad = sorted({it for it in instance_types if it not in allowed})
    if bad:
        raise LimitExceeded(
            f"instance type(s) not allowed: {', '.join(bad)}. "
            f"Allowed: {', '.join(sorted(allowed))} "
            f"(set SLM_ALLOWED_INSTANCE_TYPES to change)"
        )

    cmax = max_concurrent_races()
    if active_race_count >= cmax:
        raise LimitExceeded(
            f"{active_race_count} races already running; the limit is {cmax} "
            f"concurrent (wait for one to finish, or set SLM_MAX_CONCURRENT_RACES)"
        )

    # Optional global (cross-tenant) ceiling. Only checked when enabled AND a count
    # was supplied — None means the caller didn't run the (opt-in) cross-tenant scan.
    gmax = max_global_concurrent_races()
    if gmax > 0 and global_active_race_count is not None and global_active_race_count >= gmax:
        raise LimitExceeded(
            f"{global_active_race_count} races are running across all users; the "
            f"platform-wide limit is {gmax} concurrent. Please wait for one to finish "
            f"(an administrator can raise SLM_MAX_GLOBAL_CONCURRENT_RACES)."
        )


def limits_summary() -> dict:
    """Current effective limits (for display on the Settings page)."""
    return {
        "maxModelsPerRace": max_models_per_race(),
        "maxConcurrentRaces": max_concurrent_races(),
        # 0 = no global cap (disabled). Surfaced so operators can see whether the
        # cross-tenant backstop is active on this deployment.
        "maxGlobalConcurrentRaces": max_global_concurrent_races(),
        "allowedInstanceTypes": sorted(allowed_instance_types()),
    }
