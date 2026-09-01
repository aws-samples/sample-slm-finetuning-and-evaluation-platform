# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Cost guardrails."""
import pytest

from app.limits import LimitExceeded, check_race_launch, limits_summary


def test_within_limits_ok():
    # default: 16 models (guided "Thorough" ceiling), g5 instances, 5 concurrent
    check_race_launch(num_models=3, instance_types=["ml.g5.2xlarge"], active_race_count=0)


def test_too_many_models_rejected():
    # Cap raised to 16 to match the guided "Thorough" up-to-16 ceiling.
    with pytest.raises(LimitExceeded, match="limit is 16"):
        check_race_launch(num_models=17, instance_types=["ml.g5.2xlarge"], active_race_count=0)


def test_disallowed_instance_rejected():
    with pytest.raises(LimitExceeded, match="not allowed"):
        check_race_launch(num_models=1, instance_types=["ml.p4d.24xlarge"], active_race_count=0)


def test_too_many_concurrent_rejected():
    with pytest.raises(LimitExceeded, match="already running"):
        check_race_launch(num_models=1, instance_types=["ml.g5.2xlarge"], active_race_count=5)


def test_env_override(monkeypatch):
    monkeypatch.setenv("SLM_MAX_MODELS_PER_RACE", "2")
    with pytest.raises(LimitExceeded, match="limit is 2"):
        check_race_launch(num_models=3, instance_types=["ml.g5.2xlarge"], active_race_count=0)


def test_summary_shape():
    s = limits_summary()
    assert s["maxModelsPerRace"] >= 1
    assert s["maxConcurrentRaces"] >= 1
    assert s["maxGlobalConcurrentRaces"] >= 0  # 0 = disabled (default)
    assert isinstance(s["allowedInstanceTypes"], list)


# --- optional GLOBAL (cross-tenant) concurrency cap -------------------------

def test_global_cap_disabled_by_default():
    """With the global cap unset (0), even a large cross-tenant count is allowed —
    the per-tenant cap is the only concurrency guardrail by default."""
    from app.limits import max_global_concurrent_races
    assert max_global_concurrent_races() == 0
    # A huge global count must NOT raise when the cap is disabled.
    check_race_launch(num_models=1, instance_types=["ml.g5.2xlarge"],
                      active_race_count=0, global_active_race_count=9999)


def test_global_cap_enforced_when_enabled(monkeypatch):
    """When SLM_MAX_GLOBAL_CONCURRENT_RACES is set, the cross-tenant total is capped
    on top of the per-tenant limit."""
    monkeypatch.setenv("SLM_MAX_GLOBAL_CONCURRENT_RACES", "10")
    # Under the global cap → fine (per-tenant count well under its own limit too).
    check_race_launch(num_models=1, instance_types=["ml.g5.2xlarge"],
                      active_race_count=1, global_active_race_count=9)
    # At/over the global cap → rejected with the platform-wide message.
    with pytest.raises(LimitExceeded, match="across all users"):
        check_race_launch(num_models=1, instance_types=["ml.g5.2xlarge"],
                          active_race_count=1, global_active_race_count=10)


def test_global_cap_skipped_when_count_none(monkeypatch):
    """Even with the cap enabled, passing global_active_race_count=None (the caller
    didn't run the opt-in scan) must NOT raise — the global check is a no-op."""
    monkeypatch.setenv("SLM_MAX_GLOBAL_CONCURRENT_RACES", "1")
    check_race_launch(num_models=1, instance_types=["ml.g5.2xlarge"],
                      active_race_count=0, global_active_race_count=None)


def test_count_global_active_races_sums_across_tenants(temp_store, monkeypatch):
    """count_global_active_races sums non-terminal races across every tenant, skips
    the samples namespace, and never double-counts."""
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams
    from app.tenancy import tenant_scope

    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    monkeypatch.setattr(rm, "launch_training_job", lambda **kw: {"jobName": "x"})
    monkeypatch.setattr(rm, "launch_base_eval_job", lambda **kw: {"jobName": "b"})

    # Two distinct tenants each start one (active/training) race.
    for t in ("tenant-a", "tenant-b"):
        with tenant_scope(t):
            rm.start_race("abc123def456", [rm.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())],
                          DecodingParams(), f"20260630-{t}")
    # Cross-tenant total sees BOTH (called outside any single tenant scope).
    assert rm.count_global_active_races() >= 2
    # Per-tenant count for one tenant sees only its own.
    with tenant_scope("tenant-a"):
        assert rm._count_active_races() == 1


def test_race_launch_blocked_by_limit(temp_store, monkeypatch):
    """Integration: start_race rejects an over-limit race before launching jobs."""
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams

    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    launched = []
    monkeypatch.setattr(rm, "launch_training_job", lambda **kw: launched.append(kw) or {"jobName": "x"})
    monkeypatch.setenv("SLM_MAX_MODELS_PER_RACE", "1")

    models = [rm.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams()),
              rm.RaceModel(model_id="qwen3-0.6b", hp=Hyperparams())]
    with pytest.raises(LimitExceeded):
        rm.start_race("s", models, DecodingParams(), "20260603-1")
    assert launched == []  # NO billable job launched when the limit is breached


def test_every_catalog_instance_is_allowed():
    """Guardrail/catalog coupling: every model's suggested_instance must be in the
    default allowlist, else that model is blocked at launch. Regression for the
    g6e sizing change (9B+ models moved off g5)."""
    from app.limits import allowed_instance_types
    from app.catalog import list_models
    allowed = allowed_instance_types()
    bad = sorted({m["suggestedInstance"] for m in list_models()
                  if m["suggestedInstance"] not in allowed})
    assert not bad, f"catalog suggests instances not in the allowlist: {bad}"
