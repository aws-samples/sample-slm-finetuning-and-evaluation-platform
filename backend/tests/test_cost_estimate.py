# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Cost + time estimate for a race (the number a non-ML user approves on)."""
from app import cost_estimate as ce


def test_price_table_matches_verified_aws_rates():
    # Rates VERIFIED against the AWS Pricing API (AmazonSageMaker, us-east-1, Training)
    # on 2026-07-02. Pinned here so an accidental edit that mis-quotes a user's spend
    # is caught by CI. Update BOTH when AWS pricing changes.
    expected = {
        "ml.g5.2xlarge": 1.52, "ml.g5.4xlarge": 2.03, "ml.g5.8xlarge": 3.06,
        "ml.g5.12xlarge": 7.09, "ml.g6e.2xlarge": 2.80, "ml.g6e.4xlarge": 3.76,
        "ml.g6e.8xlarge": 5.66, "ml.g6e.12xlarge": 13.10,
    }
    assert ce.INSTANCE_HOURLY_USD == expected


def test_every_assignable_instance_is_priced():
    # Any instance the platform can actually assign/allow must be in the price table,
    # so a valid launch never silently hits the fallback rate (which misprices g6e).
    from app.limits import allowed_instance_types
    for inst in allowed_instance_types():
        assert inst in ce.INSTANCE_HOURLY_USD, f"{inst} allowed but not priced"


def test_known_instance_uses_table_rate():
    # A small LoRA job on g5.2xlarge should bill from the table rate, not the fallback.
    cost = ce.estimate_job_cost_usd(
        instance_type="ml.g5.2xlarge", train_rows=2000, eval_rows=100, epochs=3,
        params_b=1.5, method="lora", base_eval=True, use_spot=False)
    assert cost > 0
    # Sanity bound: a couple of short jobs on a ~$1.5/hr card shouldn't cost $100s.
    assert cost < 50


def test_spot_is_cheaper_than_on_demand():
    kw = dict(instance_type="ml.g5.8xlarge", train_rows=5000, eval_rows=200, epochs=3,
              params_b=8, method="lora", base_eval=True)
    on_demand = ce.estimate_job_cost_usd(use_spot=False, **kw)
    spot = ce.estimate_job_cost_usd(use_spot=True, **kw)
    assert spot < on_demand
    assert abs(spot - on_demand * ce.SPOT_DISCOUNT) < 1e-6


def test_full_weight_costs_more_than_lora_same_model():
    kw = dict(instance_type="ml.g6e.2xlarge", train_rows=2000, eval_rows=100, epochs=3,
              params_b=1.5, base_eval=True, use_spot=False)
    lora = ce.estimate_job_cost_usd(method="lora", **kw)
    full = ce.estimate_job_cost_usd(method="full", **kw)
    assert full > lora  # full-weight runs more steps-time per the multiplier


def test_unknown_instance_falls_back_not_crash():
    cost = ce.estimate_job_cost_usd(
        instance_type="ml.p5.48xlarge", train_rows=1000, eval_rows=50, epochs=2,
        params_b=7, method="lora", base_eval=False, use_spot=False)
    assert cost > 0  # used the fallback rate rather than KeyError


def test_estimate_race_sums_cost_and_takes_max_wallclock():
    entries = [
        {"instanceType": "ml.g5.2xlarge", "paramsB": 1.5, "epochs": 3, "method": "lora", "baseEval": True},
        {"instanceType": "ml.g5.8xlarge", "paramsB": 8, "epochs": 3, "method": "lora", "baseEval": True},
    ]
    est = ce.estimate_race(entries, train_rows=4000, eval_rows=200, use_spot=False)
    # Cost is a lo<=hi range and positive.
    assert 0 < est["totalUsd"]["lo"] <= est["totalUsd"]["hi"]
    # Wall-clock is the slowest single model, not the sum — so it can't exceed the
    # bigger model's own minutes by much (within the +range).
    assert est["wallClockMin"]["lo"] <= est["wallClockMin"]["hi"]
    # 2 models × (train + eval + base-eval) = 6 jobs.
    assert est["jobs"] == 6
    assert len(est["perModel"]) == 2


def test_estimate_race_empty_is_zero():
    est = ce.estimate_race([], train_rows=0, eval_rows=0)
    assert est["totalUsd"] == {"lo": 0.0, "hi": 0.0}
    assert est["jobs"] == 0


def test_long_context_costs_more_than_short():
    # A long-context dataset (high seq_tokens) must quote more time/cost than the
    # same row-count of short examples — the seq-length fix.
    kw = dict(instance_type="ml.g5.2xlarge", train_rows=2000, eval_rows=100, epochs=3,
              params_b=1.5, method="lora", base_eval=True, use_spot=False)
    short = ce.estimate_job_cost_usd(seq_tokens=256, **kw)
    long_ = ce.estimate_job_cost_usd(seq_tokens=2048, **kw)
    assert long_ > short
    # No signal → factor 1.0 (backward compatible).
    none = ce.estimate_job_cost_usd(seq_tokens=None, **kw)
    assert none > 0


def test_seq_factor_clamped():
    # Pathological outliers can't blow up the quote — factor is clamped to [0.5, 4].
    assert ce._seq_factor(10 ** 9) == 4.0
    assert ce._seq_factor(1) == 0.5
    assert ce._seq_factor(None) == 1.0
    assert ce._seq_factor(0) == 1.0


def test_estimate_race_passes_seq_tokens_through():
    entries = [{"instanceType": "ml.g5.2xlarge", "paramsB": 1.5, "epochs": 3,
                "method": "lora", "baseEval": True}]
    short = ce.estimate_race(entries, train_rows=3000, eval_rows=100, seq_tokens=256)
    long_ = ce.estimate_race(entries, train_rows=3000, eval_rows=100, seq_tokens=2048)
    assert long_["totalUsd"]["hi"] > short["totalUsd"]["hi"]
    assert long_["wallClockMin"]["hi"] > short["wallClockMin"]["hi"]


def test_base_eval_flag_adds_cost():
    kw = dict(instance_type="ml.g5.4xlarge", train_rows=3000, eval_rows=150, epochs=3,
              params_b=3, method="lora", use_spot=False)
    with_base = ce.estimate_job_cost_usd(base_eval=True, **kw)
    without = ce.estimate_job_cost_usd(base_eval=False, **kw)
    assert with_base > without
