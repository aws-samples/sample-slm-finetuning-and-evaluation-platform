# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Spot + checkpoint/resume — cost accounting + render wiring."""
from __future__ import annotations

import yaml

from app import leaderboard as lb
from app.aws_config import spot_discount_factor
from app.catalog import Hyperparams, get_model
from app.render import render_train_yaml, render_export_yaml, CHECKPOINT_DIR


class _FakeSm:
    """describe_training_job returns a fixed job; list/resolve short-circuit."""

    def __init__(self, billable, instance, spot):
        self._d = {
            "TrainingJobStatus": "Completed",
            "BillableTimeInSeconds": billable,
            "ResourceConfig": {"InstanceType": instance},
            "EnableManagedSpotTraining": spot,
        }

    def describe_training_job(self, TrainingJobName):  # noqa: N803
        return self._d


def test_spot_cost_discounted_and_flagged():
    # On-demand g5.2xlarge = $1.515/hr; 3600s billable → $1.515 on-demand.
    on_demand = lb._train_cost(_FakeSm(3600, "ml.g5.2xlarge", spot=False), "slm-x-abc")
    assert on_demand["trainSpot"] is False
    assert on_demand["trainCostIsEstimate"] is False
    assert abs(on_demand["trainCostUsd"] - 1.515) < 1e-6

    # Same job on spot: cost is discounted by the spot factor + flagged estimate.
    spot = lb._train_cost(_FakeSm(3600, "ml.g5.2xlarge", spot=True), "slm-x-abc")
    assert spot["trainSpot"] is True
    assert spot["trainCostIsEstimate"] is True
    assert abs(spot["trainCostUsd"] - 1.515 * spot_discount_factor()) < 1e-3
    # Spot must be cheaper than on-demand for the same billable time.
    assert spot["trainCostUsd"] < on_demand["trainCostUsd"]


def test_render_writes_checkpoint_dir_for_every_job():
    """Checkpointing is now UNIFIED: every job (spot AND on-demand) writes to the
    SageMaker-synced checkpoint dir with FULL optimizer state, so a failed
    on-demand run can be resumed from its last checkpoint (not just spot reclaims).
    Previously on-demand wrote to the adapter dir with save_only_model=True."""
    model = get_model("qwen3-1.7b")
    hp = Hyperparams()
    for spot in (False, True):
        train = yaml.safe_load(render_train_yaml(model, hp, "s", use_spot=spot))
        exp = yaml.safe_load(render_export_yaml(model, "s", use_spot=spot))
        assert train["output_dir"] == CHECKPOINT_DIR, f"use_spot={spot}"
        assert exp["adapter_name_or_path"] == CHECKPOINT_DIR, f"use_spot={spot}"
        # full checkpoint (optimizer/scheduler/RNG) so a resume can continue mid-epoch
        assert train["save_only_model"] is False, f"use_spot={spot}"


def test_discount_factor_in_unit_range():
    f = spot_discount_factor()
    assert 0.0 < f < 1.0  # spot is a fraction of on-demand
