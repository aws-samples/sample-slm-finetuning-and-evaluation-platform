# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LLM hyperparameter advisor (Tier 3) — clamp, fallback, JSON extraction."""
from app import advisor
from app.catalog import Hyperparams, get_model


def test_clamp_keeps_values_in_bounds():
    base = Hyperparams()
    b = Hyperparams.bounds()
    # Wildly out-of-range LLM output is clamped, not passed through.
    hp = advisor._clamp_hp({"loraRank": 99999, "learningRate": 5.0, "numTrainEpochs": 999}, base)
    assert hp.lora_rank <= b["loraRank"]["max"]
    assert hp.learning_rate <= b["learningRate"]["max"]
    assert hp.num_train_epochs <= b["numTrainEpochs"]["max"]


def test_clamp_falls_back_on_garbage():
    base = Hyperparams(lora_rank=16, learning_rate=1e-4)
    hp = advisor._clamp_hp({"loraRank": "not-a-number"}, base)
    assert hp.lora_rank == 16  # kept the baseline when the value was unparseable


def test_clamp_preserves_derived_save_steps_and_es():
    # The advisor varies LR/rank/epochs but must NOT clobber the carefully
    # derived save_steps or the early-stopping settings from the baseline.
    base = Hyperparams(save_steps=130, early_stopping_enabled=True, early_stopping_patience=3)
    hp = advisor._clamp_hp({"learningRate": 2e-4}, base)
    assert hp.save_steps == 130
    assert hp.early_stopping_enabled is True and hp.early_stopping_patience == 3


def test_extract_json_tolerates_prose_and_fences():
    assert advisor._extract_json('blah {"configs": [1,2]} trailing') == {"configs": [1, 2]}
    assert advisor._extract_json("no json here") == {}


def test_deterministic_sweep_varies_lr_and_is_distinct():
    model = get_model("qwen3-4b")
    base = Hyperparams(learning_rate=1e-4)
    sweep = advisor._deterministic_sweep(model, base)
    lrs = [c["hyperparams"]["learningRate"] for c in sweep]
    assert len(set(lrs)) == 3  # three distinct learning rates
    assert min(lrs) < 1e-4 < max(lrs)  # brackets the baseline


def test_advise_route_registered():
    import app.main as m

    assert "/api/advise" in {r.path for r in m.app.routes}
