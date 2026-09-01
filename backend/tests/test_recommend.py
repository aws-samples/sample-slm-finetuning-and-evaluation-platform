# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic hyperparameter recommender (Tier 2)."""
from app.catalog import Hyperparams, get_model
from app.recommend import suggest_config


def _rec(model_id, rows, has_val):
    return suggest_config(get_model(model_id), rows, has_val)


def test_deterministic_same_inputs_same_output():
    a = _rec("qwen3-4b", 5375, True)
    b = _rec("qwen3-4b", 5375, True)
    assert a.hp == b.hp  # dataclass equality — no RNG/LLM


def test_save_steps_gives_several_evals_per_epoch():
    # 5375 rows, effective batch 8 → ~672 steps/epoch. save_steps must be small
    # enough to evaluate several times per epoch (the early-stopping footgun).
    rec = _rec("qwen3-4b", 5375, True)
    steps_per_epoch = 5375 / 8
    evals = steps_per_epoch / rec.hp.save_steps
    assert 3 <= evals <= 10, f"got {evals:.1f} evals/epoch (save_steps={rec.hp.save_steps})"
    # Never the coarse default 500 that breaks ES on small data.
    assert rec.hp.save_steps < 500


def test_early_stopping_gated_on_val():
    assert _rec("qwen3-4b", 5375, True).hp.early_stopping_enabled is True
    assert _rec("qwen3-4b", 5375, False).hp.early_stopping_enabled is False


def test_rank_scales_with_dataset_size():
    assert _rec("qwen3-4b", 500, True).hp.lora_rank == 8       # small
    assert _rec("qwen3-4b", 5375, True).hp.lora_rank == 16     # medium
    assert _rec("qwen3-4b", 50000, True).hp.lora_rank == 32    # large


def test_sft_lora_lr_is_size_independent():
    # LoRA LR is objective-driven, NOT size-driven (arXiv:2602.04998): two SFT LoRA
    # runs at different model sizes get the SAME standard 2e-4 (the old size gate was
    # mis-conditioned).
    small = _rec("qwen3-1.7b", 5375, True).hp.learning_rate
    big = _rec("qwen3-4b", 5375, True).hp.learning_rate
    assert small == big == 2.0e-4


def test_preference_objectives_get_much_lower_lr():
    # DPO/KTO are far more LR-sensitive than SFT — they must NOT inherit the SFT-scale
    # LoRA LR (the ~30× too-hot footgun). ~5e-6, an order of magnitude below SFT.
    sft = suggest_config(get_model("qwen2.5-1.5b"), 5375, True, objective="sft").hp.learning_rate
    dpo = suggest_config(get_model("qwen2.5-1.5b"), 5375, True, objective="dpo").hp.learning_rate
    kto = suggest_config(get_model("qwen2.5-1.5b"), 5375, True, objective="kto").hp.learning_rate
    assert sft == 2.0e-4
    assert dpo == kto == 5.0e-6
    assert dpo < sft / 10  # decisively lower, not a nudge


def test_full_weight_lr_unaffected_by_objective():
    # full/freeze keep their own full-weight LR clamp (~1e-5) regardless of objective.
    rec = suggest_config(get_model("qwen2.5-1.5b"), 5375, True,
                         finetuning_type="full", objective="sft")
    assert rec.hp.learning_rate == Hyperparams.FULL_WEIGHT_DEFAULT_LR


def test_all_values_within_bounds():
    b = Hyperparams.bounds()
    for rows in (100, 5375, 100000):
        hp = _rec("qwen3-4b", rows, True).hp
        assert b["loraRank"]["min"] <= hp.lora_rank <= b["loraRank"]["max"]
        assert b["learningRate"]["min"] <= hp.learning_rate <= b["learningRate"]["max"]
        assert b["numTrainEpochs"]["min"] <= hp.num_train_epochs <= b["numTrainEpochs"]["max"]
        assert b["saveSteps"]["min"] <= hp.save_steps <= b["saveSteps"]["max"]


def test_rationale_covers_every_decision():
    rec = _rec("qwen3-4b", 5375, True)
    fields = {r["field"] for r in rec.rationale}
    for f in ("loraRank", "learningRate", "numTrainEpochs", "saveSteps", "earlyStopping"):
        assert f in fields
    # every rationale entry has a non-empty reason
    assert all(r["reason"] for r in rec.rationale)


def test_recommend_route_registered():
    import app.main as m

    assert "/api/recommend" in {r.path for r in m.app.routes}


# --- Tier 1: card-aware (architecture from model_card.fetch_arch) ------------

def test_no_arch_falls_back_to_deterministic():
    """Absent/empty arch → identical to the pure size×dataset heuristic (the card
    fetch is best-effort; a miss must never change the recommendation)."""
    base = suggest_config(get_model("qwen3-4b"), 5375, True)
    with_empty = suggest_config(get_model("qwen3-4b"), 5375, True, arch={})
    assert base.hp == with_empty.hp


def test_freeze_layers_capped_by_card_layer_count():
    # Large dataset would normally pick 8 freeze layers; a model the card says has
    # only 4 transformer layers must cap below that (leave ≥1 frozen → still 'freeze').
    rec = suggest_config(get_model("qwen3-4b"), 50000, True,
                         finetuning_type="freeze", arch={"numHiddenLayers": 4})
    assert rec.hp.freeze_trainable_layers == 3  # max(1, 4-1)
    fr = next(r for r in rec.rationale if r["field"] == "freezeTrainableLayers")
    assert "model card" in fr["reason"] and "4" in fr["reason"]


def test_freeze_layers_not_capped_when_card_has_enough_layers():
    # 36-layer model: the data-driven 8 stays (well under the cap).
    rec = suggest_config(get_model("qwen3-4b"), 50000, True,
                         finetuning_type="freeze", arch={"numHiddenLayers": 36})
    assert rec.hp.freeze_trainable_layers == 8


def test_cutoff_value_cites_card_when_context_smaller():
    # Card reports a context SMALLER than the catalog default → recommend the card
    # ceiling, citing it in the value.
    model = get_model("qwen3-4b")  # default_cutoff_len 2048
    rec = suggest_config(model, 5375, True, arch={"maxPositionEmbeddings": 1024})
    cutoff = next(r for r in rec.rationale if r["field"] == "cutoffLen")
    assert "from model card" in cutoff["value"] and "1024" in cutoff["value"]


def test_cutoff_reason_mentions_card_ceiling_when_larger():
    model = get_model("qwen3-4b")
    rec = suggest_config(model, 5375, True, arch={"maxPositionEmbeddings": 32768})
    cutoff = next(r for r in rec.rationale if r["field"] == "cutoffLen")
    assert "32768" in cutoff["reason"]  # surfaced as the card's max


def test_fetch_arch_degrades_to_empty_on_fetch_error(monkeypatch):
    # A fetch failure (offline/gated/private) must yield {} so the recommender
    # falls back cleanly. Force the underlying HTTP to raise.
    from app import model_card, onboard

    def boom(*_a, **_k):
        raise RuntimeError("offline")

    monkeypatch.setattr(onboard, "_http_json", boom)
    model_card.fetch_arch.cache_clear()
    try:
        assert model_card.fetch_arch("Qwen/Qwen3-4B-Instruct-2507") == {}
    finally:
        model_card.fetch_arch.cache_clear()


def test_fetch_arch_extracts_known_fields(monkeypatch):
    from app import model_card, onboard

    monkeypatch.setattr(onboard, "_http_json", lambda *_a, **_k: {
        "max_position_embeddings": 32768, "num_hidden_layers": 36,
        "model_type": "qwen3", "torch_dtype": "bfloat16", "ignored": "x",
    })
    model_card.fetch_arch.cache_clear()
    try:
        arch = model_card.fetch_arch("Qwen/Qwen3-4B-Instruct-2507")
        assert arch == {"maxPositionEmbeddings": 32768, "numHiddenLayers": 36,
                        "modelType": "qwen3", "torchDtype": "bfloat16"}
    finally:
        model_card.fetch_arch.cache_clear()
