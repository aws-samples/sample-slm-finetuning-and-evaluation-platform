# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Model catalog manifest + LLaMA-Factory YAML render (the only engine-aware code)."""
import yaml

from app.catalog import Hyperparams, get_model, list_models
from app.render import render_all, render_export_yaml, render_train_yaml


def test_catalog_nonempty_and_ungated_majority():
    models = list_models()
    assert len(models) >= 15
    # Gated models are flagged so the UI can disable them.
    assert any(m["gated"] for m in models)
    assert any(not m["gated"] for m in models)


def test_get_model_known_and_unknown():
    assert get_model("qwen3-1.7b") is not None
    assert get_model("nope-not-real") is None


def test_every_model_has_template_and_instance():
    for m in list_models():
        assert m["template"], f"{m['id']} missing template"
        assert m["suggestedInstance"].startswith("ml."), f"{m['id']} bad instance"


def test_every_catalog_template_is_a_known_llamafactory_template():
    """Regression guard for the phi4_mini bug: a catalog template that LLaMA-
    Factory doesn't ship fails the job at config-parse. Every template must be in
    the known set (kept in sync with the engine's registry)."""
    from app.catalog import CATALOG
    from app.onboard import KNOWN_TEMPLATES

    bad = sorted({m.template for m in CATALOG} - KNOWN_TEMPLATES)
    assert not bad, f"catalog templates not in KNOWN_TEMPLATES (would fail the engine): {bad}"


def test_render_train_yaml_carries_hyperparams():
    model = get_model("qwen3-1.7b")
    hp = Hyperparams(lora_rank=64, learning_rate=2e-5, num_train_epochs=5)
    y = render_train_yaml(model, hp, "abc123def456")
    doc = yaml.safe_load(y)
    assert doc["lora_rank"] == 64
    assert float(doc["learning_rate"]) == 2e-5
    assert float(doc["num_train_epochs"]) == 5
    # Template comes from the catalog manifest, not the hyperparams.
    assert doc["template"] == model.template


def test_lora_alpha_defaults_to_double_rank_when_unset():
    model = get_model("qwen3-1.7b")
    y = render_train_yaml(model, Hyperparams(lora_rank=16, lora_alpha=None), "s")
    doc = yaml.safe_load(y)
    # LLaMA-Factory convention: alpha = 2 * rank when unset.
    assert doc["lora_alpha"] == 32


def test_render_all_produces_both_yamls():
    model = get_model("qwen3-1.7b")
    out = render_all(model, Hyperparams(), "split123")
    assert "trainYaml" in out and "exportYaml" in out
    assert yaml.safe_load(out["trainYaml"])  # parses
    assert yaml.safe_load(out["exportYaml"])


def test_hyperparam_bounds_present():
    b = Hyperparams.bounds()
    assert b["loraRank"]["min"] >= 1
    assert b["learningRate"]["max"] <= 1e-2


# --- Phase 1: parameterization (LoRA vs QLoRA) as method-config ---


def test_default_lora_yaml_is_unchanged_no_quantization_key():
    """The default SFT-LoRA path must stay byte-for-byte: finetuning_type=lora
    and NO quantization_bit key (a stray key would silently change every existing
    run's config)."""
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(), "split123"))
    assert doc["finetuning_type"] == "lora"
    assert "quantization_bit" not in doc


def test_qlora_maps_to_engine_lora_plus_quantization_bit():
    """LLaMA-Factory has NO 'qlora' finetuning_type — QLoRA is engine
    finetuning_type:lora + quantization_bit:4. Emitting 'qlora' fails the arg
    parser ('Invalid fine-tuning method'), caught by a real run."""
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(finetuning_type="qlora"), "s"))
    assert doc["finetuning_type"] == "lora"  # NOT "qlora" — the engine rejects that
    assert doc["quantization_bit"] == 4  # this is what makes it QLoRA
    # QLoRA is still LoRA under the hood — the LoRA knobs are still emitted.
    assert "lora_rank" in doc and doc["stage"] == "sft"


def test_export_never_quantizes_even_for_qlora():
    """Merging an adapter onto a quantized base is unsupported — the export/merge
    config must NEVER carry quantization_bit, for LoRA or QLoRA alike."""
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_export_yaml(model, "s"))
    assert "quantization_bit" not in doc


def test_hyperparams_normalizes_method_and_bit():
    # lora clears any stray quantization_bit; qlora without an explicit bit -> 4.
    assert Hyperparams(finetuning_type="lora", quantization_bit=4).quantization_bit is None
    assert Hyperparams(finetuning_type="qlora").quantization_bit == 4


def test_hyperparams_rejects_unsupported_method():
    import pytest

    # Truly unknown methods are still rejected.
    with pytest.raises(ValueError):
        Hyperparams(finetuning_type="bogus")


def test_full_freeze_methods_supported_for_sft():
    # full/freeze are now first-class (full-weight) methods for SFT on llama_factory.
    for m in ("full", "freeze"):
        hp = Hyperparams(finetuning_type=m, stage="sft")
        assert hp.finetuning_type == m
        assert hp.quantization_bit is None  # full-weight never quantizes
        # The LR safety net snaps an inherited LoRA-scale LR (1e-4) down to the
        # full-weight default (1e-5) so a full-FT run can't silently diverge.
        assert hp.learning_rate <= Hyperparams.FULL_WEIGHT_MAX_LR


def test_full_freeze_rejected_for_non_sft_stage():
    import pytest

    # full/freeze are SFT-only in this phase (DPO/KTO full-FT deferred).
    for stage in ("dpo", "kto"):
        with pytest.raises(ValueError):
            Hyperparams(finetuning_type="full", stage=stage)


def test_full_freeze_lr_clamp():
    # An explicit LoRA-scale LR on a full-weight run is clamped to the safe default.
    hp = Hyperparams(finetuning_type="full", stage="sft", learning_rate=1.0e-4)
    assert hp.learning_rate == Hyperparams.FULL_WEIGHT_DEFAULT_LR
    # A sane low LR is preserved untouched.
    hp2 = Hyperparams(finetuning_type="full", stage="sft", learning_rate=2.0e-5)
    assert hp2.learning_rate == 2.0e-5


def test_catalog_exposes_allowed_methods_hint():
    for m in list_models():
        assert "lora" in m["allowedMethods"], f"{m['id']} should allow lora"


def test_default_sft_yaml_has_no_dpo_keys():
    """SFT (default) stays byte-identical — no pref_loss/pref_beta keys."""
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(), "s"))
    assert doc["stage"] == "sft"
    assert "pref_loss" not in doc and "pref_beta" not in doc


def test_dpo_emits_preference_loss_keys():
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(stage="dpo", pref_beta=0.2), "s"))
    assert doc["stage"] == "dpo"
    assert doc["pref_loss"] == "sigmoid"
    assert doc["pref_beta"] == 0.2


def test_dpo_and_qlora_compose():
    model = get_model("qwen3-1.7b")
    hp = Hyperparams(stage="dpo", finetuning_type="qlora")
    doc = yaml.safe_load(render_train_yaml(model, hp, "s"))
    assert doc["stage"] == "dpo" and doc["finetuning_type"] == "lora"  # qlora→lora+quant
    assert doc["quantization_bit"] == 4 and doc["pref_loss"] == "sigmoid"


def test_full_method_emits_full_weight_train_yaml():
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(finetuning_type="full"), "s"))
    assert doc["finetuning_type"] == "full"
    # full-weight carries NO lora_* keys and NO quantization_bit.
    for k in ("lora_rank", "lora_alpha", "lora_target", "quantization_bit", "freeze_trainable_layers"):
        assert k not in doc, f"full run should not emit {k}"


def test_freeze_method_emits_freeze_layers():
    model = get_model("qwen3-1.7b")
    hp = Hyperparams(finetuning_type="freeze", freeze_trainable_layers=4)
    doc = yaml.safe_load(render_train_yaml(model, hp, "s"))
    assert doc["finetuning_type"] == "freeze"
    assert doc["freeze_trainable_layers"] == 4
    for k in ("lora_rank", "lora_alpha", "lora_target"):
        assert k not in doc


def test_full_freeze_render_all_omits_export_yaml():
    # No adapter to merge → render_all must NOT emit exportYaml (the entrypoint
    # branches on its absence to skip the merge).
    model = get_model("qwen3-1.7b")
    for m in ("full", "freeze"):
        out = render_all(model, Hyperparams(finetuning_type=m), "s")
        assert "trainYaml" in out
        assert "exportYaml" not in out, f"{m} should not emit an export.yaml"
    # adapter methods still emit it.
    assert "exportYaml" in render_all(model, Hyperparams(finetuning_type="lora"), "s")


def test_full_weight_instance_is_method_aware():
    # The size-based LoRA pick routes a ≤2B model to a cheap g5; a full-weight run
    # of the SAME size must route to the bigger g6e (L40S) instead — else it OOMs
    # on the 24GB g5. This is the seam the race launch must use (race.py).
    from app.catalog import _instance_for

    assert _instance_for(1.7, "lora") == "ml.g5.2xlarge"
    assert _instance_for(1.7, "full") == "ml.g6e.2xlarge"
    assert _instance_for(1.7, "freeze") == "ml.g6e.2xlarge"
    assert _instance_for(2.0, "full") == "ml.g6e.2xlarge"


def test_full_freeze_offered_only_for_small_models():
    # full/freeze are offered (allowed_methods) for small models, withheld for big.
    small = get_model("qwen3-1.7b")
    assert "full" in small.allowed_methods and "freeze" in small.allowed_methods
    big = get_model("qwen3-8b")
    assert "full" not in big.allowed_methods and "freeze" not in big.allowed_methods


def test_hyperparams_rejects_unsupported_stage():
    import pytest

    for bad in ("ppo", "grpo", "bogus"):
        with pytest.raises(ValueError):
            Hyperparams(stage=bad)


def test_job_name_tags_qlora_for_distinct_leaderboard_rows():
    """QLoRA must train under a job name that the leaderboard label heuristic
    surfaces as a SEPARATE row, while the default LoRA name stays byte-identical
    to pre-change runs (no silent leaderboard churn)."""
    from app.orchestrate import _job_name
    from app.leaderboard import _model_label

    split = "abc123def456"
    lora = _job_name("qwen3-4b", split, "20260611-1", method="lora")
    qlora = _job_name("qwen3-4b", split, "20260611-1", method="qlora")
    # LoRA name unchanged vs the original 3-arg signature.
    assert lora == _job_name("qwen3-4b", split, "20260611-1")
    assert lora == f"slm-qwen3-4b-{split}-20260611-1"
    # Distinct names → distinct leaderboard labels (the comparison is visible).
    assert lora != qlora
    assert _model_label(lora) == "qwen3-4b"
    assert _model_label(qlora) == "qwen3-4b-qlora"


def test_kto_emits_pref_beta_no_pref_loss():
    """KTO loss is selected by stage=kto; pref_beta tunes it. No pref_loss key
    (that's DPO). finetuning_type stays lora."""
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(stage="kto", pref_beta=0.15), "s"))
    assert doc["stage"] == "kto" and doc["pref_beta"] == 0.15
    assert "pref_loss" not in doc and doc["finetuning_type"] == "lora"


def test_kto_default_weights_byte_identical_no_weight_keys():
    """A KTO run with default (1.0/1.0) weights emits NEITHER kto_chosen_weight
    NOR kto_rejected_weight — byte-identical to before the knob existed."""
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(stage="kto"), "s"))
    assert "kto_chosen_weight" not in doc
    assert "kto_rejected_weight" not in doc


def test_kto_nondefault_weights_emitted():
    """Non-default KTO weights are emitted with LLaMA-Factory's exact key names
    (verified present in FinetuningArguments at v0.9.4 and v0.9.5). Only the
    changed weight appears — the one left at 1.0 is omitted."""
    model = get_model("qwen3-1.7b")
    doc = yaml.safe_load(
        render_train_yaml(model, Hyperparams(stage="kto", kto_rejected_weight=3.0), "s"))
    assert doc["kto_rejected_weight"] == 3.0
    assert "kto_chosen_weight" not in doc  # left at default 1.0 → omitted
    # both set
    doc2 = yaml.safe_load(
        render_train_yaml(model, Hyperparams(stage="kto", kto_chosen_weight=2.0, kto_rejected_weight=1.5), "s"))
    assert doc2["kto_chosen_weight"] == 2.0 and doc2["kto_rejected_weight"] == 1.5


def test_kto_weights_ignored_for_non_kto_stages():
    """KTO weights never leak into SFT/DPO configs even if set on the Hyperparams."""
    model = get_model("qwen3-1.7b")
    for stage in ("sft", "dpo"):
        doc = yaml.safe_load(
            render_train_yaml(model, Hyperparams(stage=stage, kto_chosen_weight=2.0,
                                                 kto_rejected_weight=3.0), "s"))
        assert "kto_chosen_weight" not in doc and "kto_rejected_weight" not in doc


def test_kto_weight_clamp_and_bounds():
    """Weights clamp to [0.1, 4.0]; bounds() exposes the range for the UI."""
    hp = Hyperparams(stage="kto", kto_chosen_weight=99.0, kto_rejected_weight=-5.0)
    assert hp.kto_chosen_weight == 4.0 and hp.kto_rejected_weight == 0.1
    b = Hyperparams.bounds()
    assert b["ktoChosenWeight"] == {"min": 0.1, "max": 4.0}
    assert b["ktoRejectedWeight"] == {"min": 0.1, "max": 4.0}


def test_model_id_for_hf_reverse_lookup():
    """Base-model leaderboard rows must label by the catalog id (same as the
    fine-tuned row), not the raw HF repo name — else base and fine-tuned look
    like different models (e.g. 'qwen2.5-0.5b' vs 'Qwen2.5-0.5B-Instruct')."""
    from app.catalog import model_id_for_hf, get_model
    spec = get_model("qwen2.5-0.5b")
    assert model_id_for_hf(spec.hf_model_id) == "qwen2.5-0.5b"
    assert model_id_for_hf("not/in-catalog") is None
    assert model_id_for_hf("") is None


def test_base_label_matches_finetuned_label():
    from app.leaderboard import _hf_to_label
    from app.catalog import get_model
    spec = get_model("qwen2.5-0.5b")
    # base row label == the catalog id (fine-tuned rows use the same id)
    assert _hf_to_label(spec.hf_model_id) == "qwen2.5-0.5b"
    # graceful fallback for an unknown HF id
    assert _hf_to_label("foo/Bar-7B") == "Bar-7B"


# --- reasoning-aware eval token budget (item 4) -----------------------------
# Reasoning families (Qwen3, R1, GLM-Z1, gpt-oss) emit a <think> CoT before the
# answer; an unclosed block gets stripped by eval.extract_answer → score 0. The
# eval max_new_tokens floor gives the CoT room to close.

def test_reasoning_flag_detects_cot_families():
    from app.catalog import get_model
    # reasoning families
    assert get_model("qwen3-1.7b").reasoning is True
    assert get_model("deepseek-r1-distill-qwen-1.5b").reasoning is True
    assert get_model("glm-z1-9b").reasoning is True
    assert get_model("gpt-oss-20b").reasoning is True
    # non-reasoning families
    assert get_model("qwen2.5-7b").reasoning is False
    assert get_model("granite-3.1-2b").reasoning is False
    assert get_model("phi-4").reasoning is False
    # surfaced in the manifest for the UI
    assert get_model("qwen3-1.7b").to_dict()["reasoning"] is True


def test_reasoning_eval_floor():
    from app.catalog import reasoning_eval_floor, REASONING_EVAL_MAX_NEW_TOKENS
    F = REASONING_EVAL_MAX_NEW_TOKENS
    # a reasoning model bumps a too-low request up to the floor
    assert reasoning_eval_floor(["qwen3-1.7b"], 256) == F
    assert reasoning_eval_floor(["deepseek-r1-distill-qwen-1.5b"], 64) == F
    # a non-reasoning model leaves it untouched
    assert reasoning_eval_floor(["qwen2.5-7b"], 256) == 256
    # mixed race bumps (ANY reasoning racer needs the budget; eval is shared)
    assert reasoning_eval_floor(["qwen2.5-7b", "qwen3-1.7b"], 256) == F
    # an explicit higher request is never lowered
    assert reasoning_eval_floor(["qwen3-1.7b"], 1024) == 1024
    # unknown model id is ignored gracefully (no crash, no bump)
    assert reasoning_eval_floor(["nonexistent"], 256) == 256
