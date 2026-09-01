# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""ORPO / SimPO preference objectives + efficiency knobs (NEFTune / Liger / packing).

ORPO and SimPO are NOT new stages — they are stage=dpo with a different `pref_loss`
(the same chosen/rejected preference dataset, a different algorithm). KTO stays its
own stage. These are correctness-critical:
  * a wrong/dropped pref_loss silently trains plain DPO,
  * SimPO without simpo_gamma uses the wrong margin,
  * DPO vs ORPO vs SimPO of the same model MUST race as distinct entries (no key
    collision), and
  * plain DPO + SFT must stay byte-identical to before this feature.
Efficiency knobs (neftune_noise_alpha / enable_liger_kernel / packing) ride any/SFT
stage and must emit ONLY when non-default.
"""
import yaml

from app.catalog import Hyperparams, get_model
from app.race import entry_key_for
from app.render import render_train_yaml


def _doc(hp: Hyperparams, model_id: str = "qwen3-1.7b") -> dict:
    return yaml.safe_load(render_train_yaml(get_model(model_id), hp, "abc123def456"))


# --- ORPO / SimPO render emission --------------------------------------------


def test_plain_dpo_pref_loss_byte_identical():
    """Default pref_loss=sigmoid renders identically to the pre-feature DPO config:
    pref_loss=sigmoid, no simpo_gamma. (The old code hardcoded 'sigmoid'.)"""
    doc = _doc(Hyperparams(stage="dpo"))
    assert doc["pref_loss"] == "sigmoid"
    assert "simpo_gamma" not in doc
    # explicit sigmoid == default → byte-identical whole YAML
    a = render_train_yaml(get_model("qwen3-1.7b"), Hyperparams(stage="dpo"), "s")
    b = render_train_yaml(get_model("qwen3-1.7b"), Hyperparams(stage="dpo", pref_loss="sigmoid"), "s")
    assert a == b


def test_orpo_emits_pref_loss_orpo_no_gamma():
    doc = _doc(Hyperparams(stage="dpo", pref_loss="orpo"))
    assert doc["pref_loss"] == "orpo"
    assert "simpo_gamma" not in doc  # gamma is SimPO-only
    assert doc["pref_beta"] == 0.1   # ORPO still uses beta


def test_simpo_emits_pref_loss_and_gamma():
    doc = _doc(Hyperparams(stage="dpo", pref_loss="simpo", simpo_gamma=0.8))
    assert doc["pref_loss"] == "simpo"
    assert float(doc["simpo_gamma"]) == 0.8


def test_sft_never_emits_pref_loss():
    """pref_loss is a stage=dpo key; an SFT config must never carry it (would crash
    the LF arg parser). __post_init__ normalizes a stray pref_loss on SFT to sigmoid,
    and render only emits it for stage=dpo."""
    doc = _doc(Hyperparams(stage="sft", pref_loss="orpo"))
    assert "pref_loss" not in doc
    # the stray value was normalized away
    assert Hyperparams(stage="sft", pref_loss="orpo").pref_loss == "sigmoid"


def test_kto_does_not_emit_pref_loss():
    """KTO is its own stage; it tunes via pref_beta + kto weights, never pref_loss."""
    doc = _doc(Hyperparams(stage="kto"))
    assert "pref_loss" not in doc
    assert "simpo_gamma" not in doc


# --- entry_key distinctness (no race collision) ------------------------------


def test_dpo_orpo_simpo_race_as_distinct_entries():
    """The bug this prevents: DPO/ORPO/SimPO of the same model all have stage=dpo,
    so without a pref_loss token they collide on the race's per-entry state map and
    all but one are silently lost."""
    k_dpo = entry_key_for("m", {"stage": "dpo"})
    k_orpo = entry_key_for("m", {"stage": "dpo", "pref_loss": "orpo"})
    k_simpo = entry_key_for("m", {"stage": "dpo", "pref_loss": "simpo"})
    assert k_dpo == "m"  # plain DPO bare — byte-identical to pre-feature
    assert k_orpo == "m::preforpo"
    assert k_simpo == "m::prefsimpo"
    assert len({k_dpo, k_orpo, k_simpo}) == 3


def test_pref_loss_token_only_for_dpo_stage():
    """A non-DPO stage never gets a pref_loss token (KTO/SFT key unchanged)."""
    assert entry_key_for("m", {"stage": "sft"}) == "m"
    assert entry_key_for("m", {"stage": "kto"}) == "m"
    # pref_loss on a non-dpo entry is ignored in the key (defensive)
    assert entry_key_for("m", {"stage": "kto", "pref_loss": "orpo"}) == "m"


def test_pref_loss_token_composes_with_qlora():
    """ORPO on QLoRA stacks both tokens (method first, then pref_loss last)."""
    assert entry_key_for("m", {"stage": "dpo", "finetuning_type": "qlora",
                               "pref_loss": "simpo"}) == "m::qlora::prefsimpo"


# --- validation guards (reject before a billable launch) ---------------------


def test_unsupported_pref_loss_rejected():
    try:
        Hyperparams(stage="dpo", pref_loss="bogus")
    except ValueError as e:
        assert "pref_loss" in str(e)
    else:
        raise AssertionError("expected ValueError for unsupported pref_loss")


def test_orpo_simpo_serverless_rejected():
    """ORPO/SimPO are LLaMA-Factory-only (serverless recipe exposes neither)."""
    for loss in ("orpo", "simpo"):
        try:
            # serverless only supports stage in (sft,dpo,rlvr,rlaif) + lora; dpo is
            # allowed, but the reference-free loss is not on serverless.
            Hyperparams(stage="dpo", pref_loss=loss, engine="sagemaker_serverless")
        except ValueError as e:
            assert "llama_factory" in str(e)
        else:
            raise AssertionError(f"expected ValueError for serverless+{loss}")


# --- efficiency knobs ---------------------------------------------------------


def test_efficiency_knobs_byte_identical_by_default():
    """Default (all off) emits no knob keys — an unchanged run is byte-identical."""
    doc = _doc(Hyperparams(stage="sft"))
    for k in ("neftune_noise_alpha", "enable_liger_kernel", "packing"):
        assert k not in doc, f"default leaked {k!r}"


def test_neftune_emitted_only_when_positive():
    assert "neftune_noise_alpha" not in _doc(Hyperparams(stage="sft", neftune_noise_alpha=0.0))
    doc = _doc(Hyperparams(stage="sft", neftune_noise_alpha=5.0))
    assert float(doc["neftune_noise_alpha"]) == 5.0


def test_liger_and_packing_emitted_when_true():
    doc = _doc(Hyperparams(stage="sft", enable_liger_kernel=True, packing=True))
    assert doc["enable_liger_kernel"] is True
    assert doc["packing"] is True


def test_neftune_alpha_clamped():
    """A degenerate alpha is clamped to the [0,15] band, not passed through."""
    assert Hyperparams(stage="sft", neftune_noise_alpha=999.0).neftune_noise_alpha == 15.0
    assert Hyperparams(stage="sft", neftune_noise_alpha=-3.0).neftune_noise_alpha == 0.0


def test_packing_rejected_for_non_sft():
    """Packing concatenates samples — unsafe for preference/KTO/RL losses; SFT-only."""
    for stage in ("dpo", "kto"):
        try:
            Hyperparams(stage=stage, packing=True)
        except ValueError as e:
            assert "packing" in str(e).lower()
        else:
            raise AssertionError(f"expected ValueError for packing+{stage}")


def test_knobs_efficiency_compose_with_objectives():
    """NEFTune/Liger ride a DPO/ORPO run too (orthogonal to objective). Packing is
    the only SFT-restricted one."""
    doc = _doc(Hyperparams(stage="dpo", pref_loss="orpo",
                           neftune_noise_alpha=5.0, enable_liger_kernel=True))
    assert doc["pref_loss"] == "orpo"
    assert float(doc["neftune_noise_alpha"]) == 5.0
    assert doc["enable_liger_kernel"] is True


def test_simpo_gamma_clamped():
    # Floor is a POSITIVE 0.1 (not 0) so "SimPO" can't degenerate to a margin-free loss.
    assert Hyperparams(stage="dpo", pref_loss="simpo", simpo_gamma=99.0).simpo_gamma == 2.0
    assert Hyperparams(stage="dpo", pref_loss="simpo", simpo_gamma=-1.0).simpo_gamma == 0.1


# --- request → Hyperparams threading -----------------------------------------


def test_race_model_config_threads_pref_loss_and_knobs():
    from app.main import RaceModelConfig

    cfg = RaceModelConfig(modelId="m", stage="dpo", prefLoss="simpo", simpoGamma=0.7,
                          neftuneNoiseAlpha=5.0, enableLigerKernel=True)
    hp = cfg.to_hp()
    assert hp.pref_loss == "simpo"
    assert hp.simpo_gamma == 0.7
    assert hp.neftune_noise_alpha == 5.0
    assert hp.enable_liger_kernel is True


def test_render_request_threads_pref_loss():
    from app.main import RenderRequest, _hyperparams_from

    req = RenderRequest(modelId="m", splitId="s", stage="dpo", prefLoss="orpo")
    hp = _hyperparams_from(req)
    assert hp.pref_loss == "orpo"
