# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LoRA adapter variants (DoRA / rsLoRA / PiSSA / LoRA+).

The variants are MODIFIERS on finetuning_type=lora|qlora (not new methods): each
emits a specific LLaMA-Factory flag in the train YAML, rides the same merge/export
path, and plain LoRA must stay byte-identical to before the feature existed. These
are correctness-critical — a typo'd or dropped flag silently trains plain LoRA, so
we pin the exact emitted keys here plus the request→Hyperparams plumbing.
"""
import dataclasses

import yaml

from app.catalog import Hyperparams, get_model
from app.render import render_train_yaml


def _doc(hp: Hyperparams, model_id: str = "qwen3-1.7b") -> dict:
    return yaml.safe_load(render_train_yaml(get_model(model_id), hp, "abc123def456"))


def test_plain_lora_emits_no_variant_keys():
    """Default variant=lora must not leak ANY variant flag (byte-identity guard)."""
    doc = _doc(Hyperparams())
    for k in ("use_dora", "use_rslora", "pissa_init", "pissa_iter",
              "pissa_convert", "loraplus_lr_ratio"):
        assert k not in doc, f"plain LoRA leaked {k!r}"


def test_plain_lora_yaml_byte_identical_to_pre_variant():
    """The whole rendered YAML for a default run is unchanged by this feature: an
    explicit variant='lora' renders identically to not passing the field at all."""
    a = render_train_yaml(get_model("qwen3-1.7b"), Hyperparams(), "s")
    b = render_train_yaml(get_model("qwen3-1.7b"), Hyperparams(lora_variant="lora"), "s")
    assert a == b


def test_dora_emits_use_dora():
    doc = _doc(Hyperparams(lora_variant="dora"))
    assert doc["use_dora"] is True
    # Only DoRA's flag — not another variant's.
    assert "use_rslora" not in doc and "pissa_init" not in doc


def test_rslora_emits_use_rslora():
    doc = _doc(Hyperparams(lora_variant="rslora"))
    assert doc["use_rslora"] is True
    assert "use_dora" not in doc


def test_pissa_emits_init_iter_and_convert():
    """PiSSA needs pissa_convert so the adapter merges back onto the ORIGINAL base
    (not the residual) — without it export produces a broken merge."""
    doc = _doc(Hyperparams(lora_variant="pissa"))
    assert doc["pissa_init"] is True
    assert doc["pissa_iter"] == 16
    assert doc["pissa_convert"] is True


def test_loraplus_emits_lr_ratio():
    doc = _doc(Hyperparams(lora_variant="loraplus", loraplus_lr_ratio=8.0))
    assert float(doc["loraplus_lr_ratio"]) == 8.0
    assert "use_dora" not in doc


def test_quant_compatible_variant_rides_qlora():
    """rsLoRA / LoRA+ need no full-precision weights, so they're valid on QLoRA (4-bit
    base) alongside quantization_bit — independent knobs."""
    doc = _doc(Hyperparams(finetuning_type="qlora", lora_variant="rslora"))
    assert doc["use_rslora"] is True
    assert doc["quantization_bit"] == 4
    doc2 = _doc(Hyperparams(finetuning_type="qlora", lora_variant="loraplus", loraplus_lr_ratio=8))
    assert float(doc2["loraplus_lr_ratio"]) == 8.0
    assert doc2["quantization_bit"] == 4


def test_qlora_rejects_dora_and_pissa():
    """DoRA + PiSSA need the full-precision weight matrix a 4-bit base lacks — PEFT
    rejects them at model-load, so Hyperparams must reject the combo BEFORE a billable
    launch (this exact pairing burned 2 real jobs in the my-golden-lora-varients race)."""
    import pytest

    for bad in ("dora", "pissa"):
        with pytest.raises(ValueError, match="incompatible with QLoRA"):
            Hyperparams(finetuning_type="qlora", lora_variant=bad)


def test_unknown_variant_rejected():
    import pytest

    with pytest.raises(ValueError, match="unsupported lora_variant"):
        Hyperparams(lora_variant="nope")


def test_variant_suppressed_for_full_weight():
    """Full/freeze have no adapter, so a stale variant is NORMALIZED to plain lora
    (not an error — old/cloned configs mustn't blow up) and never emitted. Crucially
    this normalization runs BEFORE the quant guard, so full/freeze + 'dora' is fine."""
    for method in ("full", "freeze"):
        # full/freeze are gated to small models — use a ≤2B model.
        hp = Hyperparams(finetuning_type=method, lora_variant="dora")
        assert hp.lora_variant == "lora", f"{method} should normalize a stale variant"
        doc = _doc(hp, model_id="qwen3-1.7b")
        assert "use_dora" not in doc, f"{method} leaked a variant flag"
        # And it carries no lora_* keys at all (it's full-weight).
        assert "lora_rank" not in doc


def test_variant_persists_through_asdict():
    """The orchestrator persists hp via dataclasses.asdict — the variant fields must
    survive so a cloned run can replay them (clone reads the persisted hp dict)."""
    d = dataclasses.asdict(Hyperparams(lora_variant="dora", loraplus_lr_ratio=8.0))
    assert d["lora_variant"] == "dora"
    assert d["loraplus_lr_ratio"] == 8.0


# --- API plumbing: request models must carry the variant through to Hyperparams ---


def test_render_request_threads_variant_to_hp():
    """RenderRequest (the /api/render + /api/train body) → _hyperparams_from → hp,
    so the rendered YAML reflects the requested variant. Accepts the camelCase alias."""
    import app.main as m

    req = m.RenderRequest(modelId="qwen3-1.7b", splitId="abc123def456",
                          finetuningType="lora", loraVariant="dora")
    hp = m._hyperparams_from(req)
    assert hp.lora_variant == "dora"
    doc = yaml.safe_load(render_train_yaml(get_model("qwen3-1.7b"), hp, "s"))
    assert doc["use_dora"] is True


def test_loraplus_ratio_alias_threads_through():
    import app.main as m

    req = m.RenderRequest(modelId="qwen3-1.7b", splitId="abc123def456",
                          loraVariant="loraplus", loraplusLrRatio=8.0)
    hp = m._hyperparams_from(req)
    assert hp.lora_variant == "loraplus"
    assert hp.loraplus_lr_ratio == 8.0


def test_race_model_config_to_hp_carries_variant():
    """RaceModelConfig.to_hp (the launch path) must carry the variant + ratio."""
    import app.main as m

    rc = m.RaceModelConfig(modelId="qwen3-1.7b", finetuningType="lora",
                           loraVariant="rslora")
    hp = rc.to_hp()
    assert hp.lora_variant == "rslora"


def test_request_models_default_to_plain_lora():
    """Omitting the field defaults to plain LoRA on every request model — existing
    callers (and old persisted runs) are unaffected."""
    import app.main as m

    assert m.RenderRequest(modelId="x", splitId="abc123def456").lora_variant == "lora"
    assert m.RaceModelConfig(modelId="x").lora_variant == "lora"


# --- LoRA+ ratio clamp (__post_init__ safety net, mirrors the KTO-weight clamp) ---


def test_loraplus_ratio_clamped_to_bounds():
    # Above the ceiling snaps down to max; a degenerate <1 snaps up to min.
    assert Hyperparams(lora_variant="loraplus", loraplus_lr_ratio=999).loraplus_lr_ratio == 128.0
    assert Hyperparams(lora_variant="loraplus", loraplus_lr_ratio=0).loraplus_lr_ratio == 1.0
    # An in-range value is untouched.
    assert Hyperparams(lora_variant="loraplus", loraplus_lr_ratio=16).loraplus_lr_ratio == 16.0


def test_clamped_ratio_reaches_rendered_yaml():
    hp = Hyperparams(lora_variant="loraplus", loraplus_lr_ratio=999)
    doc = _doc(hp)
    assert float(doc["loraplus_lr_ratio"]) == 128.0


# --- Advisor DoRA arm (one-click "try the recommended richer adapter") ---


def test_advisor_appends_dora_arm_for_plain_lora():
    """A plain-LoRA sweep always offers a DoRA comparison arm (even on the
    deterministic fallback, which runs with no Bedrock available in tests)."""
    from app.advisor import advise_sweep

    r = advise_sweep(get_model("qwen3-1.7b"), train_rows=500, has_val=True)
    dora = [c for c in r["configs"] if c["hyperparams"].get("loraVariant") == "dora"]
    assert len(dora) == 1, "expected exactly one DoRA arm"
    assert "recommended" in dora[0]["label"].lower()
    # The non-DoRA arms stay plain LoRA — the arm varies the ADAPTER, not the rest.
    assert all(c["hyperparams"].get("loraVariant") in (None, "lora")
               for c in r["configs"] if c is not dora[0])


def test_advisor_no_dora_arm_for_full_weight():
    """full/freeze carry no adapter — the DoRA arm must not appear."""
    from app.advisor import advise_sweep

    r = advise_sweep(get_model("qwen3-1.7b"), train_rows=500, has_val=True,
                     finetuning_type="full")
    assert all(c["hyperparams"].get("loraVariant") in (None, "lora") for c in r["configs"])


def test_advisor_no_dora_arm_for_qlora():
    """QLoRA base must NOT get a DoRA arm — DoRA is invalid on a 4-bit base, so
    proposing it would just produce a config that fails Hyperparams validation /
    a billable launch. rsLoRA/LoRA+ would be fine, but the auto-arm is DoRA-only."""
    from app.advisor import advise_sweep

    r = advise_sweep(get_model("qwen3-1.7b"), train_rows=500, has_val=True,
                     finetuning_type="qlora")
    assert all(c["hyperparams"].get("loraVariant") in (None, "lora") for c in r["configs"])
