# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LLM hyperparameter advisor (Tier 3) — proposes a SWEEP to race.

The deterministic recommender (recommend.py) gives one sound starting point.
This layers an LLM (Bedrock Sonnet) on top to do the thing an LLM is genuinely
good at: judgment about a *search space*. It proposes a small set of distinct
configs (varying the knob most worth exploring — usually learning rate / rank)
WITH a rationale, so the user races them and lets the leaderboard pick the
winner. The LLM never decides the answer; the race does.

Guardrails (the discipline learned from shipping engine-rejected configs):
  * grounded — prompt includes the model facts + deterministic baseline, not a
    blank ask; the LLM ADJUSTS a known-good config rather than inventing one.
  * validated — every returned config is clamped to Hyperparams.bounds() and
    only known fields are accepted; malformed output falls back to the
    deterministic recommendation. (Engine dry-parse happens at launch via the
    existing render path; here we guarantee in-range, well-typed configs.)
  * deterministic fallback — any LLM/parse error returns the Tier-2 sweep, so
    the feature degrades to "still useful" rather than failing.
"""

from __future__ import annotations

import json
from typing import Any

from .aws_config import load_aws_config
from .catalog import Hyperparams, ModelSpec
from .orchestrate import _session
from .recommend import suggest_config

# Default model id (Sonnet 4.5). The LIVE model is resolved per call from the
# 'advisor' role via agent_models.resolve_model_id (Settings-overridable); this
# constant is the documented default + a fallback reference.
ADVISOR_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _clamp_hp(d: dict[str, Any], base: Hyperparams) -> Hyperparams:
    """Build a bounds-clamped Hyperparams from a loose dict, falling back to
    `base` for anything missing/invalid. Only known knobs are honored."""
    b = Hyperparams.bounds()

    def num(key, lo_hi_key, default, *, integer=False):
        v = d.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        lo, hi = b[lo_hi_key]["min"], b[lo_hi_key]["max"]
        v = max(lo, min(hi, v))
        return int(round(v)) if integer else v

    # Carry the base method through (full/freeze keep their method + freeze depth;
    # the __post_init__ LR clamp protects full/freeze regardless of what the LLM
    # proposes). For full/freeze the lora_rank knob is irrelevant — keep the base.
    # LoRA variant: only an explicit, known value overrides the base (the LLM may
    # omit it). Full/freeze carry no adapter, so the variant is forced to "lora".
    variant = d.get("loraVariant", base.lora_variant)
    if variant not in ("lora", "dora", "rslora", "pissa", "loraplus"):
        variant = base.lora_variant

    return Hyperparams(
        finetuning_type=base.finetuning_type,
        lora_rank=num("loraRank", "loraRank", base.lora_rank, integer=True),
        lora_alpha=None,  # keep auto 2×rank
        lora_variant=variant,
        loraplus_lr_ratio=base.loraplus_lr_ratio,
        freeze_trainable_layers=num("freezeTrainableLayers", "loraRank",
                                    base.freeze_trainable_layers, integer=True),
        learning_rate=num("learningRate", "learningRate", base.learning_rate),
        num_train_epochs=num("numTrainEpochs", "numTrainEpochs", base.num_train_epochs),
        per_device_train_batch_size=base.per_device_train_batch_size,
        gradient_accumulation_steps=base.gradient_accumulation_steps,
        cutoff_len=base.cutoff_len,
        save_steps=base.save_steps,  # keep the derived eval cadence
        max_samples=base.max_samples,
        early_stopping_enabled=base.early_stopping_enabled,
        early_stopping_patience=base.early_stopping_patience,
    )


def _hp_out(hp: Hyperparams) -> dict[str, Any]:
    return {
        "finetuningType": hp.finetuning_type,
        "loraRank": hp.lora_rank,
        "loraAlpha": hp.lora_alpha,
        "loraVariant": hp.lora_variant,
        "loraplusLrRatio": hp.loraplus_lr_ratio,
        "freezeTrainableLayers": hp.freeze_trainable_layers,
        "learningRate": hp.learning_rate,
        "numTrainEpochs": hp.num_train_epochs,
        "perDeviceTrainBatchSize": hp.per_device_train_batch_size,
        "gradientAccumulationSteps": hp.gradient_accumulation_steps,
        "cutoffLen": hp.cutoff_len,
        "saveSteps": hp.save_steps,
        "maxSamples": hp.max_samples,
        "earlyStoppingEnabled": hp.early_stopping_enabled,
        "earlyStoppingPatience": hp.early_stopping_patience,
    }


def _deterministic_sweep(model: ModelSpec, base: Hyperparams) -> list[dict[str, Any]]:
    """Fallback sweep when the LLM is unavailable: vary LR around the base."""
    lr = base.learning_rate
    out = []
    for factor, label in ((0.5, "conservative"), (1.0, "baseline"), (2.0, "aggressive")):
        hp = _clamp_hp({"learningRate": lr * factor}, base)
        out.append({"label": label, "hyperparams": _hp_out(hp),
                    "reason": f"learning rate {hp.learning_rate:g} ({label})"})
    return out


def _dora_arm(base: Hyperparams) -> dict[str, Any] | None:
    """An extra sweep arm that keeps the baseline knobs but switches the adapter to
    DoRA — so a sweep always offers the 'does the richer adapter beat plain LoRA?'
    comparison in one click. Only for PLAIN LoRA (finetuning_type=="lora"): DoRA
    needs the full-precision weight matrix, so it's INVALID on a QLoRA (4-bit) base
    (PEFT rejects it — see Hyperparams.QUANT_INCOMPATIBLE_VARIANTS), meaningless for
    full/freeze, and we don't override a variant the caller already chose. Returns
    None when it doesn't apply."""
    if base.finetuning_type != "lora" or base.lora_variant != "lora":
        return None
    hp = _clamp_hp({"loraVariant": "dora"}, base)
    return {
        "label": "DoRA (recommended)",
        "reason": ("Same baseline config, but with the DoRA adapter — weight-decomposed "
                   "LoRA that often closes more of the gap to full fine-tuning at the same "
                   "cost. Race it against plain LoRA to measure the lift on your data."),
        "hyperparams": _hp_out(hp),
    }


def advise_sweep(model: ModelSpec, train_rows: int, has_val: bool, n: int = 3,
                 finetuning_type: str = "lora") -> dict[str, Any]:
    """Return up to `n` distinct configs to RACE, with rationale. Grounded in the
    deterministic baseline + model facts; bounds-validated; deterministic
    fallback on any error. `source` says whether the LLM or the fallback produced
    it (so the UI is honest about provenance).

    METHOD-AWARE: for full/freeze (full-weight) the prompt is reframed off LoRA —
    the safe LR scale is ~1e-5 (NOT 1e-4), there is no lora_rank to vary, and freeze
    can vary the trainable-layer depth. The __post_init__ clamp is the backstop."""
    base_rec = suggest_config(model, train_rows, has_val, finetuning_type=finetuning_type)
    base = base_rec.hp
    is_full_weight = finetuning_type in ("full", "freeze")

    if is_full_weight:
        knob_hint = ("learning_rate (keep it LOW, ~1e-5 scale; full-weight DIVERGES at "
                     "the LoRA 1e-4) and num_train_epochs"
                     + (", and freezeTrainableLayers (how many top layers to train)"
                        if finetuning_type == "freeze" else ""))
        resp_fields = ('"learningRate":float,"numTrainEpochs":float'
                       + (',"freezeTrainableLayers":int' if finetuning_type == "freeze" else ""))
        expertise = f"full-weight ({finetuning_type}) fine-tuning of small language models"
        extra = (" There is NO lora_rank for full-weight training — do not propose one. "
                 "The learning rate MUST stay near 1e-5 (ceiling 5e-5).")
    else:
        knob_hint = "usually learning_rate, sometimes lora_rank"
        resp_fields = '"loraRank":int,"learningRate":float,"numTrainEpochs":float'
        expertise = "LoRA fine-tuning small language models"
        extra = ""

    prompt = (
        f"You are an expert at {expertise}. Propose a "
        f"sweep of {n} DISTINCT hyperparameter configurations to race against each "
        "other for the model and dataset below. Vary the one or two knobs most "
        f"worth exploring ({knob_hint}); keep the others at the provided baseline. "
        "Return STRICT JSON only.\n\n"
        f"Method: {finetuning_type}.\n"
        f"Model: {model.display_name} ({model.params_b}B params, family {model.family}).\n"
        f"Dataset: {train_rows} training rows, validation set: {has_val}.\n"
        f"Deterministic baseline config: {json.dumps(_hp_out(base))}\n"
        f"Allowed ranges: {json.dumps(Hyperparams.bounds())}\n\n"
        'Respond as: {"configs":[{"label":"...","reason":"...",' + resp_fields + "}, ...]}. "
        "Keep every value within the allowed ranges." + extra
    )

    try:
        from .agent_models import resolve_model_id

        cfg = load_aws_config()
        _, boto_sess = _session(cfg)
        client = boto_sess.client("bedrock-runtime", region_name=cfg.region)
        # Converse API — model-AGNOSTIC (Claude AND Nova/Llama/etc.), so the model
        # is user-selectable in Settings ('advisor' role). The old invoke_model used
        # the Anthropic-only schema, which would 400 on a non-Claude pick.
        resp = client.converse(
            modelId=resolve_model_id("advisor"),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
        )
        out_msg = resp.get("output", {}).get("message", {})
        text = "".join(b.get("text", "") for b in out_msg.get("content", []))
        parsed = _extract_json(text)
        raw_configs = parsed.get("configs", []) if isinstance(parsed, dict) else []
        configs = []
        for rc in raw_configs[:n]:
            hp = _clamp_hp(rc, base)
            configs.append({
                "label": str(rc.get("label", "config"))[:40],
                "reason": str(rc.get("reason", ""))[:300],
                "hyperparams": _hp_out(hp),
            })
        if not configs:
            raise ValueError("LLM returned no usable configs")
        # Always offer the DoRA comparison arm for plain-LoRA adapter sweeps (the
        # LLM varies LR/rank; this varies the ADAPTER), so "try the recommended
        # richer adapter" is one click. No-op for full/freeze or a chosen variant.
        dora = _dora_arm(base)
        if dora:
            configs.append(dora)
        return {"source": "llm", "baseline": _hp_out(base), "configs": configs}
    except Exception as e:  # noqa: BLE001 — degrade to the deterministic sweep
        configs = _deterministic_sweep(model, base)
        dora = _dora_arm(base)
        if dora:
            configs.append(dora)
        return {
            "source": "fallback",
            "baseline": _hp_out(base),
            "configs": configs,
            "note": f"LLM advisor unavailable ({e}); showing a deterministic sweep.",
        }


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of the model's text (tolerates prose/fences)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}
