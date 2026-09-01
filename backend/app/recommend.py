# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic hyperparameter recommender (Tier 2).

Getting good defaults per model is hard, and the right value depends on the
DATASET as much as the model — so this is NOT an oracle. It produces a sensible,
explainable STARTING POINT from a few facts (model size, dataset size, whether a
validation set exists), each choice carrying a human-readable reason. The race
remains the source of truth: these defaults just save the user from guessing,
and fix the recurring footgun where `save_steps` is too coarse for the dataset
(so early stopping never gets a signal).

Deterministic by design (no LLM, no RNG) — same inputs always give the same
config, which keeps runs reproducible. An LLM *advisor* layers on top later
(Tier 3) to propose sweeps; it never silently overrides these.

All values are clamped to Hyperparams.bounds() so a recommendation can never be
out of the range the API/engine accept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .catalog import Hyperparams, ModelSpec

# Effective batch = per_device_batch_size × gradient_accumulation_steps. We hold
# these fixed (memory-safe on the g5 instances) and reason about optimizer steps
# from the effective batch, matching what render.py emits.
_PER_DEVICE_BATCH = 1
_GRAD_ACCUM = 8
_EFFECTIVE_BATCH = _PER_DEVICE_BATCH * _GRAD_ACCUM

# Target number of evaluations per epoch — enough for the early-stopping signal
# and a smooth loss curve without evaluating so often it slows training.
_EVALS_PER_EPOCH = 5

# LoRA learning rate is set by the OBJECTIVE, not the model size. Model size is
# NOT the primary driver of the right LoRA LR (arXiv:2602.04998 "Learning Rate
# Matters" — rank + objective dominate, params_b barely moves it), so the old
# size gate (1.5e-4 ≤2B / 1e-4 >2B) was mis-conditioned. What DOES change the LR
# by an order of magnitude is the training objective: SFT tolerates the standard
# ~2e-4 LoRA LR, but preference tuning (DPO/KTO) is far more sensitive and
# DIVERGES at that scale — TRL "LoRA Without Regret" keys SFT ~2e-4 vs DPO/KTO
# ~5e-6. (Full/freeze keep their own ~1e-5 full-weight LR, clamped in
# Hyperparams.__post_init__.)
_SFT_LORA_LR = 2.0e-4
_PREF_LORA_LR = 5.0e-6  # DPO / KTO — ~40× lower than SFT; the SFT LR diverges here


@dataclass
class Recommendation:
    hp: Hyperparams
    rationale: list[dict[str, str]]  # [{field, value, reason}]

    def to_dict(self) -> dict[str, Any]:
        b = Hyperparams.bounds()  # noqa: F841 (kept for parity; not serialized)
        return {
            "hyperparams": {
                "finetuningType": self.hp.finetuning_type,
                "loraRank": self.hp.lora_rank,
                "loraAlpha": self.hp.lora_alpha,
                "freezeTrainableLayers": self.hp.freeze_trainable_layers,
                "learningRate": self.hp.learning_rate,
                "numTrainEpochs": self.hp.num_train_epochs,
                "perDeviceTrainBatchSize": self.hp.per_device_train_batch_size,
                "gradientAccumulationSteps": self.hp.gradient_accumulation_steps,
                "cutoffLen": self.hp.cutoff_len,
                "saveSteps": self.hp.save_steps,
                "maxSamples": self.hp.max_samples,
                "earlyStoppingEnabled": self.hp.early_stopping_enabled,
                "earlyStoppingPatience": self.hp.early_stopping_patience,
            },
            "rationale": self.rationale,
        }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _round_to_10(n: float) -> int:
    return max(10, int(round(n / 10.0) * 10))


def suggest_config(
    model: ModelSpec,
    train_rows: int,
    has_val: bool,
    finetuning_type: str = "lora",
    arch: dict[str, Any] | None = None,
    objective: str = "sft",
) -> Recommendation:
    """Recommend a starting hyperparameter config for `model` on a dataset of
    `train_rows` rows, with/without a validation set. Returns the config plus a
    per-field rationale. Deterministic + bounds-clamped.

    METHOD-AWARE: full/freeze (full-weight) are NOT adapter methods — they have no
    LoRA rank, and at the LoRA-scale LR (1e-4) they DIVERGE / catastrophically
    forget, so they need a ~1e-5 LR (matches Hyperparams.FULL_WEIGHT_DEFAULT_LR +
    the backend clamp). freeze additionally trains only the top N layers. So for
    full/freeze we skip the LoRA-rank reasoning and recommend a full-weight LR (+ a
    freeze-layer count for freeze) instead of a LoRA-scale one.

    OBJECTIVE-AWARE (LoRA LR): the right LoRA learning rate is set by the OBJECTIVE,
    not the model size (arXiv:2602.04998). SFT uses the standard ~2e-4; preference
    tuning (dpo/kto) is far more sensitive and needs ~5e-6 — the SFT LR would
    diverge. This fixes the footgun where a guided DPO/KTO run silently inherited
    the SFT-scale 1.5e-4 (~30× too hot). `objective` ∈ {sft, dpo, kto, …}; anything
    non-preference is treated as SFT-scale.

    CARD-AWARE (Tier 1): `arch` is the model card's architecture facts (from
    model_card.fetch_arch — maxPositionEmbeddings / numHiddenLayers). When present,
    it seeds the values a card can LEGITIMATELY constrain (the cutoff_len ceiling +
    the freeze-layer cap), with the rationale citing the card. The card does NOT
    carry training hyperparameters, so LR / epochs / rank / save_steps still come
    from the deterministic size×dataset heuristics. Absent/empty arch → pure
    deterministic fallback (the card fetch is best-effort)."""
    bounds = Hyperparams.bounds()
    reasons: list[dict[str, str]] = []
    params_b = model.params_b
    is_full_weight = finetuning_type in ("full", "freeze")
    arch = arch or {}
    card_layers = arch.get("numHiddenLayers")
    card_ctx = arch.get("maxPositionEmbeddings")

    rank: int | None = None
    if not is_full_weight:
        # --- LoRA rank: scale capacity with dataset size (more data tolerates a
        #     larger adapter without overfitting); keep modest for tiny datasets. ---
        if train_rows < 1000:
            rank = 8
            rank_reason = f"small dataset ({train_rows} rows) — a light rank-8 adapter resists overfitting"
        elif train_rows < 20000:
            rank = 16
            rank_reason = f"medium dataset ({train_rows} rows) — rank 16 adds capacity without overfitting"
        else:
            rank = 32
            rank_reason = f"large dataset ({train_rows} rows) — rank 32 gives the adapter room to learn"
        rank = int(_clamp(rank, bounds["loraRank"]["min"], bounds["loraRank"]["max"]))
        reasons.append({"field": "loraRank", "value": str(rank), "reason": rank_reason})
        # alpha left as None → engine auto-uses 2×rank (the LLaMA-Factory convention).
        reasons.append({"field": "loraAlpha", "value": "auto (2×rank)",
                        "reason": "LLaMA-Factory convention: alpha = 2 × rank"})

    # --- freeze: number of top transformer layers to train (scale with data). ---
    freeze_layers = Hyperparams.__dataclass_fields__["freeze_trainable_layers"].default
    if finetuning_type == "freeze":
        freeze_layers = 2 if train_rows < 1000 else 4 if train_rows < 20000 else 8
        freeze_reason = (f"{train_rows} rows — train the top {freeze_layers} layers "
                         "(more data → more trainable depth without forgetting the rest)")
        # CARD-AWARE cap: never recommend freezing-and-training more layers than the
        # model HAS (would silently train the whole model). The card's layer count
        # bounds it — leave ≥1 layer frozen so 'freeze' stays distinct from 'full'.
        if isinstance(card_layers, int) and card_layers > 0:
            cap = max(1, card_layers - 1)
            if freeze_layers > cap:
                freeze_layers = cap
                freeze_reason = (f"capped at {cap} — the model card reports {card_layers} "
                                 "transformer layers (can't train more than exist)")
        reasons.append({"field": "freezeTrainableLayers", "value": str(freeze_layers),
                        "reason": freeze_reason})

    # --- Learning rate: OBJECTIVE-conditioned for LoRA; full-weight is separate. ---
    # full/freeze need a much lower full-weight LR (~1e-5). For LoRA the driver is the
    # OBJECTIVE, not the model size: SFT ~2e-4, but preference tuning (dpo/kto) needs
    # ~5e-6 or it diverges (arXiv:2602.04998; TRL "LoRA Without Regret").
    is_pref = objective in ("dpo", "kto")
    if is_full_weight:
        lr = Hyperparams.FULL_WEIGHT_DEFAULT_LR  # 1e-5
        lr_reason = (f"{finetuning_type} fine-tuning is full-weight — needs a low LR (~1e-5); "
                     "the LoRA-scale 1e-4 would diverge / catastrophically forget")
    elif is_pref:
        lr = _PREF_LORA_LR  # 5e-6
        lr_reason = (f"{objective.upper()} is preference tuning — needs a much lower LR (~5e-6); "
                     "the SFT-scale 2e-4 diverges on a preference loss")
    else:
        lr = _SFT_LORA_LR  # 2e-4
        lr_reason = "SFT LoRA — 2e-4 is the standard, size-independent LoRA default"
    lr = _clamp(lr, bounds["learningRate"]["min"], bounds["learningRate"]["max"])
    reasons.append({"field": "learningRate", "value": f"{lr:g}", "reason": lr_reason})

    # --- Epochs: fewer passes for large datasets, more for small ones. With a
    #     val set this is a CEILING (early stopping ends sooner). ---
    if train_rows < 1000:
        epochs = 6.0
    elif train_rows < 10000:
        epochs = 4.0
    else:
        epochs = 3.0
    epochs = _clamp(epochs, bounds["numTrainEpochs"]["min"], bounds["numTrainEpochs"]["max"])
    epoch_reason = (
        f"{'ceiling — early stopping ends sooner' if has_val else 'fixed schedule'}; "
        f"{'fewer passes suit a large set' if train_rows >= 10000 else 'more passes help a smaller set'}"
    )
    reasons.append({"field": "numTrainEpochs", "value": f"{epochs:g}", "reason": epoch_reason})

    # --- save_steps: derive from steps/epoch so we evaluate ~_EVALS_PER_EPOCH
    #     times per epoch. THIS is the recurring footgun: the default 500 is far
    #     too coarse for small datasets, so early stopping never gets a signal. ---
    steps_per_epoch = max(1, math.ceil(train_rows / _EFFECTIVE_BATCH))
    save_steps = _round_to_10(steps_per_epoch / _EVALS_PER_EPOCH)
    save_steps = int(_clamp(save_steps, bounds["saveSteps"]["min"], bounds["saveSteps"]["max"]))
    reasons.append({
        "field": "saveSteps",
        "value": str(save_steps),
        "reason": (
            f"{steps_per_epoch} steps/epoch (eff. batch {_EFFECTIVE_BATCH}); "
            f"evaluate ~{_EVALS_PER_EPOCH}×/epoch for a clean curve + early-stopping signal"
        ),
    })

    # --- Cutoff: the model card's max_position_embeddings is the model's true
    #     context ceiling. When the card reports a SMALLER ceiling than the catalog
    #     default (rare but real for some small models), recommend the card's value
    #     so we never train past what the model supports. Otherwise keep the catalog
    #     default. We don't scan token lengths here — surfaced so the user can lower
    #     it for short texts to save cost. (cutoff_len stays None below = engine uses
    #     the model default; this rationale just tells the user where it came from.) ---
    if isinstance(card_ctx, int) and card_ctx > 0 and card_ctx < model.default_cutoff_len:
        cutoff_reason = (f"model card max context is {card_ctx} (< the {model.default_cutoff_len} "
                         "default) — capping there; lower further for short examples to cut cost")
        cutoff_value = f"{card_ctx} (from model card)"
    elif isinstance(card_ctx, int) and card_ctx > 0:
        cutoff_reason = (f"model default ({model.default_cutoff_len}); the model card supports up to "
                         f"{card_ctx} tokens. Lower it if your examples are short to cut cost/time")
        cutoff_value = f"model default ({model.default_cutoff_len})"
    else:
        cutoff_reason = "fits the model; lower it if your examples are short to cut cost/time"
        cutoff_value = f"model default ({model.default_cutoff_len})"
    reasons.append({"field": "cutoffLen", "value": cutoff_value, "reason": cutoff_reason})

    # --- Early stopping: on iff a validation set exists (the eval-loss signal). ---
    es_enabled = has_val
    es_patience = 3
    reasons.append({
        "field": "earlyStopping",
        "value": f"on, patience {es_patience}" if es_enabled else "off",
        "reason": (
            "validation set present — stop at convergence + export the best checkpoint"
            if es_enabled
            else "no validation set — add one to enable early stopping"
        ),
    })

    hp = Hyperparams(
        finetuning_type=finetuning_type,
        lora_rank=rank if rank is not None else 8,  # ignored for full/freeze
        lora_alpha=None,  # auto 2×rank
        freeze_trainable_layers=freeze_layers,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=_PER_DEVICE_BATCH,
        gradient_accumulation_steps=_GRAD_ACCUM,
        cutoff_len=None,  # model default
        save_steps=save_steps,
        max_samples=None,  # all rows
        early_stopping_enabled=es_enabled,
        early_stopping_patience=es_patience,
    )
    return Recommendation(hp=hp, rationale=reasons)
