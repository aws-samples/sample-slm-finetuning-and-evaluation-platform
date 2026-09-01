# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Render LLaMA-Factory YAML from an engine-neutral manifest entry.

THIS IS THE ONLY ENGINE-AWARE MODULE. It takes a ModelSpec (catalog),
Hyperparams, and a persisted split id, and emits two YAML documents:

  train.yaml   — `llamafactory-cli train` config (stage: sft, LoRA)
  export.yaml  — `llamafactory-cli export` config (merge LoRA → safetensors)

Field set + names are taken verbatim from the live LLaMA-Factory examples
(examples/train_lora/qwen3_lora_sft.yaml and examples/merge_lora/qwen3_lora_sft.yaml,
repo commit a98a1ef). If the engine ever changes, only this file changes.

The generic container entrypoint (one image for all models, per the architecture)
runs: `llamafactory-cli train train.yaml` then `llamafactory-cli export export.yaml`.
"""

from __future__ import annotations

from typing import Any

import yaml

from .catalog import Hyperparams, ModelSpec
from .storage import VAL_DATASET_NAME, EVAL_DATASET_NAME, split_meta

# Where the container expects the dataset_info.json + jsonl files to be mounted,
# and where it writes adapters / merged weights. A launched job maps these to S3
# via SageMaker channels; in a rendered YAML alone they are illustrative paths.
DATA_DIR = "/opt/ml/input/data/dataset"
ADAPTER_OUTPUT_DIR = "/opt/ml/model/adapter"
MERGED_OUTPUT_DIR = "/opt/ml/model/merged"
# The trainer ALWAYS writes its adapter/checkpoints here — SageMaker syncs this
# dir to checkpoint_s3_uri continuously and restores it before a re-run, so the
# entrypoint can resume after an interruption (spot reclaim) OR a failure that the
# user re-submits (on-demand resume). Originally spot-only; unified so on-demand
# jobs are also resumable-from-last-checkpoint (a failed run isn't wasted). The
# merged export still goes to /opt/ml/model/merged either way (the eval consumer).
CHECKPOINT_DIR = "/opt/ml/checkpoints"


class _Dumper(yaml.SafeDumper):
    """YAML dumper that keeps key insertion order and avoids anchors."""


def _dump(d: dict[str, Any]) -> str:
    return yaml.dump(d, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=100)


def render_train_yaml(model: ModelSpec, hp: Hyperparams, split_id: str, use_spot: bool = False) -> str:
    cutoff = hp.cutoff_len or model.default_cutoff_len
    lora_alpha = hp.lora_alpha if hp.lora_alpha is not None else hp.lora_rank * 2

    # In-training eval set: prefer the dataset's VALIDATION split when it has one,
    # so the held-out eval_split stays untouched (fair leaderboard comparison +
    # the signal early stopping needs). Datasets with no val fall back to eval.
    meta = split_meta(split_id)
    has_val = bool(meta.get("hasVal"))
    # Non-SFT objectives (DPO ranking, KTO labelled) have a train/val file shaped
    # for that objective, NOT plain messages. The held-out eval_split is always
    # messages (it backs the shared generation eval), so it CANNOT be the
    # in-training eval for these — only a matching-shape val_split can. With no val
    # set, such a run does no in-training eval (in_train_eval=None drops it below).
    # SFT keeps the original behaviour (val if present, else the eval_split).
    non_sft_shape = meta.get("shape") in ("preference", "kto")
    if non_sft_shape:
        in_train_eval = VAL_DATASET_NAME if has_val else None
    else:
        in_train_eval = VAL_DATASET_NAME if has_val else EVAL_DATASET_NAME

    # Write checkpoints to the SageMaker-synced checkpoint dir for EVERY job (spot
    # AND on-demand), so a reclaim OR a failure-then-resubmit can resume from the
    # last checkpoint instead of retraining from scratch. (Was spot-only; unified
    # so on-demand jobs are resumable too — a failed on-demand run is no longer
    # wasted. The use_spot arg is retained for the estimator/cost path.)
    output_dir = CHECKPOINT_DIR

    # LLaMA-Factory's `finetuning_type` accepts lora|oft|freeze|full — there is NO
    # "qlora" value. QLoRA = finetuning_type:lora PLUS quantization_bit:4 (a 4-bit-
    # quantized base under the same LoRA adapter). So our method "qlora" maps to the
    # engine's "lora"; the quantization_bit key (added below) is the only thing that
    # makes it QLoRA. "full"/"freeze" pass through unchanged — they are FULL-WEIGHT
    # methods (no adapter): they need NO lora_* keys, write standalone weights, and
    # MUST NOT be merged at export (entrypoint.sh skips the merge for them). (Caught
    # by a real run: emitting finetuning_type:qlora fails the arg parser.)
    engine_finetuning_type = "lora" if hp.finetuning_type == "qlora" else hp.finetuning_type
    is_full_weight = hp.finetuning_type in ("full", "freeze")

    cfg: dict[str, Any] = {
        # model
        "model_name_or_path": model.hf_model_id,
        "trust_remote_code": model.trust_remote_code,
        # method — objective (stage) + parameterization (finetuning_type) both come
        # from the hyperparams. stage="sft"+"lora" reproduces the original path
        # byte-for-byte; "qlora" → engine finetuning_type:lora + quantization_bit:4
        # (set below); "full"/"freeze" → full-weight (lora_* keys omitted below);
        # stage="dpo" trains on preference pairs (chosen/rejected) with pref_loss.
        "stage": hp.stage,
        "do_train": True,
        "finetuning_type": engine_finetuning_type,
        # dataset (resolved via dataset_info.json in DATA_DIR)
        "dataset_dir": DATA_DIR,
        "dataset": "train_split",
        "eval_dataset": in_train_eval,
        "template": model.template,
        "cutoff_len": cutoff,
        "preprocessing_num_workers": 16,
        "dataloader_num_workers": 4,
        # output
        "output_dir": output_dir,
        "logging_steps": 10,
        "save_steps": hp.save_steps,
        # Keep the 2 most recent checkpoints (not all): trims the artifact while
        # leaving a fallback if the latest checkpoint is corrupted/half-written.
        "save_total_limit": 2,
        # Save the FULL checkpoint (optimizer/scheduler/RNG state), not weights
        # only, so ANY re-run (spot reclaim or on-demand resubmit) can resume
        # mid-epoch rather than restarting. (Was weights-only on-demand; unified
        # for resumability — checkpoints are slightly larger but recoverable.)
        "save_only_model": False,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "report_to": "none",
        # train
        "per_device_train_batch_size": hp.per_device_train_batch_size,
        "gradient_accumulation_steps": hp.gradient_accumulation_steps,
        "learning_rate": hp.learning_rate,
        "num_train_epochs": hp.num_train_epochs,
        "lr_scheduler_type": hp.lr_scheduler_type,
        "warmup_ratio": hp.warmup_ratio,
        "bf16": True,
        # Gradient checkpointing: recompute activations in the backward pass
        # instead of storing them all. Activation memory is what scales with
        # cutoff_len × batch_size, so this is the main defence against a
        # DATA-driven OOM (a user feeding long examples). Costs ~20% compute,
        # always worth it for LoRA on a single GPU. Combined with the
        # size-aware instance pick (catalog._instance_for), training fits.
        "gradient_checkpointing": True,
        "ddp_timeout": 180000000,
        # Resume is driven by the container entrypoint: when SageMaker restores a
        # checkpoint-* dir into CHECKPOINT_DIR (spot reclaim or a resubmit pointed
        # at a prior checkpoint S3 prefix), entrypoint.sh appends
        # `resume_from_checkpoint=true` to the CLI, overriding this None. Left None
        # here so a FRESH run (no restored checkpoint) starts clean.
        "resume_from_checkpoint": None,
        # eval
        "per_device_eval_batch_size": hp.per_device_train_batch_size,
        "eval_strategy": "steps",
        "eval_steps": hp.save_steps,
    }
    # In-training eval needs an eval set of the RIGHT shape. in_train_eval is None
    # only for a DPO run with no ranking val set (the messages eval_split can't be
    # DPO's eval target) — in that case drop the eval keys entirely so the trainer
    # doesn't look for an eval dataset that isn't there. SFT is unaffected (it
    # always has a value here), so its config stays byte-identical.
    if in_train_eval is None:
        for k in ("eval_dataset", "per_device_eval_batch_size", "eval_strategy", "eval_steps"):
            cfg.pop(k, None)
    # Adapter (lora/qlora) keys: only emitted for adapter methods, so a full/freeze
    # config carries NO lora_* keys (LLaMA-Factory ignores them for full-weight, but
    # keeping them out is cleaner and matches the engine's full_sft examples).
    if not is_full_weight:
        cfg["lora_rank"] = hp.lora_rank
        cfg["lora_alpha"] = lora_alpha
        cfg["lora_target"] = model.lora_target
        # LoRA VARIANT modifiers (ride the same LoRA path). Only emitted for a
        # non-default variant, so plain LoRA/QLoRA configs stay byte-identical.
        # DoRA/rsLoRA/PiSSA are booleans; LoRA+ carries an LR ratio. PiSSA also
        # needs pissa_iter (SVD steps); 16 is LLaMA-Factory's example default.
        variant = getattr(hp, "lora_variant", "lora")
        if variant == "dora":
            cfg["use_dora"] = True
        elif variant == "rslora":
            cfg["use_rslora"] = True
        elif variant == "pissa":
            cfg["pissa_init"] = True
            cfg["pissa_iter"] = 16
            # PiSSA writes a converted base; convert at save so the adapter merges
            # back onto the ORIGINAL base (not the residual). Required for export.
            cfg["pissa_convert"] = True
        elif variant == "loraplus":
            cfg["loraplus_lr_ratio"] = hp.loraplus_lr_ratio
    # freeze trains only the top N transformer layers (LLaMA-Factory
    # freeze_trainable_layers). Emitted only for the freeze method.
    if hp.finetuning_type == "freeze":
        cfg["freeze_trainable_layers"] = hp.freeze_trainable_layers
    # QLoRA = LoRA + a 4-bit-quantized base. Only emitted when set, so the plain
    # LoRA config stays byte-identical to the original (no stray key). The
    # adapter it produces still merges onto the FULL-precision base at export —
    # so the export/merge path below is unchanged. (full/freeze never set this.)
    if hp.quantization_bit is not None:
        cfg["quantization_bit"] = hp.quantization_bit

    # DPO (preference) objective: LLaMA-Factory needs the preference-loss type +
    # beta. The pref_loss FAMILY selects the algorithm on the SAME chosen/rejected
    # dataset: sigmoid = standard DPO (default), orpo = ORPO, simpo = SimPO. ORPO/
    # SimPO are REFERENCE-FREE (LF derives use_ref_model=False for them — we don't
    # emit it). Only emitted for stage=dpo so SFT configs stay byte-identical; the
    # plain-DPO config also stays byte-identical because pref_loss defaults to
    # "sigmoid" (the value the old hardcoded line emitted).
    if hp.stage == "dpo":
        pref_loss = getattr(hp, "pref_loss", "sigmoid")
        cfg["pref_loss"] = pref_loss
        cfg["pref_beta"] = hp.pref_beta
        # SimPO's target reward margin γ (LF simpo_gamma). Only meaningful for SimPO;
        # emit it only there so DPO/ORPO configs carry no stray key.
        if pref_loss == "simpo":
            cfg["simpo_gamma"] = hp.simpo_gamma
    # KTO objective: independently-labelled desirable/undesirable completions.
    # LLaMA-Factory's KTO loss is selected by stage=kto; pref_beta tunes it the
    # same way. The dataset must be kto-labelled (messages + a boolean kto_tag) —
    # guaranteed by the KTO split (see storage.persist_kto_split).
    if hp.stage == "kto":
        cfg["pref_beta"] = hp.pref_beta
        # Per-class loss weights (KTO paper λ_D/λ_U; LF kto_chosen_weight/
        # kto_rejected_weight, default 1.0). Emit ONLY when a weight is non-default
        # so an unchanged KTO run stays byte-identical to before this knob existed.
        # Verified present in LF 0.9.4 AND 0.9.5 FinetuningArguments (RLHFArguments)
        # with no validation constraints — safe to pass.
        if hp.kto_chosen_weight != 1.0:
            cfg["kto_chosen_weight"] = hp.kto_chosen_weight
        if hp.kto_rejected_weight != 1.0:
            cfg["kto_rejected_weight"] = hp.kto_rejected_weight

    # Efficiency knobs (orthogonal to objective/method; LLaMA-Factory engine only).
    # Each is emitted ONLY when non-default so an unchanged run stays byte-identical
    # to before these existed. Validation in Hyperparams.__post_init__ already gated
    # them (packing→SFT only; all three→llama_factory only), so here we just emit.
    #   neftune_noise_alpha → HF TrainingArguments passthrough (embedding noise).
    #   enable_liger_kernel → LF model_arg (fused kernels: less memory, faster).
    #   packing → LF data_arg (concatenate short SFT samples to one cutoff_len seq).
    if getattr(hp, "neftune_noise_alpha", 0.0) and hp.neftune_noise_alpha > 0.0:
        cfg["neftune_noise_alpha"] = hp.neftune_noise_alpha
    if getattr(hp, "enable_liger_kernel", False):
        cfg["enable_liger_kernel"] = True
    if getattr(hp, "packing", False):
        cfg["packing"] = True

    if hp.max_samples is not None:
        # place near the dataset block conceptually; dict order still fine
        cfg["max_samples"] = hp.max_samples

    # Early stopping: only when requested AND the dataset has a val set
    # to provide the eval_loss signal. num_train_epochs then acts as a ceiling.
    # load_best_model_at_end makes export use the BEST checkpoint, not the last —
    # a free quality win. metric_for_best_model=eval_loss, lower is better.
    es_on = hp.early_stopping_enabled and has_val
    if es_on:
        # LLaMA-Factory 0.9.4 key is `early_stopping_steps` — despite the name its
        # value is the PATIENCE (passed straight to HF EarlyStoppingCallback's
        # early_stopping_patience, see train/tuner.py). There is NO
        # `early_stopping_patience` top-level key; emitting it crashes the arg
        # parser ("Some keys are not used by the HfArgumentParser").
        cfg["early_stopping_steps"] = hp.early_stopping_patience
        cfg["load_best_model_at_end"] = True
        cfg["metric_for_best_model"] = "eval_loss"
        cfg["greater_is_better"] = False
        # load_best_model_at_end requires save & eval cadence to match.
        cfg["save_strategy"] = "steps"
        cfg["save_steps"] = hp.save_steps
        cfg["eval_steps"] = hp.save_steps

    header_lines = [
        f"# Generated by SLM platform — train config for {model.display_name}",
        f"# split: {split_id}",
    ]
    if hp.stage == "dpo":
        _pl = getattr(hp, "pref_loss", "sigmoid")
        if _pl == "orpo":
            header_lines.append(
                f"# ORPO: reference-free preference fine-tuning on chosen/rejected pairs "
                f"(pref_loss=orpo, beta={hp.pref_beta:g})"
            )
        elif _pl == "simpo":
            header_lines.append(
                f"# SimPO: reference-free preference fine-tuning on chosen/rejected pairs "
                f"(pref_loss=simpo, gamma={hp.simpo_gamma:g})"
            )
        else:
            header_lines.append(
                f"# DPO: preference fine-tuning on chosen/rejected pairs "
                f"(pref_loss=sigmoid, beta={hp.pref_beta:g})"
            )
    if hp.stage == "kto":
        header_lines.append(
            f"# KTO: binary-feedback fine-tuning on labelled good/bad completions "
            f"(beta={hp.pref_beta:g})"
        )
    if hp.quantization_bit is not None:
        header_lines.append(
            f"# QLoRA: LoRA adapter on a {hp.quantization_bit}-bit base "
            f"(smaller footprint; merges onto the full-precision base at export)"
        )
    if es_on:
        header_lines.append(
            f"# early stopping: patience={hp.early_stopping_patience} on eval_loss "
            f"(num_train_epochs={hp.num_train_epochs:g} is a CEILING); exports best checkpoint"
        )
    header_lines.append("# run: llamafactory-cli train train.yaml")
    header = "\n".join(header_lines) + "\n"
    return header + _dump(cfg)


def render_export_yaml(model: ModelSpec, split_id: str, use_spot: bool = False) -> str:
    # The adapter to merge lives wherever training wrote it — now always the
    # SageMaker-synced checkpoint dir (output_dir=CHECKPOINT_DIR for every job).
    # use_spot is retained for signature symmetry with render_train_yaml.
    adapter_dir = CHECKPOINT_DIR
    cfg: dict[str, Any] = {
        # model + adapter to merge
        "model_name_or_path": model.hf_model_id,
        "adapter_name_or_path": adapter_dir,
        "template": model.template,
        "trust_remote_code": model.trust_remote_code,
        # export
        "export_dir": MERGED_OUTPUT_DIR,
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    header = (
        f"# Generated by SLM platform — export/merge config for {model.display_name}\n"
        f"# split: {split_id}\n"
        f"# Note: DO NOT use a quantized model or quantization_bit when merging.\n"
        f"# run: llamafactory-cli export export.yaml\n"
    )
    return header + _dump(cfg)


def render_all(model: ModelSpec, hp: Hyperparams, split_id: str, use_spot: bool = False) -> dict[str, str]:
    out = {
        "trainYaml": render_train_yaml(model, hp, split_id, use_spot=use_spot),
    }
    # FULL-WEIGHT (full/freeze) runs produce standalone weights with NO adapter, so
    # there is nothing to merge — the export.yaml is omitted entirely. The container
    # entrypoint skips its export step when no export.yaml is present (this absence
    # IS the signal). The entrypoint ALSO honors an optional SLM_SKIP_EXPORT=1 env as
    # a manual override, but the platform does not set it — omitting export.yaml is
    # the mechanism. Adapter methods (lora/qlora) keep the merge config as before.
    if hp.finetuning_type not in ("full", "freeze"):
        out["exportYaml"] = render_export_yaml(model, split_id, use_spot=use_spot)
    return out


def eval_env(decoding: "DecodingParams") -> dict[str, str]:
    """Env vars consumed by the container's eval.py.

    Eval has no LLaMA-Factory YAML — it's our own deterministic batch scorer. We
    pass decoding params as env so the SAME settings apply to every candidate in
    a comparison (methodology: identical decoding across models).
    """
    return {
        "SLM_MODE": "eval",
        "SLM_EVAL_BACKEND": decoding.backend,
        "SLM_EVAL_MAX_NEW_TOKENS": str(decoding.max_new_tokens),
        "SLM_EVAL_TEMPERATURE": str(decoding.temperature),
        "SLM_EVAL_TOP_P": str(decoding.top_p),
        "SLM_EVAL_SEED": str(decoding.seed),
    }
