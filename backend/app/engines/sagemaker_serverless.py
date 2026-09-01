# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""SageMaker serverless model-customization engine (SFT / DPO).

Validated end-to-end: SFT and RLVR jobs completed on qwen3-4b and produced a
merged HF artifact that loads in container/eval.py. This module
implements SFT + DPO launch (RLVR, which needs a reward Evaluator, is not launched
here).

Gated by the saved enableSagemakerServerless flag (see engines/base.get_engine);
it defaults ON and the Settings toggle is the single switch. Heavy SDK deps
(sagemaker-train + sagemaker-core) are imported lazily
INSIDE launch_training_job, never at module load — so this file is import-safe
even when those dists aren't installed, and the LLaMA-Factory path is never at
risk from a serverless dependency problem.

What it does at launch:
  1. Reshape the on-disk split (messages / ranking) → the recipe's required
     prompt/completion (SFT) or prompt/chosen/rejected (DPO) string JSONL.
  2. Upload to S3 under the SAME per-tenant prefix scheme as orchestrate.
  3. Ensure the per-tenant ModelPackageGroup exists (the trainer resolves its
     name→ARN and does NOT auto-create it).
  4. Map our Hyperparams → the recipe knob names (enum/range-snapped).
  5. Launch SFTTrainer/DPOTrainer with our exec role + base_job_name in the
     slm-<model>-<split>-<stamp> convention.
The produced merged model lives at <S3ModelArtifacts>/checkpoints/hf_merged/ —
the race eval bridge points there (see race.py engine-aware eval).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DEFAULT_ENGINE  # noqa: F401  (kept for symmetry)

# Recipe enum sets (verified against the live qwen3-4b recipe override params,
# s3://jumpstart-cache-prod-us-east-1/recipes/*_override_params_sm_jobs_*.json).
_LORA_RANK_ENUM = (8, 16, 32, 64, 128)
_LORA_ALPHA_ENUM = (16, 32, 64, 128, 256)
# SFT/DPO global_batch_size enum (floor 8).
_GLOBAL_BATCH_ENUM = (8, 16, 32, 64, 128, 256, 512, 1024)
# RLVR/GRPO uses a DIFFERENT, much larger global_batch_size enum — floor 128, NOT
# 8 — verified against the live verl-grpo-rlvr-qwen-3-{0.6,4,8,14}b recipes
# (all share [128,256,512,1024]). This is why an RLVR job needs at least ~128
# training rows: there is no valid batch below 128. Snapping to the SFT enum (8)
# produces an off-enum value the recipe silently rejects → the recipe default of
# 128 stays → a split under ~128 rows builds zero batches ("Train dataloader is
# empty").
_RLVR_GLOBAL_BATCH_ENUM = (128, 256, 512, 1024)
_LR_MAX = 1.0e-4  # SFT/DPO recipe caps learning_rate at 1e-4
# GRPO (RLVR/RLAIF) is RL, not supervised — it needs a MUCH lower LR than SFT/DPO.
# The live managed GRPO recipes (verl-grpo-rlvr/-rlaif *_override_params_sm_jobs)
# specify learning_rate {default 1e-5, min 1e-7, max 1e-3} for every model we serve
# (verified 2026-06-29 against the qwen3-{1.7,4,8}b + deepseek-llama-70b recipes).
# Our shared SFT default is 1e-4 — 10x the GRPO recipe default and a poor RL LR
# (risks an unstable / wasted billable GRPO run). So for GRPO we (a) treat an LR at
# or above the SFT default as "not deliberately lowered for RL" and snap it to the
# recipe default, and (b) clamp the ceiling to the recipe max. A user who explicitly
# sets a low GRPO LR (< the SFT default) is respected, only range-clamped.
_GRPO_DEFAULT_LR = 1.0e-5
_GRPO_LR_MAX = 1.0e-3
_GRPO_LR_MIN = 1.0e-7
# An LR >= this is assumed to be the inherited SFT default (not an RL-tuned value),
# so GRPO snaps it down to _GRPO_DEFAULT_LR. Set to the SFT default.
_GRPO_INHERITED_SFT_LR = 1.0e-4
# When no validation file is uploaded, the recipe carves its own val split from
# the train set at this ratio (default 0.9 train — observed "40 -> 37 train" in a
# real job log). We use it to estimate the EFFECTIVE training rows the dataloader
# sees, so the batch is clamped to fit and the RLVR floor check is honest.
_RECIPE_TRAIN_SPLIT = 0.9


def _snap(value: int, allowed: tuple[int, ...]) -> int:
    """Nearest allowed enum value (recipe rejects off-enum values)."""
    return min(allowed, key=lambda a: abs(a - value))


def _snap_down(value: int, allowed: tuple[int, ...]) -> int:
    """Largest allowed enum value <= value (or the smallest enum if value is below
    all of them). Used to fit the batch UNDER the dataset size — _snap rounds to
    NEAREST and could round 100 UP to 128, which would overflow a 100-row split."""
    below = [a for a in allowed if a <= value]
    return max(below) if below else min(allowed)


def map_hyperparameters(hp: Any, train_rows: int | None = None,
                        raw_train_rows: int | None = None) -> dict[str, Any]:
    """Translate our Hyperparams → serverless recipe knobs (enum/range-snapped).

    Pure + unit-testable. Returns the overrides we set on trainer.hyperparameters.
    Only knobs the recipe exposes; others (warmup, save_steps, …) are left at the
    recipe default.

    global_batch_size is derived from our per-device batch × grad-accum, snapped to
    the recipe's per-stage enum. When `train_rows` (the EFFECTIVE train-split row
    count the recipe will actually use — after its internal val carve-out) is
    provided, the batch is also clamped DOWN so it never exceeds the dataset —
    otherwise the trainer builds zero batches and aborts with "Train dataloader is
    empty". RLVR's batch enum floor is 128, so a too-small RLVR dataset raises a
    clear, actionable error BEFORE a billable launch (the only remedy is more
    examples — the batch can't go below 128). `raw_train_rows` (the count the user
    actually uploaded, before the recipe's carve-out) is used only to make that
    error message accurate — it tells the user the real raw-row bar, not the
    post-split number."""
    stage = getattr(hp, "stage", "sft")
    # RLVR and RLAIF are both GRPO → the larger batch enum (floor 128). SFT/DPO
    # use the floor-8 enum.
    batch_enum = _RLVR_GLOBAL_BATCH_ENUM if stage in ("rlvr", "rlaif") else _GLOBAL_BATCH_ENUM
    floor = min(batch_enum)
    rank = _snap(int(hp.lora_rank), _LORA_RANK_ENUM)
    alpha = _snap(int(hp.lora_alpha if hp.lora_alpha is not None else hp.lora_rank * 2),
                  _LORA_ALPHA_ENUM)
    desired = max(floor, int(hp.per_device_train_batch_size) * int(hp.gradient_accumulation_steps))
    # RLVR/GRPO is the hard case: its batch enum floor is 128 and there is NO valid
    # smaller batch, so a split below 128 (AFTER the recipe's internal val carve-out)
    # CANNOT run (VERL aborts at dataloader construction). Fail fast with an
    # actionable error. The check is on EFFECTIVE rows (what VERL trains on), but the
    # message states the RAW-row bar so the user knows exactly how many examples to
    # supply — int(raw*0.9) >= 128 means raw >= 143 with no val file. (SFT/DPO floor
    # is 8 and the recipe tolerates tiny splits, so they only snap down, never
    # hard-fail — keeps small smoke tests working.)
    if stage in ("rlvr", "rlaif") and train_rows is not None and train_rows < floor:
        # Smallest raw upload that clears the floor after an int(raw*0.9) carve-out.
        import math as _math
        raw_needed = _math.ceil(floor / _RECIPE_TRAIN_SPLIT) if raw_train_rows is not None else floor
        have = raw_train_rows if raw_train_rows is not None else train_rows
        carve_note = (
            f" (the recipe holds out ~{int((1 - _RECIPE_TRAIN_SPLIT) * 100)}% for "
            f"validation, leaving only {train_rows} to train on)"
            if raw_train_rows is not None and raw_train_rows != train_rows
            else ""
        )
        raise ValueError(
            f"{stage.upper()}/GRPO needs at least {raw_needed} training examples "
            f"(the recipe's smallest global_batch_size is {floor} and there is no "
            f"smaller valid batch), but you provided {have}{carve_note}. "
            f"Add more examples, or upload a separate validation file so the full "
            f"training set is used."
        )
    if train_rows is not None:
        # Fit the batch under the dataset (snap DOWN so it stays a valid enum value).
        gbs = _snap_down(min(desired, max(1, train_rows)), batch_enum)
    else:
        gbs = _snap(desired, batch_enum)
    if stage in ("rlvr", "rlaif"):
        # GRPO needs a much lower LR than SFT (recipe default 1e-5). An LR at/above
        # the SFT default is treated as inherited (not RL-tuned) and snapped to the
        # GRPO default; everything is range-clamped to the recipe's [1e-7, 1e-3].
        lr = float(hp.learning_rate)
        if lr >= _GRPO_INHERITED_SFT_LR:
            lr = _GRPO_DEFAULT_LR
        lr = max(_GRPO_LR_MIN, min(lr, _GRPO_LR_MAX))
    else:
        lr = min(float(hp.learning_rate), _LR_MAX)
    out: dict[str, Any] = {
        "max_epochs": max(1, int(round(hp.num_train_epochs))),
        "learning_rate": lr,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "global_batch_size": gbs,
    }
    if hp.cutoff_len:
        out["dataset_max_len"] = int(hp.cutoff_len)
    return out


def _serverless_python() -> str:
    """The interpreter that has the V3 SageMaker SDK (sagemaker.train.*). The
    main process runs the V2 SDK (sagemaker.estimator) which is MUTUALLY EXCLUSIVE
    with V3, so the trainer call must run under this separate interpreter. Set via
    SLM_SERVERLESS_PYTHON (a venv with sagemaker-train + sagemaker-core 2.x). No
    default guess — if unset, the engine raises a clear, actionable error."""
    import os

    py = os.environ.get("SLM_SERVERLESS_PYTHON", "").strip()
    if not py:
        raise RuntimeError(
            "SLM_SERVERLESS_PYTHON is not set. The serverless engine needs a "
            "separate Python with the V3 SageMaker SDK (sagemaker-train + "
            "sagemaker-core 2.x), because V3 is incompatible with the V2 SDK the "
            "LLaMA-Factory path uses. Point SLM_SERVERLESS_PYTHON at that venv's "
            "python."
        )
    return py


def _launch_via_subprocess(spec: dict[str, Any]) -> dict[str, Any]:
    """Run the trainer launch in the V3-SDK interpreter (process isolation).
    Writes the spec to a temp file, runs serverless_launcher.py, parses the JSON
    result line. Raises on a non-zero exit / error payload."""
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    launcher = str(Path(__file__).with_name("serverless_launcher.py"))
    py = _serverless_python()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
        spec_path = f.name
    try:
        # args are a trusted, env-pinned interpreter (SLM_SERVERLESS_PYTHON) + our own
        # launcher script + a temp spec path; no shell, no untrusted input.
        proc = subprocess.run(  # nosec B603
            [py, launcher, spec_path],
            capture_output=True, text=True, timeout=600,
        )
    finally:
        try:
            Path(spec_path).unlink()
        except OSError:
            pass
    # The launcher prints one JSON line; find the last JSON object in stdout.
    out = (proc.stdout or "").strip().splitlines()
    parsed = None
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
                break
            except ValueError:
                continue
    if parsed is None:
        # No JSON result line at all (e.g. interpreter/import crash, OOM). Surface
        # the tail of stderr so the failure is diagnosable, not silent.
        raise RuntimeError(
            f"serverless launcher produced no result "
            f"(exit {proc.returncode}): {(proc.stderr or proc.stdout or '')[-800:]}"
        )
    if "error" in parsed:
        raise RuntimeError(f"serverless launch failed: {parsed['error']}")
    # The subprocess serves two ops: a TRAINING launch returns {jobName}, and the
    # register-evaluator op returns {evaluatorArn}. Accept either success shape.
    if "jobName" not in parsed and "evaluatorArn" not in parsed:
        raise RuntimeError(f"serverless launcher returned no jobName/evaluatorArn: {parsed}")
    if proc.returncode != 0:
        # Defensive: a result WITHOUT a clean exit shouldn't happen, but if it
        # does, don't trust it — treat as a failure.
        raise RuntimeError(
            f"serverless launcher exited {proc.returncode} despite a result: {parsed}"
        )
    return parsed


class SagemakerServerlessEngine:
    name = "sagemaker_serverless"

    def launch_training_job(
        self,
        model: Any,
        split_id: str,
        hp: Any,
        instance_type: str,
        stamp: str,
        max_run_seconds: int,
        use_spot: bool,
        image_tag: str | None,
    ) -> dict[str, Any]:
        serverless_id = getattr(model, "serverless_model_id", "") or ""
        if not serverless_id:
            raise ValueError(
                f"model {model.id!r} has no serverless_model_id — the serverless "
                "engine is not available for it (no SageMaker Public Hub mapping)."
            )
        stage = getattr(hp, "stage", "sft")
        if stage not in ("sft", "dpo", "rlvr", "rlaif"):
            raise ValueError(
                f"serverless engine supports stage sft|dpo|rlvr|rlaif, got {stage!r}"
            )
        # RLVR needs a verifiable reward: EITHER a PRESET (gsm8k/prime_math/
        # prime_code) OR a CUSTOM reward function (reward_function_id → its
        # Evaluator ARN). Validated in Hyperparams; re-checked here so a direct
        # engine call can't launch a billable reward-less job. A custom reward must
        # already be DEPLOYED (Lambda + Evaluator) — we resolve its ARN from the
        # registry and fail fast if it isn't.
        # RLAIF needs a reward PROMPT (an AI judge): reward_function_id → a
        # 'reward_prompt'-kind record holding the prompt text + optional judge model
        # id, passed INLINE to RLAIFTrainer (no Lambda/Evaluator to resolve).
        preset_reward = ""
        # The preset NAME the user chose (gsm8k/prime_math), preserved for display in
        # the run result even though it's no longer sent to the recipe — the reward
        # is now carried by an auto-provisioned built-in reward Evaluator ARN.
        preset_reward_label = ""
        custom_reward_evaluator_arn = ""
        reward_prompt_evaluator_arn = ""
        reward_model_id = ""
        if stage == "rlvr":
            preset_reward = getattr(hp, "preset_reward_function", "") or ""
            preset_reward_label = preset_reward
            reward_fn_id = getattr(hp, "reward_function_id", "") or ""
            if not preset_reward and not reward_fn_id:
                raise ValueError(
                    "RLVR requires a preset_reward_function (gsm8k|prime_math) "
                    "or a custom reward_function_id"
                )
            if reward_fn_id:
                from ..reward_functions import get_reward_function

                rf = get_reward_function(reward_fn_id)
                if rf is None:
                    raise ValueError(f"reward function {reward_fn_id!r} not found")
                custom_reward_evaluator_arn = rf.get("evaluatorArn") or ""
                if not custom_reward_evaluator_arn:
                    raise ValueError(
                        f"reward function {reward_fn_id!r} is not deployed yet "
                        "(no Evaluator ARN) — deploy it before launching RLVR"
                    )
            elif preset_reward:
                # AWS removed `preset_reward_function` from the open-weight GRPO
                # recipe (the RLVRTrainer takes ONLY a custom-reward Evaluator ARN
                # and deletes the preset key). So we RECONSTRUCT a preset as an
                # auto-provisioned built-in reward function and route it through the
                # SAME custom-reward Evaluator path — the recipe never sees the
                # rejected preset key. Idempotent per tenant (first use deploys the
                # Lambda + Evaluator; later launches reuse it).
                from ..reward_functions import ensure_preset_reward_evaluator_arn

                custom_reward_evaluator_arn = ensure_preset_reward_evaluator_arn(
                    preset_reward, stamp=stamp)
                # Don't send the preset key to the recipe (it would be rejected);
                # the reward is now carried entirely by the Evaluator ARN.
                preset_reward = ""
        elif stage == "rlaif":
            reward_fn_id = getattr(hp, "reward_function_id", "") or ""
            if not reward_fn_id:
                raise ValueError(
                    "RLAIF requires a reward_function_id pointing at a reward prompt"
                )
            from ..reward_functions import get_reward_function

            rf = get_reward_function(reward_fn_id)
            if rf is None:
                raise ValueError(f"reward function {reward_fn_id!r} not found")
            if rf.get("kind") != "reward_prompt":
                raise ValueError(
                    f"reward function {reward_fn_id!r} is a {rf.get('kind')!r} reward, "
                    "but RLAIF needs a 'reward_prompt' (AI-judge) reward"
                )
            # The reward prompt must be DEPLOYED (registered as a REWARD_PROMPT
            # Evaluator) — RLAIFTrainer takes the Evaluator ARN, not raw prompt text.
            reward_prompt_evaluator_arn = rf.get("evaluatorArn") or ""
            if not reward_prompt_evaluator_arn:
                raise ValueError(
                    f"reward prompt {reward_fn_id!r} is not deployed yet "
                    "(no Evaluator ARN) — deploy it before launching RLAIF"
                )
            # The judge model: per-reward override, else the hp's reward_model_id,
            # else "" (the recipe's default judge).
            reward_model_id = rf.get("rewardModelId") or getattr(hp, "reward_model_id", "") or ""

        # boto3 is V2/V3-agnostic (it's not the sagemaker SDK), so data prep +
        # upload + MPG ensure run in THIS process. The trainer call (V3-SDK-only)
        # is shelled out to a launcher under a V3 interpreter — see _launch_via_subprocess.
        from ..aws_clients import get_session
        from ..aws_config import load_aws_config
        from ..storage import split_dir
        from . import serverless_data as sd
        from .serverless_naming import (
            serverless_job_name,
            tenant_model_package_group,
            ensure_model_package_group,
            jobs_key_prefix,
        )

        cfg = load_aws_config()
        run_dir = split_dir(split_id)
        if run_dir is None:
            raise ValueError(f"split {split_id} not found on disk")

        boto_sess = get_session(profile_name=cfg.profile or None, region_name=cfg.region)
        s3 = boto_sess.client("s3")

        # 1. Reshape train (+val) to the recipe format. Keep the converted row
        # count — it gates the batch size (a too-small split builds zero batches).
        train_local = run_dir / "_serverless_train.jsonl"
        train_rows = sd.convert_file(run_dir / "train.jsonl", train_local, stage)
        val_src = run_dir / "val.jsonl"
        val_local = None
        if val_src.exists():
            val_local = run_dir / "_serverless_val.jsonl"
            sd.convert_file(val_src, val_local, stage)
        # When we DON'T pass a validation file, the recipe carves one out of the
        # train set itself (default train_val_split_ratio 0.9 — the failed e2e log
        # showed "40 -> 37 train"). The dataloader then sees only ~90% of the rows,
        # so the batch must fit UNDER that. Use the conservative effective count.
        effective_train_rows = train_rows if val_local is not None else int(train_rows * _RECIPE_TRAIN_SPLIT)

        # 2. Upload under the per-tenant jobs prefix (mirrors orchestrate layout).
        job_name = serverless_job_name(model.id, split_id, stamp, stage=stage)
        key_base = f"{jobs_key_prefix()}/{job_name}/dataset"
        s3.upload_file(str(train_local), cfg.bucket, f"{key_base}/train.jsonl")
        train_uri = f"s3://{cfg.bucket}/{key_base}/train.jsonl"
        val_uri = None
        if val_local is not None:
            s3.upload_file(str(val_local), cfg.bucket, f"{key_base}/val.jsonl")
            val_uri = f"s3://{cfg.bucket}/{key_base}/val.jsonl"

        # 3. Ensure the ModelPackageGroup exists (trainer won't create it).
        mpg = tenant_model_package_group()
        ensure_model_package_group(boto_sess, mpg)

        # 4. Launch the trainer in the V3-SDK subprocess.
        output_s3 = f"s3://{cfg.bucket}/{jobs_key_prefix()}/{job_name}/output/"
        spec = {
            "stage": stage,
            "serverlessModelId": serverless_id,
            "role": cfg.role_arn,
            "region": cfg.region,
            "profile": cfg.profile,
            "bucket": cfg.bucket,
            "trainUri": train_uri,
            "valUri": val_uri,
            "modelPackageGroup": mpg,
            "baseJobName": job_name,
            "outputS3": output_s3,
            "hyperparameters": map_hyperparameters(
                hp, train_rows=effective_train_rows, raw_train_rows=train_rows),
            # RLVR reward is ALWAYS a custom-reward Evaluator ARN now (the recipe no
            # longer accepts preset_reward_function). A user-chosen preset was
            # resolved above to an auto-provisioned built-in reward's ARN, so
            # presetRewardFunction is left EMPTY — the launcher must not setattr the
            # rejected key. customRewardEvaluatorArn carries the reward for both the
            # preset and custom paths. Empty for sft/dpo (launcher ignores both).
            "presetRewardFunction": preset_reward,
            "customRewardEvaluatorArn": custom_reward_evaluator_arn,
            # RLAIF-only reward: the registered REWARD_PROMPT Evaluator ARN (passed
            # to RLAIFTrainer.reward_prompt) + the judge model id (reward_model_id).
            # Empty for every other stage.
            "rewardPromptEvaluatorArn": reward_prompt_evaluator_arn,
            "rewardModelId": reward_model_id,
        }
        result = _launch_via_subprocess(spec)

        return {
            "jobName": result["jobName"],
            "engine": self.name,
            "stage": stage,
            "serverlessModelId": serverless_id,
            "instanceType": "serverless",
            "datasetS3": train_uri,
            "outputS3": output_s3,
            "modelPackageGroup": mpg,
            "appliedHyperparameters": result.get("appliedHyperparameters", {}),
            # The preset NAME the user chose (for display); the actual reward is the
            # built-in reward Evaluator ARN in customRewardEvaluatorArn.
            "presetRewardFunction": preset_reward_label,
            "customRewardEvaluatorArn": custom_reward_evaluator_arn,
            "rewardPromptEvaluatorArn": reward_prompt_evaluator_arn,
            "rewardModelId": reward_model_id,
            "useSpot": False,  # serverless manages capacity; spot is N/A
            "region": cfg.region,
        }
