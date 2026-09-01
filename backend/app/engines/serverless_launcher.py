# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Standalone launcher for a SageMaker serverless customization job.

WHY THIS IS A SEPARATE SCRIPT (run via subprocess, not imported):
The SageMaker V3 SDK that provides the serverless trainers (sagemaker.train.*,
sagemaker-core 2.x) is MUTUALLY EXCLUSIVE with the classic V2 SDK
(sagemaker.estimator / sagemaker-core 1.x) the LLaMA-Factory path depends on —
the resolver reports them "unsatisfiable" together, and V3 (`sagemaker` 3.x)
dropped `sagemaker.estimator` entirely. So we CANNOT import the trainers in the
main API/worker process. Instead the engine adapter shells out to THIS script,
which runs in a dedicated venv (SLM_SERVERLESS_PYTHON) that has the V3 SDK.

Contract: reads a JSON job spec on argv[1] (a file path), launches the trainer
with wait=False, and prints a single JSON line {"jobName": ...} to stdout (or
{"error": ...} + non-zero exit). Everything it needs is in the spec — it makes
NO imports from app.* so it's safe to run under a foreign interpreter.

Spec keys: stage, serverlessModelId, role, region, bucket, trainUri, valUri,
modelPackageGroup, baseJobName, outputS3, hyperparameters (dict),
customRewardEvaluatorArn (RLVR reward — the recipe takes ONLY this; presets are
resolved to a built-in reward's Evaluator ARN upstream), rewardPromptEvaluatorArn +
rewardModelId (RLAIF).
"""

from __future__ import annotations

import json
import sys


def _apply_global_batch_size(hpobj, overrides: dict, applied: dict, stage: str) -> None:
    """Force-set global_batch_size on the recipe hyperparameter object and verify
    it stuck. UNCONDITIONAL re-verify: the generic _specs loop may have setattr'd it
    WITHOUT checking the recipe actually stored our value (it records `applied` on
    any non-raising setattr), so membership in `applied` can't be trusted. For RLVR
    a mismatch is FATAL (RLVR AND RLAIF — both are GRPO) — the GRPO batch enum
    floor is 128 and a coerced/dropped value launches a job that builds zero/wrong
    batches; fail fast BEFORE the billable launch. For SFT/DPO a mismatch is
    advisory (just not recorded).

    Pure-ish: mutates hpobj + applied, raises ValueError on an RLVR mismatch. Split
    out so it's unit-testable without the V3 SDK."""
    if "global_batch_size" not in overrides:
        return
    want = overrides["global_batch_size"]
    applied.pop("global_batch_size", None)  # re-verify from scratch
    rejected_exc = None
    try:
        setattr(hpobj, "global_batch_size", want)
    except Exception as e:  # noqa: BLE001 — recipe rejected the assignment
        rejected_exc = e
    readback = getattr(hpobj, "global_batch_size", None)
    if rejected_exc is None and readback == want:
        applied["global_batch_size"] = want
    elif stage in ("rlvr", "rlaif"):
        raise ValueError(
            f"recipe did not accept global_batch_size={want!r} "
            f"(read back {readback!r}"
            + (f", error {rejected_exc}" if rejected_exc else "")
            + f"). {stage.upper()}/GRPO needs this exact batch to fit the dataset; the "
            "recipe coerced or dropped it, which would launch a job that builds zero "
            "batches. Aborting before the billable launch."
        )


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: serverless_launcher.py <spec.json>"}))
        return 2
    try:
        spec = json.loads(open(sys.argv[1], encoding="utf-8").read())
    except Exception as e:  # noqa: BLE001 — surface as structured error
        print(json.dumps({"error": f"could not read spec: {e}"}))
        return 2
    try:
        import boto3

        # Session class location differs between SDK layouts: the V3 modular SDK
        # (what the serverless trainers ship in) exposes it at
        # sagemaker.core.helper.session_helper; classic V2 has sagemaker.session.
        # Try V3 first, fall back to V2, so the launcher works under either venv.
        try:
            from sagemaker.core.helper.session_helper import Session
        except Exception:  # noqa: BLE001
            from sagemaker.session import Session

        # Attribute this process's AWS calls to the solution, like the rest of
        # the app does. This script deliberately imports nothing from app.* (it
        # runs under a foreign interpreter), so the shared helper is inlined here
        # from the environment the parent process passes down. Absent variable =
        # simply unattributed.
        #
        # It patches the session CONSTRUCTOR, not just the session built below.
        # Handing our session to the trainer does not make the SDK route
        # everything through it: sagemaker-core builds clients off sessions of
        # its own, and those are most of the calls this script causes — the
        # CreateTrainingJob included. Only the constructor reaches them.
        import os

        import botocore.session

        # These two names are the ones app.aws_clients marks its own patch with.
        # They are repeated as literals because this module may not import from
        # app.*; keeping them identical is what makes the patch idempotent (a
        # second call must not wrap it again) and removable by that module.
        _installed_flag = "_slm_solution_user_agent_installed"
        _original_attr = "_slm_solution_user_agent_original_init"

        ua_suffix = os.environ.get("USER_AGENT_STRING", "").strip()
        if ua_suffix and not getattr(
            botocore.session.Session.__init__, _installed_flag, False
        ):
            _botocore_session_init = botocore.session.Session.__init__

            def _init_with_solution_user_agent(self, *a, **kw):
                _botocore_session_init(self, *a, **kw)
                # The user-agent is a space-delimited token list: append to
                # whatever is already there, and only once.
                existing = self.user_agent_extra or ""
                if ua_suffix not in existing.split():
                    self.user_agent_extra = f"{existing} {ua_suffix}".strip()

            setattr(_init_with_solution_user_agent, _installed_flag, True)
            setattr(_init_with_solution_user_agent, _original_attr, _botocore_session_init)
            botocore.session.Session.__init__ = _init_with_solution_user_agent

        boto_sess = boto3.Session(
            profile_name=spec.get("profile") or None,
            region_name=spec["region"],
        )
        sm_sess = Session(boto_session=boto_sess)

        # --- op: register a SageMaker Evaluator (V3 SDK, so done HERE in the
        # subprocess, not the main V2 process). Two kinds:
        #   * RLVR custom reward  → type=RewardFunction, source=<lambda ARN>
        #   * RLAIF reward prompt → type=REWARD_PROMPT, source=<S3 prompt URI>
        # Returns the hub-content ARN. No trainer. ---
        if spec.get("op") == "register_evaluator":
            from sagemaker.ai_registry.evaluator import Evaluator

            ev_type = spec.get("evaluatorType") or "RewardFunction"
            source = spec.get("source") or spec.get("lambdaArn")
            ev = Evaluator.create(
                name=spec["evaluatorName"],
                type=ev_type,
                source=source,
                role=spec["role"],
                sagemaker_session=sm_sess,
            )
            arn = (getattr(ev, "arn", None) or getattr(ev, "evaluator_arn", None)
                   or getattr(ev, "hub_content_arn", None))
            print(json.dumps({"evaluatorArn": arn}))
            return 0

        from sagemaker.train.common import TrainingType
        from sagemaker.train.sft_trainer import SFTTrainer
        from sagemaker.train.dpo_trainer import DPOTrainer

        stage = spec["stage"]
        common_kwargs = dict(
            model=spec["serverlessModelId"],
            training_type=TrainingType.LORA,
            model_package_group=spec["modelPackageGroup"],
            training_dataset=spec["trainUri"],
            validation_dataset=spec.get("valUri"),
            s3_output_path=spec["outputS3"],
            accept_eula=True,
            sagemaker_session=sm_sess,
            role=spec["role"],
            base_job_name=spec["baseJobName"],
        )
        if stage == "rlvr":
            # RLVR (GRPO against a verifiable reward). The current AWS open-weight
            # GRPO recipe / RLVRTrainer takes the reward as a SINGLE constructor arg
            # `custom_reward_function` (an Evaluator ARN) and EXPLICITLY deletes any
            # `preset_reward_function` recipe hyperparameter (verified against
            # sagemaker-train 1.13.1). There is no longer a preset recipe knob, so the
            # engine resolves a user-chosen preset to an auto-provisioned built-in
            # reward's Evaluator ARN BEFORE calling us — both the preset and custom
            # paths arrive here as customRewardEvaluatorArn.
            from sagemaker.train.rlvr_trainer import RLVRTrainer

            custom_arn = spec.get("customRewardEvaluatorArn") or ""
            if not custom_arn:
                raise ValueError(
                    "RLVR launch needs a custom reward Evaluator ARN "
                    "(presets are resolved to a built-in reward Evaluator upstream)"
                )
            trainer = RLVRTrainer(custom_reward_function=custom_arn, **common_kwargs)
        elif stage == "rlaif":
            # RLAIF (GRPO from AI feedback). The reward is an AI JUDGE driven by a
            # reward PROMPT registered as a SageMaker Evaluator (type=REWARD_PROMPT).
            # `reward_prompt` here is that Evaluator's ARN (the SDK's Union[str,...]
            # string form is an Evaluator ARN / hub-content NAME / "Builtin.*", NOT
            # the raw prompt text — a raw string fails the hubContentName regex).
            # reward_model_id names the judge model ("" = recipe default).
            from sagemaker.train.rlaif_trainer import RLAIFTrainer

            reward_prompt_arn = spec.get("rewardPromptEvaluatorArn") or ""
            if not reward_prompt_arn.strip():
                raise ValueError("RLAIF launch needs a registered reward-prompt Evaluator ARN")
            rlaif_kwargs = dict(common_kwargs)
            reward_model_id = spec.get("rewardModelId") or ""
            if reward_model_id:
                rlaif_kwargs["reward_model_id"] = reward_model_id
            trainer = RLAIFTrainer(reward_prompt=reward_prompt_arn, **rlaif_kwargs)
        elif stage == "dpo":
            trainer = DPOTrainer(**common_kwargs)
        else:
            trainer = SFTTrainer(**common_kwargs)
        # Apply hyperparameter overrides the recipe exposes (skip unknown knobs).
        # The _specs membership gate is for GENERIC numeric overrides (lr, rank, …)
        # where skipping an unknown knob is the right thing.
        overrides = dict(spec.get("hyperparameters") or {})
        hpobj = trainer.hyperparameters
        applied = {}
        for k, v in overrides.items():
            if hasattr(hpobj, "_specs") and k not in getattr(hpobj, "_specs", {}):
                continue
            try:
                setattr(hpobj, k, v)
                applied[k] = v
            except Exception:  # noqa: BLE001 — recipe rejected the value; leave default
                pass

        # global_batch_size is LOAD-BEARING for RLVR — verify it stuck (fail fast on
        # a coerced/dropped value before the billable launch). See the helper.
        _apply_global_batch_size(hpobj, overrides, applied, stage)

        # NOTE: there is intentionally NO preset_reward_function handling here. The
        # current GRPO recipe / RLVRTrainer rejects (deletes) that hyperparameter;
        # the reward is supplied ONLY as the custom_reward_function constructor arg
        # above. A user-chosen preset is resolved to a built-in reward's Evaluator
        # ARN by the engine BEFORE this launcher runs.

        tj = trainer.train(
            training_dataset=spec["trainUri"],
            validation_dataset=spec.get("valUri"),
            wait=False,
        )
        name = getattr(tj, "training_job_name", None) or getattr(tj, "name", None) or spec["baseJobName"]
        print(json.dumps({"jobName": name, "appliedHyperparameters": applied}))
        return 0
    except Exception as e:  # noqa: BLE001 — surface as structured error to the parent
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
