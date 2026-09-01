# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deploy a reward function to AWS: create/update the reward Lambda (boto3) and
register it as a SageMaker Evaluator (V3 SDK, via the serverless subprocess).

Split from reward_functions.py so the build/validation/registry layer stays
unit-testable with no AWS. Everything here touches AWS and is exercised by a real
run, not unit tests (same as the serverless launcher).

Two steps:
  1. Lambda  — create (or update code of) `slm-rlvr-reward-<id>` from the zip
     (handler + scoring + user_reward), on the dedicated reward-Lambda exec role.
     boto3 is V2/V3-agnostic so this runs in-process.
  2. Evaluator — `Evaluator.create(type="RewardFunction", source=<lambda_arn>,
     role=<sm_exec>)` returns a JsonDoc hub-content ARN. Evaluator ships in the V3
     SDK (sagemaker.ai_registry.evaluator), which is mutually exclusive with the
     V2 SDK the main process uses — so it's delegated to the V3 subprocess
     (engines/serverless_launcher.py grows an `op: "register_evaluator"` mode).
"""
from __future__ import annotations

from typing import Any

# Reward Lambda runtime/config. 60s/512MB is plenty —
# scoring is pure-python string work over a rollout batch.
_LAMBDA_RUNTIME = "python3.12"
_LAMBDA_HANDLER = "handler.handler"
_LAMBDA_TIMEOUT = 60
_LAMBDA_MEMORY = 512


def reward_lambda_name(reward_id: str) -> str:
    """Deterministic Lambda name for a reward id (idempotent create/update)."""
    return f"slm-rlvr-reward-{reward_id}"[:64]


def ensure_reward_lambda(boto_sess: Any, reward_id: str, zip_bytes: bytes,
                         reward_lambda_role_arn: str) -> str:
    """Create the reward Lambda, or update its code if it already exists. Returns
    the function ARN. Idempotent: re-deploying the same reward id updates in place
    (the name is derived from the snippet hash, so identical snippets share one)."""
    lam = boto_sess.client("lambda")
    name = reward_lambda_name(reward_id)
    try:
        existing = lam.get_function(FunctionName=name)
        lam.update_function_code(FunctionName=name, ZipFile=zip_bytes, Publish=True)
        return existing["Configuration"]["FunctionArn"]
    except Exception:  # noqa: BLE001 — not found (or transient); create it
        pass
    resp = lam.create_function(
        FunctionName=name,
        Runtime=_LAMBDA_RUNTIME,
        Role=reward_lambda_role_arn,
        Handler=_LAMBDA_HANDLER,
        Code={"ZipFile": zip_bytes},
        Timeout=_LAMBDA_TIMEOUT,
        MemorySize=_LAMBDA_MEMORY,
        Description=f"SLM RLVR custom reward {reward_id}",
        Publish=True,
        Tags={"slm-platform": "rlvr-reward"},
    )
    return resp["FunctionArn"]


def register_evaluator_via_subprocess(spec: dict[str, Any]) -> dict[str, Any]:
    """Register the reward Lambda as a SageMaker Evaluator in the V3 subprocess
    (Evaluator lives in the V3 SDK). spec: {op:"register_evaluator", region,
    profile, role, evaluatorName, lambdaArn}. Returns {"evaluatorArn": ...}."""
    from .engines.sagemaker_serverless import _launch_via_subprocess

    return _launch_via_subprocess(spec)


def deploy_reward_prompt(reward_id: str, prompt_text: str,
                         prompt_key: str | None = None) -> dict[str, str]:
    """Deploy an RLAIF reward PROMPT: upload the prompt text to S3, then register
    it as a SageMaker Evaluator (type=REWARD_PROMPT). Returns {"evaluatorArn",
    "promptS3Uri"}. NO Lambda — the prompt IS the artifact (the judge reads it).

    Mirrors deploy_reward_function's Evaluator step but with an S3 prompt URI source
    instead of a Lambda ARN. `prompt_key` (the prompt hash) names the S3 object +
    Evaluator so identical prompts reuse one set of resources (idempotent)."""
    from .aws_clients import get_session
    from .aws_config import load_aws_config
    from .reward_naming import reward_evaluator_name

    key = prompt_key or reward_id
    cfg = load_aws_config()
    boto_sess = get_session(profile_name=cfg.profile or None, region_name=cfg.region)
    s3 = boto_sess.client("s3")

    # Upload the prompt to a stable, hash-keyed S3 object (idempotent reuse).
    s3_key = f"slm-platform/reward_prompts/{key}.txt"
    s3.put_object(Bucket=cfg.bucket, Key=s3_key, Body=prompt_text.encode("utf-8"))
    prompt_s3_uri = f"s3://{cfg.bucket}/{s3_key}"

    result = register_evaluator_via_subprocess({
        "op": "register_evaluator",
        "region": cfg.region,
        "profile": cfg.profile,
        "role": cfg.role_arn,
        "evaluatorName": reward_evaluator_name(key),
        "evaluatorType": "RewardPrompt",  # air_constants.REWARD_PROMPT value
        "source": prompt_s3_uri,
    })
    evaluator_arn = result.get("evaluatorArn")
    if not evaluator_arn:
        raise RuntimeError(f"reward-prompt evaluator registration returned no ARN: {result}")
    return {"evaluatorArn": evaluator_arn, "promptS3Uri": prompt_s3_uri}


def deploy_reward_function(reward_id: str, name: str, zip_bytes: bytes,
                           lambda_key: str | None = None) -> dict[str, str]:
    """Full deploy: ensure the Lambda, then register the Evaluator. Returns
    {"lambdaArn", "evaluatorArn"}. Resolves the exec role + reward-Lambda role +
    region/profile from aws_config. Raises on failure (the caller records the
    reward function as not-yet-deployed).

    `lambda_key` is the AWS resource key (the snippet hash) used to NAME the Lambda
    + Evaluator, so identical snippets reuse one set of AWS resources (idempotent).
    Falls back to `reward_id` for back-compat when not supplied."""
    from .aws_clients import get_session
    from .aws_config import load_aws_config
    from .reward_naming import reward_evaluator_name, reward_lambda_exec_role_arn

    key = lambda_key or reward_id
    cfg = load_aws_config()
    boto_sess = get_session(profile_name=cfg.profile or None, region_name=cfg.region)

    lambda_arn = ensure_reward_lambda(
        boto_sess, key, zip_bytes, reward_lambda_exec_role_arn(cfg))

    result = register_evaluator_via_subprocess({
        "op": "register_evaluator",
        "region": cfg.region,
        "profile": cfg.profile,
        "role": cfg.role_arn,  # the SageMaker exec role the Evaluator runs under
        "evaluatorName": reward_evaluator_name(key),
        "lambdaArn": lambda_arn,
    })
    evaluator_arn = result.get("evaluatorArn")
    if not evaluator_arn:
        raise RuntimeError(f"evaluator registration returned no ARN: {result}")
    return {"lambdaArn": lambda_arn, "evaluatorArn": evaluator_arn}
