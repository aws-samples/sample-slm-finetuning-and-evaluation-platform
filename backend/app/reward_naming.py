# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Naming + role helpers for RLVR reward functions. Pure (no AWS) so they're
unit-testable. The reward Lambda runs on a DEDICATED least-priv exec role
(`slm-rlvr-reward-lambda-exec`, AWSLambdaBasicExecutionRole only — it does
pure string scoring, needs nothing else), provisioned in CDK. The SageMaker exec
role gets `lambda:InvokeFunction` on these reward Lambdas (also in CDK)."""
from __future__ import annotations

from typing import Any


def reward_evaluator_name(reward_id: str) -> str:
    """SageMaker Evaluator (hub-content) name for a reward id."""
    return f"slm-rlvr-reward-{reward_id}"[:63]


def reward_lambda_exec_role_arn(cfg: Any) -> str:
    """ARN of the dedicated reward-Lambda execution role. Overridable via env
    SLM_REWARD_LAMBDA_ROLE_ARN (set in CDK); falls back to the conventional name
    in the same account so a dev run works without extra config."""
    import os

    env = os.environ.get("SLM_REWARD_LAMBDA_ROLE_ARN", "").strip()
    if env:
        return env
    return f"arn:aws:iam::{cfg.account_id}:role/slm-rlvr-reward-lambda-exec"
