#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CDK app for the SLM fine-tune/eval platform — full hosted web app.

One `cdk deploy` stands up the entire platform in the caller's own account and
outputs a CloudFront URL that serves a working app: data + SPA S3 buckets, the
API/reconcile Lambdas (container images CDK builds), HTTP API Gateway, Cognito,
CloudFront (+OAC, +WAF), an EventBridge reconcile schedule, the ECR repo, and a
CodeBuild project that builds the big GPU training image.

Region: the WAF WebACL for CloudFront must live in us-east-1, so the whole
stack deploys there (keeps it a single stack / single deploy). SageMaker g5 +
Bedrock Sonnet 4.5 are both available in us-east-1. Override with
`-c region=<other>` only if you also move the WAF out-of-stack — the pin ignores
CDK_DEFAULT_REGION on purpose, because the CDK CLI always sets it.
"""
import os

import aws_cdk as cdk
from aws_cdk import Aspects

import solution_config
from slm_platform_infra.stack import SlmPlatformInfraStack
from slm_platform_infra.user_agent import SolutionUserAgentAspect

app = cdk.App()

# Identify the AWS API calls this solution makes, so its API usage can be
# attributed to the solution and its version. Applied at app scope so every
# Lambda in every stack — including any added later — is covered; compute that
# is not a Lambda function has to be handed the value another way (see
# user_agent.py).
Aspects.of(app).add(SolutionUserAgentAspect(solution_config.user_agent_string()))

prefix = app.node.try_get_context("prefix") or "slm-platform"

# CloudFront-scoped WAF requires us-east-1, so the stack is pinned there.
#
# This reads CONTEXT, deliberately, and not CDK_DEFAULT_REGION: the CDK CLI sets
# CDK_DEFAULT_REGION on every invocation from the caller's resolved profile, so a
# `... or "us-east-1"` fallback on that variable can never fire under `cdk deploy`
# and the pin silently became "whatever region your AWS profile happens to name".
# A deployer whose profile said eu-west-1 got a stack that synthesized there and
# then failed, because CloudFormation only accepts scope=CLOUDFRONT WebACLs in
# us-east-1 (see the WebACL in slm_platform_infra/stack.py). Context is never
# auto-populated, so the default here actually holds, and `-c region=<other>`
# remains available for anyone who moves the WAF out of this stack.
region = app.node.try_get_context("region") or "us-east-1"

# Cost-alerting (optional, opt-in): an email to subscribe to the cost-alert SNS
# topic + a monthly USD budget threshold. Provide via -c alertEmail=you@example.com
# (and optionally -c monthlyBudgetUsd=500). With no email the topic + budget +
# billing alarm are still created (so spend is tracked + alarms fire), just with
# no email subscription — set it later in the console or via a redeploy.
alert_email = app.node.try_get_context("alertEmail") or os.environ.get("SLM_ALERT_EMAIL") or ""
try:
    monthly_budget_usd = float(app.node.try_get_context("monthlyBudgetUsd") or 500)
except (TypeError, ValueError):
    monthly_budget_usd = 500.0

# SES from-address for race-finished notifications: a property of THIS
# deployment, not of the code, so the shipped default lives in
# project_config.json rather than as a literal here or in the stack. Empty is a
# supported value — notify.py then skips sending (and logs it) instead of
# failing every race against an SES identity the account cannot verify.
# -c notifyFromEmail=... overrides for a single deploy.
notify_from_email = (
    app.node.try_get_context("notifyFromEmail")
    or os.environ.get("SLM_NOTIFY_FROM_EMAIL_OVERRIDE")
    or solution_config.notify_from_email()
)

SlmPlatformInfraStack(
    app,
    "SlmPlatformInfra",
    prefix=prefix,
    alert_email=alert_email,
    monthly_budget_usd=monthly_budget_usd,
    notify_from_email=notify_from_email,
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=region,
    ),
    # Deployments of this solution are counted by matching the solution id in
    # the description of each deployed CloudFormation stack — see
    # solution_config.stack_description(). This app deploys ONE stack, which is
    # therefore the top of the (single-node) dependency graph and takes the bare
    # id; a supporting stack added later must pass a component name so that one
    # install still counts as one deployment.
    description=solution_config.stack_description(),
)

app.synth()
