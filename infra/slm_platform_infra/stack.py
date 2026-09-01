# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""SLM platform — full hosted web-app stack (one `cdk deploy`).

Clone the repo and run `cdk deploy` in your own account; this stack stands up
everything and outputs a CloudFront URL that serves a working app. No manual
copy-paste, no manual image push, no credential wiring.

What it creates (all fresh, `slm-platform-*` named):

  data S3 bucket        datasets + job artifacts + app state (CloudStore prefix)
  SPA S3 bucket         the built React/Cloudscape frontend (private, OAC-only)
  ECR repo              the 16.7GB LLaMA-Factory training/eval image
  CodeBuild project     builds+pushes that training image during deploy
  API Lambda            FastAPI via Mangum (container image CDK builds)
  reconcile Lambda      same image, scheduled (EventBridge) to advance races
  SageMaker exec role   least-priv role SageMaker jobs assume
  HTTP API Gateway      → API Lambda, JWT-authorized by Cognito
  Cognito user pool     auth for the app
  CloudFront + OAC      serves SPA (default) and /api/* → API Gateway
  WAFv2 WebACL          on CloudFront (managed rules + rate limit)

The app's AWS identity is the API/reconcile Lambda execution role — no creds
anywhere. Account/region come from the CDK env (the deploying account).
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    Fn,
    RemovalPolicy,
    Stack,
    aws_apigatewayv2 as apigw,
    aws_apigatewayv2_authorizers as apigw_auth,
    aws_apigatewayv2_integrations as apigw_int,
    aws_budgets as budgets,
    aws_certificatemanager as acm,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_route53 as route53,
    aws_route53_targets as r53_targets,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_codebuild as codebuild,
    aws_cognito as cognito,
    aws_ecr as ecr,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    aws_secretsmanager as secretsmanager,
    aws_s3_deployment as s3deploy,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
    aws_wafv2 as wafv2,
    custom_resources as cr,
)
from constructs import Construct

# Repo paths (this file: infra/slm_platform_infra/stack.py → repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"
CONTAINER_DIR = REPO_ROOT / "container"

IMAGE_TAG = "0.9.4"  # the `stable` tier tag — also the default SLM_TRAINING_IMAGE_URI
STATE_PREFIX = "slm-platform/state"

# Multi-image tiers. ONE Docker image can't run EVERY model: newer models need a
# newer transformers/LLaMA-Factory stack than older ones, and a global bump risks
# breaking the ~30 models already proven on 0.9.4. So we build a SMALL fixed set
# of image tiers (not one-image-per-model) — each its own ECR tag, off its own
# LLaMA-Factory base — and a model declares which tier it runs on
# (ModelSpec.image_tag, resolved by the backend's aws_config.image_tiers()).
# Keep this in sync with backend/app/aws_config.IMAGE_TIER_TAGS.
#   image tag → (LLaMA-Factory base tag, vllm version targeting that base's torch)
IMAGE_TIERS: dict[str, dict[str, str]] = {
    "0.9.4": {"lf_base": "0.9.4", "vllm": "0.8.5.post1"},  # stable
    "0.9.5": {"lf_base": "0.9.5", "vllm": "0.8.5.post1"},  # latest (transformers v5)
}

# Where each built image records the set of model architectures + LLaMA-Factory
# templates it supports — so the Model Catalog's "find new models" can diff a new
# image against an old one WITHOUT needing a GPU to introspect transformers at
# request time (the build writes this JSON; the backend just reads it from S3).
IMAGE_META_PREFIX = "slm-platform/image-meta"


def _training_image_buildspec() -> dict:
    """Shared buildspec for every training-image build (per-tier + adhoc).

    Parameterized entirely by env vars (TAG / LF_BASE_TAG / VLLM_VERSION / REPO /
    ACCOUNT / REGION / BUCKET) so ONE spec serves all projects. Steps:
      1. ECR login.
      2. Pull the LF base with retry+backoff (rides out Docker Hub's 429 limit).
      3. docker build (passing the base tag + vllm version as build args).
      4. push to ECR.
      5. introspect the built image's transformers arch registry + LLaMA-Factory
         templates, and upload that capability manifest to s3://$BUCKET/image-meta/
         so model-discovery can diff images cheaply later.
    """
    # Python that introspects the image and prints a capability manifest as JSON
    # to STDOUT. Run inside the freshly-built image via `python -c`, captured to a
    # host file, then uploaded to S3. Kept on one logical line per statement so it
    # embeds cleanly in a `python -c '...'` shell command.
    capture_py = (
        "import json,os;"
        "out={'tag':os.environ.get('TAG'),'lf_base':os.environ.get('LF_BASE_TAG')};"
        "\ntry:\n from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C\n out['model_types']=sorted(C.keys())\nexcept Exception as e:\n out['model_types_error']=str(e)"
        "\ntry:\n import transformers; out['transformers']=transformers.__version__\nexcept Exception as e:\n out['transformers_error']=str(e)"
        "\ntry:\n from llamafactory.data.template import TEMPLATES; out['templates']=sorted(TEMPLATES.keys())\nexcept Exception as e:\n out['templates_error']=str(e)"
        "\nprint(json.dumps(out))"
    )
    img = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG"
    return {
        "version": "0.2",
        "phases": {
            "pre_build": {
                "commands": [
                    "echo logging in to ECR",
                    "aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com",
                ]
            },
            "build": {
                "commands": [
                    "echo building training image $REPO:$TAG from LLaMA-Factory $LF_BASE_TAG",
                    "ls -la",
                    # Pull the base image. PREFERRED path: the ECR PULL-THROUGH CACHE
                    # for Docker Hub ($LF_BASE_REGISTRY = …/docker-hub/hiyouga), which
                    # needs the Docker Hub PAT secret set (authenticated → no 429).
                    # The PAT is OPTIONAL: if the cache pull fails (PAT not set yet, or
                    # a transient miss), we FALL BACK to a direct Docker Hub pull with
                    # retry+backoff and a clear warning — the old flow, which usually
                    # works but can hit the unauthenticated 429. EFFECTIVE_REGISTRY is
                    # then passed to `docker build` so the Dockerfile FROM resolves to
                    # whichever source actually succeeded.
                    'EFFECTIVE_REGISTRY="$LF_BASE_REGISTRY"',
                    'if docker pull $LF_BASE_REGISTRY/llamafactory:$LF_BASE_TAG; then '
                    'echo "pulled base via ECR pull-through cache (authenticated)"; '
                    'else '
                    'echo "WARNING: cache pull failed (Docker Hub PAT secret may be unset — see README First-time setup). '
                    'Falling back to a DIRECT Docker Hub pull, which can hit the unauthenticated rate-limit (429)."; '
                    'EFFECTIVE_REGISTRY="hiyouga"; '
                    'for i in 1 2 3 4 5 6; do docker pull hiyouga/llamafactory:$LF_BASE_TAG && break || '
                    '(echo "direct base pull attempt $i failed (likely Docker Hub 429); backing off"; sleep $((i*30))); done; '
                    'fi',
                    f"docker build --build-arg LF_BASE_TAG=$LF_BASE_TAG --build-arg LF_BASE_REGISTRY=$EFFECTIVE_REGISTRY --build-arg VLLM_VERSION=$VLLM_VERSION -t {img} -f Dockerfile .",
                ]
            },
            "post_build": {
                "commands": [
                    f"docker push {img}",
                    # Capability manifest: introspect the built image (arch registry +
                    # LF templates), write JSON to S3 so "find new models" can diff
                    # images without a GPU. Best-effort — never fail the build on it.
                    f"docker run --rm -e TAG=$TAG -e LF_BASE_TAG=$LF_BASE_TAG --entrypoint python {img} -c {_shq(capture_py)} > /tmp/image_meta.json || echo '{{}}' > /tmp/image_meta.json",
                    "cat /tmp/image_meta.json",
                    f"aws s3 cp /tmp/image_meta.json s3://$BUCKET/{IMAGE_META_PREFIX}/$TAG.json || echo 'meta upload skipped'",
                ]
            },
        },
    }


def _shq(s: str) -> str:
    """Single-quote a string for safe embedding in a shell command."""
    return "'" + s.replace("'", "'\\''") + "'"


class SlmPlatformInfraStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, prefix: str,
        alert_email: str = "", monthly_budget_usd: float = 500.0,
        # Empty by default on purpose: with no sender, notify.py skips sending
        # and logs it. Baking in an address means every fresh deployment tries
        # to send from an SES identity it can never verify, so every race ends
        # in MessageRejected. Supply one per deployment (see app.py).
        notify_from_email: str = "",
        notify_from_name: str = "SLM Fine-tuning Platform", **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ===================================== Docker Hub pull-through cache
        # CodeBuild builds FROM hiyouga/llamafactory, which kept failing on Docker
        # Hub's unauthenticated pull-rate-limit (429). Fix: an ECR pull-through
        # cache for Docker Hub — builds pull `…/docker-hub/hiyouga/llamafactory`
        # from ECR, which fetches+caches upstream (authenticated) on a miss and
        # serves cached layers after. AWS REQUIRES the credential secret name to
        # start with `ecr-pullthroughcache/` and the value to be JSON
        # {"username","accessToken"} (a Docker Hub PAT). CDK seeds a placeholder;
        # the real PAT is set once via `aws secretsmanager put-secret-value`
        # (same operate-once pattern as the HF token).
        dockerhub_secret = secretsmanager.Secret(
            self,
            "DockerHubPullThroughCreds",
            secret_name="ecr-pullthroughcache/dockerhub",  # nosec B106 — Secrets Manager secret NAME (resource id), not a credential value

            description="Docker Hub PAT for the ECR pull-through cache (set real value via CLI).",
            removal_policy=RemovalPolicy.DESTROY,
        )
        ecr.CfnPullThroughCacheRule(
            self,
            "DockerHubPullThroughCacheRule",
            ecr_repository_prefix="docker-hub",
            upstream_registry="docker-hub",
            upstream_registry_url="registry-1.docker.io",
            credential_arn=dockerhub_secret.secret_arn,
        )

        # ============================================================== ECR
        repo = ecr.Repository(
            self,
            "Repository",
            repository_name=f"{prefix}-llamafactory",
            image_scan_on_push=True,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Expire untagged images after 14 days",
                    tag_status=ecr.TagStatus.UNTAGGED,
                    max_image_age=Duration.days(14),
                )
            ],
        )
        training_image_uri = (
            f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/"
            f"{repo.repository_name}:{IMAGE_TAG}"
        )

        # ====================================================== S3 buckets
        data_bucket = s3.Bucket(
            self,
            "DataBucket",
            bucket_name=f"{prefix}-data-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            # CORS is attached later (search add_cors_rule): the allowed origins are
            # this deployment's own URLs, which are not known until the CloudFront
            # distribution below has been created.
        )

        spa_bucket = s3.Bucket(
            self,
            "SpaBucket",
            bucket_name=f"{prefix}-spa-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # ============================================ SageMaker exec role
        sm_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name=f"{prefix}-sagemaker-exec",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Least-privilege execution role for SLM platform "
            "SageMaker training/eval jobs.",
        )
        data_bucket.grant_read_write(sm_role)
        repo.grant_pull(sm_role)
        sm_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchLogs",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"],
            )
        )
        sm_role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchMetrics",
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": "/aws/sagemaker/TrainingJobs"}},
            )
        )
        # Serverless model-customization (the 2nd engine, SFT/DPO). The managed
        # AWS policy covers serverless training + hub-content + model-package +
        # the JumpStart cache the recipe pulls from. Attached to the EXEC role
        # (which the job assumes) — the engine is gated in-app by the saved
        # enableSagemakerServerless flag (Settings toggle), so granting it changes
        # nothing while the engine is off. The managed policy is broader than this
        # app strictly needs; it is used because the recipe's exact action set is not
        # documented per action, so a hand-rolled equivalent has to be derived by
        # tracing AccessDenied failures. Narrow it if your environment requires
        # least privilege here.
        _mc_serverless = iam.ManagedPolicy.from_aws_managed_policy_name(
            "AmazonSageMakerModelCustomizationCoreAccess"
        )
        sm_role.add_managed_policy(_mc_serverless)

        # --- RLVR custom reward functions -----------------------------------
        # A reward function = a user-authored scoring Lambda the RLVR GRPO loop
        # invokes per rollout batch. It runs on a DEDICATED least-priv exec role
        # (basic logging only — it does pure string scoring, needs nothing else).
        reward_lambda_role = iam.Role(
            self,
            "RewardLambdaExecRole",
            role_name=f"{prefix}-rlvr-reward-lambda-exec",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Least-priv exec role for SLM RLVR custom reward Lambdas "
            "(scoring only: basic logging, nothing else).",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )
        # The SageMaker exec role (which the RLVR job assumes) invokes the reward
        # Lambdas during training. Scope to our reward-Lambda name prefix.
        reward_lambda_arn_glob = (
            f"arn:aws:lambda:{self.region}:{self.account}:function:slm-rlvr-reward-*"
        )
        sm_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeRlvrRewardLambdas",
                actions=["lambda:InvokeFunction"],
                resources=[reward_lambda_arn_glob],
            )
        )

        # ============================================ HF token (Secrets Manager)
        # Stores HuggingFace tokens that unlock gated models (Llama, Mistral,
        # Gemma). PER USER: the secret holds a JSON map {tenant_sub: token} so
        # each user brings their own token + their own license approval (see
        # secrets.py) — one secret keeps IAM simple while isolating by caller.
        # The API/worker Lambdas read+write it; the CURRENT user's token is
        # injected as HF_TOKEN into SageMaker jobs at launch (so the exec role
        # doesn't need secret access — the token travels as job env). Created with
        # a random placeholder (Secrets Manager rejects an empty string); the
        # app's get_hf_token() requires an 'hf_' prefix so it never counts as real.
        hf_secret = secretsmanager.Secret(
            self,
            "HfToken",
            secret_name=f"{prefix}/hf-token",
            description="HuggingFace token for pulling gated models in SageMaker jobs.",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # =================================================== API Lambda(s)
        # Container image built by CDK from backend/Dockerfile (the small app
        # image — NOT the GPU training image).
        common_env = {
            "SLM_STORAGE_BACKEND": "cloud",
            "SLM_S3_BUCKET": data_bucket.bucket_name,
            "SLM_STATE_PREFIX": STATE_PREFIX,
            "SLM_AWS_REGION": self.region,
            "SLM_AWS_ACCOUNT": self.account,
            "SLM_SAGEMAKER_ROLE_ARN": sm_role.role_arn,
            # Dedicated exec role for RLVR custom-reward scoring Lambdas (created at
            # runtime by the app). reward_naming reads this; falls back to a
            # conventional name if unset.
            "SLM_REWARD_LAMBDA_ROLE_ARN": reward_lambda_role.role_arn,
            "SLM_TRAINING_IMAGE_URI": training_image_uri,
            "SLM_HF_SECRET_NAME": hf_secret.secret_name,
            "SLM_SCRATCH_DIR": "/tmp/slm-state",
            # Per-user resource isolation: state is keyed by the Cognito `sub` the
            # API Gateway JWT authorizer forwards (requestContext.authorizer.jwt.
            # claims.sub). Existing pre-tenancy state was migrated under
            # users/<owner-sub>/, so the owner keeps their history; everyone else
            # starts with a private empty workspace.
            "SLM_MULTI_TENANT": "true",
            # Cost-guardrail limits (backend/app/limits.py). These are PER TENANT
            # (the state store partitions races under users/<sub>/), so they bound
            # EACH user, not the account: N users each sitting at their cap can still
            # exceed the account's training-instance quota, which is why the
            # platform-wide backstop below exists too.
            #   - 8 races / 16 models per user keeps one user's worst case bounded.
            #   - SLM_MAX_MODELS_PER_RACE MUST be >= the guided "Thorough" ceiling (16),
            #     or that ceiling silently clamps below what the UI offers (the planner
            #     stops early anyway when fewer arms add signal, so 16 is only the hard
            #     upper bound).
            #   - SLM_MAX_GLOBAL_CONCURRENT_RACES caps the CROSS-TENANT total so the
            #     sum across all users can't exceed the account's training-instance
            #     quota. Each race entry uses ~1 instance, so 40 leaves headroom for
            #     base-eval + retries against a 60-instance quota — check your own
            #     account's SageMaker training-instance quota and lower this if it is
            #     smaller. 0 disables the cap; it is set explicitly so the backstop is
            #     on by default.
            # An operator running a deliberate big sweep can raise these per env.
            "SLM_MAX_CONCURRENT_RACES": "8",
            "SLM_MAX_MODELS_PER_RACE": "16",
            "SLM_MAX_GLOBAL_CONCURRENT_RACES": "40",
            # 2nd training engine (SageMaker serverless customization): SFT/DPO/RLVR
            # with LoRA, no infra to manage. It defaults ON in-app (engines.base:
            # _ENGINE_DEFAULT_ENABLED). Its on/off state is the saved
            # enableSagemakerServerless config (Settings toggle) — the single source
            # of truth, no env var, no redeploy. The V3-SDK interpreter is baked into
            # the image at SLM_SERVERLESS_PYTHON (Dockerfile), independent of this.
            # Race-completion email (notify.py): the verified SES sender + friendly
            # display name. The address MUST be an SES-verified identity (sandbox: it
            # also gates who can RECEIVE until production access is granted). SLM_APP_URL
            # is added below once the CloudFront domain exists (deep-link in the email).
            "SLM_NOTIFY_FROM_EMAIL": notify_from_email,
            "SLM_NOTIFY_FROM_NAME": notify_from_name,
        }

        api_fn = lambda_.DockerImageFunction(
            self,
            "ApiFunction",
            function_name=f"{prefix}-api",
            code=lambda_.DockerImageCode.from_image_asset(str(BACKEND_DIR)),
            memory_size=2048,
            timeout=Duration.seconds(29),  # API GW HTTP API max integration timeout
            environment=common_env,
            description="SLM platform FastAPI backend (Mangum).",
        )

        reconcile_fn = lambda_.DockerImageFunction(
            self,
            "ReconcileFunction",
            function_name=f"{prefix}-reconcile",
            code=lambda_.DockerImageCode.from_image_asset(
                str(BACKEND_DIR),
                cmd=["app.lambda_handler.reconcile_handler"],
            ),
            memory_size=1024,
            timeout=Duration.minutes(5),
            environment=common_env,
            description="Advances non-terminal races on a schedule (headless).",
        )

        # Worker Lambda: long tasks invoked async by the API (e.g. the Sonnet
        # baseline, which makes one Bedrock call per eval row and can exceed API
        # Gateway's 29s limit). 15-min timeout covers large eval sets.
        worker_fn = lambda_.DockerImageFunction(
            self,
            "WorkerFunction",
            function_name=f"{prefix}-worker",
            code=lambda_.DockerImageCode.from_image_asset(
                str(BACKEND_DIR),
                cmd=["app.lambda_handler.worker_handler"],
            ),
            memory_size=1024,
            timeout=Duration.minutes(15),
            environment=common_env,
            description="Runs long backend tasks off the request path (baseline).",
        )
        # The API needs to know the worker's name + be allowed to invoke it.
        api_fn.add_environment("SLM_WORKER_FUNCTION", worker_fn.function_name)
        worker_fn.grant_invoke(api_fn)

        # All three Lambdas ARE the app's AWS identity. Grant what the app calls.
        for fn in (api_fn, reconcile_fn, worker_fn):
            data_bucket.grant_read_write(fn)
            repo.grant_pull(fn)
            # HF token: API writes it (Settings), all read it to inject into jobs.
            hf_secret.grant_read(fn)
            hf_secret.grant_write(fn)
            # SageMaker: create/describe/list training jobs + pass the exec role.
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="SageMakerJobs",
                    actions=[
                        "sagemaker:CreateTrainingJob",
                        "sagemaker:DescribeTrainingJob",
                        "sagemaker:ListTrainingJobs",
                        "sagemaker:AddTags",
                    ],
                    resources=["*"],
                )
            )
            # SES: send the "your run finished" email (worker/reconcile) + kick off
            # recipient verification at launch (API). SESv2 send/identity actions are
            # account-level (no resource ARN), so they go on "*"; sending is still
            # gated by SES's own verified-sender + sandbox rules.
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="SesNotifications",
                    actions=[
                        "ses:SendEmail",          # SESv2 SendEmail maps to this action
                        "ses:GetEmailIdentity",
                        "ses:CreateEmailIdentity",  # request recipient verification (sandbox)
                    ],
                    resources=["*"],
                )
            )
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="PassSageMakerRole",
                    actions=["iam:PassRole"],
                    resources=[sm_role.role_arn],
                    conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
                )
            )
            # Serverless engine (2nd engine): the Lambda (and the V3 launcher
            # subprocess it spawns, which runs under THIS Lambda's role via boto3)
            # resolves the Public-Hub model, ensures the per-tenant
            # ModelPackageGroup, and calls SFTTrainer/DPOTrainer.train(). That
            # train() call reads the recipe artifacts from the JumpStart cache
            # bucket and creates the training job — all with the Lambda's creds
            # (NOT the passed exec role, which only applies once the job is
            # running). So the Lambda needs the hub/model-package control-plane
            # actions AND read access to the JumpStart cache S3. Harmless while the
            # engine is off (the saved enableSagemakerServerless flag gates it).
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="ServerlessCustomization",
                    actions=[
                        "sagemaker:DescribeHubContent",
                        "sagemaker:ListHubContents",
                        "sagemaker:CreateModelPackageGroup",
                        "sagemaker:DescribeModelPackageGroup",
                        "sagemaker:ListModelPackages",
                        "sagemaker:DescribeModelPackage",
                        "sagemaker:CreateTrainingJob",   # launcher creates the serverless job
                        "sagemaker:DescribeTrainingJob",
                        "sagemaker:AddTags",
                    ],
                    resources=["*"],
                )
            )
            # The serverless trainer's .train() (run in the launcher under this
            # Lambda's role) GETs recipe templates/params from the regional
            # JumpStart cache bucket. Without this the launch fails with
            # 'AccessDenied ... GetObject' (seen on a real serverless run).
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="JumpStartCacheRead",
                    actions=["s3:GetObject", "s3:ListBucket"],
                    resources=[
                        f"arn:aws:s3:::jumpstart-cache-prod-{self.region}",
                        f"arn:aws:s3:::jumpstart-cache-prod-{self.region}/*",
                    ],
                )
            )
            # RLVR custom reward functions: the Lambda (+ its V3 launcher
            # subprocess) creates/updates the reward scoring Lambda, passes it the
            # dedicated reward exec role, and registers it as a SageMaker Evaluator.
            # Scoped to the slm-rlvr-reward-* name prefix. Harmless until RLVR with
            # a custom reward is used.
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="ManageRewardLambdas",
                    actions=[
                        "lambda:CreateFunction",
                        "lambda:UpdateFunctionCode",
                        "lambda:GetFunction",
                        "lambda:TagResource",
                    ],
                    resources=[reward_lambda_arn_glob],
                )
            )
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="PassRewardLambdaRole",
                    actions=["iam:PassRole"],
                    resources=[reward_lambda_role.role_arn],
                    conditions={"StringEquals": {"iam:PassedToService": "lambda.amazonaws.com"}},
                )
            )
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="RegisterRewardEvaluator",
                    # Evaluator.create registers the reward as JsonDoc hub-content,
                    # which creates/uses an AI Registry HUB under the hood — so it
                    # needs the hub control-plane actions (Create/Describe/List Hub),
                    # not just the *HubContent ones (the missing Create/DescribeHub
                    # is what failed the first real reward deploy).
                    actions=[
                        "sagemaker:CreateHub",
                        "sagemaker:DescribeHub",
                        "sagemaker:ListHubs",
                        "sagemaker:CreateHubContent",
                        "sagemaker:DescribeHubContent",
                        "sagemaker:ListHubContents",
                        "sagemaker:ImportHubContent",
                        "sagemaker:CreateHubContentReference",
                    ],
                    resources=["*"],
                )
            )
            # Preflight (Settings → Check environment) verifies the exec role
            # exists and the training image is present: iam:GetRole on the exec
            # role + ecr:DescribeImages (grant_pull doesn't include Describe).
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="PreflightGetRole",
                    actions=["iam:GetRole"],
                    resources=[sm_role.role_arn],
                )
            )
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="PreflightDescribeImages",
                    actions=["ecr:DescribeImages", "ecr:DescribeRepositories"],
                    resources=[repo.repository_arn],
                )
            )
            # Training curves: read the loss/lr/epoch series SageMaker scrapes
            # into CloudWatch (Races detail chart). GetMetricData has no
            # resource-level scoping, so it must be "*".
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="TrainingCurves",
                    actions=["cloudwatch:GetMetricData"],
                    resources=["*"],
                )
            )
            # Bedrock: Sonnet baseline (cross-region inference profile + the
            # underlying foundation models it routes to).
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="BedrockInvoke",
                    actions=["bedrock:InvokeModel"],
                    resources=[
                        f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                        "arn:aws:bedrock:*::foundation-model/*",
                    ],
                )
            )
            # Bedrock AgentCore: invoke the dataset-investigation Strands agent
            # runtime (deployed separately via the agentcore CLI). The runtime id
            # has a random suffix, so the backend resolves the ARN by the runtime's
            # stable NAME via ListAgentRuntimes (control plane) then invokes it
            # (data plane). Scope to all runtimes in this account (the id varies).
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="AgentCoreInvoke",
                    actions=["bedrock-agentcore:InvokeAgentRuntime"],
                    resources=[
                        f"arn:aws:bedrock-agentcore:*:{self.account}:runtime/*",
                    ],
                )
            )
            # ListAgentRuntimes (control plane) is a collection op with no
            # resource-level scoping → must be "*". Used to resolve the runtime
            # ARN by its stable name (the id suffix is random per deploy).
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="AgentCoreList",
                    actions=["bedrock-agentcore:ListAgentRuntimes"],
                    resources=["*"],
                )
            )

        # ================================================ Cognito user pool
        user_pool = cognito.UserPool(
            self,
            "UserPool",
            user_pool_name=f"{prefix}-users",
            self_sign_up_enabled=False,  # no public sign-up: an admin creates each user
            sign_in_aliases=cognito.SignInAliases(email=True, username=True),
            # Deliberately permissive: 8+ chars, lower+digit+symbol, no forced
            # uppercase, no MFA. Every account is created by an administrator with
            # `admin-create-user`, and Cognito's hosted UI forces a permanent
            # password on first sign-in. Tighten this (and add MFA) before exposing
            # a deployment beyond a small trusted group.
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_lowercase=True,
                require_digits=True,
                require_symbols=True,
                require_uppercase=False,
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )
        # Hosted-UI domain prefix. Cognito requires it to be globally unique, so it
        # needs a per-deployment discriminator — but it MUST NOT be the account id.
        # The prefix ends up in a DNS-resolvable public hostname AND in the
        # unauthenticated /config.json the SPA fetches before sign-in (see
        # cognito_domain_fqdn below), so an account id here publishes the
        # deployer's 12-digit account number to anonymous visitors. The stack's
        # own unique id is a CloudFormation-generated GUID: equally unique, not
        # sensitive, and resolved at deploy time.
        stack_unique_id = Fn.select(2, Fn.split("/", self.stack_id))
        cognito_domain_prefix = f"{prefix}-{stack_unique_id}"
        user_pool_domain = user_pool.add_domain(
            "UserPoolDomain",
            cognito_domain=cognito.CognitoDomainOptions(
                domain_prefix=cognito_domain_prefix,
            ),
        )

        # ===================================================== HTTP API GW
        http_api = apigw.HttpApi(
            self,
            "HttpApi",
            api_name=f"{prefix}-api",
            create_default_stage=True,
        )
        api_domain = f"{http_api.api_id}.execute-api.{self.region}.amazonaws.com"

        # ====================================================== CloudFront
        # WAF (CLOUDFRONT scope) MUST be us-east-1 — this whole stack deploys
        # there (see app.py), so a same-stack WebACL is valid.
        web_acl = wafv2.CfnWebACL(
            self,
            "WebAcl",
            default_action=wafv2.CfnWebACL.DefaultActionProperty(allow={}),
            scope="CLOUDFRONT",
            visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=f"{prefix}-webacl",
                sampled_requests_enabled=True,
            ),
            rules=[
                wafv2.CfnWebACL.RuleProperty(
                    name="AWSManagedCommon",
                    priority=1,
                    override_action=wafv2.CfnWebACL.OverrideActionProperty(none={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=wafv2.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesCommonRuleSet",
                            # The managed SizeRestrictions_BODY rule blocks any
                            # request body > 8 KB, which kills dataset/eval JSONL
                            # uploads. Set it to Count (observe, don't block) so
                            # uploads work; the FastAPI app enforces its own
                            # 100 MB cap instead (MAX_UPLOAD_BYTES in
                            # backend/app/main.py). The rest of the common set
                            # stays active.
                            rule_action_overrides=[
                                wafv2.CfnWebACL.RuleActionOverrideProperty(
                                    name="SizeRestrictions_BODY",
                                    action_to_use=wafv2.CfnWebACL.RuleActionProperty(count={}),
                                )
                            ],
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="AWSManagedCommon",
                        sampled_requests_enabled=True,
                    ),
                ),
                wafv2.CfnWebACL.RuleProperty(
                    name="RateLimit",
                    priority=2,
                    action=wafv2.CfnWebACL.RuleActionProperty(block={}),
                    statement=wafv2.CfnWebACL.StatementProperty(
                        rate_based_statement=wafv2.CfnWebACL.RateBasedStatementProperty(
                            limit=2000, aggregate_key_type="IP"
                        )
                    ),
                    visibility_config=wafv2.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="RateLimit",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )

        # API origin: forward everything except Host (HTTP APIs reject a foreign
        # Host header). Use the all-viewer-except-host managed policy.
        api_origin = origins.HttpOrigin(
            api_domain,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
        )

        # >>> CUSTOM DOMAIN. Context-gated so the default build is unaffected.
        #     Deploy with:
        #       -c customDomain=app.example.com \
        #       -c customDomainZoneId=Z0123456789ABCDEFGHIJ
        #     The ACM cert is DNS-validated against that hosted zone (must be
        #     us-east-1 for CloudFront — this whole stack is, see app.py).
        custom_domain = self.node.try_get_context("customDomain")
        custom_domain_zone_id = self.node.try_get_context("customDomainZoneId")
        domain_names = None
        certificate = None
        hosted_zone = None
        if custom_domain:
            if not custom_domain_zone_id:
                raise ValueError("customDomain requires -c customDomainZoneId")
            hosted_zone = route53.HostedZone.from_hosted_zone_attributes(
                self,
                "CustomDomainZone",
                hosted_zone_id=custom_domain_zone_id,
                zone_name=custom_domain,
            )
            certificate = acm.Certificate(
                self,
                "CustomDomainCert",
                domain_name=custom_domain,
                validation=acm.CertificateValidation.from_dns(hosted_zone),
            )
            domain_names = [custom_domain]
        # <<< CUSTOM DOMAIN

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            comment=f"{prefix} app",
            default_root_object="index.html",
            domain_names=domain_names,
            certificate=certificate,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(spa_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            additional_behaviors={
                "/api/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                ),
            },
            # NO custom error responses. The SPA uses HASH routing (#finetune,
            # #races, …) + useState — there are no deep server paths to fall
            # back to, so the only document path is "/" → index.html (handled by
            # default_root_object). A 404→index.html remap would be unnecessary
            # AND harmful: CloudFront error responses are distribution-wide, so
            # they'd mask legitimate API 404s/4xx (e.g. "unknown race") as HTML
            # and break the SPA's JSON parsing. Let all API responses pass
            # through verbatim.
            web_acl_id=web_acl.attr_arn,
        )

        cf_url = f"https://{distribution.distribution_domain_name}"

        # >>> CUSTOM DOMAIN — A/AAAA alias records → this distribution.
        if custom_domain:
            route53.ARecord(
                self,
                "CustomDomainAlias",
                zone=hosted_zone,
                target=route53.RecordTarget.from_alias(
                    r53_targets.CloudFrontTarget(distribution)
                ),
            )
            route53.AaaaRecord(
                self,
                "CustomDomainAliasV6",
                zone=hosted_zone,
                target=route53.RecordTarget.from_alias(
                    r53_targets.CloudFrontTarget(distribution)
                ),
            )
        # <<< CUSTOM DOMAIN

        # The race-completion email deep-links back to the run-detail page; the app
        # URL only exists now (CloudFront domain), so inject it post-hoc on all three
        # Lambdas (same pattern as SLM_WORKER_FUNCTION).
        for fn in (api_fn, reconcile_fn, worker_fn):
            fn.add_environment("SLM_APP_URL", cf_url)

        # App client for the Hosted UI. Implicit grant (response_type=token) so
        # the Hosted UI returns the id_token in the URL hash and the SPA needs no
        # backend token exchange.
        #
        # Callback/logout URLs must exactly match the app origin: always the
        # CloudFront domain, plus the custom domain when configured (both kept so
        # bookmarks keep working).
        app_urls = [f"{cf_url}/"]
        if custom_domain:
            app_urls.append(f"https://{custom_domain}/")

        # Data-bucket CORS, scoped to THIS deployment's own origins. It was `["*"]`,
        # which let any web page a signed-in user visited drive a leaked presigned URL
        # from their browser. An Origin header never carries a path or a trailing
        # slash, so strip the slash app_urls needs for Cognito callbacks. Headers are
        # narrowed to content-type — the SPA's presigned PUT (frontend/src/api.ts:65)
        # sets no other request header.
        data_bucket.add_cors_rule(
            allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT],
            allowed_origins=[u.rstrip("/") for u in app_urls],
            allowed_headers=["content-type"],
        )

        user_pool_client = user_pool.add_client(
            "AppClient",
            user_pool_client_name=f"{prefix}-app",
            generate_secret=False,  # public SPA client
            # USER_PASSWORD_AUTH is deliberately NOT enabled: it accepts a raw
            # username+password over the API and is the flow credential-stuffing
            # targets. The hosted UI signs in through SRP, so nothing needs it.
            auth_flows=cognito.AuthFlow(user_srp=True),
            o_auth=cognito.OAuthSettings(
                # Authorization CODE grant with PKCE, not the implicit grant. Implicit
                # returns the id_token in the URL fragment, where it lands in history
                # and referrers and — because the SPA had no way to bind the response
                # to the request it made — any token pasted into the fragment was
                # accepted, so one crafted link could drop a visitor into someone
                # else's session. The code grant hands back a single-use code that is
                # worthless without the PKCE verifier held in the browser that started
                # the flow (see frontend/src/auth.ts). Cognito supports PKCE for
                # public clients, so this needs no client secret.
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE],
                callback_urls=app_urls,
                logout_urls=app_urls,
            ),
        )

        # API routes: gate everything behind a Cognito JWT, except /api/health
        # (left public so CloudFront/uptime checks don't need a token).
        authorizer = apigw_auth.HttpUserPoolAuthorizer(
            "ApiAuthorizer", user_pool, user_pool_clients=[user_pool_client]
        )
        api_integration = apigw_int.HttpLambdaIntegration("ApiIntegration", api_fn)
        http_api.add_routes(
            path="/api/health",
            methods=[apigw.HttpMethod.GET],
            integration=api_integration,
        )
        http_api.add_routes(
            path="/api/{proxy+}",
            methods=[apigw.HttpMethod.ANY],
            integration=api_integration,
            authorizer=authorizer,
        )

        # =============================================== Frontend build+deploy
        # CDK builds the SPA (npm ci && npm run build) in a Node bundling
        # container, then uploads dist/ to the SPA bucket and invalidates the
        # distribution — all part of `cdk deploy`. The SPA calls relative /api
        # paths, so no API URL needs baking in.
        #
        # The runtime config.json (Cognito params the SPA fetches on load) is a
        # SECOND source in the SAME deployment — NOT a separate BucketDeployment,
        # which would prune the other's files. No rebuild needed when these
        # values change; the SPA reads them at runtime.
        # Built from the same prefix the domain was created with (above), NOT from
        # the account id — this value is served to anonymous visitors in
        # /config.json, so it must not carry the account number.
        cognito_domain_fqdn = (
            f"{cognito_domain_prefix}.auth.{self.region}.amazoncognito.com"
        )
        s3deploy.BucketDeployment(
            self,
            "DeploySpa",
            sources=[
                s3deploy.Source.asset(
                    str(FRONTEND_DIR),
                    bundling={
                        "image": lambda_.Runtime.NODEJS_20_X.bundling_image,
                        # HOME/npm cache point at a writable path: CDK runs the
                        # bundler as the host uid, which has no home in the image.
                        "command": [
                            "bash",
                            "-c",
                            "export HOME=/tmp && npm_config_cache=/tmp/.npm "
                            "npm ci && npm run build && cp -r dist/* /asset-output/",
                        ],
                    },
                ),
                s3deploy.Source.json_data(
                    "config.json",
                    {
                        "cognitoDomain": cognito_domain_fqdn,
                        "cognitoClientId": user_pool_client.user_pool_client_id,
                        "region": self.region,
                    },
                ),
            ],
            destination_bucket=spa_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # =============================================== EventBridge schedule
        events.Rule(
            self,
            "ReconcileSchedule",
            rule_name=f"{prefix}-reconcile",
            schedule=events.Schedule.rate(Duration.minutes(1)),
            targets=[targets.LambdaFunction(reconcile_fn)],
            description="Advance non-terminal races every minute (headless).",
        )

        # ============================================ CodeBuild: training images
        # Builds the ~16.7GB GPU image(s) from container/Dockerfile and pushes to
        # ECR. Too large for CDK Docker bundling; runs on a 128GB-disk fleet.
        # Triggered automatically on deploy by a per-tier AwsCustomResource keyed
        # to the container source hash (see "Auto-build every tier's image" below).
        # StartBuild is async, so the ~10-15 min build does NOT gate the stack.
        # To rebuild without a deploy: aws codebuild start-build
        # --project-name <project output>.
        #
        # ONE project PER image tier (the multi-image design): the same Dockerfile
        # source builds each tier off a different LLaMA-Factory base (--build-arg
        # LF_BASE_TAG), tagged with the tier's ECR tag. Building tiers side by side
        # means a new stack (0.9.5) lands ALONGSIDE the proven one (0.9.4), never
        # replacing it — old models keep their image, new models get the new one.
        container_asset = s3_assets.Asset(
            self, "TrainingImageSource", path=str(CONTAINER_DIR)
        )
        build_projects: dict[str, codebuild.Project] = {}
        for image_tag, spec in IMAGE_TIERS.items():
            # Construct ids must be alnum; turn "0.9.5" → "095".
            cid = image_tag.replace(".", "")
            project = codebuild.Project(
                self,
                f"TrainingImageBuild{cid}",
                project_name=f"{prefix}-training-image-build-{cid}",
                environment=codebuild.BuildEnvironment(
                    build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                    compute_type=codebuild.ComputeType.LARGE,  # 128GB disk
                    privileged=True,  # docker build
                ),
                environment_variables={
                    "ACCOUNT": codebuild.BuildEnvironmentVariable(value=self.account),
                    "REGION": codebuild.BuildEnvironmentVariable(value=self.region),
                    "REPO": codebuild.BuildEnvironmentVariable(value=repo.repository_name),
                    "BUCKET": codebuild.BuildEnvironmentVariable(value=data_bucket.bucket_name),
                    "TAG": codebuild.BuildEnvironmentVariable(value=image_tag),
                    "LF_BASE_TAG": codebuild.BuildEnvironmentVariable(value=spec["lf_base"]),
                    "VLLM_VERSION": codebuild.BuildEnvironmentVariable(value=spec["vllm"]),
                    # Pull the LLaMA-Factory base through this account's ECR
                    # pull-through cache for Docker Hub (dodges the Docker Hub 429).
                    "LF_BASE_REGISTRY": codebuild.BuildEnvironmentVariable(
                        value=f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/docker-hub/hiyouga"
                    ),
                },
                # Source = the container/ dir packaged as an S3 asset by CDK at
                # synth, so the build is self-contained (no git remote needed).
                source=codebuild.Source.s3(
                    bucket=container_asset.bucket,
                    path=container_asset.s3_object_key,
                ),
                build_spec=codebuild.BuildSpec.from_object(_training_image_buildspec()),
            )
            repo.grant_pull_push(project)
            container_asset.grant_read(project)
            data_bucket.grant_write(project)  # upload the image capability manifest
            build_projects[image_tag] = project
        # Back-compat alias for the stable project (kept for the original output).
        build_project = build_projects[IMAGE_TAG]

        # ADHOC build project: builds ANY hiyouga/llamafactory:<tag> via env-var
        # overrides at start_build time (LF_BASE_TAG / TAG / VLLM_VERSION). This is
        # what lets the Images page build a NEWLY-RELEASED LF version one-click —
        # without a per-release CDK project + redeploy. The env values here are
        # defaults; the backend overrides them per call. Same shared buildspec, so
        # it also writes the capability manifest for model-discovery.
        adhoc_project = codebuild.Project(
            self,
            "TrainingImageBuildAdhoc",
            project_name=f"{prefix}-training-image-build-adhoc",
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=codebuild.ComputeType.LARGE,
                privileged=True,
            ),
            environment_variables={
                "ACCOUNT": codebuild.BuildEnvironmentVariable(value=self.account),
                "REGION": codebuild.BuildEnvironmentVariable(value=self.region),
                "REPO": codebuild.BuildEnvironmentVariable(value=repo.repository_name),
                "BUCKET": codebuild.BuildEnvironmentVariable(value=data_bucket.bucket_name),
                "TAG": codebuild.BuildEnvironmentVariable(value="adhoc"),
                "LF_BASE_TAG": codebuild.BuildEnvironmentVariable(value="0.9.5"),
                "VLLM_VERSION": codebuild.BuildEnvironmentVariable(value="0.8.5.post1"),
                "LF_BASE_REGISTRY": codebuild.BuildEnvironmentVariable(
                    value=f"{self.account}.dkr.ecr.{self.region}.amazonaws.com/docker-hub/hiyouga"
                ),
            },
            source=codebuild.Source.s3(
                bucket=container_asset.bucket,
                path=container_asset.s3_object_key,
            ),
            build_spec=codebuild.BuildSpec.from_object(_training_image_buildspec()),
        )
        repo.grant_pull_push(adhoc_project)
        container_asset.grant_read(adhoc_project)
        data_bucket.grant_write(adhoc_project)
        build_projects["__adhoc__"] = adhoc_project

        # Every build project pulls the LLaMA-Factory base through the Docker Hub
        # pull-through cache. On a cache MISS (e.g. a new LF tag), ECR must create
        # the cached repo + import the upstream image — these actions aren't
        # covered by repo.grant_pull_push (that's the OUR-image repo only). Grant
        # the pull-through-cache import perms (scoped to the docker-hub/* cache
        # repos) + GetAuthorizationToken to all build projects.
        for _proj in build_projects.values():
            _proj.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchImportUpstreamImage",
                        "ecr:CreateRepository",
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                    resources=["*"],  # GetAuthorizationToken needs *, others scoped by ECR
                )
            )

        # Auto-build every tier's image ON DEPLOY so a single `cdk deploy` ends up
        # with BOTH images (0.9.4 + 0.9.5) in the deploying account's ECR — no
        # manual `start-build`. Each tier gets an AwsCustomResource that calls
        # codebuild:StartBuild. The trigger is keyed to the container source hash
        # (container_asset.asset_hash), so it fires on the first deploy and again
        # whenever the Dockerfile/entrypoint changes — but NOT on unrelated stack
        # updates (idempotent). StartBuild returns immediately (async); the build
        # runs ~10-15 min in the background and does NOT gate the stack, so a slow
        # or failing image build can't roll back the app. (Re)build manually any
        # time from the Images page or `aws codebuild start-build`.
        for image_tag, project in build_projects.items():
            if image_tag == "__adhoc__":
                continue  # adhoc builds are on-demand only — never auto-fire on deploy
            cid = image_tag.replace(".", "")
            trigger = cr.AwsCustomResource(
                self,
                f"TrainingImageBuildTrigger{cid}",
                # output_paths trims the SDK response to JUST the build id —
                # StartBuild's full response is large and otherwise overflows the
                # CFN custom-resource response limit ("Response object is too
                # long"), which previously failed the update + wedged a rollback.
                on_create=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="startBuild",
                    parameters={"projectName": project.project_name},
                    output_paths=["build.id"],
                    # New physical id when the source changes → CFN re-runs on_update.
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"build-{cid}-{container_asset.asset_hash}"
                    ),
                ),
                on_update=cr.AwsSdkCall(
                    service="CodeBuild",
                    action="startBuild",
                    parameters={"projectName": project.project_name},
                    output_paths=["build.id"],
                    physical_resource_id=cr.PhysicalResourceId.of(
                        f"build-{cid}-{container_asset.asset_hash}"
                    ),
                ),
                policy=cr.AwsCustomResourcePolicy.from_statements(
                    [
                        iam.PolicyStatement(
                            actions=["codebuild:StartBuild"],
                            resources=[project.project_arn],
                        )
                    ]
                ),
                install_latest_aws_sdk=False,
            )
            # The build can only start once the project (+ its source asset) exist.
            trigger.node.add_dependency(project)

        # The API/worker Lambdas drive image management from the Images page +
        # self-healing flow: start a tier rebuild (start_build) and read each
        # tier's latest build status (list_builds_for_project + batch_get_builds).
        # Scope start/read to OUR per-tier projects; the list/batch-get reads are
        # account-level APIs with no resource ARN, so they go on "*".
        for fn in (api_fn, worker_fn, reconcile_fn):
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="CodeBuildStart",
                    actions=["codebuild:StartBuild", "codebuild:BatchGetBuilds"],
                    resources=[p.project_arn for p in build_projects.values()],
                )
            )
            fn.add_to_role_policy(
                iam.PolicyStatement(
                    sid="CodeBuildListReads",
                    actions=["codebuild:ListBuildsForProject"],
                    resources=[p.project_arn for p in build_projects.values()],
                )
            )

        # ================================================= Cost alerting
        # Backstop for a multi-user deployment: the in-app per-tenant + global race caps
        # bound how many GPU jobs CAN start, but a runaway (many users, long jobs,
        # big instances) still spends real money. These give an operator visibility
        # the moment spend climbs — independent of the app, so they fire even if the
        # app's own guardrails are misconfigured.
        #
        #   1. SNS topic  — the single notification channel. An email subscription is
        #      added only when an alertEmail was provided (else set one later in the
        #      console / redeploy with -c alertEmail=…). Email confirmation is manual
        #      (AWS sends a confirm link) — expected, one-time.
        #   2. AWS Budget — a MONTHLY cost budget (USD) with notifications at 80% and
        #      100% of ACTUAL spend plus 100% of FORECASTED spend, all → the SNS topic.
        #      Budgets is a global service; thresholds are evaluated by AWS, not us.
        #   3. CloudWatch billing alarm — on AWS/Billing EstimatedCharges (USD). That
        #      metric is published ONLY in us-east-1, which is exactly where this stack
        #      lives, so the alarm is in-region. Fires when estimated month-to-date
        #      charges cross the monthly budget → the SNS topic.
        cost_topic = sns.Topic(
            self, "CostAlertTopic",
            topic_name=f"{prefix}-cost-alerts",
            display_name="SLM platform cost alerts",
        )
        if alert_email.strip():
            cost_topic.add_subscription(sns_subs.EmailSubscription(alert_email.strip()))

        # Budgets needs to publish to the topic — grant SNS publish to the budgets
        # service principal, scoped to this topic.
        cost_topic.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowBudgetsPublish",
                actions=["sns:Publish"],
                principals=[iam.ServicePrincipal("budgets.amazonaws.com")],
                resources=[cost_topic.topic_arn],
            )
        )

        _budget_subscribers = [
            budgets.CfnBudget.SubscriberProperty(
                subscription_type="SNS", address=cost_topic.topic_arn
            )
        ]
        budgets.CfnBudget(
            self, "MonthlyCostBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name=f"{prefix}-monthly-budget",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=monthly_budget_usd, unit="USD"
                ),
            ),
            notifications_with_subscribers=[
                # 80% of ACTUAL month-to-date spend — early warning.
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="ACTUAL", comparison_operator="GREATER_THAN",
                        threshold=80, threshold_type="PERCENTAGE",
                    ),
                    subscribers=_budget_subscribers,
                ),
                # 100% of ACTUAL — budget reached.
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="ACTUAL", comparison_operator="GREATER_THAN",
                        threshold=100, threshold_type="PERCENTAGE",
                    ),
                    subscribers=_budget_subscribers,
                ),
                # 100% of FORECASTED — on track to overshoot by month end.
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="FORECASTED", comparison_operator="GREATER_THAN",
                        threshold=100, threshold_type="PERCENTAGE",
                    ),
                    subscribers=_budget_subscribers,
                ),
            ],
        )

        # CloudWatch billing alarm: estimated month-to-date charges over the monthly
        # budget. Period 6h (the billing metric updates a few times/day); 1 datapoint
        # is enough since it's a slow cumulative gauge.
        billing_metric = cloudwatch.Metric(
            namespace="AWS/Billing",
            metric_name="EstimatedCharges",
            dimensions_map={"Currency": "USD"},
            statistic="Maximum",
            period=Duration.hours(6),
        )
        billing_alarm = cloudwatch.Alarm(
            self, "EstimatedChargesAlarm",
            alarm_name=f"{prefix}-estimated-charges",
            alarm_description=(
                f"Estimated AWS charges exceeded the ${monthly_budget_usd:.0f} monthly "
                "budget for the SLM platform account."
            ),
            metric=billing_metric,
            threshold=monthly_budget_usd,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        billing_alarm.add_alarm_action(cw_actions.SnsAction(cost_topic))

        # ===================================================== Outputs
        CfnOutput(self, "CloudFrontUrl", value=f"https://{distribution.distribution_domain_name}")
        CfnOutput(self, "OutApiEndpoint", value=http_api.api_endpoint)
        CfnOutput(self, "OutCostAlertTopic", value=cost_topic.topic_arn)
        CfnOutput(self, "OutDataBucket", value=data_bucket.bucket_name)
        CfnOutput(self, "OutTrainingImageUri", value=training_image_uri)
        CfnOutput(self, "OutTrainingImageBuildProject", value=build_project.project_name)
        for image_tag, project in build_projects.items():
            cid = image_tag.replace(".", "")
            CfnOutput(self, f"OutTrainingImageBuildProject{cid}", value=project.project_name)
        CfnOutput(self, "OutUserPoolId", value=user_pool.user_pool_id)
        CfnOutput(self, "OutUserPoolClientId", value=user_pool_client.user_pool_client_id)
        CfnOutput(self, "OutCognitoDomain", value=user_pool_domain.domain_name)
        CfnOutput(self, "OutRegion", value=self.region)
        CfnOutput(self, "OutAccount", value=self.account)
        # Consumed by infra/apply_outputs.py, which merges the deployed resource
        # names into the backend's data/config.json. Without this the roleArn
        # Settings field has no output to read.
        CfnOutput(self, "OutSageMakerRoleArn", value=sm_role.role_arn)
