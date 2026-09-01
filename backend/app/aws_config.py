# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""AWS / SageMaker configuration for the platform.

Values resolve from the persisted Settings document, then environment variables,
then a built-in default — see _resolve(). No account id, bucket name or role ARN
is hardcoded here: the account is read from the ambient AWS credentials when it
is not configured explicitly (see _resolve_account), and the bucket/role/image
defaults are derived from it. The backend reads these once at startup; the
orchestrator uses them to launch jobs.

Cost: training jobs are billable. The launch path is gated behind an explicit
user action in the UI, never automatic.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .store import get_store

_log = logging.getLogger(__name__)

# Persisted runtime overrides (set via the Settings page). Layered OVER env vars,
# which are layered over the built-in defaults. Lets an operator point the app at
# their own account/bucket/role without editing code or env. Lives as the
# root-level config.json document in the state store (local disk or, hosted, S3).
_CONFIG_FILE = "config.json"


# Short-lived read cache for the config doc. image_tiers()/spot_discount_factor()/…
# call _saved(), and image_tiers() runs once PER MODEL inside model_status_map — so
# /api/models fanned out 30+ identical S3 GETs of config.json (a big chunk of the
# ~13s page load). Cache read-only access; read-modify-write callers pass
# use_cache=False so they mutate a FRESH doc. Writes invalidate. 5s TTL is a
# freshness backstop; monotonic() is a cache clock, not a persisted timestamp.
_CONFIG_CACHE_TTL_S = 5.0
_config_cache: dict | None = None
_config_cache_at: float = 0.0


def _saved(use_cache: bool = True) -> dict:
    global _config_cache, _config_cache_at
    import time

    if use_cache and _config_cache is not None and (time.monotonic() - _config_cache_at) < _CONFIG_CACHE_TTL_S:
        return _config_cache
    doc = get_store().read_root_json(_CONFIG_FILE)
    if use_cache:
        _config_cache = doc
        _config_cache_at = time.monotonic()
    return doc


def _invalidate_config_cache() -> None:
    global _config_cache
    _config_cache = None


def save_config(values: dict) -> None:
    """Persist non-empty config overrides to the config doc (merges)."""
    current = _saved(use_cache=False)  # read-modify-write → fresh, never the cache
    for k, v in values.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        current[k] = v.strip() if isinstance(v, str) else v
    get_store().write_root_json(_CONFIG_FILE, current)
    _invalidate_config_cache()


def _resolve(key: str, env: str, default: str) -> str:
    """saved config.json  >  environment variable  >  built-in default."""
    saved = _saved().get(key)
    if saved:
        return saved
    return os.environ.get(env, default)


def reset_cutoff() -> str | None:
    """ISO-8601 timestamp; SageMaker jobs created at/before this are hidden from
    all listings (a soft 'start blank' — SageMaker job records can't be deleted).
    Set via the reset action; None means show everything."""
    return _saved().get("resetCutoff")


# Image TIERS (the multi-image design). One Docker image cannot run EVERY model:
# new models need a newer transformers/LLaMA-Factory stack than older ones, and a
# global bump risks breaking the models that already work. So we keep a SMALL,
# fixed set of image tiers (not one-image-per-model — that's image sprawl) and a
# model declares which tier it runs on (ModelSpec.image_tag). The tier name maps
# to a concrete ECR image tag; the resolved URI is what the SageMaker job uses.
#
#   stable = 0.9.4 — the proven image the ~30 existing models run on.
#   latest = 0.9.5 — transformers v5 / LLaMA-Factory v0.9.5 for newer models
#                    (Phi-4-mini, Qwen3.5/3.6, Gemma4) that 0.9.4 can't load.
#
# Keep this in sync with the CDK CodeBuild projects that build each tag.
DEFAULT_IMAGE_TIER = "stable"
IMAGE_TIER_TAGS: dict[str, str] = {
    "stable": "0.9.4",
    "latest": "0.9.5",
}

# Name prefix the CDK stack gives every resource it creates (its `prefix`
# context value). The default bucket/role/repo names below are derived from it,
# exactly as the stack derives them — "<prefix>-data-<account>-<region>",
# "<prefix>-sagemaker-exec", "<prefix>-llamafactory". A stack deployed with a
# different prefix passes the resolved names to the backend as env vars instead
# (see the Lambda environment in infra/slm_platform_infra/stack.py).
RESOURCE_PREFIX = "slm-platform"

# Region the stack deploys into unless overridden. The stack deploys ENTIRELY in
# one region, so this is also the region every derived name is built against.
DEFAULT_REGION = "us-east-1"

# S3 prefix (under the data bucket) where each built image writes its capability
# manifest (supported architectures + templates). MUST match the CDK buildspec's
# IMAGE_META_PREFIX. The new-model discovery reads these to diff images cheaply.
IMAGE_META_PREFIX_DEFAULT = "slm-platform/image-meta"


def image_tiers() -> dict[str, str]:
    """Tier → ECR image tag. saved config.json ("imageTierTags") can extend/override
    the built-in map (so an operator can register a new tier without code), but the
    built-ins always remain as a floor."""
    tiers = dict(IMAGE_TIER_TAGS)
    saved = _saved().get("imageTierTags")
    if isinstance(saved, dict):
        tiers.update({str(k): str(v) for k, v in saved.items()})
    return tiers


def register_image_tier(tier: str, tag: str) -> dict[str, str]:
    """Persist a new (or updated) tier → tag mapping into config.json at RUNTIME.

    This is how a newly-built LLaMA-Factory release becomes a usable image tier
    WITHOUT a cdk deploy: the adhoc CodeBuild project builds + pushes the image,
    then this records the tier so image_tiers()/the catalog/orchestrate can use it
    immediately. Built-in tiers (stable/latest) can't be clobbered — they're the
    floor in image_tiers() — but a new name (e.g. "v096") is added to the saved map."""
    saved = _saved(use_cache=False)  # read-modify-write → fresh, never the cache
    tier_map = dict(saved.get("imageTierTags") or {})
    tier_map[str(tier)] = str(tag)
    saved["imageTierTags"] = tier_map
    get_store().write_root_json(_CONFIG_FILE, saved)
    _invalidate_config_cache()
    return image_tiers()


@dataclass(frozen=True)
class AwsConfig:
    region: str
    account_id: str
    role_arn: str
    bucket: str
    image_uri: str
    profile: str | None

    @property
    def s3_prefix(self) -> str:
        return f"s3://{self.bucket}/slm-platform"

    @property
    def image_repo(self) -> str:
        """The ECR repository URI without a tag (…/slm-platform-llamafactory)."""
        return self.image_uri.rsplit(":", 1)[0]

    def image_uri_for_tier(self, tier: str | None) -> str:
        """Resolve a model's image tier → full ECR image URI.

        Unknown/empty tier falls back to the default (stable) so a job is never
        launched against a non-existent tag. `stable` resolves to the same URI as
        the global `image_uri`, keeping every existing model byte-identical."""
        tiers = image_tiers()
        tag = tiers.get(tier or DEFAULT_IMAGE_TIER) or tiers[DEFAULT_IMAGE_TIER]
        return f"{self.image_repo}:{tag}"


# On-demand $/hr for the SageMaker TRAINING instances we use. g5 are list-price
# estimates; g6e/g7e are the REAL us-east-1 Training rates from the AWS Pricing API
# (verified 2026-06-19) — these back the cost estimate for the full/freeze (g6e)
# and future g7e paths. Without the g6e entries, instance_hourly() returned None
# for the g6e instances _instance_for already assigned (a pre-existing gap).
# Used to turn measured job duration into a measured train cost and a DERIVED
# projected self-host cost. Update deliberately; clearly a list-price estimate.
INSTANCE_HOURLY_USD: dict[str, float] = {
    "ml.g5.2xlarge": 1.515,
    "ml.g5.4xlarge": 2.03,
    "ml.g5.8xlarge": 2.94,
    "ml.g5.12xlarge": 7.09,
    # g6e (L40S 48 GB) — full/freeze training target. us-east-1 Training, Pricing API.
    "ml.g6e.xlarge": 2.61,
    "ml.g6e.2xlarge": 2.80,
    "ml.g6e.12xlarge": 13.12,
    # g7e (RTX PRO 6000 96 GB) — deferred (needs a Blackwell-ready image), priced
    # for when it lands. us-east-1 Training, Pricing API.
    "ml.g7e.2xlarge": 4.2039,
    "ml.g7e.4xlarge": 4.9977,
    "ml.g7e.12xlarge": 10.3576,
}


def instance_hourly(instance_type: str) -> float | None:
    return INSTANCE_HOURLY_USD.get(instance_type)


# Managed spot typically costs ~30% of on-demand for these GPU instances (a
# 60–70% discount). BillableTimeInSeconds on a spot job is still billed at the
# spot rate, so multiplying it by the ON-DEMAND rate overstates cost ~3×. We
# apply this factor to spot jobs and label the figure an estimate. Configurable
# via config.json ("spotDiscountFactor") or env, since the real discount drifts.
DEFAULT_SPOT_DISCOUNT_FACTOR = 0.35


def spot_discount_factor() -> float:
    """Fraction of the on-demand rate that managed spot is billed at (estimate).
    saved config.json > env SLM_SPOT_DISCOUNT_FACTOR > built-in default."""
    saved = _saved().get("spotDiscountFactor")
    if saved is not None:
        try:
            return float(saved)
        except (TypeError, ValueError):
            pass
    try:
        return float(os.environ.get("SLM_SPOT_DISCOUNT_FACTOR", DEFAULT_SPOT_DISCOUNT_FACTOR))
    except ValueError:
        return DEFAULT_SPOT_DISCOUNT_FACTOR


class AwsAccountUnresolvedError(RuntimeError):
    """The AWS account id could not be determined from Settings, env, or STS."""


# Account ids discovered via STS, keyed by (profile, region). A given set of
# credentials belongs to exactly one account, and load_aws_config() runs on
# nearly every request, so the lookup is cached for the life of the process.
# Failures are deliberately NOT cached: the next call then picks up credentials
# that have since been configured or refreshed.
_sts_account_cache: dict[tuple[str | None, str], str] = {}


def _sts_account(region: str, profile: str | None) -> str | None:
    """Account id of the ambient AWS credentials, or None if unavailable."""
    key = (profile, region)
    cached = _sts_account_cache.get(key)
    if cached:
        return cached
    try:
        from .aws_clients import get_session

        sess = get_session(profile_name=profile or None, region_name=region)
        account = str(sess.client("sts").get_caller_identity()["Account"])
    except Exception as e:  # noqa: BLE001 — no credentials / expired / no network
        _log.debug("STS GetCallerIdentity failed (%s): %s", type(e).__name__, e)
        return None
    _sts_account_cache[key] = account
    return account


def _resolve_account(region: str, profile: str | None) -> str:
    """The AWS account this deployment's bucket, role and image live in.

    saved config.json > SLM_AWS_ACCOUNT > the account of the ambient AWS
    credentials (STS GetCallerIdentity). There is deliberately no built-in
    default: a literal account id here would belong to whoever wrote it, and
    every ARN, bucket name and image URI derived from it would silently point
    somewhere the operator does not own.
    """
    explicit = _resolve("account", "SLM_AWS_ACCOUNT", "").strip()
    if explicit:
        return explicit
    discovered = _sts_account(region, profile)
    if discovered:
        return discovered
    raise AwsAccountUnresolvedError(
        "Cannot determine the AWS account for this deployment. Set "
        "SLM_AWS_ACCOUNT to your 12-digit account id (or set 'account' on the "
        "Settings page), or make AWS credentials available so the account can "
        "be read from STS GetCallerIdentity."
    )


def resolve_region() -> str:
    """The deployment region (saved config > SLM_AWS_REGION > DEFAULT_REGION)."""
    return _resolve("region", "SLM_AWS_REGION", DEFAULT_REGION)


def resolve_profile() -> str | None:
    """The named AWS profile to use, or None for ambient credentials.

    In Lambda (or any role-based env) there is NO named profile — boto3 must fall
    back to the ambient execution-role credentials. We detect that via
    AWS_LAMBDA_FUNCTION_NAME and force None. Locally the default is still the
    "default" profile (the dev-box workflow).
    """
    in_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    default_profile = "" if in_lambda else "default"
    return _resolve("profile", "SLM_AWS_PROFILE", default_profile) or None


def load_aws_config() -> AwsConfig:
    """The effective AWS config: saved config.json > env var > derived default.

    Raises AwsAccountUnresolvedError if the account cannot be determined; the
    bucket, role and image names are all derived from it.
    """
    # Defaults mirror what the CDK stack (SlmPlatformInfra) creates. Keeping the
    # local defaults identical means a job launched from a dev box lands in the
    # SAME region/bucket/role/image as the hosted app — so the deployed UI (and
    # its /curves) can see it too.
    region = resolve_region()
    # Resolved before the account because the account lookup uses these
    # credentials.
    profile = resolve_profile()
    account = _resolve_account(region, profile)
    role = _resolve(
        "roleArn", "SLM_SAGEMAKER_ROLE_ARN",
        f"arn:aws:iam::{account}:role/{RESOURCE_PREFIX}-sagemaker-exec",
    )
    bucket = _resolve("bucket", "SLM_S3_BUCKET", f"{RESOURCE_PREFIX}-data-{account}-{region}")
    image = _resolve(
        "imageUri", "SLM_TRAINING_IMAGE_URI",
        f"{account}.dkr.ecr.{region}.amazonaws.com/{RESOURCE_PREFIX}-llamafactory"
        f":{IMAGE_TIER_TAGS[DEFAULT_IMAGE_TIER]}",
    )
    return AwsConfig(
        region=region,
        account_id=account,
        role_arn=role,
        bucket=bucket,
        image_uri=image,
        profile=profile,
    )
