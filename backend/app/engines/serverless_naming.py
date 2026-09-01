# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Naming + ModelPackageGroup helpers for the serverless engine.

Kept separate so they're unit-testable without the SDK. Job names follow the
platform convention `slm-<modelid>-<splitid>-<stamp>` (so leaderboard/job-listing
heuristics parse them like LLaMA-Factory jobs); the S3 prefix is per-tenant
(mirrors orchestrate._jobs_key_prefix), and the ModelPackageGroup is per-tenant
so one user's custom models don't mix with another's.
"""

from __future__ import annotations

import re
from typing import Any


# The V3 SDK's _get_unique_name(base) builds f"{base}-{YYYYMMDDHHMMSS}" then HARD
# TRUNCATES to 63 chars. A 14-char timestamp + its leading hyphen needs 15 chars of
# headroom; if our base is too long the cut lands mid-name — and if it lands on a
# hyphen the result fails SageMaker's trainingJobName regex (must end alphanumeric),
# which is exactly what broke an RLAIF launch (the longer `-serverless-rlaif` head
# pushed the base to 62, leaving the SDK no room). So OUR base must be ≤48 chars
# (63 − 15) AND end alphanumeric. But we can't blindly truncate the tail: the
# leaderboard groups runs by parsing `slm-<modelid>-<splitid>` (the 12-hex split id
# — see leaderboard._resolve_train_job), so the split id + stage marker are
# load-bearing and must SURVIVE. We therefore shorten only the FLEXIBLE part (the
# model-id) and keep the structural tail (`-serverless[-stage]-<split>-<stamp>`)
# whole, dropping the stamp only as a last resort for an extreme-length model id.
_SDK_SUFFIX_HEADROOM = 15  # "-" + 14-char timestamp the V3 SDK appends
_MAX_BASE_LEN = 63 - _SDK_SUFFIX_HEADROOM  # 48


def serverless_job_name(model_id: str, split_id: str, stamp: str, stage: str = "sft") -> str:
    """`slm-<model>[-serverless[-stage]]-<split>-<stamp>`, sanitized, ≤48 chars so
    the V3 SDK's appended `-<timestamp>` (then truncate-to-63) still yields a valid
    trainingJobName (ends alphanumeric). Only the model-id is shortened to fit; the
    `-serverless[-stage]-<splitid>` tail is preserved so the leaderboard heuristic
    still parses the split id and the engine/stage marker.

    The `serverless` marker (and non-sft stage) ride in the model-id HEAD so the
    leaderboard's label heuristic (which keeps parts until the 12-hex split id)
    surfaces a serverless run as a DISTINCT row from the LLaMA-Factory one — the
    same trick the LF path uses for qlora. SFT stays unmarked-by-stage but tagged
    `serverless` so the two engines never collide on a job name."""
    def _san(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9-]", "-", s)

    model = _san(model_id)
    marker = "serverless" if stage == "sft" else f"serverless-{stage}"
    split = _san(split_id)
    st = _san(stamp)

    # Fixed (must-survive) structural parts: slm- + -marker- + split + -stamp.
    # Reserve their length, then give the model-id whatever budget remains. The
    # split id is never truncated (leaderboard grouping depends on it).
    def _assemble(model_part: str, stamp_part: str) -> str:
        tail = f"-{marker}-{split}" + (f"-{stamp_part}" if stamp_part else "")
        return f"slm-{model_part}{tail}"

    candidate = _assemble(model, st)
    if len(candidate) > _MAX_BASE_LEN:
        # Shrink the model-id first (the flexible, non-structural part).
        fixed_len = len(_assemble("", st))  # everything except the model-id
        model_budget = max(1, _MAX_BASE_LEN - fixed_len)
        candidate = _assemble(model[:model_budget].strip("-"), st)
    if len(candidate) > _MAX_BASE_LEN:
        # Still too long (extreme model id) → drop the stamp; the SDK's appended
        # timestamp still makes the final name unique.
        fixed_len = len(_assemble("", ""))
        model_budget = max(1, _MAX_BASE_LEN - fixed_len)
        candidate = _assemble(model[:model_budget].strip("-"), "")
    return candidate[:_MAX_BASE_LEN].strip("-")


def jobs_key_prefix() -> str:
    """Per-tenant S3 key prefix for jobs — reuse orchestrate's helper so the
    serverless engine writes under the SAME users/<sub>/jobs scheme."""
    from ..orchestrate import _jobs_key_prefix

    return _jobs_key_prefix()


def tenant_model_package_group() -> str:
    """ModelPackageGroup name for the current tenant. Default tenant → a shared
    group; other tenants get their own so registered custom models are isolated."""
    from ..tenancy import DEFAULT_TENANT, current_tenant

    tenant = current_tenant()
    if tenant == DEFAULT_TENANT:
        return "slm-platform-serverless"
    # MPG names: alphanumeric + hyphens, ≤63 chars.
    safe = re.sub(r"[^a-zA-Z0-9-]", "-", tenant)[:40].strip("-")
    return f"slm-serverless-{safe}"


def ensure_model_package_group(boto_sess: Any, name: str) -> str:
    """Create the ModelPackageGroup if missing (idempotent). The serverless
    trainer resolves the group NAME→ARN via describe and does NOT auto-create it,
    so we must ensure it exists first. Returns the group name."""
    sm = boto_sess.client("sagemaker")
    try:
        sm.describe_model_package_group(ModelPackageGroupName=name)
        return name
    except Exception:  # noqa: BLE001 — not found (or transient); try to create
        pass
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription="SLM platform serverless fine-tuned models",
        )
    except Exception as e:  # noqa: BLE001 — a concurrent create / already-exists race is fine
        # describe again; if it now exists we're good, else re-raise.
        try:
            sm.describe_model_package_group(ModelPackageGroupName=name)
        except Exception:
            raise e
    return name
