# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Self-healing image selection — the agentic layer of the multi-image design.

When a model FAILS on its current image, the platform should not just surface a
red error: it should figure out WHETHER the failure means "this model needs a
newer stack" and, if so, route the model to a newer image tier, smoke-test it
there, and update the catalog — so the next run just works. This is the user's
vision: an agent that makes image selection self-healing.

Design (consistent with the platform's "deterministic core, LLM for fuzzy
judgment" principle):
  * DETERMINISTIC triage first. A curated set of signature patterns
    (ImportError: LossKwargs, "requires transformers>=", unknown-template at the
    arg parser, CUDA/OOM, spot-capacity) classifies the most common failures with
    zero cost and full reproducibility. OOM / capacity / transient are explicitly
    NOT stack problems — a newer image won't help, so we never escalate them.
  * LLM CLASSIFIER fallback for the genuinely-fuzzy unknown reason. Bedrock
    Sonnet reads the failure text and decides needs_newer_stack (bounded JSON,
    deterministic temperature). Any error degrades to "unknown — don't escalate"
    so a Bedrock outage can never trigger a wrong image switch.
  * The ACTION is deterministic: pick the next tier up from the model's current
    one, ensure that image exists in ECR (build it via CodeBuild if missing), then
    launch a capped smoke-test on the new tier. The verification store records the
    result; if it passes, the operator (or a follow-up) flips the model's tier.

Nothing here is automatic-by-surprise: escalation is triggered explicitly (an API
action / the agent loop), classification is conservative (default = do NOT
escalate), and every billable step is a smoke test the user already understands.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .aws_config import DEFAULT_IMAGE_TIER, image_tiers, load_aws_config
from .catalog import ModelSpec

# Default classifier model id. The LIVE model is resolved per call from the
# 'selfheal' role via agent_models.resolve_model_id (Settings-overridable); this
# constant documents the default.
CLASSIFIER_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Deterministic failure signatures. Each maps a regex over the failure reason /
# log text → a classification. `needs_newer_stack=True` means a newer image
# COULD fix it; False means it definitively won't (don't waste a build).
_SIGNATURES: list[dict[str, Any]] = [
    {
        "pattern": r"cannot import name '?LossKwargs'?",
        "category": "stack_too_old",
        "needs_newer_stack": True,
        "explanation": "transformers is too old for this model (LossKwargs needs v5)",
    },
    {
        "pattern": r"requires? transformers[>=]",
        "category": "stack_too_old",
        "needs_newer_stack": True,
        "explanation": "model explicitly requires a newer transformers version",
    },
    {
        "pattern": r"requires? .*accelerate[>=]|requires? .*peft[>=]",
        "category": "stack_too_old",
        "needs_newer_stack": True,
        "explanation": "model requires a newer training-stack dependency",
    },
    {
        "pattern": r"is not (a )?valid template|Some keys are not used by the HfArgumentParser",
        "category": "config_rejected",
        "needs_newer_stack": True,
        "explanation": "engine rejected the template/config — a newer engine may add it",
    },
    {
        "pattern": r"out of memory|CUDA out of memory|OOM",
        "category": "oom",
        "needs_newer_stack": False,
        # Method-dependent: for LoRA/QLoRA a bigger instance usually fixes it; for
        # full/freeze (full-weight) the fix is more often freeze/fewer trainable
        # layers/grad-checkpointing/smaller model — not just a bigger box. Left
        # generic so the triage agent (which sees finetuning_type) decides.
        "explanation": "out of memory — not an image problem; the fix depends on the "
                       "method (bigger instance for LoRA; freeze/grad-checkpointing/"
                       "smaller model for full-weight)",
    },
    {
        "pattern": r"Insufficient capacity|CapacityError|spot",
        "category": "capacity",
        "needs_newer_stack": False,
        "explanation": "transient capacity issue — retry; not an image problem",
    },
]


# A still-RUNNING job stuck waiting for spot capacity emits this in its secondary
# status / failure reason. Narrower than the broad capacity signature above (no
# bare "spot", which would match benign mentions) — used by the spot→on-demand
# fallback to decide a job is genuinely capacity-stalled, not just slow to start.
_CAPACITY_STALL_RE = re.compile(
    r"Insufficient capacity|CapacityError|InsufficientInstanceCapacity|"
    r"capacity error|waiting for (spot )?capacity|spot.*capacity",
    re.IGNORECASE,
)


def is_capacity_stall(text: str | None) -> bool:
    """True if a job's secondary-status/failure text indicates it's stuck WAITING
    for (spot) capacity — the trigger for the spot→on-demand fallback. Deliberately
    narrow so a generic 'Starting'/'Downloading' status never matches."""
    return bool(text and _CAPACITY_STALL_RE.search(text))


def classify_failure(reason: str | None) -> dict[str, Any]:
    """Classify a training failure reason. Deterministic signatures first, then
    an LLM fallback for an unrecognized reason. Returns
    {category, needsNewerStack, explanation, source}."""
    text = (reason or "").strip()
    if not text:
        return {"category": "unknown", "needsNewerStack": False,
                "explanation": "no failure reason recorded", "source": "deterministic"}

    for sig in _SIGNATURES:
        if re.search(sig["pattern"], text, re.IGNORECASE):
            return {
                "category": sig["category"],
                "needsNewerStack": sig["needs_newer_stack"],
                "explanation": sig["explanation"],
                "source": "deterministic",
            }

    return _classify_with_llm(text)


def _classify_with_llm(text: str) -> dict[str, Any]:
    """Bedrock fallback for an unrecognized failure. Conservative: any error →
    'unknown, do not escalate' so an outage never triggers a wrong image switch."""
    prompt = (
        "You triage machine-learning training-job failures. Decide whether the "
        "failure below means the model needs a NEWER software stack (newer "
        "transformers / LLaMA-Factory / PyTorch image) to load or train. "
        "Out-of-memory, disk, network, spot-capacity, and dataset errors do NOT "
        "need a newer stack. Respond with STRICT JSON only.\n\n"
        f"Failure text:\n{text[:4000]}\n\n"
        'Respond as: {"needsNewerStack": true|false, "category": "short_snake_case", '
        '"explanation": "one sentence"}.'
    )
    try:
        from .orchestrate import _session
        from .agent_models import resolve_model_id

        cfg = load_aws_config()
        _, boto_sess = _session(cfg)
        client = boto_sess.client("bedrock-runtime", region_name=cfg.region)
        # Converse API — model-AGNOSTIC, so the classifier model is user-selectable
        # in Settings ('selfheal' role). Any error still degrades to "don't escalate".
        resp = client.converse(
            modelId=resolve_model_id("selfheal"),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 400, "temperature": 0.0},
        )
        out_msg = resp.get("output", {}).get("message", {})
        out = "".join(b.get("text", "") for b in out_msg.get("content", []))
        start, end = out.find("{"), out.rfind("}")
        parsed = json.loads(out[start : end + 1]) if start != -1 and end > start else {}
        return {
            "category": str(parsed.get("category", "unknown"))[:40],
            "needsNewerStack": bool(parsed.get("needsNewerStack", False)),
            "explanation": str(parsed.get("explanation", ""))[:300],
            "source": "llm",
        }
    except Exception as e:  # noqa: BLE001 — never escalate on a classifier outage
        return {"category": "unknown", "needsNewerStack": False,
                "explanation": f"classifier unavailable ({e}); not escalating", "source": "fallback"}


def next_tier_up(current_tag: str | None) -> str | None:
    """The next image tier a model should try when its current one can't load it.

    Tiers are ordered by their ECR tag (a version string), so the "next" tier is
    the smallest tag strictly greater than the current. Returns None when the
    model is already on the newest tier (nothing newer to try)."""
    tiers = image_tiers()  # {tierName: tag}
    current = tiers.get(current_tag or DEFAULT_IMAGE_TIER, tiers.get(DEFAULT_IMAGE_TIER))

    def ver(tag: str) -> tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", tag))

    # Candidates strictly newer than the current tag, by version.
    newer = [(name, tag) for name, tag in tiers.items() if ver(tag) > ver(current)]
    if not newer:
        return None
    newer.sort(key=lambda nt: ver(nt[1]))
    return newer[0][0]  # the tier NAME (e.g. "latest")


def image_exists_in_ecr(image_tag: str) -> bool:
    """Whether the tier's image tag is already pushed to ECR (so we can smoke-test
    it without first building). Read-only."""
    from .orchestrate import _session

    cfg = load_aws_config()
    tag = image_tiers().get(image_tag, image_tag)
    repo = cfg.image_repo.split("/")[-1]
    try:
        _, boto_sess = _session(cfg)
        boto_sess.client("ecr").describe_images(
            repositoryName=repo, imageIds=[{"imageTag": tag}]
        )
        return True
    except Exception:  # noqa: BLE001 — missing tag or no access → treat as absent
        return False


# The shared CodeBuild project that builds ANY LLaMA-Factory tag via env-var
# overrides (LF_BASE_TAG / TAG / VLLM_VERSION). Tiers registered at RUNTIME (via
# the Images page "Build <tag>" action, releases.build_release) have NO per-tier
# CDK project, so their rebuild + status must route through THIS one, keyed to the
# tag. Mirrors releases.py:123. Confirmed live as a stack output (...-build-adhoc).
ADHOC_BUILD_PROJECT = "slm-platform-training-image-build-adhoc"
# Default vLLM pin used by the adhoc build (matches releases.build_release).
DEFAULT_VLLM_VERSION = "0.8.5.post1"


def build_project_for(image_tag: str) -> str:
    """The per-tier CodeBuild project name (CDK convention: ...-build-<tag-
    without-dots>). NOTE: this project only exists for tiers baked into the CDK
    stack (e.g. 094/095). Tiers added at runtime via "Build <tag>" have no such
    project — callers must fall back to ADHOC_BUILD_PROJECT (see _codebuild_project_exists)."""
    cid = image_tiers().get(image_tag, image_tag).replace(".", "")
    return f"slm-platform-training-image-build-{cid}"


def _codebuild_project_exists(cb, project: str) -> bool:
    """True if a CodeBuild project with this exact name exists in the account."""
    try:
        found = cb.batch_get_projects(names=[project]).get("projects", [])
        return any(p.get("name") == project for p in found)
    except Exception:  # noqa: BLE001 — no access / transient → assume absent, use adhoc
        return False


def image_tier_status() -> list[dict[str, Any]]:
    """Rich status for every image tier, for the Images management page.

    Per tier: the ECR tag it maps to, whether that tag is present in ECR (+ when
    it was pushed and its size), the CodeBuild project that builds it, and that
    project's most-recent build status. One ECR describe_images call covers all
    tags; one batch_get_builds covers all projects' latest builds. Read-only."""
    from .orchestrate import _session
    from .verifications import all_verifications

    cfg = load_aws_config()
    tiers = image_tiers()  # {tierName: tag}
    repo = cfg.image_repo.split("/")[-1]
    _, boto_sess = _session(cfg)
    ecr = boto_sess.client("ecr")
    cb = boto_sess.client("codebuild")

    # ECR: pull all of the repo's image details once, index by tag.
    by_tag: dict[str, dict[str, Any]] = {}
    try:
        paginator = ecr.get_paginator("describe_images")
        for page in paginator.paginate(repositoryName=repo):
            for d in page.get("imageDetails", []):
                for t in d.get("imageTags", []) or []:
                    by_tag[t] = d
    except Exception:  # noqa: BLE001 — repo missing / no access → all absent
        by_tag = {}

    # Count verified models per tier (informational on the management page).
    verifs = all_verifications()
    verified_counts: dict[str, int] = {}
    for _model, per in verifs.items():
        for tier_tag, rec in per.items():
            if rec.get("status") == "verified":
                verified_counts[tier_tag] = verified_counts.get(tier_tag, 0) + 1

    out: list[dict[str, Any]] = []
    for tier_name, tag in tiers.items():
        det = by_tag.get(tag)
        # Built-in tiers have a per-tier project; runtime ("Build <tag>") tiers were
        # built by the shared adhoc project and have none — for those, report the
        # adhoc project and find THIS tag's latest build among the adhoc builds
        # (so the row shows a real last-build status instead of "never built").
        per_tier = build_project_for(tier_name)
        adhoc = not _codebuild_project_exists(cb, per_tier)
        project = ADHOC_BUILD_PROJECT if adhoc else per_tier
        last_build = None
        try:
            ids = cb.list_builds_for_project(projectName=project, sortOrder="DESCENDING").get("ids", [])
            if ids:
                # For the adhoc project (shared across tags), scan recent builds and
                # pick the latest whose TAG env-override matches this tier's tag.
                # For a per-tier project every build is this tier's, so take the top.
                scan = ids[:20] if adhoc else ids[:1]
                builds = cb.batch_get_builds(ids=scan).get("builds", []) if scan else []
                chosen = None
                if adhoc:
                    for b in builds:  # builds are DESCENDING by recency
                        envs = {e.get("name"): e.get("value")
                                for e in (b.get("environment", {}) or {}).get("environmentVariables", [])}
                        ov = {e.get("name"): e.get("value") for e in (b.get("environmentVariablesOverride") or [])}
                        if ov.get("TAG") == tag or envs.get("TAG") == tag:
                            chosen = b
                            break
                else:
                    chosen = builds[0] if builds else None
                if chosen:
                    last_build = {
                        "status": chosen.get("buildStatus"),
                        "id": chosen.get("id"),
                        "startTime": str(chosen.get("startTime")) if chosen.get("startTime") else None,
                    }
        except Exception:  # noqa: BLE001 — project may not exist yet
            last_build = None
        out.append({
            "tier": tier_name,
            "tag": tag,
            "imageUri": f"{cfg.image_repo}:{tag}",
            "existsInEcr": det is not None,
            "pushedAt": str(det.get("imagePushedAt")) if det and det.get("imagePushedAt") else None,
            "sizeMB": round(det["imageSizeInBytes"] / 1e6) if det and det.get("imageSizeInBytes") else None,
            "buildProject": project,
            "lastBuild": last_build,
            "verifiedModels": verified_counts.get(tier_name, 0),
        })
    return out


def trigger_image_build(image_tag: str, project_name: str | None = None) -> dict[str, Any]:
    """Kick off the CodeBuild project that builds a tier's image. Returns the
    build id so the caller can poll.

    Built-in tiers (094/095) have a dedicated per-tier project. Tiers added at
    RUNTIME via "Build <tag>" have NO per-tier project — their image was built by
    the shared adhoc project. So if the per-tier project doesn't exist we route
    the (re)build through ADHOC_BUILD_PROJECT with TAG/LF_BASE_TAG/VLLM_VERSION
    overrides (mirrors releases.build_release), instead of 502-ing on a missing
    project. An explicit project_name still wins (caller override)."""
    from .orchestrate import _session

    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    cb = boto_sess.client("codebuild")
    # Resolve the ECR tag this tier maps to (the adhoc build keys on it).
    tag = image_tiers().get(image_tag, image_tag)

    if project_name:
        project, overrides = project_name, None
    else:
        per_tier = build_project_for(image_tag)
        if _codebuild_project_exists(cb, per_tier):
            project, overrides = per_tier, None
        else:
            # Runtime/adhoc tier (or a per-tier project that was never created):
            # rebuild via the shared adhoc project, parameterized to this tag.
            project = ADHOC_BUILD_PROJECT
            overrides = [
                {"name": "LF_BASE_TAG", "value": tag, "type": "PLAINTEXT"},
                {"name": "TAG", "value": tag, "type": "PLAINTEXT"},
                {"name": "VLLM_VERSION", "value": DEFAULT_VLLM_VERSION, "type": "PLAINTEXT"},
            ]

    kwargs: dict[str, Any] = {"projectName": project}
    if overrides:
        kwargs["environmentVariablesOverride"] = overrides
    resp = cb.start_build(**kwargs)
    build = resp["build"]
    return {"buildId": build["id"], "project": project, "status": build.get("buildStatus")}


def diagnose(model: ModelSpec, failure_reason: str | None) -> dict[str, Any]:
    """Full diagnosis for a failed model run: classify the failure, and if it
    looks like a stack-too-old problem, recommend the next image tier + whether
    that image already exists. Read-only — recommends, does not act.

    The orchestration (build if missing → smoke-test on the new tier → record
    verification → suggest flipping the model's tier) is driven by the caller /
    the agent loop so each billable step stays explicit."""
    classification = classify_failure(failure_reason)
    current_tag = getattr(model, "image_tag", DEFAULT_IMAGE_TIER)
    result: dict[str, Any] = {
        "modelId": model.id,
        "currentTier": current_tag,
        "classification": classification,
        "recommendedTier": None,
        "imageReady": None,
        "action": "none",
    }
    if not classification["needsNewerStack"]:
        result["action"] = "no_image_change"  # OOM/capacity/transient/unknown
        return result

    target = next_tier_up(current_tag)
    if target is None:
        result["action"] = "already_newest"  # nothing newer to escalate to
        return result

    result["recommendedTier"] = target
    ready = image_exists_in_ecr(target)
    result["imageReady"] = ready
    result["action"] = "smoke_test" if ready else "build_then_smoke_test"
    return result
