# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LLaMA-Factory release tracking + new-model discovery.

Two operator features built on the multi-image design:

1. RELEASE CHECK / ADHOC BUILD — query Docker Hub for the LLaMA-Factory tags and
   compare against the image tiers we already build. When a newer release exists,
   one call builds it: the generic adhoc CodeBuild project (parameterized by env
   overrides) builds `hiyouga/llamafactory:<tag>` → pushes to our ECR → writes a
   capability manifest to S3. The new image is then registered as a tier in
   config.json AT RUNTIME (no cdk deploy). New tier = every model "untested";
   existing models stay pinned to their proven tier (auto-build, never auto-trust).

2. MODEL DISCOVERY — each built image writes a capability manifest (the set of
   transformers model architectures + LLaMA-Factory templates it supports) to
   s3://<bucket>/slm-platform/image-meta/<tag>.json during its build. Discovery
   diffs a new image's manifest against an older one to surface the architectures
   the new release ADDED, maps them to popular Hugging Face models, and returns
   them as SUGGESTIONS to probe + smoke-test (never auto-onboards — verify first).

Deterministic core; the only network calls are Docker Hub (public, read-only) and
S3 (read the manifests). Building/verifying stays gated behind explicit actions.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from .aws_config import IMAGE_META_PREFIX_DEFAULT, image_tiers, load_aws_config, register_image_tier

DOCKERHUB_TAGS = "https://hub.docker.com/v2/repositories/hiyouga/llamafactory/tags?page_size=100"

# A LLaMA-Factory / vLLM release tag is a dotted version, optionally with a
# pre-release suffix (e.g. '0.9.5', '0.8.5.post1'). Validating against this
# allowlist BEFORE the value reaches a Docker image reference or a CodeBuild
# environment override closes the tool-misuse surface (OWASP-Agentic ASI-02 /
# CWE-918): it rejects shell/registry metacharacters, path traversal and
# alternate-image injection that a free-form string could otherwise carry.
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _validate_tag(tag: str, what: str = "tag") -> str:
    """Return a clean version tag or raise ValueError. The trust boundary for
    every tag that flows into an image reference or a build override."""
    if not isinstance(tag, str):
        raise ValueError(f"{what} must be a string")
    tag = tag.strip()
    if not tag or not _TAG_RE.match(tag):
        raise ValueError(
            f"invalid {what} '{tag}': expected a version-like tag "
            "(letters, digits, '.', '_' or '-')"
        )
    return tag


def _ver(tag: str) -> tuple[int, ...]:
    """Numeric version tuple from a tag ('0.9.5' → (0,9,5)); () if non-numeric."""
    parts = re.findall(r"\d+", tag)
    return tuple(int(p) for p in parts) if parts else ()


def list_upstream_releases() -> list[str]:
    """LLaMA-Factory release tags from Docker Hub, newest-first, NPU/latest excluded.
    Returns [] on any network error (the UI then just shows 'could not check')."""
    req = urllib.request.Request(DOCKERHUB_TAGS, headers={"User-Agent": "slm-platform"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (fixed host)
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return []
    tags = []
    for t in data.get("results", []):
        name = t.get("name", "")
        if not name or "npu" in name or name == "latest":
            continue
        if _ver(name):  # keep only version-like tags
            tags.append(name)
    return sorted(set(tags), key=_ver, reverse=True)


def check_for_updates() -> dict[str, Any]:
    """Compare upstream LLaMA-Factory releases against the tags we already build.

    Returns {upstream:[...], builtTags:[...], newest, haveNewest(bool),
    newReleases:[tags we don't build yet, newest-first]}."""
    upstream = list_upstream_releases()
    built = sorted(set(image_tiers().values()), key=_ver, reverse=True)
    built_set = set(built)
    new_releases = [t for t in upstream if t not in built_set]
    # Only flag releases NEWER than our newest built tag as actionable upgrades.
    newest_built = built[0] if built else ""
    actionable = [t for t in new_releases if not newest_built or _ver(t) > _ver(newest_built)]
    return {
        "upstream": upstream,
        "builtTags": built,
        "newest": upstream[0] if upstream else None,
        "haveNewest": bool(upstream) and upstream[0] in built_set,
        "newReleases": actionable,
    }


def _tier_name_for(tag: str) -> str:
    """A stable tier name derived from a version tag ('0.9.6' → 'v096')."""
    return "v" + re.sub(r"[^0-9]", "", tag)


def build_release(lf_tag: str, vllm_version: str = "0.8.5.post1") -> dict[str, Any]:
    """Build a specific LLaMA-Factory release via the adhoc CodeBuild project and
    register it as a new image tier (so it's usable immediately, no redeploy).

    Uses environmentVariablesOverride to point the shared adhoc project at this
    release: LF_BASE_TAG (Docker Hub base) + TAG (our ECR tag = the same version) +
    VLLM_VERSION. The image lands as ECR tag `lf_tag`; the tier name is derived
    (e.g. 'v096'). Verification for every model on the new tier starts 'untested'.
    Returns {buildId, project, tier, tag, tiers}."""
    from .orchestrate import _session

    lf_tag = _validate_tag(lf_tag, "lf_tag")  # gate before it reaches CodeBuild
    vllm_version = _validate_tag(vllm_version, "vllm_version")
    cfg = load_aws_config()
    project = "slm-platform-training-image-build-adhoc"
    _, boto_sess = _session(cfg)
    resp = boto_sess.client("codebuild").start_build(
        projectName=project,
        environmentVariablesOverride=[
            {"name": "LF_BASE_TAG", "value": lf_tag, "type": "PLAINTEXT"},
            {"name": "TAG", "value": lf_tag, "type": "PLAINTEXT"},
            {"name": "VLLM_VERSION", "value": vllm_version, "type": "PLAINTEXT"},
        ],
    )
    tier = _tier_name_for(lf_tag)
    tiers = register_image_tier(tier, lf_tag)  # usable at runtime, no cdk deploy
    return {
        "buildId": resp["build"]["id"],
        "project": project,
        "tier": tier,
        "tag": lf_tag,
        "buildStatus": resp["build"].get("buildStatus"),
        "tiers": tiers,
    }


def _read_image_meta(tag: str) -> dict[str, Any] | None:
    """Read an image's capability manifest (written to S3 during its build).
    None if absent (image never built, or built before manifests existed)."""
    from .orchestrate import _session

    cfg = load_aws_config()
    key = f"{IMAGE_META_PREFIX_DEFAULT}/{tag}.json"
    try:
        _, boto_sess = _session(cfg)
        obj = boto_sess.client("s3").get_object(Bucket=cfg.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


# Newly-supported model_type → a popular, real, public HF model to suggest. This
# curated map turns a raw architecture name into an actionable onboarding pick.
#
# IMPORTANT: only list repos that ACTUALLY EXIST on Hugging Face today. A new
# transformers release often adds an architecture BEFORE the vendor publishes
# weights (e.g. 0.9.5/transformers-5 knows `gemma4`/`qwen3_5_moe`/`mistral4`/
# `solar_open` but those repos 404), so guessing names makes the "probe & add"
# button fail with a confusing error. Verified-present 2026-06-04 (HF API 200);
# add more here only after confirming the repo resolves. Architectures with no
# entry still surface as "new arch" in the UI, just without a clickable repo.
_ARCH_TO_HF: dict[str, list[str]] = {
    "lfm2": ["LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-350M"],
    "lfm2_moe": ["LiquidAI/LFM2-8B-A1B"],
    "qwen3_5": ["Qwen/Qwen3.5-4B"],
    "ministral3": ["ministral/Ministral-3b-instruct"],
    "nemotron_h": ["nvidia/Nemotron-H-4B-Instruct-128K"],
}


def discover_new_models(new_tag: str, base_tag: str | None = None) -> dict[str, Any]:
    """Architectures the `new_tag` image supports that `base_tag` did NOT — mapped
    to suggested HF models to probe + smoke-test.

    base_tag defaults to the newest OTHER built tag older than new_tag. Returns
    {newTag, baseTag, newArchitectures:[...], newTemplates:[...],
    suggestions:[{architecture, repos:[...]}], note}. Empty (with a note) when a
    manifest is missing (e.g. the image predates capability capture)."""
    new_tag = _validate_tag(new_tag, "new_tag")  # gate before it builds an S3 key
    if base_tag is not None:
        base_tag = _validate_tag(base_tag, "base_tag")
    new_meta = _read_image_meta(new_tag)
    if not new_meta or not new_meta.get("model_types"):
        return {
            "newTag": new_tag, "baseTag": base_tag,
            "newArchitectures": [], "newTemplates": [], "suggestions": [],
            "note": f"no capability manifest for image '{new_tag}' yet — rebuild it to capture one.",
        }

    if base_tag is None:
        built = sorted(set(image_tiers().values()), key=_ver)
        older = [t for t in built if _ver(t) < _ver(new_tag)]
        base_tag = older[-1] if older else None

    base_meta = _read_image_meta(base_tag) if base_tag else None
    base_arches = set(base_meta.get("model_types", [])) if base_meta else set()
    base_templates = set(base_meta.get("templates", [])) if base_meta else set()

    new_arches = sorted(set(new_meta.get("model_types", [])) - base_arches)
    new_templates = sorted(set(new_meta.get("templates", [])) - base_templates)

    # Build suggestions from the arch→repos map, but DROP any repo already in the
    # catalog (built-in OR custom-onboarded) — "Newly supported" is for models you
    # don't have yet, so an already-added model must not keep showing up. An
    # architecture whose every suggested repo is already cataloged is dropped
    # entirely. Track how many were filtered so the UI can say so (no silent hide).
    from .catalog import model_id_for_hf

    suggestions: list[dict[str, Any]] = []
    already_in_catalog = 0
    for a in new_arches:
        if a not in _ARCH_TO_HF:
            continue
        fresh = [r for r in _ARCH_TO_HF[a] if not model_id_for_hf(r)]
        already_in_catalog += len(_ARCH_TO_HF[a]) - len(fresh)
        if fresh:
            suggestions.append({"architecture": a, "repos": fresh})
    note = ""
    if base_meta is None:
        note = (
            f"no manifest for a baseline image to diff against — showing ALL "
            f"architectures '{new_tag}' supports rather than only the new ones."
        )
        if base_tag is None:
            new_arches = sorted(new_meta.get("model_types", []))
    if already_in_catalog:
        hidden = (f"{already_in_catalog} suggested model(s) already in your catalog "
                  "were hidden.")
        note = f"{note} {hidden}".strip() if note else hidden
    return {
        "newTag": new_tag,
        "baseTag": base_tag,
        "transformers": new_meta.get("transformers"),
        "newArchitectures": new_arches,
        "newTemplates": new_templates,
        "suggestions": suggestions,
        "note": note,
    }
