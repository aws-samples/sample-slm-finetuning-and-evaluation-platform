# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Export a fine-tuned winner so the user can deploy it in THEIR OWN AWS account.

A finished training job leaves a single tarball in S3:

    {s3_prefix}/jobs/{train_job}/output/model.tar.gz

which (per render.py / entrypoint.sh) contains BOTH artifacts SageMaker tarred
from /opt/ml/model:

    adapter/   the LoRA adapter (the user's actual fine-tuning delta)
    merged/    base ⊕ adapter merged to standalone safetensors

So the same tarball serves either delivery mode; the base model's license decides
which the user should deploy:

  * ungated base (Qwen/Apache, Granite/Apache, Phi/MIT, MiniCPM) → ship MERGED
    (self-contained, no HF dependency, redistribution permitted).
  * gated base (Llama/Gemma/Mistral) → ship ADAPTER-only; the deploy script pulls
    the base from HF with the user's own token. Redistributing merged gated
    weights would breach the model license, so we never hand those over.

This module produces (a) a manifest describing the model + which mode applies and
(b) a time-limited presigned URL to the weights tarball — the browser only ever
downloads a tiny scripts bundle (see bundle.py), and the weights are fetched
directly from S3 by the deploy script. The presigned URL lets the user's account
read this one object without any cross-account role setup.

Two engines, two artifact shapes:

  * llama_factory (default) — a single `model.tar.gz` holding adapter/ + merged/.
    One object → one presigned URL (`weightsUrl`); deploy.sh extracts the tarball
    and picks the licensed subdir.
  * sagemaker_serverless — an UNCOMPRESSED S3 prefix; the loadable HF model lives
    in `checkpoints/hf_merged/` (ungated → merged) and the LoRA adapter in
    `checkpoints/hf/` (gated → adapter). There's no tarball, so we presign EACH
    file under the chosen subdir (`weightsFiles`); deploy.sh downloads them
    straight into the model dir. The deploy image is still the LLaMA-Factory tag
    the model maps to — the race eval bridge already loads the serverless merged
    model on that exact image, so inference.py/Dockerfile are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .aws_config import AwsConfig, load_aws_config
from .catalog import get_model
from .orchestrate import _session
from .race import _load

# How long the weights download link stays valid. Long enough to run deploy.sh
# (which does an s3 cp early), short enough that a leaked link expires.
PRESIGN_TTL_SECONDS = 6 * 60 * 60  # 6 hours


class ExportError(Exception):
    """A model can't be exported (not found, not finished, no artifact)."""


@dataclass
class ExportManifest:
    race_id: str
    model_id: str
    model_display: str
    hf_base_model: str
    template: str
    gated: bool
    # "merged": ship the standalone merged model; "adapter": ship the LoRA
    # adapter + pull the base from HF at deploy time (gated bases).
    deploy_mode: str
    weights_subdir: str  # which dir inside the tarball to deploy ("merged"/"adapter")
    suggested_instance: str
    train_job: str
    artifact_s3_uri: str  # s3://…/model.tar.gz (the tarball with adapter/ + merged/)
    artifact_region: str  # region the source bucket lives in (for fast aws s3 cp)
    # The LLaMA-Factory base-image tag the model was TRAINED on (= its catalog
    # tier's tag, e.g. "0.9.4"/"0.9.5"). The deploy bundle's Dockerfile does
    # `FROM hiyouga/llamafactory:<this>` so inference runs on the EXACT same
    # transformers/torch/vllm versions training used — no version-mismatch guess
    # (the customer pulls that public LLaMA-Factory image themselves).
    base_image_tag: str
    # Which training engine produced the artifact: "llama_factory" (tarball) or
    # "sagemaker_serverless" (uncompressed prefix of loose files). deploy.sh keys
    # off this to download-and-extract vs download loose files into the model dir.
    engine: str = "llama_factory"
    # True when the downloadable artifact is MERGED weights of a GATED base — i.e. a
    # gated full/freeze fine-tune. Distributing it embeds the gated base, so the
    # download is gated behind the user accepting that base's license (Option B).
    # False for ungated (no restriction) and gated-adapter (ships only the adapter).
    requires_license_acceptance: bool = False
    # Whether the exported endpoint may execute modeling code shipped inside the
    # base model's repo (`auto_map`). Carried through from the catalog so the
    # bundle's inference.py runs with the same setting the platform trained and
    # evaluated under, instead of hardcoding it on.
    trust_remote_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "raceId": self.race_id,
            "modelId": self.model_id,
            "modelDisplay": self.model_display,
            "hfBaseModel": self.hf_base_model,
            "template": self.template,
            "gated": self.gated,
            "deployMode": self.deploy_mode,
            "weightsSubdir": self.weights_subdir,
            "artifactRegion": self.artifact_region,
            "suggestedInstance": self.suggested_instance,
            "trainJob": self.train_job,
            "artifactS3Uri": self.artifact_s3_uri,
            "baseImageTag": self.base_image_tag,
            "requiresLicenseAcceptance": self.requires_license_acceptance,
            "trustRemoteCode": self.trust_remote_code,
            # "sagemaker_serverless_adapter" is an INTERNAL delivery variant (gated
            # LLaMA-Factory adapter from a checkpoint dir, filtered to adapter
            # files). deploy.sh only knows llama_factory vs sagemaker_serverless, and
            # the loose-file branch is identical for both serverless variants, so
            # collapse it to "sagemaker_serverless" in the customer-facing manifest.
            "engine": ("sagemaker_serverless"
                       if self.engine == "sagemaker_serverless_adapter"
                       else self.engine),
        }


# Where the loadable weights live INSIDE a serverless job's (uncompressed) output
# prefix — verified on real completed runs:
#   checkpoints/hf_merged/  standalone merged HF model (config + model.safetensors)
#   checkpoints/hf/         the LoRA adapter (adapter_config.json + adapter_model.safetensors)
_SERVERLESS_MERGED_SUBDIR = "checkpoints/hf_merged"
_SERVERLESS_ADAPTER_SUBDIR = "checkpoints/hf"


def _artifact_uri(cfg: AwsConfig, train_job: str) -> str:
    """Resolve the model.tar.gz S3 URI for a finished training job. Prefer the
    authoritative path from describe_training_job; fall back to the conventional
    output path so this still works if the describe call is unavailable."""
    try:
        sm_sess, boto_sess = _session(cfg)
        sm = boto_sess.client("sagemaker")
        d = sm.describe_training_job(TrainingJobName=train_job)
        art = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
        if art:
            return art
    except Exception:  # noqa: BLE001 — fall back to the conventional path
        pass
    # Fallback (describe unavailable): the conventional output path for the
    # CURRENT tenant. Tenant jobs live under users/<sub>/jobs/, default under
    # slm-platform/jobs/ — reuse orchestrate's helper so both stay in sync.
    from .orchestrate import _jobs_s3_base

    return f"{_jobs_s3_base(cfg)}/{train_job}/output/model.tar.gz"


def _serverless_artifact_prefix(cfg: AwsConfig, train_job: str, subdir: str) -> str:
    """Resolve the S3 PREFIX holding a serverless job's loose weight files.

    Serverless output is an uncompressed prefix (`S3ModelArtifacts` points at its
    root); the deployable model lives under `subdir` (hf_merged / hf). Unlike the
    LLaMA-Factory tarball there's no reliable conventional fallback — SageMaker
    double-nests the managed output path — so if describe is unavailable we raise
    a clear error rather than hand back a wrong prefix."""
    try:
        _, boto_sess = _session(cfg)
        sm = boto_sess.client("sagemaker")
        d = sm.describe_training_job(TrainingJobName=train_job)
        art = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    except Exception as e:  # noqa: BLE001 — surface, don't guess a wrong prefix
        raise ExportError(
            f"can't resolve serverless artifact prefix for {train_job}: {e}"
        ) from e
    if not art:
        raise ExportError(f"serverless job {train_job} has no model artifact yet")
    return f"{art.rstrip('/')}/{subdir}"


def _latest_checkpoint_prefix(cfg: AwsConfig, checkpoint_s3: str) -> str | None:
    """Resolve the latest `checkpoint-N/` dir under a LLaMA-Factory job's synced
    checkpoint prefix — where the LoRA adapter (adapter_config.json +
    adapter_model.safetensors) lives (training writes output_dir=CHECKPOINT_DIR for
    every job; verified on a real on-demand run). Picks the highest step N.

    Used for the GATED adapter deploy. Returns None (rather than raising) when no
    checkpoint-N/ dir exists — e.g. an entry trained BEFORE the unify, or one whose
    checkpoint never synced — so the caller can fall back to the legacy
    model.tar.gz adapter/ path instead of failing the export."""
    try:
        bucket, prefix = _split_s3_uri(checkpoint_s3)
    except ExportError:
        return None
    prefix = prefix.rstrip("/") + "/"
    _, boto_sess = _session(cfg)
    s3 = boto_sess.client("s3")
    steps: list[tuple[int, str]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            sub = cp["Prefix"][len(prefix):].rstrip("/")  # e.g. "checkpoint-48"
            if sub.startswith("checkpoint-") and sub[len("checkpoint-"):].isdigit():
                steps.append((int(sub[len("checkpoint-"):]), cp["Prefix"]))
    if not steps:
        return None
    steps.sort()
    return f"s3://{bucket}/{steps[-1][1].rstrip('/')}"


def build_manifest(race_id: str, entry_key: str) -> ExportManifest:
    """Assemble the export manifest for one race entry's winning fine-tune.
    `entry_key` is the stable per-entry id (model_id, or model_id::method)."""
    from .race import _find_entry

    race = _load(race_id)
    if race is None:
        raise ExportError(f"race not found: {race_id}")
    entry = _find_entry(race, entry_key)
    if entry is None:
        raise ExportError(f"entry {entry_key!r} is not in race {race_id}")
    if not entry.train_job:
        raise ExportError(f"{entry_key} has no training job yet — nothing to export")

    model = get_model(entry.model_id)
    if model is None:
        raise ExportError(f"unknown model in catalog: {entry.model_id}")

    # Is this a FULL-WEIGHT (full/freeze) run? Those produce a standalone model with
    # NO adapter (the entrypoint stages it into merged/), so the adapter-delivery
    # path CANNOT apply — there is nothing adapter-shaped to ship. A gated full-FT
    # therefore MUST deliver MERGED weights, which embeds the gated base → it's a
    # redistribution that requires the user to accept the base model's license
    # (Option B). requires_license_acceptance flags exactly that case for the
    # download gate; ungated full-FT and all adapter runs are unaffected.
    method = (entry.hp or {}).get("finetuning_type", "lora")
    is_full_weight = method in ("full", "freeze")

    # License-driven delivery: gated ADAPTER runs → adapter-only (don't redistribute
    # merged gated weights); ungated → merged standalone. full/freeze are always
    # merged (no adapter exists), gated or not.
    if is_full_weight:
        deploy_mode = "merged"
        weights_subdir = "merged"
    else:
        deploy_mode = "adapter" if model.gated else "merged"
        weights_subdir = "adapter" if model.gated else "merged"
    requires_license_acceptance = bool(model.gated and is_full_weight)

    # Resolve the model's image TIER → the LLaMA-Factory version tag it trained
    # on, so the deploy image is `hiyouga/llamafactory:<tag>` (exact versions).
    # Serverless trains on AWS's managed recipe image but its merged HF model
    # loads on our LLaMA-Factory image (the eval bridge already proves this), so
    # the deploy image is resolved the SAME way for both engines.
    from .aws_config import image_tiers

    base_image_tag = image_tiers().get(model.image_tag, model.image_tag)

    engine = (entry.hp or {}).get("engine", "llama_factory")
    cfg = load_aws_config()

    if engine == "sagemaker_serverless":
        # Uncompressed prefix of loose files; weights under checkpoints/hf_merged
        # (ungated) or checkpoints/hf (gated adapter). No tarball to extract.
        # (Serverless is LoRA-only, so a full-weight run never reaches here.)
        subdir = _SERVERLESS_ADAPTER_SUBDIR if model.gated else _SERVERLESS_MERGED_SUBDIR
        artifact_s3_uri = _serverless_artifact_prefix(cfg, entry.train_job, subdir)
        engine_out = engine
    elif model.gated and not is_full_weight:
        # LLaMA-Factory GATED model → ADAPTER deploy. After the checkpoint-unify
        # change a job writes its adapter to the synced checkpoint prefix's latest
        # checkpoint-N/ dir (output_dir=CHECKPOINT_DIR), NOT into model.tar.gz. So
        # we prefer that prefix and deliver it as loose files. BUT entries trained
        # BEFORE the unify (or whose checkpoint never synced) have NO checkpoint-N/
        # — for those we must NOT hard-fail: fall back to the legacy model.tar.gz
        # adapter/ path so historical gated winners still export.
        checkpoint_s3 = getattr(entry, "checkpoint_s3", None)
        ckpt_prefix = _latest_checkpoint_prefix(cfg, checkpoint_s3) if checkpoint_s3 else None
        if ckpt_prefix:
            artifact_s3_uri = ckpt_prefix
            # Loose-file delivery (weightsFiles + deploy.sh download-loose-files),
            # filtered to adapter-only so optimizer/scheduler/RNG state isn't shipped.
            engine_out = "sagemaker_serverless_adapter"
        else:
            # Legacy/no-checkpoint path: the adapter is still inside model.tar.gz.
            artifact_s3_uri = _artifact_uri(cfg, entry.train_job)
            engine_out = engine  # llama_factory tarball delivery (deploy.sh picks adapter/)
    else:
        # LLaMA-Factory merged standalone, in model.tar.gz. Covers UNGATED (any
        # method) AND gated FULL-WEIGHT (full/freeze: no adapter → merged is the
        # only artifact; the license gate, not adapter-only, handles the licensing).
        artifact_s3_uri = _artifact_uri(cfg, entry.train_job)
        engine_out = engine

    return ExportManifest(
        race_id=race_id,
        model_id=entry.model_id,
        model_display=entry.model_display or model.display_name,
        hf_base_model=model.hf_model_id,
        template=model.template,
        gated=model.gated,
        deploy_mode=deploy_mode,
        weights_subdir=weights_subdir,
        suggested_instance=entry.instance_type or model.suggested_instance,
        train_job=entry.train_job,
        artifact_s3_uri=artifact_s3_uri,
        artifact_region=cfg.region,
        base_image_tag=base_image_tag,
        engine=engine_out,
        requires_license_acceptance=requires_license_acceptance,
        trust_remote_code=model.trust_remote_code,
    )


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """('s3://bucket/key…') → ('bucket', 'key…'). Raises on a malformed URI."""
    if not uri.startswith("s3://"):
        raise ExportError(f"not an S3 URI: {uri}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not bucket or not key:
        raise ExportError(f"malformed S3 URI: {uri}")
    return bucket, key


def _s3_presign_client(cfg: AwsConfig):
    """An S3 client pinned to SigV4 + regional virtual addressing. The data bucket
    is KMS-encrypted (SSE-KMS), and S3 REQUIRES SigV4 for presigned GETs of KMS
    objects — boto3's default signer would emit SigV2 and the download 400s with
    "require AWS Signature Version 4". Shared by the tarball + prefix presigners."""
    _, boto_sess = _session(cfg)
    from botocore.config import Config as BotoConfig

    from .aws_clients import botocore_config

    return boto_sess.client(
        "s3",
        region_name=cfg.region,
        config=botocore_config(
            BotoConfig(signature_version="s3v4", s3={"addressing_style": "virtual"})
        ),
    )


def presign_artifact(artifact_s3_uri: str, ttl: int = PRESIGN_TTL_SECONDS) -> str:
    """Time-limited GET URL for the weights tarball (LLaMA-Factory: one object),
    so the user's account can download it without any cross-account IAM setup."""
    bucket, key = _split_s3_uri(artifact_s3_uri)
    s3 = _s3_presign_client(load_aws_config())
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl,
    )


# Files an ADAPTER deploy actually needs from a raw HF-Trainer checkpoint-N/ dir.
# A checkpoint also holds optimizer.pt / scheduler.pt / rng_state.pth /
# trainer_state.json / training_args.bin (full resumable state, because we set
# save_only_model=False) — those are useless at inference and can dwarf the LoRA
# adapter, so we drop them from the gated-adapter loose-file delivery.
_ADAPTER_DEPLOY_FILES = frozenset({
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "config.json", "generation_config.json",
})


def presign_prefix(prefix_s3_uri: str, ttl: int = PRESIGN_TTL_SECONDS,
                   adapter_only: bool = False) -> list[dict[str, Any]]:
    """Presign EVERY object under a prefix — for the serverless engine, whose
    output is an uncompressed directory of loose files (no tarball). Returns a
    list of {name, url, size} where `name` is the object's path RELATIVE to the
    prefix (so deploy.sh recreates the dir). Skips "directory marker" keys (those
    ending in "/"). Raises if the prefix is empty (nothing to deploy).

    When `adapter_only` is set (gated deploy from a raw HF-Trainer checkpoint-N/
    dir), only adapter + tokenizer files are presigned — the optimizer/scheduler/
    RNG/trainer-state files are dropped so the deploy bundle isn't bloated with
    training-only state the inference container never loads."""
    bucket, prefix = _split_s3_uri(prefix_s3_uri)
    prefix = prefix.rstrip("/") + "/"  # ensure a trailing slash for clean relpaths
    cfg = load_aws_config()
    s3 = _s3_presign_client(cfg)

    files: list[dict[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # S3 console "folder" marker — not a real file
                continue
            rel = key[len(prefix):]
            if not rel:
                continue
            if adapter_only and rel.split("/")[-1] not in _ADAPTER_DEPLOY_FILES:
                continue  # skip optimizer.pt / scheduler.pt / rng_state.pth / etc.
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=ttl,
            )
            files.append({"name": rel, "url": url, "size": obj.get("Size", 0)})
    if not files:
        raise ExportError(f"no weight files found under {prefix_s3_uri}")
    return files


def export_info(race_id: str, model_id: str, license_accepted: bool = False) -> dict[str, Any]:
    """Manifest + presigned weights — the full payload the export endpoint returns.
    Presigning is done LAST so a manifest-only error (bad race/model) doesn't make
    an S3 call.

    LLaMA-Factory → one `weightsUrl` (the tarball). Serverless → `weightsFiles`,
    a list of per-file presigned URLs (the uncompressed prefix). deploy.sh keys off
    the `engine`/presence of `weightsFiles` to choose download-and-extract vs
    download-loose-files.

    LICENSE GATE (Option B): when the artifact is MERGED weights of a GATED base (a
    gated full/freeze fine-tune — `requires_license_acceptance`), the presigned
    weights are WITHHELD until `license_accepted` is True. The manifest still
    returns (so the UI can show the model + render the license prompt), but no
    download URL is minted — distributing merged gated weights requires the user to
    accept the base model's license first."""
    manifest = build_manifest(race_id, model_id)
    out = manifest.to_dict()
    if manifest.requires_license_acceptance and not license_accepted:
        # Gate: surface the requirement + base model, mint NO presigned URL.
        out["licenseRequired"] = True
        out["licenseModel"] = manifest.hf_base_model
        return out
    # The gated-LF-from-checkpoint path tags engine "sagemaker_serverless_adapter":
    # it delivers loose files like serverless BUT from a raw HF-Trainer checkpoint
    # dir, so it must filter to adapter-only files (drop optimizer/scheduler state).
    if manifest.engine in ("sagemaker_serverless", "sagemaker_serverless_adapter"):
        adapter_only = manifest.engine == "sagemaker_serverless_adapter"
        out["weightsFiles"] = presign_prefix(manifest.artifact_s3_uri, adapter_only=adapter_only)
    else:
        out["weightsUrl"] = presign_artifact(manifest.artifact_s3_uri)
    out["weightsTtlSeconds"] = PRESIGN_TTL_SECONDS
    return out
