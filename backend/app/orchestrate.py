# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""SageMaker training orchestration — deterministic, no agents.

Flow (per the architecture):
  1. Render train.yaml + export.yaml for (model, hyperparams, split).
  2. Upload the persisted split (train.jsonl, eval.jsonl, dataset_info.json) and
     the two YAMLs to S3 under a per-job prefix.
  3. Launch a SageMaker training job with the frozen LLaMA-Factory image and two
     input channels (dataset, config), wait=False.
  4. Status is polled separately via describe_training_job.

The generic container entrypoint (/usr/local/bin/train) runs
`llamafactory-cli train` then `export`. No per-model code here either — the
model is just a manifest entry + rendered YAML.

Costs money: only `launch_training_job` creates a billable job; it's called
solely from the explicit UI launch action.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import json
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

from .aws_clients import get_session
from .aws_config import AwsConfig, load_aws_config, reset_cutoff
from .catalog import DecodingParams, Hyperparams, ModelSpec, get_model
from .render import eval_env, render_all
from .secrets import get_hf_token
from .storage import split_dir

# SageMaker container dir conventions (mirror entrypoint.sh).
CONFIG_CHANNEL = "config"
DATASET_CHANNEL = "dataset"


def _job_tags() -> list[dict[str, str]]:
    """Tags applied to EVERY SageMaker training job we submit (train, fine-tuned
    eval, base eval): an `owner` tag naming the launching tenant, for cost
    attribution and so a user's jobs are identifiable in the SageMaker console /
    Cost Explorer. Default tenant → no owner tag (keeps pre-tenancy jobs' tag set
    unchanged). SageMaker SDK v2 takes tags as a list of {Key, Value}."""
    from .tenancy import DEFAULT_TENANT, current_tenant

    tenant = current_tenant()
    if tenant == DEFAULT_TENANT:
        return []
    # SageMaker tag values: <=256 chars, [\w\s+=.:/@-]. A Cognito sub is safe.
    return [{"Key": "owner", "Value": tenant[:256]}]


# S3 layout for a job's inputs + outputs. Per-tenant so one user's job data/
# artifacts live under their own prefix (users/<sub>/jobs/…), mirroring the state
# isolation in store.py. Consumers (eval, export, judge) resolve artifacts from
# describe_training_job (the recorded S3 path), so they are path-agnostic and need
# no change — and EXISTING jobs keep working since their paths are already baked
# into their SageMaker records. Default tenant → the historical slm-platform/jobs/.
def _jobs_key_prefix() -> str:
    """The S3 KEY prefix (no bucket) for this tenant's jobs."""
    from .tenancy import DEFAULT_TENANT, current_tenant

    tenant = current_tenant()
    if tenant == DEFAULT_TENANT:
        return "slm-platform/jobs"
    return f"slm-platform/users/{tenant}/jobs"


def _jobs_s3_base(cfg: AwsConfig) -> str:
    """The s3:// base URI for this tenant's jobs."""
    return f"s3://{cfg.bucket}/{_jobs_key_prefix()}"

# Regex scrapers SageMaker applies to the training job's stdout to publish
# CloudWatch metrics — this is how we get a loss curve with NO image rebuild
# (the HF Trainer underneath LLaMA-Factory already prints these log dicts, e.g.
#   {'loss': 0.6577, 'grad_norm': 1.2, 'learning_rate': 4.7e-05, 'epoch': 0.92}
#   {'eval_loss': 0.5123, 'eval_runtime': 3.4, 'epoch': 1.0}
# ). Each definition captures ONE numeric value; SageMaker timestamps each hit,
# giving us a time series we read back via CloudWatch get_metric_data. `epoch`
# rides along so the chart can label progress. Names become CloudWatch metrics
# under the job's namespace (/aws/sagemaker/TrainingJobs, dim TrainingJobName).
TRAINING_METRIC_DEFINITIONS: list[dict[str, str]] = [
    {"Name": "train:loss", "Regex": r"'loss':\s*([0-9.]+)"},
    {"Name": "eval:loss", "Regex": r"'eval_loss':\s*([0-9.]+)"},
    {"Name": "train:learning_rate", "Regex": r"'learning_rate':\s*([0-9.eE+-]+)"},
    {"Name": "train:epoch", "Regex": r"'epoch':\s*([0-9.]+)"},
    # grad_norm is logged on the same train lines; spots instability/spikes.
    {"Name": "train:grad_norm", "Regex": r"'grad_norm':\s*([0-9.eE+-]+)"},
]

# CloudWatch metric name → the friendly key the UI charts on.
_CURVE_METRICS = {
    "train:loss": "trainLoss",
    "eval:loss": "evalLoss",
    "train:learning_rate": "learningRate",
    "train:epoch": "epoch",
    "train:grad_norm": "gradNorm",
}


def _session(cfg: AwsConfig) -> tuple[sagemaker.Session, boto3.Session]:
    # profile=None → use ambient credentials (Lambda execution role, instance
    # profile, or env). A named profile is only for the local dev-box workflow.
    boto_sess = get_session(profile_name=cfg.profile or None, region_name=cfg.region)
    sm_sess = sagemaker.Session(boto_session=boto_sess)
    return sm_sess, boto_sess


def _job_name(model_id: str, split_id: str, stamp: str, method: str = "lora") -> str:
    # SageMaker job names: alphanumeric + hyphens, <= 63 chars.
    #
    # The method tag goes in the model-id HEAD (before the split id) so the
    # leaderboard's label heuristic — which keeps parts until the 12-hex split id
    # (see leaderboard._model_label) — surfaces it: a QLoRA run of qwen3-4b labels
    # as `qwen3-4b-qlora`, a DISTINCT leaderboard row from the LoRA `qwen3-4b`.
    # The default LoRA path is left UNTAGGED so its job names stay byte-identical
    # to every run launched before this change (no silent leaderboard churn).
    head = model_id if method == "lora" else f"{model_id}-{method}"
    base = f"slm-{head}-{split_id}-{stamp}"
    base = re.sub(r"[^a-zA-Z0-9-]", "-", base)
    return base[:63].strip("-")


def _job_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Base job env + the HF token (if stored) so gated models can be pulled.

    The HF libraries read HF_TOKEN automatically; injecting it here means the
    frozen image needs no change. No token → gated pulls just fail (and gated
    models are disabled in the UI until a token is set)."""
    env = {"PYTHONUNBUFFERED": "1"}
    token = get_hf_token()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token  # legacy var some versions still read
    if extra:
        env.update(extra)
    return env


def resolve_image_uri(cfg: AwsConfig, model: ModelSpec) -> str:
    """The ECR image a model trains/evals on — resolved from its image TIER.

    This is the per-model image selection at the heart of the multi-image design:
    old models stay on `stable` (0.9.4), models that need a newer stack declare
    `latest` (0.9.5). `stable` resolves to the same URI as cfg.image_uri, so every
    existing model is byte-identical to before this change."""
    return cfg.image_uri_for_tier(getattr(model, "image_tag", None))


def build_estimator(
    cfg: AwsConfig,
    sm_sess: sagemaker.Session,
    model: ModelSpec,
    job_name: str,
    instance_type: str,
    max_run_seconds: int,
    output_s3: str,
    use_spot: bool = False,
    checkpoint_s3: str | None = None,
    image_uri: str | None = None,
) -> Estimator:
    """Construct (but do not launch) the SageMaker Estimator.

    `image_uri` is the model's resolved per-tier image (defaults to the model's
    tier when not passed).

    Checkpointing is now UNCONDITIONAL: every job (spot AND on-demand) gets a
    checkpoint dir synced to checkpoint_s3, so a failed on-demand run can be
    resubmitted and resume from the last checkpoint rather than retraining from
    scratch — not just spot reclaims. When use_spot is set, the job ALSO runs on
    managed spot capacity (cheaper, interruptible) and SageMaker auto-restores the
    checkpoint dir before each retry; max_wait (>= max_run) bounds the total wait
    incl. interruptions."""
    extra_kwargs: dict[str, Any] = {}
    # checkpoint_s3 is supplied by every engine launch now; guard defensively so a
    # caller that passes None (e.g. an eval-only estimator) doesn't break.
    if checkpoint_s3:
        extra_kwargs["checkpoint_s3_uri"] = checkpoint_s3
        extra_kwargs["checkpoint_local_path"] = "/opt/ml/checkpoints"
    if use_spot:
        extra_kwargs["use_spot_instances"] = True
        # max_wait bounds queue + interruption wait; must be >= max_run.
        extra_kwargs["max_wait"] = max(max_run_seconds * 2, max_run_seconds + 3600)
    return Estimator(
        image_uri=image_uri or resolve_image_uri(cfg, model),
        role=cfg.role_arn,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_s3,
        base_job_name=job_name,
        sagemaker_session=sm_sess,
        max_run=max_run_seconds,
        # The frozen image already carries everything; no hyperparameters passed
        # to the container — config travels as YAML on the `config` channel.
        environment=_job_environment(),
        # Scrape loss/lr/epoch from stdout into CloudWatch so the UI can draw a
        # live training curve. No image change — the Trainer already logs these.
        metric_definitions=TRAINING_METRIC_DEFINITIONS,
        tags=_job_tags(),
        **extra_kwargs,
    )


def upload_job_inputs(
    sm_sess: sagemaker.Session,
    cfg: AwsConfig,
    model: ModelSpec,
    hp: Hyperparams,
    split_id: str,
    job_name: str,
    use_spot: bool = False,
) -> dict[str, str]:
    """Render YAML + upload dataset and config to S3. Returns channel S3 URIs."""
    run_dir = split_dir(split_id)
    if run_dir is None:
        raise ValueError(f"split {split_id} not found on disk")

    # Render the YAMLs and drop them into the split dir alongside the data, then
    # upload the dataset + config channels. use_spot redirects the trainer output to
    # the checkpoint dir for resumability. render_all OMITS exportYaml for
    # full-weight (full/freeze) methods — they produce standalone weights with no
    # adapter to merge — so the export.yaml is conditional. Its ABSENCE on the
    # config channel is the signal the container entrypoint uses to skip the merge.
    yamls = render_all(model, hp, split_id, use_spot=use_spot)
    (run_dir / "train.yaml").write_text(yamls["trainYaml"], encoding="utf-8")
    config_files = [run_dir / "train.yaml"]
    export_yaml = yamls.get("exportYaml")
    export_path = run_dir / "export.yaml"
    if export_yaml is not None:
        export_path.write_text(export_yaml, encoding="utf-8")
        config_files.append(export_path)
    else:
        # Stale export.yaml from a previous (adapter) run in the same split dir would
        # wrongly re-trigger a merge — remove it so the config channel is clean.
        export_path.unlink(missing_ok=True)

    key_base = f"{_jobs_key_prefix()}/{job_name}"
    base = f"{_jobs_s3_base(cfg)}/{job_name}"

    # Dataset channel: train.jsonl, eval.jsonl, dataset_info.json (+ val.jsonl if
    # this dataset carries a validation split, which the train YAML references).
    dataset_files = [run_dir / "train.jsonl", run_dir / "eval.jsonl", run_dir / "dataset_info.json"]
    val_file = run_dir / "val.jsonl"
    if val_file.exists():
        dataset_files.append(val_file)
    dataset_prefix = _upload_files(
        sm_sess, cfg, dataset_files, f"{key_base}/dataset",
    )
    # Config channel: train.yaml (+ export.yaml only for adapter methods).
    config_prefix = _upload_files(
        sm_sess, cfg, config_files,
        f"{key_base}/config",
    )
    return {"dataset": dataset_prefix, "config": config_prefix, "base": base}


def _upload_files(sm_sess: sagemaker.Session, cfg: AwsConfig, files: list[Path], key_prefix: str) -> str:
    s3 = sm_sess.boto_session.client("s3")
    for f in files:
        s3.upload_file(str(f), cfg.bucket, f"{key_prefix}/{f.name}")
    return f"s3://{cfg.bucket}/{key_prefix}"


def launch_training_job(
    model_id: str,
    split_id: str,
    hp: Hyperparams,
    instance_type: str,
    stamp: str,
    max_run_seconds: int = 3600,
    use_spot: bool = False,
    image_tag: str | None = None,
    resume_checkpoint_s3: str | None = None,
) -> dict[str, Any]:
    """Render, upload, and LAUNCH a SageMaker training job (wait=False).

    `stamp` must be supplied by the caller (no time/RNG in library code).
    `use_spot` runs on managed spot with checkpoint/resume (cheaper, slower,
    interruptible). `image_tag` overrides the model's catalog tier — used by the
    catalog "verify on image X" / re-test action to smoke-test a model on a tier
    it isn't pinned to yet (e.g. proving phi-4-mini on `latest` before flipping
    it). Defaults to the model's own tier.

    `resume_checkpoint_s3` (LLaMA-Factory engine only): point the new job's
    checkpoint dir at a PRIOR failed job's checkpoint S3 prefix so SageMaker
    restores it and training resumes from the last step instead of from scratch.

    Returns the job name + S3 locations.
    """
    model = get_model(model_id)
    if model is None:
        raise ValueError(f"unknown model id: {model_id}")

    # Dispatch to the run's training ENGINE (a per-run Hyperparams choice; falls
    # back to the model's default). The default ("llama_factory") wraps the
    # original launch logic verbatim — same S3 keys, YAML, job name, tags, and
    # return dict — so existing runs are byte-identical. "sagemaker_serverless"
    # routes to the managed serverless engine, gated behind a feature flag in
    # get_engine.
    from .engines import get_engine

    engine_name = getattr(hp, "engine", None) or getattr(model, "engine", None)
    engine = get_engine(engine_name)
    kwargs: dict[str, Any] = dict(
        model=model,
        split_id=split_id,
        hp=hp,
        instance_type=instance_type,
        stamp=stamp,
        max_run_seconds=max_run_seconds,
        use_spot=use_spot,
        image_tag=image_tag,
    )
    # Resume is a LLaMA-Factory capability (checkpoint-prefix reuse); the serverless
    # engine manages its own capacity and ignores it. Only pass it to an engine
    # whose launch accepts it, so the serverless signature stays clean.
    if resume_checkpoint_s3 and engine_name in (None, "llama_factory"):
        kwargs["resume_checkpoint_s3"] = resume_checkpoint_s3
    return engine.launch_training_job(**kwargs)


def preflight() -> dict[str, Any]:
    """Verify the AWS environment is usable BEFORE any (billable) launch.

    Read-only checks: caller identity (creds valid), the S3 bucket is reachable,
    the SageMaker execution role exists, and the training image is in ECR. Returns
    one entry per check with ok/detail so the Settings page can show a clear
    green/red panel instead of a cryptic 502 mid-launch.
    """
    cfg = load_aws_config()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    def fail(name: str, detail: str, exc: Exception) -> None:
        """Record a failed check WITHOUT the raw exception text.

        The resource NAMES in these messages are fine to return — the caller is the
        deployment's own operator and `_public_config` echoes bucket/role/image back
        anyway. The botocore message body is not: on an AccessDenied it spells out the
        execution-role ARN that was denied, and it carries SDK internals besides.
        What the operator actually needs to act is the error CODE (AccessDenied,
        NoSuchBucket, 404), so return that and log the rest.
        """
        from .obs import log_event

        code = ""
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str(response.get("Error", {}).get("Code") or "")
        log_event("config.preflight.check_failed", level="WARNING",
                  check=name, error=f"{type(exc).__name__}: {exc}")
        add(name, False, f"{detail} ({code or type(exc).__name__})")

    try:
        _, boto_sess = _session(cfg)
    except Exception as e:  # noqa: BLE001
        fail("credentials", "could not create an AWS session", e)
        return {"ok": False, "config": _public_config(cfg), "checks": checks}

    # 1. Caller identity (are credentials valid + which account?)
    try:
        ident = boto_sess.client("sts").get_caller_identity()
        acct = ident["Account"]
        add("credentials", True, f"authenticated as account {acct}")
        if acct != cfg.account_id:
            add("account_match", False,
                f"configured account {cfg.account_id} != credentials account {acct}")
        else:
            add("account_match", True, f"account {acct} matches config")
    except Exception as e:  # noqa: BLE001
        fail("credentials", "STS get-caller-identity failed", e)
        return {"ok": False, "config": _public_config(cfg), "checks": checks}

    # 2. S3 bucket reachable
    try:
        boto_sess.client("s3").head_bucket(Bucket=cfg.bucket)
        add("s3_bucket", True, f"bucket {cfg.bucket} is reachable")
    except Exception as e:  # noqa: BLE001
        fail("s3_bucket", f"cannot access bucket {cfg.bucket}", e)

    # 3. SageMaker execution role exists
    try:
        role_name = cfg.role_arn.split("/")[-1]
        boto_sess.client("iam").get_role(RoleName=role_name)
        add("execution_role", True, f"role {role_name} exists")
    except Exception as e:  # noqa: BLE001
        fail("execution_role", f"cannot find role {cfg.role_arn}", e)

    # 4. Training image present in ECR
    try:
        repo = cfg.image_uri.split("/")[-1].split(":")[0]
        tag = cfg.image_uri.split(":")[-1]
        ecr = boto_sess.client("ecr")
        ecr.describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])
        add("training_image", True, f"image {repo}:{tag} found in ECR")
    except Exception as e:  # noqa: BLE001
        fail("training_image", f"training image not found ({cfg.image_uri})", e)

    return {
        "ok": all(c["ok"] for c in checks),
        "config": _public_config(cfg),
        "checks": checks,
    }


def _public_config(cfg: AwsConfig) -> dict[str, Any]:
    """Config values safe to show in the UI (no secrets — these are ARNs/ids)."""
    return {
        "region": cfg.region,
        "account": cfg.account_id,
        "bucket": cfg.bucket,
        "roleArn": cfg.role_arn,
        "imageUri": cfg.image_uri,
        "profile": cfg.profile,
    }


def job_log_tail(job_name: str, limit: int = 40) -> str:
    """Last few CloudWatch log lines for a training job, or "" if unavailable.
    SageMaker's FailureReason is often a bare "AlgorithmError: exit code 1" — the
    real cause (e.g. a gated-repo 403) lives only in the logs, so verification
    classification reads this to tell access-denied from true incompatibility."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    try:
        logs = boto_sess.client("logs")
        grp = "/aws/sagemaker/TrainingJobs"
        streams = logs.describe_log_streams(
            logGroupName=grp, logStreamNamePrefix=job_name,
            orderBy="LastEventTime", descending=True, limit=1,
        ).get("logStreams", [])
        if not streams:
            return ""
        ev = logs.get_log_events(
            logGroupName=grp, logStreamName=streams[0]["logStreamName"],
            startFromHead=False, limit=limit,
        ).get("events", [])
        return "\n".join(e["message"].rstrip() for e in ev)[-6000:]
    except Exception:  # noqa: BLE001 — logs are best-effort
        return ""


def describe_job(job_name: str) -> dict[str, Any]:
    """Poll a training job's status (read-only)."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    d = sm.describe_training_job(TrainingJobName=job_name)

    billable = d.get("BillableTimeInSeconds")
    # Serverless customization jobs have NO ResourceConfig (managed capacity) —
    # use .get() so describe_job (called on EVERY job by the race state machine)
    # doesn't KeyError on a serverless entry. Reports "serverless" for those.
    instance = (d.get("ResourceConfig") or {}).get("InstanceType") or (
        "serverless" if d.get("ServerlessJobConfig") is not None else None
    )
    return {
        "jobName": job_name,
        "status": d["TrainingJobStatus"],  # InProgress | Completed | Failed | Stopped
        "secondaryStatus": d.get("SecondaryStatus"),
        "failureReason": d.get("FailureReason"),
        "instanceType": instance,
        "billableTimeSeconds": billable,
        # CreationTime = when the job was created (the capacity-wait clock starts
        # here). TrainingStartTime is null until the container actually starts —
        # so for a spot job stuck waiting on capacity, CreationTime is the only
        # elapsed reference (used by the spot→on-demand fallback).
        "creationTime": str(d.get("CreationTime")) if d.get("CreationTime") else None,
        "trainingStartTime": str(d.get("TrainingStartTime")) if d.get("TrainingStartTime") else None,
        "trainingEndTime": str(d.get("TrainingEndTime")) if d.get("TrainingEndTime") else None,
        "modelArtifacts": d.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
        "region": cfg.region,
    }


def stop_training_job(job_name: str) -> bool:
    """Stop a running SageMaker training job (best-effort, idempotent). Returns
    True if the stop was issued, False if the job was already terminal / gone
    (ValidationException) — so callers can stop-and-relaunch without racing the
    job's own completion. Never raises on the already-stopped case."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    try:
        sm.stop_training_job(TrainingJobName=job_name)
        return True
    except Exception as e:  # noqa: BLE001 — already terminal/absent → nothing to stop
        from .obs import log_event

        log_event("train.stop.noop", level="INFO", jobName=job_name, detail=str(e))
        return False


def fetch_training_curves(job_name: str) -> dict[str, Any]:
    """Read a training job's loss/lr/epoch curves.

    Prefers a durable SNAPSHOT (written when the job finished — see
    snapshot_curves_if_terminal) so finished jobs keep their full-resolution
    curve forever, immune to CloudWatch down-sampling/expiry. Falls back to a
    live CloudWatch query for in-progress jobs (or any job not yet snapshotted).

    SageMaker scrapes `TRAINING_METRIC_DEFINITIONS` from the job's stdout into
    metrics under namespace /aws/sagemaker/TrainingJobs (dimension
    TrainingJobName). We pull each as a time series via get_metric_data and
    convert absolute timestamps to minutes-since-training-start so the X axis is
    a clean elapsed-time axis the UI can plot directly.

    Returns {"jobName", "status", "startTime", series: {trainLoss:[{x,y}], ...}}.
    Series are empty (not an error) when a job is too new to have logged yet, or
    predates this feature (older jobs were launched without metric_definitions).
    """
    # Durable snapshot wins if present (finished job → permanent curve).
    from .storage import load_curves

    snap = load_curves(job_name)
    if snap is not None:
        return snap

    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    d = sm.describe_training_job(TrainingJobName=job_name)
    status = d["TrainingJobStatus"]
    start = d.get("TrainingStartTime")
    end = d.get("TrainingEndTime")

    series: dict[str, list[dict[str, float]]] = {v: [] for v in _CURVE_METRICS.values()}
    if start is None:
        # Job hasn't begun training (still provisioning) → no metrics yet.
        return {
            "jobName": job_name,
            "status": status,
            "startTime": None,
            "series": series,
        }

    # CloudWatch window: from just before training started to a bit after it
    # ended (or now, for in-progress jobs). Pad so boundary points aren't lost.
    cw = boto_sess.client("cloudwatch")
    win_start = start - timedelta(minutes=2)
    win_end = (end or _now_utc(boto_sess)) + timedelta(minutes=2)

    queries = []
    for i, (cw_name, key) in enumerate(_CURVE_METRICS.items()):
        queries.append(
            {
                "Id": f"m{i}",
                "Label": key,
                "MetricStat": {
                    "Metric": {
                        "Namespace": "/aws/sagemaker/TrainingJobs",
                        "MetricName": cw_name,
                        "Dimensions": [{"Name": "TrainingJobName", "Value": job_name}],
                    },
                    # 60s is the finest period CloudWatch keeps for recent data;
                    # Average collapses multiple log lines in the same minute.
                    "Period": 60,
                    "Stat": "Average",
                },
                "ReturnData": True,
            }
        )

    resp = cw.get_metric_data(
        MetricDataQueries=queries,
        StartTime=win_start,
        EndTime=win_end,
        ScanBy="TimestampAscending",
    )

    start_ts = start.timestamp()
    for res in resp.get("MetricDataResults", []):
        key = res.get("Label")
        if key not in series:
            continue
        for ts, val in zip(res.get("Timestamps", []), res.get("Values", [])):
            # Minutes elapsed since training started — clean X axis for the chart.
            minutes = (ts.timestamp() - start_ts) / 60.0
            series[key].append({"x": round(minutes, 3), "y": val})
        series[key].sort(key=lambda p: p["x"])

    return {
        "jobName": job_name,
        "status": status,
        "startTime": str(start),
        "series": series,
    }


def _now_utc(boto_sess: boto3.Session):
    """Timezone-aware UTC now for the CloudWatch read window of an in-progress
    job. The no-time-in-lib-code rule is about deterministic outputs (job names,
    ids); this is just an I/O query bound, so a plain now() is fine."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def snapshot_curves_if_terminal(job_name: str) -> bool:
    """If a training job is finished (Completed/Failed/Stopped) and not yet
    snapshotted, fetch its curves from CloudWatch once and persist them durably.

    Idempotent + best-effort: called from the race reconcile loop when an entry
    leaves the TRAINING state. Returns True if it wrote a snapshot this call.
    Safe to call repeatedly — does nothing once a snapshot exists.
    """
    from .storage import has_curves, persist_curves

    if has_curves(job_name):
        return False
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    try:
        status = sm.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"]
    except Exception:  # noqa: BLE001 — job may not be describable; skip
        return False
    if status not in ("Completed", "Failed", "Stopped"):
        return False  # still running — its curve isn't final yet
    curves = fetch_training_curves(job_name)  # CloudWatch (no snapshot exists yet)
    persist_curves(job_name, curves)
    return True


# --- Offline batch eval ---


def job_after_cutoff(summary: dict[str, Any]) -> bool:
    """True if a SageMaker job summary is newer than the reset cutoff (or no
    cutoff is set). Used to hide pre-reset jobs from all UI listings."""
    cutoff = reset_cutoff()
    if not cutoff:
        return True
    ct = summary.get("CreationTime")
    return ct is None or ct.isoformat() > cutoff


def list_completed_jobs(max_results: int = 20) -> list[dict[str, Any]]:
    """List recent COMPLETED slm-* training jobs that produced a model artifact.

    These are the candidates evalable on the Eval page. We filter to our naming
    prefix and require a model.tar.gz (so train jobs, not prior eval jobs).
    """
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    resp = sm.list_training_jobs(
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=max_results,
    )
    out: list[dict[str, Any]] = []
    for s in resp.get("TrainingJobSummaries", []):
        name = s["TrainingJobName"]
        if not name.startswith("slm-") or name.startswith("slm-eval-"):
            continue
        if not job_after_cutoff(s):
            continue
        d = sm.describe_training_job(TrainingJobName=name)
        art = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
        if not art:
            continue
        out.append(
            {
                "jobName": name,
                "modelArtifacts": art,
                "instanceType": (d.get("ResourceConfig") or {}).get("InstanceType", "serverless"),
                "creationTime": str(s.get("CreationTime")),
            }
        )
    return out


def launch_eval_job(
    source_job_name: str,
    split_id: str,
    decoding: DecodingParams,
    stamp: str,
    instance_type: str = "ml.g5.2xlarge",
    max_run_seconds: int = 3600,
    engine: str | None = None,
) -> dict[str, Any]:
    """Launch an eval job: load a prior job's merged model, score the held-out set.

    The model channel points at the source job's model.tar.gz (SageMaker extracts
    it into /opt/ml/input/data/model). The dataset channel carries eval.jsonl.

    `engine` selects how the source model is resolved. Default/None/"llama_factory"
    is the original behavior (artifact root + the source job's own training image).
    "sagemaker_serverless" points the model channel at the `checkpoints/hf_merged/`
    subdir of the (uncompressed) artifact prefix and runs eval on OUR eval image
    (the source job's container is AWS's managed recipe image, which can't run our
    eval.py).
    """
    cfg = load_aws_config()
    sm_sess, boto_sess = _session(cfg)

    # Resolve the source model artifact — but only if training actually COMPLETED.
    # Guards against evaluating a model whose training is still running or failed
    # (there's no usable artifact yet); the UI also hides such models.
    sm = boto_sess.client("sagemaker")
    src = sm.describe_training_job(TrainingJobName=source_job_name)
    src_status = src.get("TrainingJobStatus")
    if src_status != "Completed":
        raise ValueError(
            f"source training job {source_job_name} is {src_status}, not Completed — "
            "cannot evaluate a model whose training hasn't finished"
        )
    model_artifact = src.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not model_artifact:
        raise ValueError(f"source job {source_job_name} has no model artifact")

    is_serverless = (engine == "sagemaker_serverless")
    if is_serverless:
        # Serverless output is an UNCOMPRESSED S3 prefix; the loadable merged HF
        # model lives in checkpoints/hf_merged/. Point the model channel THERE so
        # only those files land at the channel root (the artifact root also holds
        # multi-GB FSDP checkpoints we don't want to download). Trailing slash so
        # the S3Prefix grabs the directory contents.
        base = model_artifact.rstrip("/")
        model_channel_uri = f"{base}/checkpoints/hf_merged/"
        # Serverless trains on AWS's managed recipe container, which is NOT our
        # eval image — eval MUST run on OUR image (resolve from default tier).
        eval_image_uri = cfg.image_uri
    else:
        model_channel_uri = model_artifact
        # Eval MUST run on the same image the model was trained/merged on — the merged
        # safetensors + tokenizer match that exact transformers/stack, so a mismatched
        # image could fail to load them. Read the training job's own image rather than
        # the global default, so per-model image tiers carry through to eval.
        eval_image_uri = src.get("AlgorithmSpecification", {}).get("TrainingImage") or cfg.image_uri

    run_dir = split_dir(split_id)
    if run_dir is None:
        raise ValueError(f"split {split_id} not found on disk")

    # Prefix with slm-eval- so list_completed_jobs excludes eval jobs as candidates.
    job_name = re.sub(r"[^a-zA-Z0-9-]", "-", f"slm-eval-{split_id}-{stamp}")[:63].strip("-")

    # Upload the eval dataset (eval.jsonl + dataset_info.json) for this job.
    dataset_prefix = _upload_files(
        sm_sess, cfg,
        [run_dir / "eval.jsonl", run_dir / "dataset_info.json"],
        f"{_jobs_key_prefix()}/{job_name}/dataset",
    )
    output_s3 = f"{_jobs_s3_base(cfg)}/{job_name}/output"

    est = Estimator(
        image_uri=eval_image_uri,  # match the source training job's image tier
        role=cfg.role_arn,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_s3,
        base_job_name=job_name,
        sagemaker_session=sm_sess,
        max_run=max_run_seconds,
        environment=_job_environment(eval_env(decoding)),  # SLM_MODE=eval + decoding + HF token
        tags=_job_tags(),
    )
    est.fit(
        inputs={
            "model": TrainingInput(model_channel_uri, input_mode="File"),
            DATASET_CHANNEL: TrainingInput(dataset_prefix, input_mode="File"),
        },
        wait=False,
    )
    return {
        "jobName": est.latest_training_job.name,
        "sourceJob": source_job_name,
        "sourceModel": model_channel_uri,
        "instanceType": instance_type,
        "imageUri": eval_image_uri,
        "datasetS3": dataset_prefix,
        "outputS3": output_s3,
        "region": cfg.region,
    }


def launch_base_eval_job(
    model_id: str,
    split_id: str,
    decoding: DecodingParams,
    stamp: str,
    instance_type: str = "ml.g5.2xlarge",
    max_run_seconds: int = 3600,
) -> dict[str, Any]:
    """Evaluate the UNTRAINED (base) model on the held-out test set — the control
    that quantifies how much fine-tuning actually helped (base → fine-tuned lift).

    Unlike launch_eval_job there's no trained artifact: eval.py loads the base
    weights straight from Hugging Face by id (SLM_EVAL_BASE_MODEL). It needs no
    source training job, so it can run in PARALLEL with training. Uses the SAME
    eval harness + decoding as the fine-tuned eval (fair comparison) and the SAME
    per-model image tier (resolve_image_uri) — so when a future LLaMA-Factory
    release adds a new tier/image, base eval automatically uses the right one,
    exactly like the train/fine-tuned-eval paths."""
    cfg = load_aws_config()
    sm_sess, boto_sess = _session(cfg)

    model = get_model(model_id)
    if model is None:
        raise ValueError(f"unknown model: {model_id}")
    # Image from the model's OWN tier — future-proof across LLaMA-Factory releases.
    eval_image_uri = resolve_image_uri(cfg, model)

    run_dir = split_dir(split_id)
    if run_dir is None:
        raise ValueError(f"split {split_id} not found on disk")

    # slm-eval- prefix keeps it out of the training-candidate list; -base- marks it.
    job_name = re.sub(
        r"[^a-zA-Z0-9-]", "-", f"slm-eval-{split_id}-base-{stamp}"
    )[:63].strip("-")

    dataset_prefix = _upload_files(
        sm_sess, cfg,
        [run_dir / "eval.jsonl", run_dir / "dataset_info.json"],
        f"{_jobs_key_prefix()}/{job_name}/dataset",
    )
    output_s3 = f"{_jobs_s3_base(cfg)}/{job_name}/output"

    env = eval_env(decoding)
    env["SLM_EVAL_BASE_MODEL"] = model.hf_model_id  # entrypoint → eval.py loads this from HF
    # Repo-side modeling code stays OFF unless this model declares it needs it.
    if model.trust_remote_code:
        env["SLM_EVAL_TRUST_REMOTE_CODE"] = "true"

    est = Estimator(
        image_uri=eval_image_uri,
        role=cfg.role_arn,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_s3,
        base_job_name=job_name,
        sagemaker_session=sm_sess,
        max_run=max_run_seconds,
        environment=_job_environment(env),  # +HF token (gated base models need it)
        tags=_job_tags(),
    )
    # Only the dataset channel — no model channel (base loads from HF).
    est.fit(
        inputs={DATASET_CHANNEL: TrainingInput(dataset_prefix, input_mode="File")},
        wait=False,
    )
    return {
        "jobName": est.latest_training_job.name,
        "baseModel": model.hf_model_id,
        "instanceType": instance_type,
        "imageUri": eval_image_uri,
        "region": cfg.region,
    }


def fetch_metrics(job_name: str) -> dict[str, Any] | None:
    """If an eval job is Completed, download + parse metrics.json from its output."""
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    sm = boto_sess.client("sagemaker")
    d = sm.describe_training_job(TrainingJobName=job_name)
    if d["TrainingJobStatus"] != "Completed":
        return None
    artifact = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not artifact:
        return None

    # artifact is s3://bucket/key/model.tar.gz — download + extract metrics.json
    s3 = boto_sess.client("s3")
    _, _, rest = artifact.partition("s3://")
    bucket, _, key = rest.partition("/")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "model.tar.gz"
        s3.download_file(bucket, key, str(local))
        with tarfile.open(local) as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith("metrics.json")), None)
            if member is None:
                return None
            f = tar.extractfile(member)
            if f is None:
                return None
            return json.loads(f.read().decode("utf-8"))
