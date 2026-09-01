# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The default engine — LLaMA-Factory on a SageMaker training job.

This is a VERBATIM wrapper around the original launch logic that used to live
inline in orchestrate.launch_training_job. It calls the same module functions
in the same order, so a launch routed through this engine produces byte-for-byte
the same S3 keys, train/export YAML, job name, tags, metric definitions, and
return dict as before the engine seam existed.

NOTHING engine-specific or new happens here — it exists purely so the default
path goes through the same Engine interface a new engine will.
"""

from __future__ import annotations

from typing import Any

from .base import DEFAULT_ENGINE


class LlamaFactoryEngine:
    name = DEFAULT_ENGINE

    def launch_training_job(
        self,
        model: Any,
        split_id: str,
        hp: Any,
        instance_type: str,
        stamp: str,
        max_run_seconds: int,
        use_spot: bool,
        image_tag: str | None,
        resume_checkpoint_s3: str | None = None,
    ) -> dict[str, Any]:
        # Import here (not at module load) to avoid any import cycle with
        # orchestrate, which imports this package's get_engine.
        from .. import orchestrate as orch
        from ..aws_config import load_aws_config

        cfg = load_aws_config()
        sm_sess, _ = orch._session(cfg)
        job_name = orch._job_name(model.id, split_id, stamp, method=hp.finetuning_type)
        # Per-model tier → ECR image; explicit image_tag overrides it (cross-tier verify).
        image_uri = (
            cfg.image_uri_for_tier(image_tag)
            if image_tag
            else orch.resolve_image_uri(cfg, model)
        )

        channels = orch.upload_job_inputs(
            sm_sess, cfg, model, hp, split_id, job_name, use_spot=use_spot
        )
        output_s3 = f"{channels['base']}/output"
        # Checkpoints sync to S3 for EVERY job now (resumable on-demand + spot).
        # For a RESUME, point the new job's checkpoint dir at the FAILED job's
        # checkpoint prefix so SageMaker restores it into /opt/ml/checkpoints and
        # the entrypoint resumes from the last step; otherwise use this job's own.
        checkpoint_s3 = resume_checkpoint_s3 or f"{channels['base']}/checkpoints"

        est = orch.build_estimator(
            cfg, sm_sess, model, job_name, instance_type, max_run_seconds, output_s3,
            use_spot=use_spot, checkpoint_s3=checkpoint_s3, image_uri=image_uri,
        )
        est.fit(
            inputs={
                orch.DATASET_CHANNEL: orch.TrainingInput(channels["dataset"], input_mode="File"),
                orch.CONFIG_CHANNEL: orch.TrainingInput(channels["config"], input_mode="File"),
            },
            wait=False,
        )
        return {
            "jobName": est.latest_training_job.name,
            "engine": self.name,
            "instanceType": instance_type,
            "imageUri": image_uri,
            "imageTag": image_tag or getattr(model, "image_tag", "stable"),
            "datasetS3": channels["dataset"],
            "configS3": channels["config"],
            "outputS3": output_s3,
            # The S3 prefix SageMaker syncs checkpoints to — recorded so a later
            # resume can re-point a new job at it (resume from last checkpoint).
            "checkpointS3": checkpoint_s3,
            "useSpot": use_spot,
            "region": cfg.region,
        }
