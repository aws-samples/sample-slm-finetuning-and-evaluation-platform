# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# API Lambda image for the SLM platform backend (FastAPI via Mangum).
# Built by CDK (DockerImageFunction) — no manual push. This is the SMALL app
# image; the big GPU training/eval image lives in container/ and is built by a
# CodeBuild project, not here.
FROM public.ecr.aws/lambda/python:3.12

# Dependencies first for layer caching.
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# --- Serverless engine: ISOLATED V3-SDK venv (process isolation) ------------
# The SageMaker serverless trainers (sagemaker.train.*) need the V3 SDK
# (sagemaker-train + sagemaker-core 2.x), which is MUTUALLY EXCLUSIVE with the V2
# SDK the LLaMA-Factory path imports (sagemaker.estimator) — installing it into
# the app env would break the default engine. So we put it in a SEPARATE venv
# and run the standalone launcher (app/engines/serverless_launcher.py, which
# imports NO app.* modules) under it via subprocess. SLM_SERVERLESS_PYTHON tells
# the engine where that interpreter is. requirements-serverless.txt pins the V3
# set. This adds ~150 MB to the image (well under Lambda's 10 GB), and only the
# worker Lambda ever invokes it — the API/reconcile paths never import it.
COPY requirements-serverless.txt ${LAMBDA_TASK_ROOT}/
RUN python -m venv /opt/serverless-venv \
 && /opt/serverless-venv/bin/pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-serverless.txt
ENV SLM_SERVERLESS_PYTHON=/opt/serverless-venv/bin/python

# Application package.
COPY app ${LAMBDA_TASK_ROOT}/app

# Hosted backend uses the S3 state store; account/region/bucket come from env
# vars set by CDK at deploy time. SLM_SCRATCH_DIR points at Lambda's writable
# /tmp (the only writable path in the runtime).
ENV SLM_STORAGE_BACKEND=cloud \
    SLM_SCRATCH_DIR=/tmp/slm-state

# Default entrypoint = the API. The reconcile Lambda overrides CMD to
# app.lambda_handler.reconcile_handler.
CMD ["app.lambda_handler.handler"]
