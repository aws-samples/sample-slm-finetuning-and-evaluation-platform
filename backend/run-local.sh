#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Run the backend locally on port 8000 — the port the Vite dev server proxies
# /api to (see frontend/vite.config.ts).
#
# Storage defaults to LOCAL disk (SLM_STORAGE_BACKEND=local, state under
# backend/data), so a fresh checkout runs with no AWS account and no cloud state.
#
#   SLM_STORAGE_BACKEND=cloud points the local backend at a DEPLOYED stack's S3
#   state instead, so the local UI lists the same datasets / verifications / runs
#   as the hosted app. That mode requires SLM_S3_BUCKET, and it READS AND WRITES
#   the state the deployed app serves: browsing is safe, but the action buttons
#   (verify, smoke-test, backfill, delete) mutate it.
#
# Usage:  ./run-local.sh
#         PORT=8001 ./run-local.sh
#         SLM_STORAGE_BACKEND=cloud SLM_S3_BUCKET=<state-bucket> ./run-local.sh
#
# Pair with the frontend:  cd ../frontend && npm run dev   (proxies /api -> :8000)
set -euo pipefail

cd "$(dirname "$0")"

# --- state store
export SLM_STORAGE_BACKEND="${SLM_STORAGE_BACKEND:-local}"
export SLM_AWS_REGION="${SLM_AWS_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-$SLM_AWS_REGION}"

# store.py accepts "cloud" or "s3" for the S3-backed store, case-insensitively.
backend_lc="$(printf '%s' "$SLM_STORAGE_BACKEND" | tr '[:upper:]' '[:lower:]')"
cloud_store=0
[[ "$backend_lc" == "cloud" || "$backend_lc" == "s3" ]] && cloud_store=1

if (( cloud_store )); then
  # No default bucket: the name is specific to a deployment (the CDK stack
  # creates it and reports it as the OutDataBucket stack output), and guessing
  # one would either fail or write into a bucket you don't own.
  if [[ -z "${SLM_S3_BUCKET:-}" ]]; then
    echo "error: SLM_STORAGE_BACKEND=$SLM_STORAGE_BACKEND requires SLM_S3_BUCKET=<state bucket>." >&2
    echo "       Use the OutDataBucket output of your deployed SlmPlatformInfra stack," >&2
    echo "       or drop SLM_STORAGE_BACKEND to run against local disk instead." >&2
    exit 1
  fi
  export SLM_S3_BUCKET
  export SLM_STATE_PREFIX="${SLM_STATE_PREFIX:-slm-platform/state}"
fi

# --- AWS credentials (only needed for the cloud store, SageMaker launches and
# Secrets Manager). Whatever the ambient AWS config resolves to is used; export
# AWS_PROFILE=<name> before running to pick a specific profile. SLM_AWS_PROFILE
# is what the backend's own boto clients read (aws_config.load_aws_config), so it
# is kept in lockstep — otherwise those clients fall back to the "default"
# profile while the rest of the process uses AWS_PROFILE, and calls like the HF
# token read silently hit the wrong account.
if [[ -n "${AWS_PROFILE:-}" ]]; then
  export SLM_AWS_PROFILE="${SLM_AWS_PROFILE:-$AWS_PROFILE}"
fi

# --- HF token (Secrets Manager): unlocks gated models (Llama/Mistral/Gemma).
# Set SLM_HF_SECRET_NAME to the secret your deployment stores tokens in (the CDK
# stack names it "<prefix>/hf-token"). Left unset, hf_token_is_set() is always
# False and gated models stay locked — which is the right local default.
if [[ -n "${SLM_HF_SECRET_NAME:-}" ]]; then
  export SLM_HF_SECRET_NAME
fi

# --- multi-tenancy: per-user resource isolation (state keyed by the caller's
# stable Cognito username). There is no JWT on a dev box, so SLM_DEV_TENANT
# simulates a logged-in user; state then lives under users/<value>/. Set it to ""
# to use the legacy shared (default-tenant) view, or SLM_MULTI_TENANT="" to turn
# isolation off entirely.
export SLM_MULTI_TENANT="${SLM_MULTI_TENANT:-true}"
export SLM_DEV_TENANT="${SLM_DEV_TENANT-local-dev}"

PORT="${PORT:-8000}"

if (( cloud_store )); then
  echo "→ backend: cloud store  s3://${SLM_S3_BUCKET}/${SLM_STATE_PREFIX}"
  echo "  (reads AND WRITES the state the deployed app serves)"
else
  echo "→ backend: local store  ${SLM_DATA_DIR:-backend/data}"
fi
echo "→ HF secret: ${SLM_HF_SECRET_NAME:-<unset → gated models locked>}"
echo "→ multi-tenant: ${SLM_MULTI_TENANT}  dev-tenant: ${SLM_DEV_TENANT:-<none → default>}"
echo "→ profile: ${AWS_PROFILE:-<ambient credentials>}   region: ${SLM_AWS_REGION}   port: ${PORT}"

# Use the venv's interpreter explicitly — a bare `uvicorn` may resolve to a
# system one that lacks the deps.
if [[ ! -x .venv/bin/python ]]; then
  echo "error: no interpreter at backend/.venv/bin/python — create the venv and" >&2
  echo "       install the dependencies from backend/pyproject.toml first" >&2
  echo "       (see the 'Backend' setup steps in the top-level README)." >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn app.main:app --reload --port "${PORT}"
