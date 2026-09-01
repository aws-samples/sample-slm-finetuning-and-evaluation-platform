#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Generic SageMaker entrypoint for the SLM platform — ONE script for ALL models.
#
# The platform renders per-model config as YAML (see backend/app/render.py) and
# ships it to the job; this script just runs the engine. No per-model code.
#
# Mode is the first arg (SageMaker passes the value of the `train` toml key, but
# we read $1 / $SLM_MODE explicitly):
#   train (default) : llamafactory-cli train -> export
#   eval            : python eval.py over the held-out set
#
# SageMaker channel/dir conventions:
#   /opt/ml/input/data/config    : train.yaml + export.yaml (train mode)
#   /opt/ml/input/data/dataset   : train.jsonl, eval.jsonl, dataset_info.json
#   /opt/ml/input/data/model     : a prior merged model (eval mode input)
#   /opt/ml/model                : final artifacts (SageMaker uploads to S3 after)
#   /opt/ml/checkpoints          : checkpoints (spot resume)
set -euo pipefail

MODE="${SLM_MODE:-${1:-train}}"
CONFIG_DIR=/opt/ml/input/data/config
MODEL_DIR=/opt/ml/input/data/model

echo "[entrypoint] mode=${MODE}"
echo "[entrypoint] llamafactory version:"
llamafactory-cli version || true

if [ "${MODE}" = "eval" ]; then
  echo "[entrypoint] ===== EVAL ====="
  if [ -n "${SLM_EVAL_BASE_MODEL:-}" ]; then
    # BASE-MODEL eval: no trained artifact — score the untrained model straight
    # from Hugging Face (the "did fine-tuning help?" control). eval.py loads it by
    # HF id; no model channel / tarball involved. Same harness + decoding as the
    # fine-tuned eval, so the comparison is apples-to-apples.
    echo "[entrypoint] base-model eval: ${SLM_EVAL_BASE_MODEL}"
    export SLM_EVAL_MODEL_DIR="${SLM_EVAL_BASE_MODEL}"
  else
    # The model channel ships the prior job's model.tar.gz. SageMaker training
    # input channels do NOT auto-extract, so do it ourselves.
    EXTRACT_DIR="${MODEL_DIR}"
    TARBALL="$(find "${MODEL_DIR}" -maxdepth 2 -name 'model.tar.gz' | head -1 || true)"
    if [ -n "${TARBALL}" ]; then
      echo "[entrypoint] extracting ${TARBALL}"
      EXTRACT_DIR="${MODEL_DIR}/extracted"
      mkdir -p "${EXTRACT_DIR}"
      tar -xzf "${TARBALL}" -C "${EXTRACT_DIR}"
    fi
    # Locate the dir that actually holds the model (config.json). The merged
    # weights live under merged/ inside the tar (export_dir=/opt/ml/model/merged).
    CONFIG_PATH="$(find "${EXTRACT_DIR}" -maxdepth 3 -name 'config.json' | head -1 || true)"
    if [ -n "${CONFIG_PATH}" ]; then
      export SLM_EVAL_MODEL_DIR="$(dirname "${CONFIG_PATH}")"
    else
      export SLM_EVAL_MODEL_DIR="${EXTRACT_DIR}"
    fi
  fi
  echo "[entrypoint] model dir/id: ${SLM_EVAL_MODEL_DIR}"
  python /usr/local/bin/eval.py
  echo "[entrypoint] ===== DONE (eval) ====="
  ls -R /opt/ml/model || true
  exit 0
fi

# Default: train -> export
TRAIN_YAML="${CONFIG_DIR}/train.yaml"
EXPORT_YAML="${CONFIG_DIR}/export.yaml"
CHECKPOINT_DIR="${SLM_CHECKPOINT_DIR:-/opt/ml/checkpoints}"

echo "[entrypoint] ===== TRAIN ====="
echo "[entrypoint] using ${TRAIN_YAML}"
cat "${TRAIN_YAML}"

# Spot-interruption resume: when a job runs on spot, render.py sets the train
# output_dir to ${CHECKPOINT_DIR}, which SageMaker syncs to checkpoint_s3_uri
# continuously AND restores BEFORE the container starts on a retry. So if a
# prior checkpoint-* dir is present, resume from it; otherwise start fresh.
# (On-demand runs have no checkpoint dir → always a fresh start, unchanged.)
RESUME_ARG=""
if [ -d "${CHECKPOINT_DIR}" ] && find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
  echo "[entrypoint] found existing checkpoint(s) in ${CHECKPOINT_DIR} — resuming"
  ls -la "${CHECKPOINT_DIR}" || true
  # LLaMA-Factory passes through HF Trainer's resume; key=value overrides the YAML.
  RESUME_ARG="resume_from_checkpoint=true"
else
  echo "[entrypoint] no checkpoint to resume from — fresh start"
fi

llamafactory-cli train "${TRAIN_YAML}" ${RESUME_ARG}

# EXPORT (merge adapter -> safetensors) — ONLY for adapter methods (lora/qlora).
# FULL-WEIGHT methods (finetuning_type: full/freeze) write standalone safetensors
# directly to output_dir and have NO adapter to merge, so the platform omits
# export.yaml from the config channel for them (render.render_all). Branch on its
# presence: no export.yaml => full-weight run => skip the merge. SLM_SKIP_EXPORT=1
# is honored too as an explicit belt-and-suspenders override. (A merge with no
# adapter would error or silently no-op, so skipping is required, not just an
# optimization.)
MERGED_DIR=/opt/ml/model/merged
if [ -f "${EXPORT_YAML}" ] && [ "${SLM_SKIP_EXPORT:-0}" != "1" ]; then
  echo "[entrypoint] ===== EXPORT (merge adapter -> safetensors) ====="
  echo "[entrypoint] using ${EXPORT_YAML}"
  cat "${EXPORT_YAML}"
  llamafactory-cli export "${EXPORT_YAML}"
else
  # FULL-WEIGHT run (full/freeze): no adapter, nothing to merge. Training wrote the
  # standalone model to output_dir (CHECKPOINT_DIR). SageMaker packs /opt/ml/model
  # into model.tar.gz (eval/export/deploy consume that), but CHECKPOINT_DIR syncs
  # to a SEPARATE checkpoint S3 path — so the final weights would be STRANDED out
  # of model.tar.gz. Stage the FINAL model (top-level files, not the checkpoint-*
  # subdirs that hold optimizer state) into ${MERGED_DIR} so the downstream sees
  # the SAME merged/ layout as a LoRA run — eval/export/deploy stay method-agnostic.
  echo "[entrypoint] ===== EXPORT SKIPPED (full-weight run: no adapter to merge) ====="
  echo "[entrypoint] staging final full-weight model ${CHECKPOINT_DIR} -> ${MERGED_DIR}"
  mkdir -p "${MERGED_DIR}"
  # Top-level model files only (config.json, *.safetensors[.index.json], tokenizer*,
  # generation_config.json, *.model, *.txt). The checkpoint-* subdirs (optimizer
  # state) are left behind. EXCLUDE SageMaker's transient sync markers
  # (*.sagemaker-uploaded / *.sagemaker-uploading) — they're 0-byte bookkeeping
  # files that would otherwise clutter the merged model dir.
  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f \
    ! -name '*.sagemaker-uploaded' ! -name '*.sagemaker-uploading' \
    -exec cp -v {} "${MERGED_DIR}/" \;
  if [ ! -f "${MERGED_DIR}/config.json" ]; then
    echo "[entrypoint] ERROR: full-weight staging produced no config.json in ${MERGED_DIR}" >&2
    ls -la "${CHECKPOINT_DIR}" || true
    exit 1
  fi
  ls -la "${MERGED_DIR}" || true
fi

echo "[entrypoint] ===== DONE (train) ====="
ls -R /opt/ml/model || true
