# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Frozen LLaMA-Factory training image for the SLM platform.
#
# ONE image per LLaMA-Factory release (architecture decision). Pinned to the
# official GPU image so the heavy CUDA/torch/LLaMA-Factory stack is reproducible
# and vendor-maintained — we only add the generic SageMaker entrypoint on top.
#
# Bump POLICY: change the base tag deliberately (human-picked release), rebuild,
# re-push, re-smoke-test. Do NOT float to :latest.
#
# Multi-image: the SAME Dockerfile builds every tier — the base LLaMA-Factory
# release is a build ARG so one source produces both `stable` (0.9.4) and
# `latest` (0.9.5). CodeBuild passes --build-arg LF_BASE_TAG=<tag>; default 0.9.4
# keeps a plain `docker build` reproducing the stable image.
#
# BASE REGISTRY: defaults to the official Docker Hub org (`hiyouga`), so a plain
# `docker build` works with no extra setup. In CI, repeated unauthenticated
# Docker Hub pulls hit the rate limit (429) and fail the build, so point
# LF_BASE_REGISTRY at a registry that mirrors the same repository — e.g. an ECR
# pull-through cache for Docker Hub, whose URL looks like
# <account-id>.dkr.ecr.<region>.amazonaws.com/docker-hub/hiyouga. The image path
# `${LF_BASE_REGISTRY}/llamafactory:${LF_BASE_TAG}` must resolve either way.
ARG LF_BASE_TAG=0.9.4
ARG LF_BASE_REGISTRY=hiyouga
FROM ${LF_BASE_REGISTRY}/llamafactory:${LF_BASE_TAG}

# vLLM for offline batch eval. Pinned to a release that targets the
# image's torch; installed without deps churn where possible. Overridable per
# tier (the newer LLaMA-Factory base ships a newer torch needing a newer vllm).
# If this layer ever conflicts with the base torch/CUDA, the eval falls back to
# HF generate (the entrypoint handles either backend).
ARG VLLM_VERSION=0.8.5.post1
RUN pip install --no-cache-dir "vllm==${VLLM_VERSION}" || \
    echo "WARN: vllm install failed; eval will use HF generate backend"

# bitsandbytes for QLoRA (4-bit quantized base). The frozen LLaMA-Factory image
# does NOT ship it, so a QLoRA run fails at "No package metadata was found for
# bitsandbytes" (caught by a real smoke-test). QLoRA = finetuning_type:lora +
# quantization_bit:4, and LLaMA-Factory loads the 4-bit base via bitsandbytes.
# Pinned >=0.43 for current torch/transformers; CPU-less wheels are fine (the
# training instances are GPU). If this ever conflicts, only QLoRA is affected —
# plain LoRA/SFT/DPO don't import bitsandbytes.
RUN pip install --no-cache-dir "bitsandbytes>=0.43.0" || \
    echo "WARN: bitsandbytes install failed; QLoRA will be unavailable"

# liger-kernel for the optional `enable_liger_kernel` efficiency knob (fused Triton
# kernels → less memory + faster training). The frozen LLaMA-Factory image does NOT
# ship it, and llamafactory-cli HARD-REQUIRES it when enable_liger_kernel:true
# (check_version("liger-kernel", mandatory=True) → "No package metadata was found
# for liger-kernel"), proven by a real smoke-test. Only the Liger knob imports it;
# plain runs / NEFTune / packing don't, so a failed install degrades only that knob.
RUN pip install --no-cache-dir liger-kernel || \
    echo "WARN: liger-kernel install failed; the Liger efficiency knob will be unavailable"

# SageMaker "bring your own container" invokes `docker run <image> train`.
# Install our generic entrypoint as /usr/local/bin/train so that invocation
# runs train -> export (or eval, via SLM_MODE) for any model.
COPY entrypoint.sh /usr/local/bin/train
COPY eval.py /usr/local/bin/eval.py
RUN chmod +x /usr/local/bin/train

# Quiet/robust defaults for headless training.
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false

# Default to running our entrypoint; SageMaker passes `train` which it ignores.
ENTRYPOINT ["/usr/local/bin/train"]
