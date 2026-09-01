# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# SageMaker inference image for an exported SLM fine-tune.
#
# Built FROM the SAME LLaMA-Factory image the model was TRAINED on (tag passed by
# deploy.sh as --build-arg LF_TAG=<baseImageTag from manifest.json>). That image
# pins the exact transformers / torch / vllm versions training used, so the
# weights load with ZERO version-mismatch risk — no guessing a compatible
# transformers version. The image is public on Docker Hub; you pull it yourself.
ARG LF_TAG=0.9.4
FROM hiyouga/llamafactory:${LF_TAG}

# inference.py is a self-contained FastAPI server (GET /ping + POST /invocations,
# SageMaker's container contract). FastAPI + uvicorn already ship in the
# LLaMA-Factory base (it serves its own API), so no extra install is needed;
# PEFT (for adapter mode) is also present. We install nothing → faster, no risk
# of a pip resolver conflict against the base's pinned stack.
COPY inference.py /opt/ml/code/inference.py
ENV SLM_MODEL_DIR=/opt/ml/model

# SageMaker launches an inference image as `docker run <image> serve`. Provide a
# `serve` executable on PATH that starts our server. Using a script (not a bare
# ENTRYPOINT) keeps the base image's own entrypoint from interfering and matches
# what SageMaker expects for a bring-your-own-container model.
RUN printf '#!/usr/bin/env bash\nexec python /opt/ml/code/inference.py\n' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

# Clear any inherited entrypoint so `serve` (passed by SageMaker as the CMD) runs.
ENTRYPOINT []
CMD ["serve"]
