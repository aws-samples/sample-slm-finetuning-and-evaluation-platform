# Deploy your fine-tuned model

This bundle deploys the model you fine-tuned on the SLM platform to a **SageMaker
real-time endpoint in your own AWS account**. The model weights are **not** in
this zip — they're downloaded on demand via a time-limited link in `manifest.json`
(see "Weights link expiry" below).

## Contents

| File | What it is |
|------|------------|
| `manifest.json` | Model facts: base model, template, deploy mode, instance, **LLaMA-Factory base image tag**, the training **engine**, and the presigned weights link(s) |
| `deploy.sh` | Re-runnable deploy → SageMaker endpoint (build image → stage model → create endpoint) |
| `inference.py` | A small FastAPI server (`/ping` + `/invocations`) — same generate path as the platform's eval harness |
| `Dockerfile` | `FROM hiyouga/llamafactory:<tag>` — the exact image the model trained on |
| `requirements.txt` | Hook for extra inference deps (normally empty; the base image has everything) |

## Why the LLaMA-Factory image?

The inference container is built **FROM the same `hiyouga/llamafactory` image your
model was fine-tuned on** (the tag is in `manifest.json` → `baseImageTag`). That
image pins the exact `transformers` / `torch` / `vllm` versions training used, so
the merged weights **always load** — no guessing a compatible transformers version
(which is the usual failure when deploying newer architectures like Qwen3 on a
generic inference image). The base image is **public on Docker Hub**; `deploy.sh`
pulls it during the build.

## Prerequisites

- **Docker** (to build the inference image) and the **AWS CLI**, configured with a
  profile for your account.

## Quick start

```bash
chmod +x deploy.sh
./deploy.sh --profile <your_aws_profile> --region <region>
```

The script builds the inference image from the matching LLaMA-Factory base and
pushes it to your ECR, downloads + stages the model in your S3, and stands up the
endpoint. When it finishes it prints an `invoke-endpoint` test command and the
delete command.

### Re-runnable (idempotent)

If a run fails partway (image pushed but endpoint creation failed, network blip,
etc.), **just run the exact same command again** — it resumes instead of redoing
work:

- ECR image with this tag already present → **skips the build + push**
- `model.tar.gz` already staged in S3 → **skips the download + repackage**
- execution role already exists → **reuses it**
- SageMaker model / endpoint-config already exist → **reuses them**
- endpoint already exists → **updates it** to the latest config

The endpoint name is deterministic from the model id (so re-runs target the same
endpoint, not a new one each time). Pin a custom name with `--name <endpoint>`.

## Two delivery modes (set automatically in `manifest.json`)

- **`merged`** (permissive-license bases: Qwen, Granite, Phi, MiniCPM): the bundle
  deploys a **standalone merged model** — no Hugging Face dependency, no token.
- **`adapter`** (gated bases: Llama, Gemma, Mistral): the bundle deploys the **LoRA
  adapter only**, and the base model is pulled from Hugging Face at load time. You
  must **export your own HF token** first (and have accepted the base model's
  license on HF):

  ```bash
  export HF_TOKEN=hf_xxx
  ./deploy.sh --profile <profile> --region <region>
  ```

  This is because redistributing merged *gated* weights would violate the model
  license — the adapter is your own fine-tuning delta, which is safe to move.

## Two training engines (set automatically in `manifest.json` → `engine`)

The weights are packaged differently depending on which engine trained the model,
but `deploy.sh` handles both transparently — you run the same command either way:

- **`llama_factory`**: weights are one `model.tar.gz` (`weightsUrl`). The script
  downloads + extracts it and picks the licensed subdir.
- **`sagemaker_serverless`**: the output is an uncompressed set of loose files
  (`weightsFiles` — each presigned individually), already in the right layout.
  The script downloads them straight into the model dir (no tarball to extract).

In both cases the inference image is still built `FROM hiyouga/llamafactory:<tag>`
— the serverless merged model loads on that image too (it's what the platform's
own eval uses), so versions match with no extra work.

## Weights link expiry

The presigned link(s) in `manifest.json` are valid for **6 hours**. If they
expire, re-export from the platform's Runs page to get a fresh bundle.

## Cost

A real-time endpoint bills **per hour while it exists**, regardless of traffic.
Delete it when you're done (the script prints the exact command):

```bash
aws sagemaker delete-endpoint --profile <profile> --region <region> --endpoint-name <name>
```

## The inference image

`deploy.sh` builds `Dockerfile` (`FROM hiyouga/llamafactory:<baseImageTag>`) and
pushes it to an `slm-inference` repo in your ECR, then points the SageMaker model
at it. To pin a different base tag, override `LF_TAG` — but the default (the tag
the model trained on) is what guarantees version compatibility, so change it only
if you know what you're doing.
