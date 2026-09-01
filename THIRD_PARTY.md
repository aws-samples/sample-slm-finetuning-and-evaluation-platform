# Third-Party Notices

This project incorporates and/or redistributes third-party open-source software.
The components below are the **direct, notable dependencies**; each transitively
pulls in further open-source packages whose licenses are carried in their own
distribution metadata (Python: `*.dist-info/`; npm: `node_modules/<pkg>/LICENSE`).

License identifiers use SPDX short forms. Where a component is redistributed in
binary form (the GPU training/eval container), attribution travels with the
package metadata inside the image as well.

> **Maintenance note:** this file lists *direct* dependencies declared in
> `frontend/package.json`, `backend/pyproject.toml`,
> `backend/requirements-lambda.txt`, `backend/requirements-serverless.txt`,
> `agent/pyproject.toml`, `agent/requirements.txt`, `infra/requirements.txt`, and
> `container/Dockerfile`. That is every dependency manifest in the tree;
> `backend/app/export_templates/requirements.txt` also exists but declares no
> packages (it is a documented hook for the generated deploy bundle). Regenerate
> the exhaustive transitive list at build time before any external distribution
> (see *Generating the full attribution list* below). SPDX identifiers were taken
> from each project's published metadata and should be re-verified at release time.

---

## Frontend (`frontend/`, npm)

| Component | Version (range) | License (SPDX) | Project |
| --- | --- | --- | --- |
| `@cloudscape-design/components` | ^3.0.0 | Apache-2.0 | https://github.com/cloudscape-design/components |
| `@cloudscape-design/chat-components` | ^1.0.146 | Apache-2.0 | https://github.com/cloudscape-design/chat-components |
| `@cloudscape-design/code-view` | ^3.0.139 | Apache-2.0 | https://github.com/cloudscape-design/code-view |
| `@cloudscape-design/collection-hooks` | ^1.0.96 | Apache-2.0 | https://github.com/cloudscape-design/collection-hooks |
| `@cloudscape-design/global-styles` | ^1.0.0 | Apache-2.0 | https://github.com/cloudscape-design/global-styles |
| `@lobehub/icons` | ^5.10.0 | MIT | https://github.com/lobehub/lobe-icons |
| `react` | ^18.3.1 | MIT | https://github.com/facebook/react |
| `react-dom` | ^18.3.1 | MIT | https://github.com/facebook/react |
| `recharts` | ^2.15.4 | MIT | https://github.com/recharts/recharts |
| `typescript` (build) | ^5.5.3 | Apache-2.0 | https://github.com/microsoft/TypeScript |
| `vite` (build) | ^5.4.0 | MIT | https://github.com/vitejs/vite |
| `@vitejs/plugin-react` (build) | ^4.3.1 | MIT | https://github.com/vitejs/vite-plugin-react |
| `@types/react`, `@types/react-dom` (build) | ^18.3.x | MIT | https://github.com/DefinitelyTyped/DefinitelyTyped |

> `@lobehub/icons` bundles provider/model brand monograms (used as catalog
> icons). Brand logos may carry separate trademark terms from their respective
> owners independent of the package's MIT license; they are used here for
> nominative identification only.

## Backend — API/runtime (`backend/`, Python)

| Component | Version (range) | License (SPDX) | Project |
| --- | --- | --- | --- |
| `fastapi` | >=0.115 | MIT | https://github.com/fastapi/fastapi |
| `uvicorn[standard]` | >=0.30 | BSD-3-Clause | https://github.com/encode/uvicorn |
| `python-multipart` | >=0.0.9 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| `pydantic` | >=2.7 | MIT | https://github.com/pydantic/pydantic |
| `pyyaml` | >=6.0 | MIT | https://github.com/yaml/pyyaml |
| `sagemaker` (SDK v2) | >=2.220,<3 | Apache-2.0 | https://github.com/aws/sagemaker-python-sdk |
| `boto3` | >=1.34 | Apache-2.0 | https://github.com/boto/boto3 |
| `mangum` | >=0.17 | MIT | https://github.com/jordaneremieff/mangum |
| `requests` | (transitive/runtime) | Apache-2.0 | https://github.com/psf/requests |
| `certifi` | (transitive, via `requests`/`httpx`) | MPL-2.0 | https://github.com/certifi/python-certifi |

> `certifi` is called out by name because **MPL-2.0** is the only non-permissive
> license this review found anywhere in the tree. It is weak, *file-level* copyleft:
> the reciprocity attaches to modified MPL-licensed files, not to code that merely
> imports the package. This project uses `certifi` unmodified and does not
> incorporate its source, so it places no obligation on the rest of this repository
> — but its license text must travel wherever the package is redistributed (which
> the wheel's own `*.dist-info/` already handles). It reaches both the backend and
> the agent environments transitively; nothing depends on it directly.

## Backend — serverless engine venv (`backend/requirements-serverless.txt`)

Installed into an **isolated** interpreter (`/opt/serverless-venv`), not the app env.

| Component | Version | License (SPDX) | Project |
| --- | --- | --- | --- |
| `sagemaker-train` | 1.13.1 | Apache-2.0 | https://github.com/aws/sagemaker-python-sdk |
| `sagemaker-core` | 2.13.1 | Apache-2.0 | https://github.com/aws/sagemaker-core |
| `sagemaker-mlflow` | 0.4.0 | Apache-2.0 | https://github.com/aws/sagemaker-mlflow |

## Agent — dataset investigator (`agent/`, Python)

| Component | Version (range) | License (SPDX) | Project |
| --- | --- | --- | --- |
| `strands-agents` | >=1.42.0 | Apache-2.0 | https://github.com/strands-agents/sdk-python |
| `strands-agents-tools` | >=0.8.0 | Apache-2.0 | https://github.com/strands-agents/tools |
| `bedrock-agentcore` | >=1.14.0 | Apache-2.0 | https://github.com/aws/bedrock-agentcore-sdk-python |

> The three rows above are the direct dependencies, from `agent/pyproject.toml`.
> The committed `agent/requirements.txt` is a different thing: a fully pinned
> `uv export` of the *resolved* tree (~80 packages, transitive included), kept in
> the repo so the AgentCore build is reproducible. It is generated, not curated, so
> this file does not restate it — read `agent/pyproject.toml` for what the agent
> asks for and `agent/requirements.txt` for the exact versions it got. Each pinned
> wheel carries its own license in `*.dist-info/`; the only non-permissive one in
> that set is `certifi` (MPL-2.0, noted above).

## Infrastructure as code (`infra/`, Python CDK)

Deploy-time only — these run on the operator's machine or in CI to synthesize
CloudFormation. They are not installed into any runtime artifact and are not
redistributed.

| Component | Version (range) | License (SPDX) | Project |
| --- | --- | --- | --- |
| `aws-cdk-lib` | >=2.140,<3 | Apache-2.0 | https://github.com/aws/aws-cdk |
| `constructs` | >=10.0,<11 | Apache-2.0 | https://github.com/aws/constructs |

## Training / eval container image (`container/Dockerfile`, redistributed via ECR)

The GPU training/eval image is built **FROM** the official LLaMA-Factory image
and adds inference/quantization libraries. These are redistributed in binary form
inside the container; each package's license + copyright travel in its own
on-image metadata (`*.dist-info/`, `/usr/share/doc/*`).

| Component | License (SPDX) | Project |
| --- | --- | --- |
| LLaMA-Factory (base image `hiyouga/llamafactory`) | Apache-2.0 | https://github.com/hiyouga/LLaMA-Factory |
| vLLM | Apache-2.0 | https://github.com/vllm-project/vllm |
| bitsandbytes | MIT | https://github.com/bitsandbytes-foundation/bitsandbytes |
| Liger-Kernel (`liger-kernel`) | BSD-2-Clause | https://github.com/linkedin/Liger-Kernel |
| PyTorch, transformers, peft, accelerate, etc. (from the base image) | BSD-3-Clause / Apache-2.0 (respectively) | bundled by the LLaMA-Factory base image; see on-image metadata |

> The base image transitively includes CUDA/PyTorch and the Hugging Face stack.
> Their full attribution is carried inside the published base image; this project
> does not modify those packages.

---

## Models and datasets (user-supplied / referenced — NOT redistributed here)

This tool fine-tunes open-weight models and imports datasets that the **user**
selects; it does not bundle third-party model weights or datasets in this
repository. The boundaries that matter:

- **Base models** are downloaded from Hugging Face at training time under each
  model's own license. A gated repo (e.g. Llama, Gemma, Mistral) needs the user's
  own Hugging Face token, and access to it is granted by the publisher on Hugging
  Face, not by this tool. The catalog records each model's license alongside its
  `gated` flag (`backend/app/catalog.py`) — the two are independent: Mistral 7B is
  Apache-2.0 behind a gated repo, and some ungated repos carry restrictive custom
  terms.
- **Exporting a fine-tune of a gated base** follows one of two paths, chosen by the
  training method in `backend/app/export.py`:
  - *LoRA/QLoRA* produces an adapter, and the bundle ships the **adapter only**.
    `deploy.sh` pulls the base from Hugging Face at deploy time using the user's
    own `HF_TOKEN`, so no gated weights are redistributed by this project.
  - *full/freeze* produces no adapter — the only artifact is **merged weights that
    embed the gated base**. Those **are** handed to the user, behind a click-through:
    the presigned download URL is withheld until the export request carries
    `license_accepted=true`, which the UI sets when the user clicks through a prompt
    naming the base model. That is the user *asserting* they have accepted the
    publisher's license; the backend does not verify the assertion against Hugging
    Face, and responsibility for its truth sits with the user. Training, eval and
    leaderboard comparison all work without accepting — only the download is gated.
- **Imported datasets** are fetched from Hugging Face by the user. The import UI
  surfaces the dataset's declared license as an advisory banner
  (`backend/app/hf_ingest.py` buckets the slug as permissive / restrictive /
  unknown; anything non-commercial, copyleft, use-restricted such as the
  RAIL/OpenRAIL family, or simply undeclared gets an amber "read the terms"
  warning). It never blocks an import — Hugging Face license tags are self-reported
  and often absent, so an allow/deny decision would over-claim. The user is
  responsible for confirming they may use and redistribute the data.
- The only data files committed in this repo are the small **synthetic** demo
  datasets in `frontend/src/demoDatasets.ts`, authored for this project.

---

## Generating the full attribution list (before external distribution)

This file covers direct dependencies. Before publishing externally, generate the
exhaustive transitive attribution and verify SPDX identifiers:

```bash
# Frontend (npm)
cd frontend && npx license-checker --production --summary

# Backend / agent (Python) — per environment
pip install pip-licenses && pip-licenses --format=markdown --with-urls

# Container image — extract on-image package metadata
#   docker run --rm <image> pip-licenses --format=markdown
```

Keeping this file current: when a dependency is added, removed, or version-bumped in
any manifest named in the maintenance note at the top, update the matching row here
in the same change. Verify each SPDX identifier against the project's own `LICENSE`
file or published package metadata rather than a third-party summary — projects do
relicense between releases.
