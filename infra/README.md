# CDK application

One `cdk deploy` stands up the whole platform in your own AWS account and prints
a CloudFront URL that serves a working app. No manual copy-paste, no manual image
push, no credential wiring.

Read the [top-level README](../README.md) first for prerequisites with versions,
the cost breakdown, the teardown checklist and the sample-code disclaimer. This
file covers the stack itself.

## What it creates

One stack, `SlmPlatformInfra`, in `us-east-1`.

| Resource | Purpose |
| --- | --- |
| CloudFront distribution (+OAC, +WAF) | One URL: serves the SPA, routes `/api/*` to API Gateway |
| S3 bucket (SPA) | The built React and Cloudscape frontend. Private, reachable only through OAC |
| S3 bucket (data) | Datasets, job artifacts and app state (the CloudStore prefix) |
| API Lambda | FastAPI via Mangum, from a container image CDK builds from `backend/Dockerfile` |
| Reconcile Lambda | Same image, triggered by EventBridge every minute to advance races headless |
| Worker Lambda | Same image, 15-minute timeout, invoked asynchronously for work that exceeds API Gateway's 29 seconds |
| HTTP API Gateway | Routes to the API Lambda |
| Cognito user pool + hosted-UI domain + app client | JWT-gates every `/api/*` route except `/api/health` |
| WAFv2 WebACL | On CloudFront: AWS managed common rules plus a 2000 request per IP rate limit |
| ECR repository | The LLaMA-Factory training and eval image, one tag per tier |
| ECR pull-through cache rule + Secrets Manager secret | Pulls the LLaMA-Factory base from Docker Hub through ECR |
| Secrets Manager secret (`<prefix>/hf-token`) | Per-user Hugging Face tokens for gated models |
| CodeBuild projects | One per image tier, plus an adhoc project the Images page drives, that build and push the training image to ECR |
| SageMaker execution role | The role SageMaker training and eval jobs assume |
| Reward Lambda execution role | Least-privilege role for the RLVR reward scoring Lambdas the app creates at runtime |
| EventBridge rule | Runs the reconcile Lambda every minute |
| SNS topic, AWS Budgets budget, CloudWatch billing alarm | Cost alerting |
| ACM certificate + Route 53 alias records | Only when `-c customDomain` is set |

The region is `us-east-1` because a CloudFront-scoped WAF WebACL must live there,
which keeps this a single stack. SageMaker `g5` instances and the Bedrock Claude
models the baselines use are both available there. `app.py` pins the region to
`us-east-1` and reads only `-c region=<other>` as an override — not
`CDK_DEFAULT_REGION`, which the CDK CLI sets on every invocation from your
resolved profile and which would therefore silently relocate the stack to
whatever region your profile names. Overriding the pin means moving the WebACL
out of the stack yourself.

## Deploy

Prerequisites: Python 3.11 or newer, Node 20 or newer, Docker running, the
account bootstrapped in `us-east-1` (`cdk bootstrap aws://<account-id>/us-east-1`),
and credentials for the target account.

```bash
cd infra
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export PATH=".venv/bin:$PATH"

# Export credentials into the environment so the CDK Node SDK picks them up
# reliably; a stray AWS_PROFILE in the shell can otherwise shadow them.
eval "$(aws configure export-credentials --profile <your-profile> --format env)"
unset AWS_PROFILE
export CDK_DEFAULT_ACCOUNT=<your-account-id>
export CDK_DEFAULT_REGION=us-east-1

cdk deploy --require-approval never --outputs-file cdk-outputs.json
```

`cdk deploy` builds the backend Lambda container image, bundles the SPA (`npm ci`
and `npm run build` inside a Node container), and creates roughly 40 resources.
The first deploy takes about 8 to 12 minutes; the CloudFront distribution
dominates. It prints `CloudFrontUrl` when it finishes.

Nothing can sign in until you create a Cognito user. See "Create the first user"
in the [top-level README](../README.md).

## Deploy-time context flags

`cdk.json` sets only two CDK feature flags,
`@aws-cdk/core:newStyleStackSynthesis` and `@aws-cdk/aws-iam:minimizePolicies`.
Everything configurable is a `-c` flag read by `app.py` or `stack.py`:

| Flag | Default | Effect |
| --- | --- | --- |
| `-c prefix=<name>` | `slm-platform` | Name prefix for every named resource, and the Cognito hosted-UI domain prefix |
| `-c alertEmail=you@example.com` | none | Subscribes the address to the cost-alert SNS topic. Also readable from `SLM_ALERT_EMAIL`. Without it the topic, budget and alarm still exist, just with no subscriber |
| `-c monthlyBudgetUsd=<amount>` | `500` | Threshold for both the AWS Budgets budget and the CloudWatch billing alarm |
| `-c notifyFromEmail=<address>` | from `project_config.json`, shipped empty | SES sender for the race-completion email. Empty means `notify.py` skips sending and logs it, rather than failing every race against an unverifiable identity. Also readable from `SLM_NOTIFY_FROM_EMAIL_OVERRIDE` |
| `-c customDomain=app.example.com` | none | Serves the app on your own domain. Creates a DNS-validated ACM certificate and A and AAAA alias records. Requires `customDomainZoneId` |
| `-c customDomainZoneId=<zone-id>` | none | Route 53 hosted zone ID for `customDomain`. The stack raises a `ValueError` if `customDomain` is set without it |

## Training images

The GPU training image is too large for CDK Docker bundling, so a CodeBuild
project builds and pushes it. `cdk deploy` starts one build per tier through an
`AwsCustomResource` whose physical ID is keyed to the hash of `container/`, so
the build fires on the first deploy and again whenever the Dockerfile or
entrypoint changes, but not on unrelated stack updates. `StartBuild` returns
immediately, so a slow or failing image build cannot roll the stack back.

Two tiers ship: `0.9.4` (stable, and the default `SLM_TRAINING_IMAGE_URI`) and
`0.9.5` (latest, transformers v5). One Dockerfile builds both; the LLaMA-Factory
base tag is a build argument. A model picks its tier through
`ModelSpec.image_tag`, resolved by `backend/app/aws_config.image_tiers()`, so a
new stack lands alongside the proven one instead of replacing it. Keep
`IMAGE_TIERS` in `stack.py` in sync with `IMAGE_TIER_TAGS` in
`backend/app/aws_config.py`.

A build takes roughly 10 to 20 minutes on the LARGE compute type, which has a
128 GB disk. The app UI works immediately after deploy; the first training run is
what needs the image. Rebuild from the Docker images page in the UI, or:

```bash
aws codebuild start-build \
  --project-name <OutTrainingImageBuildProject output> --region us-east-1
```

Each build also introspects the image it just pushed for its transformers
architecture registry and LLaMA-Factory template list, and writes that manifest
to `s3://<data-bucket>/slm-platform/image-meta/<tag>.json`. The Model Catalog's
"find new models" diffs those manifests, so it never needs a GPU to work out
which architectures an image supports.

## Identity and config

The app's AWS identity is the Lambda execution role CDK creates. There are no
credentials in the tree or in the deployment. Account, region, bucket, role ARN
and the training-image URI reach the Lambdas as environment variables set at
deploy time, and the Settings page reflects them. Nothing to set up.

`apply_outputs.py` merges the deploy outputs into `backend/data/config.json` so a
locally running backend points at the deployed resources:

```bash
cdk deploy --outputs-file cdk-outputs.json
python3 apply_outputs.py cdk-outputs.json
```

It writes only the five Settings fields (region, account, bucket, roleArn,
imageUri), each read from the corresponding stack output via the
`FIELD_BY_OUTPUT` map in that script, and preserves any other keys already in the
file.

## Stack outputs

`CloudFrontUrl`, `OutApiEndpoint`, `OutCostAlertTopic`, `OutDataBucket`,
`OutTrainingImageUri`, `OutTrainingImageBuildProject` and a per-tier variant of
it, `OutUserPoolId`, `OutUserPoolClientId`, `OutCognitoDomain`, `OutRegion`,
`OutAccount`, `OutSageMakerRoleArn`.

`OutCognitoDomain` is derived from the CloudFormation stack id, not the AWS
account id: the hosted-UI prefix becomes a public DNS name and is also served in
the unauthenticated `/config.json`, so it must not carry the account number.

## Teardown

```bash
cdk destroy
```

Both S3 buckets auto-empty, and the ECR repository empties on delete. Several
things are created outside this stack and survive `cdk destroy`, including any
SageMaker endpoint from an export bundle, the AgentCore runtime, the pull-through
cache repositories and every CloudWatch log group. The full checklist is in the
[top-level README](../README.md). SageMaker training job records cannot be
deleted through any AWS API; use the in-app reset cutoff to hide them from the
UI.

## Notes

- Authentication is a plain Amazon Cognito user pool with the hosted UI. An
  `HttpUserPoolAuthorizer` is attached to `/api/{proxy+}`, so every route except
  `/api/health` returns 401 without a valid Cognito JWT, and per-user state
  isolation keys off a stable username claim the authorizer forwards — not the
  `sub` claim, which regenerates for external-IdP users (see
  `_stable_tenant_key` in `backend/app/tenancy.py`). The SPA uses the
  implicit grant (`response_type=token`), so the hosted UI returns the `id_token`
  in the URL hash and the SPA needs no backend token exchange
  (`frontend/src/auth.ts`). Self-signup is disabled: users are created by an
  administrator. If the SPA cannot fetch `/config.json` it runs with
  authentication off, which is the local-development path.
- The SPA calls relative `/api` paths, so CloudFront same-origins the API. No API
  URL is baked into the build, and a redeploy needs no rebuild for that reason.
- The distribution declares no custom error responses. The SPA uses hash routing,
  so `/` is the only document path and `default_root_object` handles it. A
  404-to-index remap would apply distribution-wide and would turn legitimate API
  4xx responses into HTML, breaking the SPA's JSON parsing.
- The WebACL sets the managed `SizeRestrictions_BODY` rule to Count instead of
  Block, because that rule rejects any body over 8 KB and would break dataset and
  eval JSONL uploads. The FastAPI app enforces its own cap instead
  (`MAX_UPLOAD_BYTES` in `backend/app/main.py`, currently 100 MB). The rest of the
  common rule set stays active.
- Long backend operations, such as a frontier baseline over many rows or a
  leaderboard scan, run inside the API Lambda under API Gateway's 29-second
  integration timeout. The worker Lambda exists for the ones that exceed it, with
  a 15-minute timeout and an async invoke plus poll from the API.
- The per-tenant and cross-tenant race caps are Lambda environment variables in
  `stack.py`: `SLM_MAX_CONCURRENT_RACES`, `SLM_MAX_MODELS_PER_RACE` and
  `SLM_MAX_GLOBAL_CONCURRENT_RACES`. `SLM_MAX_MODELS_PER_RACE` must stay at or
  above the guided flow's ceiling of 16, or that ceiling silently clamps.
- The stack renders the AWS Solution ID into the CloudFormation stack description
  and attaches a user-agent suffix to the Lambdas' AWS SDK calls. See
  "Anonymized data collection" in the [top-level README](../README.md).
