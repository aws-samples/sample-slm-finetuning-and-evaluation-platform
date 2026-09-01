# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-28

Initial public release of the SLM Finetuning and Evaluation Platform.

### Added

- FastAPI backend holding the platform logic, with a storage abstraction that runs against local
  disk for development and Amazon S3 when deployed to AWS Lambda.
- Fine-tuning orchestration on Amazon SageMaker driven by a single frozen LLaMA-Factory training
  image, so adding a model is a catalog entry rather than new code.
- Parallel fine-tuning races: several models trained on the same split and scored on the same
  held-out eval set, advanced by a reconcile loop that survives restarts.
- Offline batch evaluation with vLLM inside the same SageMaker job, with no model hosting.
- Frontier Claude baselines through Amazon Bedrock, scored on the same held-out eval set.
- Split-scoped leaderboard comparing fine-tuned models against those baselines.
- Three Strands agents on Amazon Bedrock AgentCore Runtime: dataset investigation, failed-run
  diagnosis, and a model recommendation over the leaderboard.
- React, TypeScript and Cloudscape single-page frontend built with Vite.
- AWS CDK application deploying the stack: CloudFront with WAF, Amazon Cognito, Amazon API Gateway,
  AWS Lambda, Amazon S3, AWS CodeBuild and Amazon ECR.
- pytest suite for the backend with AWS calls mocked.
