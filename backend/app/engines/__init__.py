# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Training ENGINES — the additive multi-engine seam.

The platform's original (and default) engine is LLaMA-Factory: a frozen ECR
image trained as a SageMaker training job, rendered from a ModelSpec by
render.py. This package lets a model declare a DIFFERENT engine (e.g. SageMaker
serverless model-customization) WITHOUT touching the LLaMA-Factory path.

Design rules (so existing runs + the 327 tests stay byte-identical):
  - `get_engine(name)` returns an Engine. The DEFAULT is "llama_factory", whose
    implementation calls the EXISTING orchestrate functions verbatim — same S3
    keys, same YAML, same job names, same tags.
  - A model with no `engine` field resolves to "llama_factory".
  - Non-default engines are GATED behind a feature flag (SLM_ENABLE_<ENGINE> /
    config.json) so they can ship dark. get_engine raises for a disabled engine.
  - Engine implementations lazy-import any heavy/optional deps INSIDE their
    methods, never at module import, so a missing SDK can't break catalog loading
    or the LLaMA-Factory launch path.
"""

from __future__ import annotations

from .base import Engine, EngineNotEnabled, get_engine, DEFAULT_ENGINE

__all__ = ["Engine", "EngineNotEnabled", "get_engine", "DEFAULT_ENGINE"]
