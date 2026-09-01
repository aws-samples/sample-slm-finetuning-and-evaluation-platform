# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Baseline model catalog (multi-provider) + base-eval leaderboard labeling."""
from app.baseline import BASELINE_MODELS, DEFAULT_BASELINE, baseline_model
from app.leaderboard import _hf_to_label


def test_every_baseline_has_a_provider():
    for key, spec in BASELINE_MODELS.items():
        assert spec.get("provider"), f"{key} missing provider"
        assert spec.get("modelId") and spec.get("label")


def test_providers_span_multiple_vendors():
    provs = {v["provider"] for v in BASELINE_MODELS.values()}
    # Anthropic + Amazon were original; Meta/Mistral/Cohere added.
    assert {"Anthropic", "Amazon", "Meta", "Mistral", "Cohere"} <= provs


def test_default_baseline_resolves():
    assert DEFAULT_BASELINE in BASELINE_MODELS
    assert baseline_model(None)["modelId"] == BASELINE_MODELS[DEFAULT_BASELINE]["modelId"]
    # Unknown key falls back to default (not a crash).
    assert baseline_model("nope")["label"] == BASELINE_MODELS[DEFAULT_BASELINE]["label"]


def test_hf_to_label_uses_catalog_id_then_falls_back():
    # A base-eval row now labels by the CATALOG id (matching its fine-tuned
    # sibling row) when the HF id is a known catalog model — so base and
    # fine-tuned don't look like two different models on the leaderboard.
    assert _hf_to_label("Qwen/Qwen3-0.6B") == "qwen3-0.6b"
    # Not in the catalog → fall back to the HF repo name (org stripped).
    assert _hf_to_label("some-org/Mystery-7B-Instruct") == "Mystery-7B-Instruct"
    assert _hf_to_label("") == "base model"
