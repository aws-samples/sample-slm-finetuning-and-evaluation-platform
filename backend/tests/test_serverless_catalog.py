# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for serverless model discovery + the runtime serverless-tag overlay.

The live SageMaker Public Hub query (list_customizable_hub_models) is the network
boundary — mocked here, mirroring how test_releases mocks list_upstream_releases.
Covers: keyword parsing, classification (untagged matches / stale tags / new
candidates), the config.json overlay register/merge (a discovered tag reaches the
ModelSpec gates), the hub-id trust boundary, and the endpoints.
"""
from __future__ import annotations

import pytest


def test_parse_keywords_requires_capability_and_recipe():
    from app.serverless_catalog import _parse_keywords

    # Customizable: capability + at least one finetuning recipe.
    out = _parse_keywords([
        "@capability:customization", "@recipe:finetuning_sft_lora",
        "@recipe:finetuning_dpo_lora", "@huggingface-id:qwen/qwen3-4b",
    ])
    assert out == {"hf": "qwen/qwen3-4b", "recipes": ["dpo_lora", "sft_lora"]}
    # Capability but NO finetuning recipe → not customizable.
    assert _parse_keywords(["@capability:customization", "@recipe:evaluation_x"]) is None
    # Recipe but no customization capability → not customizable.
    assert _parse_keywords(["@recipe:finetuning_sft_lora"]) is None
    # Missing hf id is tolerated (empty string).
    out2 = _parse_keywords(["@capability:customization", "@recipe:finetuning_sft_lora"])
    assert out2 == {"hf": "", "recipes": ["sft_lora"]}


def test_validate_hub_id_rejects_malicious():
    from app.serverless_catalog import _validate_hub_id

    assert _validate_hub_id("huggingface-reasoning-qwen3-06b") == "huggingface-reasoning-qwen3-06b"
    for bad in ["../etc/passwd", "a b", "Foo;rm -rf", "", "UPPER_not_allowed_$", "x" * 200]:
        with pytest.raises(ValueError):
            _validate_hub_id(bad)


def test_discover_classifies_untagged_match(monkeypatch, temp_store):
    """A catalog row that is customizable on the hub but has no tag → untaggedMatch.
    qwen2.5-3b is in the catalog (no serverless tag) and its hub name is guessable."""
    from app import serverless_catalog as sc

    monkeypatch.setattr(sc, "list_customizable_hub_models", lambda: [
        # matches qwen2.5-3b by the name guess (huggingface-llm-qwen2-5-3b-instruct)
        {"name": "huggingface-llm-qwen2-5-3b-instruct", "hf": "Qwen/Qwen2.5-3B-Instruct",
         "recipes": ["sft_lora", "dpo_lora"]},
        # already tagged statically (qwen3-4b) — must NOT appear as untagged
        {"name": "huggingface-reasoning-qwen3-4b", "hf": "qwen/qwen3-4b",
         "recipes": ["sft_lora", "rlvr_lora"]},
    ])
    out = sc.discover_serverless_models()
    untagged_ids = {m["id"] for m in out["untaggedMatches"]}
    assert "qwen2.5-3b" in untagged_ids
    assert "qwen3-4b" not in untagged_ids  # already tagged, not re-suggested
    match = next(m for m in out["untaggedMatches"] if m["id"] == "qwen2.5-3b")
    assert match["hubId"] == "huggingface-llm-qwen2-5-3b-instruct"
    assert "sft_lora" in match["recipes"]
    assert out["customizableCount"] == 2


def test_discover_flags_new_candidate_and_stale_tag(monkeypatch, temp_store):
    from app import serverless_catalog as sc

    # Hub lists a model with NO catalog row → newCandidate. And it does NOT list
    # qwen3-4b's tagged id (so that static tag is stale per this hub snapshot).
    monkeypatch.setattr(sc, "list_customizable_hub_models", lambda: [
        {"name": "nova-textgeneration-micro", "hf": "", "recipes": ["sft_lora"]},
    ])
    out = sc.discover_serverless_models()
    cand_ids = {c["hubId"] for c in out["newCandidates"]}
    assert "nova-textgeneration-micro" in cand_ids
    # qwen3-4b is statically tagged 'huggingface-reasoning-qwen3-4b' which is NOT
    # in this hub snapshot → stale.
    stale_ids = {s["id"] for s in out["staleTags"]}
    assert "qwen3-4b" in stale_ids


def test_discover_graceful_when_hub_unreachable(monkeypatch, temp_store):
    from app import serverless_catalog as sc

    monkeypatch.setattr(sc, "list_customizable_hub_models", lambda: [])
    out = sc.discover_serverless_models()
    assert out["customizableCount"] == 0
    assert out["untaggedMatches"] == [] and out["newCandidates"] == []
    assert out["allModels"] == []
    assert "could not list" in out["note"]


def test_discover_flat_allmodels_states(monkeypatch, temp_store):
    """The flat allModels list classifies EVERY customizable hub model into one of
    enabled | addable | onboardable | unavailable, for a browsable pick-and-add
    table. A statically-tagged catalog row → enabled; an untagged catalog row →
    addable; a new HF-backed candidate → onboardable; a no-HF (Nova) → unavailable."""
    from app import serverless_catalog as sc

    monkeypatch.setattr(sc, "list_customizable_hub_models", lambda: [
        # statically tagged in CATALOG (qwen3-4b → enabled)
        {"name": "huggingface-reasoning-qwen3-4b", "hf": "qwen/qwen3-4b", "recipes": ["sft_lora"]},
        # in catalog, untagged (qwen2.5-3b → addable, matched by name guess)
        {"name": "huggingface-llm-qwen2-5-3b-instruct", "hf": "Qwen/Qwen2.5-3B-Instruct", "recipes": ["sft_lora"]},
        # new HF-backed candidate (onboardable)
        {"name": "deepseek-llm-r1-distill-qwen-32b", "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "recipes": ["sft_lora"]},
        # no HF repo → unavailable
        {"name": "nova-textgeneration-micro", "hf": "", "recipes": ["sft_lora"]},
    ])
    monkeypatch.setattr(sc, "describe_hub_hf", lambda name: {})  # no fallback recovery for Nova
    out = sc.discover_serverless_models()
    by_hub = {m["hubId"]: m for m in out["allModels"]}
    assert by_hub["huggingface-reasoning-qwen3-4b"]["state"] == "enabled"
    assert by_hub["huggingface-llm-qwen2-5-3b-instruct"]["state"] == "addable"
    assert by_hub["deepseek-llm-r1-distill-qwen-32b"]["state"] == "onboardable"
    assert by_hub["nova-textgeneration-micro"]["state"] == "unavailable"
    # Every customizable model appears exactly once in the flat list.
    assert len(out["allModels"]) == 4


def test_register_overlay_persists_and_reaches_modelspec(temp_store):
    """Registering a tag via the overlay makes get_model return a ModelSpec whose
    serverless_model_id is set — so launch/verification gates honor it. And
    list_models surfaces the engine + serverlessModelId."""
    from app import catalog
    from app.serverless_catalog import register_serverless_model_id, serverless_overlay

    # qwen2.5-3b has no static serverless tag.
    assert catalog.get_model("qwen2.5-3b").serverless_model_id == ""

    overlay = register_serverless_model_id("qwen2.5-3b", "huggingface-llm-qwen2-5-3b-instruct")
    assert overlay["qwen2.5-3b"] == "huggingface-llm-qwen2-5-3b-instruct"
    assert serverless_overlay()["qwen2.5-3b"] == "huggingface-llm-qwen2-5-3b-instruct"

    # The OVERLAY reaches the ModelSpec (not just the dict) → gates honor it.
    spec = catalog.get_model("qwen2.5-3b")
    assert spec.serverless_model_id == "huggingface-llm-qwen2-5-3b-instruct"
    # And the served catalog row shows the serverless engine.
    row = next(m for m in catalog.list_models() if m["id"] == "qwen2.5-3b")
    assert "sagemaker_serverless" in row["engines"]
    assert row["serverlessModelId"] == "huggingface-llm-qwen2-5-3b-instruct"


def test_static_tag_is_the_floor_overlay_never_overrides(temp_store):
    """A hand-curated static tag must win over an overlay entry (the overlay only
    fills empties)."""
    from app import catalog
    from app.serverless_catalog import register_serverless_model_id

    # qwen3-4b is statically tagged. Try to override via overlay.
    register_serverless_model_id("qwen3-4b", "some-other-hub-id")
    assert catalog.get_model("qwen3-4b").serverless_model_id == "huggingface-reasoning-qwen3-4b"


def test_register_rejects_unknown_model_and_bad_hub_id(temp_store):
    from app.serverless_catalog import register_serverless_model_id

    with pytest.raises(ValueError):
        register_serverless_model_id("no-such-model", "huggingface-llm-x")
    with pytest.raises(ValueError):
        register_serverless_model_id("qwen2.5-3b", "../bad")


def test_unregister_clears_overlay_not_static(temp_store):
    from app import catalog
    from app.serverless_catalog import register_serverless_model_id, unregister_serverless_model_id

    register_serverless_model_id("qwen2.5-3b", "huggingface-llm-qwen2-5-3b-instruct")
    assert catalog.get_model("qwen2.5-3b").serverless_model_id != ""
    unregister_serverless_model_id("qwen2.5-3b")
    assert catalog.get_model("qwen2.5-3b").serverless_model_id == ""
    # Clearing a non-overlay (static) id is a harmless no-op.
    unregister_serverless_model_id("qwen3-4b")
    assert catalog.get_model("qwen3-4b").serverless_model_id == "huggingface-reasoning-qwen3-4b"


def test_three_static_tags_present():
    """The 3 hand-added static tags are wired."""
    from app.catalog import get_model

    assert get_model("qwen3-0.6b").serverless_model_id == "huggingface-reasoning-qwen3-06b"
    assert get_model("llama-3.2-1b").serverless_model_id == "meta-textgeneration-llama-3-2-1b-instruct"
    assert get_model("llama-3.1-8b").serverless_model_id == "meta-textgeneration-llama-3-1-8b-instruct"


def test_serverless_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    assert "/api/models/serverless-candidates" in paths
    assert "/api/models/{model_id}/serverless-tag" in paths


def test_set_serverless_tag_endpoint(temp_store, monkeypatch):
    """POST applies the overlay; blank hubId clears it."""
    import app.main as m
    from app import catalog

    req = m.ServerlessTagRequest(hubId="huggingface-llm-qwen2-5-3b-instruct")
    out = m.set_serverless_tag("qwen2.5-3b", req)
    assert out["ok"] and out["hubId"] == "huggingface-llm-qwen2-5-3b-instruct"
    assert catalog.get_model("qwen2.5-3b").serverless_model_id == "huggingface-llm-qwen2-5-3b-instruct"

    cleared = m.set_serverless_tag("qwen2.5-3b", m.ServerlessTagRequest(hubId=""))
    assert cleared["ok"]
    assert catalog.get_model("qwen2.5-3b").serverless_model_id == ""


def test_set_serverless_tag_endpoint_rejects_bad(temp_store):
    import app.main as m
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        m.set_serverless_tag("qwen2.5-3b", m.ServerlessTagRequest(hubId="../evil"))
    assert ei.value.status_code == 400


# --- onboard a NEW serverless candidate end-to-end (custom model carries the tag) --

def test_custom_model_carries_serverless_tag(temp_store):
    """A custom (onboarded) model can now hold a serverless_model_id — so a brand-new
    serverless-customizable hub model can be added WITH the serverless engine on.
    Previously save_custom_model/_spec_from_dict dropped it."""
    from app import catalog
    from app.onboard import get_custom_model, save_custom_model

    save_custom_model({
        "id": "gpt-oss-120b", "displayName": "GPT-OSS 120B", "hfModelId": "openai/gpt-oss-120b",
        "template": "gpt_oss", "family": "GPT-OSS", "paramsB": 120.0,
        "defaultCutoffLen": 4096, "suggestedInstance": "ml.g6e.12xlarge",
        "serverlessModelId": "openai-reasoning-gpt-oss-120b",
    })
    # Round-trips through the store as a ModelSpec WITH the tag.
    spec = get_custom_model("gpt-oss-120b")
    assert spec.serverless_model_id == "openai-reasoning-gpt-oss-120b"
    # And the served catalog shows the serverless engine for the new row.
    row = next(m for m in catalog.list_models() if m["id"] == "gpt-oss-120b")
    assert "sagemaker_serverless" in row["engines"]
    assert row.get("custom") is True


def test_save_custom_model_validates_serverless_id(temp_store):
    """A malformed serverless id is rejected at the trust boundary on save."""
    from app.onboard import save_custom_model

    with pytest.raises(ValueError):
        save_custom_model({
            "id": "bad", "displayName": "Bad", "hfModelId": "x/y", "template": "qwen",
            "paramsB": 1.0, "defaultCutoffLen": 2048, "suggestedInstance": "ml.g5.2xlarge",
            "serverlessModelId": "../evil",
        })


def test_add_custom_model_endpoint_forwards_serverless_tag(temp_store):
    """POST /api/models/custom accepts serverlessModelId and persists it."""
    import app.main as m
    from app import catalog

    req = m.SaveModelRequest(
        id="ds-distill-32b", displayName="DeepSeek-R1-Distill-Qwen 32B",
        hfModelId="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", template="deepseekr1",
        family="DeepSeek-R1-Distill", paramsB=32.0, defaultCutoffLen=4096,
        suggestedInstance="ml.g6e.12xlarge", serverlessModelId="deepseek-llm-r1-distill-qwen-32b",
    )
    out = m.add_custom_model(req)
    assert out["ok"]
    assert catalog.get_model("ds-distill-32b").serverless_model_id == "deepseek-llm-r1-distill-qwen-32b"


def test_discover_flags_candidate_onboardability(monkeypatch, temp_store):
    """newCandidates carry onboardable + reason: a text model with an HF id is
    one-click; a VLM or a no-HF (Nova) model is listed but not onboardable."""
    from app import serverless_catalog as sc

    monkeypatch.setattr(sc, "list_customizable_hub_models", lambda: [
        {"name": "deepseek-llm-r1-distill-qwen-32b", "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
         "recipes": ["sft_lora"]},  # onboardable
        {"name": "huggingface-vlm-qwen3-5-9b", "hf": "qwen/qwen3.5-9b", "recipes": ["sft_lora"]},  # VLM
        {"name": "nova-textgeneration-micro", "hf": "", "recipes": ["sft_lora"]},  # no HF repo
    ])
    out = sc.discover_serverless_models()
    by_hub = {c["hubId"]: c for c in out["newCandidates"]}
    assert by_hub["deepseek-llm-r1-distill-qwen-32b"]["onboardable"] is True
    assert by_hub["huggingface-vlm-qwen3-5-9b"]["onboardable"] is False
    assert "vision-language" in by_hub["huggingface-vlm-qwen3-5-9b"]["reason"]
    assert by_hub["nova-textgeneration-micro"]["onboardable"] is False
    assert "Hugging Face" in by_hub["nova-textgeneration-micro"]["reason"]


def test_discover_describe_fallback_recovers_empty_hf_id(monkeypatch, temp_store):
    """A candidate whose list-summary @huggingface-id is EMPTY (e.g. qwen2.5-32b)
    must NOT be mislabeled 'no HF repo' — describe_hub_content recovers it from the
    content document, making it onboardable. Authoritative modalities flag VLMs."""
    from app import serverless_catalog as sc

    monkeypatch.setattr(sc, "list_customizable_hub_models", lambda: [
        {"name": "huggingface-llm-qwen2-5-32b-instruct", "hf": "", "recipes": ["sft_lora"]},
        {"name": "huggingface-vlm-qwen3-5-9b", "hf": "", "recipes": ["sft_lora"]},
    ])

    def fake_describe(name):
        if name == "huggingface-llm-qwen2-5-32b-instruct":
            return {"hf": "Qwen/Qwen2.5-32B-Instruct", "modalities": ["text"]}
        return {"hf": "Qwen/Qwen3.5-9B", "modalities": ["text", "image"]}  # a VLM

    monkeypatch.setattr(sc, "describe_hub_hf", fake_describe)
    out = sc.discover_serverless_models()
    by_hub = {c["hubId"]: c for c in out["newCandidates"]}
    # The empty-id text model is now recovered + onboardable.
    q = by_hub["huggingface-llm-qwen2-5-32b-instruct"]
    assert q["hf"] == "Qwen/Qwen2.5-32B-Instruct" and q["onboardable"] is True
    # The image-modality model is flagged VLM (not onboardable) via modalities.
    v = by_hub["huggingface-vlm-qwen3-5-9b"]
    assert v["onboardable"] is False and "vision-language" in v["reason"]
