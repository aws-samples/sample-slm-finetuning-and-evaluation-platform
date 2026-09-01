# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Auto-onboard models from a Hugging Face id (Tier 1)."""
import pytest

from app import onboard


def test_template_match_by_architecture():
    assert onboard._match_template(["Qwen3ForCausalLM"], "qwen3")[0] == "qwen3"
    assert onboard._match_template(["LlamaForCausalLM"], "llama")[0] == "llama3"
    assert onboard._match_template(["MistralForCausalLM"], "mistral")[0] == "mistral"
    # Unknown architecture → no confident match (None), not a wrong guess.
    tmpl, how = onboard._match_template(["TotallyNovelArch"], "novel")
    assert tmpl is None and "no confident" in how


def test_template_match_deepseek_r1_distill_by_repo_name():
    # DeepSeek-R1-Distill models report their BASE architecture (LlamaForCausalLM /
    # Qwen2ForCausalLM), but were distilled with DeepSeek's R1 reasoning format —
    # so the repo-name rule must override the base-arch match to 'deepseekr1'
    # (mirrors the built-in catalog). Without this they'd wrongly get llama3/qwen.
    assert onboard._match_template(
        ["LlamaForCausalLM"], "llama",
        repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")[0] == "deepseekr1"
    assert onboard._match_template(
        ["Qwen2ForCausalLM"], "qwen2",
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")[0] == "deepseekr1"
    # A PLAIN Llama/Qwen (not an R1-distill) is unaffected — still base-arch match.
    assert onboard._match_template(
        ["LlamaForCausalLM"], "llama", repo="meta-llama/Llama-3.1-8B-Instruct")[0] == "llama3"
    assert onboard._match_template(
        ["Qwen3ForCausalLM"], "qwen3", repo="Qwen/Qwen3-4B")[0] == "qwen3"


def test_template_match_qwen3_nothink_and_qwen35():
    # Qwen3-*-Instruct-2507 is NON-thinking → qwen3_nothink (NOT the thinking qwen3,
    # which would inject empty <think></think> tokens with loss during SFT).
    assert onboard._match_template(
        ["Qwen3ForCausalLM"], "qwen3", repo="Qwen/Qwen3-4B-Instruct-2507")[0] == "qwen3_nothink"
    # Plain thinking Qwen3 stays qwen3.
    assert onboard._match_template(
        ["Qwen3ForCausalLM"], "qwen3", repo="Qwen/Qwen3-8B")[0] == "qwen3"
    # Qwen3.5 family → dedicated qwen3_5 template (distinct from qwen3).
    assert onboard._match_template(
        ["Qwen3_5ForConditionalGeneration"], "qwen3_5", repo="Qwen/Qwen3.5-4B")[0] == "qwen3_5"
    # All the corrected templates must be in the engine's KNOWN set (else jobs 400).
    for t in ("qwen3_nothink", "qwen3_5", "qwen3_5_nothink"):
        assert onboard.is_known_template(t), f"{t} must be in KNOWN_TEMPLATES"


def test_known_template_guard():
    assert onboard.is_known_template("qwen3") is True
    assert onboard.is_known_template("not_a_real_template") is False
    assert onboard.is_known_template(None) is False


def test_params_billions_prefers_safetensors():
    meta = {"safetensors": {"total": 4_000_000_000}}
    assert onboard._params_billions(meta, {}) == 4.0
    # falls back to a dim-based estimate when safetensors metadata is absent
    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "vocab_size": 150000}
    est = onboard._params_billions({}, cfg)
    assert est > 0


def test_slug_is_safe_catalog_id():
    assert onboard._slug("Qwen/Qwen3-4B-Instruct-2507") == "qwen3-4b-instruct-2507"


def test_validate_repo_accepts_legit_ids():
    # Real HF repo ids pass through unchanged (whitespace trimmed).
    assert onboard._validate_repo("Qwen/Qwen3-4B-Instruct-2507") == "Qwen/Qwen3-4B-Instruct-2507"
    assert onboard._validate_repo("  meta-llama/Llama-3.1-8B  ") == "meta-llama/Llama-3.1-8B"


@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",          # path traversal
        "Qwen/Qwen3@evil.com",       # host override / embedded credentials
        "https://evil.com/x",        # protocol smuggling
        "Qwen/Q\r\nHost: evil",      # CRLF injection
        "org/name?x=1",              # query injection
        "no-slash",                  # not org/name
        "a/b/c",                     # too many segments
        "",                          # empty
    ],
)
def test_validate_repo_rejects_malicious_ids(bad):
    # ASI-02 / CWE-918: malformed repo ids must never reach the HF URL.
    with pytest.raises(ValueError):
        onboard._validate_repo(bad)
    assert onboard._slug("org/Weird Name!!") == "weird-name"


def test_custom_model_roundtrip_and_catalog_merge(temp_store):
    from app.catalog import get_model, list_models

    spec = {
        "id": "my-custom-7b",
        "displayName": "My Custom 7B",
        "hfModelId": "org/my-custom-7b",
        "template": "qwen",
        "family": "Custom",
        "paramsB": 7.0,
        "defaultCutoffLen": 4096,
        "suggestedInstance": "ml.g5.8xlarge",
        "gated": False,
    }
    onboard.save_custom_model(spec)
    # Resolvable via the unified catalog lookup + appears in list_models.
    m = get_model("my-custom-7b")
    assert m is not None and m.template == "qwen" and m.params_b == 7.0
    assert any(x["id"] == "my-custom-7b" and x.get("custom") for x in list_models())
    # Delete removes it.
    assert onboard.delete_custom_model("my-custom-7b") is True
    assert get_model("my-custom-7b") is None


def test_save_rejects_missing_template(temp_store):
    import pytest

    with pytest.raises(ValueError):
        onboard.save_custom_model({"id": "x", "displayName": "X", "hfModelId": "o/x",
                                   "template": None, "paramsB": 1.0, "defaultCutoffLen": 2048,
                                   "suggestedInstance": "ml.g5.2xlarge"})


def test_onboard_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in ["/api/models/probe", "/api/models/custom", "/api/models/{model_id}/smoke-test"]:
        assert p in paths
