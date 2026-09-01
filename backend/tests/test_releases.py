# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LLaMA-Factory release tracking + new-model discovery (pure logic, AWS stubbed)."""
import pytest

from app import releases


def test_ver_parsing_and_ordering():
    assert releases._ver("0.9.5") == (0, 9, 5)
    assert releases._ver("0.9.10") > releases._ver("0.9.9")
    assert releases._ver("latest") == ()


def test_tier_name_for():
    assert releases._tier_name_for("0.9.6") == "v096"
    assert releases._tier_name_for("1.0.0") == "v100"


def test_check_for_updates_flags_newer(monkeypatch, temp_store):
    # Upstream has a newer 0.9.6 than our built {0.9.4, 0.9.5}.
    monkeypatch.setattr(releases, "list_upstream_releases", lambda: ["0.9.6", "0.9.5", "0.9.4", "0.9.3"])
    out = releases.check_for_updates()
    assert out["newest"] == "0.9.6"
    assert out["haveNewest"] is False
    assert out["newReleases"] == ["0.9.6"]  # only the actionable (newer) one
    assert "0.9.4" in out["builtTags"] and "0.9.5" in out["builtTags"]


def test_check_for_updates_up_to_date(monkeypatch, temp_store):
    monkeypatch.setattr(releases, "list_upstream_releases", lambda: ["0.9.5", "0.9.4"])
    out = releases.check_for_updates()
    assert out["haveNewest"] is True
    assert out["newReleases"] == []


def test_discover_new_models_diffs_manifests(monkeypatch, temp_store):
    # base image supports lfm2 already; new image adds lfm2_moe + qwen3_5.
    metas = {
        "0.9.4": {"model_types": ["llama", "qwen2", "lfm2"], "templates": ["llama3", "qwen"]},
        "0.9.6": {
            "model_types": ["llama", "qwen2", "lfm2", "lfm2_moe", "qwen3_5", "some_unmapped_arch"],
            "templates": ["llama3", "qwen", "lfm2"],
            "transformers": "5.7.0",
        },
    }
    monkeypatch.setattr(releases, "_read_image_meta", lambda tag: metas.get(tag))
    out = releases.discover_new_models("0.9.6", base_tag="0.9.4")
    assert out["baseTag"] == "0.9.4"
    assert set(out["newArchitectures"]) == {"lfm2_moe", "qwen3_5", "some_unmapped_arch"}
    assert out["newTemplates"] == ["lfm2"]
    # Mapped arches become suggestions with HF repos; unmapped ones don't — AND any
    # repo already in the catalog is filtered out ("Newly supported" = models you
    # don't have yet). lfm2_moe's only suggested repo (LiquidAI/LFM2-8B-A1B) is a
    # built-in catalog model, so that arch is dropped; qwen3_5's repo (Qwen/Qwen3.5-4B)
    # is NOT in the catalog, so it remains.
    sugg_arches = {s["architecture"] for s in out["suggestions"]}
    assert sugg_arches == {"qwen3_5"}
    assert all("LFM2-8B-A1B" not in r for s in out["suggestions"] for r in s["repos"])
    # The hidden already-cataloged model is surfaced in the note (no silent hide).
    assert "already in your catalog" in out["note"]


def test_discover_handles_missing_manifest(monkeypatch, temp_store):
    monkeypatch.setattr(releases, "_read_image_meta", lambda tag: None)
    out = releases.discover_new_models("0.9.9")
    assert out["suggestions"] == []
    assert "no capability manifest" in out["note"]


def test_validate_tag_accepts_version_like():
    assert releases._validate_tag("0.9.5") == "0.9.5"
    assert releases._validate_tag("  0.8.5.post1  ", "vllm_version") == "0.8.5.post1"


@pytest.mark.parametrize(
    "bad",
    [
        "latest; rm -rf /",   # command injection
        "0.9.5 && curl evil", # shell chaining
        "../secret",          # path traversal
        "tag$(whoami)",       # command substitution
        "a/b",                # slash (alternate image ref)
        "",                   # empty
    ],
)
def test_validate_tag_rejects_malicious(bad):
    # ASI-02 / CWE-918: a tag must never carry shell/registry metacharacters
    # into a CodeBuild override or an image reference.
    with pytest.raises(ValueError):
        releases._validate_tag(bad)


def test_discover_new_models_rejects_bad_tag(temp_store):
    with pytest.raises(ValueError):
        releases.discover_new_models("../../etc/passwd")


def test_register_image_tier_persists(temp_store):
    from app.aws_config import image_tiers, register_image_tier

    tiers = register_image_tier("v096", "0.9.6")
    assert tiers["v096"] == "0.9.6"
    # round-trips via config.json
    assert image_tiers()["v096"] == "0.9.6"
    # built-ins remain the floor
    assert image_tiers()["stable"] == "0.9.4"


def test_release_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in [
        "/api/images/check-updates",
        "/api/images/build-release",
        "/api/images/{image_tag}/new-models",
    ]:
        assert p in paths, f"missing route {p}"
