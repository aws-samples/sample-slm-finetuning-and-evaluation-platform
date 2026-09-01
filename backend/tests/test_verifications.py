# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Verification status classification — gated access-denied vs incompatible."""
import pytest

from app import verifications as v


def test_is_access_error_matches_gated_signatures():
    assert v._is_access_error("huggingface_hub.errors.GatedRepoError: 403 ...")
    assert v._is_access_error("Access to model meta-llama/Llama-3.2-1B is restricted")
    assert v._is_access_error("you are not in the authorized list")
    # Real incompatibility / unrelated errors are NOT access errors.
    assert not v._is_access_error("ImportError: cannot import name LossKwargs")
    assert not v._is_access_error("CUDA out of memory")
    assert not v._is_access_error(None)


def test_classify_failure_access_denied_from_reason(monkeypatch):
    # Reason itself carries the signature → ACCESS_DENIED, no log fetch needed.
    monkeypatch.setattr(v, "_is_access_error", lambda s: "gated" in (s or "").lower())
    status, reason = v.classify_failure("job-x", "GatedRepoError: gated repo")
    assert status == v.ACCESS_DENIED
    assert "license" in reason.lower() and "hugging face" in reason.lower()


def test_classify_failure_reads_logs_when_reason_bare(monkeypatch):
    # Bare FailureReason (the real-world case) → must consult the log tail.
    import app.orchestrate as orch
    monkeypatch.setattr(orch, "job_log_tail",
                        lambda j, limit=40: "Cannot access gated repo ... 403 Forbidden")
    status, reason = v.classify_failure("job-y", "AlgorithmError: , exit code: 1")
    assert status == v.ACCESS_DENIED


def test_classify_failure_real_incompatibility(monkeypatch):
    import app.orchestrate as orch
    monkeypatch.setattr(orch, "job_log_tail", lambda j, limit=40: "ImportError: LossKwargs")
    status, reason = v.classify_failure("job-z", "AlgorithmError: , exit code: 1")
    assert status == v.INCOMPATIBLE
    assert reason == "AlgorithmError: , exit code: 1"  # original reason preserved


def test_access_denied_is_a_valid_status(temp_store):
    # set_status accepts ACCESS_DENIED and it ranks with incompatible (overridable
    # by a later verified).
    rec = v.set_status("m1", "stable", v.ACCESS_DENIED, reason="accept license")
    assert rec["status"] == v.ACCESS_DENIED
    # A later success wins.
    rec2 = v.set_status("m1", "stable", v.VERIFIED)
    assert rec2["status"] == v.VERIFIED
    # And access-denied does NOT override a prior verified without force.
    rec3 = v.set_status("m1", "stable", v.ACCESS_DENIED, reason="x")
    assert rec3["status"] == v.VERIFIED


# --- method-keyed verification (LoRA vs QLoRA proven independently) ---


def test_lora_proof_does_not_leak_to_qlora(temp_store):
    v.set_status("m1", "stable", v.VERIFIED, job_name="j-lora")
    assert v.get_status("m1", "stable")["status"] == v.VERIFIED  # LoRA (bare key)
    # QLoRA on the same model+tier is a DIFFERENT key — still untested.
    assert v.get_status("m1", "stable", "qlora")["status"] == v.UNTESTED


def test_qlora_failure_does_not_affect_lora(temp_store):
    v.set_status("m1", "stable", v.VERIFIED, job_name="j-lora")
    v.set_status("m1", "stable", v.INCOMPATIBLE, method="qlora", reason="4-bit load failed")
    assert v.get_status("m1", "stable", "qlora")["status"] == v.INCOMPATIBLE
    assert v.get_status("m1", "stable")["status"] == v.VERIFIED  # LoRA untouched


def test_lora_stored_under_bare_key_qlora_under_composite(temp_store):
    v.set_status("m1", "stable", v.VERIFIED)
    v.set_status("m1", "stable", v.VERIFIED, method="qlora")
    raw = v._all()["m1"]
    assert set(raw.keys()) == {"stable", "stable::qlora"}


def test_list_pending_carries_method(temp_store):
    v.set_pending("m1", "stable", "j1")  # LoRA (default)
    v.set_pending("m1", "stable", "j2", method="qlora")
    by_job = {p["jobName"]: p for p in v.list_pending()}
    assert by_job["j1"]["method"] == "lora" and by_job["j1"]["imageTag"] == "stable"
    assert by_job["j2"]["method"] == "qlora" and by_job["j2"]["imageTag"] == "stable"


def test_reset_tier_clears_all_methods_on_that_tier(temp_store):
    v.set_status("m1", "stable", v.VERIFIED)
    v.set_status("m1", "stable", v.VERIFIED, method="qlora")
    v.set_status("m1", "latest", v.VERIFIED)  # other tier survives
    cleared = v.reset_tier("stable")
    assert cleared == 2  # both stable keys
    assert v.get_status("m1", "stable")["status"] == v.UNTESTED
    assert v.get_status("m1", "stable", "qlora")["status"] == v.UNTESTED
    assert v.get_status("m1", "latest")["status"] == v.VERIFIED


def test_seed_covers_lora_only_not_qlora(temp_store):
    # A model in the shipped seed is LoRA-verified — its QLoRA must NOT inherit that.
    from app.verification_seed import _SEED, seed_status

    seeded = next(m for m, tiers in _SEED.items() if "stable" in tiers)
    assert seed_status(seeded, "stable") is not None  # LoRA seed present
    assert v.get_status(seeded, "stable", "qlora")["status"] == v.UNTESTED


# --- engine-keyed verification (LLaMA-Factory vs serverless proven independently) ---


def test_default_engine_key_scheme_unchanged(temp_store):
    assert v._key("stable", "lora", "llama_factory") == "stable"
    assert v._key("stable", "qlora", "llama_factory") == "stable::qlora"
    v.set_status("m1", "stable", v.VERIFIED)  # default engine + lora
    v.set_status("m1", "stable", v.VERIFIED, method="qlora")
    assert set(v._all()["m1"].keys()) == {"stable", "stable::qlora"}  # no engine token


def test_serverless_uses_serverless_surface_not_image_tag(temp_store):
    # A serverless proof is keyed on the "serverless" surface regardless of the
    # image_tag passed (serverless has no image of ours).
    v.set_status("m1", "stable", v.VERIFIED, engine="sagemaker_serverless")
    assert "serverless" in v._all()["m1"]  # stored under the engine surface
    assert "stable" not in v._all()["m1"]  # NOT under the image tier
    assert v.get_status("m1", "stable", engine="sagemaker_serverless")["status"] == v.VERIFIED


def test_serverless_proof_does_not_leak_to_llama_factory_and_vice_versa(temp_store):
    v.set_status("m1", "stable", v.VERIFIED, engine="sagemaker_serverless")
    # LLaMA-Factory on the same model+tier is a DIFFERENT axis → still untested.
    assert v.get_status("m1", "stable")["status"] == v.UNTESTED
    assert v.get_status("m1", "stable", engine="llama_factory")["status"] == v.UNTESTED
    # And the reverse: an LF proof doesn't make serverless verified.
    v.set_status("m2", "stable", v.VERIFIED)  # LF
    assert v.get_status("m2", "stable", engine="sagemaker_serverless")["status"] == v.UNTESTED


def test_seed_does_not_leak_to_serverless(temp_store):
    # A seeded (LF+LoRA) model must NOT show its serverless surface as verified.
    from app.verification_seed import _SEED, seed_status

    seeded = next(m for m, tiers in _SEED.items() if "stable" in tiers)
    assert seed_status(seeded, "stable") is not None
    assert v.get_status(seeded, "stable", engine="sagemaker_serverless")["status"] == v.UNTESTED


def test_reset_tier_leaves_serverless_proofs_intact(temp_store):
    # An ECR image rebuild (reset_tier) is irrelevant to the serverless managed
    # recipe → serverless proofs survive a tier reset.
    v.set_status("m1", "stable", v.VERIFIED)  # LF on stable
    v.set_status("m1", "stable", v.VERIFIED, engine="sagemaker_serverless")
    cleared = v.reset_tier("stable")
    assert cleared == 1  # only the LF stable key
    assert v.get_status("m1", "stable")["status"] == v.UNTESTED  # LF reset
    assert v.get_status("m1", "stable", engine="sagemaker_serverless")["status"] == v.VERIFIED


def test_list_pending_carries_engine(temp_store):
    v.set_pending("m1", "stable", "j-lf")  # default engine
    v.set_pending("m1", "stable", "j-sv", engine="sagemaker_serverless")
    by_job = {p["jobName"]: p for p in v.list_pending()}
    assert by_job["j-lf"]["engine"] == "llama_factory"
    assert by_job["j-sv"]["engine"] == "sagemaker_serverless"
    assert by_job["j-sv"]["imageTag"] == "serverless"  # surface recovered


def test_split_key_roundtrip_engine(temp_store):
    # surface[::method[::variant]] round-trips to (surface, method, engine, variant).
    assert v._split_key("serverless") == ("serverless", "lora", "sagemaker_serverless", "lora")
    assert v._split_key("stable") == ("stable", "lora", "llama_factory", "lora")
    assert v._split_key("stable::qlora") == ("stable", "qlora", "llama_factory", "lora")
    # A non-plain LoRA variant carries an explicit method token, so the 3rd ::
    # segment is the variant — never confused with a method.
    assert v._split_key("stable::lora::dora") == ("stable", "lora", "llama_factory", "dora")
    assert v._split_key("latest::qlora::rslora") == ("latest", "qlora", "llama_factory", "rslora")


# --- variant-keyed verification (DoRA/rsLoRA/PiSSA/LoRA+ proven independently) ---


def test_plain_lora_key_byte_identical_with_variant_axis(temp_store):
    # The newest axis must not change ANY pre-variant key: plain LoRA stays bare,
    # qlora stays surface::qlora, serverless stays serverless — no migration.
    assert v._key("stable", "lora") == "stable"
    assert v._key("stable", "lora", lora_variant="lora") == "stable"
    assert v._key("stable", "qlora") == "stable::qlora"
    assert v._key("stable", "lora", "sagemaker_serverless") == "serverless"


def test_variant_key_carries_explicit_method_token(temp_store):
    # A non-plain variant is surface::method::variant — the method token is ALWAYS
    # present so the variant can't be parsed as a method.
    assert v._key("stable", "lora", lora_variant="dora") == "stable::lora::dora"
    assert v._key("stable", "qlora", lora_variant="rslora") == "stable::qlora::rslora"
    assert v._key("latest", "lora", lora_variant="loraplus") == "latest::lora::loraplus"


def test_dora_proof_does_not_leak_to_plain_lora(temp_store):
    # THE bug this feature fixes: a DoRA proof must NOT make plain LoRA look verified.
    v.set_status("m1", "stable", v.VERIFIED, lora_variant="dora", job_name="j-dora")
    assert v.get_status("m1", "stable", lora_variant="dora")["status"] == v.VERIFIED
    assert v.get_status("m1", "stable")["status"] == v.UNTESTED  # plain LoRA untouched
    assert v.get_status("m1", "stable", lora_variant="lora")["status"] == v.UNTESTED


def test_plain_lora_proof_does_not_leak_to_dora(temp_store):
    # And the reverse: a plain LoRA proof doesn't cover DoRA (it changes the merge).
    v.set_status("m1", "stable", v.VERIFIED, job_name="j-lora")
    assert v.get_status("m1", "stable")["status"] == v.VERIFIED
    assert v.get_status("m1", "stable", lora_variant="dora")["status"] == v.UNTESTED
    assert v.get_status("m1", "stable", lora_variant="rslora")["status"] == v.UNTESTED


def test_each_variant_keyed_independently(temp_store):
    v.set_status("m1", "stable", v.VERIFIED, lora_variant="dora")
    v.set_status("m1", "stable", v.INCOMPATIBLE, lora_variant="pissa", reason="svd oom")
    assert v.get_status("m1", "stable", lora_variant="dora")["status"] == v.VERIFIED
    assert v.get_status("m1", "stable", lora_variant="pissa")["status"] == v.INCOMPATIBLE
    assert v.get_status("m1", "stable", lora_variant="rslora")["status"] == v.UNTESTED
    raw = set(v._all()["m1"].keys())
    assert raw == {"stable::lora::dora", "stable::lora::pissa"}


def test_full_weight_method_never_carries_variant_token(temp_store):
    # Variants only ride adapter methods; a stray variant on full/freeze normalizes
    # to plain "lora" so it shares the full/freeze method key, not a variant sibling.
    assert v._key("stable", "full", lora_variant="dora") == "stable::full"
    assert v._key("stable", "freeze", lora_variant="pissa") == "stable::freeze"


def test_unknown_variant_falls_back_to_plain_key(temp_store):
    # A bad variant value can't forge a new key (defensive — the API Literal already
    # rejects it, but the store must not store a junk sibling either).
    assert v._key("stable", "lora", lora_variant="bogus") == "stable"


def test_seed_covers_plain_lora_only_not_variants(temp_store):
    # A seeded (plain LoRA) model must NOT show DoRA as verified by inheritance.
    from app.verification_seed import _SEED, seed_status

    seeded = next(m for m, tiers in _SEED.items() if "stable" in tiers)
    assert seed_status(seeded, "stable") is not None
    assert v.get_status(seeded, "stable", lora_variant="dora")["status"] == v.UNTESTED


def test_list_pending_carries_variant(temp_store):
    v.set_pending("m1", "stable", "j-plain")  # plain LoRA
    v.set_pending("m1", "stable", "j-dora", lora_variant="dora")
    by_job = {p["jobName"]: p for p in v.list_pending()}
    assert by_job["j-plain"]["loraVariant"] == "lora"
    assert by_job["j-dora"]["loraVariant"] == "dora"
    assert by_job["j-dora"]["imageTag"] == "stable"


def test_reset_tier_clears_variant_siblings_too(temp_store):
    v.set_status("m1", "stable", v.VERIFIED)  # plain
    v.set_status("m1", "stable", v.VERIFIED, lora_variant="dora")
    v.set_status("m1", "stable", v.VERIFIED, lora_variant="rslora")
    v.set_status("m1", "latest", v.VERIFIED, lora_variant="dora")  # other tier survives
    cleared = v.reset_tier("stable")
    assert cleared == 3  # plain + dora + rslora on stable
    assert v.get_status("m1", "stable", lora_variant="dora")["status"] == v.UNTESTED
    assert v.get_status("m1", "latest", lora_variant="dora")["status"] == v.VERIFIED
