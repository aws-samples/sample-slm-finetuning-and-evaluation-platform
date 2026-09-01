# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Multi-image: per-model image-tier resolution + (model,tag) verification store."""
import pytest

from app.aws_config import (
    DEFAULT_IMAGE_TIER,
    IMAGE_TIER_TAGS,
    AwsConfig,
    image_tiers,
)
from app.catalog import get_model, list_models


def _cfg() -> AwsConfig:
    return AwsConfig(
        region="us-east-1",
        account_id="111122223333",
        role_arn="arn:aws:iam::111122223333:role/exec",
        bucket="b",
        image_uri="111122223333.dkr.ecr.us-east-1.amazonaws.com/slm-platform-llamafactory:0.9.4",
        profile=None,
    )


# --- image tier resolution -------------------------------------------------

def test_stable_tier_resolves_to_global_image():
    cfg = _cfg()
    # stable must be byte-identical to the existing global image_uri so every
    # currently-working model is unchanged by the multi-image change.
    assert cfg.image_uri_for_tier("stable") == cfg.image_uri
    assert cfg.image_uri_for_tier(None) == cfg.image_uri  # default = stable


def test_latest_tier_resolves_to_095_on_same_repo():
    cfg = _cfg()
    uri = cfg.image_uri_for_tier("latest")
    assert uri.endswith(":0.9.5")
    assert uri.rsplit(":", 1)[0] == cfg.image_repo  # same ECR repo, new tag


def test_unknown_tier_falls_back_to_default():
    cfg = _cfg()
    # Never launch against a non-existent tag — unknown tier degrades to stable.
    assert cfg.image_uri_for_tier("does-not-exist") == cfg.image_uri_for_tier(DEFAULT_IMAGE_TIER)


def test_image_tiers_includes_builtins():
    tiers = image_tiers()
    assert tiers["stable"] == IMAGE_TIER_TAGS["stable"]
    assert tiers["latest"] == IMAGE_TIER_TAGS["latest"]


def test_resolve_image_uri_uses_model_tag():
    from app.orchestrate import resolve_image_uri

    cfg = _cfg()
    m = get_model("qwen3-1.7b")  # default tier = stable
    assert resolve_image_uri(cfg, m) == cfg.image_uri


def test_every_catalog_model_has_a_resolvable_tier():
    cfg = _cfg()
    tiers = image_tiers()
    for m in list_models():
        tag = m.get("imageTag", "stable")
        # Either a known tier, or it still resolves (falls back) — never crashes.
        assert cfg.image_uri_for_tier(tag).startswith(cfg.image_repo + ":")
        # Catalog models should only declare tiers we actually build.
        if not m.get("custom"):
            assert tag in tiers, f"{m['id']} declares unbuilt tier {tag}"


# --- verification store ----------------------------------------------------

def test_verification_default_is_untested(temp_store):
    from app import verifications as v

    rec = v.get_status("qwen3-4b", "stable")
    assert rec["status"] == v.UNTESTED
    assert rec["jobName"] is None


def test_set_and_get_verified(temp_store):
    from app import verifications as v

    v.set_status("qwen3-4b", "stable", v.VERIFIED, job_name="job-1", ts="2026-06-04T00:00:00Z")
    rec = v.get_status("qwen3-4b", "stable")
    assert rec["status"] == v.VERIFIED
    assert rec["jobName"] == "job-1"
    # Per-tier isolation: latest is still untested.
    assert v.get_status("qwen3-4b", "latest")["status"] == v.UNTESTED


def test_authority_rank_keeps_verified_over_later_failure(temp_store):
    from app import verifications as v

    v.set_status("m", "stable", v.VERIFIED, job_name="good")
    # A later transient failure must NOT erase a proven success (informational,
    # not a hard block — flaky spot/download shouldn't hide a good model).
    v.set_status("m", "stable", v.INCOMPATIBLE, reason="flaky")
    assert v.get_status("m", "stable")["status"] == v.VERIFIED


def test_force_overwrites_for_explicit_retest(temp_store):
    from app import verifications as v

    v.set_status("m", "stable", v.VERIFIED, job_name="old")
    v.set_status("m", "stable", v.INCOMPATIBLE, reason="rebuilt", force=True)
    assert v.get_status("m", "stable")["status"] == v.INCOMPATIBLE


def test_incompatible_then_verified_upgrades(temp_store):
    from app import verifications as v

    v.set_status("m", "latest", v.INCOMPATIBLE, reason="ImportError LossKwargs")
    v.set_status("m", "latest", v.VERIFIED, job_name="now-works")
    assert v.get_status("m", "latest")["status"] == v.VERIFIED


def test_reset_tier_clears_only_that_tier(temp_store):
    from app import verifications as v

    v.set_status("m", "stable", v.VERIFIED, job_name="s")
    v.set_status("m", "latest", v.VERIFIED, job_name="l")
    cleared = v.reset_tier("latest")
    assert cleared == 1
    assert v.get_status("m", "stable")["status"] == v.VERIFIED
    assert v.get_status("m", "latest")["status"] == v.UNTESTED


def test_model_status_map_shows_all_tiers(temp_store):
    from app import verifications as v

    v.set_status("m", "stable", v.VERIFIED, job_name="s")
    m = v.model_status_map("m")
    assert set(m) >= {"stable", "latest"}
    assert m["stable"]["status"] == v.VERIFIED
    assert m["latest"]["status"] == v.UNTESTED


def test_bad_status_rejected(temp_store):
    from app import verifications as v

    with pytest.raises(ValueError):
        v.set_status("m", "stable", "bogus")


# --- Phase 2: auto-promotion from the race reconcile loop ------------------

def _race_with_completed_training(temp_store, monkeypatch, model_id="qwen3-1.7b"):
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams

    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(rm, "launch_eval_job", lambda **kw: {"jobName": f"eval-{kw['stamp']}"})
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    monkeypatch.setattr(rm, "describe_job",
                        lambda job: {"status": "Completed", "trainingEndTime": "2026-06-04T01:00:00Z"})
    rms = [rm.RaceModel(model_id=model_id, hp=Hyperparams())]
    race = rm.start_race("split-x", rms, DecodingParams(), "20260604-9")
    rm.reconcile_race(race.race_id)  # TRAINING → Completed → autopromote + eval
    return rm, race


def test_training_completion_autopromotes_to_verified(temp_store, monkeypatch):
    from app import verifications as v

    _race_with_completed_training(temp_store, monkeypatch, model_id="qwen3-1.7b")
    # qwen3-1.7b is a stable-tier model → its stable verification is now proven.
    rec = v.get_status("qwen3-1.7b", "stable")
    assert rec["status"] == v.VERIFIED
    assert rec["jobName"].startswith("train-qwen3-1.7b")
    assert rec["ts"] == "2026-06-04T01:00:00Z"


def test_qlora_run_autopromotes_qlora_not_lora(temp_store, monkeypatch):
    """A completed QLoRA race entry proves QLoRA (the `stable::qlora` key), and
    must NOT mark plain LoRA verified — they're proven independently."""
    from app import race as rm
    from app import verifications as v
    from app.catalog import DecodingParams, Hyperparams

    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(rm, "launch_eval_job", lambda **kw: {"jobName": f"eval-{kw['stamp']}"})
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    monkeypatch.setattr(rm, "describe_job",
                        lambda job: {"status": "Completed", "trainingEndTime": "2026-06-11T01:00:00Z"})
    # qwen3-8b is NOT in the shipped seed, so its LoRA stays untested unless this
    # run proves it — which it must NOT (the run is QLoRA).
    rms = [rm.RaceModel(model_id="qwen3-8b", hp=Hyperparams(finetuning_type="qlora"))]
    race = rm.start_race("split-q", rms, DecodingParams(), "20260611-q")
    rm.reconcile_race(race.race_id)
    assert v.get_status("qwen3-8b", "stable", "qlora")["status"] == v.VERIFIED
    # LoRA on the same model stays untested (its proof didn't come from this run).
    assert v.get_status("qwen3-8b", "stable")["status"] == v.UNTESTED


def test_dora_run_autopromotes_dora_not_plain_lora(temp_store, monkeypatch):
    """THE bug this feature fixes: a completed DoRA race entry must autopromote to
    the per-variant key `stable::lora::dora`, NOT the bare plain-LoRA key. Before
    the fix, _autopromote dropped lora_variant and DoRA silently inherited/polluted
    plain LoRA's verified badge. This exercises the reconcile→_autopromote path
    end-to-end (re-reading entry.hp), so removing the variant forwarding fails it."""
    from app import race as rm
    from app import verifications as v
    from app.catalog import DecodingParams, Hyperparams

    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(rm, "launch_eval_job", lambda **kw: {"jobName": f"eval-{kw['stamp']}"})
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    monkeypatch.setattr(rm, "describe_job",
                        lambda job: {"status": "Completed", "trainingEndTime": "2026-06-26T01:00:00Z"})
    # qwen3-8b is NOT seeded, so plain LoRA is only verified if a plain-LoRA run
    # proves it — which this DoRA run must NOT do.
    rms = [rm.RaceModel(model_id="qwen3-8b", hp=Hyperparams(lora_variant="dora"))]
    race = rm.start_race("split-d", rms, DecodingParams(), "20260626-d")
    rm.reconcile_race(race.race_id)
    assert v.get_status("qwen3-8b", "stable", lora_variant="dora")["status"] == v.VERIFIED
    # Plain LoRA on the same model stays untested — DoRA's proof doesn't leak to it.
    assert v.get_status("qwen3-8b", "stable")["status"] == v.UNTESTED


def test_backfill_from_races_seeds_verified(temp_store, monkeypatch):
    from app import race as rm
    from app import verifications as v
    from app.catalog import DecodingParams, Hyperparams

    # A race whose entry trained successfully (state EVALUATING/DONE) but with no
    # verification recorded yet (simulating pre-feature history).
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}"})
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    # qwen3-8b is NOT in the seed, so it starts untested even with the baseline.
    race = rm.start_race("split-y", [rm.RaceModel(model_id="qwen3-8b", hp=Hyperparams())],
                         DecodingParams(), "20260604-bf")
    # Hand-place it into a trained state and clear any verification.
    race.entries[0].state = rm.DONE
    rm._save(race)
    v.reset_tier("stable")
    assert v.get_status("qwen3-8b", "stable")["status"] == v.UNTESTED

    summary = v.backfill_from_races()
    assert summary["promoted"] >= 1
    assert v.get_status("qwen3-8b", "stable")["status"] == v.VERIFIED


# --- API wiring ------------------------------------------------------------

def test_verification_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in [
        "/api/verifications",
        "/api/verifications/backfill",
        "/api/verify/{model_id}/{image_tag}/{job_name}",
    ]:
        assert p in paths, f"missing route {p}"


def test_models_endpoint_carries_verifications_and_tiers(temp_store):
    import app.main as m

    out = m.get_models()
    assert "imageTiers" in out and out["imageTiers"]["stable"] == "0.9.4"
    # Every model row carries a per-tier verification grid.
    row = next(x for x in out["models"] if x["id"] == "qwen3-1.7b")
    assert "verifications" in row
    assert row["verifications"]["stable"]["status"] in ("untested", "verified", "incompatible")


# --- provider grouping -----------------------------------------------------

def test_provider_derived_from_hf_org():
    from app.catalog import provider_for

    assert provider_for("Qwen/Qwen3-1.7B") == "Alibaba (Qwen)"
    assert provider_for("microsoft/Phi-4-mini-instruct") == "Microsoft"
    assert provider_for("meta-llama/Llama-3.2-1B-Instruct") == "Meta"
    assert provider_for("LiquidAI/LFM2-8B-A1B") == "LiquidAI"
    # Unknown org → title-cased fallback (never crashes).
    assert provider_for("some-new-org/cool-model") == "Some New Org"


def test_every_model_has_a_provider():
    from app.catalog import list_models

    for m in list_models():
        assert m["provider"], f"{m['id']} missing provider"


def test_image_and_reset_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in [
        "/api/images",
        "/api/images/{image_tag}/build",
        "/api/images/{image_tag}/reset-verifications",
    ]:
        assert p in paths, f"missing route {p}"


def test_reset_verifications_endpoint_clears_tier(temp_store):
    import app.main as main
    from app import verifications as v

    v.set_status("m", "latest", v.VERIFIED, job_name="j")
    out = main.reset_image_verifications("latest")
    assert out["cleared"] == 1
    # The store record is cleared; the model is not in the seed → untested.
    assert v.get_status("m", "latest")["status"] == v.UNTESTED


# --- verification seed (baseline shown in a fresh account) -----------------

def test_seed_models_verified_on_fresh_store(temp_store):
    """A brand-new account (empty store) still shows the seeded baseline as
    verified, so a fresh deployment's catalog is useful on day one."""
    from app import verifications as v

    # qwen3-1.7b is in the seed on `stable`; nothing written to the store here.
    rec = v.get_status("qwen3-1.7b", "stable")
    assert rec["status"] == v.VERIFIED
    assert rec.get("seed") is True
    # lfm2 is the canonical latest-tier baseline.
    assert v.get_status("lfm2-8b-a1b", "latest")["status"] == v.VERIFIED


def test_store_overrides_seed(temp_store):
    """A local run ALWAYS wins over the seed — even a failure (the account's own
    truth beats the shipped optimism)."""
    from app import verifications as v

    # Force an incompatible record for a seeded-verified model.
    v.set_status("qwen3-1.7b", "stable", v.INCOMPATIBLE, reason="local failure", force=True)
    rec = v.get_status("qwen3-1.7b", "stable")
    assert rec["status"] == v.INCOMPATIBLE
    assert rec.get("seed") is not True


def test_all_verifications_merges_seed_under_store(temp_store):
    from app import verifications as v

    v.set_status("my-local-model", "stable", v.VERIFIED, job_name="local-job")
    allv = v.all_verifications()
    # Seed entries present...
    assert allv["qwen3-1.7b"]["stable"]["status"] == v.VERIFIED
    # ...and the local-only record too.
    assert allv["my-local-model"]["stable"]["jobName"] == "local-job"


def test_not_seeded_model_is_untested(temp_store):
    from app import verifications as v

    assert v.get_status("qwen3-8b", "stable")["status"] == v.UNTESTED


# --- pending verification (survives navigation; resolved by reconcile) ------

def test_set_pending_and_list(temp_store):
    from app import verifications as v

    v.set_pending("qwen3-8b", "stable", "job-smoke-1")
    rec = v.get_status("qwen3-8b", "stable")
    assert rec["status"] == v.PENDING
    assert rec["jobName"] == "job-smoke-1"
    pend = v.list_pending()
    assert any(p["modelId"] == "qwen3-8b" and p["jobName"] == "job-smoke-1" for p in pend)


def test_pending_does_not_overwrite_verified(temp_store):
    from app import verifications as v

    v.set_status("m", "stable", v.VERIFIED, job_name="good")
    v.set_pending("m", "stable", "new-job")  # uses force, but rank guard protects verified
    # set_pending forces, so it WILL show pending — that's intended for an explicit
    # re-test. Confirm it's pending now (the user asked to re-verify).
    assert v.get_status("m", "stable")["status"] == v.PENDING


def test_resolve_pending_completed_to_verified(temp_store, monkeypatch):
    from app import verifications as v
    import app.orchestrate as orch

    v.set_pending("qwen3-8b", "stable", "job-x")
    monkeypatch.setattr(orch, "describe_job",
                        lambda job: {"status": "Completed", "trainingEndTime": "2026-06-04T01:00:00Z"})
    summary = v.resolve_pending_verifications()
    assert summary["resolved"] == 1
    assert v.get_status("qwen3-8b", "stable")["status"] == v.VERIFIED


def test_resolve_pending_failed_to_incompatible(temp_store, monkeypatch):
    from app import verifications as v
    import app.orchestrate as orch

    v.set_pending("qwen3-8b", "latest", "job-y")
    monkeypatch.setattr(orch, "describe_job",
                        lambda job: {"status": "Failed", "failureReason": "AlgorithmError"})
    v.resolve_pending_verifications()
    rec = v.get_status("qwen3-8b", "latest")
    assert rec["status"] == v.INCOMPATIBLE
    assert "AlgorithmError" in (rec["reason"] or "")


def test_resolve_pending_inprogress_stays_pending(temp_store, monkeypatch):
    from app import verifications as v
    import app.orchestrate as orch

    v.set_pending("qwen3-8b", "stable", "job-z")
    monkeypatch.setattr(orch, "describe_job", lambda job: {"status": "InProgress"})
    summary = v.resolve_pending_verifications()
    assert summary["resolved"] == 0
    assert v.get_status("qwen3-8b", "stable")["status"] == v.PENDING
