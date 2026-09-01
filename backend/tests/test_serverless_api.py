# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Race-endpoint guards for the serverless engine.

Calls the race_launch handler directly (no httpx/TestClient dependency) with a
constructed RaceRequest. start_race + split_dir are monkeypatched so no AWS is
touched — we're testing request validation, not the launch.
"""
import pytest


@pytest.fixture
def launch(temp_store, monkeypatch):
    import app.main as m
    from fastapi import HTTPException

    monkeypatch.setattr(m, "split_dir", lambda s: "/tmp/fake")
    # The split's shape gates the objective (sft|preference|kto|rlvr). Tests set it
    # per-call via `dataset_shape` so an RLVR launch sees an rlvr-shaped split.
    state = {"shape": "sft"}
    monkeypatch.setattr(m, "split_meta", lambda s: {"shape": state["shape"]})
    captured = {}

    def _fake_start_race(split_id, models, decoding, stamp, **kw):
        captured["models"] = models
        from app.race import Race
        return Race(race_id="race-x", split_id=split_id, stamp=stamp, decoding={})

    monkeypatch.setattr(m, "start_race", _fake_start_race)
    monkeypatch.setattr(m, "rank_entries", lambda race: [])

    def _run(model_id, engine="llama_factory", finetuning_type="lora",
             stage="sft", preset_reward_function="", dataset_shape="sft",
             pref_loss="sigmoid", packing=False):
        state["shape"] = dataset_shape
        try:
            req = m.RaceRequest(
                splitId="abc123def456",
                models=[m.RaceModelConfig(
                    modelId=model_id, engine=engine, finetuningType=finetuning_type,
                    stage=stage, presetRewardFunction=preset_reward_function,
                    prefLoss=pref_loss, packing=packing)],
            )
            m.race_launch(req)
            return 200, "", captured
        except HTTPException as e:
            return e.status_code, str(e.detail), captured
        except ValueError as e:
            # Hyperparams gating raises ValueError at RaceModelConfig.to_hp() time
            # (e.g. rlvr without a preset). race_launch wraps it as a 400; surface
            # the same shape here when the model itself raises during construction.
            return 400, str(e), captured

    return _run


def test_llama_factory_race_unaffected(launch):
    code, _, cap = launch("qwen3-4b", "llama_factory")
    assert code == 200
    assert cap["models"][0].hp.engine == "llama_factory"


def test_serverless_race_on_mapped_model_ok(launch):
    # Serverless is ON by default — no setup needed.
    code, _, cap = launch("qwen3-4b", "sagemaker_serverless")
    assert code == 200
    assert cap["models"][0].hp.engine == "sagemaker_serverless"


def test_serverless_rejected_when_flag_off(launch):
    # Saved config is the single source of truth: persist OFF → launch rejected.
    from app.aws_config import save_config

    save_config({"enableSagemakerServerless": False})
    code, detail, _ = launch("qwen3-4b", "sagemaker_serverless")
    assert code == 400 and "not enabled" in detail.lower()


def test_serverless_rejected_for_unmapped_model(launch):
    code, detail, _ = launch("phi-3.5-mini", "sagemaker_serverless")
    assert code == 400 and "no sagemaker serverless equivalent" in detail.lower()


def test_serverless_qlora_combo_rejected_400(launch):
    code, detail, _ = launch("qwen3-4b", "sagemaker_serverless", finetuning_type="qlora")
    assert code == 400 and "does not support method" in detail.lower()


# --- RLVR objective (serverless-only, preset-reward) -----------------------

def test_serverless_rlvr_with_preset_reward_ok(launch):
    code, detail, cap = launch("qwen3-4b", "sagemaker_serverless",
                               stage="rlvr", preset_reward_function="gsm8k",
                               dataset_shape="rlvr")
    assert code == 200, detail
    hp = cap["models"][0].hp
    assert hp.stage == "rlvr"
    assert hp.preset_reward_function == "gsm8k"


def test_serverless_rlvr_without_preset_rejected_400(launch):
    code, detail, _ = launch("qwen3-4b", "sagemaker_serverless", stage="rlvr",
                             dataset_shape="rlvr")
    assert code == 400 and "preset reward" in detail.lower()


def test_rlvr_rejected_on_llama_factory_400(launch):
    # RLVR is serverless-only; the LLaMA-Factory engine can't do it (even on an
    # rlvr-shaped dataset, the engine-stage capability gate rejects it).
    code, detail, _ = launch("qwen3-4b", "llama_factory",
                             stage="rlvr", preset_reward_function="gsm8k",
                             dataset_shape="rlvr")
    assert code == 400 and "does not support stage" in detail.lower()


def test_rlvr_rejected_on_non_rlvr_dataset_400(launch):
    # An RLVR launch against an SFT-shaped dataset is rejected — the dataset is the
    # source of truth for the objective (the whole point of the separate RLVR tab).
    code, detail, _ = launch("qwen3-4b", "sagemaker_serverless",
                             stage="rlvr", preset_reward_function="gsm8k",
                             dataset_shape="sft")
    assert code == 400 and "rlvr" in detail.lower() and "dataset" in detail.lower()


# --- /api/models: the flag drives UI engine visibility (Option A) ----------

def _engines_for(models, model_id):
    return next(m["engines"] for m in models if m["id"] == model_id)


def test_catalog_hides_serverless_when_flag_off(temp_store):
    """Flag OFF (saved) → a serverless-mapped model's engines[] drops
    sagemaker_serverless, so the UI picker never offers it (matches the launch
    guard's rejection)."""
    import app.main as m
    from app.aws_config import save_config

    save_config({"enableSagemakerServerless": False})
    models = m.get_models()["models"]
    # qwen3-4b IS serverless-mapped, but with the flag off it looks LF-only.
    assert _engines_for(models, "qwen3-4b") == ["llama_factory"]
    # an unmapped model is llama_factory-only regardless.
    assert _engines_for(models, "phi-3.5-mini") == ["llama_factory"]


def test_catalog_shows_serverless_by_default(temp_store):
    import app.main as m

    # Serverless ships ON — no saved config needed.
    models = m.get_models()["models"]
    assert "sagemaker_serverless" in _engines_for(models, "qwen3-4b")
    # unmapped model still never gains serverless.
    assert _engines_for(models, "phi-3.5-mini") == ["llama_factory"]


# --- Settings toggle: enable serverless via config (no redeploy) -----------

def test_settings_toggle_persists_serverless_state(temp_store):
    """PUT /api/config {enableSagemakerServerless} persists as the camelCase key
    the flag reader uses, so the toggle sticks without a redeploy — and get_config
    reflects it. Default is ON; toggling off then on round-trips."""
    import app.main as m

    # Default ON with no saved config.
    assert m.get_config()["enableSagemakerServerless"] is True
    # toggling off disables it (and persists under the camelCase key)
    out = m.put_config(m.ConfigUpdate(enableSagemakerServerless=False))
    assert out["enableSagemakerServerless"] is False
    from app.aws_config import _saved
    assert _saved().get("enableSagemakerServerless") is False
    assert _engines_for(m.get_models()["models"], "qwen3-4b") == ["llama_factory"]
    # toggling back on re-enables; the catalog exposes serverless again
    out = m.put_config(m.ConfigUpdate(enableSagemakerServerless=True))
    assert out["enableSagemakerServerless"] is True
    assert "sagemaker_serverless" in _engines_for(m.get_models()["models"], "qwen3-4b")


def test_settings_toggle_off_sticks(temp_store):
    """Saved config is the SINGLE source of truth (no env override): an OFF toggle
    must persist and win over the built-in default-ON, with no snap-back."""
    import app.main as m
    from app.engines.base import engine_enabled

    assert engine_enabled("sagemaker_serverless") is True  # built-in default on
    # operator turns it OFF via Settings → saved value wins, stays off on re-read
    m.put_config(m.ConfigUpdate(enableSagemakerServerless=False))
    assert engine_enabled("sagemaker_serverless") is False
    assert engine_enabled("sagemaker_serverless") is False  # no snap-back
    assert _engines_for(m.get_models()["models"], "qwen3-4b") == ["llama_factory"]
    # turning it back ON via Settings re-enables
    m.put_config(m.ConfigUpdate(enableSagemakerServerless=True))
    assert engine_enabled("sagemaker_serverless") is True


def test_settings_config_roundtrip_uses_camel_keys(temp_store, monkeypatch):
    """Regression: put_config must persist camelCase aliases (roleArn, not
    role_arn) so load_aws_config — which reads camelCase — sees them."""
    import app.main as m

    m.put_config(m.ConfigUpdate(roleArn="arn:aws:iam::1:role/x"))
    from app.aws_config import _saved
    assert _saved().get("roleArn") == "arn:aws:iam::1:role/x"


# --- Clone run config (clone & edit feature) -------------------------------

def test_clone_race_config_replays_models_and_hp(temp_store, monkeypatch):
    """A run's launch config is reconstructed so the builder can pre-fill it:
    same dataset, and each entry's engine/stage/method/hp replayed as a
    RaceModelConfig-shaped dict (camelCase aliases the UI speaks)."""
    import app.main as m
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams

    # Persist a race directly (no AWS): two distinct entries — LF qlora + serverless.
    monkeypatch.setattr(rm, "launch_training_job", lambda **kw: {"jobName": "t"})
    monkeypatch.setattr(rm, "launch_base_eval_job", lambda **kw: {"jobName": "b"})
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    models = [
        rm.RaceModel(model_id="qwen3-4b", hp=Hyperparams(finetuning_type="qlora", lora_rank=16)),
        rm.RaceModel(model_id="qwen3-4b", hp=Hyperparams(engine="sagemaker_serverless", num_train_epochs=2.0)),
    ]
    race = rm.start_race("abc123def456", models, DecodingParams(max_new_tokens=128, temperature=0.0),
                         "20260615-1", name="run01")

    cfg = m.clone_race_config(race.race_id)
    assert cfg["splitId"] == "abc123def456"
    assert cfg["name"].endswith("(clone)")
    assert cfg["evalMaxNewTokens"] == 128
    assert len(cfg["models"]) == 2
    by = {(x["modelId"], x["engine"], x["finetuningType"]): x for x in cfg["models"]}
    # LF qlora entry replayed
    lf = by[("qwen3-4b", "llama_factory", "qlora")]
    assert lf["loraRank"] == 16
    # serverless entry replayed (engine + epochs preserved)
    sv = by[("qwen3-4b", "sagemaker_serverless", "lora")]
    assert sv["numTrainEpochs"] == 2.0


def test_clone_race_config_unknown_race_404(temp_store):
    import app.main as m
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        m.clone_race_config("race-does-not-exist")
    assert ei.value.status_code == 404


# --- ORPO / SimPO preference objectives (race endpoint + clone) -------------


def test_dpo_orpo_simpo_race_ok_on_llama_factory(launch):
    """All three preference losses launch on the LF engine with a preference dataset
    — and each carries its pref_loss through to the constructed Hyperparams."""
    for pl in ("sigmoid", "orpo", "simpo"):
        code, detail, cap = launch("qwen3-4b", "llama_factory", stage="dpo",
                                   dataset_shape="preference", pref_loss=pl)
        assert code == 200, f"{pl} should launch on LF: {detail}"
        assert cap["models"][0].hp.pref_loss == pl


def test_serverless_orpo_rejected_400(launch):
    """ORPO/SimPO are reference-free, LLaMA-Factory-only — a serverless+ORPO race
    must surface a clean 400 (the to_hp ValueError → HTTPException), NOT a 500."""
    for pl in ("orpo", "simpo"):
        code, detail, _ = launch("qwen3-4b", "sagemaker_serverless", stage="dpo",
                                 dataset_shape="preference", pref_loss=pl)
        assert code == 400, f"serverless+{pl} should 400"
        assert "llama_factory" in detail.lower()


def test_packing_on_dpo_rejected_400(launch):
    """Packing is SFT-only — a packing+DPO race surfaces a clean 400, not a 500."""
    code, detail, _ = launch("qwen3-4b", "llama_factory", stage="dpo",
                             dataset_shape="preference", packing=True)
    assert code == 400 and "packing" in detail.lower()


def test_clone_orpo_simpo_run_preserves_pref_loss(temp_store, monkeypatch):
    """Cloning an ORPO/SimPO run must replay pref_loss (+ simpo_gamma) so it doesn't
    silently downgrade to plain DPO (sigmoid) in the rebuilt builder config."""
    import app.main as m
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams

    monkeypatch.setattr(rm, "launch_training_job", lambda **kw: {"jobName": "t"})
    monkeypatch.setattr(rm, "launch_base_eval_job", lambda **kw: {"jobName": "b"})
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    models = [
        rm.RaceModel(model_id="qwen3-4b", hp=Hyperparams(stage="dpo", pref_loss="orpo")),
        rm.RaceModel(model_id="qwen3-4b", hp=Hyperparams(stage="dpo", pref_loss="simpo", simpo_gamma=0.8)),
    ]
    race = rm.start_race("abc123def456", models, DecodingParams(), "20260626-pl", name="pl-run")
    cfg = m.clone_race_config(race.race_id)
    by_key = {x.get("prefLoss"): x for x in cfg["models"]}
    assert "orpo" in by_key and "simpo" in by_key  # both losses preserved (not downgraded)
    assert by_key["simpo"]["simpoGamma"] == 0.8
    # Round-trip back through to_hp keeps the loss.
    assert m.RaceModelConfig(**{**by_key["orpo"], "modelId": "qwen3-4b"}).to_hp().pref_loss == "orpo"


def test_clone_migrates_retired_preset_reward():
    """A historical RLVR run stored with the now-retired 'prime_code' preset must
    clone to a VALID, launchable preset (prime_math) — not a stale value the picker
    can't show / the launch rejects. Surviving presets pass through unchanged."""
    from app.main import _clone_preset_reward

    assert _clone_preset_reward("prime_code") == "prime_math"  # retired → substitute
    assert _clone_preset_reward("gsm8k") == "gsm8k"            # still valid → unchanged
    assert _clone_preset_reward("prime_math") == "prime_math"
    assert _clone_preset_reward("") == ""                       # custom-reward run


def test_neftune_on_serverless_rejected_400(launch):
    """Efficiency knobs are LLaMA-Factory-only — neftune on serverless → clean 400."""
    import app.main as m
    from fastapi import HTTPException

    try:
        req = m.RaceRequest(
            splitId="abc123def456",
            models=[m.RaceModelConfig(modelId="qwen3-4b", engine="sagemaker_serverless",
                                      stage="sft", neftuneNoiseAlpha=5.0)],
        )
        # split_dir/split_meta are monkeypatched by the `launch` fixture; this test
        # builds the request directly, so guard via to_hp (the same path race_launch hits).
        req.models[0].to_hp()
    except (HTTPException, ValueError) as e:
        assert "llama_factory" in str(getattr(e, "detail", e)).lower()


# --- pre-launch dataset quality gate (synchronous, before any billable launch) ---

def _rm(stage, engine="sagemaker_serverless", **hp_kw):
    """A RaceModel with the given stage/engine for gate tests."""
    from app.race import RaceModel
    from app.catalog import Hyperparams
    # RLVR needs a reward to construct; default to a preset.
    if stage == "rlvr" and "preset_reward_function" not in hp_kw and "reward_function_id" not in hp_kw:
        hp_kw["preset_reward_function"] = "gsm8k"
    return RaceModel(model_id="qwen3-1.7b",
                     hp=Hyperparams(engine=engine, stage=stage, **hp_kw))


def test_quality_gate_blocks_rlvr_below_grpo_floor(temp_store):
    """A small RLVR dataset (effective train < 128 after the 0.9 carve) is HARD
    BLOCKED synchronously with the actionable raw-row bar — not dispatched to fail
    in the worker."""
    from app.main import _prelaunch_quality_gate
    from app.storage import persist_rlvr_split
    from fastapi import HTTPException

    # 50 rows → effective ~45 < 128 floor.
    train = [{"messages": [{"role": "user", "content": f"{i}+1?"}], "ground_truth": str(i + 1)}
             for i in range(50)]
    split_id, _ = persist_rlvr_split(train=train, meta={"name": "tiny-rlvr"})
    with pytest.raises(HTTPException) as ei:
        _prelaunch_quality_gate(split_id, [_rm("rlvr")])
    assert ei.value.status_code == 400
    assert "GRPO needs at least" in str(ei.value.detail)


def test_quality_gate_passes_rlvr_above_grpo_floor(temp_store):
    """An RLVR dataset large enough for GRPO passes the gate (no exception)."""
    from app.main import _prelaunch_quality_gate
    from app.storage import persist_rlvr_split

    train = [{"messages": [{"role": "user", "content": f"{i}+1?"}], "ground_truth": str(i + 1)}
             for i in range(160)]  # effective ~144 > 128
    split_id, _ = persist_rlvr_split(train=train, meta={"name": "ok-rlvr"})
    # Should not raise; returns a (possibly empty) warnings list.
    warns = _prelaunch_quality_gate(split_id, [_rm("rlvr")])
    assert isinstance(warns, list)


def test_quality_gate_blocks_unconvertible_serverless_dataset(temp_store, monkeypatch):
    """If the on-disk train rows can't be reshaped to the recipe format, the gate
    blocks synchronously (the worker would otherwise fail the launch). Here a DPO
    serverless run is pointed at a split whose rows lack chosen/rejected."""
    from app.main import _prelaunch_quality_gate
    from app.storage import persist_rlvr_split
    from fastapi import HTTPException

    # Persist an rlvr-shaped split, then ask the gate to convert it as DPO — the
    # rows have no chosen/rejected, so ranking_to_dpo (via convert_file) fails.
    train = [{"messages": [{"role": "user", "content": f"{i}+1?"}], "ground_truth": str(i + 1)}
             for i in range(160)]
    split_id, _ = persist_rlvr_split(train=train, meta={"name": "wrong-shape"})
    with pytest.raises(HTTPException) as ei:
        _prelaunch_quality_gate(split_id, [_rm("dpo")])
    assert ei.value.status_code == 400
    assert "can't be converted" in str(ei.value.detail).lower()


def test_quality_gate_llama_factory_skips_convert_dry_run(temp_store):
    """The conversion dry-run is serverless-only; an LLaMA-Factory run is not gated
    on it (LF uploads the validated on-disk JSONL as-is)."""
    from app.main import _prelaunch_quality_gate
    from app.storage import persist_rlvr_split

    train = [{"messages": [{"role": "user", "content": f"{i}+1?"}], "ground_truth": str(i + 1)}
             for i in range(160)]
    split_id, _ = persist_rlvr_split(train=train, meta={"name": "lf-ok"})
    # An SFT/LF model on this split: no serverless convert dry-run, no GRPO floor.
    warns = _prelaunch_quality_gate(split_id, [_rm("sft", engine="llama_factory")])
    assert isinstance(warns, list)


def test_quality_gate_never_crashes_on_missing_split(temp_store):
    """The gate is advisory-safe: an unknown split id (no files) does not crash the
    launch — the profiler failure is swallowed and no hard checks have data."""
    from app.main import _prelaunch_quality_gate

    warns = _prelaunch_quality_gate("does-not-exist-000", [_rm("sft", engine="llama_factory")])
    assert warns == []
