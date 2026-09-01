# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Engine seam — additive multi-engine dispatch.

Contract under test: the default "llama_factory" engine is byte-identical to the
original inline launch (same orchestrate calls, same return dict), non-default
engines are flag-gated, and the ModelSpec.engine field defaults so every
existing model + persisted artifact is unchanged.
"""
import pytest

from app.catalog import get_model, list_models
from app.engines import DEFAULT_ENGINE, EngineNotEnabled, get_engine
from app.engines.base import engine_enabled


# --- registry / resolution -------------------------------------------------

def test_default_engine_is_llama_factory():
    assert DEFAULT_ENGINE == "llama_factory"


def test_get_engine_none_resolves_to_default():
    eng = get_engine(None)
    assert eng.name == "llama_factory"
    assert get_engine("").name == "llama_factory"
    assert get_engine("llama_factory").name == "llama_factory"


def test_unknown_engine_raises_valueerror():
    with pytest.raises(ValueError):
        get_engine("totally-made-up-engine")


def test_default_engine_always_enabled():
    assert engine_enabled(None) is True
    assert engine_enabled("llama_factory") is True


# --- feature-flag gating for non-default engines ---------------------------

def test_sagemaker_serverless_enabled_by_default(temp_store):
    # The serverless engine ships ON: with no saved config it resolves and
    # engine_enabled reports True (so list_models surfaces its models).
    eng = get_engine("sagemaker_serverless")
    assert eng.name == "sagemaker_serverless"
    assert engine_enabled("sagemaker_serverless") is True


def test_sagemaker_serverless_disabled_by_saved_config(temp_store):
    # The saved config toggle is the single source of truth: persisting OFF gates
    # the engine — requesting it raises EngineNotEnabled and engine_enabled is False.
    from app.aws_config import save_config

    save_config({"enableSagemakerServerless": False})
    with pytest.raises(EngineNotEnabled):
        get_engine("sagemaker_serverless")
    assert engine_enabled("sagemaker_serverless") is False


# --- ModelSpec.engine field is additive ------------------------------------

def test_every_catalog_model_defaults_to_llama_factory():
    for m in list_models():
        assert m["engine"] == "llama_factory"


def test_model_to_dict_carries_engine():
    d = get_model("qwen3-4b").to_dict()
    assert d["engine"] == "llama_factory"


# --- llama_factory engine is byte-identical to the original inline launch ---

def test_launch_routes_through_engine_with_identical_args(monkeypatch):
    """orchestrate.launch_training_job must call the SAME orchestrate functions,
    in the same way, the inline code did — proving the default path is unchanged."""
    from app import orchestrate as orch
    from app.catalog import Hyperparams

    cfg = object()
    calls = {}

    class _FakeEst:
        class latest_training_job:
            name = "slm-qwen3-4b-split-stamp"

        def fit(self, inputs, wait):
            calls["fit_inputs"] = inputs
            calls["fit_wait"] = wait

    monkeypatch.setattr(orch, "load_aws_config", lambda: cfg)
    # _session returns (sm_sess, boto_sess) in the real code.
    monkeypatch.setattr(orch, "_session", lambda c: ("SM_SESS", "BOTO"))
    monkeypatch.setattr(orch, "_job_name",
                        lambda mid, sid, st, method="lora": f"JOB::{mid}::{method}")
    monkeypatch.setattr(orch, "resolve_image_uri", lambda c, m: "IMG_FROM_TIER")
    monkeypatch.setattr(orch, "upload_job_inputs",
                        lambda sm, c, m, hp, sid, jn, use_spot=False: {
                            "dataset": "s3://d", "config": "s3://c", "base": "s3://base",
                        })

    def _fake_build_estimator(c, sm, m, jn, it, mrs, out, use_spot=False,
                              checkpoint_s3=None, image_uri=None):
        calls["build"] = dict(job_name=jn, instance_type=it, max_run=mrs,
                              output=out, use_spot=use_spot, image_uri=image_uri)
        return _FakeEst()

    monkeypatch.setattr(orch, "build_estimator", _fake_build_estimator)
    monkeypatch.setattr(orch, "TrainingInput", lambda uri, input_mode: f"TI({uri})")

    hp = Hyperparams()  # default lora
    out = orch.launch_training_job(
        model_id="qwen3-4b", split_id="split", hp=hp,
        instance_type="ml.g5.4xlarge", stamp="stamp", max_run_seconds=1234,
    )

    # job name uses the model's id + method; image resolved from the tier.
    assert calls["build"]["job_name"] == "JOB::qwen3-4b::lora"
    assert calls["build"]["image_uri"] == "IMG_FROM_TIER"
    assert calls["build"]["instance_type"] == "ml.g5.4xlarge"
    assert calls["build"]["max_run"] == 1234
    # two channels, wait=False — exactly as the inline launch did.
    assert set(calls["fit_inputs"].keys()) == {orch.DATASET_CHANNEL, orch.CONFIG_CHANNEL}
    assert calls["fit_wait"] is False
    # return dict preserves the original keys (+ engine tag).
    assert out["jobName"] == "slm-qwen3-4b-split-stamp"
    assert out["imageUri"] == "IMG_FROM_TIER"
    assert out["outputS3"] == "s3://base/output"
    assert out["engine"] == "llama_factory"


def test_unknown_model_still_raises(monkeypatch):
    from app import orchestrate as orch
    from app.catalog import Hyperparams

    with pytest.raises(ValueError):
        orch.launch_training_job(
            model_id="does-not-exist", split_id="s", hp=Hyperparams(),
            instance_type="ml.g5.2xlarge", stamp="x",
        )


# --- unified checkpointing (on-demand AND spot sync to S3) ------------------

def _capture_estimator_kwargs(monkeypatch):
    """Patch orchestrate.Estimator to capture the kwargs build_estimator passes."""
    from app import orchestrate as orch
    captured = {}

    class _Est:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(orch, "Estimator", _Est)
    monkeypatch.setattr(orch, "_job_environment", lambda: {})
    monkeypatch.setattr(orch, "_job_tags", lambda: [])
    monkeypatch.setattr(orch, "resolve_image_uri", lambda c, m: "IMG")
    return captured


def test_build_estimator_checkpoints_on_demand_and_spot(monkeypatch):
    """Checkpointing is UNCONDITIONAL now: an on-demand job ALSO gets
    checkpoint_s3_uri/checkpoint_local_path (so a failure can resume), but NO
    use_spot_instances/max_wait. Spot adds those two on top."""
    from app import orchestrate as orch
    captured = _capture_estimator_kwargs(monkeypatch)
    cfg = type("C", (), {"role_arn": "arn:role"})()
    model = get_model("qwen3-1.7b")

    # on-demand: checkpoint dir set, but not a spot job
    orch.build_estimator(cfg, "SM", model, "job", "ml.g5.2xlarge", 3600,
                         "s3://o", use_spot=False, checkpoint_s3="s3://ck", image_uri="IMG")
    assert captured["checkpoint_s3_uri"] == "s3://ck"
    assert captured["checkpoint_local_path"] == "/opt/ml/checkpoints"
    assert "use_spot_instances" not in captured  # NOT a spot job
    assert "max_wait" not in captured

    # spot: checkpoint dir PLUS spot capacity + max_wait
    captured.clear()
    orch.build_estimator(cfg, "SM", model, "job", "ml.g5.2xlarge", 3600,
                         "s3://o", use_spot=True, checkpoint_s3="s3://ck", image_uri="IMG")
    assert captured["checkpoint_s3_uri"] == "s3://ck"
    assert captured["use_spot_instances"] is True
    assert captured["max_wait"] >= 3600
