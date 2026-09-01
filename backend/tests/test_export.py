# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Model export: license-driven manifest, presigned URL, and the deploy bundle.

All AWS is stubbed — these check the export DECISIONS (which weights subdir, the
S3 artifact URI, the bundle contents), not real S3/SageMaker calls.
"""
import json
import zipfile

import pytest

from app.catalog import DecodingParams, Hyperparams


@pytest.fixture
def race_with_trained(temp_store, monkeypatch):
    """A persisted race whose entries have train_job set (as after launch)."""
    from app import race as race_mod

    monkeypatch.setattr(race_mod, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(race_mod, "launch_eval_job", lambda **kw: {"jobName": "eval-x"})
    monkeypatch.setattr(race_mod, "launch_base_eval_job",
                        lambda **kw: {"jobName": f"base-{kw['model_id']}"})
    monkeypatch.setattr(race_mod, "split_dir", lambda s: "/tmp/fake")

    # One ungated (qwen) + one gated (llama) model so both modes are exercised.
    rms = [race_mod.RaceModel(model_id=m, hp=Hyperparams()) for m in ("qwen3-0.6b", "llama-3.1-8b")]
    race = race_mod.start_race("split-x", rms, DecodingParams(), "20260610-1", name="export-test")
    return race_mod, race


@pytest.fixture
def stub_aws(monkeypatch):
    """Stub the AWS surface export.py touches: config, session, and presign."""
    from app import export as exp

    class _Cfg:
        bucket = "test-bucket"
        region = "us-east-1"

        @property
        def s3_prefix(self):
            return "s3://test-bucket/slm-platform"

    monkeypatch.setattr(exp, "load_aws_config", lambda: _Cfg())
    # describe_training_job raises → falls back to the conventional output path.
    monkeypatch.setattr(exp, "_session", lambda cfg: (_ for _ in ()).throw(RuntimeError("no aws")))
    return exp


def test_ungated_model_exports_merged(race_with_trained, stub_aws):
    _, race = race_with_trained
    m = stub_aws.build_manifest(race.race_id, "qwen3-0.6b")
    assert m.deploy_mode == "merged"
    assert m.weights_subdir == "merged"
    assert m.gated is False
    assert m.hf_base_model == "Qwen/Qwen3-0.6B"
    # Fell back to the conventional artifact path (describe stubbed to fail).
    assert m.artifact_s3_uri.endswith("/output/model.tar.gz")
    assert "test-bucket" in m.artifact_s3_uri
    # Deploy image = the LLaMA-Factory tag the model's tier maps to (stable→0.9.4),
    # so the inference container matches the training transformers version.
    assert m.base_image_tag == "0.9.4"
    assert m.to_dict()["baseImageTag"] == "0.9.4"


def test_gated_model_exports_adapter(race_with_trained, monkeypatch):
    """A GATED LLaMA-Factory model ships the LoRA adapter. Since checkpointing was
    unified, the adapter is in the synced checkpoint prefix's latest checkpoint-N/
    dir (NOT model.tar.gz, which now holds only merged/) — so the manifest resolves
    that prefix and delivers it as loose files (engine=sagemaker_serverless path)."""
    from app import export as exp

    race_mod, race = race_with_trained
    # record a checkpoint prefix on the gated entry (as a real launch now does)
    for e in race.entries:
        if e.model_id == "llama-3.1-8b":
            e.checkpoint_s3 = "s3://test-bucket/slm-platform/jobs/train-llama-3.1-8b-x/checkpoints"
    race_mod._save(race)

    class _Cfg:
        bucket = "test-bucket"
        region = "us-east-1"
        profile = None

    # S3 paginator returns two checkpoint-N/ dirs as CommonPrefixes; latest wins.
    ckpt_prefix = "slm-platform/jobs/train-llama-3.1-8b-x/checkpoints/"

    class _Pag:
        def paginate(self, Bucket, Prefix, Delimiter):  # noqa: N803
            yield {"CommonPrefixes": [
                {"Prefix": ckpt_prefix + "checkpoint-20/"},
                {"Prefix": ckpt_prefix + "checkpoint-48/"},  # highest step
            ]}

    class _S3:
        def get_paginator(self, _n):
            return _Pag()

    class _Boto:
        def client(self, n, **_k):
            return _S3()

    monkeypatch.setattr(exp, "load_aws_config", lambda: _Cfg())
    monkeypatch.setattr(exp, "_session", lambda cfg: (None, _Boto()))

    m = exp.build_manifest(race.race_id, "llama-3.1-8b")
    assert m.deploy_mode == "adapter"
    assert m.weights_subdir == "adapter"
    assert m.gated is True
    # resolves the LATEST checkpoint dir + delivers via the adapter-only loose-file path
    assert m.artifact_s3_uri.endswith("/checkpoint-48")
    assert m.engine == "sagemaker_serverless_adapter"  # internal adapter-filter variant
    # customer-facing manifest collapses it to the engine deploy.sh understands
    assert m.to_dict()["engine"] == "sagemaker_serverless"


@pytest.fixture
def race_gated_full(temp_store, monkeypatch):
    """A persisted race whose GATED ≤2B entry was trained with FULL fine-tuning
    (finetuning_type=full) — the Option-B case (merged gated weights, license gate)."""
    from app import race as race_mod

    monkeypatch.setattr(race_mod, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(race_mod, "launch_eval_job", lambda **kw: {"jobName": "eval-x"})
    monkeypatch.setattr(race_mod, "launch_base_eval_job",
                        lambda **kw: {"jobName": f"base-{kw['model_id']}"})
    monkeypatch.setattr(race_mod, "split_dir", lambda s: "/tmp/fake")

    # llama-3.2-1b is gated AND ≤2B, so full is in its allowed_methods.
    rms = [race_mod.RaceModel(model_id="llama-3.2-1b",
                              hp=Hyperparams(finetuning_type="full", stage="sft"))]
    race = race_mod.start_race("split-x", rms, DecodingParams(), "20260619-1", name="gated-full-test")
    return race_mod, race


def test_gated_full_ft_exports_merged_not_adapter(race_gated_full, stub_aws):
    """A GATED FULL fine-tune has NO adapter → it must ship MERGED weights (the
    merged tarball path), NOT the gated-adapter path, and flag the license gate."""
    _, race = race_gated_full
    m = stub_aws.build_manifest(race.race_id, "llama-3.2-1b")
    assert m.gated is True
    assert m.deploy_mode == "merged"       # NOT "adapter" — there is no adapter
    assert m.weights_subdir == "merged"
    assert m.engine == "llama_factory"     # merged tarball, not the serverless-adapter variant
    assert m.requires_license_acceptance is True
    assert m.to_dict()["requiresLicenseAcceptance"] is True


def test_gated_full_ft_download_gated_until_license_accepted(race_gated_full, stub_aws, monkeypatch):
    """export_info withholds the presigned weights until the license is accepted."""
    monkeypatch.setattr(stub_aws, "presign_artifact",
                        lambda uri, ttl=0: "https://signed.example/model.tar.gz")
    _, race = race_gated_full
    # Not accepted → no URL, just the license-required marker.
    gated = stub_aws.export_info(race.race_id, "llama-3.2-1b", license_accepted=False)
    assert gated.get("licenseRequired") is True
    assert gated.get("licenseModel") == "meta-llama/Llama-3.2-1B-Instruct"
    assert "weightsUrl" not in gated and "weightsFiles" not in gated
    # Accepted → the presigned tarball URL is minted.
    ok = stub_aws.export_info(race.race_id, "llama-3.2-1b", license_accepted=True)
    assert not ok.get("licenseRequired")
    assert ok.get("weightsUrl")


def test_gated_legacy_entry_falls_back_to_tarball(race_with_trained, monkeypatch):
    """A gated LLaMA-Factory entry with NO checkpoint prefix (trained before the
    unify, or checkpoint never synced) must NOT hard-fail — it falls back to the
    legacy model.tar.gz adapter/ delivery so historical winners still export."""
    from app import export as exp

    race_mod, race = race_with_trained
    for e in race.entries:  # gated entry has checkpoint_s3 = None (default)
        if e.model_id == "llama-3.1-8b":
            assert e.checkpoint_s3 is None
    race_mod._save(race)

    class _Cfg:
        bucket = "test-bucket"
        region = "us-east-1"
        profile = None

    class _EmptyPag:
        def paginate(self, Bucket, Prefix, Delimiter):  # noqa: N803
            yield {"CommonPrefixes": []}  # no checkpoint-N/ dirs anywhere

    class _S3:
        def get_paginator(self, _n):
            return _EmptyPag()

    class _Boto:
        def client(self, n, **_k):
            return _S3()

    monkeypatch.setattr(exp, "load_aws_config", lambda: _Cfg())
    # describe fails → _artifact_uri falls back to the conventional tarball path
    monkeypatch.setattr(exp, "_session", lambda cfg: (None, _Boto()))

    m = exp.build_manifest(race.race_id, "llama-3.1-8b")
    assert m.gated is True
    assert m.deploy_mode == "adapter"
    # NO checkpoint prefix → legacy tarball path (engine stays llama_factory)
    assert m.engine == "llama_factory"
    assert m.artifact_s3_uri.endswith("/output/model.tar.gz")


def test_presign_prefix_adapter_only_filters_optimizer_state(stub_aws_serverless):
    """The adapter-only filter drops optimizer/scheduler/RNG/trainer-state files
    from a raw HF-Trainer checkpoint dir, keeping only adapter + tokenizer files."""
    exp = stub_aws_serverless
    # reuse the adapter prefix from the fixture but add training-only junk
    art_key = _SERVERLESS_ART[len("s3://test-bucket/"):]
    ckpt = f"{art_key}/checkpoints/hf/"
    # monkeypatch a richer listing for this prefix
    objs = {ckpt: [
        {"Key": ckpt + "adapter_config.json", "Size": 1053},
        {"Key": ckpt + "adapter_model.safetensors", "Size": 36981072},
        {"Key": ckpt + "tokenizer.json", "Size": 11422778},
        {"Key": ckpt + "optimizer.pt", "Size": 70059562},   # must be dropped
        {"Key": ckpt + "scheduler.pt", "Size": 1064},        # must be dropped
        {"Key": ckpt + "rng_state.pth", "Size": 14244},      # must be dropped
        {"Key": ckpt + "trainer_state.json", "Size": 2201},  # must be dropped
    ]}
    sess = _FakeBotoSession(_SERVERLESS_ART, objs)
    import app.export as _exp
    _exp_session = _exp._session
    _exp_loadcfg = _exp.load_aws_config
    try:
        _exp._session = lambda cfg: (None, sess)
        files = exp.presign_prefix(f"{_SERVERLESS_ART}/checkpoints/hf/", adapter_only=True)
        names = {f["name"] for f in files}
        assert names == {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json"}
        assert "optimizer.pt" not in names and "scheduler.pt" not in names
    finally:
        _exp._session = _exp_session
        _exp.load_aws_config = _exp_loadcfg


def test_export_unknown_race_or_model_raises(race_with_trained, stub_aws):
    _, race = race_with_trained
    with pytest.raises(stub_aws.ExportError):
        stub_aws.build_manifest("nope", "qwen3-0.6b")
    with pytest.raises(stub_aws.ExportError):
        stub_aws.build_manifest(race.race_id, "not-in-race")


def test_export_entry_without_train_job_raises(race_with_trained, stub_aws):
    race_mod, race = race_with_trained
    # Wipe the train job to simulate a not-yet-launched entry.
    race.entries[0].train_job = None
    race_mod._save(race)
    with pytest.raises(stub_aws.ExportError):
        stub_aws.build_manifest(race.race_id, "qwen3-0.6b")


def test_presign_rejects_non_s3_uri(stub_aws):
    with pytest.raises(stub_aws.ExportError):
        stub_aws.presign_artifact("https://example.com/x")


def test_bundle_is_small_zip_with_manifest(race_with_trained, stub_aws, monkeypatch):
    from app import bundle, export

    # Stub the presign so export_info doesn't hit S3.
    monkeypatch.setattr(export, "presign_artifact", lambda uri, ttl=0: "https://signed.example/model.tar.gz")
    _, race = race_with_trained

    filename, data = bundle.build_bundle(race.race_id, "qwen3-0.6b")
    assert filename == "slm-deploy-qwen3-0-6b.zip"
    # Tiny — scripts only, no weights.
    assert len(data) < 100_000

    import io

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert {"deploy.sh", "inference.py", "Dockerfile", "README.md", "manifest.json"} <= names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["deployMode"] == "merged"
        assert manifest["weightsUrl"] == "https://signed.example/model.tar.gz"


# --- serverless engine export ------------------------------------------------
# Serverless winners have a different artifact shape: an UNCOMPRESSED S3 prefix
# (no tarball), with the loadable model under checkpoints/hf_merged (ungated) or
# the LoRA adapter under checkpoints/hf (gated). export.py must resolve the right
# prefix per-license and presign EACH file.

# A completed serverless job's artifact ROOT (describe → S3ModelArtifacts).
_SERVERLESS_ART = (
    "s3://test-bucket/slm-platform/users/u/jobs/"
    "slm-qwen3-1-7b-serverless-x/output/slm-qwen3-1-7b-serverless-x/output/model"
)


class _FakePaginator:
    """Mimics boto3's list_objects_v2 paginator for one fixed prefix listing."""

    def __init__(self, objects_by_prefix):
        self._objects_by_prefix = objects_by_prefix

    def paginate(self, Bucket, Prefix):  # noqa: N803 — boto kwarg names
        contents = self._objects_by_prefix.get(Prefix, [])
        # Split across two pages to exercise pagination.
        yield {"Contents": contents[:1]}
        yield {"Contents": contents[1:]}


class _FakeS3:
    def __init__(self, objects_by_prefix):
        self._objects_by_prefix = objects_by_prefix

    def get_paginator(self, _name):
        return _FakePaginator(self._objects_by_prefix)

    def generate_presigned_url(self, _op, Params, ExpiresIn):  # noqa: N803
        return f"https://signed.example/{Params['Key']}?ttl={ExpiresIn}"


class _FakeSageMaker:
    def __init__(self, artifact):
        self._artifact = artifact

    def describe_training_job(self, TrainingJobName):  # noqa: N803
        return {"ModelArtifacts": {"S3ModelArtifacts": self._artifact}}


class _FakeBotoSession:
    def __init__(self, artifact, objects_by_prefix):
        self._artifact = artifact
        self._objects_by_prefix = objects_by_prefix

    def client(self, name, **_kw):
        if name == "sagemaker":
            return _FakeSageMaker(self._artifact)
        if name == "s3":
            return _FakeS3(self._objects_by_prefix)
        raise AssertionError(f"unexpected client {name}")


@pytest.fixture
def serverless_race(temp_store, monkeypatch):
    """A persisted race with two serverless entries (ungated qwen + gated llama),
    each with train_job set as it would be after the worker launch."""
    from app import race as race_mod

    monkeypatch.setattr(race_mod, "split_dir", lambda s: "/tmp/fake")
    # Serverless launch dispatches to a worker; no-op it so start_race just persists.
    monkeypatch.setattr(race_mod, "dispatch_worker", lambda *a, **k: True)

    sv = Hyperparams(engine="sagemaker_serverless")
    rms = [race_mod.RaceModel(model_id=m, hp=sv) for m in ("qwen3-1.7b", "llama-3.2-3b")]
    race = race_mod.start_race("split-x", rms, DecodingParams(), "20260615-1", name="sv-export")
    for e in race.entries:
        e.train_job = f"train-{e.model_id}"
    race_mod._save(race)
    return race_mod, race


@pytest.fixture
def stub_aws_serverless(monkeypatch):
    """Config + a fake boto session covering describe_training_job (prefix
    resolution) AND the S3 paginator (per-file presign)."""
    from app import export as exp

    class _Cfg:
        bucket = "test-bucket"
        region = "us-east-1"
        profile = None
        role_arn = "arn:aws:iam::123:role/x"

    # boto's paginate(Prefix=…) receives the BARE S3 key (no s3://bucket/), so the
    # fake's lookup keys must be bare keys too.
    art_key = _SERVERLESS_ART[len("s3://test-bucket/"):]
    merged_prefix = f"{art_key}/checkpoints/hf_merged/"
    adapter_prefix = f"{art_key}/checkpoints/hf/"
    objects = {
        merged_prefix: [
            {"Key": merged_prefix, "Size": 0},  # dir marker — must be skipped
            {"Key": merged_prefix + "config.json", "Size": 1336},
            {"Key": merged_prefix + "model.safetensors", "Size": 3554214752},
            {"Key": merged_prefix + "tokenizer.json", "Size": 11422778},
        ],
        adapter_prefix: [
            {"Key": adapter_prefix + "adapter_config.json", "Size": 1053},
            {"Key": adapter_prefix + "adapter_model.safetensors", "Size": 36981072},
        ],
    }
    sess = _FakeBotoSession(_SERVERLESS_ART, objects)
    monkeypatch.setattr(exp, "load_aws_config", lambda: _Cfg())
    monkeypatch.setattr(exp, "_session", lambda cfg: (None, sess))
    return exp


def test_serverless_ungated_resolves_merged_prefix(serverless_race, stub_aws_serverless):
    _, race = serverless_race
    m = stub_aws_serverless.build_manifest(race.race_id, "qwen3-1.7b")
    assert m.engine == "sagemaker_serverless"
    assert m.deploy_mode == "merged"
    # Prefix points at checkpoints/hf_merged (NOT a model.tar.gz).
    assert m.artifact_s3_uri.endswith("/checkpoints/hf_merged")
    assert not m.artifact_s3_uri.endswith(".tar.gz")
    # Deploy image still the LLaMA-Factory tag (eval bridge loads it on that image).
    assert m.base_image_tag == "0.9.4"
    assert m.to_dict()["engine"] == "sagemaker_serverless"


def test_serverless_gated_resolves_adapter_prefix(serverless_race, stub_aws_serverless):
    _, race = serverless_race
    m = stub_aws_serverless.build_manifest(race.race_id, "llama-3.2-3b")
    assert m.engine == "sagemaker_serverless"
    assert m.deploy_mode == "adapter"
    assert m.gated is True
    # Gated → the LoRA adapter dir (checkpoints/hf), never the merged weights.
    assert m.artifact_s3_uri.endswith("/checkpoints/hf")


def test_presign_prefix_lists_every_file_relative(stub_aws_serverless):
    files = stub_aws_serverless.presign_prefix(f"{_SERVERLESS_ART}/checkpoints/hf_merged/")
    names = {f["name"] for f in files}
    # The dir-marker key is skipped; names are RELATIVE to the prefix.
    assert names == {"config.json", "model.safetensors", "tokenizer.json"}
    assert all(f["url"].startswith("https://signed.example/") for f in files)
    big = next(f for f in files if f["name"] == "model.safetensors")
    assert big["size"] == 3554214752


def test_presign_prefix_raises_on_empty(stub_aws_serverless):
    with pytest.raises(stub_aws_serverless.ExportError):
        stub_aws_serverless.presign_prefix(f"{_SERVERLESS_ART}/nonexistent/")


def test_serverless_export_info_has_weights_files_not_url(serverless_race, stub_aws_serverless):
    _, race = serverless_race
    info = stub_aws_serverless.export_info(race.race_id, "qwen3-1.7b")
    assert info["engine"] == "sagemaker_serverless"
    assert "weightsFiles" in info and "weightsUrl" not in info
    assert {f["name"] for f in info["weightsFiles"]} == {
        "config.json", "model.safetensors", "tokenizer.json"
    }
    assert info["weightsTtlSeconds"] == stub_aws_serverless.PRESIGN_TTL_SECONDS


def test_serverless_prefix_describe_unavailable_raises(serverless_race, monkeypatch):
    """No conventional fallback for serverless — a failed describe must raise, not
    hand back a wrong/guessed prefix."""
    from app import export as exp

    class _Cfg:
        bucket = "test-bucket"
        region = "us-east-1"
        profile = None
        role_arn = "arn:aws:iam::123:role/x"

    monkeypatch.setattr(exp, "load_aws_config", lambda: _Cfg())
    monkeypatch.setattr(exp, "_session",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("no aws")))
    _, race = serverless_race
    with pytest.raises(exp.ExportError):
        exp.build_manifest(race.race_id, "qwen3-1.7b")
