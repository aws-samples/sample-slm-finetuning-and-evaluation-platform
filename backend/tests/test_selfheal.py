# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Self-healing image selection — deterministic failure triage + tier routing."""
from app import selfheal
from app.catalog import get_model


# --- deterministic failure classification ----------------------------------

def test_losskwargs_import_means_newer_stack():
    # The exact failure that triggered the multi-image design (Phi-4-mini on 0.9.4).
    c = selfheal.classify_failure(
        "ImportError: cannot import name 'LossKwargs' from transformers.utils"
    )
    assert c["needsNewerStack"] is True
    assert c["category"] == "stack_too_old"
    assert c["source"] == "deterministic"


def test_requires_transformers_version_means_newer_stack():
    c = selfheal.classify_failure("This model requires transformers>=5.0.0")
    assert c["needsNewerStack"] is True
    assert c["source"] == "deterministic"


def test_unknown_template_arg_parser_means_newer_stack():
    c = selfheal.classify_failure(
        "Some keys are not used by the HfArgumentParser: ['some_new_key']"
    )
    assert c["needsNewerStack"] is True


def test_oom_does_not_need_newer_stack():
    c = selfheal.classify_failure("RuntimeError: CUDA out of memory. Tried to allocate ...")
    assert c["needsNewerStack"] is False
    assert c["category"] == "oom"


def test_capacity_does_not_need_newer_stack():
    c = selfheal.classify_failure(
        "Insufficient capacity error from EC2 while launching instances, retrying!"
    )
    assert c["needsNewerStack"] is False
    assert c["category"] == "capacity"


def test_is_capacity_stall_matches_capacity_waits_only():
    # Real capacity-stall messages → True (the spot→on-demand fallback trigger).
    assert selfheal.is_capacity_stall("Insufficient capacity error from EC2 ... retrying!")
    assert selfheal.is_capacity_stall("InsufficientInstanceCapacity")
    assert selfheal.is_capacity_stall("Still waiting for spot capacity")
    # Benign provisioning statuses → False (must NOT trigger a conversion).
    assert not selfheal.is_capacity_stall("Downloading the training image")
    assert not selfheal.is_capacity_stall("Starting")
    assert not selfheal.is_capacity_stall("Training")
    assert not selfheal.is_capacity_stall(None)
    assert not selfheal.is_capacity_stall("")


def test_empty_reason_is_unknown_no_escalation():
    c = selfheal.classify_failure(None)
    assert c["needsNewerStack"] is False
    assert c["source"] == "deterministic"


# --- tier routing ----------------------------------------------------------

def test_next_tier_up_from_stable_is_latest():
    assert selfheal.next_tier_up("stable") == "latest"


def test_next_tier_up_from_newest_is_none():
    assert selfheal.next_tier_up("latest") is None


def test_build_project_name_convention():
    assert selfheal.build_project_for("stable") == "slm-platform-training-image-build-094"
    assert selfheal.build_project_for("latest") == "slm-platform-training-image-build-095"


# --- adhoc-tier (re)build fallback -----------------------------------------
# A tier added at RUNTIME via "Build <tag>" has NO per-tier CodeBuild project —
# its image was built by the shared adhoc project. Rebuild/status must route
# through the adhoc project (keyed to the tag) instead of 502-ing.

class _FakeCB:
    """Minimal CodeBuild client: knows which project names 'exist' and records
    the start_build call so we can assert the project + env overrides used."""

    def __init__(self, existing_projects):
        self._existing = set(existing_projects)
        self.started = None

    def batch_get_projects(self, names):  # noqa: N803 — boto kwarg
        return {"projects": [{"name": n} for n in names if n in self._existing]}

    def start_build(self, **kwargs):
        self.started = kwargs
        return {"build": {"id": "build-123", "buildStatus": "IN_PROGRESS"}}


def _patch_session(monkeypatch, cb):
    import types as _t
    monkeypatch.setattr(selfheal, "load_aws_config",
                        lambda: _t.SimpleNamespace(region="us-east-1", profile=None))
    # selfheal does `from .orchestrate import _session` inside the function.
    import app.orchestrate as orch
    monkeypatch.setattr(orch, "_session",
                        lambda cfg: (None, _t.SimpleNamespace(client=lambda svc: cb)))


def test_trigger_build_per_tier_project_when_it_exists(monkeypatch):
    """Built-in tier (stable→...-094): the per-tier project exists → use it, no overrides."""
    cb = _FakeCB({"slm-platform-training-image-build-094"})
    _patch_session(monkeypatch, cb)
    out = selfheal.trigger_image_build("stable")
    assert out["project"] == "slm-platform-training-image-build-094"
    assert "environmentVariablesOverride" not in cb.started  # per-tier build, no override


def test_trigger_build_falls_back_to_adhoc_for_runtime_tier(monkeypatch):
    """A runtime tier whose per-tier project does NOT exist → route through the
    shared adhoc project with TAG/LF_BASE_TAG overrides (was a 502 before)."""
    cb = _FakeCB(set())  # no per-tier project exists
    _patch_session(monkeypatch, cb)
    # image_tiers() maps the tier name → tag; for an unknown tier the tag == name.
    out = selfheal.trigger_image_build("v096")
    assert out["project"] == selfheal.ADHOC_BUILD_PROJECT
    ov = {e["name"]: e["value"] for e in cb.started["environmentVariablesOverride"]}
    assert ov["TAG"] == "v096" and ov["LF_BASE_TAG"] == "v096"
    assert "VLLM_VERSION" in ov
    assert out["buildId"] == "build-123"


# --- diagnosis (no AWS; classification + routing only) ----------------------

def test_diagnose_escalates_phi4_mini_style_failure(monkeypatch):
    # Pretend the recommended image isn't built yet → action says build first.
    monkeypatch.setattr(selfheal, "image_exists_in_ecr", lambda tag: False)
    model = get_model("qwen3-1.7b")  # a stable-tier model
    d = selfheal.diagnose(model, "ImportError: cannot import name 'LossKwargs'")
    assert d["classification"]["needsNewerStack"] is True
    assert d["recommendedTier"] == "latest"
    assert d["imageReady"] is False
    assert d["action"] == "build_then_smoke_test"


def test_diagnose_ready_image_goes_straight_to_smoke_test(monkeypatch):
    monkeypatch.setattr(selfheal, "image_exists_in_ecr", lambda tag: True)
    model = get_model("qwen3-1.7b")
    d = selfheal.diagnose(model, "requires transformers>=5")
    assert d["action"] == "smoke_test"
    assert d["imageReady"] is True


def test_diagnose_oom_does_not_change_image():
    model = get_model("qwen3-1.7b")
    d = selfheal.diagnose(model, "CUDA out of memory")
    assert d["action"] == "no_image_change"
    assert d["recommendedTier"] is None


def test_diagnose_already_newest_tier(monkeypatch):
    model = get_model("phi-4-mini")  # pinned to latest
    assert getattr(model, "image_tag") == "latest"
    d = selfheal.diagnose(model, "cannot import name 'LossKwargs'")
    assert d["action"] == "already_newest"


def test_selfheal_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in ["/api/models/{model_id}/diagnose", "/api/images/{image_tag}/build"]:
        assert p in paths, f"missing route {p}"
