# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Presigned direct-to-S3 upload helpers (large dataset files)."""
import pytest

from app import uploads


def test_safe_name_strips_paths_and_unsafe_chars():
    assert uploads._safe_name("train.jsonl") == "train.jsonl"
    assert uploads._safe_name("../../etc/passwd") == "passwd"
    assert uploads._safe_name("a b/c$d.jsonl") == "c_d.jsonl"
    assert uploads._safe_name("") == "upload"


def test_fetch_rejects_keys_outside_upload_prefix(monkeypatch):
    # Guard: only objects under the uploads prefix may be read back, so a crafted
    # upload_id can't exfiltrate arbitrary bucket objects.
    with pytest.raises(ValueError):
        uploads.fetch_upload_bytes("slm-platform/state/config.json", 1024)
    with pytest.raises(ValueError):
        uploads.fetch_upload_bytes("../secrets", 1024)


def test_delete_upload_ignores_foreign_keys():
    # Never deletes outside the uploads prefix (no exception, just a no-op).
    uploads.delete_upload("slm-platform/jobs/something/model.tar.gz")


def test_owns_upload_rejects_other_tenant_keys():
    from app import tenancy
    # Acting as tenant "alice": her own tenant-scoped key is owned...
    tok = tenancy.set_tenant("alice")
    try:
        assert uploads._owns_upload("slm-platform/uploads/alice/20260701-1-ab12-train.jsonl")
        # ...but bob's key is NOT (cross-tenant read blocked).
        assert not uploads._owns_upload("slm-platform/uploads/bob/20260701-1-cd34-train.jsonl")
        # A legacy (no-tenant-segment) key is refused for a NON-default tenant.
        assert not uploads._owns_upload("slm-platform/uploads/20260701-train.jsonl")
    finally:
        tenancy._current.reset(tok)


def test_legacy_key_allowed_for_default_tenant():
    # With no tenant set (default), pre-tenant-scoping keys still read (back-compat).
    assert uploads._owns_upload("slm-platform/uploads/20260701-train.jsonl")


def test_fetch_rejects_other_tenant_key(monkeypatch):
    from app import tenancy
    tok = tenancy.set_tenant("alice")
    try:
        with pytest.raises(ValueError):
            uploads.fetch_upload_bytes("slm-platform/uploads/bob/x-y-train.jsonl", 1024)
    finally:
        tenancy._current.reset(tok)


def test_make_upload_url_key_is_tenant_scoped_and_random(monkeypatch):
    # The generated key embeds the tenant segment + a random suffix (unguessable).
    captured = {}

    class _FakeS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            captured["key"] = Params["Key"]
            return "https://example/put"

    class _Cfg:
        bucket = "b"

    monkeypatch.setattr(uploads, "load_aws_config", lambda: _Cfg())
    monkeypatch.setattr(uploads, "_session", lambda cfg: (None, type("S", (), {"client": lambda self, *a, **k: _FakeS3()})()))
    from app import tenancy
    tok = tenancy.set_tenant("alice")
    try:
        out = uploads.make_upload_url("train.jsonl", "20260701-120000")
    finally:
        tenancy._current.reset(tok)
    assert out["uploadId"].startswith("slm-platform/uploads/alice/")
    assert out["uploadId"].endswith("-train.jsonl")
    # Two calls → different keys (random suffix).
    tok2 = tenancy.set_tenant("alice")
    try:
        out2 = uploads.make_upload_url("train.jsonl", "20260701-120000")
    finally:
        tenancy._current.reset(tok2)
    assert out["uploadId"] != out2["uploadId"]
