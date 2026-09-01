# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Per-user HF token storage — isolation, legacy migration, worker propagation."""
import json

import pytest

from app import secrets as sec
from app import tenancy


@pytest.fixture
def fake_secret(monkeypatch):
    """In-memory stand-in for the Secrets Manager blob, so tests don't hit AWS.
    Returns a mutable holder whose .raw is the SecretString the secret holds."""
    holder = {"raw": ""}
    monkeypatch.setenv("SLM_HF_SECRET_NAME", "test/hf-token")

    def fake_read():
        raw = holder["raw"].strip()
        if not raw:
            return {}
        if raw.startswith("{"):
            return {k: str(v) for k, v in json.loads(raw).items() if isinstance(v, str)}
        return {tenancy.DEFAULT_TENANT: raw}

    def fake_write(blob):
        holder["raw"] = json.dumps(blob)

    monkeypatch.setattr(sec, "_read_blob", fake_read)
    monkeypatch.setattr(sec, "_write_blob", fake_write)
    return holder


@pytest.fixture
def as_tenant():
    """Run as a given tenant, restoring after."""
    tokens = []

    def _set(t):
        tokens.append(tenancy.set_tenant(t))

    yield _set
    for tok in reversed(tokens):
        tenancy._current.reset(tok)


def test_tokens_are_isolated_per_tenant(fake_secret, as_tenant):
    as_tenant("alice")
    sec.set_hf_token("hf_alice_tok")
    as_tenant("bob")
    # Bob has no token yet — alice's is not visible to him.
    assert sec.get_hf_token() is None
    assert sec.hf_token_is_set() is False
    sec.set_hf_token("hf_bob_tok")

    as_tenant("alice")
    assert sec.get_hf_token() == "hf_alice_tok"
    as_tenant("bob")
    assert sec.get_hf_token() == "hf_bob_tok"


def test_non_hf_prefix_is_rejected(fake_secret, as_tenant):
    as_tenant("alice")
    sec.set_hf_token("not-a-real-token")  # stored, but...
    assert sec.get_hf_token() is None  # ...rejected for lacking the hf_ prefix


def test_legacy_raw_token_reads_as_default_tenant(fake_secret, as_tenant):
    # Secret historically held a RAW token string (the single global token).
    fake_secret["raw"] = "hf_legacy_global"
    as_tenant(tenancy.DEFAULT_TENANT)
    assert sec.get_hf_token() == "hf_legacy_global"
    # A named tenant BORROWS the legacy global token via the shared fallback
    # (so new users work before setting their own)...
    as_tenant("carol")
    assert sec.get_hf_token() == "hf_legacy_global"
    # ...but it is never reported as THEIR token (Settings/banner stay honest).
    assert sec.hf_token_is_set() is False


def test_fallback_borrows_shared_default_slot(fake_secret, as_tenant, monkeypatch):
    monkeypatch.delenv("SLM_HF_FALLBACK_TENANTS", raising=False)
    as_tenant(tenancy.DEFAULT_TENANT)
    sec.set_hf_token("hf_shared_tok")
    as_tenant("newuser")
    # Effective token = the shared slot's; own-token status stays False.
    assert sec.get_hf_token() == "hf_shared_tok"
    assert sec.hf_token_is_set() is False
    # Their own token, once set, wins over the fallback.
    sec.set_hf_token("hf_newuser_tok")
    assert sec.get_hf_token() == "hf_newuser_tok"
    assert sec.hf_token_is_set() is True


def test_another_users_token_is_never_borrowed(fake_secret, as_tenant, monkeypatch):
    """Only the SHARED default slot is a fallback: one user's personal token must
    never be lent to another user who hasn't set one."""
    monkeypatch.delenv("SLM_HF_FALLBACK_TENANTS", raising=False)
    as_tenant("alice")
    sec.set_hf_token("hf_alice_tok")
    as_tenant("newuser")
    assert sec.get_hf_token() is None
    assert sec.hf_token_is_set() is False


def test_fallback_disabled_by_empty_env(fake_secret, as_tenant, monkeypatch):
    monkeypatch.setenv("SLM_HF_FALLBACK_TENANTS", "")
    # Even the shared slot is withheld when the fallback list is empty.
    as_tenant(tenancy.DEFAULT_TENANT)
    sec.set_hf_token("hf_shared_tok")
    as_tenant("newuser")
    assert sec.get_hf_token() is None


def test_fallback_env_override_order(fake_secret, as_tenant, monkeypatch):
    monkeypatch.setenv("SLM_HF_FALLBACK_TENANTS", "alice, bob")
    as_tenant("bob")
    sec.set_hf_token("hf_bob_tok")
    as_tenant("newuser")
    # alice has no token → falls through to bob's.
    assert sec.get_hf_token() == "hf_bob_tok"
    # An invalid owner token is never lent out.
    as_tenant("alice")
    sec.set_hf_token("garbage-token")
    as_tenant("newuser")
    assert sec.get_hf_token() == "hf_bob_tok"


def test_first_write_migrates_legacy_blob_to_map(fake_secret, as_tenant):
    fake_secret["raw"] = "hf_legacy_global"  # legacy raw form
    as_tenant("dave")
    sec.set_hf_token("hf_dave_tok")
    # Blob is now a JSON map preserving the legacy token under DEFAULT_TENANT.
    blob = json.loads(fake_secret["raw"])
    assert blob["dave"] == "hf_dave_tok"
    assert blob[tenancy.DEFAULT_TENANT] == "hf_legacy_global"


def test_no_secret_configured_degrades(monkeypatch):
    monkeypatch.delenv("SLM_HF_SECRET_NAME", raising=False)
    assert sec.get_hf_token() is None
    assert sec.hf_token_is_set() is False
    sec.set_hf_token("hf_x")  # no-op, no raise


def test_dispatch_stamps_current_tenant(monkeypatch):
    """dispatch_worker injects the caller's tenant so the worker can rebind it."""
    from app import dispatch

    monkeypatch.setenv("SLM_WORKER_FUNCTION", "worker-fn")
    captured = {}

    class _Lam:
        def invoke(self, **kw):
            captured.update(json.loads(kw["Payload"]))

    monkeypatch.setattr("boto3.client", lambda *a, **k: _Lam())
    tok = tenancy.set_tenant("erin")
    try:
        assert dispatch.dispatch_worker({"task": "judge", "evalJob": "e1"}) is True
    finally:
        tenancy._current.reset(tok)
    assert captured["tenant"] == "erin"
    assert captured["task"] == "judge"
