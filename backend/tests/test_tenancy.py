# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Per-user tenancy: TenantStore scoping + the resolver gate.

These verify the isolation invariant WITHOUT auth wired: collection state is
per-tenant, root docs stay global, and the default tenant is a pure passthrough
(so existing un-prefixed state is untouched until multi-tenancy is enabled).
"""
from __future__ import annotations

import pytest


@pytest.fixture
def reset_tenant():
    """Restore the default tenant after a test mutates the contextvar."""
    from app import tenancy

    token = tenancy.set_tenant(tenancy.DEFAULT_TENANT)
    yield tenancy
    tenancy._current.reset(token)


def test_default_tenant_is_unprefixed_passthrough(temp_store, reset_tenant):
    """With the default tenant, a collection key lands at the historical path —
    no users/ prefix — so existing state needs no migration."""
    store = temp_store.get_store()
    wd = store.workdir("races", "r1")
    (wd / "race.json").write_text("{}")
    store.commit("races", "r1")

    # Written under <root>/races/r1, NOT <root>/users/.../races/r1.
    root = wd.parents[1]  # wd = root/races/r1
    assert (root / "races" / "r1" / "race.json").exists()
    assert not (root / "users").exists()
    assert store.list_keys("races") == ["r1"]


def test_named_tenants_are_isolated(temp_store, reset_tenant):
    """Two tenants writing the same collection+key never see each other."""
    store = temp_store.get_store()

    reset_tenant.set_tenant("alice")
    (store.workdir("races", "r1") / "race.json").write_text('{"o": "alice"}')
    store.commit("races", "r1")

    reset_tenant.set_tenant("bob")
    assert store.list_keys("races") == []  # bob sees nothing of alice's
    (store.workdir("races", "r1") / "race.json").write_text('{"o": "bob"}')
    store.commit("races", "r1")

    # Each tenant reads back only their own value.
    reset_tenant.set_tenant("alice")
    assert store.list_keys("races") == ["r1"]
    assert '"alice"' in (store.read_file("races", "r1", "race.json") or "")

    reset_tenant.set_tenant("bob")
    assert '"bob"' in (store.read_file("races", "r1", "race.json") or "")


def test_default_tenant_cannot_see_named_tenant_state(temp_store, reset_tenant):
    """Default-tenant state and a named tenant's state are separate namespaces."""
    store = temp_store.get_store()

    # Default tenant writes r1.
    (store.workdir("races", "r1") / "race.json").write_text("{}")
    store.commit("races", "r1")

    reset_tenant.set_tenant("carol")
    assert store.list_keys("races") == []  # carol's view is empty
    # And carol's own write doesn't leak back to default.
    (store.workdir("races", "r2") / "race.json").write_text("{}")
    store.commit("races", "r2")

    reset_tenant.set_tenant(reset_tenant.DEFAULT_TENANT)
    assert store.list_keys("races") == ["r1"]  # still only the default's race


def test_root_docs_are_global_across_tenants(temp_store, reset_tenant):
    """Root JSON docs (config/verifications/custom_models) are platform-global —
    every tenant reads/writes the same document."""
    store = temp_store.get_store()

    reset_tenant.set_tenant("alice")
    store.write_root_json("verifications.json", {"qwen3-4b": {"stable": "verified"}})

    reset_tenant.set_tenant("bob")
    # Bob sees alice's write — it's a shared fact, not per-user.
    assert store.read_root_json("verifications.json") == {"qwen3-4b": {"stable": "verified"}}


def test_list_tenants_finds_named_tenants(temp_store, reset_tenant):
    """list_tenants() (used by the reconcile loop) surfaces every user namespace
    and excludes the default (un-prefixed) tenant."""
    store = temp_store.get_store()

    # Default-tenant write creates no users/ entry.
    (store.workdir("races", "r0") / "race.json").write_text("{}")
    store.commit("races", "r0")
    assert temp_store.list_tenants() == []

    for who in ("alice", "bob"):
        reset_tenant.set_tenant(who)
        (store.workdir("races", "r1") / "race.json").write_text("{}")
        store.commit("races", "r1")

    assert sorted(temp_store.list_tenants()) == ["alice", "bob"]


def test_resolver_gated_off_ignores_jwt(monkeypatch):
    """With SLM_MULTI_TENANT unset, the resolver returns the default tenant even
    when a verified JWT claim is present — auth is decoupled from tenancy."""
    from app import tenancy

    monkeypatch.delenv("SLM_MULTI_TENANT", raising=False)

    class _Req:
        scope = {"aws.event": {"requestContext": {"authorizer": {"jwt": {"claims": {"cognito:username": "MyIdP_u123", "sub": "s-1"}}}}}}

    assert tenancy.resolve_tenant_from_request(_Req()) == tenancy.DEFAULT_TENANT


def test_resolver_keys_on_stable_username_not_sub(monkeypatch):
    """With multi-tenancy on, the STABLE username is the tenant id (NOT the Cognito
    sub, which regenerates for users signing in through an external IdP). Usernames
    of the form "<IdP>_<username>" resolve to the bare username; the sub is ignored
    even when it differs across logins."""
    from app import tenancy

    monkeypatch.setenv("SLM_MULTI_TENANT", "true")
    monkeypatch.delenv("SLM_DEV_TENANT", raising=False)

    def _req(claims):
        class _R:
            scope = {"aws.event": {"requestContext": {"authorizer": {"jwt": {"claims": claims}}}}}
        return _R()

    # Two logins of the SAME user: different sub, same username → same tenant.
    assert tenancy.resolve_tenant_from_request(
        _req({"cognito:username": "MyIdP_alice", "sub": "sub-aaa"})) == "alice"
    assert tenancy.resolve_tenant_from_request(
        _req({"cognito:username": "MyIdP_alice", "sub": "sub-bbb"})) == "alice"
    # identities[].userId is used when username is absent.
    assert tenancy.resolve_tenant_from_request(
        _req({"identities": [{"userId": "victor"}], "sub": "x"})) == "victor"
    # sub is the last-resort fallback.
    assert tenancy.resolve_tenant_from_request(_req({"sub": "only-sub"})) == "only-sub"
    # No claim → default tenant.
    assert tenancy.resolve_tenant_from_request(_req({})) == tenancy.DEFAULT_TENANT


def test_dev_tenant_override_simulates_user_locally(monkeypatch):
    """SLM_DEV_TENANT lets the dev box (no JWT) simulate a user — but only when
    multi-tenancy is on, and never overriding a real JWT claim."""
    from app import tenancy

    monkeypatch.setenv("SLM_DEV_TENANT", "dev-sub")

    class _NoClaim:
        scope = {"aws.event": {}}

    class _WithSub:
        scope = {"aws.event": {"requestContext": {"authorizer": {"jwt": {"claims": {"sub": "real-sub"}}}}}}

    # Off → ignored entirely.
    monkeypatch.delenv("SLM_MULTI_TENANT", raising=False)
    assert tenancy.resolve_tenant_from_request(_NoClaim()) == tenancy.DEFAULT_TENANT

    # On + no claim → dev override applies.
    monkeypatch.setenv("SLM_MULTI_TENANT", "true")
    assert tenancy.resolve_tenant_from_request(_NoClaim()) == "dev-sub"
    # On + real claim → the real sub WINS (dev override never masks a real user).
    assert tenancy.resolve_tenant_from_request(_WithSub()) == "real-sub"
