# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Per-request tenant identity — the seam between "who is calling" and "whose
state do we touch".

The platform is moving from single-user to multi-user (per-user resource
isolation). That split has two independent halves:

  1. tenancy  — keying state by an owner (this module + TenantStore in store.py)
  2. auth     — proving who the owner is (Cognito, optional external IdP; later)

This module is the tenancy half, and it is deliberately INERT until auth lands:

  * The current tenant is a contextvar, defaulting to DEFAULT_TENANT.
  * DEFAULT_TENANT is the historical, UN-prefixed namespace — so with the flag
    off, every request is the default tenant and state lives exactly where it
    always has (no migration, no behaviour change for the existing races).
  * resolve_tenant_from_request() CAN read a verified Cognito JWT `sub`, but only
    when SLM_MULTI_TENANT is enabled. Until then it always returns DEFAULT_TENANT.

Once auth is wired, flip SLM_MULTI_TENANT=true and the same identity flows
through this seam into TenantStore — no storage rework.
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from typing import Any, Iterator

# The historical namespace: state written before multi-tenancy lived at the bare
# collection prefix (slm-platform/state/races/<id>). TenantStore treats this
# sentinel as "no per-user prefix", so existing state stays put.
DEFAULT_TENANT = "__default__"

# The global namespace holding curated "golden"/sample runs (see samples.py). It
# lives under users/ like a tenant but is NEVER a real user: the reconcile loop
# skips it, and the store's sample-read fallback reads from it. Defined here (the
# lowest-level module) so store.py can reference it without importing samples.py.
SAMPLES_TENANT_NAME = "__samples__"

_current: ContextVar[str] = ContextVar("slm_tenant", default=DEFAULT_TENANT)


def multi_tenant_enabled() -> bool:
    """Whether per-user isolation is active. Off by default — the JWT `sub` is
    ignored and everyone shares the default tenant until this is turned on."""
    return os.environ.get("SLM_MULTI_TENANT", "").lower() in ("1", "true", "yes", "on")


def current_tenant() -> str:
    """The tenant for the current context (request or background scope)."""
    return _current.get()


def set_tenant(tenant: str) -> Any:
    """Set the current tenant; returns the reset token (see contextvars)."""
    return _current.set(tenant or DEFAULT_TENANT)


@contextlib.contextmanager
def tenant_scope(tenant: str) -> Iterator[None]:
    """Run a block as a given tenant, restoring the previous one after. Used by
    the background reconcile loop to advance each tenant's races in turn."""
    token = _current.set(tenant or DEFAULT_TENANT)
    try:
        yield
    finally:
        _current.reset(token)


def _claims_from_request(request: Any) -> dict[str, Any]:
    """Pull the API Gateway (HTTP API v2) JWT authorizer claims out of the ASGI
    scope Mangum exposes. API Gateway has already VERIFIED the token, so trusting
    these claims is safe — we never parse the raw token ourselves. Empty dict if
    not present (e.g. local dev with no authorizer)."""
    try:
        event = request.scope.get("aws.event", {})
        return (
            event.get("requestContext", {})
            .get("authorizer", {})
            .get("jwt", {})
            .get("claims", {})
        ) or {}
    except Exception:  # noqa: BLE001 — never let identity-extraction crash a request
        return {}


def _stable_tenant_key(claims: dict[str, Any]) -> str | None:
    """The DURABLE per-user id from a verified JWT, preferring the USERNAME over
    Cognito's `sub`.

    Why not `sub`: for a user signing in through an EXTERNAL IdP, Cognito
    regenerates the `sub` whenever that user record is recreated (e.g. an IdP
    attribute-mapping change + re-login, or an admin deleting the user) — which
    would orphan that user's data. The username at the IdP doesn't change, so it
    is the stable tenant key.

    Sources, in order: an explicit alias claim (custom:alias / alias / ALIAS — if
    the IdP emits one and Cognito maps it) → `cognito:username` (usernames minted
    by an external IdP arrive as "<IdP>_<username>", so take the username part;
    native users have a plain username) → the `identities` userId → finally `sub`
    (last resort). Returns None if no claim is present."""
    # Prefer an explicit alias claim when present — no string parsing, no
    # dependence on the "<IdP>_<username>" username format.
    for k in ("custom:alias", "alias", "ALIAS"):
        v = claims.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    username = claims.get("cognito:username") or claims.get("username")
    if isinstance(username, str) and username:
        # Usernames from an external IdP are "<IdP>_<username>" ("MyIdP_alice").
        return username.split("_", 1)[1] if "_" in username else username
    # Fallback: the `identities` array carries the raw external-provider userId.
    identities = claims.get("identities")
    if isinstance(identities, list) and identities:
        uid = identities[0].get("userId") if isinstance(identities[0], dict) else None
        if uid:
            return str(uid)
    sub = claims.get("sub")
    return str(sub) if sub else None


def resolve_tenant_from_request(request: Any) -> str:
    """Resolve the tenant for an incoming request.

    Gated: with SLM_MULTI_TENANT off this ALWAYS returns DEFAULT_TENANT, so the
    JWT (present in prod even today) is ignored and behaviour is unchanged. With
    the flag on, the STABLE username (see _stable_tenant_key) becomes the tenant
    id — NOT the Cognito `sub`, which regenerates for external-IdP users and would
    orphan their data; falls back to DEFAULT_TENANT if no claim is present.

    SLM_DEV_TENANT is a LOCAL-ONLY escape hatch: there's no Cognito/JWT on the dev
    box, so set it to a value to *simulate* that user and exercise per-user
    isolation locally. It's read only when multi-tenancy is on, and is ignored the
    moment a real JWT claim is present (so it can never override a real user)."""
    if not multi_tenant_enabled():
        return DEFAULT_TENANT
    key = _stable_tenant_key(_claims_from_request(request))
    if key:
        return key
    dev = os.environ.get("SLM_DEV_TENANT", "").strip()
    return dev if dev else DEFAULT_TENANT
