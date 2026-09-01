# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""HuggingFace token storage (AWS Secrets Manager) — PER USER.

Gated models (Llama, Mistral, Gemma) require an HF token to download. Each user
brings THEIR OWN token: they accept the model's license under their own HF
account, so we never share one person's credentials or license approval across
the team. This mirrors the per-user data isolation (see tenancy.py) — the token
is the one piece of a user's workspace that must also be theirs.

Storage: a single Secrets Manager secret (SLM_HF_SECRET_NAME, set by CDK) holds a
JSON map {tenant_sub: token}. One secret keeps IAM simple (the Lambdas already
have access to it) while still isolating tokens by caller. The token is resolved
at job-launch time for the CURRENT tenant and injected as HF_TOKEN into the
SageMaker job (the HF libraries read it automatically).

Back-compat: the secret historically held a RAW token string (the single global
token). We still read that — as the DEFAULT_TENANT's token — and the first
per-user write transparently migrates the blob to the map form.

Shared fallback: a user who hasn't set a token yet reads the SHARED legacy slot
(DEFAULT_TENANT — see _fallback_tenants), so HF model/dataset access works out of
the box wherever that slot is populated. One named user's personal token is never
lent to another. The UI nags users on the fallback to set their own, and
hf_token_is_set() reports only the user's OWN token so that nag (and Settings)
never lies.

Locally SLM_HF_SECRET_NAME is unset and these functions degrade gracefully (no
token → gated models stay locked).
"""
from __future__ import annotations

import json
import logging
import os

from .aws_config import load_aws_config
from .tenancy import DEFAULT_TENANT, current_tenant

_log = logging.getLogger(__name__)


def _secret_name() -> str | None:
    return os.environ.get("SLM_HF_SECRET_NAME")


def _client():
    # Local import + ambient creds (Lambda role or profile), matching orchestrate.
    from .aws_clients import get_session

    cfg = load_aws_config()
    sess = get_session(profile_name=cfg.profile or None, region_name=cfg.region)
    return sess.client("secretsmanager")


def _read_blob() -> dict[str, str]:
    """Read the secret as a {tenant: token} map. A legacy raw token string is
    interpreted as the DEFAULT_TENANT's token (so existing setups keep working)."""
    name = _secret_name()
    if not name:
        return {}
    try:
        raw = _client().get_secret_value(SecretId=name).get("SecretString", "").strip()
    except Exception as e:  # noqa: BLE001 — missing secret / no access / expired creds
        # Degrade gracefully (gated models stay locked) but DON'T swallow silently:
        # an ExpiredTokenException or AccessDenied looks identical to "no token set"
        # otherwise, which is exactly what made the local 'HF token not set' bug hard
        # to diagnose (wrong/expired profile reading the secret). Leave a breadcrumb.
        _log.warning("HF-token secret %r read failed (%s): %s — treating as no token.",
                     name, type(e).__name__, e)
        return {}
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            return {k: str(v) for k, v in obj.items() if isinstance(v, str)}
        except (json.JSONDecodeError, ValueError):
            return {}
    # Legacy raw-token form → the historical single token belongs to the default
    # tenant (the owner whose state was migrated under the default namespace).
    return {DEFAULT_TENANT: raw}


def _write_blob(blob: dict[str, str]) -> None:
    name = _secret_name()
    if not name:
        return
    client = _client()
    body = json.dumps(blob)
    try:
        client.put_secret_value(SecretId=name, SecretString=body)
    except client.exceptions.ResourceNotFoundException:
        client.create_secret(Name=name, SecretString=body)


def set_hf_token(token: str, tenant: str | None = None) -> None:
    """Store (or update) the CURRENT tenant's HF token. No-op if no secret is
    configured or the token is blank."""
    name = _secret_name()
    if not name or not token.strip():
        return
    blob = _read_blob()
    blob[tenant or current_tenant()] = token.strip()
    _write_blob(blob)


def _fallback_tenants() -> list[str]:
    """Tenant slots whose token may be BORROWED when the caller hasn't set their
    own — so new users aren't blocked on HF downloads before they've visited
    Settings. Defaults to DEFAULT_TENANT alone: that slot holds the shared,
    pre-tenancy token an operator installs for everyone, and it is the only token
    intended to be readable by other tenants. No named user's slot is a default
    fallback, so a personal token is never handed to a different user.

    SLM_HF_FALLBACK_TENANTS overrides the list (comma-separated) for deployments
    that want a different shared slot; set it to an empty string to disable the
    shared fallback entirely and require every user to bring their own token."""
    raw = os.environ.get("SLM_HF_FALLBACK_TENANTS")
    if raw is None:
        return [DEFAULT_TENANT]
    return [t.strip() for t in raw.split(",") if t.strip()]


def _valid(val: str | None) -> str | None:
    """Requires the 'hf_' prefix so an empty string or a CDK-generated placeholder
    never counts as a real token (which would wrongly enable gated models)."""
    val = (val or "").strip()
    return val if val.startswith("hf_") else None


def get_hf_token(tenant: str | None = None) -> str | None:
    """The EFFECTIVE HF token for the CURRENT tenant: their own stored token,
    falling back to the shared slot (see _fallback_tenants) when they haven't set
    one. None if neither exists.

    On the fallback a user downloads models under whichever HF account owns the
    SHARED token — that account's gated-model license approvals apply, not the
    user's. The UI nags such users to set their own (hf_token_is_set stays
    own-token-only precisely so it can tell the difference)."""
    blob = _read_blob()
    me = tenant or current_tenant()
    own = _valid(blob.get(me))
    if own:
        return own
    for fb in _fallback_tenants():
        if fb == me:
            continue
        val = _valid(blob.get(fb))
        if val:
            return val
    return None


def hf_token_is_set(tenant: str | None = None) -> bool:
    """Whether THIS tenant has THEIR OWN real HF token stored — deliberately
    ignores the shared fallback, so Settings + the set-your-token banner reflect
    the user's own state, not the borrowed one."""
    return _valid(_read_blob().get(tenant or current_tenant())) is not None
