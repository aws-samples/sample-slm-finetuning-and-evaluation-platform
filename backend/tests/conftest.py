# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Shared pytest fixtures.

Tests target the platform's PURE logic — no AWS. The state store is pointed at a
temp dir via SLM_DATA_DIR so storage/race tests never touch real data or S3.
"""
from __future__ import annotations

import importlib
import os

import pytest

# Make the suite HERMETIC. Several tests reach code that calls load_aws_config(),
# which resolves the account from Settings > SLM_AWS_ACCOUNT > STS
# GetCallerIdentity and raises AwsAccountUnresolvedError when all three fail. On a
# machine with no (or expired) credentials that is 4 failures on a fresh clone,
# even though nothing in the suite talks to AWS — the account is only needed to
# derive bucket/role/image NAMES. Pinning a documentation-range account id here
# keeps the derived names deterministic and stops the resolver from reaching for
# real credentials. Set before any `app.*` import so module-level config reads see
# it. A test that wants a different account still overrides it with monkeypatch.
os.environ.setdefault("SLM_AWS_ACCOUNT", "111122223333")
os.environ.setdefault("SLM_AWS_REGION", "us-east-1")
# No named profile: boto3 must not try to load a developer's ~/.aws profile.
os.environ.setdefault("SLM_AWS_PROFILE", "")


def _clear_doc_caches():
    """Drop the short-lived read caches for the verifications + config root docs
    (process-global; request-scoped in prod but shared across tests in one run)."""
    try:
        from app import verifications
        verifications._invalidate_all_cache()
    except Exception:
        pass
    try:
        from app import aws_config
        aws_config._invalidate_config_cache()
    except Exception:
        pass


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """A LocalStore rooted at a temp dir; clears the cached singleton so each
    test gets an isolated data root. Returns the store module."""
    monkeypatch.setenv("SLM_STORAGE_BACKEND", "local")
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    from app import store as store_mod

    store_mod.get_store.cache_clear()
    _clear_doc_caches()  # don't let a prior test's cached doc leak into this store
    yield store_mod
    store_mod.get_store.cache_clear()
    _clear_doc_caches()


@pytest.fixture
def sample_rows():
    """A few valid chat-template rows."""
    return [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(10)
    ]
