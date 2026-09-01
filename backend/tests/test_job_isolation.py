# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Per-tenant job isolation — owner tags + tenant-scoped S3 job paths."""
import pytest

from app import orchestrate as orch
from app import tenancy


@pytest.fixture
def as_tenant():
    tokens = []

    def _set(t):
        tokens.append(tenancy.set_tenant(t))

    yield _set
    for tok in reversed(tokens):
        tenancy._current.reset(tok)


class _Cfg:
    bucket = "test-bucket"

    @property
    def s3_prefix(self):
        return "s3://test-bucket/slm-platform"


def test_default_tenant_uses_historical_paths_and_tags(as_tenant):
    as_tenant(tenancy.DEFAULT_TENANT)
    assert orch._jobs_key_prefix() == "slm-platform/jobs"
    assert orch._jobs_s3_base(_Cfg()) == "s3://test-bucket/slm-platform/jobs"
    # No owner tag for the default tenant — keeps pre-tenancy jobs' tag set intact.
    tags = orch._job_tags()
    assert not any(t["Key"] == "owner" for t in tags)


def test_named_tenant_scopes_paths_and_adds_owner_tag(as_tenant):
    sub = "00000000-0000-0000-0000-000000000001"
    as_tenant(sub)
    assert orch._jobs_key_prefix() == f"slm-platform/users/{sub}/jobs"
    assert orch._jobs_s3_base(_Cfg()) == f"s3://test-bucket/slm-platform/users/{sub}/jobs"
    # Owner tag is added (cost attribution + console identification).
    tags = orch._job_tags()
    assert {"Key": "owner", "Value": sub} in tags


def test_two_tenants_get_distinct_job_prefixes(as_tenant):
    as_tenant("alice")
    a = orch._jobs_key_prefix()
    as_tenant("bob")
    b = orch._jobs_key_prefix()
    assert a != b
    assert "alice" in a and "bob" in b


def test_leaderboard_regex_handles_tenant_path():
    """The source-job parser must still find the train job in a tenant-scoped
    artifact URI (users/<sub>/jobs/...)."""
    from app.leaderboard import _source_train_job

    uri = ("s3://b/slm-platform/users/some-sub/jobs/slm-qwen3-0-6b-x/output/"
           "slm-qwen3-0-6b-x/output/model.tar.gz")
    assert _source_train_job(uri) == "slm-qwen3-0-6b-x"
