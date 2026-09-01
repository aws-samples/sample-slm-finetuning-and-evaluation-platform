# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The solution's user-agent suffix, asserted on the real request header.

API usage is attributed by matching a token in the user-agent of each AWS call,
so the only assertion worth making is on the header of a fully built request —
not on the Config object that was meant to produce it. Every test here captures
that header from botocore's `before-send` hook and aborts the request there, so
nothing is sent and no AWS is touched.

Two properties matter beyond "the token is present":
  * exactly once — botocore concatenates a client's user-agent extra onto its
    session's, so two layers both adding the token would send it twice, and a
    retried request must not accumulate copies either;
  * additive — a call site that already configures a signature version, a
    timeout or its own user-agent token must keep all of it.
"""
from __future__ import annotations

import boto3
import botocore.session
import pytest
from botocore.config import Config

from app import aws_clients

TOKEN = "AWSSOLUTION/SO0363/v1.0.0"


class _Abort(Exception):
    """Raised from the before-send hook to stop the request being sent."""


@pytest.fixture
def configured(monkeypatch):
    """A configured token and dummy credentials, with the hook NOT installed.

    The hook patches a botocore class, so every fixture here restores the
    original or the patch outlives the test.

    It also has to *remove* the hook rather than assert it is absent. Importing
    `app` installs it whenever the environment already carries a suffix, so a
    suite started from a shell that exports USER_AGENT_STRING would otherwise
    fail on fixture setup — the tests have to hold in both environments.
    """
    original_init = botocore.session.Session.__init__
    # boto3.client() reuses a cached default session. One built under different
    # settings would carry a stale user-agent into this test, so drop it and let
    # each test build its own.
    original_default_session = boto3.DEFAULT_SESSION
    boto3.DEFAULT_SESSION = None
    monkeypatch.setenv(aws_clients.ENV_VAR_NAME, TOKEN)
    # The request must be signed to be built, but it is aborted before it leaves
    # the process, so the credentials are never used against AWS.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    aws_clients.uninstall()
    assert aws_clients.hook_installed() is False
    yield
    botocore.session.Session.__init__ = original_init
    boto3.DEFAULT_SESSION = original_default_session


@pytest.fixture
def installed(configured):
    """As `configured`, plus the process-wide hook in place."""
    assert aws_clients.install() is True
    assert aws_clients.hook_installed() is True
    yield


def _user_agent(client) -> str:
    """The User-Agent of a real, fully signed request that is never sent."""
    captured = {}

    def grab(request, **_kwargs):
        captured["ua"] = request.headers["User-Agent"].decode()
        raise _Abort()

    client.meta.events.register("before-send.s3.*", grab, unique_id="ua-probe")
    try:
        client.list_buckets()
    except _Abort:
        pass
    finally:
        # Leaving the probe registered would abort the NEXT request on this
        # client before its header was built, which reads as a missing token.
        client.meta.events.unregister("before-send.s3.*", grab, unique_id="ua-probe")
    assert "ua" in captured, "no request was built, so nothing was asserted"
    return captured["ua"]


def _tokens(client) -> list[str]:
    """The user-agent as the collector sees it: a space-delimited token list."""
    return _user_agent(client).split()


# --- with the process-wide hook: the token reaches the wire, once ----------- #

def test_plain_client_carries_the_token(installed):
    """The hook is what covers the call sites this module never sees, including
    the clients the SageMaker SDK builds internally."""
    assert _tokens(boto3.client("s3", region_name="us-east-1")).count(TOKEN) == 1


def test_wrapper_client_carries_the_token_exactly_once(installed):
    """Both layers are live here; only one of them may add the token."""
    client = aws_clients.get_client("s3", region_name="us-east-1")
    assert _tokens(client).count(TOKEN) == 1


def test_session_derived_client_carries_the_token(installed):
    """Covers the many call sites that build clients off their own session."""
    client = boto3.Session(region_name="us-east-1").client("s3")
    assert _tokens(client).count(TOKEN) == 1


def test_token_appears_once_when_a_client_is_reused(installed):
    """A retry re-signs the request; the token must not accumulate."""
    client = boto3.client("s3", region_name="us-east-1")
    assert _tokens(client).count(TOKEN) == 1
    assert _tokens(client).count(TOKEN) == 1


# --- nothing pre-existing is dropped --------------------------------------- #

def test_client_level_config_survives(installed):
    """A signature version pinned at a call site must still be honoured."""
    client = boto3.client(
        "s3", region_name="us-east-1", config=Config(signature_version="s3v4")
    )
    assert _tokens(client).count(TOKEN) == 1
    assert client.meta.config.signature_version == "s3v4"


def test_another_librarys_token_is_not_clobbered(installed):
    tokens = _tokens(
        boto3.client(
            "s3", region_name="us-east-1", config=Config(user_agent_extra="OTHER/1.0")
        )
    )
    assert tokens.count(TOKEN) == 1
    assert "OTHER/1.0" in tokens


def test_config_is_left_alone_while_the_hook_is_live(installed):
    """The hook already attributes every client, so the Config layer stands
    down: touching it here is what would double the token."""
    base = Config(signature_version="s3v4", read_timeout=42)
    assert aws_clients.botocore_config(base) is base


# --- without the hook: the Config layer attributes, and preserves ----------- #

def test_wrapper_client_carries_the_token_without_the_hook(configured):
    """The fallback path: an entrypoint that never installed the hook still
    attributes the calls it makes through the wrappers."""
    client = aws_clients.get_client("s3", region_name="us-east-1")
    assert _tokens(client).count(TOKEN) == 1


def test_wrapper_session_carries_the_token_without_the_hook(configured):
    """A session takes no Config, so this is the only way its clients -- and any
    a library builds off it -- get attributed when the hook is absent."""
    client = aws_clients.get_session(region_name="us-east-1").client("s3")
    assert _tokens(client).count(TOKEN) == 1


def test_wrapper_session_honours_its_arguments(configured):
    """Region and profile must still reach the session it hands back."""
    session = aws_clients.get_session(region_name="eu-west-1")
    assert session.region_name == "eu-west-1"


def test_botocore_config_merge_preserves_every_other_field(configured):
    base = Config(signature_version="s3v4", read_timeout=42, retries={"max_attempts": 7})
    merged = aws_clients.botocore_config(base)
    assert merged.signature_version == "s3v4"
    assert merged.read_timeout == 42
    assert merged.retries == {"max_attempts": 7}
    assert TOKEN in (merged.user_agent_extra or "").split()
    # Applying it twice must not duplicate the token.
    assert (aws_clients.botocore_config(merged).user_agent_extra or "").split().count(TOKEN) == 1


def test_botocore_config_appends_to_an_existing_extra(configured):
    merged = aws_clients.botocore_config(Config(user_agent_extra="OTHER/1.0"))
    assert (merged.user_agent_extra or "").split() == ["OTHER/1.0", TOKEN]


# --- an unconfigured environment is not an error --------------------------- #

def test_unconfigured_environment_is_a_no_op(monkeypatch):
    """No configured token means "not attributed", never a failure."""
    original_init = botocore.session.Session.__init__
    monkeypatch.delenv(aws_clients.ENV_VAR_NAME, raising=False)
    try:
        assert aws_clients.user_agent_string() == ""
        assert aws_clients.install() is False
        assert botocore.session.Session.__init__ is original_init
        # The wrappers still return a usable client, just an unattributed one.
        base = Config(signature_version="s3v4")
        assert aws_clients.botocore_config(base) is base
        assert aws_clients.botocore_config().user_agent_extra is None
    finally:
        botocore.session.Session.__init__ = original_init


def test_uninstall_restores_botocores_own_constructor(configured):
    """The hook can be taken back off, which is what makes the suite hermetic."""
    original = botocore.session.Session.__init__
    assert aws_clients.install() is True
    assert aws_clients.hook_installed() is True
    assert aws_clients.uninstall() is True
    assert botocore.session.Session.__init__ is original
    assert aws_clients.hook_installed() is False
