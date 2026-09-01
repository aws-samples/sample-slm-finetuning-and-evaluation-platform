# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The solution's user-agent suffix, asserted on the real request header.

The header is captured from botocore's `before-send` hook and the request
aborted there, so nothing is sent and no AWS is touched.

What matters for this package specifically: Strands builds its own Bedrock
client inside BedrockModel, so the suffix has to be applied at session level to
reach it. Handing BedrockModel a boto_client_config instead would carry the
suffix but silently drop the 120-second read timeout Strands sets by default.
"""
from __future__ import annotations

import boto3
import botocore.session
import pytest
from botocore.config import Config

from dataset_investigator import aws_user_agent

TOKEN = "AWSSOLUTION/SO0363/v1.0.0"


class _Abort(Exception):
    pass


@pytest.fixture
def configured(monkeypatch):
    """A configured token and dummy credentials, hook NOT installed.

    The hook has to be *removed* rather than asserted absent: importing this
    package installs it whenever the environment already carries a suffix, so
    otherwise the suite would pass or fail depending on the shell it was started
    from. Same for boto3's cached default session, which would hand back a client
    built before the fixture ran.
    """
    original_init = botocore.session.Session.__init__
    original_default_session = boto3.DEFAULT_SESSION
    boto3.DEFAULT_SESSION = None
    monkeypatch.setenv(aws_user_agent.ENV_VAR_NAME, TOKEN)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    aws_user_agent.uninstall()
    assert aws_user_agent.hook_installed() is False
    yield
    botocore.session.Session.__init__ = original_init
    boto3.DEFAULT_SESSION = original_default_session


@pytest.fixture
def installed(configured):
    assert aws_user_agent.install() is True
    yield


def _tokens(client) -> list[str]:
    """The user-agent of a real signed request, as a token list."""
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
        client.meta.events.unregister("before-send.s3.*", grab, unique_id="ua-probe")
    assert "ua" in captured, "no request was built, so nothing was asserted"
    return captured["ua"].split()


def test_hook_attributes_a_client_it_never_built(installed):
    """Stands in for the Bedrock client Strands builds inside BedrockModel."""
    assert _tokens(boto3.client("s3", region_name="us-east-1")).count(TOKEN) == 1


def test_token_appears_once_with_both_layers_live(installed):
    client = aws_user_agent.get_client("s3", region_name="us-east-1")
    assert _tokens(client).count(TOKEN) == 1


def test_get_client_attributes_without_the_hook(configured):
    client = aws_user_agent.get_client("s3", region_name="us-east-1")
    assert _tokens(client).count(TOKEN) == 1


def test_get_client_preserves_a_caller_supplied_config(configured):
    client = aws_user_agent.get_client(
        "s3", region_name="us-east-1", config=Config(signature_version="s3v4")
    )
    assert _tokens(client).count(TOKEN) == 1
    assert client.meta.config.signature_version == "s3v4"


def test_unconfigured_environment_is_a_no_op(monkeypatch):
    original_init = botocore.session.Session.__init__
    monkeypatch.delenv(aws_user_agent.ENV_VAR_NAME, raising=False)
    try:
        assert aws_user_agent.user_agent_string() == ""
        assert aws_user_agent.install() is False
        assert botocore.session.Session.__init__ is original_init
    finally:
        botocore.session.Session.__init__ = original_init


def test_uninstall_restores_botocores_own_constructor(configured):
    """The hook can be taken back off, which is what makes the suite hermetic."""
    original = botocore.session.Session.__init__
    assert aws_user_agent.install() is True
    assert aws_user_agent.hook_installed() is True
    assert aws_user_agent.uninstall() is True
    assert botocore.session.Session.__init__ is original
    assert aws_user_agent.hook_installed() is False
