# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The serverless launcher attributes its whole process, exactly once.

The launcher runs under a foreign interpreter with the V3 SageMaker SDK and so
cannot import app.aws_clients; it carries its own copy of the session-constructor
patch. Two properties are worth pinning, because both were wrong at first:

  * it patches the CONSTRUCTOR, not just the session it hands the trainer — the
    SDK builds clients off sessions of its own, and those issue most of the calls
    this script causes;
  * the patch is idempotent and marked the same way app.aws_clients marks its
    own, so running the launcher twice in one process cannot stack it (which puts
    the token on the wire twice) and app.aws_clients can still remove it.

The fake SDK here is the `register_evaluator` op — the cheapest path through
main() that still executes the attribution block. Nothing is sent to AWS: the
request is aborted in botocore's before-send hook.
"""
from __future__ import annotations

import json
import sys
import types

import boto3
import botocore.session
import pytest

from app import aws_clients
from app.engines import serverless_launcher

TOKEN = "AWSSOLUTION/SO0363/v1.0.0"


class _Abort(Exception):
    pass


def _patch_layers() -> int:
    """How many attribution patches are stacked on the session constructor."""
    n = 0
    fn = botocore.session.Session.__init__
    while getattr(fn, "_slm_solution_user_agent_installed", False):
        n += 1
        fn = getattr(fn, "_slm_solution_user_agent_original_init", None)
        if fn is None:
            break
    return n


def _tokens(client) -> list[str]:
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


@pytest.fixture
def launcher_env(monkeypatch, tmp_path):
    """A spec file, a fake V3 SDK, and a clean session constructor."""
    original_init = botocore.session.Session.__init__
    original_default_session = boto3.DEFAULT_SESSION
    boto3.DEFAULT_SESSION = None
    monkeypatch.setenv("USER_AGENT_STRING", TOKEN)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    # The suite may have been started from a shell that already carries a suffix,
    # in which case importing `app` installed the hook. Start from none.
    aws_clients.uninstall()

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    class _FakeEvaluator:
        arn = "arn:aws:sagemaker:us-east-1:111122223333:hub-content/x"

        @classmethod
        def create(cls, **_kwargs):
            return cls()

    fakes = {
        "sagemaker": _mod("sagemaker"),
        "sagemaker.core": _mod("sagemaker.core"),
        "sagemaker.core.helper": _mod("sagemaker.core.helper"),
        "sagemaker.core.helper.session_helper": _mod(
            "sagemaker.core.helper.session_helper",
            Session=lambda boto_session=None: object(),
        ),
        "sagemaker.ai_registry": _mod("sagemaker.ai_registry"),
        "sagemaker.ai_registry.evaluator": _mod(
            "sagemaker.ai_registry.evaluator", Evaluator=_FakeEvaluator
        ),
    }
    for name, mod in fakes.items():
        monkeypatch.setitem(sys.modules, name, mod)

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "op": "register_evaluator",
                "evaluatorName": "ev",
                "evaluatorType": "RewardFunction",
                "source": "arn:aws:lambda:us-east-1:111122223333:function:r",
                "role": "arn:aws:iam::111122223333:role/r",
                "region": "us-east-1",
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["serverless_launcher.py", str(spec_path)])
    yield
    botocore.session.Session.__init__ = original_init
    boto3.DEFAULT_SESSION = original_default_session


def test_launcher_attributes_the_process(launcher_env):
    """A client the launcher never built still carries the token — which is the
    case that matters, since the SDK builds its own."""
    assert serverless_launcher.main() == 0
    assert _patch_layers() == 1
    assert _tokens(boto3.client("s3", region_name="us-east-1")).count(TOKEN) == 1


def test_running_the_launcher_twice_does_not_stack_the_patch(launcher_env):
    """Two runs in one process must not send the token twice."""
    assert serverless_launcher.main() == 0
    assert serverless_launcher.main() == 0
    assert _patch_layers() == 1
    assert _tokens(boto3.client("s3", region_name="us-east-1")).count(TOKEN) == 1


def test_the_launchers_patch_is_removable_by_the_shared_helper(launcher_env):
    """The launcher marks its patch the way app.aws_clients marks its own, so the
    two agree on whether this process is attributed."""
    assert serverless_launcher.main() == 0
    assert aws_clients.hook_installed() is True
    assert aws_clients.uninstall() is True
    assert aws_clients.hook_installed() is False
    assert _patch_layers() == 0


def test_launcher_is_a_no_op_without_a_configured_suffix(launcher_env, monkeypatch):
    monkeypatch.delenv("USER_AGENT_STRING", raising=False)
    before = botocore.session.Session.__init__
    assert serverless_launcher.main() == 0
    assert botocore.session.Session.__init__ is before
    assert _patch_layers() == 0
