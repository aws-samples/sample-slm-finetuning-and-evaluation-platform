# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Make this runtime's AWS SDK calls identify the solution.

The solution's API usage is measured from a suffix on the user-agent of each AWS
call, e.g. ``AWSSOLUTION/SO0363/v1.0.0``. This package is deployed on its own,
outside the CDK app, so nothing injects the value for it: the suffix comes from
the ``USER_AGENT_STRING`` environment variable of the runtime, which has to be
set when the runtime is launched. Unset simply means unattributed — never an
error.

``install()`` patches botocore's session constructor rather than configuring
individual clients, for one specific reason: almost every AWS call this agent
makes is issued by a client it never sees. Strands builds its own Bedrock client
inside ``BedrockModel``, and handing ``BedrockModel`` a ``boto_client_config``
to carry the suffix would silently drop the 120-second read timeout Strands sets
by default and fall back to botocore's 60 — long model calls would start timing
out. Patching the session leaves Strands' own configuration untouched.
"""
from __future__ import annotations

import os
from typing import Any

import boto3
import botocore.session
from botocore.config import Config

ENV_VAR_NAME = "USER_AGENT_STRING"

_INSTALLED_FLAG = "_slm_solution_user_agent_installed"
_ORIGINAL_INIT = "_slm_solution_user_agent_original_init"


def user_agent_string() -> str:
    """The solution's user-agent suffix, or "" when none is configured."""
    return os.environ.get(ENV_VAR_NAME, "").strip()


def hook_installed() -> bool:
    return bool(getattr(botocore.session.Session.__init__, _INSTALLED_FLAG, False))


def install() -> bool:
    """Make every botocore session in this process carry the suffix.

    Idempotent. Returns False when there is nothing to attribute.
    """
    if not user_agent_string():
        return False
    if hook_installed():
        return True

    original_init = botocore.session.Session.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        suffix = user_agent_string()
        if not suffix:
            return
        existing = self.user_agent_extra or ""
        # The user-agent is a space-delimited token list: append to whatever is
        # already there, and only if this token is not already present.
        if suffix not in existing.split():
            self.user_agent_extra = f"{existing} {suffix}".strip()

    setattr(patched_init, _INSTALLED_FLAG, True)
    setattr(patched_init, _ORIGINAL_INIT, original_init)
    botocore.session.Session.__init__ = patched_init
    return True


def uninstall() -> bool:
    """Put botocore's own session constructor back. Returns True if one was removed.

    For tests that have to exercise the un-hooked path. They cannot simply assert
    the hook is absent: it is installed from the package `__init__` whenever the
    environment carries a suffix, so whether it is there depends on the shell the
    suite was started from. A test that needs it gone has to remove it.
    """
    removed = False
    # Unwrap every layer, not just the outer one: a process can end up with more
    # than one patch (an entrypoint that installs its own inline copy, say), and
    # leaving an inner one in place looks exactly like no hook at all while still
    # putting the token on the wire.
    while True:
        original = getattr(botocore.session.Session.__init__, _ORIGINAL_INIT, None)
        if original is None:
            return removed
        botocore.session.Session.__init__ = original
        removed = True


def get_client(service_name: str, **kwargs: Any) -> Any:
    """A boto3 client that identifies this solution.

    The suffix is added at client level only when the process-wide hook is
    absent: botocore concatenates a client's user-agent extra onto its session's,
    so both layers acting would send the token twice.
    """
    suffix = user_agent_string()
    if suffix and not hook_installed():
        base: Config | None = kwargs.get("config")
        existing = (base.user_agent_extra or "") if base is not None else ""
        if suffix not in existing.split():
            merged = Config(user_agent_extra=f"{existing} {suffix}".strip())
            kwargs["config"] = base.merge(merged) if base is not None else merged
    return boto3.client(service_name, **kwargs)
