# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""AWS SDK clients that identify this solution on every call.

API usage is attributed to a solution by a suffix on the user-agent of each AWS
SDK call, e.g. ``AWSSOLUTION/SO0363/v1.0.0``. CDK publishes the value to the
runtime as ``USER_AGENT_STRING``; this module is what puts it on the wire.

Two layers, deliberately:

1. ``install()`` patches ``botocore.session.Session.__init__`` so that every
   botocore session created in this process carries the suffix. It is the only
   mechanism that reaches clients this code never constructs itself — the ones
   the SageMaker SDK, and any other library, build internally — and it covers
   the sixty-odd ``session.client(...)`` calls already in the codebase without
   touching a single one of them. It is installed from the package ``__init__``,
   which runs before any submodule, so no client can be built ahead of it.

2. ``get_client``/``get_resource``/``get_session`` are the sanctioned way to
   construct a client directly. They exist so attribution does not depend on
   one patch surviving a future refactor, and so new code has an obvious thing
   to call. They add the suffix at client level only when layer 1 is *not* in
   place: botocore concatenates a client's ``user_agent_extra`` onto its
   session's, so applying both would put the token on the wire twice.

The suffix is read with ``os.environ.get`` and a default of "": an unset
variable must never be an error, or this module becomes a hard dependency on a
deploy-time detail and any environment that lacks it (a test, a local run, a
developer's shell) fails at import instead of simply not being attributed.

Note on the value: it is passed as botocore's ``user_agent_extra``, verbatim.
The SDK's "app id" knob is not usable here because it sanitises the string,
replacing the slashes the collector matches on.
"""
from __future__ import annotations

import os
from typing import Any

import boto3
import botocore.session
from botocore.config import Config

#: Set by CDK on every Lambda in the stack. Empty everywhere else, which simply
#: means "not attributed" — never an error.
ENV_VAR_NAME = "USER_AGENT_STRING"

_INSTALLED_FLAG = "_slm_solution_user_agent_installed"
_ORIGINAL_INIT = "_slm_solution_user_agent_original_init"


def user_agent_string() -> str:
    """The solution's user-agent suffix, or "" when none is configured."""
    return os.environ.get(ENV_VAR_NAME, "").strip()


def hook_installed() -> bool:
    """True when every session in this process already carries the suffix.

    Read off the patched function itself rather than a module-level flag, so a
    reload of this module cannot make it disagree with reality.
    """
    return bool(getattr(botocore.session.Session.__init__, _INSTALLED_FLAG, False))


def install() -> bool:
    """Make every botocore session in this process carry the suffix.

    Idempotent: safe to call from more than one entrypoint, and re-importing or
    reloading the module cannot stack the patch on top of itself.

    Returns True if the patch is in place, False if there is nothing to attribute
    (no configured suffix), so a caller can log which happened.
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
        # Append, never replace: another library may have put its own token
        # here, and the user-agent is a space-delimited list of them. Checking
        # for the token first keeps a session from accumulating duplicates.
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


def botocore_config(base: Config | None = None) -> Config:
    """A botocore Config carrying the suffix, preserving everything in `base`.

    Use at the call sites that already pass a Config. `Config.merge` returns a
    new Config in which the argument's set fields win, so passing only
    `user_agent_extra` leaves every other field of `base` — signature version,
    timeouts, retries, addressing style — exactly as it was.

    `base` comes back untouched when there is nothing to attribute, and also
    when the process-wide hook is already applying the suffix at session level:
    botocore concatenates the two, so adding it here as well would send the
    token twice.
    """
    suffix = user_agent_string()
    if not suffix or hook_installed():
        return base if base is not None else Config()
    if base is None:
        return Config(user_agent_extra=suffix)
    existing = base.user_agent_extra or ""
    if suffix in existing.split():
        return base
    merged_extra = f"{existing} {suffix}".strip()
    return base.merge(Config(user_agent_extra=merged_extra))


def get_session(**kwargs: Any) -> boto3.Session:
    """A boto3 Session whose clients identify this solution.

    A boto3 Session takes no Config, so when the hook is absent the suffix is
    set on the underlying botocore session instead — which every client built
    from it inherits, including the ones libraries build for themselves.
    """
    suffix = user_agent_string()
    if suffix and not hook_installed() and "botocore_session" not in kwargs:
        core = botocore.session.get_session()
        existing = core.user_agent_extra or ""
        if suffix not in existing.split():
            core.user_agent_extra = f"{existing} {suffix}".strip()
        kwargs["botocore_session"] = core
    return boto3.Session(**kwargs)


def get_client(service_name: str, **kwargs: Any) -> Any:
    """A boto3 client that identifies this solution.

    Any `config=` passed in is preserved field for field; the suffix is merged
    into it rather than replacing it.
    """
    kwargs["config"] = botocore_config(kwargs.get("config"))
    return boto3.client(service_name, **kwargs)


def get_resource(service_name: str, **kwargs: Any) -> Any:
    """A boto3 resource that identifies this solution."""
    kwargs["config"] = botocore_config(kwargs.get("config"))
    return boto3.resource(service_name, **kwargs)
