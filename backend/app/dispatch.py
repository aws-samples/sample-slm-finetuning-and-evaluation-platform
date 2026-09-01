# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Async worker dispatch — invoke the worker Lambda for long tasks.

Long tasks (Sonnet baseline, LLM-as-judge) make one Bedrock call per eval row
and can exceed API Gateway's 29s limit, so the API/reconcile path fires them at
the worker Lambda asynchronously instead of running inline.

Kept in its own tiny module (no app imports) so both main.py and race.py can use
it without an import cycle. When no worker is configured (local dev), returns
False and the caller runs inline.
"""
from __future__ import annotations

import json
import os
from typing import Any


def dispatch_worker(payload: dict[str, Any]) -> bool:
    """Invoke the worker Lambda (InvocationType=Event). True if dispatched,
    False if no worker is configured (local dev → caller runs inline).

    Stamps the CURRENT tenant into the payload so the worker (which runs with no
    request context) operates on the dispatching user's state + HF token, not the
    default tenant's. Imported lazily to keep this module app-import-free."""
    fn = os.environ.get("SLM_WORKER_FUNCTION")
    if not fn:
        return False
    from .aws_clients import get_client

    if "tenant" not in payload:
        from .tenancy import current_tenant

        payload = {**payload, "tenant": current_tenant()}
    get_client("lambda").invoke(
        FunctionName=fn, InvocationType="Event", Payload=json.dumps(payload).encode()
    )
    return True
