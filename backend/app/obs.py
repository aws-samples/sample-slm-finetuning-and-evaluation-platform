# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Structured (JSON) logging — makes CloudWatch Logs queryable.

Emitting one JSON object per line lets CloudWatch Logs Insights filter on fields
(event, raceId, splitId, durationMs, …) instead of grepping free text. Keep it
dependency-free and side-effect-light: a single module-level logger, one helper.

Usage:
    from .obs import log_event
    log_event("race.launch", raceId=rid, models=8)
    log_event("baseline.done", splitId=sid, rows=72, durationMs=123000)
"""
from __future__ import annotations

import json
import logging
import os
import sys

_LEVEL = os.environ.get("SLM_LOG_LEVEL", "INFO").upper()

_logger = logging.getLogger("slm")
if not _logger.handlers:
    # In Lambda a handler is pre-attached to root; add our own stdout handler
    # only when none exists so we don't double-log locally.
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(_LEVEL)
_logger.propagate = False


def log_event(event: str, level: str = "INFO", **fields) -> None:
    """Emit one structured JSON log line: {"event": ..., **fields}."""
    record = {"event": event}
    for k, v in fields.items():
        # Keep values JSON-serializable; fall back to str().
        try:
            json.dumps(v)
            record[k] = v
        except (TypeError, ValueError):
            record[k] = str(v)
    _logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(record))
