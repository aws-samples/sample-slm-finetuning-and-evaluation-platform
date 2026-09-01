# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Eval latency timing — p50/p90/p99 tail percentiles (container/eval.py).

The leaderboard surfaces p50/p90/p99 so a good median can't hide a slow tail. These
pin the percentile math + the tail behaviour. eval.py loads only stdlib at module
level (torch/vllm are imported lazily inside the generate fns), so a path-load is
safe and cheap.
"""
from __future__ import annotations

import importlib.util
import pathlib

_EVAL_PATH = pathlib.Path(__file__).resolve().parents[2] / "container" / "eval.py"


def _load_eval():
    spec = importlib.util.spec_from_file_location("container_eval", _EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_timing_reports_p50_p90_p99():
    ev = _load_eval()
    lat = [100, 120, 150, 200, 2000]
    t = ev._timing(total_output_tokens=100, gen_seconds=10.0, latencies_ms=lat)
    # p50 is the median; p90/p99 climb toward the slow tail (the 2000ms outlier).
    assert t["p50_latency_ms"] == 150.0
    assert t["p90_latency_ms"] > t["p50_latency_ms"]
    assert t["p99_latency_ms"] >= t["p90_latency_ms"]
    assert t["tokens_per_sec"] == 10.0  # 100 tokens / 10 s


def test_timing_tail_blowup_visible_in_p99():
    """A median that looks fine while the tail is brutal — p99 must expose it."""
    ev = _load_eval()
    # 95 fast requests + 5 very slow ones: p50 stays ~100ms, p99 must be large.
    lat = [100] * 95 + [5000] * 5
    t = ev._timing(50, 5.0, lat)
    assert t["p50_latency_ms"] == 100.0
    assert t["p99_latency_ms"] >= 1000.0  # tail is visible, not hidden by the median


def test_timing_empty_latencies_are_none():
    ev = _load_eval()
    t = ev._timing(0, 0.0, [])
    assert t["p50_latency_ms"] is None
    assert t["p90_latency_ms"] is None
    assert t["p99_latency_ms"] is None


def test_percentile_monotonic():
    ev = _load_eval()
    s = [float(x) for x in range(1, 101)]  # 1..100
    assert ev._percentile(s, 0.5) <= ev._percentile(s, 0.9) <= ev._percentile(s, 0.99)
    assert ev._percentile([], 0.9) == 0.0
