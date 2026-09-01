# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Training loss curves (metric_definitions → CloudWatch → series).

Pure-logic tests — the AWS clients are stubbed so we exercise the regex metric
definitions and the get_metric_data → minutes-elapsed transform without touching
real AWS.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from app import orchestrate as orch


def test_metric_definitions_match_trainer_log_lines():
    """The regexes must extract the numbers HF Trainer actually prints, so
    SageMaker publishes a curve. Guards against drift in the log format."""
    train_line = "{'loss': 0.6577, 'grad_norm': 1.23, 'learning_rate': 4.7e-05, 'epoch': 0.92}"
    eval_line = "{'eval_loss': 0.5123, 'eval_runtime': 3.4, 'epoch': 1.0}"
    by_name = {d["Name"]: d["Regex"] for d in orch.TRAINING_METRIC_DEFINITIONS}

    assert re.search(by_name["train:loss"], train_line).group(1) == "0.6577"
    assert re.search(by_name["eval:loss"], eval_line).group(1) == "0.5123"
    assert re.search(by_name["train:learning_rate"], train_line).group(1) == "4.7e-05"
    assert re.search(by_name["train:epoch"], train_line).group(1) == "0.92"
    assert re.search(by_name["train:grad_norm"], train_line).group(1) == "1.23"
    # train:loss must NOT also capture eval_loss (distinct series).
    assert re.search(by_name["train:loss"], eval_line) is None


def test_curves_route_registered():
    import app.main as m

    assert "/api/train/{job_name}/curves" in {r.path for r in m.app.routes}


class _FakeSm:
    def __init__(self, start, end=None, status="InProgress"):
        self._start, self._end, self._status = start, end, status

    def describe_training_job(self, TrainingJobName):  # noqa: N803
        d = {"TrainingJobStatus": self._status, "TrainingStartTime": self._start}
        if self._end:
            d["TrainingEndTime"] = self._end
        return d


class _FakeCw:
    def __init__(self, results):
        self._results = results
        self.last_kwargs = None

    def get_metric_data(self, **kwargs):
        self.last_kwargs = kwargs
        return {"MetricDataResults": self._results}


class _FakeBoto:
    def __init__(self, sm, cw):
        self._sm, self._cw = sm, cw

    def client(self, name):
        return {"sagemaker": self._sm, "cloudwatch": self._cw}[name]


def _patch_session(monkeypatch, sm, cw):
    monkeypatch.setattr(orch, "load_aws_config", lambda: object())
    monkeypatch.setattr(orch, "_session", lambda cfg: (object(), _FakeBoto(sm, cw)))


def test_fetch_curves_converts_timestamps_to_minutes(temp_store, monkeypatch):
    start = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)
    cw = _FakeCw(
        [
            {
                "Label": "trainLoss",
                "Timestamps": [start + timedelta(minutes=1), start + timedelta(minutes=3)],
                "Values": [0.9, 0.4],
            },
            {"Label": "evalLoss", "Timestamps": [start + timedelta(minutes=2)], "Values": [0.5]},
            {"Label": "learningRate", "Timestamps": [], "Values": []},
            {"Label": "epoch", "Timestamps": [], "Values": []},
            {"Label": "gradNorm", "Timestamps": [], "Values": []},
        ]
    )
    _patch_session(monkeypatch, _FakeSm(start, status="InProgress"), cw)

    out = orch.fetch_training_curves("slm-qwen3-1-7b-abc-1")

    assert out["status"] == "InProgress"
    assert out["series"]["trainLoss"] == [{"x": 1.0, "y": 0.9}, {"x": 3.0, "y": 0.4}]
    assert out["series"]["evalLoss"] == [{"x": 2.0, "y": 0.5}]
    assert out["series"]["learningRate"] == []
    # All metric series are always present (UI indexes them directly).
    assert set(out["series"]) == {"trainLoss", "evalLoss", "learningRate", "epoch", "gradNorm"}


def test_fetch_curves_empty_before_training_starts(temp_store, monkeypatch):
    # No TrainingStartTime yet (still provisioning) → empty series, no CW call.
    cw = _FakeCw([])
    _patch_session(monkeypatch, _FakeSm(None, status="InProgress"), cw)

    out = orch.fetch_training_curves("slm-job-pending")

    assert out["startTime"] is None
    assert all(v == [] for v in out["series"].values())
    assert cw.last_kwargs is None  # never queried CloudWatch


# --- curve snapshot persistence (survives CloudWatch retention) ---


class _FakeSmStatus:
    def __init__(self, status):
        self._status = status

    def describe_training_job(self, TrainingJobName):  # noqa: N803
        return {"TrainingJobStatus": self._status}


def test_snapshot_skips_running_job(temp_store, monkeypatch):
    from app import storage

    monkeypatch.setattr(orch, "load_aws_config", lambda: object())
    monkeypatch.setattr(orch, "_session", lambda cfg: (object(), _FakeBoto(_FakeSmStatus("InProgress"), None)))
    assert orch.snapshot_curves_if_terminal("slm-job-running") is False
    assert storage.has_curves("slm-job-running") is False


def test_snapshot_persists_on_terminal_and_then_served(temp_store, monkeypatch):
    from app import storage

    # describe → Completed; fetch_training_curves stubbed to a known curve.
    monkeypatch.setattr(orch, "load_aws_config", lambda: object())
    monkeypatch.setattr(orch, "_session", lambda cfg: (object(), _FakeBoto(_FakeSmStatus("Completed"), None)))
    fake_curve = {"jobName": "slm-done", "status": "Completed", "startTime": "t",
                  "series": {"trainLoss": [{"x": 1.0, "y": 0.5}], "evalLoss": [], "learningRate": [],
                             "epoch": [], "gradNorm": []}}
    # Only stub the CloudWatch path (called once, before a snapshot exists).
    monkeypatch.setattr(orch, "_fetch_curves_from_cloudwatch", None, raising=False)
    calls = {"n": 0}

    real = orch.fetch_training_curves

    def fake_fetch(job):
        # First call (inside snapshot, no snapshot yet) returns the CW curve;
        # after persist, the real fn should serve the snapshot.
        if storage.has_curves(job):
            return real(job)  # exercise the snapshot-read branch
        calls["n"] += 1
        return fake_curve

    monkeypatch.setattr(orch, "fetch_training_curves", fake_fetch)

    assert orch.snapshot_curves_if_terminal("slm-done") is True
    assert storage.has_curves("slm-done") is True
    # Idempotent — second call writes nothing.
    assert orch.snapshot_curves_if_terminal("slm-done") is False
    # The persisted snapshot is what gets served now (CloudWatch not re-queried).
    served = real("slm-done")
    assert served["series"]["trainLoss"] == [{"x": 1.0, "y": 0.5}]
    assert calls["n"] == 1  # CloudWatch fetched exactly once, at snapshot time
