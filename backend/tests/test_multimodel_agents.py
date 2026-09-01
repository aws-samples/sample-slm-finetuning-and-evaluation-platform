# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for multi-model baselines + the triage/interpret agent wiring."""
from __future__ import annotations


def test_baseline_registry_and_resolution():
    from app.baseline import BASELINE_MODELS, DEFAULT_BASELINE, baseline_model

    # The four frontier tiers the user picked are present.
    for key in ("haiku-4-5", "sonnet-4-5", "sonnet-4-6", "opus-4-8"):
        assert key in BASELINE_MODELS
        assert BASELINE_MODELS[key]["modelId"].startswith(("us.", "global."))
        assert BASELINE_MODELS[key]["inPer1k"] > 0
    # Default + unknown both resolve to Sonnet 4.5.
    assert baseline_model(None)["modelId"] == BASELINE_MODELS[DEFAULT_BASELINE]["modelId"]
    assert baseline_model("nonsense")["modelId"] == BASELINE_MODELS[DEFAULT_BASELINE]["modelId"]


def test_baseline_filenames_are_model_keyed():
    """Sonnet 4.5 keeps the legacy filename (back-compat); others are namespaced."""
    from app.baseline import _baseline_file, _status_file

    assert _baseline_file("sonnet-4-5") == "sonnet_baseline.json"
    assert _baseline_file("haiku-4-5") == "baseline_haiku-4-5.json"
    assert _status_file("sonnet-4-5") == "sonnet_baseline_status.json"
    assert _status_file("opus-4-8") == "baseline_opus-4-8_status.json"


def test_back_compat_sonnet_constants():
    """judge.py + selfheal.py import these — they must stay valid."""
    from app.baseline import SONNET_INPUT_PER_1K, SONNET_MODEL_ID, SONNET_OUTPUT_PER_1K

    assert "sonnet-4-5" in SONNET_MODEL_ID
    assert SONNET_INPUT_PER_1K == 0.003 and SONNET_OUTPUT_PER_1K == 0.015


def test_load_all_baselines(temp_store):
    from app import baseline
    from app.store import get_store
    from app.storage import RUNS

    # _save_baseline only writes if the split exists (has dataset_info.json).
    store = get_store()
    wd = store.workdir(RUNS, "split-mm")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    store.commit(RUNS, "split-mm")

    assert baseline.load_all_baselines("split-mm") == []
    baseline._save_baseline("split-mm", {"token_f1": 0.5, "baseline": {"label": "Haiku"}}, "haiku-4-5")
    baseline._save_baseline("split-mm", {"token_f1": 0.7, "baseline": {"label": "Sonnet"}}, "sonnet-4-5")
    got = baseline.load_all_baselines("split-mm")
    assert len(got) == 2
    labels = {b["baseline"]["label"] for b in got}
    assert labels == {"Haiku", "Sonnet"}


def test_new_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in [
        "/api/baseline/models",
        "/api/race/{race_id}/triage",
        "/api/race/{race_id}/triage/{model_id}",  # async poll
        "/api/leaderboard/interpret",
        "/api/leaderboard/interpret/{split_id}",  # async poll
    ]:
        assert p in paths, f"missing route {p}"


def test_triage_async_inline_without_worker(temp_store, monkeypatch):
    """No worker (local) → start_triage runs inline + persists; status flips done."""
    from app import investigator

    monkeypatch.delenv("SLM_WORKER_FUNCTION", raising=False)
    monkeypatch.setattr(investigator, "triage_failure",
                        lambda job, model, reason: {"summary": "ok", "retryable": True})
    out = investigator.start_triage("race1", "qwen3-1.7b", "job-abc", "OOM")
    assert out["status"] == "done"
    assert out["result"]["summary"] == "ok"
    # persisted + pollable
    assert investigator.load_triage("race1", "qwen3-1.7b")["summary"] == "ok"
    assert investigator.triage_status("race1", "qwen3-1.7b")["status"] == "done"


def test_interpret_async_dispatches_when_worker_set(temp_store, monkeypatch):
    """Worker configured (hosted) → start_interpret dispatches + returns running."""
    from app import investigator

    monkeypatch.setenv("SLM_WORKER_FUNCTION", "slm-worker")
    dispatched = {}
    monkeypatch.setattr(investigator, "dispatch_worker" if hasattr(investigator, "dispatch_worker") else "_noop", lambda *a, **k: True, raising=False)
    # dispatch_worker is imported inside start_interpret; patch at its source.
    import app.dispatch as d
    monkeypatch.setattr(d, "dispatch_worker", lambda payload: dispatched.update(payload) or True)
    out = investigator.start_interpret("split-9", "cheapest")
    assert out["status"] == "running"
    assert dispatched["task"] == "investigate_interpret"
    assert investigator.interpret_status("split-9")["status"] == "running"


def test_interpret_needs_rows(temp_store, monkeypatch):
    """interpret_results returns an error (not a crash) when there are no evals."""
    from app import investigator

    monkeypatch.setattr(investigator, "build_leaderboard", lambda sid: [], raising=False)
    # build_leaderboard is imported inside the function; patch via the module it pulls from.
    import app.leaderboard as lb
    monkeypatch.setattr(lb, "build_leaderboard", lambda sid: [])
    monkeypatch.setattr("app.baseline.load_all_baselines", lambda sid: [])
    out = investigator.interpret_results("empty-split")
    assert "error" in out
