# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The manual Step-2 "auto-fill race" endpoint — reuses the guided plan_race brain.

Calls the autofill_race handler directly (no TestClient) against a real temp split,
so we exercise the actual planner + hp→card mapping, not a mock.
"""
import json

import pytest


def _make_sft_split(store_mod, split_id, n_train=2000):
    store = store_mod.get_store()
    wd = store.workdir("runs", split_id)
    def chat(u, a):
        return {"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
    train = [chat(f"classify ticket {i}", "billing") for i in range(n_train)]
    eval_ = [chat(f"classify e{i}", "billing") for i in range(50)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": split_id, "shape": "sft", "hasVal": False, "trainRows": n_train}), encoding="utf-8")
    store.commit("runs", split_id)


def _call(split_id, ceiling):
    import app.main as m
    return m.autofill_race(m.AutofillRequest(splitId=split_id, ceiling=ceiling))


def test_autofill_returns_portfolio_with_card_hyperparams(temp_store):
    _make_sft_split(temp_store, "af-sft")
    out = _call("af-sft", 4)
    assert out["supported"] is True and out["objective"] == "sft"
    assert 1 <= len(out["models"]) <= 4
    # Each arm carries the full staged-card hyperparameters (objective-aware LR etc.).
    m0 = out["models"][0]
    assert m0["modelId"] and m0["hp"]["stage"] == "sft"
    assert m0["hp"]["finetuningType"] in ("lora", "qlora", "full", "freeze")
    assert float(m0["hp"]["learningRate"]) == 2.0e-4  # SFT LoRA standard LR


def test_autofill_honors_exact_ceiling_not_snapped_tier(temp_store):
    # A ceiling of 6 must NOT be snapped up to 8 — the exact cap is honored.
    _make_sft_split(temp_store, "af-six", n_train=2000)
    out = _call("af-six", 6)
    assert out["ceiling"] == 6
    assert len(out["models"]) <= 6


def test_autofill_caps_and_reports_when_stopping_short(temp_store, monkeypatch):
    # A high ceiling on a simple set fills fewer arms and flags capped=True (no padding).
    import app.race_planner as rp
    _make_sft_split(temp_store, "af-cap")
    out = _call("af-cap", 16)
    assert out["ceiling"] == 16
    assert out["meaningfulCount"] == len(out["models"]) < 16
    assert out["capped"] is True
    # No duplicate model arms.
    keys = [(x["modelId"], x["hp"]["finetuningType"], x["hp"]["loraVariant"], x["hp"]["prefLoss"])
            for x in out["models"]]
    assert len(keys) == len(set(keys))


def test_autofill_dpo_uses_low_lr_and_no_full_arm(temp_store):
    store = temp_store.get_store()
    wd = store.workdir("runs", "af-dpo")
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}],
             "chosen": {"role": "assistant", "content": "good"},
             "rejected": {"role": "assistant", "content": "bad"}} for i in range(300)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(
        {"messages": [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": "good"}]}) for i in range(30)) + "\n",
        encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": "af-dpo", "shape": "preference", "hasVal": False, "trainRows": 300}), encoding="utf-8")
    store.commit("runs", "af-dpo")
    out = _call("af-dpo", 8)
    assert out["objective"] == "dpo"
    assert all(x["hp"]["stage"] == "dpo" for x in out["models"])
    assert all(x["hp"]["finetuningType"] not in ("full", "freeze") for x in out["models"])
    assert all(float(x["hp"]["learningRate"]) <= 1e-5 for x in out["models"])


def test_autofill_unknown_split_404(temp_store):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _call("does-not-exist", 4)
