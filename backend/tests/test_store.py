# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""State store + storage layer round-trips (LocalStore on a temp dir)."""
import json


def test_persist_split_round_trip(temp_store, sample_rows):
    from app import storage

    train, ev = sample_rows[:8], sample_rows[8:]
    split_id, run_dir = storage.persist_split(train, ev, {"name": "demo", "mode": "auto"})
    assert storage.split_dir(split_id) is not None
    assert storage.split_meta(split_id)["name"] == "demo"
    # Files written in the canonical layout.
    assert (run_dir / "train.jsonl").exists()
    assert (run_dir / "eval.jsonl").exists()
    assert (run_dir / "dataset_info.json").exists()


def test_persist_split_idempotent(temp_store, sample_rows):
    from app import storage

    train, ev = sample_rows[:8], sample_rows[8:]
    id1, _ = storage.persist_split(train, ev, {"name": "a"})
    id2, _ = storage.persist_split(train, ev, {"name": "a"})
    assert id1 == id2  # content hash → same id


def test_list_datasets(temp_store, sample_rows):
    from app import storage

    storage.persist_split(sample_rows[:8], sample_rows[8:], {"name": "ds1"})
    libs = storage.list_datasets()
    assert len(libs) == 1
    assert libs[0]["name"] == "ds1"
    assert libs[0]["hasBaseline"] is False


def test_eval_only_dataset(temp_store, sample_rows):
    from app import storage

    sid, run_dir = storage.persist_eval_only(sample_rows[:5], {"name": "holdout"})
    assert storage.split_dir(sid) is not None
    meta = storage.split_meta(sid)
    assert meta["evalOnly"] is True
    assert meta["trainRows"] == 0 and meta["evalRows"] == 5
    # dataset_info registers ONLY the eval set.
    info = json.loads((run_dir / "dataset_info.json").read_text())
    assert "eval_split" in info and "train_split" not in info


def test_persist_split_with_val(temp_store, sample_rows):
    from app import storage

    train, ev, val = sample_rows[:6], sample_rows[6:8], sample_rows[8:]
    split_id, run_dir = storage.persist_split(
        train, ev, {"name": "v", "hasVal": True, "valMode": "file", "valRows": len(val)}, val=val
    )
    # val.jsonl written + registered in dataset_info.
    assert (run_dir / "val.jsonl").exists()
    info = json.loads((run_dir / "dataset_info.json").read_text())
    assert "val_split" in info and info["val_split"]["file_name"] == "val.jsonl"
    # Library surfaces hasVal.
    lib = next(d for d in storage.list_datasets() if d["splitId"] == split_id)
    assert lib["hasVal"] is True and lib["valRows"] == len(val)


def test_val_changes_split_id(temp_store, sample_rows):
    from app import storage

    train, ev, val = sample_rows[:6], sample_rows[6:8], sample_rows[8:]
    no_val, _ = storage.persist_split(train, ev, {"name": "x"})
    with_val, _ = storage.persist_split(train, ev, {"name": "x"}, val=val)
    assert no_val != with_val  # val participates in the content hash


def test_render_uses_val_split_when_present(temp_store, sample_rows):
    from app import storage
    from app.render import render_train_yaml
    from app.catalog import Hyperparams, get_model

    train, ev, val = sample_rows[:6], sample_rows[6:8], sample_rows[8:]
    sid, _ = storage.persist_split(train, ev, {"hasVal": True}, val=val)
    y = render_train_yaml(get_model("qwen3-1.7b"), Hyperparams(), sid)
    assert "eval_dataset: val_split" in y  # in-training eval uses VAL, not held-out

    # A dataset with no val falls back to eval_split (unchanged behaviour).
    sid2, _ = storage.persist_split(train, ev, {})
    y2 = render_train_yaml(get_model("qwen3-1.7b"), Hyperparams(), sid2)
    assert "eval_dataset: eval_split" in y2


def test_early_stopping_keys_only_with_val(temp_store, sample_rows):
    import yaml
    from app import storage
    from app.render import render_train_yaml
    from app.catalog import Hyperparams, get_model

    train, ev, val = sample_rows[:6], sample_rows[6:8], sample_rows[8:]
    model = get_model("qwen3-1.7b")
    hp = Hyperparams(early_stopping_enabled=True, early_stopping_patience=2, num_train_epochs=10)

    # With a val set → ES keys present; num_train_epochs is the ceiling. The
    # LLaMA-Factory 0.9.4 key is `early_stopping_steps` (its value is the
    # patience); `early_stopping_patience` is NOT a valid key and must never be
    # emitted (it crashes the HfArgumentParser).
    sid, _ = storage.persist_split(train, ev, {"hasVal": True}, val=val)
    doc = yaml.safe_load(render_train_yaml(model, hp, sid))
    assert doc["early_stopping_steps"] == 2
    assert "early_stopping_patience" not in doc  # invalid key — would fail the job
    assert doc["load_best_model_at_end"] is True
    assert doc["metric_for_best_model"] == "eval_loss"
    assert doc["greater_is_better"] is False
    assert doc["eval_dataset"] == "val_split"

    # No val set → ES requested but silently ignored (no signal to stop on).
    sid2, _ = storage.persist_split(train, ev, {})
    doc2 = yaml.safe_load(render_train_yaml(model, hp, sid2))
    assert "early_stopping_steps" not in doc2
    assert "load_best_model_at_end" not in doc2


def test_config_root_json_merge_preserves(temp_store):
    from app import aws_config

    aws_config.save_config({"resetCutoff": "2026-06-02T00:00:00Z"})
    aws_config.save_config({"region": "us-east-1", "bucket": ""})  # empty skipped
    saved = aws_config._saved()
    assert saved["resetCutoff"] == "2026-06-02T00:00:00Z"  # preserved
    assert saved["region"] == "us-east-1"
    assert "bucket" not in saved  # empty value not written


def test_missing_split_returns_none(temp_store):
    from app import storage

    assert storage.split_dir("doesnotexist") is None
    assert storage.split_meta("doesnotexist") == {}


def test_archived_dataset_excluded_by_default(temp_store, sample_rows):
    from app import storage

    sid, _ = storage.persist_split(sample_rows[:8], sample_rows[8:], {"name": "ds"})
    assert len(storage.list_datasets()) == 1
    # Archive it → gone from the default list, still there with include_archived.
    assert storage.set_dataset_archived(sid, True)
    assert storage.list_datasets() == []
    assert len(storage.list_datasets(include_archived=True)) == 1
    assert storage.is_dataset_archived(sid) is True
    # Restore → back in the default list.
    storage.set_dataset_archived(sid, False)
    assert len(storage.list_datasets()) == 1
    assert storage.is_dataset_archived(sid) is False


def test_recommended_metric_persists_and_surfaces(temp_store, sample_rows):
    from app import storage

    sid, _ = storage.persist_split(sample_rows[:8], sample_rows[8:], {"name": "ds"})
    # No recommendation yet.
    ds = storage.list_datasets(include_archived=True)[0]
    assert ds.get("recommendedRankMetric") is None
    # Record one (as synthesize_proposal does) → meta + list_datasets carry it.
    assert storage.set_recommended_metric(sid, "label_accuracy", ["per_class_accuracy"])
    assert storage.split_meta(sid)["recommendedRankMetric"] == "label_accuracy"
    ds = storage.list_datasets(include_archived=True)[0]
    assert ds["recommendedRankMetric"] == "label_accuracy"
    # Unknown dataset → False.
    assert storage.set_recommended_metric("nope", "token_f1", []) is False


def test_recommended_kto_weights_persist_and_surface(temp_store, sample_rows):
    from app import storage

    sid, _ = storage.persist_split(sample_rows[:8], sample_rows[8:], {"name": "ds"})
    # No KTO recommendation yet.
    ds = storage.list_datasets(include_archived=True)[0]
    assert ds.get("recommendedChosenWeight") is None
    assert ds.get("recommendedRejectedWeight") is None
    # Record the profiler's λD/λU → meta + list_datasets carry them.
    assert storage.set_recommended_kto_weights(sid, 1.0, 3.0)
    meta = storage.split_meta(sid)
    assert meta["recommendedChosenWeight"] == 1.0 and meta["recommendedRejectedWeight"] == 3.0
    ds = storage.list_datasets(include_archived=True)[0]
    assert ds["recommendedChosenWeight"] == 1.0 and ds["recommendedRejectedWeight"] == 3.0
    # Unknown dataset → False.
    assert storage.set_recommended_kto_weights("nope", 1.0, 2.0) is False


def test_load_answers_roundtrip(temp_store):
    from app import investigator as inv

    assert inv.load_answers("missing") == {}
    inv._save("dsX", "answers.json", {"answers": {"q1": "many labels shared"}})
    assert inv.load_answers("dsX") == {"q1": "many labels shared"}


# --- Phase 2: preference (DPO) datasets ------------------------------------- #

def _pref_rows(n=4):
    return [
        {"messages": [{"role": "user", "content": f"q{i}"}],
         "chosen": {"role": "assistant", "content": f"good{i}"},
         "rejected": {"role": "assistant", "content": f"bad{i}"}}
        for i in range(n)
    ]


def test_persist_preference_split_shapes(temp_store):
    from app import storage
    import json

    rows = _pref_rows(4)
    sid, run_dir = storage.persist_preference_split(rows, {"name": "pref", "trainRows": 4})
    info = json.loads((run_dir / "dataset_info.json").read_text())
    # train is ranking-shaped; eval stays plain messages (the shared gen eval).
    assert info["train_split"]["ranking"] is True
    assert info["train_split"]["columns"] == {
        "messages": "messages", "chosen": "chosen", "rejected": "rejected"}
    assert "ranking" not in info["eval_split"]
    assert info["eval_split"]["columns"] == {"messages": "messages"}
    # eval gold = the chosen response.
    ev = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines() if l.strip()]
    assert ev[0]["messages"][-1] == {"role": "assistant", "content": "good0"}
    # eval row ALSO carries chosen_ref + rejected_ref (for the chosen_win_rate
    # metric) without changing the gold turn (disjointness invariant preserved).
    assert ev[0]["chosen_ref"] == "good0"
    assert ev[0]["rejected_ref"] == "bad0"
    # meta + list_datasets surface the shape.
    assert storage.split_meta(sid)["shape"] == "preference"
    lib = next(d for d in storage.list_datasets() if d["splitId"] == sid)
    assert lib["shape"] == "preference"


def test_preference_split_with_val(temp_store):
    from app import storage
    import json

    rows = _pref_rows(6)
    sid, run_dir = storage.persist_preference_split(
        rows[:4], {"name": "pv", "hasVal": True}, val=rows[4:])
    info = json.loads((run_dir / "dataset_info.json").read_text())
    # val is also ranking-shaped (in-training preference eval).
    assert info["val_split"]["ranking"] is True
    assert (run_dir / "val.jsonl").exists()


def test_dpo_render_eval_shape_matches_data(temp_store):
    """DPO's in-training eval must be ranking-shaped: with a val set it points at
    val_split; WITHOUT one it drops eval entirely (the messages eval_split can't
    be DPO's eval target). SFT is unaffected."""
    import yaml
    from app import storage
    from app.catalog import Hyperparams, get_model
    from app.render import render_train_yaml

    model = get_model("qwen3-1.7b")
    pref = _pref_rows(2)
    no_val, _ = storage.persist_preference_split(pref, {"trainRows": 2})
    doc = yaml.safe_load(render_train_yaml(model, Hyperparams(stage="dpo"), no_val))
    assert "eval_dataset" not in doc and "eval_strategy" not in doc

    with_val, _ = storage.persist_preference_split(
        pref, {"trainRows": 2, "hasVal": True}, val=pref)
    doc2 = yaml.safe_load(render_train_yaml(model, Hyperparams(stage="dpo"), with_val))
    assert doc2["eval_dataset"] == "val_split"


def test_race_rejects_objective_dataset_shape_mismatch(temp_store, monkeypatch):
    """The race endpoint guards objective ↔ dataset shape BEFORE any launch:
    DPO on a messages split (and SFT on a preference split) → 400, no job."""
    import pytest
    from fastapi import HTTPException
    import app.main as m
    from app import storage, race as rm

    # A messages (SFT) split and a preference (DPO) split.
    sft_id, _ = storage.persist_split(
        [{"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "a"}]}],
        [{"messages": [{"role": "user", "content": "e"},
                       {"role": "assistant", "content": "b"}]}],
        {"name": "sft", "trainRows": 1, "evalRows": 1})
    pref_id, _ = storage.persist_preference_split(
        [{"messages": [{"role": "user", "content": "q"}],
          "chosen": {"role": "assistant", "content": "good"},
          "rejected": {"role": "assistant", "content": "bad"}}],
        {"name": "pref", "trainRows": 1})

    launched = []
    monkeypatch.setattr(rm, "launch_training_job", lambda **kw: launched.append(kw) or {"jobName": "x"})

    def cfg(stage):
        return m.RaceModelConfig(modelId="qwen3-1.7b", stage=stage)

    # DPO on a messages split → 400 mentioning the needed preference shape.
    with pytest.raises(HTTPException) as ei:
        m.race_launch(m.RaceRequest(splitId=sft_id, models=[cfg("dpo")]))
    assert ei.value.status_code == 400 and "preference" in str(ei.value.detail).lower()

    # SFT on a preference split → 400 mentioning messages.
    with pytest.raises(HTTPException) as ei:
        m.race_launch(m.RaceRequest(splitId=pref_id, models=[cfg("sft")]))
    assert ei.value.status_code == 400 and "messages" in str(ei.value.detail).lower()

    assert launched == []  # neither mismatch reached a billable launch


def test_smoke_test_skips_preference_datasets(temp_store, monkeypatch):
    """The smoke test always trains SFT, so its auto-picked dataset must be
    messages-shaped — never a preference (ranking) split, which would fail the
    engine. Caught by deploy-testing: the picker grabbed a preference split."""
    import app.main as m
    from app import storage, orchestrate

    # Only a preference dataset exists → smoke test must refuse (no SFT data).
    storage.persist_preference_split(
        [{"messages": [{"role": "user", "content": "q"}],
          "chosen": {"role": "assistant", "content": "good"},
          "rejected": {"role": "assistant", "content": "bad"}}],
        {"name": "pref-only", "trainRows": 1})

    launched = []
    monkeypatch.setattr(m, "launch_training_job",
                        lambda **kw: launched.append(kw) or {"jobName": "x", "imageTag": "stable",
                                                             "imageUri": "uri"})
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        m.smoke_test_model("qwen3-1.7b")
    assert ei.value.status_code == 400 and launched == []

    # Add a messages dataset → smoke test now picks IT, not the preference one.
    sft_id, _ = storage.persist_split(
        [{"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "a"}]}],
        [{"messages": [{"role": "user", "content": "e"},
                       {"role": "assistant", "content": "b"}]}],
        {"name": "sft", "trainRows": 1, "evalRows": 1})
    res = m.smoke_test_model("qwen3-1.7b")
    assert res["splitId"] == sft_id  # the messages split, not the preference one
    assert launched and launched[0]["split_id"] == sft_id


def test_serverless_smoke_test_handles_no_image_tag(temp_store, monkeypatch):
    """Regression: the serverless engine is a MANAGED recipe → its launch result
    has NO imageTag/imageUri. smoke_test_model must not KeyError on res['imageTag']
    (the bug 500'd the UI AFTER the job already launched). It should record a
    PENDING serverless verification and return imageTag=''."""
    import pytest
    import app.main as m
    from app import storage, verifications
    from app.engines import base as engine_base

    # Enable the serverless engine for the test (the endpoint guards on the flag).
    monkeypatch.setattr(engine_base, "engine_enabled",
                        lambda name: name == "sagemaker_serverless")
    # The endpoint also preflights SLM_SERVERLESS_PYTHON (the V3-SDK interpreter the
    # serverless engine shells out to) and 400s when unset. This test mocks the
    # launch itself, so set it to any value to get past the preflight to the
    # imageTag-handling code under test.
    monkeypatch.setenv("SLM_SERVERLESS_PYTHON", "/usr/bin/true")
    # A messages dataset to train on.
    storage.persist_split(
        [{"messages": [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "a"}]}],
        [{"messages": [{"role": "user", "content": "e"},
                       {"role": "assistant", "content": "b"}]}],
        {"name": "sft", "trainRows": 1, "evalRows": 1})
    # Serverless launch returns NO imageTag/imageUri (managed recipe) — the shape
    # that triggered the KeyError.
    monkeypatch.setattr(m, "launch_training_job",
                        lambda **kw: {"jobName": "serverless-job-1", "engine": "sagemaker_serverless",
                                      "instanceType": "serverless"})
    # qwen3-8b is serverless-tagged (huggingface-reasoning-qwen3-8b).
    res = m.smoke_test_model("qwen3-8b", engine="sagemaker_serverless")
    assert res["jobName"] == "serverless-job-1"
    assert res["engine"] == "sagemaker_serverless"
    assert res["imageTag"] == "" and res["imageUri"] == ""  # gracefully absent
    # A PENDING verification was recorded on the "serverless" surface.
    st = verifications.model_status_map("qwen3-8b").get("serverless")
    assert st and st["status"] == "pending" and st["jobName"] == "serverless-job-1"


def test_persist_kto_split_shapes_and_eval(temp_store):
    from app import storage
    import json
    rows = [
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "good1"}], "kto_tag": True},
        {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "bad2"}], "kto_tag": False},
        {"messages": [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "good3"}], "kto_tag": True},
    ]
    sid, run_dir = storage.persist_kto_split(rows, {"name": "kto", "trainRows": 3})
    info = json.loads((run_dir / "dataset_info.json").read_text())
    assert info["train_split"]["columns"] == {"messages": "messages", "kto_tag": "kto_tag"}
    assert "ranking" not in info["train_split"]
    assert info["eval_split"]["columns"] == {"messages": "messages"}
    # eval gold = DESIRABLE completions only.
    ev = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines() if l.strip()]
    assert [e["messages"][-1]["content"] for e in ev] == ["good1", "good3"]
    assert storage.split_meta(sid)["shape"] == "kto"
    lib = next(d for d in storage.list_datasets() if d["splitId"] == sid)
    assert lib["shape"] == "kto"


def test_kto_three_way_split_keeps_both_classes(temp_store):
    """A KTO 3-way split must keep BOTH good/bad classes in train, val AND test
    (KTO loss needs both to contrast; the test gold needs desirable rows)."""
    import app.main as m
    good = [{"kto_tag": True, "i": i} for i in range(20)]
    bad = [{"kto_tag": False, "i": i} for i in range(20)]
    tr, val, te = m._kto_three_way_split(good + bad, 0.1, 0.1, 42)
    for split in (tr, val, te):
        assert split is not None
        assert any(r["kto_tag"] for r in split), "needs a desirable row"
        assert any(not r["kto_tag"] for r in split), "needs an undesirable row"
    # a class with <3 rows can't give to test+val and keep a train row → stays in train
    tr2, val2, te2 = m._kto_three_way_split(good + [{"kto_tag": False, "i": 0}], 0.1, 0.1, 42)
    assert sum(1 for r in tr2 if not r["kto_tag"]) == 1  # the lone bad row stayed in train


def test_persist_rlvr_split_shapes_and_eval(temp_store):
    from app import storage
    import json
    rows = [
        {"messages": [{"role": "user", "content": "2+2?"}], "ground_truth": "4"},
        {"messages": [{"role": "user", "content": "3+5?"}], "ground_truth": "8"},
        {"messages": [{"role": "user", "content": "10-1?"}], "ground_truth": "9"},
    ]
    sid, run_dir = storage.persist_rlvr_split(rows, {"name": "rlvr", "trainRows": 3})
    info = json.loads((run_dir / "dataset_info.json").read_text())
    # train entry records messages + the verifiable ground_truth column.
    assert info["train_split"]["columns"] == {"messages": "messages", "ground_truth": "ground_truth"}
    assert "ranking" not in info["train_split"]
    # eval entry is plain messages (shared leaderboard eval).
    assert info["eval_split"]["columns"] == {"messages": "messages"}
    # eval gold = the ground_truth appended as the assistant turn.
    ev = [json.loads(l) for l in (run_dir / "eval.jsonl").read_text().splitlines() if l.strip()]
    assert all(e["messages"][-1]["role"] == "assistant" for e in ev)
    assert set(e["messages"][-1]["content"] for e in ev) == {"4", "8", "9"}
    assert storage.split_meta(sid)["shape"] == "rlvr"
    lib = next(d for d in storage.list_datasets() if d["splitId"] == sid)
    assert lib["shape"] == "rlvr"


def test_create_rlvr_dataset_endpoint(temp_store, monkeypatch):
    """POST /api/datasets/rlvr: valid prompt+ground_truth JSONL → an rlvr-shaped
    split with a held-out test set. Stubs file resolution to avoid multipart."""
    import asyncio
    import json
    import app.main as m

    jsonl = "\n".join(
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"}], "ground_truth": str(i)})
        for i in range(20)
    )

    async def _fake_resolve(file, upload_id, what):
        return jsonl

    monkeypatch.setattr(m, "_resolve_text", _fake_resolve)
    out = asyncio.run(m.create_rlvr_dataset(file=None, upload_id="x", test_ratio=0.1, val_ratio=0.1, seed=42, name="math-rlvr"))
    assert out["shape"] == "rlvr"
    assert out["totalRows"] == 20
    assert out["trainRows"] >= 1 and out["testRows"] >= 1
    # the persisted split is rlvr-shaped + discoverable in the library
    from app import storage
    assert storage.split_meta(out["splitId"])["shape"] == "rlvr"


def test_create_rlvr_dataset_rejects_missing_ground_truth(temp_store, monkeypatch):
    import asyncio
    import json
    import app.main as m
    from fastapi import HTTPException

    # rows with no ground_truth → all invalid → 400
    jsonl = "\n".join(
        json.dumps({"messages": [{"role": "user", "content": f"q{i}"}]}) for i in range(5)
    )

    async def _fake_resolve(file, upload_id, what):
        return jsonl

    monkeypatch.setattr(m, "_resolve_text", _fake_resolve)
    try:
        asyncio.run(m.create_rlvr_dataset(file=None, upload_id="x", test_ratio=0.1, val_ratio=0.1, seed=42, name="bad"))
        assert False, "expected HTTPException"
    except HTTPException as e:
        assert e.status_code == 400


def test_three_way_split_disjoint_and_sized(temp_store):
    """The unified one-file split → disjoint train/val/test at the requested ratios."""
    import app.main as m
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}],
             "chosen": {"role": "assistant", "content": f"c{i}"},
             "rejected": {"role": "assistant", "content": f"r{i}"}} for i in range(100)]
    tr, val, te = m._three_way_split(rows, 0.1, 0.1, 42)
    assert (len(tr), len(val), len(te)) == (80, 10, 10)
    k = lambda L: {r["messages"][0]["content"] for r in L}
    assert k(tr).isdisjoint(k(val)) and k(tr).isdisjoint(k(te)) and k(val).isdisjoint(k(te))
    # tiny input never starves train
    tr2, val2, te2 = m._three_way_split(rows[:3], 0.1, 0.1, 42)
    assert len(tr2) >= 1


def test_dpo_test_gold_is_held_out_from_train(temp_store):
    """The DPO held-out test gold (chosen of the test portion) must be DISJOINT
    from the training rows — a genuine benchmark, not derived from train."""
    import app.main as m
    from app.storage import persist_preference_split, preference_eval_rows
    import json
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}],
             "chosen": {"role": "assistant", "content": f"c{i}"},
             "rejected": {"role": "assistant", "content": f"r{i}"}} for i in range(100)]
    tr, val, te = m._three_way_split(rows, 0.1, 0.1, 42)
    sid, run = persist_preference_split(
        tr, {"name": "x", "hasVal": True}, val=val, eval_=preference_eval_rows(te))
    ev = [json.loads(l) for l in (run / "eval.jsonl").read_text().splitlines() if l.strip()]
    ev_gold = {m_["messages"][-1]["content"] for m_ in ev}
    train_chosen = {r["chosen"]["content"] for r in tr}
    assert ev_gold and ev_gold.isdisjoint(train_chosen)


def test_three_way_split_tiny_dataset_keeps_held_out_test(temp_store):
    """A positive test_ratio that rounds to 0 on a small dataset must still yield
    ≥1 held-out test row (else persist derives eval gold from train → overlap)."""
    import app.main as m
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}],
             "chosen": {"role": "assistant", "content": f"c{i}"},
             "rejected": {"role": "assistant", "content": f"r{i}"}} for i in range(8)]
    tr, val, te = m._three_way_split(rows, 0.1, 0.1, 42)
    assert te and len(te) >= 1 and val and len(val) >= 1 and len(tr) >= 1
    # still disjoint + complete
    assert len(tr) + len(val) + len(te) == 8


def test_hf_preference_row_rejects_empty_prompt():
    """An empty prompt cell (configured prompt_field) must NOT leak an empty-user
    pair into training — mirrors the SFT/KTO guards."""
    from app.hf_ingest import preference_row_from_hf, autodetect_preference_mapping
    pm = autodetect_preference_mapping(["prompt", "chosen", "rejected"])
    assert preference_row_from_hf({"prompt": "", "chosen": "a", "rejected": "b"}, pm) is None
    assert preference_row_from_hf({"prompt": "   ", "chosen": "a", "rejected": "b"}, pm) is None
    assert preference_row_from_hf({"prompt": "q", "chosen": "a", "rejected": "b"}, pm) is not None
