# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic dataset profiler + recommended eval strategy."""
import json

from app import profiler


def _chat(user: str, assistant: str, system: str | None = None) -> dict:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return {"messages": msgs}


def _make_split(store_mod, split_id: str, train: list[dict], eval_: list[dict], meta: dict | None = None):
    """Write train.jsonl/eval.jsonl/dataset_info.json/meta.json into a RUNS split."""
    store = store_mod.get_store()
    wd = store.workdir("runs", split_id)
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(meta or {"name": split_id}), encoding="utf-8")
    store.commit("runs", split_id)


# --- task detection (shared with eval.py) ---------------------------------

def test_detect_task():
    assert profiler.detect_task('{"narrative": "x"}') == "json"
    assert profiler.detect_task("urgency: high") == "label"
    assert profiler.detect_task("42.5") == "numeric"
    assert profiler.detect_task("The user is browsing hotels in Lisbon today.") == "text"


# --- JSON-generation dataset (the travel-concierge shape, with <think>) ----

def test_profile_json_dataset_recommends_json_strategy(temp_store):
    think = "<think>\n\n</think>\n\n"
    train = [_chat(f"events {i}", think + json.dumps({"narrative": f"user does {i}"}),
                   system="You are an analytics engine. Respond with JSON.") for i in range(30)]
    eval_ = [_chat(f"events e{i}", think + json.dumps({"narrative": f"eval {i}"})) for i in range(8)]
    _make_split(temp_store, "json-ds", train, eval_)

    p = profiler.profile_dataset("json-ds")
    assert p["eval"]["dominantTask"] == "json"
    # scaffold detector flags the <think> wrapper (generalized from thinkPrefixRate)
    assert p["eval"]["scaffold"]["rate"] == 1.0
    assert "think" in p["eval"]["scaffold"]["patterns"]
    # raw JSON invalid (think prefix) but valid after stripping — the key finding.
    assert p["eval"]["json"]["goldValidRaw"] == 0.0
    assert p["eval"]["json"]["goldValidStripped"] == 1.0
    rec = p["recommendation"]
    assert rec["rankMetric"] == "json_structural"
    assert "json_valid" in rec["alsoWatch"]
    assert any("wrap the json" in n.lower() for n in rec["rationale"])
    # fixed system prompt detected + its text surfaced
    assert p["structure"]["systemPromptFixed"] is True
    assert p["structure"]["fixedSystemPrompt"] == "You are an analytics engine. Respond with JSON."


def test_profile_rlvr_dataset(temp_store):
    # RLVR train rows are prompt-only messages + a numeric ground_truth.
    train = [{"messages": [{"role": "user", "content": f"What is {i}+{i}?"}],
              "ground_truth": str(i * 2)} for i in range(20)]
    # held-out eval is messages-shaped (ground_truth appended as the gold answer)
    eval_ = [_chat(f"What is {i}+{i}?", str(i * 2)) for i in range(5)]
    _make_split(temp_store, "rlvr-ds", train, eval_, meta={"name": "rlvr-ds", "shape": "rlvr"})

    p = profiler.profile_dataset("rlvr-ds")
    # RLVR gets its OWN profile block (NOT treated as malformed SFT).
    assert "rlvr" in p and "train" not in p
    r = p["rlvr"]
    assert r["rows"] == 20 and r["malformed"] == 0
    assert r["emptyGroundTruth"] == 0
    assert r["numericGroundTruthRate"] == 1.0          # all numeric → verifiable
    assert r["groundTruthTask"] == "numeric"            # detected task for the reward guard
    # objective recommendation is rlvr (not the sft fallthrough)
    assert p["objective"]["objective"] == "rlvr"
    assert any("verifiable" in n.lower() or "reward" in n.lower() for n in p["objective"]["rationale"])


def test_profile_rlvr_flags_prose_ground_truth(temp_store):
    # Free-text ground_truth → NOT numeric; the reward-domain guard should later warn.
    train = [{"messages": [{"role": "user", "content": f"Summarize doc {i}"}],
              "ground_truth": "This is a long prose answer that is not a checkable number at all."}
             for i in range(12)]
    eval_ = [_chat("Summarize doc 0", "prose")]
    _make_split(temp_store, "rlvr-prose", train, eval_, meta={"name": "rlvr-prose", "shape": "rlvr"})
    p = profiler.profile_dataset("rlvr-prose")
    assert p["rlvr"]["groundTruthTask"] == "text"
    assert p["rlvr"]["numericGroundTruthRate"] == 0.0


def test_detect_scaffold_generalizes():
    assert "think" in profiler.detect_scaffold("<think>x</think> answer")
    assert "code_fence" in profiler.detect_scaffold("```json\n{}\n```")
    assert "gpt_oss_channel" in profiler.detect_scaffold("<|channel|>analysis<|message|>hi")
    assert "reasoning" in profiler.detect_scaffold("<reasoning>because</reasoning>")
    assert profiler.detect_scaffold("Business") == []  # plain label → no scaffold


def test_profile_label_has_full_class_list(temp_store):
    train = [_chat(f"text {i}", ["World", "Sports", "Business", "Sci/Tech"][i % 4]) for i in range(40)]
    eval_ = [_chat(f"t{i}", ["World", "Sports"][i % 2]) for i in range(8)]
    _make_split(temp_store, "label-ds", train, eval_)
    p = profiler.profile_dataset("label-ds")
    classes = p["train"]["labels"]["classes"]
    assert set(classes) == {"World", "Sports", "Business", "Sci/Tech"}
    # scaffold rate is 0 for plain labels (so the UI hides it)
    assert p["train"]["scaffold"]["rate"] == 0.0


# --- classification dataset + imbalance ------------------------------------

def test_profile_label_dataset_flags_imbalance(temp_store):
    # 48 'low' + 2 'critical' → minority 4% → imbalanced (< 5% threshold).
    train = [_chat(f"ticket {i}", "urgency: low") for i in range(48)]
    train += [_chat(f"urgent {i}", "urgency: critical") for i in range(2)]
    eval_ = [_chat("t1", "urgency: low"), _chat("t2", "urgency: critical")]
    _make_split(temp_store, "label-ds", train, eval_)

    p = profiler.profile_dataset("label-ds")
    assert p["train"]["dominantTask"] == "label"
    assert p["train"]["labels"]["numClasses"] == 2
    assert p["train"]["labels"]["imbalanced"] is True
    assert p["recommendation"]["rankMetric"] == "label_accuracy"


# --- leakage detection ------------------------------------------------------

def test_profile_detects_train_eval_leakage(temp_store):
    shared = _chat("same question", "same answer here")
    train = [shared, _chat("q2", "a2"), _chat("q3", "a3")]
    eval_ = [shared, _chat("q9", "a9")]
    _make_split(temp_store, "leak-ds", train, eval_)

    p = profiler.profile_dataset("leak-ds")
    assert p["leakage"]["exactOverlapRows"] == 1
    assert any(w["severity"] == "error" and "Leakage" in w["message"] for w in p["warnings"])


# --- truncation risk + malformed rows --------------------------------------

def test_truncation_risk_flagged(temp_store):
    long_ans = "word " * 400  # ~520 tokens at 1.3x
    train = [_chat("q", long_ans) for _ in range(10)]
    eval_ = [_chat("q", long_ans)]
    _make_split(temp_store, "long-ds", train, eval_)

    p = profiler.profile_dataset("long-ds", cutoff_len=256)
    assert p["train"]["truncation"]["estTruncatedRows"] > 0.5
    assert any("truncat" in w["message"].lower() for w in p["warnings"])


def test_malformed_rows_counted(temp_store):
    store = temp_store.get_store()
    wd = store.workdir("runs", "bad-ds")
    (wd / "train.jsonl").write_text('{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\nNOT JSON\n', encoding="utf-8")
    (wd / "eval.jsonl").write_text('{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n', encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text("{}", encoding="utf-8")
    store.commit("runs", "bad-ds")

    p = profiler.profile_dataset("bad-ds")
    assert p["train"]["malformed"] == 1
    assert any(w["severity"] == "error" for w in p["warnings"])


def test_profile_route_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    assert "/api/datasets/{split_id}/profile" in paths


# --- preference (DPO) shape detection -------------------------------------- #

def test_profile_preference_dataset_recommends_dpo(temp_store):
    from app import storage

    pref = [
        {"messages": [{"role": "user", "content": f"q{i}"}],
         "chosen": {"role": "assistant", "content": f"good answer {i}"},
         "rejected": {"role": "assistant", "content": f"bad {i}"}}
        for i in range(8)
    ]
    sid, _ = storage.persist_preference_split(pref, {"name": "pref-ds", "trainRows": 8})
    p = profiler.profile_dataset(sid)
    assert p["shape"] == "preference"
    assert p["objective"]["objective"] == "dpo"
    # preference pairs profiled (NOT the messages train path).
    assert p["preference"]["pairs"] == 8
    assert "train" not in p  # ranking train file isn't profiled as messages
    # the held-out eval is still messages-shaped → eval recommendation present.
    assert p["recommendation"]["rankMetric"]


def test_profile_sft_dataset_recommends_sft(temp_store):
    from app import storage

    sid, _ = storage.persist_split(
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}],
        [{"messages": [{"role": "user", "content": "e"}, {"role": "assistant", "content": "b"}]}],
        {"name": "sft-ds"})
    p = profiler.profile_dataset(sid)
    assert p["shape"] == "sft" and p["objective"]["objective"] == "sft"
    assert "train" in p and "preference" not in p


def test_profile_kto_dataset_recommends_kto(temp_store):
    from app import storage
    rows = [
        {"messages": [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": f"a{i}"}], "kto_tag": i % 2 == 0}
        for i in range(6)
    ]
    sid, _ = storage.persist_kto_split(rows, {"name": "kto-ds", "trainRows": 6})
    p = profiler.profile_dataset(sid)
    assert p["shape"] == "kto" and p["objective"]["objective"] == "kto"
    assert p["kto"]["rows"] == 6 and p["kto"]["desirable"] == 3
    assert "train" not in p and p["recommendation"]["rankMetric"]


def test_warnings_flag_dpo_length_bias_and_identical():
    """DPO: identical pairs + a chosen-longer length bias surface as warnings
    (grounded in the DPO paper's verbosity-bias finding)."""
    from app.profiler import warnings_from
    prof = {"preference": {
        "pairs": 100, "malformed": 0, "identicalPairs": 30,
        "chosenWordLen": {"p50": 60}, "rejectedWordLen": {"p50": 20},
    }}
    msgs = " ".join(w["message"] for w in warnings_from(prof))
    assert "identical" in msgs.lower()
    assert "longer" in msgs.lower()  # verbosity/length-bias flag


def test_warnings_flag_kto_imbalance_and_missing_class():
    from app.profiler import warnings_from, _kto_weight_recommendation
    # imbalanced → warning that cites the CONCRETE recommended weights.
    kto = {"rows": 110, "malformed": 0, "desirable": 100, "undesirable": 10}
    kto.update(_kto_weight_recommendation(100, 10))  # mirror real profiler output
    w = warnings_from({"kto": kto})
    msg = " ".join(x["message"] for x in w)
    assert any("imbalance" in x["message"].lower() for x in w)
    # the warning names the actual knobs + the recommended undesirable up-weight
    assert "kto_rejected_weight" in msg and "kto_chosen_weight" in msg
    # single-class → hard error
    w2 = warnings_from({"kto": {"rows": 50, "malformed": 0, "desirable": 50, "undesirable": 0}})
    assert any(x["severity"] == "error" for x in w2)


def test_kto_weight_recommendation_upweights_minority():
    """The recommendation raises λ on the MINORITY class so λD·nD ≈ λU·nU
    (KTO paper §4.2), capped at the catalog weight bound."""
    from app.profiler import _kto_weight_recommendation
    from app.catalog import _KTO_WEIGHT_MAX

    # 300 desirable : 100 undesirable → up-weight the undesirable (rejected) class ~3×
    r = _kto_weight_recommendation(300, 100)
    assert r["imbalanceRatio"] == 3.0 and r["weightsBalanced"] is False
    assert r["recommendedChosenWeight"] == 1.0
    assert r["recommendedRejectedWeight"] == 3.0
    # symmetric: fewer desirable → up-weight the desirable (chosen) class
    r2 = _kto_weight_recommendation(50, 200)
    assert r2["recommendedChosenWeight"] == 4.0  # 200/50=4
    assert r2["recommendedRejectedWeight"] == 1.0
    # balanced → neutral 1.0/1.0
    r3 = _kto_weight_recommendation(100, 100)
    assert r3 == {"imbalanceRatio": 1.0, "recommendedChosenWeight": 1.0,
                  "recommendedRejectedWeight": 1.0, "weightsBalanced": True}
    # extreme skew is capped at the bound (never recommends an out-of-range weight)
    r4 = _kto_weight_recommendation(1000, 10)  # ratio 100 → capped
    assert r4["recommendedRejectedWeight"] == _KTO_WEIGHT_MAX
    # missing class → neutral (the hard error is raised separately by warnings_from)
    r5 = _kto_weight_recommendation(50, 0)
    assert r5["imbalanceRatio"] is None and r5["recommendedChosenWeight"] == 1.0


def test_preference_profile_exposes_length_bias_ratio(temp_store):
    """_profile_preference returns a concrete lengthBiasRatio (median chosen ÷
    rejected words) for the DPO card."""
    from app import storage
    from app.profiler import profile_dataset
    # chosen ~2× longer than rejected → ratio ≈ 2.0
    train = [{"messages": [{"role": "user", "content": "q"}],
              "chosen": {"role": "assistant", "content": " ".join(["good"] * 20)},
              "rejected": {"role": "assistant", "content": " ".join(["bad"] * 10)}} for _ in range(12)]
    split_id, _ = storage.persist_preference_split(
        train, {"name": "len-bias", "source": "preference"})
    p = profile_dataset(split_id)
    assert p["preference"]["lengthBiasRatio"] == 2.0


def test_clean_dpo_data_no_warnings():
    """Balanced lengths + no identical pairs → no DPO warnings (clean data)."""
    from app.profiler import warnings_from
    prof = {"preference": {
        "pairs": 100, "malformed": 0, "identicalPairs": 0,
        "chosenWordLen": {"p50": 22}, "rejectedWordLen": {"p50": 20},
    }}
    pref_warnings = [w for w in warnings_from(prof) if "DPO" in w["message"]]
    assert pref_warnings == []
