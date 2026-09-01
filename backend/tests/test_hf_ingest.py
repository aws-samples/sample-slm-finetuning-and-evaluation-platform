# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""HF dataset ingestion: column→messages conversion + deterministic sampling.

Pure-logic tests — the HTTP layer (_fetch_rows / _get) is monkeypatched so no
network is touched. Conversion and sampling determinism are the real surface.
"""
from __future__ import annotations

import json

import pytest

from app import hf_ingest
from app.hf_ingest import ColumnMapping, autodetect_mapping, row_to_messages


# --- conversion ------------------------------------------------------------


def test_autodetect_alpaca():
    m = autodetect_mapping(["instruction", "input", "output"])
    assert m is not None
    assert m.user_field == "instruction"
    assert m.target_field == "output"
    assert m.context_field == "input"


def test_autodetect_classification():
    m = autodetect_mapping(["text", "label"])
    assert m is not None
    assert m.user_field == "text"
    assert m.target_field == "label"


def test_autodetect_qa():
    m = autodetect_mapping(["question", "context", "answer"])
    assert m.user_field == "question"
    assert m.target_field == "answer"
    assert m.context_field == "context"


def test_autodetect_summarization():
    # cnn_dailymail (article/highlights) + xsum-style (document/summary).
    m = autodetect_mapping(["article", "highlights", "id"])
    assert m.user_field == "article"
    assert m.target_field == "highlights"
    m2 = autodetect_mapping(["document", "summary"])
    assert (m2.user_field, m2.target_field) == ("document", "summary")


def test_autodetect_none_when_no_match():
    assert autodetect_mapping(["foo", "bar"]) is None


def test_row_to_messages_basic():
    m = ColumnMapping(user_field="text", target_field="label")
    row = row_to_messages({"text": "hello", "label": "greeting"}, m)
    assert row == {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "greeting"},
        ]
    }


def test_row_to_messages_classlabel_int_becomes_name():
    """An integer label indexing a ClassLabel is converted to the class name."""
    m = ColumnMapping(user_field="text", target_field="label")
    row = row_to_messages(
        {"text": "Wall St. rallies", "label": 2},
        m,
        class_labels={"label": ["World", "Sports", "Business", "Sci/Tech"]},
    )
    assert row["messages"][1]["content"] == "Business"


def test_row_to_messages_instruction_becomes_system():
    """A fixed instruction maps to the SYSTEM turn (it's the same for every row);
    the user turn stays the actual input data."""
    m = ColumnMapping(
        user_field="question",
        target_field="answer",
        context_field="context",
        instruction="Answer using the context.",
    )
    row = row_to_messages(
        {"question": "Who?", "answer": "Alice", "context": "Alice is here."}, m
    )
    msgs = row["messages"]
    assert msgs[0] == {"role": "system", "content": "Answer using the context."}
    user = msgs[1]["content"]
    assert msgs[1]["role"] == "user"
    assert "Who?" in user and "Alice is here." in user
    assert "Answer using the context." not in user  # instruction is NOT in user


def test_row_to_messages_instruction_plus_system_field_merge():
    m = ColumnMapping(
        user_field="q", target_field="a", system_field="sys", instruction="Be terse."
    )
    row = row_to_messages({"q": "x", "a": "y", "sys": "You are a bot."}, m)
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][0]["content"] == "Be terse.\n\nYou are a bot."


def test_row_to_messages_system_field():
    m = ColumnMapping(user_field="q", target_field="a", system_field="sys")
    row = row_to_messages({"q": "x", "a": "y", "sys": "be terse"}, m)
    assert row["messages"][0] == {"role": "system", "content": "be terse"}
    assert len(row["messages"]) == 3


def test_row_to_messages_skips_empty():
    m = ColumnMapping(user_field="text", target_field="label")
    assert row_to_messages({"text": "", "label": "x"}, m) is None
    assert row_to_messages({"text": "x", "label": ""}, m) is None


def test_row_to_messages_jsonifies_complex_target():
    m = ColumnMapping(user_field="q", target_field="a")
    row = row_to_messages({"q": "x", "a": {"k": "v"}}, m)
    assert json.loads(row["messages"][1]["content"]) == {"k": "v"}


# --- deterministic sampling offsets ---------------------------------------


def test_ordered_pages_deterministic():
    a = hf_ingest._ordered_pages(10000, 250, seed=42)
    b = hf_ingest._ordered_pages(10000, 250, seed=42)
    assert a == b
    # ceil(250/100) = 3 pages, each a page-aligned offset.
    assert len(a) == 3
    assert all(off % hf_ingest._PAGE == 0 for off in a)
    assert all(0 <= off < 10000 for off in a)


def test_ordered_pages_seed_changes_selection():
    a = hf_ingest._ordered_pages(100000, 300, seed=1)
    b = hf_ingest._ordered_pages(100000, 300, seed=2)
    assert a != b


def test_ordered_pages_caps_at_total():
    # want more pages than exist -> every page, in order.
    pages = hf_ingest._ordered_pages(250, 9999, seed=42)
    assert pages == [0, 100, 200]


def test_ordered_pages_bounds_http_calls():
    # 500 rows from a 1M-row dataset should need only 5 page fetches, not 10k.
    assert len(hf_ingest._ordered_pages(1_000_000, 500, seed=42)) == 5


# --- sample_to_jsonl (HTTP mocked) ----------------------------------------


@pytest.fixture
def fake_rows(monkeypatch):
    """Mock _fetch_rows to serve a synthetic ag_news-like classification set."""
    names = ["World", "Sports", "Business", "Sci/Tech"]
    features = [
        {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "label", "type": {"_type": "ClassLabel", "names": names}},
    ]
    total = 50

    def _fake(dataset, config, split, offset, length, token):
        rows = []
        for idx in range(offset, min(offset + length, total)):
            rows.append({"row_idx": idx, "row": {"text": f"headline {idx}", "label": idx % 4}})
        return {"features": features, "rows": rows, "num_rows_total": total}

    monkeypatch.setattr(hf_ingest, "_fetch_rows", _fake)
    # License lookup hits a different (Hub) endpoint — stub it so preview
    # tests stay hermetic (advisory; defaults to unknown anyway).
    monkeypatch.setattr(hf_ingest, "fetch_license",
                        lambda ds, token=None: {"license": None, "bucket": "unknown", "gated": None})
    return names


def test_sample_to_jsonl_converts_and_caps(fake_rows):
    m = ColumnMapping(user_field="text", target_field="label")
    text, stats = hf_ingest.sample_to_jsonl(
        "fake/ds", m, config="default", split="train", max_rows=10, seed=42
    )
    lines = [json.loads(x) for x in text.splitlines()]
    assert len(lines) == 10
    assert stats["converted"] == 10
    assert stats["numRowsTotal"] == 50
    # labels were mapped from ints to class names
    answers = {l["messages"][1]["content"] for l in lines}
    assert answers <= set(fake_rows)


def test_sample_to_jsonl_deterministic(fake_rows):
    m = ColumnMapping(user_field="text", target_field="label")
    t1, _ = hf_ingest.sample_to_jsonl("fake/ds", m, config="default", split="train", max_rows=10, seed=7)
    t2, _ = hf_ingest.sample_to_jsonl("fake/ds", m, config="default", split="train", max_rows=10, seed=7)
    assert t1 == t2


def test_sample_to_jsonl_all_rows_when_cap_exceeds_total(fake_rows):
    m = ColumnMapping(user_field="text", target_field="label")
    text, stats = hf_ingest.sample_to_jsonl(
        "fake/ds", m, config="default", split="train", max_rows=999, seed=42
    )
    assert stats["converted"] == 50  # the whole synthetic set


# --- preference (DPO) import ------------------------------------------------ #

from app.hf_ingest import autodetect_preference_mapping, preference_row_from_hf


def test_autodetect_preference_prompt_chosen_rejected():
    pm = autodetect_preference_mapping(["prompt", "chosen", "rejected"])
    assert pm and pm.chosen_field == "chosen" and pm.rejected_field == "rejected"
    assert pm.prompt_field == "prompt"


def test_autodetect_preference_no_prompt_column():
    pm = autodetect_preference_mapping(["chosen", "rejected"])
    assert pm and pm.prompt_field is None


def test_autodetect_preference_none_when_missing_pair():
    assert autodetect_preference_mapping(["instruction", "output"]) is None
    assert autodetect_preference_mapping(["chosen"]) is None  # needs both


def test_preference_row_string_layout():
    pm = autodetect_preference_mapping(["prompt", "chosen", "rejected"])
    row = preference_row_from_hf({"prompt": "2+2?", "chosen": "4", "rejected": "5"}, pm)
    assert row["messages"] == [{"role": "user", "content": "2+2?"}]
    assert row["chosen"] == {"role": "assistant", "content": "4"}
    assert row["rejected"]["content"] == "5"


def test_preference_row_chat_list_shared_prefix():
    pm = autodetect_preference_mapping(["chosen", "rejected"])
    chosen = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    rejected = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "no"}]
    row = preference_row_from_hf({"chosen": chosen, "rejected": rejected}, pm)
    assert row["messages"] == [{"role": "user", "content": "hi"}]
    assert row["chosen"]["content"] == "hello" and row["rejected"]["content"] == "no"


def test_preference_row_sharegpt_native_keys():
    pm = autodetect_preference_mapping(["chosen", "rejected"])
    c = [{"from": "human", "value": "q"}, {"from": "gpt", "value": "good"}]
    r = [{"from": "human", "value": "q"}, {"from": "gpt", "value": "bad"}]
    row = preference_row_from_hf({"chosen": c, "rejected": r}, pm)
    assert row["messages"] == [{"role": "user", "content": "q"}]
    assert row["chosen"]["content"] == "good"


def test_preference_row_identical_responses_skipped():
    pm = autodetect_preference_mapping(["prompt", "chosen", "rejected"])
    assert preference_row_from_hf({"prompt": "q", "chosen": "x", "rejected": "x"}, pm) is None


@pytest.fixture
def fake_pref_rows(monkeypatch):
    """Mock _fetch_rows to serve a synthetic prompt/chosen/rejected set."""
    features = [
        {"name": "prompt", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "chosen", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "rejected", "type": {"dtype": "string", "_type": "Value"}},
    ]
    total = 30

    def _fake(dataset, config, split, offset, length, token):
        rows = []
        for idx in range(offset, min(offset + length, total)):
            rows.append({"row_idx": idx, "row": {
                "prompt": f"q{idx}", "chosen": f"good{idx}", "rejected": f"bad{idx}"}})
        return {"features": features, "rows": rows, "num_rows_total": total}

    monkeypatch.setattr(hf_ingest, "_fetch_rows", _fake)
    # License lookup hits a different (Hub) endpoint — stub it so preview
    # tests stay hermetic (advisory; defaults to unknown anyway).
    monkeypatch.setattr(hf_ingest, "fetch_license",
                        lambda ds, token=None: {"license": None, "bucket": "unknown", "gated": None})
    return total


def test_sample_preference_to_jsonl(fake_pref_rows):
    pm = autodetect_preference_mapping(["prompt", "chosen", "rejected"])
    rows, stats = hf_ingest.sample_preference_to_jsonl(
        "fake/ds", pm, config="default", split="train", max_rows=10, seed=42)
    assert len(rows) == 10 and stats["converted"] == 10
    # canonical ranking shape
    assert all(set(r.keys()) == {"messages", "chosen", "rejected"} for r in rows)
    assert rows[0]["chosen"]["role"] == "assistant"


def test_preview_detects_preference_shape(fake_pref_rows, monkeypatch):
    # list_splits is a separate HTTP call — stub it to one split.
    monkeypatch.setattr(hf_ingest, "list_splits",
                        lambda ds, token=None: [{"config": "default", "split": "train"}])
    prev = hf_ingest.preview("fake/ds").to_dict()
    assert prev["detectedShape"] == "preference"
    assert prev["suggestedPreference"]["chosenField"] == "chosen"
    assert len(prev["preferencePreview"]) > 0


def test_preference_row_transcript_string_layout():
    """Anthropic/hh-rlhf ships chosen/rejected as ONE transcript string
    ('\\n\\nHuman: …\\n\\nAssistant: …'). The converter must split it and use the
    shared prefix as the prompt — caught only by deploy-testing against the real
    dataset (the columns are named chosen/rejected but hold strings, not lists)."""
    pm = autodetect_preference_mapping(["chosen", "rejected"])
    chosen = "\n\nHuman: What are cuss words?\n\nAssistant: A list: darn, heck."
    rejected = "\n\nHuman: What are cuss words?\n\nAssistant: I cannot help with that."
    row = preference_row_from_hf({"chosen": chosen, "rejected": rejected}, pm)
    assert row is not None
    assert row["messages"][-1] == {"role": "user", "content": "What are cuss words?"}
    assert "darn" in row["chosen"]["content"]
    assert "cannot help" in row["rejected"]["content"]


def test_preference_transcript_multi_turn_prompt():
    pm = autodetect_preference_mapping(["chosen", "rejected"])
    c = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: bye\n\nAssistant: goodbye friend"
    r = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: bye\n\nAssistant: see ya"
    row = preference_row_from_hf({"chosen": c, "rejected": r}, pm)
    # shared prefix = [user hi, assistant hello, user bye]; responses diverge after.
    assert len(row["messages"]) == 3
    assert row["chosen"]["content"] == "goodbye friend"
    assert row["rejected"]["content"] == "see ya"


# --- regression: review findings (Layout-2 divergence, transcript leading turn) ---

def test_preference_multiturn_divergence_not_at_final_turn():
    """HIGH regression: when chosen/rejected diverge at an EARLIER assistant turn,
    the response must be that first diverging turn (not the final turn), and the
    prompt is truncated to the shared prefix — else the pair is misaligned."""
    pm = autodetect_preference_mapping(["chosen", "rejected"])
    c = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A1good"},
         {"role": "user", "content": "x"}, {"role": "assistant", "content": "tail"}]
    r = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "A1bad"},
         {"role": "user", "content": "x"}, {"role": "assistant", "content": "tail"}]
    row = preference_row_from_hf({"chosen": c, "rejected": r}, pm)
    assert row["messages"] == [{"role": "user", "content": "q"}]
    assert row["chosen"]["content"] == "A1good" and row["rejected"]["content"] == "A1bad"


def test_preference_divergence_at_user_turn_skipped():
    """If the first divergence isn't an assistant/assistant pair, no clean
    preference pair can be formed → skip (return None), never crash/corrupt."""
    pm = autodetect_preference_mapping(["chosen", "rejected"])
    c = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "same"},
         {"role": "user", "content": "DIFF-a"}]
    r = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "same"},
         {"role": "user", "content": "DIFF-b"}]
    assert preference_row_from_hf({"chosen": c, "rejected": r}, pm) is None


def test_parse_transcript_without_leading_blank_keeps_first_turn():
    """Regression: a transcript that omits the leading blank line must NOT drop
    its first turn."""
    assert hf_ingest._parse_transcript("Human: hi\n\nAssistant: hello") == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


# --- KTO HF import ---------------------------------------------------------- #

from app.hf_ingest import autodetect_kto_mapping, kto_row_from_hf


def test_autodetect_kto_mapping():
    m = autodetect_kto_mapping(["prompt", "completion", "label"])
    assert m and m.completion_field == "completion" and m.label_field == "label"
    assert m.prompt_field == "prompt"
    # needs a completion AND a label column
    assert autodetect_kto_mapping(["prompt", "completion"]) is None
    assert autodetect_kto_mapping(["instruction", "output"]) is None


def test_kto_row_from_hf_label_variants():
    m = autodetect_kto_mapping(["prompt", "completion", "label"])
    good = kto_row_from_hf({"prompt": "q", "completion": "a", "label": "good"}, m)
    assert good["kto_tag"] is True
    assert good["messages"] == [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": "a"}]
    bad = kto_row_from_hf({"prompt": "q", "completion": "a", "label": "bad"}, m)
    assert bad["kto_tag"] is False
    # unrecognized label → skip
    assert kto_row_from_hf({"prompt": "q", "completion": "a", "label": "maybe"}, m) is None
    # empty completion → skip
    assert kto_row_from_hf({"prompt": "q", "completion": "", "label": "good"}, m) is None


def test_kto_row_numeric_minus_one():
    m = autodetect_kto_mapping(["question", "response", "reward"])
    r = kto_row_from_hf({"question": "q", "response": "x", "reward": -1}, m)
    assert r["kto_tag"] is False


@pytest.fixture
def fake_kto_rows(monkeypatch):
    features = [
        {"name": "prompt", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "completion", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "label", "type": {"dtype": "string", "_type": "Value"}},
    ]
    total = 20

    def _fake(dataset, config, split, offset, length, token):
        rows = []
        for idx in range(offset, min(offset + length, total)):
            rows.append({"row_idx": idx, "row": {
                "prompt": f"q{idx}", "completion": f"ans{idx}",
                "label": "good" if idx % 2 == 0 else "bad"}})
        return {"features": features, "rows": rows, "num_rows_total": total}

    monkeypatch.setattr(hf_ingest, "_fetch_rows", _fake)
    # License lookup hits a different (Hub) endpoint — stub it so preview
    # tests stay hermetic (advisory; defaults to unknown anyway).
    monkeypatch.setattr(hf_ingest, "fetch_license",
                        lambda ds, token=None: {"license": None, "bucket": "unknown", "gated": None})
    return total


def test_sample_kto_to_jsonl(fake_kto_rows):
    m = autodetect_kto_mapping(["prompt", "completion", "label"])
    rows, stats = hf_ingest.sample_kto_to_jsonl(
        "fake/ds", m, config="default", split="train", max_rows=10, seed=42)
    assert len(rows) == 10 and stats["converted"] == 10
    assert all(set(r.keys()) == {"messages", "kto_tag"} for r in rows)
    assert any(r["kto_tag"] for r in rows) and any(not r["kto_tag"] for r in rows)


def test_preview_detects_kto_shape(fake_kto_rows, monkeypatch):
    monkeypatch.setattr(hf_ingest, "list_splits",
                        lambda ds, token=None: [{"config": "default", "split": "train"}])
    prev = hf_ingest.preview("fake/ds").to_dict()
    assert prev["detectedShape"] == "kto"
    assert prev["suggestedKto"]["labelField"] == "label"
    assert len(prev["ktoPreview"]) > 0


# --- RLVR import (prompt + verifiable ground_truth; gsm8k is the canonical fit) ---

def test_autodetect_rlvr_mapping():
    from app.hf_ingest import autodetect_rlvr_mapping
    # gsm8k columns: question + answer
    m = autodetect_rlvr_mapping(["question", "answer"])
    assert m and m.prompt_field == "question" and m.ground_truth_field == "answer"
    # explicit ground_truth column
    m2 = autodetect_rlvr_mapping(["prompt", "ground_truth"])
    assert m2 and m2.ground_truth_field == "ground_truth"
    # needs BOTH a prompt and a target, and they must differ
    assert autodetect_rlvr_mapping(["question"]) is None
    assert autodetect_rlvr_mapping(["foo", "bar"]) is None


def test_rlvr_row_from_hf_gsm8k():
    from app.hf_ingest import autodetect_rlvr_mapping, rlvr_row_from_hf
    m = autodetect_rlvr_mapping(["question", "answer"])
    row = rlvr_row_from_hf(
        {"question": "Natalia sold 48 clips, half as many in May. Total?",
         "answer": "48 + 24 = 72\n#### 72"}, m)
    # prompt-only messages (NO assistant turn) + explicit ground_truth.
    assert row["messages"] == [
        {"role": "user", "content": "Natalia sold 48 clips, half as many in May. Total?"}]
    assert row["ground_truth"] == "48 + 24 = 72\n#### 72"
    # missing answer → skip
    assert rlvr_row_from_hf({"question": "q", "answer": ""}, m) is None


@pytest.fixture
def fake_rlvr_rows(monkeypatch):
    features = [
        {"name": "question", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "answer", "type": {"dtype": "string", "_type": "Value"}},
    ]
    total = 20

    def _fake(dataset, config, split, offset, length, token):
        rows = []
        for idx in range(offset, min(offset + length, total)):
            rows.append({"row_idx": idx, "row": {
                "question": f"q{idx}", "answer": f"#### {idx}"}})
        return {"features": features, "rows": rows, "num_rows_total": total}

    monkeypatch.setattr(hf_ingest, "_fetch_rows", _fake)
    # License lookup hits a different (Hub) endpoint — stub it so preview
    # tests stay hermetic (advisory; defaults to unknown anyway).
    monkeypatch.setattr(hf_ingest, "fetch_license",
                        lambda ds, token=None: {"license": None, "bucket": "unknown", "gated": None})
    return total


def test_sample_rlvr_to_jsonl(fake_rlvr_rows):
    from app.hf_ingest import autodetect_rlvr_mapping
    m = autodetect_rlvr_mapping(["question", "answer"])
    rows, stats = hf_ingest.sample_rlvr_to_jsonl(
        "fake/ds", m, config="default", split="train", max_rows=10, seed=42)
    assert len(rows) == 10 and stats["converted"] == 10
    assert all(set(r.keys()) == {"messages", "ground_truth"} for r in rows)
    # prompt has no trailing assistant turn
    assert all(r["messages"][-1]["role"] != "assistant" for r in rows)


def test_preview_suggests_rlvr_without_overriding_shape(fake_rlvr_rows, monkeypatch):
    """A Q&A dataset (gsm8k) stays detectedShape=sft (Q&A is usually SFT), but the
    preview offers a suggestedRlvr mapping so the user can opt into RLVR."""
    monkeypatch.setattr(hf_ingest, "list_splits",
                        lambda ds, token=None: [{"config": "default", "split": "train"}])
    prev = hf_ingest.preview("fake/ds").to_dict()
    assert prev["detectedShape"] == "sft"  # NOT auto-flipped to rlvr
    assert prev["suggestedRlvr"]["promptField"] == "question"
    assert prev["suggestedRlvr"]["groundTruthField"] == "answer"
    assert len(prev["rlvrPreview"]) > 0


# --- RLAIF import (prompt-only; the AI judge scores the response) ----------

def test_autodetect_rlaif_mapping():
    from app.hf_ingest import autodetect_rlaif_mapping
    # only a prompt column is needed (no answer/ground_truth)
    m = autodetect_rlaif_mapping(["prompt", "extra"])
    assert m and m.prompt_field == "prompt"
    # other prompt-like names work
    assert autodetect_rlaif_mapping(["question"]).prompt_field == "question"
    assert autodetect_rlaif_mapping(["instruction", "x"]).prompt_field == "instruction"
    # picks up a per-row system column when present
    m2 = autodetect_rlaif_mapping(["prompt", "system"])
    assert m2.system_field == "system"
    # no prompt-like column → None
    assert autodetect_rlaif_mapping(["foo", "bar"]) is None


def test_rlaif_row_from_hf_prompt_only():
    from app.hf_ingest import RlaifMapping, autodetect_rlaif_mapping, rlaif_row_from_hf
    m = autodetect_rlaif_mapping(["prompt"])
    # string prompt → prompt-only messages, NO assistant turn, NO ground_truth
    row = rlaif_row_from_hf({"prompt": "Write a friendly note about X."}, m)
    assert row == {"messages": [{"role": "user", "content": "Write a friendly note about X."}]}
    # a chat-list prompt that already has an answer → assistant turn dropped
    mc = RlaifMapping(prompt_field="conv")
    row2 = rlaif_row_from_hf(
        {"conv": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}, mc)
    assert row2 == {"messages": [{"role": "user", "content": "hi"}]}
    # empty prompt → skip
    assert rlaif_row_from_hf({"prompt": "   "}, m) is None
    # a fixed instruction becomes a system turn
    mi = RlaifMapping(prompt_field="prompt", instruction="Be concise.")
    row3 = rlaif_row_from_hf({"prompt": "Note about Y"}, mi)
    assert row3["messages"][0] == {"role": "system", "content": "Be concise."}


@pytest.fixture
def fake_rlaif_rows(monkeypatch):
    features = [{"name": "prompt", "type": {"dtype": "string", "_type": "Value"}}]
    total = 20

    def _fake(dataset, config, split, offset, length, token):
        rows = []
        for idx in range(offset, min(offset + length, total)):
            rows.append({"row_idx": idx, "row": {"prompt": f"Write a note about topic {idx}."}})
        return {"features": features, "rows": rows, "num_rows_total": total}

    monkeypatch.setattr(hf_ingest, "_fetch_rows", _fake)
    monkeypatch.setattr(hf_ingest, "fetch_license",
                        lambda ds, token=None: {"license": None, "bucket": "unknown", "gated": None})
    return total


def test_sample_rlaif_to_jsonl(fake_rlaif_rows):
    from app.hf_ingest import autodetect_rlaif_mapping
    m = autodetect_rlaif_mapping(["prompt"])
    rows, stats = hf_ingest.sample_rlaif_to_jsonl(
        "fake/ds", m, config="default", split="train", max_rows=10, seed=42)
    assert len(rows) == 10 and stats["converted"] == 10
    # prompt-only: messages only, NO ground_truth, NO trailing assistant turn
    assert all(set(r.keys()) == {"messages"} for r in rows)
    assert all(r["messages"][-1]["role"] != "assistant" for r in rows)


def test_preview_suggests_rlaif(fake_rlaif_rows, monkeypatch):
    """A prompt-only dataset offers a suggestedRlaif mapping + preview (and, since
    it has no answer column, does NOT suggest RLVR)."""
    monkeypatch.setattr(hf_ingest, "list_splits",
                        lambda ds, token=None: [{"config": "default", "split": "train"}])
    prev = hf_ingest.preview("fake/ds").to_dict()
    assert prev["suggestedRlaif"]["promptField"] == "prompt"
    assert len(prev["rlaifPreview"]) > 0
    assert prev["suggestedRlvr"] is None  # no ground-truth column → no RLVR suggestion


# --- license advisory (compliance aid) ------------------------------------


def test_classify_license_buckets():
    cl = hf_ingest.classify_license
    # permissive
    for s in ("mit", "apache-2.0", "BSD-3-Clause", "cc0-1.0", "cc-by-4.0"):
        assert cl(s) == "permissive", s
    # restrictive: non-commercial, copyleft, known-restricted families
    for s in ("cc-by-nc-4.0", "gpl-3.0", "agpl-3.0", "lgpl-2.1", "llama3.1", "other"):
        assert cl(s) == "restrictive", s
    # unknown: absent / literally unknown / unrecognized
    for s in (None, "", "unknown", "some-bespoke-license-v9"):
        assert cl(s) == "unknown", s


def test_classify_license_heuristics_for_unlisted_slugs():
    # An -nc variant not in the explicit set is still restrictive.
    assert hf_ingest.classify_license("cc-by-nc-nd-3.0") == "restrictive"
    # A cc-by (non-NC) variant not explicitly listed is permissive.
    assert hf_ingest.classify_license("cc-by-1.0") == "permissive"


def test_fetch_license_parses_cardData(monkeypatch):
    class _R:
        status_code = 200
        @staticmethod
        def json():
            return {"cardData": {"license": "mit"}, "tags": ["license:mit"], "gated": False}
    monkeypatch.setattr(hf_ingest.requests, "get", lambda *a, **k: _R())
    out = hf_ingest.fetch_license("openai/gsm8k")
    assert out == {"license": "mit", "bucket": "permissive", "gated": False}


def test_fetch_license_list_license_takes_first(monkeypatch):
    class _R:
        status_code = 200
        @staticmethod
        def json():
            return {"cardData": {"license": ["cc-by-nc-4.0", "mit"]}, "tags": [], "gated": False}
    monkeypatch.setattr(hf_ingest.requests, "get", lambda *a, **k: _R())
    out = hf_ingest.fetch_license("x/y")
    assert out["license"] == "cc-by-nc-4.0" and out["bucket"] == "restrictive"


def test_fetch_license_falls_back_to_tag(monkeypatch):
    class _R:
        status_code = 200
        @staticmethod
        def json():
            return {"cardData": {}, "tags": ["task:x", "license:apache-2.0"], "gated": "manual"}
    monkeypatch.setattr(hf_ingest.requests, "get", lambda *a, **k: _R())
    out = hf_ingest.fetch_license("x/y")
    assert out["license"] == "apache-2.0" and out["bucket"] == "permissive" and out["gated"] == "manual"


def test_fetch_license_never_raises_on_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(hf_ingest.requests, "get", _boom)
    out = hf_ingest.fetch_license("x/y")
    assert out == {"license": None, "bucket": "unknown", "gated": None}


def test_fetch_license_non_200_is_unknown(monkeypatch):
    class _R:
        status_code = 404
        @staticmethod
        def json():
            return {}
    monkeypatch.setattr(hf_ingest.requests, "get", lambda *a, **k: _R())
    assert hf_ingest.fetch_license("x/y")["bucket"] == "unknown"
