# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit tests for the eval metric harness (answer extraction + task-aware metrics).

Run from the container/ dir:  python -m pytest test_eval.py
These guard the logic that runs inside the GPU image, independent of any model —
in particular the <think>-prefix extraction that once caused a false 0% JSON score.
"""
import eval as E


# --- answer extraction ----------------------------------------------------- #

def test_strip_think_prefix():
    raw = '<think>\n\nlet me reason\n</think>\n\n{"narrative": "hi"}'
    assert E.extract_answer(raw) == '{"narrative": "hi"}'


def test_strip_code_fence():
    raw = 'Here you go:\n```json\n{"a": 1}\n```'
    assert E.extract_answer(raw).strip() == '{"a": 1}'


def test_unclosed_think_is_dropped():
    raw = '<think> reasoning that got truncated'
    assert E.extract_answer(raw) == ""


def test_plain_answer_unchanged():
    assert E.extract_answer("urgency: high") == "urgency: high"


def test_strip_thinking_and_reasoning_variants():
    assert E.extract_answer("<thinking>hmm</thinking>Business") == "Business"
    assert E.extract_answer("<reasoning>because X</reasoning>\nSports") == "Sports"
    assert E.extract_answer("<scratchpad>1+1</scratchpad> 2") == "2"


def test_gpt_oss_harmony_final_channel():
    # gpt-oss puts reasoning in `analysis` and the answer in `final`.
    raw = (
        "<|channel|>analysis<|message|>The topic is markets.<|end|>"
        "<|channel|>final<|message|>Business<|return|>"
    )
    assert E.extract_answer(raw) == "Business"


def test_gpt_oss_harmony_final_with_json():
    raw = '<|channel|>analysis<|message|>think…<|channel|>final<|message|>{"a": 1}<|end|>'
    assert E.extract_answer(raw) == '{"a": 1}'


def test_answer_tag_unwrapped():
    assert E.extract_answer("preamble <answer>42</answer> trailing") == "42"


# --- task detection -------------------------------------------------------- #

def test_detect_task():
    assert E.detect_task('{"narrative": "x"}') == "json"
    assert E.detect_task("urgency: high") == "label"
    assert E.detect_task("42") == "numeric"
    assert E.detect_task("$1,250.50") == "numeric"
    assert E.detect_task("The user is browsing hotels in Lisbon and Porto.") == "text"


# --- the regression that started it all: <think> + JSON --------------------- #

def test_think_wrapped_json_scores_valid():
    """A reasoning model wrapping valid JSON in <think></think> must NOT score 0%
    JSON (the exact bug found on the travel-concierge run)."""
    gold = '{"narrative": "The user browses Lisbon hotels."}'
    pred = '<think>\n\n</think>\n\n{"narrative": "The user is looking at hotels in Lisbon."}'
    m = E.compute_metrics([pred], [gold])
    assert m["json_rows"] == 1
    assert m["json_valid"] == 1.0          # was 0.0 before extraction
    assert m["json_structural"] == 1.0     # same {narrative: str} shape
    assert m["json_key_recall"] == 1.0     # has the 'narrative' key
    assert m["scaffold_rate"] == 1.0       # flagged the <think> wrapping
    assert m["task_mix"] == {"json": 1}


def test_invalid_json_pred_scored_zero_valid():
    gold = '{"a": 1}'
    pred = 'not json at all'
    m = E.compute_metrics([pred], [gold])
    assert m["json_valid"] == 0.0
    assert m["json_key_recall"] == 0.0


# --- numeric + label task metrics ------------------------------------------ #

def test_numeric_match_tolerant_of_formatting():
    m = E.compute_metrics(["The answer is $1,200"], ["1200"])
    # gold '1200' is numeric; pred contains 1,200 → extracted answer parses equal.
    # (extraction keeps the prose, but _as_number requires a bare number, so this
    #  exercises the 'pred has prose' path → numeric_match 0; bare pred below = 1.)
    assert m["numeric_rows"] == 1


def test_numeric_match_bare():
    m = E.compute_metrics(["1200"], ["1,200"])
    assert m["numeric_match"] == 1.0


def test_label_accuracy():
    golds = ["urgency: high", "urgency: low", "urgency: high"]
    preds = ["urgency: high", "urgency: high", "urgency: high"]
    m = E.compute_metrics(preds, golds)
    assert m["label_rows"] == 3
    assert m["label_accuracy"] == round(2 / 3, 4)


def test_mixed_task_set_reports_each():
    golds = ['{"a":1}', "urgency: high", "42", "A full sentence answer here."]
    preds = ['{"a":1}', "urgency: high", "42", "A full sentence answer here."]
    m = E.compute_metrics(preds, golds)
    assert m["task_mix"] == {"json": 1, "label": 1, "numeric": 1, "text": 1}
    assert m["json_valid"] == 1.0
    assert m["label_accuracy"] == 1.0
    assert m["numeric_match"] == 1.0


# --- load_eval_rows: prompt-only rows are skipped, not crashed ------------- #

def test_load_eval_rows_skips_prompt_only(tmp_path):
    """A PROMPT-ONLY row (no assistant turn — e.g. an RLAIF held-out prompt that
    leaked into a gold eval) must be SKIPPED, not crash `max()` over an empty
    sequence. Rows with a gold answer are still parsed normally."""
    import json
    p = tmp_path / "eval.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Q1"},
                                  {"role": "assistant", "content": "A1"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "prompt only, no answer"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "Q2"},
                                   {"role": "assistant", "content": "A2"}]}) + "\n",
        encoding="utf-8",
    )
    prompts, golds, rejecteds = E.load_eval_rows(p)
    assert golds == ["A1", "A2"]               # prompt-only row dropped, no crash
    assert [m[-1]["content"] for m in prompts] == ["Q1", "Q2"]
    assert rejecteds == [None, None]           # no rejected_ref on plain SFT rows


def test_chosen_win_rate_preference(tmp_path):
    """A DPO eval row carries chosen_ref + rejected_ref; chosen_win_rate = fraction
    of rows where the prediction is closer (token-F1) to chosen than to rejected.
    None for rows/datasets with no rejected ref (SFT/KTO/RLVR)."""
    import json
    p = tmp_path / "eval.jsonl"
    # Row 1: pred will match chosen exactly → win. Row 2: pred matches rejected → loss.
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Q1"},
                                  {"role": "assistant", "content": "the polite answer"}],
                    "rejected_ref": "rude answer"}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "Q2"},
                                   {"role": "assistant", "content": "helpful detailed reply"}],
                      "rejected_ref": "vague unhelpful reply"}) + "\n",
        encoding="utf-8",
    )
    prompts, golds, rejecteds = E.load_eval_rows(p)
    assert rejecteds == ["rude answer", "vague unhelpful reply"]
    # pred[0] == chosen (win); pred[1] == rejected (loss) → 1/2 = 0.5
    preds = ["the polite answer", "vague unhelpful reply"]
    m = E.compute_metrics(preds, golds, rejecteds)
    assert m["chosen_win_rate_rows"] == 2
    assert m["chosen_win_rate"] == 0.5
    # No rejected refs → metric is None (SFT/KTO/RLVR behavior)
    m2 = E.compute_metrics(preds, golds)
    assert m2["chosen_win_rate"] is None
    assert m2["chosen_win_rate_rows"] == 0
