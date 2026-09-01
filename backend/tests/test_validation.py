# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Chat-template JSONL validation."""
from app.validation import validate_jsonl


def test_valid_rows_pass():
    text = '\n'.join([
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}',
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}',
    ])
    r = validate_jsonl(text)
    assert r.valid_rows == 2
    assert r.invalid_rows == 0
    assert r.errors == []


def test_missing_assistant_turn_is_invalid():
    text = '{"messages":[{"role":"user","content":"hi"}]}'
    r = validate_jsonl(text)
    assert r.invalid_rows == 1
    assert r.valid_rows == 0
    assert r.errors and r.errors[0].line == 1


def test_bad_json_line_reported():
    text = '{"messages": [oops'
    r = validate_jsonl(text)
    assert r.invalid_rows == 1
    assert "json" in r.errors[0].message.lower() or "parse" in r.errors[0].message.lower() \
        or "expecting" in r.errors[0].message.lower()


def test_unknown_role_invalid():
    text = '{"messages":[{"role":"wizard","content":"x"},{"role":"assistant","content":"a"}]}'
    r = validate_jsonl(text)
    assert r.invalid_rows == 1


def test_blank_lines_ignored():
    text = '\n\n{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n\n'
    r = validate_jsonl(text)
    assert r.valid_rows == 1
    assert r.invalid_rows == 0


def test_empty_content_invalid():
    text = '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"   "}]}'
    r = validate_jsonl(text)
    assert r.invalid_rows == 1


# --- preference (DPO) parsing ----------------------------------------------- #

def test_parse_preference_messages_shape():
    from app.validation import parse_preference_jsonl

    text = (
        '{"messages":[{"role":"user","content":"q"}],'
        '"chosen":{"role":"assistant","content":"good"},'
        '"rejected":{"role":"assistant","content":"bad"}}'
    )
    rows, report = parse_preference_jsonl(text)
    assert report.valid and len(rows) == 1
    assert rows[0]["chosen"]["content"] == "good"
    assert rows[0]["rejected"]["content"] == "bad"


def test_parse_preference_prompt_and_string_responses():
    from app.validation import parse_preference_jsonl

    # bare `prompt` string + string chosen/rejected → normalized to turns.
    text = '{"prompt":"hi","chosen":"hello","rejected":"go away"}'
    rows, report = parse_preference_jsonl(text)
    assert report.valid
    assert rows[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert rows[0]["chosen"] == {"role": "assistant", "content": "hello"}


def test_parse_preference_missing_rejected_is_invalid():
    from app.validation import parse_preference_jsonl

    text = '{"prompt":"hi","chosen":"hello"}'
    rows, report = parse_preference_jsonl(text)
    assert not report.valid and report.invalid_rows == 1 and rows == []


# --- KTO (binary feedback) parsing ----------------------------------------- #

def test_parse_kto_messages_and_prompt_completion():
    from app.validation import parse_kto_jsonl
    import json
    text = "\n".join([
        json.dumps({"messages": [{"role": "user", "content": "q"},
                                 {"role": "assistant", "content": "good"}], "kto_tag": True}),
        json.dumps({"prompt": "q2", "completion": "bad", "label": "bad"}),
        json.dumps({"prompt": "q3", "completion": "ok", "label": 1}),
    ])
    rows, rep = parse_kto_jsonl(text)
    assert rep.valid and len(rows) == 3
    assert rows[0]["kto_tag"] is True and rows[1]["kto_tag"] is False and rows[2]["kto_tag"] is True
    assert rows[1]["messages"][-1] == {"role": "assistant", "content": "bad"}


def test_parse_kto_rejects_no_assistant_and_no_label():
    from app.validation import parse_kto_jsonl
    import json
    # messages not ending in assistant.
    r1, rep1 = parse_kto_jsonl(json.dumps({"messages": [{"role": "user", "content": "q"}], "kto_tag": True}))
    assert not rep1.valid
    # no usable label.
    r2, rep2 = parse_kto_jsonl(json.dumps({"prompt": "q", "completion": "a"}))
    assert not rep2.valid


def test_preference_prompt_ending_in_assistant_rejected():
    """Regression: the prompt `messages` must not already contain the response
    (a trailing assistant turn) — that would train DPO on a double assistant turn."""
    from app.validation import parse_preference_jsonl
    import json
    text = json.dumps({"messages": [{"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "a"}],
                       "chosen": "c", "rejected": "r"})
    rows, rep = parse_preference_jsonl(text)
    assert not rep.valid and rows == []


def test_kto_label_minus_one_is_bad():
    """Regression: the -1/1 (bad/good) numeric convention must be recognized."""
    from app.validation import _coerce_kto_tag
    assert _coerce_kto_tag(-1) is False
    assert _coerce_kto_tag(1) is True
    assert _coerce_kto_tag(2) is None


def test_parse_rlvr_messages_and_prompt():
    from app.validation import parse_rlvr_jsonl
    import json
    text = "\n".join([
        json.dumps({"messages": [{"role": "user", "content": "2+2?"}], "ground_truth": "4"}),
        json.dumps({"prompt": "3+5?", "ground_truth": "8"}),       # bare prompt string
        json.dumps({"prompt": "10-1?", "answer": "9"}),            # `answer` alias
    ])
    rows, rep = parse_rlvr_jsonl(text)
    assert rep.valid and len(rows) == 3
    assert rows[0]["ground_truth"] == "4"
    assert rows[1]["messages"] == [{"role": "user", "content": "3+5?"}]
    assert rows[2]["ground_truth"] == "9"


def test_parse_rlvr_rejects_missing_ground_truth_and_answer_in_prompt():
    from app.validation import parse_rlvr_jsonl
    import json
    # missing ground_truth — the verifiable target is required (NOT derived).
    r1, rep1 = parse_rlvr_jsonl(json.dumps({"messages": [{"role": "user", "content": "q"}]}))
    assert not rep1.valid and r1 == []
    # blank ground_truth.
    r2, rep2 = parse_rlvr_jsonl(json.dumps({"prompt": "q", "ground_truth": "  "}))
    assert not rep2.valid
    # prompt must not already contain the answer (trailing assistant turn).
    r3, rep3 = parse_rlvr_jsonl(json.dumps({
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "ground_truth": "a"}))
    assert not rep3.valid
