# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Raw-JSONL shape autodetection for the Guided Fine-tuning agent."""
import json

from app.validation import detect_shape


def _jsonl(rows):
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def test_detect_sft_messages():
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}]} for i in range(20)]
    res = detect_shape(_jsonl(rows))
    assert res["shape"] == "sft"
    assert res["confidence"] >= 0.8
    assert res["rows"] == 20


def test_detect_preference_chosen_rejected():
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}],
             "chosen": "good answer", "rejected": "bad answer"} for i in range(15)]
    res = detect_shape(_jsonl(rows))
    assert res["shape"] == "preference"
    assert "better and a worse" in res["label"]


def test_detect_kto_labelled():
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}],
             "kto_tag": (i % 2 == 0)} for i in range(16)]
    res = detect_shape(_jsonl(rows))
    assert res["shape"] == "kto"


def test_detect_rlvr_ground_truth():
    rows = [{"messages": [{"role": "user", "content": f"what is {i}+{i}?"}],
             "ground_truth": str(i + i)} for i in range(12)]
    res = detect_shape(_jsonl(rows))
    assert res["shape"] == "rlvr"


def test_detect_rlaif_prompts_only():
    # Prompt-only rows (no assistant turn, no chosen/rejected, no ground_truth).
    rows = [{"messages": [{"role": "user", "content": f"write a poem about {i}"}]}
            for i in range(10)]
    res = detect_shape(_jsonl(rows))
    # Prompt-only is the RLAIF signature (no more-specific shape matches).
    assert res["shape"] == "rlaif"


def test_empty_is_unknown_not_crash():
    res = detect_shape("")
    assert res["shape"] == "unknown"
    assert res["rows"] == 0


def test_garbage_is_unknown():
    res = detect_shape("not json at all\n{also not valid\n")
    assert res["shape"] == "unknown"


def test_preference_wins_over_sft_when_both_present():
    # A preference row also carries `messages`; detection must pick the MORE specific
    # preference shape, not fall through to sft.
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}],
             "chosen": "yes", "rejected": "no"} for i in range(10)]
    res = detect_shape(_jsonl(rows))
    assert res["shape"] == "preference"
    # The match-rate map should record that sft did NOT broadly match (these rows
    # have no trailing assistant turn), so preference is unambiguously correct.
    assert res["matchRates"]["preference"] >= 0.8
