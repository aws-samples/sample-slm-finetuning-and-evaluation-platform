# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LLM-as-judge — pure parsing + persistence (no Bedrock calls)."""
from app.judge import _detect_task, _parse_judge

# Default dimension set used in most tests (the rich JSON-task rubric).
DIMS = ["correctness", "faithfulness", "format", "completeness", "conciseness"]


def test_parse_multidimension_json():
    per_dim, overall, reason = _parse_judge(
        '{"correctness":5,"faithfulness":4,"format":5,"overall":4,"reason":"good but slightly off"}',
        DIMS,
    )
    assert overall == 4
    assert per_dim["correctness"] == 5 and per_dim["faithfulness"] == 4
    assert "off" in reason


def test_parse_json_with_surrounding_prose():
    per_dim, overall, _ = _parse_judge(
        'Sure: {"correctness":5,"format":5,"overall":5,"reason":"exact"} done', DIMS
    )
    assert overall == 5
    assert per_dim["correctness"] == 5


def test_parse_clamps_out_of_range():
    assert _parse_judge('{"overall": 9, "reason": "x"}', DIMS)[1] == 5
    assert _parse_judge('{"overall": 0, "reason": "x"}', DIMS)[1] == 1


def test_overall_derived_from_dimensions_when_missing():
    # No explicit overall → mean of the per-dimension scores (rounded).
    per_dim, overall, _ = _parse_judge('{"correctness":4,"format":2,"reason":"x"}', ["correctness", "format"])
    assert overall == 3  # round((4+2)/2)


def test_parse_fallback_to_digit():
    _, overall, _ = _parse_judge("I would say 3 out of 5", DIMS)
    assert overall == 3


def test_parse_garbage_returns_zero():
    _, overall, _ = _parse_judge("no number here", DIMS)
    assert overall == 0


def test_detect_task_picks_rubric():
    assert _detect_task('{"narrative": "x"}') == "json"
    assert _detect_task("urgency: high") == "label"
    assert _detect_task("42.5") == "numeric"
    assert _detect_task("The user is browsing hotels in Lisbon.") == "text"


def test_judge_status_none_then_running(temp_store):
    from app.judge import judge_status, set_judge_status

    assert judge_status("eval-job-x")["status"] == "none"
    set_judge_status("eval-job-x", "running")
    assert judge_status("eval-job-x")["status"] == "running"


def test_judge_done_inferred_from_saved_result(temp_store):
    from app.judge import _save_judge, judge_status, load_judge

    _save_judge("eval-job-y", {"judgeScore": 4.2, "judgedRows": 10})
    # 'done' is inferred whenever a result exists, regardless of status marker.
    assert judge_status("eval-job-y")["status"] == "done"
    assert load_judge("eval-job-y")["judgeScore"] == 4.2


class _FakeConverseClient:
    """Records the converse() call and returns a canned judge reply in the
    Converse response shape (output.message.content[].text + usage.*Tokens)."""

    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [
                {"text": '{"correctness":5,"format":4,"overall":5,"reason":"ok"}'}
            ]}},
            "usage": {"inputTokens": 12, "outputTokens": 7},
        }


def test_judge_one_uses_converse_not_invoke_model():
    """_judge_one calls the model-agnostic Converse API (so Nova works too) and
    parses the Converse response/usage shape."""
    from app.judge import _judge_one

    client = _FakeConverseClient()
    per_dim, overall, reason, it, ot = _judge_one(
        client, prompt="P", gold="cat", pred="cat", model_id="us.amazon.nova-pro-v1:0"
    )
    # Made exactly one converse() call with the right model + Converse arg shape.
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["modelId"] == "us.amazon.nova-pro-v1:0"
    assert call["system"][0]["text"]  # system prompt present (Converse shape)
    assert call["messages"][0]["content"][0]["text"]  # Converse content block (no "type")
    assert "inferenceConfig" in call
    # Parsed the canned reply + Converse usage keys (inputTokens/outputTokens).
    assert overall == 5 and per_dim["correctness"] == 5
    assert it == 12 and ot == 7


def test_judge_key_selects_model(monkeypatch, temp_store):
    """run_judge resolves judge_key → a baseline model spec (Claude or Nova) and
    records which model judged."""
    from app import judge as judge_mod

    captured = {}

    def fake_session(cfg):
        class _B:
            def client(self, *a, **k):
                return _FakeConverseClient()
        return None, _B()

    monkeypatch.setattr(judge_mod, "_session", fake_session)
    monkeypatch.setattr(judge_mod, "load_aws_config", lambda: type("C", (), {"region": "us-east-1"})())
    monkeypatch.setattr(judge_mod, "_load_predictions",
                        lambda j: [{"gold": "cat", "prediction": "cat", "prompt": "p"}])

    res = judge_mod.run_judge("eval-z", judge_key="nova-pro")
    assert res["judgeKey"] == "nova-pro"
    assert res["model"] == "us.amazon.nova-pro-v1:0"
    assert res["label"] == "Amazon Nova Pro"
    assert res["judgedRows"] == 1


def test_judge_empty_eval_short_circuits(monkeypatch, temp_store):
    """An empty eval set must NOT emit a misleading judgeScore=0.0 over 0 rows —
    it returns an explicit no_eval_rows status (judgeScore None) instead."""
    from app import judge as judge_mod

    def fake_session(cfg):
        class _B:
            def client(self, *a, **k):
                return _FakeConverseClient()
        return None, _B()

    monkeypatch.setattr(judge_mod, "_session", fake_session)
    monkeypatch.setattr(judge_mod, "load_aws_config", lambda: type("C", (), {"region": "us-east-1"})())
    monkeypatch.setattr(judge_mod, "_load_predictions", lambda j: [])  # empty eval

    res = judge_mod.run_judge("eval-empty")
    assert res["status"] == "no_eval_rows"
    assert res["judgeScore"] is None
    assert res["judgedRows"] == 0
