# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Custom RLVR reward functions: snippet validation, Lambda packaging, the
per-tenant registry, the metric-mirror scorer, and the reward-curve parse. AWS is
not touched — these check the LOGIC (what gets packaged, what's rejected, what the
registry persists), not real Lambda/Evaluator creation (that's a real-run check)."""
import io
import json
import zipfile

import pytest


# --- scoring (eval.py mirror, torch-free) ----------------------------------

def test_scoring_extract_answer_strips_think():
    from app.reward_templates import scoring
    assert scoring.extract_answer("<think>blah</think>72") == "72"
    # unclosed think → everything from the opener dropped
    assert scoring.extract_answer("<think>still thinking") == ""


def test_scoring_score_metrics():
    from app.reward_templates import scoring
    # exact-ish numeric
    assert scoring.score("numeric_match", "<think>..</think>72", "72") == 1.0
    assert scoring.score("numeric_match", "73", "72") == 0.0
    # token_f1 partial credit, always 0..1
    v = scoring.score("token_f1", "the quick brown fox", "quick brown fox")
    assert 0.0 < v <= 1.0
    # label = normalized exact
    assert scoring.score("label_accuracy", "Urgent.", "urgent") == 1.0
    # json_valid
    assert scoring.score("json_valid", '```json\n{"a":1}\n```', "{}") == 1.0
    assert scoring.score("json_valid", "not json", "{}") == 0.0
    # unknown metric falls back to token_f1, never raises
    assert 0.0 <= scoring.score("nope", "a", "a") <= 1.0
    # a None/garbage input scores 0, never raises
    assert scoring.score("token_f1", None, None) in (0.0, 1.0)


# --- reward Lambda handler envelope normalisation --------------------------

def _load_handler(monkeypatch):
    """Import the handler template with a stub user_reward (it imports it at load).
    The handler.py + scoring.py live in app/reward_templates and are packaged into
    the Lambda zip verbatim, so testing them here covers the deployed artifact."""
    import sys
    import types as _t
    from pathlib import Path
    import importlib.util

    stub = _t.ModuleType("user_reward")
    stub.reward = lambda response, ground_truth: 1.0 if str(response).strip() == str(ground_truth).strip() else 0.0
    monkeypatch.setitem(sys.modules, "user_reward", stub)
    # scoring is imported by snippets via `import scoring`; make it importable too.
    tmpl = Path(__file__).resolve().parents[1] / "app" / "reward_templates"
    monkeypatch.syspath_prepend(str(tmpl))
    spec = importlib.util.spec_from_file_location("_reward_handler", tmpl / "handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_handler_accepts_bare_list_envelope(monkeypatch):
    """Regression for the real GRPO failure: the RLVR loop sends the batch as a
    BARE LIST, not {"batch": [...]}. The handler must score it, not crash with
    'list' object has no attribute 'get'."""
    h = _load_handler(monkeypatch)
    event = [{"id": "0", "messages": [{"role": "assistant", "content": "4"}],
              "reference_answer": "4"}]
    out = h.handler(event, None)
    body = json.loads(out["body"])
    assert out["statusCode"] == 200
    assert len(body) == 1
    assert body[0]["id"] == "0"
    assert body[0]["aggregate_reward_score"] == 1.0


def test_handler_still_accepts_batch_envelope(monkeypatch):
    """The documented {"batch": [...]} shape (and the deploy-time validation invoke)
    must keep working."""
    h = _load_handler(monkeypatch)
    out = h.handler({"batch": [{"id": "1", "messages": [{"role": "assistant", "content": "9"}],
                                "reference_answer": "8"}]}, None)
    body = json.loads(out["body"])
    assert body[0]["aggregate_reward_score"] == 0.0  # 9 != 8


def test_handler_tolerates_garbage_envelopes(monkeypatch):
    """None / unexpected shapes return an empty result, never raise (a crash would
    fail the whole billable training job)."""
    h = _load_handler(monkeypatch)
    assert json.loads(h.handler(None, None)["body"]) == []
    assert json.loads(h.handler({"weird": 1}, None)["body"]) == []
    # a single un-wrapped row is scored alone
    one = json.loads(h.handler({"id": "x", "messages": [{"role": "assistant", "content": "4"}],
                                "reference_answer": "4"}, None)["body"])
    assert one and one[0]["aggregate_reward_score"] == 1.0


# --- snippet validation ----------------------------------------------------

def test_validate_snippet_accepts_good():
    from app.reward_functions import validate_snippet
    validate_snippet(
        "import scoring\n"
        "def reward(response, ground_truth):\n"
        "    return scoring.score('token_f1', response, ground_truth)\n"
    )


def test_validate_snippet_rejects_bad():
    from app.reward_functions import RewardError, validate_snippet
    # no reward() function
    with pytest.raises(RewardError):
        validate_snippet("def other(a, b):\n    return 1.0\n")
    # wrong arity
    with pytest.raises(RewardError):
        validate_snippet("def reward(x):\n    return 1.0\n")
    # forbidden import
    with pytest.raises(RewardError):
        validate_snippet("import os\ndef reward(r, g):\n    return 1.0\n")
    # forbidden builtin
    with pytest.raises(RewardError):
        validate_snippet("def reward(r, g):\n    return eval('1')\n")
    # syntax error
    with pytest.raises(RewardError):
        validate_snippet("def reward(r, g):\n    return (\n")
    # empty
    with pytest.raises(RewardError):
        validate_snippet("   ")


def test_validate_snippet_rejects_dunder_bypass():
    """Sandbox-escape regression: the AST blocklist stopped `__import__` only as a
    bare-name CALL, so `__builtins__["__import__"]("os").system(...)` slipped
    through — a subscript, with no dunder attribute access. Dunder NAMES are now
    rejected outright."""
    from app.reward_functions import RewardError, validate_snippet
    # dunder name via subscript — the exact bypass
    with pytest.raises(RewardError):
        validate_snippet(
            "def reward(r, g):\n"
            "    return float(__builtins__['__import__']('os').getpid())\n"
        )
    # bare dunder name reference
    with pytest.raises(RewardError):
        validate_snippet("def reward(r, g):\n    return len(__builtins__)\n")
    # dunder attribute escape chain (already blocked, keep covered)
    with pytest.raises(RewardError):
        validate_snippet("def reward(r, g):\n    return ().__class__\n")


def test_validate_snippet_rejects_private_name_pivot():
    """Sandbox-escape regression. A dunder-only rule was not enough: `collections`
    is on the import allowlist and a real module keeps private aliases to its own
    imports, so `collections._sys` IS the `sys` module and
    `collections._sys.modules["os"]` was a full escape using one underscore and two
    ordinary lookups. Every private (`_`-prefixed) identifier is now rejected, and
    an allowlisted module can no longer be reached through a from-import either."""
    from app.reward_functions import RewardError, validate_snippet

    escapes = [
        # the exact pivot
        "def reward(r, g):\n    import collections\n    return float(collections._sys.modules['os'].getpid())\n",
        # same target, bound by a from-import so no attribute access appears
        "def reward(r, g):\n    from collections import _sys\n    return 1.0\n",
        "def reward(r, g):\n    from collections import _sys as z\n    return 1.0\n",
        "import json as _j\ndef reward(r, g):\n    return 1.0\n",
        # a submodule of an allowlisted root is still not allowlisted
        "def reward(r, g):\n    import json.decoder\n    return 1.0\n",
        "from math import *\ndef reward(r, g):\n    return 1.0\n",
        "from . import scoring\ndef reward(r, g):\n    return 1.0\n",
        # a public name that is simply not exported
        "def reward(r, g):\n    from collections import ChainMap\n    return 1.0\n",
        # private helpers of our own scoring mirror
        "def reward(r, g):\n    import scoring\n    return scoring._token_f1(r, g)\n",
        # getattr would defeat the static attribute rule, so it is a forbidden call
        "def reward(r, g):\n    import collections\n    return float(getattr(collections, '_sys') is None)\n",
    ]
    for snip in escapes:
        with pytest.raises(RewardError):
            validate_snippet(snip)


def test_safe_module_facade_has_no_escape_edge():
    """The second layer, independent of the AST pass: _safe_import hands out a
    _SafeModule façade, not the real module, so even a snippet that defeated
    validate_snippet has nothing to walk. `__getattribute__` is overridden rather
    than `__getattr__`, so dunders are intercepted too."""
    from app.reward_functions import safe_module

    proxy = safe_module("collections")
    for attr in ("_sys", "__class__", "__dict__", "__getattribute__", "abc"):
        with pytest.raises(AttributeError):
            getattr(proxy, attr)
    # ...while the exported names still work.
    assert proxy.Counter("aab")["a"] == 2
    # And the façade is read-only.
    with pytest.raises(AttributeError):
        proxy.Counter = None


def test_from_import_of_exported_names_still_works():
    """The tightened import rules must not break ordinary reward logic."""
    from app.reward_functions import try_reward

    snip = (
        "from collections import Counter\n"
        "from math import sqrt\n\n"
        "def reward(response, ground_truth):\n"
        "    return min(1.0, sqrt(len(Counter(response.split()))) / 10.0)\n"
    )
    assert 0.0 <= try_reward(snip, "a b c", "a b") <= 1.0


def test_try_reward_sandbox_blocks_escape():
    """Defence-in-depth: even if a dangerous snippet reaches try_reward, the
    restricted-__builtins__ sandbox has no import/open/eval, so it cannot touch the
    filesystem/network/other-tenant state from the shared backend process."""
    from app.reward_functions import RewardError, try_reward
    # `import os` resolves through the sandbox's allowlist-gated __import__ → blocked
    with pytest.raises(RewardError):
        try_reward("def reward(r, g):\n    import os\n    return 1.0\n", "x", "y")
    # open() isn't in the safe-builtins set → NameError surfaced as RewardError
    with pytest.raises(RewardError):
        try_reward("def reward(r, g):\n    open('/etc/passwd')\n    return 1.0\n", "x", "y")


def test_try_reward_sandbox_allows_legit_builtins():
    """The safe-builtins subset must still cover what real reward logic uses."""
    from app.reward_functions import try_reward
    snip = (
        "import scoring\n"
        "def reward(response, ground_truth):\n"
        "    toks = sorted(set(str(response).split()))\n"
        "    base = float(len(toks)) / max(1, len(str(ground_truth).split()))\n"
        "    return min(1.0, round(base, 3))\n"
    )
    assert 0.0 <= try_reward(snip, "a b c", "a b") <= 1.0


def test_metric_reward_snippet_roundtrips():
    from app.reward_functions import metric_reward_snippet, validate_snippet
    snip = metric_reward_snippet("token_f1")
    validate_snippet(snip)  # generated snippet must itself be valid
    assert "scoring.score('token_f1'" in snip


# --- Lambda packaging ------------------------------------------------------

def test_build_lambda_zip_contains_three_files():
    from app.reward_functions import build_lambda_zip
    snip = "def reward(response, ground_truth):\n    return 1.0\n"
    data = build_lambda_zip(snip)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert {"handler.py", "scoring.py", "user_reward.py"} <= names
        # the user's snippet is verbatim user_reward.py
        assert zf.read("user_reward.py").decode() == snip


def test_reward_hash_stable():
    from app.reward_functions import reward_hash
    a = "def reward(r, g):\n    return 1.0\n"
    assert reward_hash(a) == reward_hash("  " + a + "  ")  # strips
    assert reward_hash(a) != reward_hash("def reward(r, g):\n    return 0.5\n")


# --- registry (per-tenant, persisted) --------------------------------------

def test_registry_crud(temp_store):
    from app import reward_functions as rf
    fns = rf.list_reward_functions()
    assert fns == []
    obj = rf.make_reward_function("my-reward", metric="token_f1", stamp="20260615-1")
    rf.save_reward_function(obj)
    assert len(rf.list_reward_functions()) == 1
    got = rf.get_reward_function(obj.id)
    assert got["name"] == "my-reward" and got["kind"] == "metric"
    assert got["status"] == "draft" and got["deployed"] is False
    assert rf.delete_reward_function(obj.id) is True
    assert rf.list_reward_functions() == []
    assert rf.delete_reward_function(obj.id) is False


def test_make_reward_function_requires_exactly_one_source():
    from app.reward_functions import RewardError, make_reward_function
    with pytest.raises(RewardError):
        make_reward_function("x")  # neither
    with pytest.raises(RewardError):
        make_reward_function("x", snippet="def reward(r,g): return 1.0", metric="token_f1")  # both


def test_deploy_reward_task_records_failure(temp_store, monkeypatch):
    """If the AWS deploy raises, the record flips to failed with the error (never
    crashes the worker)."""
    from app import reward_functions as rf
    obj = rf.make_reward_function("r", metric="token_f1", stamp="s")
    rf.save_reward_function(obj)

    def _boom(reward_id, name, zip_bytes, lambda_key=None):
        raise RuntimeError("no aws in test")

    monkeypatch.setattr("app.reward_deploy.deploy_reward_function", _boom)
    rf.run_deploy_reward_task(obj.id)
    rec = rf.get_reward_function(obj.id)
    assert rec["status"] == "failed" and "no aws" in rec["error"]


def test_deploy_reward_task_records_success(temp_store, monkeypatch):
    from app import reward_functions as rf
    obj = rf.make_reward_function("r", metric="token_f1", stamp="s")
    rf.save_reward_function(obj)
    captured = {}
    monkeypatch.setattr("app.reward_deploy.deploy_reward_function",
                        lambda rid, name, zb, lambda_key=None: captured.update(lambda_key=lambda_key)
                        or {"lambdaArn": "arn:lam", "evaluatorArn": "arn:ev"})
    rf.run_deploy_reward_task(obj.id)
    rec = rf.get_reward_function(obj.id)
    assert rec["status"] == "deployed" and rec["deployed"] is True
    assert rec["evaluatorArn"] == "arn:ev"
    # the Lambda is named by the snippet hash (idempotent reuse), not the record id
    assert captured["lambda_key"] == rec["lambdaHash"]


def test_ensure_preset_reward_provisions_builtin(temp_store, monkeypatch):
    """A preset (gsm8k) auto-provisions a built-in numeric_match reward function:
    deploy is called once, the registry record is created + flipped to deployed, and
    its Evaluator ARN is returned. This is how the serverless engine consumes a
    preset now that the recipe no longer accepts preset_reward_function."""
    from app import reward_functions as rf
    calls = []
    monkeypatch.setattr("app.reward_deploy.deploy_reward_function",
                        lambda rid, name, zb, lambda_key=None: calls.append(rid)
                        or {"lambdaArn": "arn:lam", "evaluatorArn": "arn:ev-builtin"})
    arn = rf.ensure_preset_reward_evaluator_arn("gsm8k", stamp="s")
    assert arn == "arn:ev-builtin"
    rec = rf.get_reward_function(rf.builtin_reward_id("numeric_match"))
    assert rec is not None
    assert rec["kind"] == "metric" and rec["metric"] == "numeric_match"
    assert rec["status"] == "deployed" and rec["deployed"] is True
    assert len(calls) == 1


def test_ensure_preset_reward_is_idempotent_and_shared(temp_store, monkeypatch):
    """gsm8k and prime_math SHARE one built-in (both → numeric_match). Once deployed,
    a second resolution reuses the stored ARN without re-deploying."""
    from app import reward_functions as rf
    calls = []
    monkeypatch.setattr("app.reward_deploy.deploy_reward_function",
                        lambda rid, name, zb, lambda_key=None: calls.append(rid)
                        or {"lambdaArn": "arn:lam", "evaluatorArn": "arn:ev-shared"})
    a = rf.ensure_preset_reward_evaluator_arn("gsm8k", stamp="s")
    b = rf.ensure_preset_reward_evaluator_arn("prime_math", stamp="s")  # same metric
    assert a == b == "arn:ev-shared"
    assert len(calls) == 1  # deployed once, reused on the second call


def test_ensure_preset_reward_unknown_preset_raises(temp_store):
    from app import reward_functions as rf
    with pytest.raises(rf.RewardError, match="unknown preset"):
        rf.ensure_preset_reward_evaluator_arn("prime_code")  # dropped preset


def test_same_metric_two_rewards_are_distinct_records(temp_store):
    """Two reward functions on the SAME metric (or same snippet) must be DISTINCT
    registry rows — the old snippet-only-hash id silently overwrote the first and
    made it un-deletable. Regression for the UI 'added but can't remove' bug."""
    from app import reward_functions as rf
    a = rf.make_reward_function("first", metric="token_f1", stamp="20260615-1")
    rf.save_reward_function(a)
    b = rf.make_reward_function("second", metric="token_f1", stamp="20260615-2")
    rf.save_reward_function(b)
    assert a.id != b.id                       # distinct ids despite identical snippet
    assert a.lambda_hash == b.lambda_hash      # but SAME AWS lambda (idempotent reuse)
    ids = {r["id"] for r in rf.list_reward_functions()}
    assert {a.id, b.id} <= ids                 # BOTH rows present (no overwrite)
    assert rf.delete_reward_function(a.id) is True
    assert rf.get_reward_function(b.id) is not None  # deleting one leaves the other


# --- reward curve parse ----------------------------------------------------

def test_parse_reward_curve():
    from app.reward_curve import parse_reward_curve
    text = "\n".join([
        json.dumps({"step": 1, "critic/rewards/mean": 0.1, "critic/rewards/max": 0.3, "critic/rewards/min": 0.0}),
        json.dumps({"step": 2, "critic/rewards/mean": 0.4, "val-core/customized/reward/mean@1": 0.35}),
        "not json",
        json.dumps({"no_step": True}),  # skipped (no step)
    ])
    c = parse_reward_curve(text)
    assert c["hasData"] is True
    assert c["steps"] == [1, 2]
    assert c["rewardMean"] == [0.1, 0.4]
    assert c["valReward"] == [{"step": 2, "value": 0.35}]


def test_parse_reward_curve_empty():
    from app.reward_curve import parse_reward_curve
    c = parse_reward_curve("")
    assert c["hasData"] is False and c["steps"] == []


# --- reward-domain guard (reward must be able to grade the ground_truth) ----

def test_reward_domain_warning_preset():
    from app.reward_functions import reward_domain_warning
    # gsm8k on numeric ground_truth → fine (no warning)
    assert reward_domain_warning(preset="gsm8k", ground_truth_task="numeric") is None
    # gsm8k on prose → warn (would score ~0)
    w = reward_domain_warning(preset="gsm8k", ground_truth_task="text")
    assert w and "numeric" in w.lower()
    # prime_math on label → warn
    assert reward_domain_warning(preset="prime_math", ground_truth_task="label") is not None
    # prime_math on numeric → fine
    assert reward_domain_warning(preset="prime_math", ground_truth_task="numeric") is None
    # prime_code was DROPPED as a preset — unknown preset → can't judge → no warning
    assert reward_domain_warning(preset="prime_code", ground_truth_task="numeric") is None
    # unknown task → can't judge → no warning
    assert reward_domain_warning(preset="gsm8k", ground_truth_task=None) is None


def test_reward_domain_warning_custom_metric():
    from app.reward_functions import reward_domain_warning
    # numeric_match on prose → warn
    assert reward_domain_warning(metric="numeric_match", ground_truth_task="text") is not None
    # numeric_match on numeric → fine
    assert reward_domain_warning(metric="numeric_match", ground_truth_task="numeric") is None
    # a metric we don't domain-restrict (token_f1) → never warns
    assert reward_domain_warning(metric="token_f1", ground_truth_task="text") is None


# --- dry-run (try_reward): score one sample in-process, no AWS ---------------

def test_try_reward_metric_snippet_matches_scoring():
    from app.reward_functions import metric_reward_snippet, try_reward
    snip = metric_reward_snippet("numeric_match")
    # extract_answer strips <think> then numeric_match → exact number wins
    assert try_reward(snip, "<think>hmm</think>72", "72") == 1.0
    assert try_reward(snip, "73", "72") == 0.0


def test_try_reward_custom_snippet():
    from app.reward_functions import try_reward
    snip = "def reward(response, ground_truth):\n    return 1.0 if ground_truth in response else 0.0\n"
    assert try_reward(snip, "the answer is cat", "cat") == 1.0
    assert try_reward(snip, "the answer is dog", "cat") == 0.0


def test_try_reward_clamps_and_guards_nan():
    from app.reward_functions import try_reward
    # >1 clamps to 1.0; a NaN-returning reward guards to 0.0
    assert try_reward("def reward(r, g):\n    return 5.0\n", "x", "y") == 1.0
    assert try_reward("def reward(r, g):\n    return float('nan')\n", "x", "y") == 0.0
    assert try_reward("def reward(r, g):\n    return -3.0\n", "x", "y") == 0.0


def test_reward_metric_for_rank_loop():
    """The reward↔leaderboard-metric loop: a rank metric maps to a reward ONLY
    when it's a verifiable per-row scorer; non-verifiable metrics map to None."""
    from app.reward_functions import reward_metric_for_rank
    from app.reward_templates import scoring

    # every verifiable scorer is its own reward (identity within the scorer set)
    for m in scoring.METRIC_NAMES:
        assert reward_metric_for_rank(m) == m
    # leaderboard metrics with NO verifiable per-row reward → None (UI explains why)
    for m in ("llm_judge:overall", "llm_judge:faithfulness", "json_structural",
              "json_key_recall", "", None):
        assert reward_metric_for_rank(m) is None


def test_try_reward_rejects_invalid_and_raising():
    from app.reward_functions import RewardError, try_reward
    # invalid snippet (no reward fn) → RewardError
    with pytest.raises(RewardError):
        try_reward("def nope(): return 1", "x", "y")
    # a reward that raises on the sample → RewardError (so the user sees it)
    with pytest.raises(RewardError):
        try_reward("def reward(r, g):\n    return 1/0\n", "x", "y")
    # disallowed import is rejected before exec
    with pytest.raises(RewardError):
        try_reward("import os\ndef reward(r, g):\n    return 1.0\n", "x", "y")


# --- RLAIF reward PROMPT kind (AI-judge reward; no Lambda) ------------------

def test_validate_reward_prompt_requires_placeholders():
    from app.reward_functions import validate_reward_prompt, RewardError
    # valid: both placeholders present (whitespace inside braces tolerated)
    validate_reward_prompt("Rate {{ prompt }} answered by {{response}} as JSON {\"score\":0..1}")
    # empty / missing placeholders rejected
    for bad in ("", "   ", "no placeholders", "only {{prompt}} here", "only {{response}} here"):
        with pytest.raises(RewardError):
            validate_reward_prompt(bad)


def test_make_reward_prompt_record():
    from app.reward_functions import make_reward_prompt
    rp = make_reward_prompt("helpfulness",
                            "Judge {{prompt}} vs {{response}} → {\"score\":0..1}",
                            reward_model_id="openai.gpt-oss-20b-1:0", stamp="20260617-1")
    assert rp.kind == "reward_prompt"
    assert rp.reward_model_id == "openai.gpt-oss-20b-1:0"
    assert rp.snippet == ""           # no code
    assert rp.lambda_hash != ""        # hash NAMES the S3 object + Evaluator
    d = rp.to_dict()
    assert d["kind"] == "reward_prompt"
    assert d["prompt"].startswith("Judge")
    assert d["rewardModelId"] == "openai.gpt-oss-20b-1:0"
    assert d["deployed"] is False      # draft until the Evaluator is registered


def test_reward_prompt_deploy_registers_evaluator(temp_store, monkeypatch):
    # A reward_prompt deploy registers a REWARD_PROMPT Evaluator (no Lambda). The
    # actual Evaluator.create runs in the V3 subprocess (AWS) — mock the deploy fn
    # so this unit test stays AWS-free, then assert the ARN is stored + deployed.
    import app.reward_functions as rf
    import app.reward_deploy as rd
    monkeypatch.setattr(rd, "deploy_reward_prompt",
                        lambda rid, text, prompt_key=None: {
                            "evaluatorArn": "arn:aws:sagemaker:us-east-1:1:hub-content/h/JsonDoc/x/2.0.0",
                            "promptS3Uri": "s3://b/slm-platform/reward_prompts/x.txt"})
    rec = rf.make_reward_prompt("hp", "Score {{prompt}} / {{response}} → {\"score\":0..1}", stamp="s")
    rec.status = "deploying"
    rf.save_reward_function(rec)
    rf.run_deploy_reward_task(rec.id)
    got = rf.get_reward_function(rec.id)
    assert got["status"] == "deployed"
    assert got["deployed"] is True
    assert got["evaluatorArn"].startswith("arn:aws:sagemaker:")
    assert got["lambdaArn"] == ""  # NO Lambda for a reward prompt
    assert got["promptS3Uri"].startswith("s3://")


def test_make_reward_prompt_rejects_invalid_judge_model():
    from app.reward_functions import make_reward_prompt, RewardError, ALLOWED_JUDGE_MODELS
    good = "Score {{prompt}} / {{response}} → {\"score\":0..1}"
    # an Amazon Nova id is NOT a valid RLAIF judge (the real launch rejects it)
    with pytest.raises(RewardError):
        make_reward_prompt("hp", good, reward_model_id="us.amazon.nova-pro-v1:0", stamp="s")
    # a valid judge from the allowed list is accepted
    rp = make_reward_prompt("hp", good, reward_model_id=ALLOWED_JUDGE_MODELS[0], stamp="s")
    assert rp.reward_model_id == ALLOWED_JUDGE_MODELS[0]
    # blank = recipe default, allowed
    assert make_reward_prompt("hp", good, stamp="s").reward_model_id == ""


def test_reward_prompt_deploy_fails_on_bad_prompt(temp_store):
    # If a record somehow has a bad prompt, deploy marks it failed (not deployed) —
    # validation runs BEFORE any AWS call, so no mock is needed.
    import app.reward_functions as rf
    rec = rf.make_reward_prompt("hp", "Score {{prompt}} / {{response}}", stamp="s")
    # corrupt the stored prompt to drop a placeholder, then deploy
    rec.prompt = "missing the response placeholder {{prompt}}"
    rec.status = "deploying"
    rf.save_reward_function(rec)
    rf.run_deploy_reward_task(rec.id)
    got = rf.get_reward_function(rec.id)
    assert got["status"] == "failed"
    assert got["deployed"] is False


# --- try_reward_prompt: dry-run a judge RUBRIC (no AWS — stub Converse) -------

RUBRIC = ('Rate the response to {{prompt}}: {{response}} — reply with ONLY JSON '
         '{"score": <0..1>, "reasoning": "<one sentence>"}')


class _StubConverse:
    """A fake bedrock-runtime client whose converse() returns a canned reply text.
    Records the modelId + the user message it was called with."""
    def __init__(self, reply_text: str):
        self._reply = reply_text
        self.calls: list[dict] = []

    def converse(self, modelId, messages, inferenceConfig, **kw):  # noqa: N803
        self.calls.append({"modelId": modelId, "messages": messages})
        return {"output": {"message": {"content": [{"text": self._reply}]}}}


def test_fill_reward_prompt_substitutes_placeholders():
    from app.reward_functions import _fill_reward_prompt
    out = _fill_reward_prompt("P={{ prompt }} R={{response}}", "hello", "world")
    assert out == "P=hello R=world"


def test_try_reward_prompt_scores_and_clamps():
    from app.reward_functions import try_reward_prompt
    stub = _StubConverse('{"score": 0.83, "reasoning": "friendly + concise"}')
    out = try_reward_prompt(RUBRIC, "Greet a user", "Hi! Happy to help.", "qwen.qwen3-32b-v1:0", _client=stub)
    assert out["error"] is None
    assert out["score"] == 0.83
    assert "friendly" in out["reasoning"]
    # the rubric's placeholders were filled into the judge message
    sent = stub.calls[0]["messages"][0]["content"][0]["text"]
    assert "Greet a user" in sent and "Hi! Happy to help." in sent
    assert stub.calls[0]["modelId"] == "qwen.qwen3-32b-v1:0"
    # out-of-range score clamps to [0,1]
    stub2 = _StubConverse('{"score": 5.0, "reasoning": "x"}')
    assert try_reward_prompt(RUBRIC, "p", "r", "qwen.qwen3-32b-v1:0", _client=stub2)["score"] == 1.0


def test_try_reward_prompt_blank_judge_uses_dry_run_fallback():
    from app.reward_functions import try_reward_prompt, _DRY_RUN_FALLBACK_JUDGE
    stub = _StubConverse('{"score": 0.5, "reasoning": "ok"}')
    out = try_reward_prompt(RUBRIC, "p", "r", "", _client=stub)  # blank judge
    assert out["score"] == 0.5
    assert stub.calls[0]["modelId"] == _DRY_RUN_FALLBACK_JUDGE  # previewed with the fallback


def test_try_reward_prompt_never_raises_on_bad_judge_reply():
    from app.reward_functions import try_reward_prompt
    # non-JSON reply → graceful error result, score 0, no exception
    out = try_reward_prompt(RUBRIC, "p", "r", "qwen.qwen3-32b-v1:0",
                            _client=_StubConverse("I think this is pretty good honestly"))
    assert out["score"] == 0.0 and out["error"] is not None
    # JSON without a numeric score → graceful
    out2 = try_reward_prompt(RUBRIC, "p", "r", "qwen.qwen3-32b-v1:0",
                             _client=_StubConverse('{"reasoning": "no score field"}'))
    assert out2["score"] == 0.0 and out2["error"] is not None
    # a Converse call that raises → graceful error, not a crash
    class _Boom:
        def converse(self, **kw): raise RuntimeError("bedrock down")
    out3 = try_reward_prompt(RUBRIC, "p", "r", "qwen.qwen3-32b-v1:0", _client=_Boom())
    assert out3["score"] == 0.0 and "failed" in out3["error"]


def test_try_reward_prompt_rejects_bad_inputs():
    from app.reward_functions import try_reward_prompt, RewardError
    # missing placeholder → RewardError (before any judge call)
    with pytest.raises(RewardError):
        try_reward_prompt("no placeholders here", "p", "r", "qwen.qwen3-32b-v1:0", _client=_StubConverse("{}"))
    # non-allowlisted judge id → RewardError (before any billable Converse)
    with pytest.raises(RewardError):
        try_reward_prompt(RUBRIC, "p", "r", "us.amazon.nova-pro-v1:0", _client=_StubConverse("{}"))


def test_try_prompt_endpoint_computes_spread(monkeypatch):
    """POST /api/reward-functions/try-prompt scores good/bad candidates + reports a
    discrimination spread. Stub try_reward_prompt so no AWS is touched."""
    import app.main as m

    def _fake(prompt_text, prompt, response, judge_model_id, _client=None):
        # "good" candidates score high, "bad" low — simulate a discriminating rubric.
        hi = "good" in response
        return {"score": 0.9 if hi else 0.1, "reasoning": "stub", "error": None}

    monkeypatch.setattr("app.reward_functions.try_reward_prompt", _fake)
    req = m.RewardPromptTryRequest(
        prompt=RUBRIC, rewardModelId="qwen.qwen3-32b-v1:0",
        samples=[
            {"prompt": "greet", "response": "a good warm reply", "intendedLabel": "good"},
            {"prompt": "greet", "response": "a bad rude reply", "intendedLabel": "bad"},
        ],
    )
    out = m.try_reward_prompt_fn(req)
    assert len(out["samples"]) == 2
    assert out["scoreSpread"]["goodMean"] == 0.9
    assert out["scoreSpread"]["badMean"] == 0.1
    assert out["scoreSpread"]["discriminates"] is True
    assert out["indicative"] is True


def test_try_prompt_endpoint_caps_samples(monkeypatch):
    """The endpoint caps the candidate count so a user can't spam billable calls."""
    import app.main as m
    calls = {"n": 0}

    def _fake(*a, **k):
        calls["n"] += 1
        return {"score": 0.5, "reasoning": "", "error": None}

    monkeypatch.setattr("app.reward_functions.try_reward_prompt", _fake)
    req = m.RewardPromptTryRequest(
        prompt=RUBRIC, samples=[{"prompt": "p", "response": f"r{i}"} for i in range(50)])
    m.try_reward_prompt_fn(req)
    assert calls["n"] == m._MAX_DRYRUN_SAMPLES  # capped, not 50
