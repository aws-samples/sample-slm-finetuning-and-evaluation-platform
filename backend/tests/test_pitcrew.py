# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Guided Fine-tuning agent — the deterministic conversation state machine.

Stubs the LLM narration (no Bedrock) and the race launch (no AWS), so this tests
the STATE MACHINE: greet → goal → data → profile → confirm → effort → plan →
review → (approve) → launch, plus the no-auto-launch and version guards.
"""
import json

import pytest

from app import pitcrew


def _make_sft_split(store_mod, split_id, n_train=2000, n_eval=100):
    store = store_mod.get_store()
    wd = store.workdir("runs", split_id)
    train = [{"messages": [{"role": "user", "content": f"classify {i}"},
                           {"role": "assistant", "content": "billing"}]} for i in range(n_train)]
    eval_ = [{"messages": [{"role": "user", "content": f"classify e{i}"},
                          {"role": "assistant", "content": "billing"}]} for i in range(n_eval)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in eval_) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps(
        {"name": split_id, "shape": "sft", "hasVal": False, "trainRows": n_train, "evalRows": n_eval}),
        encoding="utf-8")
    store.commit("runs", split_id)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Narration falls back to the deterministic string (no Bedrock call in tests)."""
    monkeypatch.setattr(pitcrew, "_narrate", lambda prompt, fallback: fallback)


def _drive_to_review(store_mod, monkeypatch, split_id="sft-ds"):
    _make_sft_split(store_mod, split_id)
    # No HF token in tests.
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    s = pitcrew.start_session("sess-1", "20260630-120000")
    assert s["phase"] == pitcrew.COLLECT_GOAL
    pitcrew.advance("sess-1", "goal", {"goal": "sort tickets into categories"}, "20260630-120001")
    s = pitcrew.advance("sess-1", "use_dataset", {"splitId": split_id}, "20260630-120002")
    assert s["phase"] == pitcrew.CONFIRM_TASK
    s = pitcrew.advance("sess-1", "confirm", {}, "20260630-120003")
    assert s["phase"] == pitcrew.CHOOSE_EFFORT
    s = pitcrew.advance("sess-1", "effort", {"effort": "easy"}, "20260630-120004")
    return s


def test_full_flow_reaches_review_with_plan_and_estimate(temp_store, monkeypatch):
    s = _drive_to_review(temp_store, monkeypatch)
    assert s["phase"] == pitcrew.REVIEW_PLAN
    assert s["plan"] and s["plan"]["supported"]
    # "quick" is an up-to-4 ceiling; the plan fills as many arms as add signal (≤4).
    assert 2 <= len(s["plan"]["models"]) <= 4
    assert len(s["plan"]["models"]) == s["plan"]["meaningfulCount"]
    assert s["estimate"] and s["estimate"]["totalUsd"]["hi"] > 0
    # The review message carries the structured plan + estimate for the UI.
    last = s["messages"][-1]
    assert last.get("reviewPlan") and "plan" in last and "estimate" in last


def test_review_does_not_auto_launch(temp_store, monkeypatch):
    s = _drive_to_review(temp_store, monkeypatch)
    # No raceId until the user explicitly approves.
    assert s["phase"] == pitcrew.REVIEW_PLAN
    assert not s["raceId"]


def test_approve_launches_via_start_race(temp_store, monkeypatch):
    launched = {}

    class _FakeRace:
        race_id = "race-sft-ds-20260630"
        entries = []

    def _fake_start_race(split_id, race_models, decoding, stamp, **kw):
        launched["split_id"] = split_id
        launched["models"] = [rm.model_id for rm in race_models]
        launched["notify"] = kw.get("notify_emails")
        return _FakeRace()

    monkeypatch.setattr("app.race.start_race", _fake_start_race)
    s = _drive_to_review(temp_store, monkeypatch)
    s = pitcrew.advance("sess-1", "approve", {"notifyEmail": "me@example.com"}, "20260630-120010")
    assert s["phase"] == pitcrew.LAUNCHED
    assert s["raceId"] == "race-sft-ds-20260630"
    assert launched["split_id"] == "sft-ds"
    assert 2 <= len(launched["models"]) <= 4  # "quick" ceiling, filled as far as useful
    assert launched["notify"] == ["me@example.com"]


def test_approve_is_idempotent_no_duplicate_race(temp_store, monkeypatch):
    # THE duplicate-race guard: a second approve (e.g. the user retried after a slow
    # launch appeared to error) must NOT launch a second race.
    calls = {"n": 0}

    class _FakeRace:
        race_id = "race-sft-ds-20260630"
        entries = []

    def _fake_start_race(split_id, race_models, decoding, stamp, **kw):
        calls["n"] += 1
        return _FakeRace()

    monkeypatch.setattr("app.race.start_race", _fake_start_race)
    s = _drive_to_review(temp_store, monkeypatch)
    v = s["version"]
    s = pitcrew.advance("sess-1", "approve", {}, "20260630-120010", expected_version=v)
    assert s["phase"] == pitcrew.LAUNCHED and s["raceId"] == "race-sft-ds-20260630"
    # Second approve (stale button / retry) — must be a no-op launch-wise.
    s = pitcrew.advance("sess-1", "approve", {}, "20260630-120011")
    assert calls["n"] == 1, "start_race must be called exactly once across two approves"
    assert s["phase"] == pitcrew.LAUNCHED and s["raceId"] == "race-sft-ds-20260630"


def test_launch_dispatches_to_worker_when_configured(temp_store, monkeypatch):
    # When a worker Lambda is configured, approve must dispatch the launch OFF the
    # request path (returns instantly, phase=launched, launching flag set) and NOT call
    # start_race inline — then the worker task runs the real launch.
    dispatched = {}

    def _fake_dispatch(payload):
        dispatched.update(payload)
        return True  # simulate a configured worker

    def _boom_start(*a, **k):
        raise AssertionError("start_race must NOT run inline when a worker is configured")

    monkeypatch.setattr("app.dispatch.dispatch_worker", _fake_dispatch)
    monkeypatch.setattr("app.race.start_race", _boom_start)
    s = _drive_to_review(temp_store, monkeypatch)
    s = pitcrew.advance("sess-1", "approve", {}, "20260630-120010")
    assert s["phase"] == pitcrew.LAUNCHED
    assert s.get("launching") is True and not s["raceId"]  # provisional state, no race yet
    assert dispatched.get("task") == "pitcrew_launch"
    assert dispatched.get("sessionId") == "sess-1"

    # Now simulate the worker running the launch task → real race appears, flag cleared.
    class _FakeRace:
        race_id = "race-sft-ds-worker"
        entries = []

    monkeypatch.setattr("app.race.start_race", lambda *a, **k: _FakeRace())
    pitcrew.run_pitcrew_launch("sess-1", dispatched["stamp"])
    s = pitcrew.get_session("sess-1") if hasattr(pitcrew, "get_session") else pitcrew._load("sess-1")
    assert s["raceId"] == "race-sft-ds-worker"
    assert not s.get("launching")


def test_cancel_does_not_launch(temp_store, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("start_race must not be called on cancel")

    monkeypatch.setattr("app.race.start_race", _boom)
    s = _drive_to_review(temp_store, monkeypatch)
    s = pitcrew.advance("sess-1", "cancel", {}, "20260630-120010")
    assert s["phase"] == pitcrew.CHOOSE_EFFORT
    assert not s["raceId"]


def test_stale_version_is_rejected(temp_store, monkeypatch):
    _drive_to_review(temp_store, monkeypatch)
    # Advancing with an obviously stale version must raise (a second tab moved on).
    with pytest.raises(ValueError, match="another tab"):
        pitcrew.advance("sess-1", "approve", {}, "20260630-120010", expected_version=0)


def test_unknown_session_raises(temp_store):
    with pytest.raises(ValueError, match="unknown session"):
        pitcrew.advance("nope", "goal", {"goal": "x"}, "20260630-120000")


def test_greeting_prompts_for_goal_input(temp_store):
    # The opening message MUST flag the goal text box, else the conversation can't
    # start (the bug a user hit: no input rendered).
    s = pitcrew.start_session("g", "20260630-143000")
    assert s["messages"][-1].get("collectGoal") is True
    assert s["phase"] == pitcrew.COLLECT_GOAL


def test_default_title_is_distinguishable_per_session(temp_store):
    a = pitcrew.start_session("a", "20260630-143000")
    b = pitcrew.start_session("b", "20260630-150500")
    # Default titles carry the start time, so two fresh sessions aren't identical.
    assert a["title"] != b["title"]
    assert "New session" in a["title"] and "30 Jun" in a["title"]


def test_goal_becomes_title_unless_manually_renamed(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    pitcrew.start_session("t1", "20260630-143000")
    s = pitcrew.advance("t1", "goal", {"goal": "sort support tickets by topic"}, "20260630-143100")
    assert s["title"] == "sort support tickets by topic"

    # A manual rename sticks even after a later goal turn.
    pitcrew.start_session("t2", "20260630-143000")
    renamed = pitcrew.rename_session("t2", "My ticket classifier")
    assert renamed["title"] == "My ticket classifier"
    assert renamed["titleManual"] is True
    s2 = pitcrew.advance("t2", "goal", {"goal": "something else entirely"}, "20260630-143100")
    assert s2["title"] == "My ticket classifier"  # not overwritten by the goal


def test_rename_blank_keeps_current_title(temp_store):
    pitcrew.start_session("r", "20260630-143000")
    before = pitcrew.get_session("r")["title"]
    after = pitcrew.rename_session("r", "   ")
    assert after["title"] == before  # blank ignored


def test_rename_unknown_session_returns_none(temp_store):
    assert pitcrew.rename_session("nope", "x") is None


def test_archive_hides_session_from_list(temp_store):
    pitcrew.start_session("keep", "20260630-143000")
    pitcrew.start_session("drop", "20260630-144000")
    assert {s["sessionId"] for s in pitcrew.list_sessions()} == {"keep", "drop"}
    assert pitcrew.archive_session("drop") is True
    # Archived session is hidden from the sidebar list...
    assert {s["sessionId"] for s in pitcrew.list_sessions()} == {"keep"}
    # ...but still loadable (the race it launched is never orphaned).
    assert pitcrew.get_session("drop") is not None


def test_archive_unknown_session_returns_false(temp_store):
    assert pitcrew.archive_session("nope") is False


def test_archive_restore(temp_store):
    pitcrew.start_session("s", "20260630-143000")
    pitcrew.archive_session("s", True)
    assert pitcrew.list_sessions() == []
    pitcrew.archive_session("s", False)
    assert {x["sessionId"] for x in pitcrew.list_sessions()} == {"s"}


def test_sessions_listed_newest_first(temp_store):
    pitcrew.start_session("a", "20260630-120000")
    pitcrew.start_session("b", "20260630-130000")
    sessions = pitcrew.list_sessions()
    ids = [s["sessionId"] for s in sessions]
    assert set(ids) == {"a", "b"}


def test_review_carries_editable_models_and_add_pool(temp_store, monkeypatch):
    s = _drive_to_review(temp_store, monkeypatch)
    last = s["messages"][-1]
    assert last.get("reviewPlan")
    # The plan lists the proposed models, and an add-pool of more eligible models.
    assert 2 <= len(last["plan"]["models"]) <= 4
    assert isinstance(last.get("addPool"), list) and len(last["addPool"]) > 0
    # The pool must not include models already in the plan.
    chosen = {m["modelId"] for m in last["plan"]["models"]}
    assert all(m["modelId"] not in chosen for m in last["addPool"])


def test_edit_models_remove_then_add_updates_plan(temp_store, monkeypatch):
    s = _drive_to_review(temp_store, monkeypatch)
    chosen = [m["entryKey"] for m in s["messages"][-1]["plan"]["models"]]
    pool = [m["modelId"] for m in s["messages"][-1]["addPool"]]
    # Remove one ARM (by entryKey), add one model from the pool (by modelId).
    s = pitcrew.advance("sess-1", "edit_models",
                        {"remove": [chosen[0]], "add": [pool[0]]}, "20260630-120010")
    assert s["phase"] == pitcrew.REVIEW_PLAN  # still review — never auto-launches
    new_keys = [m["entryKey"] for m in s["messages"][-1]["plan"]["models"]]
    new_ids = [m["modelId"] for m in s["messages"][-1]["plan"]["models"]]
    assert chosen[0] not in new_keys
    assert pool[0] in new_ids
    assert not s["raceId"]


def test_edit_models_adds_several_models_at_once(temp_store, monkeypatch):
    # The guided review card's Multiselect add-picker sends add:[id1, id2, ...] — the
    # handler must add ALL of them in one edit (not just the first).
    s = _drive_to_review(temp_store, monkeypatch)
    pool = [m["modelId"] for m in s["messages"][-1]["addPool"]]
    assert len(pool) >= 2, "need >=2 pool models to test multi-add"
    before = {m["modelId"] for m in s["messages"][-1]["plan"]["models"]}
    s = pitcrew.advance("sess-1", "edit_models", {"add": pool[:2]}, "20260630-120010")
    after = {m["modelId"] for m in s["messages"][-1]["plan"]["models"]}
    assert pool[0] in after and pool[1] in after
    assert len(after) >= len(before) + 2 - len(before & set(pool[:2]))
    assert s["phase"] == pitcrew.REVIEW_PLAN and not s["raceId"]


def test_edit_models_updates_review_in_place_no_new_bubble(temp_store, monkeypatch):
    # Editing the model set must REFINE the existing review card, not append a new
    # message each time (the jarring "reload" the user reported).
    s = _drive_to_review(temp_store, monkeypatch)
    before = len(s["messages"])
    review_count_before = sum(1 for m in s["messages"] if m.get("reviewPlan"))
    pool = [m["modelId"] for m in s["messages"][-1]["addPool"]]
    s = pitcrew.advance("sess-1", "edit_models", {"add": [pool[0]]}, "20260630-120010")
    # Same message count, still exactly one review bubble (updated in place).
    assert len(s["messages"]) == before
    assert sum(1 for m in s["messages"] if m.get("reviewPlan")) == review_count_before == 1


def test_edit_models_then_approve_launches_edited_set(temp_store, monkeypatch):
    launched = {}

    class _FakeRace:
        race_id = "race-x"
        entries = []

    def _fake_start_race(split_id, race_models, decoding, stamp, **kw):
        launched["models"] = [rm.model_id for rm in race_models]
        return _FakeRace()

    monkeypatch.setattr("app.race.start_race", _fake_start_race)
    s = _drive_to_review(temp_store, monkeypatch)
    pool = [m["modelId"] for m in s["messages"][-1]["addPool"]]
    models = s["messages"][-1]["plan"]["models"]
    # Pick a victim whose modelId appears as exactly ONE arm, so removing that arm's
    # entry key truly removes the model (a model with a DoRA A/B arm has two arms sharing
    # a modelId — removing one leaves the other, which is correct behavior).
    id_counts: dict[str, int] = {}
    for m in models:
        id_counts[m["modelId"]] = id_counts.get(m["modelId"], 0) + 1
    victim = next(m for m in models if id_counts[m["modelId"]] == 1)
    victim_key, victim_id = victim["entryKey"], victim["modelId"]
    pitcrew.advance("sess-1", "edit_models", {"remove": [victim_key], "add": [pool[0]]}, "20260630-120010")
    pitcrew.advance("sess-1", "approve", {}, "20260630-120011")
    # The launched set reflects the EDIT, not the original proposal.
    assert pool[0] in launched["models"]
    assert victim_id not in launched["models"]


def test_emptying_the_plan_blocks_until_a_model_is_added(temp_store, monkeypatch):
    s = _drive_to_review(temp_store, monkeypatch)
    chosen = [m["entryKey"] for m in s["messages"][-1]["plan"]["models"]]
    s = pitcrew.advance("sess-1", "edit_models", {"remove": chosen}, "20260630-120010")
    # No models left → stays in review with a nudge, not a launchable empty plan.
    assert s["phase"] == pitcrew.REVIEW_PLAN
    assert any("no models left" in m.get("text", "").lower() for m in s["messages"])


def test_out_of_order_action_reprompts_not_silent(temp_store, monkeypatch):
    # Firing an action that doesn't match the current phase must ALWAYS produce a
    # response (re-prompt), never a silent no-op where the user sees nothing happen.
    pitcrew.start_session("oo", "20260630-120000")
    for bad in ("approve", "effort", "use_dataset", "confirm", "edit_models"):
        before = len(pitcrew.get_session("oo")["messages"])
        s = pitcrew.advance("oo", bad, {"effort": "easy", "splitId": "x"}, "20260630-120001")
        assert len(s["messages"]) > before, f"'{bad}' at collect_goal produced no message"
        assert s["messages"][-1].get("collectGoal"), "re-prompt should re-render the goal input"


def test_plan_arms_have_unique_entry_keys_and_distinct_labels(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    _make_sft_split(temp_store, "sft-arms", n_train=2000)
    pitcrew.start_session("ak", "20260630-120000")
    pitcrew.advance("ak", "goal", {"goal": "classify"}, "20260630-120001")
    pitcrew.advance("ak", "use_dataset", {"splitId": "sft-arms"}, "20260630-120002")
    pitcrew.advance("ak", "confirm", {}, "20260630-120003")
    s = pitcrew.advance("ak", "effort", {"effort": "huge"}, "20260630-120004")
    models = s["messages"][-1]["plan"]["models"]
    keys = [m["entryKey"] for m in models]
    assert len(keys) == len(set(keys)), f"duplicate entryKey: {keys}"
    # Where a modelId repeats across arms (plain/DoRA/full), labels must differ.
    from collections import defaultdict
    by_id = defaultdict(list)
    for m in models:
        by_id[m["modelId"]].append(m["label"])
    for mid, labels in by_id.items():
        assert len(set(labels)) == len(labels), f"{mid} arms share a label: {labels}"


def test_remove_one_arm_keeps_other_arms_of_same_model(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    _make_sft_split(temp_store, "sft-rm", n_train=2000)
    pitcrew.start_session("rm", "20260630-120000")
    pitcrew.advance("rm", "goal", {"goal": "classify"}, "20260630-120001")
    pitcrew.advance("rm", "use_dataset", {"splitId": "sft-rm"}, "20260630-120002")
    pitcrew.advance("rm", "confirm", {}, "20260630-120003")
    s = pitcrew.advance("rm", "effort", {"effort": "huge"}, "20260630-120004")
    models = s["messages"][-1]["plan"]["models"]
    # Find a modelId with multiple arms; remove ONE arm and assert the others survive.
    from collections import defaultdict
    arms = defaultdict(list)
    for m in models:
        arms[m["modelId"]].append(m["entryKey"])
    multi = next((mid for mid, ks in arms.items() if len(ks) > 1), None)
    if multi:  # huge tier adds a DoRA/full arm on the small model
        victim = arms[multi][0]
        s2 = pitcrew.advance("rm", "edit_models", {"remove": [victim]}, "20260630-120005")
        surviving = [m["entryKey"] for m in s2["messages"][-1]["plan"]["models"]]
        assert victim not in surviving
        assert any(k in surviving for k in arms[multi][1:]), "removing one arm dropped the model's other arms"


def test_big_file_reports_true_row_count(temp_store, monkeypatch):
    # A large SFT file must report its TRUE row count, not the 5000-row profiler cap.
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    import json as _json
    store = temp_store.get_store()
    wd = store.workdir("runs", "big")
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": "billing"}]} for i in range(8000)]
    (wd / "train.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(_json.dumps(r) for r in rows[:50]) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(_json.dumps({"name": "big", "shape": "sft", "trainRows": 8000}), encoding="utf-8")
    store.commit("runs", "big")
    pitcrew.start_session("bf", "20260630-120000")
    pitcrew.advance("bf", "goal", {"goal": "classify"}, "20260630-120001")
    s = pitcrew.advance("bf", "use_dataset", {"splitId": "big"}, "20260630-120002")
    summary = next((m["taskSummary"] for m in s["messages"] if m.get("confirmTask")), {})
    assert summary.get("rows", 0) >= 8000, f"under-reported rows: {summary.get('rows')} (5000-cap bug)"


def test_edit_goal_after_dataset_unlinks_but_keeps_it(temp_store, monkeypatch):
    # THE lifecycle question: editing the goal after a dataset was created must
    # UNLINK the dataset from the session but KEEP it on disk (reusable / may back a
    # race) — never orphan-delete it.
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    _make_sft_split(temp_store, "eg", n_train=1500)
    pitcrew.start_session("editg", "20260630-120000")
    pitcrew.advance("editg", "goal", {"goal": "classify tickets"}, "20260630-120001")
    s = pitcrew.advance("editg", "use_dataset", {"splitId": "eg"}, "20260630-120002")
    assert s["splitId"] == "eg"
    idx = next(i for i, m in enumerate(s["messages"]) if m.get("editKind") == "goal")
    s = pitcrew.edit_message("editg", idx, "detect review sentiment instead", "20260630-120003")
    assert s["splitId"] == ""                    # unlinked
    assert s["phase"] == pitcrew.AWAIT_DATA       # rewound to the data step
    assert s["goal"] == "detect review sentiment instead"
    from app.storage import split_dir
    assert split_dir("eg") is not None            # dataset KEPT on disk


def test_edit_blocked_after_launch(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    monkeypatch.setattr("app.race.start_race",
                        lambda *a, **k: type("R", (), {"race_id": "r1", "entries": []})())
    _make_sft_split(temp_store, "el", n_train=1000)
    pitcrew.start_session("editl", "20260630-120000")
    pitcrew.advance("editl", "goal", {"goal": "classify"}, "20260630-120001")
    pitcrew.advance("editl", "use_dataset", {"splitId": "el"}, "20260630-120002")
    pitcrew.advance("editl", "confirm", {}, "20260630-120003")
    pitcrew.advance("editl", "effort", {"effort": "easy"}, "20260630-120004")
    s = pitcrew.advance("editl", "approve", {}, "20260630-120005")
    assert s["phase"] == pitcrew.LAUNCHED
    idx = next(i for i, m in enumerate(s["messages"]) if m.get("editKind") == "goal")
    s = pitcrew.edit_message("editl", idx, "different goal", "20260630-120006")
    assert s["phase"] == pitcrew.LAUNCHED and s["raceId"] == "r1"  # launch untouched


def test_edit_correction_keeps_dataset_linked(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    _make_sft_split(temp_store, "ec", n_train=1500)
    pitcrew.start_session("editc", "20260630-120000")
    pitcrew.advance("editc", "goal", {"goal": "classify"}, "20260630-120001")
    pitcrew.advance("editc", "use_dataset", {"splitId": "ec"}, "20260630-120002")
    s = pitcrew.advance("editc", "confirm", {"correction": "3 classes"}, "20260630-120003")
    idx = next(i for i, m in enumerate(s["messages"]) if m.get("editKind") == "correction")
    s = pitcrew.edit_message("editc", idx, "actually 5 classes", "20260630-120004")
    # A correction lives after the data step → dataset stays linked.
    assert s["splitId"] == "ec"
    assert s["phase"] == pitcrew.CHOOSE_EFFORT


def test_edit_message_rejects_stale_version(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    _make_sft_split(temp_store, "ev2", n_train=1000)
    pitcrew.start_session("editv", "20260630-120000")
    s = pitcrew.advance("editv", "goal", {"goal": "classify"}, "20260630-120001")
    idx = next(i for i, m in enumerate(s["messages"]) if m.get("editKind") == "goal")
    with pytest.raises(ValueError, match="another tab"):
        pitcrew.edit_message("editv", idx, "new goal", "20260630-120002", expected_version=0)


def test_refresh_race_bumps_version_so_stale_action_is_rejected(temp_store, monkeypatch):
    # A completed race reconciled by a GET must bump version, so a stale tab's next
    # advance() 409s instead of clobbering the finished state.
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    _make_sft_split(temp_store, "rr", n_train=1000)

    class _DoneRace:
        race_id = "rdone"
        class _E:
            state = "done"
        entries = [_E()]
    monkeypatch.setattr("app.race.start_race", lambda *a, **k: _DoneRace())
    monkeypatch.setattr("app.race.reconcile_race", lambda rid: _DoneRace())
    monkeypatch.setattr("app.race.rank_entries",
                        lambda race: [{"isWinner": True, "model_id": "qwen2.5-1.5b",
                                       "model_display": "Qwen2.5 1.5B", "state": "done"}])
    monkeypatch.setattr("app.race.TERMINAL", {"done", "failed"}, raising=False)
    pitcrew.start_session("rrs", "20260630-120000")
    pitcrew.advance("rrs", "goal", {"goal": "x"}, "20260630-120001")
    pitcrew.advance("rrs", "use_dataset", {"splitId": "rr"}, "20260630-120002")
    pitcrew.advance("rrs", "confirm", {}, "20260630-120003")
    pitcrew.advance("rrs", "effort", {"effort": "easy"}, "20260630-120004")
    s = pitcrew.advance("rrs", "approve", {}, "20260630-120005")
    v_before = s["version"]
    # A GET reconciles → race is DONE → version must bump.
    s2 = pitcrew.get_session("rrs", "20260630-120006")
    assert s2["phase"] == pitcrew.DONE
    assert s2["version"] > v_before


def test_below_floor_dataset_blocked_at_data_step(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    _make_sft_split(temp_store, "tiny", n_train=5)   # below MIN_USABLE_ROWS
    pitcrew.start_session("tf", "20260630-120000")
    pitcrew.advance("tf", "goal", {"goal": "classify"}, "20260630-120001")
    s = pitcrew.advance("tf", "use_dataset", {"splitId": "tiny"}, "20260630-120002")
    assert s["phase"] == pitcrew.AWAIT_DATA                     # bounced, not planned
    assert any("usable example" in m.get("text", "") for m in s["messages"])


def test_race_name_is_clean_goal_without_dataset(temp_store):
    # Race name is the cleaned GOAL only — the dataset is NOT appended (the Races table
    # already has a dataset column, so repeating it in the title is redundant).
    assert pitcrew._race_name(
        {"goal": "i want to classify the new support tickets by topic", "datasetName": "t"}
    ) == "Classify the new support tickets by topic"
    assert pitcrew._race_name(
        {"goal": "detect sentiment", "datasetName": "reviews"}
    ) == "Detect sentiment"
    # No usable goal → a stamped generic (still no dataset name).
    empty = pitcrew._race_name({"goal": "", "datasetName": "my-data", "createdAt": "20260701-120000"})
    assert empty.startswith("Guided fine-tune ·") and "my-data" not in empty
    # Never leaks the raw filler opener, and never appends the dataset name.
    n = pitcrew._race_name({"goal": "please build a model to summarize articles", "datasetName": "ds"})
    assert n == "Summarize articles"
    assert "ds" not in n


def test_launch_failure_rerenders_review_card(temp_store, monkeypatch):
    from app.limits import LimitExceeded
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    monkeypatch.setattr("app.secrets.get_hf_token", lambda: "", raising=False)
    monkeypatch.setattr("app.race.start_race",
                        lambda *a, **k: (_ for _ in ()).throw(LimitExceeded("platform busy")))
    _make_sft_split(temp_store, "lf", n_train=1000)
    pitcrew.start_session("lfs", "20260630-120000")
    pitcrew.advance("lfs", "goal", {"goal": "x"}, "20260630-120001")
    pitcrew.advance("lfs", "use_dataset", {"splitId": "lf"}, "20260630-120002")
    pitcrew.advance("lfs", "confirm", {}, "20260630-120003")
    pitcrew.advance("lfs", "effort", {"effort": "easy"}, "20260630-120004")
    s = pitcrew.advance("lfs", "approve", {}, "20260630-120005")
    # Blocked → stays in review AND the last message is an actionable plan card.
    assert s["phase"] == pitcrew.REVIEW_PLAN
    assert s["messages"][-1].get("reviewPlan")   # buttons come back, not a dead-end
    assert not s["raceId"]


def test_edit_noop_on_non_editable_or_bad_index(temp_store, monkeypatch):
    monkeypatch.setattr(pitcrew, "_narrate", lambda p, f: f)
    pitcrew.start_session("en", "20260630-120000")
    s = pitcrew.advance("en", "goal", {"goal": "x"}, "20260630-120001")
    # Editing the assistant greeting (index 0, not editable) is a no-op.
    before = len(s["messages"])
    s = pitcrew.edit_message("en", 0, "hack", "20260630-120002")
    assert len(s["messages"]) == before
    # Out-of-range index is a no-op.
    s = pitcrew.edit_message("en", 999, "x", "20260630-120003")
    assert s is not None


def test_rl_dataset_declined_in_guided_flow(temp_store, monkeypatch):
    # An rlvr-shaped dataset should be politely declined at the confirm step, not
    # advanced into a plan.
    store = temp_store.get_store()
    wd = store.workdir("runs", "rlvr-ds")
    rows = [{"messages": [{"role": "user", "content": f"{i}+{i}?"}], "ground_truth": str(2 * i)}
            for i in range(200)]
    (wd / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (wd / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:20]) + "\n", encoding="utf-8")
    (wd / "dataset_info.json").write_text("{}", encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps({"name": "rlvr-ds", "shape": "rlvr", "trainRows": 200}),
                                  encoding="utf-8")
    store.commit("runs", "rlvr-ds")

    pitcrew.start_session("sess-rl", "20260630-120000")
    pitcrew.advance("sess-rl", "goal", {"goal": "solve math"}, "20260630-120001")
    s = pitcrew.advance("sess-rl", "use_dataset", {"splitId": "rlvr-ds"}, "20260630-120002")
    # Declined → bounced back to AWAIT_DATA with a clear message, not CONFIRM_TASK.
    assert s["phase"] == pitcrew.AWAIT_DATA
    assert any("reward-based" in m.get("text", "") for m in s["messages"])
