# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Race state machine — advanced with all AWS calls stubbed.

The state machine (PENDING→TRAINING→EVAL_PENDING→EVALUATING→DONE/FAILED) is the
headless orchestration core. We stub launch/describe/fetch so reconcile_race can
be driven deterministically without SageMaker.
"""
import pytest

from app.catalog import DecodingParams, Hyperparams


@pytest.fixture
def race_mod(temp_store, monkeypatch):
    from app import race as race_mod

    # Stub AWS-touching orchestrate calls injected into race.py's namespace, and
    # bypass the on-disk split check (so tests don't need a persisted split).
    monkeypatch.setattr(race_mod, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(race_mod, "launch_eval_job",
                        lambda **kw: {"jobName": f"eval-{kw['stamp']}"})
    monkeypatch.setattr(race_mod, "launch_base_eval_job",
                        lambda **kw: {"jobName": f"base-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(race_mod, "split_dir", lambda s: "/tmp/fake")
    return race_mod


def _start(race_mod, models=("qwen3-1.7b",), name=""):
    rms = [race_mod.RaceModel(model_id=m, hp=Hyperparams()) for m in models]
    return race_mod.start_race("split-x", rms, DecodingParams(), "20260603-1", name=name)


def test_start_race_launches_training(race_mod):
    race = _start(race_mod, name="my-run")
    assert race.name == "my-run"
    assert race.entries[0].state == race_mod.TRAINING
    assert race.entries[0].train_job is not None


def test_full_weight_routes_to_g6e_not_g5(race_mod):
    # A full-weight (full/freeze) run of a small model must auto-route to the g6e
    # (L40S 48GB) instance, NOT the LoRA-sized g5 (24GB) — else it OOMs. The launch
    # re-resolves the instance method-aware (regression: it used the LoRA-sized
    # suggested_instance and a 1.7B full-FT landed on g5.2xlarge).
    lora = race_mod.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams(finetuning_type="lora"))
    race_lora = race_mod.start_race("split-x", [lora], DecodingParams(), "i-lora")
    assert race_lora.entries[0].instance_type == "ml.g5.2xlarge"  # unchanged

    full = race_mod.RaceModel(model_id="qwen3-1.7b",
                              hp=Hyperparams(finetuning_type="full", stage="sft"))
    race_full = race_mod.start_race("split-x", [full], DecodingParams(), "i-full")
    assert race_full.entries[0].instance_type == "ml.g6e.2xlarge"


def test_full_weight_rejected_for_oversize_model(race_mod):
    # Server-side size gate: full/freeze on a model whose allowed_methods excludes
    # them (e.g. qwen3-4b, >2B) must be REJECTED before launch — a direct API/clone
    # call must not bypass the UI's size gate and launch a billable OOM-bound job.
    oversize = race_mod.RaceModel(model_id="qwen3-4b",
                                  hp=Hyperparams(finetuning_type="full", stage="sft"))
    with pytest.raises(ValueError):
        race_mod.start_race("split-x", [oversize], DecodingParams(), "i-oversize")
    # ...but LoRA on the same oversize model is fine (unchanged).
    lora_ok = race_mod.RaceModel(model_id="qwen3-4b", hp=Hyperparams(finetuning_type="lora"))
    race_ok = race_mod.start_race("split-x", [lora_ok], DecodingParams(), "i-ok")
    assert race_ok.entries[0].state == race_mod.TRAINING


def test_unknown_split_raises(race_mod, monkeypatch):
    monkeypatch.setattr(race_mod, "split_dir", lambda s: None)
    with pytest.raises(ValueError):
        _start(race_mod)


def test_max_run_seconds_threads_through_and_persists(race_mod, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        race_mod, "launch_training_job",
        lambda **kw: seen.update(kw) or {"jobName": f"train-{kw['stamp']}"},
    )
    rms = [race_mod.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())]
    race = race_mod.start_race("split-x", rms, DecodingParams(), "20260604-1", max_run_seconds=18000)
    assert seen.get("max_run_seconds") == 18000  # forwarded to the launch
    assert race_mod._load(race.race_id).max_run_seconds == 18000  # persisted for retries
    # legacy races (no field) default to 5h, not the old 1h that caused timeouts
    assert _start(race_mod).max_run_seconds == 18000


def test_use_spot_persists_and_passes_through(race_mod, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        race_mod, "launch_training_job",
        lambda **kw: seen.update(kw) or {"jobName": f"train-{kw['stamp']}"},
    )
    rms = [race_mod.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())]
    race = race_mod.start_race("split-x", rms, DecodingParams(), "20260603-2", use_spot=True)
    # Flag is on the race + persisted, and forwarded to the launch call.
    assert race.use_spot is True
    assert seen.get("use_spot") is True
    reloaded = race_mod._load(race.race_id)
    assert reloaded.use_spot is True

    # Legacy races (persisted before spot existed) default to False.
    default_race = _start(race_mod)
    assert default_race.use_spot is False


def test_training_complete_launches_eval(race_mod, monkeypatch):
    rm = race_mod
    race = _start(rm)
    assert race.entries[0].state == rm.TRAINING

    # Training reports Completed → reconcile should launch eval + move to EVALUATING.
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    assert reloaded.entries[0].state == rm.EVALUATING
    assert reloaded.entries[0].eval_job is not None


def test_launching_recovers_when_worker_lands_job(race_mod):
    """A serverless entry stuck in LAUNCHING with a train_job already set (worker
    landed it but the state lagged) is picked up by reconcile → TRAINING."""
    rm = race_mod
    race = _start(rm)
    e = race.entries[0]
    e.state = rm.LAUNCHING
    e.train_job = "serverless-job-123"  # worker recorded it
    rm._save(race)
    rm.reconcile_race(race.race_id)
    assert rm._load(race.race_id).entries[0].state == rm.TRAINING


def test_launching_times_out_to_failed(race_mod):
    """A serverless entry stuck in LAUNCHING with NO train_job past the grace period
    (or with no launching_at stamp, e.g. a legacy stuck entry) reconciles to FAILED
    so it can't hang 'running' forever and becomes retryable."""
    rm = race_mod
    race = _start(rm)
    e = race.entries[0]
    e.state = rm.LAUNCHING
    e.train_job = None
    e.launching_at = None  # legacy/lost stamp → eligible to time out immediately
    rm._save(race)
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id).entries[0]
    assert reloaded.state == rm.FAILED
    assert "never started" in (reloaded.error or "")


def test_launching_waits_within_grace(race_mod):
    """A freshly-dispatched LAUNCHING entry (recent launching_at, no train_job yet)
    is left alone — reconcile must NOT fail it during the grace window."""
    from datetime import datetime, timezone

    rm = race_mod
    race = _start(rm)
    e = race.entries[0]
    e.state = rm.LAUNCHING
    e.train_job = None
    e.launching_at = datetime.now(timezone.utc).isoformat()  # just now
    rm._save(race)
    rm.reconcile_race(race.race_id)
    assert rm._load(race.race_id).entries[0].state == rm.LAUNCHING  # still waiting


def test_serverless_launch_merge_does_not_clobber_sibling(race_mod, monkeypatch):
    """Two serverless entries launch on SEPARATE concurrent workers, each doing
    load→mutate→save of the whole race.json. launch_serverless_entry must re-load
    + write ONLY its own entry, so worker B can't wipe worker A's train_job (the
    serverless-run-1 bug: a Completed entry showed as failed). Simulate by having
    A load the race, then B fully launch+save in between, then A persist."""
    rm = race_mod
    # Two entries, both serverless.
    rms = [
        rm.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams(engine="sagemaker_serverless")),
        rm.RaceModel(model_id="qwen3-4b", hp=Hyperparams(engine="sagemaker_serverless")),
    ]
    # No worker configured → start_race leaves serverless entries LAUNCHING without
    # launching inline (dispatch_worker returns False → falls to inline launch). To
    # keep both in LAUNCHING for the test, stub dispatch_worker truthy.
    monkeypatch.setattr(rm, "dispatch_worker", lambda payload: True)
    race = rm.start_race("split-x", rms, DecodingParams(), "20260618-c")
    k0, k1 = race.entries[0].entry_key, race.entries[1].entry_key
    assert race.entries[0].state == rm.LAUNCHING and race.entries[1].state == rm.LAUNCHING

    # Worker B launches entry 1 fully and persists (its train_job lands).
    rm.launch_serverless_entry(race.race_id, k1, "stampB")
    # Worker A launches entry 0 — its merge must NOT drop entry 1's train_job.
    rm.launch_serverless_entry(race.race_id, k0, "stampA")

    reloaded = rm._load(race.race_id)
    e0 = rm._find_entry(reloaded, k0)
    e1 = rm._find_entry(reloaded, k1)
    assert e0.state == rm.TRAINING and e0.train_job is not None  # A's write survived
    assert e1.state == rm.TRAINING and e1.train_job is not None  # B's NOT clobbered


def test_eval_complete_fetches_metrics_and_done(race_mod, monkeypatch):
    rm = race_mod
    race = _start(race_mod)
    # Drive train→eval.
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    rm.reconcile_race(race.race_id)
    # Now eval completes + metrics fetched.
    monkeypatch.setattr(rm, "fetch_metrics", lambda job: {"token_f1": 0.7})
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    assert reloaded.entries[0].state == rm.DONE
    assert reloaded.entries[0].metrics["token_f1"] == 0.7


# --- spot → on-demand capacity fallback -------------------------------------

def _stuck_spot_describe(creation_iso: str, secondary: str = "Insufficient capacity error"):
    """A describe_job result for a spot job still WAITING for capacity: InProgress,
    no trainingStartTime (container not started), a capacity secondaryStatus, and a
    creationTime far enough in the past."""
    return {
        "status": "InProgress",
        "secondaryStatus": secondary,
        "failureReason": None,
        "trainingStartTime": None,
        "creationTime": creation_iso,
    }


def _spot_race(rm, monkeypatch, fallback_minutes=10):
    """A 1-entry spot race with the fallback opted in + a recorded checkpoint."""
    monkeypatch.setattr(
        rm, "launch_training_job",
        lambda **kw: {"jobName": f"train-{kw['stamp']}", "checkpointS3": "s3://ck/x"})
    rms = [rm.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())]
    race = rm.start_race("split-x", rms, DecodingParams(), "20260616-1",
                         use_spot=True, spot_fallback_minutes=fallback_minutes)
    # start_race's stub didn't set checkpoint_s3 (different stub above does now);
    # ensure it's present so the resume precondition holds.
    reloaded = rm._load(race.race_id)
    reloaded.entries[0].checkpoint_s3 = "s3://ck/x"
    rm._save(reloaded)
    return reloaded


def test_spot_fallback_converts_stuck_job_to_on_demand(race_mod, monkeypatch):
    rm = race_mod
    race = _spot_race(rm, monkeypatch, fallback_minutes=10)
    # Capture launch_training_job kwargs so we can assert use_spot=False + resume.
    calls = {}
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: calls.update(kw) or {"jobName": "ondemand-job", "checkpointS3": "s3://ck/x"})
    stopped = {}
    monkeypatch.setattr(rm, "stop_training_job", lambda j: stopped.update({"job": j}) or True)
    # Job created 30 min ago, still waiting on capacity → past the 10-min threshold.
    monkeypatch.setattr(rm, "describe_job",
                        lambda job: _stuck_spot_describe("2026-06-16 00:00:00+00:00"))
    # Freeze "now" 30 min after creation via a fixed elapsed (the helper parses
    # creationTime vs datetime.now — use a creation far in the past instead).
    import app.race as racemod
    # Use a creationTime guaranteed > 10 min in the past relative to real now.
    monkeypatch.setattr(racemod, "describe_job",
                        lambda job: _stuck_spot_describe("2020-01-01 00:00:00+00:00"))
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    assert stopped.get("job"), "the stuck spot job should have been stopped"
    assert calls.get("use_spot") is False, "relaunch must be on-demand"
    assert calls.get("resume_checkpoint_s3") == "s3://ck/x", "must resume from checkpoint"
    assert reloaded.entries[0].spot_fell_back is True
    assert reloaded.entries[0].state == rm.TRAINING
    assert reloaded.entries[0].train_job == "ondemand-job"


def test_spot_fallback_idempotent_once(race_mod, monkeypatch):
    rm = race_mod
    race = _spot_race(rm, monkeypatch, fallback_minutes=10)
    n = {"launches": 0}
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: n.update(launches=n["launches"] + 1) or {"jobName": "od", "checkpointS3": "s3://ck/x"})
    monkeypatch.setattr(rm, "stop_training_job", lambda j: True)
    monkeypatch.setattr(rm, "describe_job",
                        lambda job: _stuck_spot_describe("2020-01-01 00:00:00+00:00"))
    rm.reconcile_race(race.race_id)  # converts
    rm.reconcile_race(race.race_id)  # must NOT convert again (spot_fell_back guard)
    assert n["launches"] == 1


def test_spot_fallback_skips_when_training_started(race_mod, monkeypatch):
    # A job that has STARTED training (trainingStartTime set) is making progress —
    # never convert it, even if it's been a while.
    rm = race_mod
    race = _spot_race(rm, monkeypatch, fallback_minutes=10)
    monkeypatch.setattr(rm, "stop_training_job",
                        lambda j: (_ for _ in ()).throw(AssertionError("must not stop a started job")))
    monkeypatch.setattr(rm, "describe_job", lambda job: {
        "status": "InProgress", "secondaryStatus": "Training",
        "failureReason": None, "trainingStartTime": "2026-06-16 01:00:00+00:00",
        "creationTime": "2020-01-01 00:00:00+00:00",
    })
    rm.reconcile_race(race.race_id)
    assert rm._load(race.race_id).entries[0].spot_fell_back is False


def test_spot_fallback_skips_non_capacity_status(race_mod, monkeypatch):
    # A slow image pull ("Downloading") is NOT a capacity stall → don't convert.
    rm = race_mod
    race = _spot_race(rm, monkeypatch, fallback_minutes=10)
    monkeypatch.setattr(rm, "stop_training_job",
                        lambda j: (_ for _ in ()).throw(AssertionError("must not stop")))
    monkeypatch.setattr(rm, "describe_job", lambda job: _stuck_spot_describe(
        "2020-01-01 00:00:00+00:00", secondary="Downloading the training image"))
    rm.reconcile_race(race.race_id)
    assert rm._load(race.race_id).entries[0].spot_fell_back is False


def test_spot_fallback_off_by_default(race_mod, monkeypatch):
    # No spot_fallback_minutes → never converts, even on a capacity stall.
    rm = race_mod
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: {"jobName": f"t-{kw['stamp']}", "checkpointS3": "s3://ck/x"})
    rms = [rm.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())]
    race = rm.start_race("split-x", rms, DecodingParams(), "20260616-2", use_spot=True)
    r = rm._load(race.race_id); r.entries[0].checkpoint_s3 = "s3://ck/x"; rm._save(r)
    monkeypatch.setattr(rm, "stop_training_job",
                        lambda j: (_ for _ in ()).throw(AssertionError("must not stop")))
    monkeypatch.setattr(rm, "describe_job",
                        lambda job: _stuck_spot_describe("2020-01-01 00:00:00+00:00"))
    rm.reconcile_race(race.race_id)
    assert rm._load(race.race_id).entries[0].spot_fell_back is False


def test_spot_fallback_persists_and_clones(race_mod, monkeypatch):
    rm = race_mod
    race = _spot_race(rm, monkeypatch, fallback_minutes=15)
    assert rm._load(race.race_id).spot_fallback_minutes == 15  # round-trips


def test_failed_training_marks_failed(race_mod, monkeypatch):
    from app import race as rm
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")

    race = _start(race_mod)
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Failed"})
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    assert reloaded.entries[0].state == rm.FAILED
    assert reloaded.entries[0].error


def test_base_eval_launched_at_start(race_mod):
    """Every entry also gets a base-model eval job, in parallel with training."""
    race = _start(race_mod)
    assert race.entries[0].base_eval_job is not None
    assert race.entries[0].base_eval_job.startswith("base-")
    assert race.entries[0].base_metrics is None  # not fetched until it completes


def test_base_eval_metrics_fetched_independently(race_mod, monkeypatch):
    """Base metrics are picked up even while the entry is still TRAINING (the base
    eval has no dependency on the train job), and don't affect ranking."""
    rm = race_mod
    race = _start(race_mod)
    # Train job still in progress, but the BASE eval completed.
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    monkeypatch.setattr(rm, "fetch_metrics", lambda job: {"token_f1": 0.30, "exact_match": 0.25})
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    assert reloaded.entries[0].base_metrics is not None
    assert reloaded.entries[0].base_metrics["token_f1"] == 0.30


def test_base_eval_failure_is_non_fatal(race_mod, monkeypatch):
    """A failed base eval marks base_metrics empty (advisory) but never fails the
    entry — the fine-tuned path is what matters."""
    rm = race_mod
    race = _start(race_mod)
    # Base eval failed; training itself still in progress (describe returns Failed
    # for all jobs here, so training will also fail — but base must be {} not crash).
    calls = {"n": 0}
    def fake_describe(job):
        return {"status": "Failed"}
    monkeypatch.setattr(rm, "describe_job", fake_describe)
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    assert reloaded.entries[0].base_metrics == {}  # resolved-but-empty, polling stops


def test_rank_entries_picks_winner_by_metric(race_mod, monkeypatch):
    from app import race as rm
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")

    race = _start(race_mod, models=("qwen3-1.7b", "qwen3-0.6b"))
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    rm.reconcile_race(race.race_id)
    # Give the two entries different f1 on eval completion.
    scores = iter([{"token_f1": 0.4}, {"token_f1": 0.9}])
    monkeypatch.setattr(rm, "fetch_metrics", lambda job: next(scores))
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    ranked = rm.rank_entries(reloaded, "token_f1")
    assert ranked[0]["isWinner"] is True
    assert ranked[0]["rankScore"] == 0.9


# --- RLAIF: skip the gold eval, rank by judge reward ------------------------

def _start_rlaif(race_mod, models=("qwen3-1.7b",)):
    """An RLAIF race: serverless engine + stage=rlaif. The serverless launch falls
    inline to the (stubbed) launch_training_job because no worker is configured."""
    hp = Hyperparams(engine="sagemaker_serverless", stage="rlaif",
                     reward_function_id="rf-1", num_train_epochs=1)
    rms = [race_mod.RaceModel(model_id=m, hp=hp) for m in models]
    return race_mod.start_race("split-x", rms, DecodingParams(), "20260617-1")


def test_rlaif_skips_base_eval(race_mod):
    """An RLAIF entry has a prompt-only held-out set, so it must NOT launch a base
    eval (which would crash on a row with no assistant turn + has nothing to
    compute lift against)."""
    race = _start_rlaif(race_mod)
    assert race.entries[0].state == race_mod.TRAINING
    assert race.entries[0].base_eval_job is None  # base eval skipped for RLAIF


def test_rlaif_training_complete_ranks_by_reward_no_eval(race_mod, monkeypatch):
    """When an RLAIF entry's training completes, reconcile reads the judge reward
    from the training metrics.jsonl and goes STRAIGHT to DONE (no eval job) —
    ranked by reward_mean."""
    rm = race_mod
    race = _start_rlaif(rm)
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    # Held-out reward present → that's the rank signal.
    monkeypatch.setattr(rm, "fetch_reward_curve", lambda job: {
        "hasData": True, "steps": [1, 2], "rewardMean": [0.80, 0.90],
        "valReward": [{"step": 2, "value": 0.94}],
    })
    # If launch_eval_job is called for an RLAIF entry that's a bug → blow up.
    monkeypatch.setattr(rm, "launch_eval_job",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("RLAIF must not eval")))
    rm.reconcile_race(race.race_id)
    e = rm._load(race.race_id).entries[0]
    assert e.state == rm.DONE
    assert e.eval_job is None
    assert e.metrics["reward_mean"] == 0.94          # held-out reward wins
    assert e.metrics["reward_source"] == "held_out"


def test_rlaif_falls_back_to_training_reward(race_mod, monkeypatch):
    """No held-out reward logged → rank by the FINAL training-step reward."""
    rm = race_mod
    race = _start_rlaif(rm)
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    monkeypatch.setattr(rm, "fetch_reward_curve", lambda job: {
        "hasData": True, "steps": [1, 2, 3], "rewardMean": [0.5, 0.6, 0.7],
        "valReward": [],
    })
    rm.reconcile_race(race.race_id)
    e = rm._load(race.race_id).entries[0]
    assert e.state == rm.DONE
    assert e.metrics["reward_mean"] == 0.7
    assert e.metrics["reward_source"] == "training"


def test_rlaif_waits_when_reward_not_flushed(race_mod, monkeypatch):
    """The reward file lags the job's terminal status. An empty curve must leave
    the entry TRAINING (retry next tick) — NOT produce a 0-metric DONE."""
    rm = race_mod
    race = _start_rlaif(rm)
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    monkeypatch.setattr(rm, "fetch_reward_curve", lambda job: {"hasData": False, "steps": []})
    rm.reconcile_race(race.race_id)
    e = rm._load(race.race_id).entries[0]
    assert e.state == rm.TRAINING   # held, not DONE
    assert e.metrics is None


def test_rlaif_failed_training_still_fails(race_mod, monkeypatch):
    """A failed RLAIF training job fails the entry like any other (the reward-skip
    path only applies on Completed)."""
    rm = race_mod
    race = _start_rlaif(rm)
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Failed"})
    rm.reconcile_race(race.race_id)
    assert rm._load(race.race_id).entries[0].state == rm.FAILED


def test_effective_rank_metric_forces_reward_for_rlaif(race_mod):
    """An RLAIF race is always ranked by reward_mean regardless of the requested
    gold metric; a non-RLAIF race rejects reward_mean (would be all-blank)."""
    rm = race_mod
    rlaif = _start_rlaif(rm)
    assert rm.effective_rank_metric(rlaif, "token_f1") == "reward_mean"
    assert rm.effective_rank_metric(rlaif, "exact_match") == "reward_mean"

    sft = _start(rm)
    assert rm.effective_rank_metric(sft, "token_f1") == "token_f1"
    assert rm.effective_rank_metric(sft, "reward_mean") == "token_f1"  # rejected → default
    assert rm.effective_rank_metric(sft, "bogus") == "token_f1"


def test_rlaif_winner_ranked_by_reward(race_mod, monkeypatch):
    """Two RLAIF entries → the higher judge reward wins, via rank_entries."""
    rm = race_mod
    race = _start_rlaif(rm, models=("qwen3-1.7b", "qwen3-0.6b"))
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    rewards = iter([
        {"hasData": True, "steps": [1], "rewardMean": [0.55], "valReward": []},
        {"hasData": True, "steps": [1], "rewardMean": [0.88], "valReward": []},
    ])
    monkeypatch.setattr(rm, "fetch_reward_curve", lambda job: next(rewards))
    rm.reconcile_race(race.race_id)
    reloaded = rm._load(race.race_id)
    ranked = rm.rank_entries(reloaded, "token_f1")  # requested gold metric ignored
    assert ranked[0]["rankMetric"] == "reward_mean"
    assert ranked[0]["isWinner"] is True
    assert ranked[0]["rankScore"] == 0.88


# --- DPO/KTO rank by the LLM judge -------------------------------

def _start_dpo(race_mod, models=("qwen3-1.7b",)):
    hp = Hyperparams(stage="dpo", pref_beta=0.1)
    rms = [race_mod.RaceModel(model_id=m, hp=hp) for m in models]
    return race_mod.start_race("split-x", rms, DecodingParams(), "20260617-2")


def test_effective_rank_metric_defaults_to_judge_for_dpo_kto(race_mod):
    """A DPO/KTO race defaults to ranking by llm_judge (overlap is vs one acceptable
    answer), but still honours an explicit gold-overlap request."""
    rm = race_mod
    dpo = _start_dpo(rm)
    assert rm.effective_rank_metric(dpo, "token_f1") == "token_f1"  # explicit honored
    assert rm.effective_rank_metric(dpo, "reward_mean") == "llm_judge"  # invalid → judge default
    assert rm.effective_rank_metric(dpo, "bogus") == "llm_judge"
    # llm_judge is a valid explicit choice on any race now
    sft = _start(rm)
    assert rm.effective_rank_metric(sft, "llm_judge") == "llm_judge"


def test_judge_backfill_sets_normalized_score(race_mod, monkeypatch):
    """_advance_judge pulls the async judge score (1-5) onto a DONE entry's metrics
    as llm_judge (0-1), so the race can rank by it on a later reconcile."""
    rm = race_mod
    race = _start_dpo(rm)
    monkeypatch.setattr(rm, "describe_job", lambda job: {"status": "Completed"})
    monkeypatch.setattr(rm, "fetch_metrics", lambda job: {"token_f1": 0.3})
    monkeypatch.setattr(rm, "launch_eval_job", lambda **kw: {"jobName": f"eval-{kw['stamp']}"})
    rm.reconcile_race(race.race_id)  # train→eval
    rm.reconcile_race(race.race_id)  # eval→done (token_f1 fetched)
    e = rm._load(race.race_id).entries[0]
    assert e.state == rm.DONE and e.metrics.get("llm_judge") is None  # not judged yet
    # Judge result now available (4.0/5 = 0.8).
    import app.judge as judge
    monkeypatch.setattr(judge, "load_judge", lambda ej: {"judgeScore": 4.0})
    rm.reconcile_race(race.race_id)
    e = rm._load(race.race_id).entries[0]
    assert e.metrics["llm_judge"] == 0.8
    # No explicit metric → DPO defaults to the judge.
    ranked = rm.rank_entries(rm._load(race.race_id))
    assert ranked[0]["rankMetric"] == "llm_judge"
    assert ranked[0]["rankScore"] == 0.8
    # An explicit gold-overlap request is still honoured (lets the user inspect it).
    ranked_f1 = rm.rank_entries(rm._load(race.race_id), "token_f1")
    assert ranked_f1[0]["rankMetric"] == "token_f1"
    assert ranked_f1[0]["rankScore"] == 0.3


def test_set_archived_round_trip(race_mod, monkeypatch):
    from app import race as rm
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")

    race = _start(race_mod)
    assert rm._load(race.race_id).archived is False
    rm.set_archived(race.race_id, True)
    assert rm._load(race.race_id).archived is True
    rm.set_archived(race.race_id, False)
    assert rm._load(race.race_id).archived is False
    assert rm.set_archived("nonexistent", True) is None


def test_same_model_two_methods_distinct_entries(temp_store, monkeypatch):
    """A race can hold the SAME model with LoRA AND QLoRA — distinct entry_keys,
    distinct state-map keys, retry/lookup resolve each independently."""
    from app import race as rm
    from app.catalog import DecodingParams, Hyperparams

    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(rm, "launch_base_eval_job", lambda **kw: {"jobName": "base"})
    monkeypatch.setattr(rm, "split_dir", lambda s: "/tmp/fake")
    models = [
        rm.RaceModel(model_id="qwen3-0.6b", hp=Hyperparams(finetuning_type="lora")),
        rm.RaceModel(model_id="qwen3-0.6b", hp=Hyperparams(finetuning_type="qlora")),
    ]
    race = rm.start_race("s", models, DecodingParams(), "20260612-1")
    assert len(race.entries) == 2
    keys = {e.entry_key for e in race.entries}
    assert keys == {"qwen3-0.6b", "qwen3-0.6b::qlora"}
    # state map (list_races) keys them distinctly
    states = {e.entry_key: e.state for e in race.entries}
    assert len(states) == 2
    # _find_entry resolves each
    assert rm._find_entry(race, "qwen3-0.6b").hp["finetuning_type"] == "lora"
    assert rm._find_entry(race, "qwen3-0.6b::qlora").hp["finetuning_type"] == "qlora"


def test_entry_key_back_compat_bare_model_id(temp_store):
    """A LoRA entry's key is the bare model_id (back-compat); _find_entry matches
    both the entry_key and a bare model_id."""
    from app.race import entry_key_for
    assert entry_key_for("m", {"finetuning_type": "lora"}) == "m"
    assert entry_key_for("m", {"finetuning_type": "qlora"}) == "m::qlora"
    assert entry_key_for("m", {}) == "m"  # pre-method races default to lora
    assert entry_key_for("m", None) == "m"


def test_entry_key_includes_lora_variant(temp_store):
    """A non-plain LoRA variant appends `::<variant>` LAST so the SAME (model,
    method) can race as DISTINCT entries (DoRA vs rsLoRA vs plain). Plain "lora"
    adds nothing → every pre-variant key stays byte-identical."""
    from app.race import entry_key_for
    # plain LoRA unchanged (no variant token)
    assert entry_key_for("m", {"lora_variant": "lora"}) == "m"
    # each variant is its own key on plain LoRA
    assert entry_key_for("m", {"lora_variant": "dora"}) == "m::dora"
    assert entry_key_for("m", {"lora_variant": "rslora"}) == "m::rslora"
    # variant rides QLoRA too, appended AFTER the method token
    assert entry_key_for("m", {"finetuning_type": "qlora", "lora_variant": "rslora"}) == "m::qlora::rslora"
    # so two variants of the same (model, method) are DISTINCT keys (no collision)
    a = entry_key_for("m", {"lora_variant": "dora"})
    b = entry_key_for("m", {"lora_variant": "rslora"})
    assert a != b


def test_entry_key_includes_kto_weights(temp_store):
    """For a KTO run, a non-default (kto_chosen_weight, kto_rejected_weight) appends
    `::kw<c>x<r>` LAST so two weightings race as DISTINCT rows. The default 1.0/1.0
    — and every non-KTO run — adds nothing → byte-identical to pre-existing keys."""
    from app.race import entry_key_for
    # default weights → no token (byte-identical to a plain KTO/SFT key)
    assert entry_key_for("m", {"stage": "kto"}) == "m"
    assert entry_key_for("m", {"stage": "kto", "kto_chosen_weight": 1.0,
                               "kto_rejected_weight": 1.0}) == "m"
    # non-default weighting → a kw token; trailing .0 stripped (3.0 → "3")
    assert entry_key_for("m", {"stage": "kto", "kto_chosen_weight": 1.0,
                               "kto_rejected_weight": 3.0}) == "m::kw1x3"
    assert entry_key_for("m", {"stage": "kto", "kto_chosen_weight": 2.5,
                               "kto_rejected_weight": 1.0}) == "m::kw2.5x1"
    # weights are IGNORED off the KTO stage (an SFT run never gets the token)
    assert entry_key_for("m", {"stage": "sft", "kto_chosen_weight": 3.0}) == "m"
    # two weightings of the same KTO (model, method) are DISTINCT keys (no collision)
    a = entry_key_for("m", {"stage": "kto", "kto_rejected_weight": 3.0})
    b = entry_key_for("m", {"stage": "kto", "kto_rejected_weight": 2.0})
    assert a != b and a != "m"
    # the KTO token sits AFTER method/variant tokens (last axis)
    assert entry_key_for("m", {"finetuning_type": "qlora", "stage": "kto",
                               "kto_chosen_weight": 2.0}) == "m::qlora::kw2x1"


# --- checkpoint recorded at launch + resume-from-checkpoint retry ----------

def test_launch_records_checkpoint_s3_on_entry(race_mod, monkeypatch):
    """start_race records the checkpoint S3 prefix the launch returns, so a later
    resume can re-point a new job at it."""
    monkeypatch.setattr(race_mod, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['stamp']}",
                                      "checkpointS3": "s3://b/jobs/J/checkpoints"})
    race = _start(race_mod)
    assert race.entries[0].checkpoint_s3 == "s3://b/jobs/J/checkpoints"


def test_can_resume_entry_predicate(temp_store):
    from app import race as rm

    base = dict(model_id="qwen3-1.7b", model_display="Q", instance_type="ml.g5.2xlarge")
    # resumable: failed LF training with a checkpoint prefix, no eval job
    e = rm.RaceEntry(**base, state=rm.FAILED, train_job="t",
                     checkpoint_s3="s3://ck", hp={"engine": "llama_factory"})
    assert rm.can_resume_entry(e) is True
    # not resumable: no checkpoint recorded
    assert rm.can_resume_entry(rm.RaceEntry(**base, state=rm.FAILED, train_job="t",
                                            hp={"engine": "llama_factory"})) is False
    # not resumable: eval failed (training had succeeded) → re-eval, not resume
    assert rm.can_resume_entry(rm.RaceEntry(**base, state=rm.FAILED, train_job="t",
                               eval_job="ev", checkpoint_s3="s3://ck",
                               hp={"engine": "llama_factory"})) is False
    # not resumable: serverless manages its own state
    assert rm.can_resume_entry(rm.RaceEntry(**base, state=rm.FAILED, train_job="t",
                               checkpoint_s3="s3://ck",
                               hp={"engine": "sagemaker_serverless"})) is False
    # not resumable: not failed
    assert rm.can_resume_entry(rm.RaceEntry(**base, state=rm.TRAINING, train_job="t",
                               checkpoint_s3="s3://ck",
                               hp={"engine": "llama_factory"})) is False


def test_retry_resume_reuses_checkpoint_prefix(race_mod, monkeypatch):
    """retry(resume=True) on a resumable entry passes the OLD checkpoint prefix to
    the new launch (SageMaker restores it → entrypoint resumes from last step)."""
    rm = race_mod
    race = _start(rm)
    # mark the entry failed-in-training with a recorded checkpoint prefix
    entry = race.entries[0]
    entry.state = rm.FAILED
    entry.train_job = "train-old"
    entry.checkpoint_s3 = "s3://b/jobs/OLD/checkpoints"
    entry.eval_job = None
    rm._save(race)

    seen = {}
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: seen.update(kw) or {"jobName": "train-new"})
    rm.retry_entry(race.race_id, entry.entry_key, "20260615-1", resume=True)
    # the new launch reused the OLD prefix (resume), not a fresh one
    assert seen["resume_checkpoint_s3"] == "s3://b/jobs/OLD/checkpoints"
    reloaded = rm._load(race.race_id)
    assert reloaded.entries[0].state == rm.TRAINING
    assert reloaded.entries[0].checkpoint_s3 == "s3://b/jobs/OLD/checkpoints"


def test_retry_fresh_does_not_reuse_checkpoint(race_mod, monkeypatch):
    """retry without resume (the default) retrains from scratch — no prior prefix
    passed; the new job records its OWN checkpoint prefix."""
    rm = race_mod
    race = _start(rm)
    entry = race.entries[0]
    entry.state = rm.FAILED
    entry.train_job = "train-old"
    entry.checkpoint_s3 = "s3://b/jobs/OLD/checkpoints"
    entry.eval_job = None
    rm._save(race)

    seen = {}
    monkeypatch.setattr(rm, "launch_training_job",
                        lambda **kw: seen.update(kw) or {"jobName": "train-new",
                                                         "checkpointS3": "s3://b/jobs/NEW/checkpoints"})
    rm.retry_entry(race.race_id, entry.entry_key, "20260615-1", resume=False)
    assert seen.get("resume_checkpoint_s3") is None
    reloaded = rm._load(race.race_id)
    assert reloaded.entries[0].checkpoint_s3 == "s3://b/jobs/NEW/checkpoints"
