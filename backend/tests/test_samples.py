# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Sample ("golden") runs overlay — per-user flag + cross-namespace overlay.

Uses the temp_store fixture (LocalStore under a temp dir) so the sample namespace
+ per-user prefs are isolated. Builds a sample run by seeding from a 'source'
tenant into SAMPLES_TENANT, then asserts the overlay surfaces it only when enabled.
"""
from app import samples
from app.tenancy import tenant_scope


def _make_race(name: str = "golden", split_id: str = "abc123split"):
    """Persist a minimal completed race in the CURRENT tenant; return its id."""
    from app.race import Race, RaceEntry, _save

    r = Race(
        race_id=f"race-{split_id}-20260101-000000",
        name=name,
        archived=False,
        use_spot=False,
        max_run_seconds=18000,
        spot_fallback_minutes=None,
        split_id=split_id,
        stamp="20260101-000000",
        decoding={},
        entries=[RaceEntry(model_id="qwen3-0.6b", model_display="Qwen3 0.6B",
                           instance_type="ml.g5.2xlarge", hp={}, state="done",
                           train_job="job-1")],
    )
    _save(r)
    return r.race_id


def _make_dataset(split_id: str = "abc123split"):
    """Drop a minimal dataset_info.json so list_datasets surfaces this split."""
    import json

    from app.storage import RUNS
    from app.store import get_store

    store = get_store()
    wd = store.workdir(RUNS, split_id)
    (wd / "dataset_info.json").write_text(json.dumps({"train_split": {}}), encoding="utf-8")
    (wd / "meta.json").write_text(json.dumps({"name": "sample-data", "trainRows": 10}), encoding="utf-8")
    store.commit(RUNS, split_id)


# --- per-user flag ----------------------------------------------------------

def test_samples_disabled_by_default(temp_store):
    assert samples.samples_enabled() is False


def test_set_and_read_flag(temp_store):
    samples.set_samples_enabled(True)
    assert samples.samples_enabled() is True
    samples.set_samples_enabled(False)
    assert samples.samples_enabled() is False


# --- seed + overlay ---------------------------------------------------------

def test_seed_copies_race_and_dataset_into_samples_namespace(temp_store):
    # Build a run in a 'source' tenant, then seed it into the sample namespace.
    with tenant_scope("owner"):
        rid = _make_race()
        _make_dataset()
    res = samples.seed_run_as_sample(rid, src_tenant="owner")
    assert res["raceId"] == rid and res["dataset"] is True
    # It now lives in the sample namespace.
    assert rid in samples.sample_race_ids()
    assert "abc123split" in samples.sample_split_ids()


def test_overlay_hidden_until_enabled(temp_store):
    with tenant_scope("owner"):
        rid = _make_race()
        _make_dataset()
    samples.seed_run_as_sample(rid, src_tenant="owner")

    # A fresh user (default tenant) with samples OFF sees none of it.
    assert samples.overlay_races([]) == []
    assert samples.overlay_datasets([]) == []

    # Enable → the sample race + dataset appear, tagged isSample.
    samples.set_samples_enabled(True)
    races = samples.overlay_races([])
    assert len(races) == 1 and races[0]["raceId"] == rid and races[0]["isSample"] is True
    ds = samples.overlay_datasets([])
    assert len(ds) == 1 and ds[0]["splitId"] == "abc123split" and ds[0]["isSample"] is True


def test_overlay_does_not_duplicate_own_rows(temp_store):
    with tenant_scope("owner"):
        rid = _make_race()
    samples.seed_run_as_sample(rid, src_tenant="owner")
    samples.set_samples_enabled(True)
    # If the user already has a row with the same id, it isn't duplicated.
    own = [{"raceId": rid, "name": "mine"}]
    out = samples.overlay_races(own)
    assert len(out) == 1 and out[0]["name"] == "mine"


def test_read_fallback_resolves_sample_split(temp_store):
    # split_dir / split_meta / split_name must resolve a sample-only split when the
    # user has samples on (the store read-fallback) — this is what stopped the
    # run-detail "unknown dataset" + empty leaderboard.
    from app.storage import split_dir, split_meta, split_name

    with tenant_scope("owner"):
        _make_dataset("smpl999")
    # Re-key the dataset into the sample namespace (simulate a seeded sample split).
    from app.store import get_store
    src = None
    with tenant_scope("owner"):
        src = get_store().workdir("runs", "smpl999")
    import shutil
    with tenant_scope(samples.SAMPLES_TENANT):
        dst = get_store().workdir("runs", "smpl999")
        for p in src.rglob("*"):
            if p.is_file():
                d = dst / p.relative_to(src)
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)
        get_store().commit("runs", "smpl999")

    with tenant_scope("fresh"):
        # Off → invisible.
        assert split_dir("smpl999") is None
        # On → resolves via the sample fallback.
        samples.set_samples_enabled(True)
        assert split_dir("smpl999") is not None
        assert split_name("smpl999") == "sample-data"
        assert split_meta("smpl999").get("trainRows") == 10


def test_write_on_sample_split_forks_into_own_tenant_not_samples(temp_store):
    # Regression: running a baseline (workdir→write→commit) on a sample-only split
    # must persist in the USER's tenant (not vanish, not mutate the shared sample).
    import json

    from app.store import get_store

    # Seed a sample split.
    with tenant_scope("owner"):
        _make_dataset("fork123")
    import shutil
    with tenant_scope("owner"):
        src = get_store().workdir("runs", "fork123")
    with tenant_scope(samples.SAMPLES_TENANT):
        dst = get_store().workdir("runs", "fork123")
        for p in src.rglob("*"):
            if p.is_file():
                d = dst / p.relative_to(src); d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)
        get_store().commit("runs", "fork123")

    with tenant_scope("fresh"):
        samples.set_samples_enabled(True)
        store = get_store()
        wd = store.workdir("runs", "fork123")  # copy-on-read materializes sample files
        (wd / "baseline.json").write_text(json.dumps({"score": 1}), encoding="utf-8")
        store.commit("runs", "fork123")
        # The write persisted in the FRESH user's tenant.
        assert store.read_file("runs", "fork123", "baseline.json") is not None

    # The shared sample namespace was NOT mutated (no baseline file there).
    with tenant_scope(samples.SAMPLES_TENANT):
        assert get_store().read_file("runs", "fork123", "baseline.json") is None

    # A DIFFERENT fresh user does NOT see the first user's baseline.
    with tenant_scope("other"):
        samples.set_samples_enabled(True)
        assert get_store().read_file("runs", "fork123", "baseline.json") is None


def test_is_sample_race_gated_on_flag(temp_store):
    with tenant_scope("owner"):
        rid = _make_race()
    samples.seed_run_as_sample(rid, src_tenant="owner")
    assert samples.is_sample_race(rid) is False  # flag off
    samples.set_samples_enabled(True)
    assert samples.is_sample_race(rid) is True
    assert samples.is_sample_race("race-does-not-exist") is False
