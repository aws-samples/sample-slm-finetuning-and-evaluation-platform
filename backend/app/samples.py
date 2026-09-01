# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Sample ("golden") runs — a curated, read-only showcase a new user can enable
so they land on something to explore (a real dataset → fine-tune → leaderboard)
instead of a blank app.

Design (overlay, NOT per-user copy):
  * The curated runs live ONCE in a dedicated global tenant namespace,
    SAMPLES_TENANT (`users/__samples__/...`). You drop a run in there (seed_*),
    and it's the single source of truth — add more later and every user who has
    samples enabled sees them immediately, no re-import.
  * "Import sample run" just flips a PER-USER flag (samples_enabled). When on, the
    list/detail read paths UNION the sample namespace into the user's view, tagged
    isSample=true so the UI renders them read-only (view + clone, never
    delete/retry/relaunch — those would mutate shared state).
  * The leaderboard already discovers evals globally via SageMaker
    list_training_jobs keyed on split_id (see leaderboard.py), so once a user can
    SEE the sample dataset (its split surfaces via the dataset overlay), the
    leaderboard + eval scores populate for free — no per-tenant eval state needed.

Why this is safe to read cross-tenant: sample runs are COMPLETED (all entries
terminal), so reconcile is a no-op on them; we still only ever READ the sample
namespace from the request paths (the import endpoint is the sole writer, and it
only writes the per-user flag + the one-time seed).
"""

from __future__ import annotations

from typing import Any

from .tenancy import SAMPLES_TENANT_NAME, current_tenant, tenant_scope

# Dedicated global namespace holding the curated runs. Lives under users/ like a
# tenant, but is NEVER a real user — excluded from the reconcile loop (main.py) so
# the background state machine never mutates the showcase. The canonical name is
# defined in tenancy.py so store.py's read-fallback can use it without a cycle.
SAMPLES_TENANT = SAMPLES_TENANT_NAME

# Per-user preference doc: a tenant-scoped collection blob (root JSON is global, so
# it can't be per-user — a collection key is). Keyed by a fixed id under `prefs`.
_PREFS = "prefs"
_PREFS_KEY = "user"
_PREFS_FILE = "prefs.json"


# --- per-user "show samples" flag (tenant-scoped) ---------------------------

def _read_prefs() -> dict[str, Any]:
    import json

    from .store import get_store

    raw = get_store().read_file(_PREFS, _PREFS_KEY, _PREFS_FILE)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def _write_prefs(prefs: dict[str, Any]) -> None:
    import json

    from .store import get_store

    store = get_store()
    wd = store.workdir(_PREFS, _PREFS_KEY)
    (wd / _PREFS_FILE).write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    store.commit(_PREFS, _PREFS_KEY)


def samples_enabled() -> bool:
    """Whether the CURRENT tenant has opted into seeing the sample runs."""
    return bool(_read_prefs().get("showSamples"))


def set_samples_enabled(on: bool) -> None:
    """Flip the current tenant's show-samples flag (the 'Import sample run' button).
    No copy — just enables the overlay onto the shared sample namespace."""
    prefs = _read_prefs()
    prefs["showSamples"] = bool(on)
    _write_prefs(prefs)


# --- sample-namespace catalog (always reads the SAMPLES_TENANT) -------------

def sample_race_ids() -> list[str]:
    """Race ids present in the sample namespace (empty if none seeded yet)."""
    from .race import RACES
    from .store import get_store

    with tenant_scope(SAMPLES_TENANT):
        return get_store().list_keys(RACES)


def sample_split_ids() -> set[str]:
    """Dataset split ids that back the sample races — the splits to surface in the
    dataset overlay so the leaderboard (split-keyed) lights up for the user."""
    from .race import _load
    from .store import get_store

    ids: set[str] = set()
    with tenant_scope(SAMPLES_TENANT):
        for rid in get_store().list_keys("races"):
            r = _load(rid)
            if r:
                ids.add(r.split_id)
    return ids


def is_sample_race(race_id: str) -> bool:
    """True if `race_id` exists in the sample namespace (and the user has samples
    on). Lets the detail/clone paths route reads to SAMPLES_TENANT."""
    if current_tenant() == SAMPLES_TENANT:
        return True  # already scoped (the seed/admin path)
    if not samples_enabled():
        return False
    with tenant_scope(SAMPLES_TENANT):
        from .store import get_store

        return get_store().dir_exists("races", race_id)


# --- overlay readers (union sample state into the user's view) --------------

def overlay_races(own: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append the sample races (tagged isSample=true) to the user's own race list
    when samples are enabled. De-dupes on raceId so a user who has both a real run
    and (somehow) the same id sees their own. Newest-first ordering is re-applied
    by the caller's existing sort, but we keep samples after own by default."""
    if not samples_enabled():
        return own
    from .race import list_races

    own_ids = {r["raceId"] for r in own}
    with tenant_scope(SAMPLES_TENANT):
        samples = list_races()
    extra = [{**r, "isSample": True} for r in samples if r["raceId"] not in own_ids]
    return own + extra


def _copy_collection_key(src_tenant: str, dst_tenant: str, collection: str, key: str) -> bool:
    """Copy one collection/key blob (all its files) from src→dst tenant namespace
    via the store's workdir(materialize)→commit(upload). Returns False if the
    source key doesn't exist. Used by the seed step only (curation)."""
    import shutil

    from .store import get_store

    store = get_store()
    with tenant_scope(src_tenant):
        if not store.dir_exists(collection, key):
            return False
        src_dir = store.workdir(collection, key)  # materialize source files
        files = {p.relative_to(src_dir): p.read_bytes() for p in src_dir.rglob("*") if p.is_file()}
    with tenant_scope(dst_tenant):
        dst_dir = store.workdir(collection, key)
        for rel, data in files.items():
            dest = dst_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        store.commit(collection, key)
    return True


def seed_run_as_sample(race_id: str, src_tenant: str) -> dict[str, Any]:
    """Curate ONE run into the sample namespace: copy its race record, its backing
    dataset (split), and any training-curve snapshots from `src_tenant` into
    SAMPLES_TENANT. Idempotent (re-copy overwrites). The eval scores themselves are
    NOT copied — the leaderboard re-discovers them from SageMaker by split_id, so
    surfacing the split is enough. Returns what was copied.

    This is an ADMIN/curation step (run once per golden run), not a user action."""
    from .race import RACES, _load
    from .storage import CURVES, RUNS

    with tenant_scope(src_tenant):
        race = _load(race_id)
    if race is None:
        raise ValueError(f"race {race_id} not found in tenant {src_tenant}")

    copied: dict[str, Any] = {"raceId": race_id, "splitId": race.split_id}
    _copy_collection_key(src_tenant, SAMPLES_TENANT, RACES, race_id)
    copied["dataset"] = _copy_collection_key(src_tenant, SAMPLES_TENANT, RUNS, race.split_id)
    # Curve snapshots are keyed by training-job name — copy each entry's, if present.
    jobs = [j for e in race.entries for j in (e.train_job, e.base_eval_job) if j]
    copied["curves"] = sum(
        1 for j in jobs if _copy_collection_key(src_tenant, SAMPLES_TENANT, CURVES, j)
    )
    return copied


def overlay_datasets(own: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append the sample datasets (tagged isSample=true) to the user's dataset
    library when samples are enabled — so the sample run's split is visible (and
    its split-keyed leaderboard populates) and the user can clone against it."""
    if not samples_enabled():
        return own
    from .storage import list_datasets

    own_ids = {d["splitId"] for d in own}
    with tenant_scope(SAMPLES_TENANT):
        samples = list_datasets()
    extra = [{**d, "isSample": True} for d in samples if d["splitId"] not in own_ids]
    return own + extra
