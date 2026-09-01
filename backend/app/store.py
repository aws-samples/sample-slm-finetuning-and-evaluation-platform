# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Pluggable state store — the seam between the app and where state lives.

The platform persists three kinds of state:

  * dataset splits   collection "runs",   key=<split_id>   (train/eval jsonl,
                     dataset_info.json, meta.json, rendered YAMLs, baseline)
  * fine-tuning races collection "races", key=<race_id>    (race.json)
  * AWS config       a single root JSON document (config.json)

Locally this is just the filesystem under backend/data/ — identical to the
behaviour the app has always had. Hosted on Lambda there is no durable local
disk, so the same operations must map onto S3 (and/or DynamoDB).

The abstraction is a **materialize → mutate → commit** model:

    wd = store.workdir("runs", split_id)   # a real local dir to read/write files
    (wd / "train.jsonl").write_text(...)    # mutate files in it
    store.commit("runs", split_id)          # make the changes durable

Locally `workdir` is the on-disk dir and `commit` is a no-op. In the cloud
`workdir` syncs the prefix from S3 into Lambda's writable /tmp and
`commit` uploads it back — which is how you'd stage SageMaker job inputs anyway.

Backend is selected by SLM_STORAGE_BACKEND (default "local"); the local data
root is SLM_DATA_DIR (default backend/data). This module imports nothing from
the rest of the app, so it can be used during early startup (e.g. aws_config).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any


class StateStore(ABC):
    """Backend-agnostic persistence primitive (see module docstring)."""

    # --- collection/key blobs (splits, races) -------------------------------
    @abstractmethod
    def workdir(self, collection: str, key: str) -> Path:
        """Return a local directory for (collection, key), creating it. Write
        files into it, then call commit() to make them durable."""

    @abstractmethod
    def commit(self, collection: str, key: str) -> None:
        """Make the workdir's current contents durable. No-op for local disk."""

    @abstractmethod
    def dir_exists(self, collection: str, key: str) -> bool:
        """Whether any state exists for (collection, key)."""

    @abstractmethod
    def list_keys(self, collection: str) -> list[str]:
        """All keys present under a collection."""

    @abstractmethod
    def key_mtime(self, collection: str, key: str) -> float | None:
        """Last-modified epoch seconds for a key (newest-first ordering)."""

    @abstractmethod
    def read_file(self, collection: str, key: str, filename: str) -> str | None:
        """Read one file's text without materializing the whole dir, or None."""

    @abstractmethod
    def file_exists(self, collection: str, key: str, filename: str) -> bool:
        """Whether a single file exists under (collection, key)."""

    # --- single root-level JSON document (config.json) ----------------------
    @abstractmethod
    def read_root_json(self, filename: str) -> dict[str, Any]:
        """Read a root-level JSON doc; {} if missing or unparseable."""

    @abstractmethod
    def write_root_json(self, filename: str, data: dict[str, Any]) -> None:
        """Write (overwrite) a root-level JSON doc."""


class LocalStore(StateStore):
    """Filesystem backend rooted at a data directory (the historical layout)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir(self, collection: str, key: str) -> Path:
        return self.root / collection / key

    def workdir(self, collection: str, key: str) -> Path:
        d = self._dir(collection, key)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def commit(self, collection: str, key: str) -> None:
        # Files were written straight to disk; nothing to flush.
        return None

    def dir_exists(self, collection: str, key: str) -> bool:
        return self._dir(collection, key).is_dir()

    def list_keys(self, collection: str) -> list[str]:
        base = self.root / collection
        if not base.exists():
            return []
        return [d.name for d in base.iterdir() if d.is_dir()]

    def key_mtime(self, collection: str, key: str) -> float | None:
        d = self._dir(collection, key)
        return d.stat().st_mtime if d.exists() else None

    def read_file(self, collection: str, key: str, filename: str) -> str | None:
        p = self._dir(collection, key) / filename
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None

    def file_exists(self, collection: str, key: str, filename: str) -> bool:
        return (self._dir(collection, key) / filename).exists()

    def read_root_json(self, filename: str) -> dict[str, Any]:
        p = self.root / filename
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError, OSError):
            return {}

    def write_root_json(self, filename: str, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")


class CloudStore(StateStore):
    """S3-backed store for the hosted (Lambda) deployment.

    State lives under s3://<bucket>/<prefix>/<collection>/<key>/<file>. The
    config doc is s3://<bucket>/<prefix>/<filename>.

    `workdir` syncs a key's objects down into a per-process /tmp dir so existing
    file-based code (orchestrate's YAML writes, baseline's eval.jsonl reads) work
    unchanged; `commit` uploads the dir's current contents back. Listings/reads
    hit S3 directly. There's no cross-invocation locking — fine here because race
    docs are advanced by a single scheduled reconcile, and split/baseline writes
    are last-writer-wins on immutable-ish content.

    Bucket + prefix + region come from the environment (set by CDK as Lambda env
    vars), keeping this module independent of the app's config layer.
    """

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str | None = None,
        region: str | None = None,
    ) -> None:
        # local import: only needed in the cloud backend
        from .aws_clients import get_client

        self.bucket = bucket or os.environ.get("SLM_S3_BUCKET")
        if not self.bucket:
            raise ValueError("CloudStore requires SLM_S3_BUCKET (the state bucket)")
        # State prefix kept distinct from the job-inputs prefix (slm-platform/jobs/).
        self.prefix = (prefix or os.environ.get("SLM_STATE_PREFIX", "slm-platform/state")).strip("/")
        region = region or os.environ.get("SLM_AWS_REGION") or os.environ.get("AWS_REGION")
        self._s3 = get_client("s3", region_name=region)
        # Per-process scratch root for materialized workdirs (Lambda: /tmp).
        self._scratch = Path(os.environ.get("SLM_SCRATCH_DIR", "/tmp/slm-state"))

    # --- key/prefix helpers -------------------------------------------------
    def _key_prefix(self, collection: str, key: str) -> str:
        return f"{self.prefix}/{collection}/{key}"

    def _obj_key(self, collection: str, key: str, filename: str) -> str:
        return f"{self._key_prefix(collection, key)}/{filename}"

    def _local_dir(self, collection: str, key: str) -> Path:
        return self._scratch / collection / key

    def _list_objects(self, prefix: str, delimiter: str = "") -> Any:
        paginator = self._s3.get_paginator("list_objects_v2")
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        if delimiter:
            kwargs["Delimiter"] = delimiter
        return paginator.paginate(**kwargs)

    # --- collection/key blobs ----------------------------------------------
    def workdir(self, collection: str, key: str) -> Path:
        d = self._local_dir(collection, key)
        d.mkdir(parents=True, exist_ok=True)
        prefix = self._key_prefix(collection, key) + "/"
        for page in self._list_objects(prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix):]
                if not rel:  # the "directory" placeholder, if any
                    continue
                dest = d / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(self.bucket, obj["Key"], str(dest))
        return d

    def commit(self, collection: str, key: str) -> None:
        d = self._local_dir(collection, key)
        if not d.exists():
            return
        prefix = self._key_prefix(collection, key)
        for path in d.rglob("*"):
            if path.is_file():
                rel = path.relative_to(d).as_posix()
                self._s3.upload_file(str(path), self.bucket, f"{prefix}/{rel}")

    def dir_exists(self, collection: str, key: str) -> bool:
        prefix = self._key_prefix(collection, key) + "/"
        resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
        return resp.get("KeyCount", 0) > 0

    def list_keys(self, collection: str) -> list[str]:
        prefix = f"{self.prefix}/{collection}/"
        keys: list[str] = []
        for page in self._list_objects(prefix, delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                keys.append(cp["Prefix"][len(prefix):].rstrip("/"))
        return keys

    def key_mtime(self, collection: str, key: str) -> float | None:
        prefix = self._key_prefix(collection, key) + "/"
        latest: float | None = None
        for page in self._list_objects(prefix):
            for obj in page.get("Contents", []):
                ts = obj["LastModified"].timestamp()
                if latest is None or ts > latest:
                    latest = ts
        return latest

    def read_file(self, collection: str, key: str, filename: str) -> str | None:
        return self._get_text(self._obj_key(collection, key, filename))

    def file_exists(self, collection: str, key: str, filename: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._obj_key(collection, key, filename))
            return True
        except ClientError:
            return False

    # --- root-level JSON doc ------------------------------------------------
    def read_root_json(self, filename: str) -> dict[str, Any]:
        raw = self._get_text(f"{self.prefix}/{filename}")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}

    def write_root_json(self, filename: str, data: dict[str, Any]) -> None:
        self._s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}/{filename}",
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def _get_text(self, key: str) -> str | None:
        from botocore.exceptions import ClientError

        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read().decode("utf-8")
        except ClientError:
            return None


class TenantStore(StateStore):
    """Per-user view over a base store: isolates user-owned COLLECTION state
    (runs/races/curves) under a `users/<tenant>/` prefix, while leaving root-level
    JSON docs (config.json, verifications.json, custom_models.json) GLOBAL/shared.

    The tenant is resolved per-call from the tenancy contextvar, so a single
    process-wide instance serves every request/background scope correctly. For
    the DEFAULT_TENANT sentinel the collection name is passed through UNCHANGED —
    so with multi-tenancy off, state lives exactly where it historically did
    (no prefix, no migration). For a real tenant `t`, collection "races" becomes
    "users/<t>/races", which both LocalStore and CloudStore key off transparently.

    Rationale for prefixing the COLLECTION (not the key): every collection method
    routes through `_scope(collection)`, so one transform covers workdir/commit/
    list_keys/dir_exists/read_file/file_exists/key_mtime with no per-method logic.
    Root-doc methods deliberately bypass it — those facts are platform-global.
    """

    def __init__(self, base: StateStore) -> None:
        self._base = base

    def _scope(self, collection: str) -> str:
        # Imported lazily so store.py stays importable during early startup
        # (aws_config) without pulling the tenancy module in a cycle.
        from .tenancy import DEFAULT_TENANT, current_tenant

        tenant = current_tenant()
        if tenant == DEFAULT_TENANT:
            return collection
        return f"users/{tenant}/{collection}"

    # Sample-namespace READ fallback. The curated "golden" runs live once under
    # users/__samples__/<collection>. When the current tenant has opted into
    # samples, READS for a key that doesn't exist in the user's own scope fall
    # back to the sample namespace — so EVERY consumer (split_dir/split_meta,
    # leaderboard split resolution, run-detail, curve snapshots) sees the showcase
    # with no per-call-site patching. WRITES never fall back (commit always targets
    # the user's own tenant), so the shared showcase is read-only by construction.
    # Collections the fallback must NEVER touch, to avoid infinite recursion:
    # reading the per-user "prefs" doc is how samples_enabled() is computed, so it
    # can't itself trigger a samples_enabled() check.
    _SAMPLE_FALLBACK_SKIP = {"prefs"}

    def _sample_fallback_active(self, collection: str) -> bool:
        from .tenancy import SAMPLES_TENANT_NAME, current_tenant

        if collection in self._SAMPLE_FALLBACK_SKIP:
            return False
        if current_tenant() == SAMPLES_TENANT_NAME:
            return False  # already scoped to the sample namespace; don't recurse
        try:
            from .samples import samples_enabled

            return samples_enabled()
        except Exception:  # noqa: BLE001 — never let the fallback break a normal read
            return False

    def _sample_scope(self, collection: str) -> str:
        from .tenancy import SAMPLES_TENANT_NAME

        return f"users/{SAMPLES_TENANT_NAME}/{collection}"

    # --- collection/key blobs: tenant-scoped --------------------------------
    def workdir(self, collection: str, key: str) -> Path:
        own = self._scope(collection)
        own_dir = self._base.workdir(own, key)
        # Sample COPY-ON-READ: if the key lives only in the shared sample namespace
        # (not the user's own tenant), materialize the sample's files INTO the
        # user's own workdir. Crucially this keeps workdir + commit consistent: a
        # caller that does workdir→write→commit (e.g. running a baseline on a sample
        # split) now reads the sample content AND its commit lands in the USER'S
        # tenant — "forking" the sample into the user's own copy, never mutating the
        # shared showcase. (A plain read just sees the sample files; nothing commits.)
        if self._sample_fallback_active(collection) and not self._base.dir_exists(own, key):
            sample = self._sample_scope(collection)
            if self._base.dir_exists(sample, key):
                import shutil

                sample_dir = self._base.workdir(sample, key)
                for src in sample_dir.rglob("*"):
                    if src.is_file():
                        dest = own_dir / src.relative_to(sample_dir)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            shutil.copy2(src, dest)
        return own_dir

    def commit(self, collection: str, key: str) -> None:
        # Writes ALWAYS target the user's own tenant — never the shared samples.
        return self._base.commit(self._scope(collection), key)

    def dir_exists(self, collection: str, key: str) -> bool:
        if self._base.dir_exists(self._scope(collection), key):
            return True
        if self._sample_fallback_active(collection):
            return self._base.dir_exists(self._sample_scope(collection), key)
        return False

    def list_keys(self, collection: str) -> list[str]:
        # list_keys is NOT fallback-merged: the overlay readers (samples.py) union
        # the sample list explicitly + tag isSample, so merging here would
        # double-count and lose the tag. Own scope only.
        return self._base.list_keys(self._scope(collection))

    def key_mtime(self, collection: str, key: str) -> float | None:
        own = self._base.key_mtime(self._scope(collection), key)
        if own is None and self._sample_fallback_active(collection):
            return self._base.key_mtime(self._sample_scope(collection), key)
        return own

    def read_file(self, collection: str, key: str, filename: str) -> str | None:
        own = self._base.read_file(self._scope(collection), key, filename)
        if own is None and self._sample_fallback_active(collection):
            return self._base.read_file(self._sample_scope(collection), key, filename)
        return own

    def file_exists(self, collection: str, key: str, filename: str) -> bool:
        if self._base.file_exists(self._scope(collection), key, filename):
            return True
        if self._sample_fallback_active(collection):
            return self._base.file_exists(self._sample_scope(collection), key, filename)
        return False

    # --- root-level JSON docs: GLOBAL (never tenant-scoped) ------------------
    def read_root_json(self, filename: str) -> dict[str, Any]:
        return self._base.read_root_json(filename)

    def write_root_json(self, filename: str, data: dict[str, Any]) -> None:
        return self._base.write_root_json(filename, data)


def _default_data_dir() -> Path:
    # backend/data, the historical location (this file lives in backend/app/).
    return Path(__file__).resolve().parent.parent / "data"


def _make_base_store() -> StateStore:
    backend = os.environ.get("SLM_STORAGE_BACKEND", "local").lower()
    if backend == "local":
        root = Path(os.environ.get("SLM_DATA_DIR", str(_default_data_dir())))
        return LocalStore(root)
    if backend in ("cloud", "s3"):
        return CloudStore()
    raise ValueError(f"unknown SLM_STORAGE_BACKEND: {backend!r} (use 'local' or 'cloud')")


@lru_cache(maxsize=1)
def get_store() -> StateStore:
    """The process-wide state store, chosen by SLM_STORAGE_BACKEND and wrapped in
    a TenantStore so collection state is per-user while root docs stay global.
    The wrapper is always present but is a no-op for the default tenant, so this
    is transparent until multi-tenancy is enabled."""
    return TenantStore(_make_base_store())


def list_tenants() -> list[str]:
    """All tenant ids that have any state, by listing the `users/` namespace on
    the BASE store (bypassing per-request scoping). Used by the background
    reconcile loop to advance every tenant's races. With multi-tenancy off this
    is normally empty (existing state is un-prefixed under the default tenant),
    so callers should reconcile the default tenant regardless."""
    store = get_store()
    base = store._base if isinstance(store, TenantStore) else store
    return base.list_keys("users")
