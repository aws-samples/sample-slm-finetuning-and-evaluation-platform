# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Guard against import/wiring regressions in the FastAPI app.

A real bug once shipped where main.py called dispatch_worker() without importing
it — invisible locally (the inline fallback path wasn't hit) but a 500 in the
hosted dispatch path. These cheap checks catch that class of error.
"""


def test_app_imports_and_dispatch_worker_bound():
    import app.main as m

    assert callable(m.dispatch_worker)  # imported, not a NameError at runtime


def test_key_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in [
        "/api/race",
        "/api/baseline/sonnet",
        "/api/judge/{eval_job}",
        "/api/limits",
        "/api/datasets/upload-url",  # presigned direct-to-S3 upload (large files)
        "/api/datasets/preference",  # DPO preference dataset upload
        "/api/datasets/hf/import-preference",  # HF preference-dataset import
        "/api/datasets/hf/import-kto",  # HF KTO-dataset import
        "/api/datasets/kto",  # KTO binary-feedback dataset upload
        "/api/pitcrew/sessions",  # Guided Fine-tuning agent
        "/api/pitcrew/sessions/{session_id}/advance",
        "/api/pitcrew/sessions/{session_id}/edit",
        "/api/datasets/detect-shape",
    ]:
        assert p in paths, f"missing route {p}"


def test_worker_handler_imports():
    from app.lambda_handler import handler, reconcile_handler, worker_handler

    assert callable(handler) and callable(reconcile_handler) and callable(worker_handler)


def test_reconcile_handler_walks_all_tenants(temp_store, monkeypatch):
    """MULTI-TENANT: the scheduled reconcile MUST advance races in EVERY tenant, not
    just the default — else per-user (guided-flow) races hang forever. Simulate two
    tenants each with a non-terminal race and assert both get reconciled."""
    from app import lambda_handler as lh

    # list_tenants is imported INSIDE the handler (from .store), so patch it there.
    monkeypatch.setattr("app.store.list_tenants", lambda: ["alice", "bob"], raising=False)
    # Per-tenant race summaries: each tenant has one in-progress race.
    per_tenant = {
        "__default__": [],
        "alice": [{"raceId": "race-alice", "states": {"m": "training"}}],
        "bob": [{"raceId": "race-bob", "states": {"m": "training"}}],
    }
    from app import tenancy
    monkeypatch.setattr(lh, "list_races",
                        lambda: per_tenant.get(tenancy.current_tenant(), []))
    reconciled = []
    monkeypatch.setattr(lh, "reconcile_race", lambda rid: reconciled.append(rid))
    monkeypatch.setattr("app.verifications.resolve_pending_verifications",
                        lambda: {"checked": 0, "resolved": 0}, raising=False)
    out = lh.reconcile_handler({}, None)
    assert set(reconciled) == {"race-alice", "race-bob"}, reconciled
    assert out["count"] == 2


def test_maybe_gunzip_roundtrip_and_passthrough():
    """Uploads are gzipped in-browser to fit API Gateway's 10 MB cap; the server
    transparently decompresses (gzip magic bytes) or passes plain bytes through."""
    import gzip

    import app.main as m

    plain = b'{"messages":[{"role":"user","content":"hi"}]}\n' * 100
    gz = gzip.compress(plain)
    assert gz[:2] == b"\x1f\x8b"  # gzip magic
    assert m._maybe_gunzip(gz) == plain  # decompresses
    assert m._maybe_gunzip(plain) == plain  # plain passes through unchanged


def test_maybe_gunzip_rejects_zip_bomb(monkeypatch):
    """A small gzip that expands past the cap must 413, not OOM."""
    import gzip

    import app.main as m
    from fastapi import HTTPException

    monkeypatch.setattr(m, "MAX_UPLOAD_BYTES", 1024)  # tiny cap for the test
    bomb = gzip.compress(b"a" * (1024 * 50))  # 50 KB expands from a few hundred bytes
    try:
        m._maybe_gunzip(bomb)
        assert False, "expected HTTPException"
    except HTTPException as e:
        assert e.status_code == 413
