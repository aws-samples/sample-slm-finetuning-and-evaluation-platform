# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Race-completion email notifications: normalization, persistence/back-compat,
the exactly-once reconcile fire, and the (AWS-free) email-body + verification logic.
All SES calls are stubbed — no AWS is touched.
"""
import types

import pytest

from app.catalog import DecodingParams, Hyperparams


# --- email normalization (pure) --------------------------------------------- #

def test_normalize_notify_emails_validates_dedupes_caps():
    from app.race import _normalize_notify_emails, _MAX_NOTIFY_EMAILS

    # lowercases, trims, drops blanks + non-emails, de-dupes order-preserving
    assert _normalize_notify_emails(
        ["  Foo@Bar.com ", "foo@bar.com", "", "not-an-email", "x@y.zz"]
    ) == ["foo@bar.com", "x@y.zz"]
    # None / empty
    assert _normalize_notify_emails(None) == []
    assert _normalize_notify_emails([]) == []
    # non-strings are skipped (never raises)
    assert _normalize_notify_emails([None, 123, "a@b.co"]) == ["a@b.co"]  # type: ignore[list-item]
    # capped
    many = [f"u{i}@x.co" for i in range(_MAX_NOTIFY_EMAILS + 5)]
    assert len(_normalize_notify_emails(many)) == _MAX_NOTIFY_EMAILS


# --- persistence + back-compat ---------------------------------------------- #

@pytest.fixture
def race_mod(temp_store, monkeypatch):
    from app import race as race_mod
    monkeypatch.setattr(race_mod, "launch_training_job",
                        lambda **kw: {"jobName": f"train-{kw['model_id']}-{kw['stamp']}"})
    monkeypatch.setattr(race_mod, "launch_eval_job", lambda **kw: {"jobName": f"eval-{kw['stamp']}"})
    monkeypatch.setattr(race_mod, "launch_base_eval_job", lambda **kw: {"jobName": "b"})
    monkeypatch.setattr(race_mod, "split_dir", lambda s: "/tmp/fake")
    return race_mod


def test_notify_emails_persist_and_normalize_through_start_race(race_mod):
    rms = [race_mod.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())]
    race = race_mod.start_race("split-x", rms, DecodingParams(), "20260630-1",
                               notify_emails=["Me@Example.com", "me@example.com", "junk"])
    # normalized (lowercased + deduped + junk dropped) and persisted
    assert race.notify_emails == ["me@example.com"]
    assert race.notified is False
    reloaded = race_mod._load(race.race_id)
    assert reloaded.notify_emails == ["me@example.com"]
    assert reloaded.notified is False


def test_load_tolerates_race_persisted_without_notify_fields(race_mod):
    """Back-compat: a race.json written before this feature (no notifyEmails/notified
    keys) must load with safe defaults, never KeyError."""
    race = race_mod.start_race("split-x", [race_mod.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())],
                               DecodingParams(), "20260630-2")
    # Simulate a legacy doc: drop the new keys from the stored JSON.
    import json
    from app.store import get_store
    store = get_store()
    raw = json.loads(store.read_file(race_mod.RACES, race.race_id, race_mod.RACE_FILE))
    raw.pop("notifyEmails", None)
    raw.pop("notified", None)
    wd = store.workdir(race_mod.RACES, race.race_id)
    (wd / race_mod.RACE_FILE).write_text(json.dumps(raw), encoding="utf-8")
    store.commit(race_mod.RACES, race.race_id)
    reloaded = race_mod._load(race.race_id)
    assert reloaded.notify_emails == []
    assert reloaded.notified is False


# --- exactly-once reconcile fire -------------------------------------------- #

def _terminal_race(race_mod, notify_emails, notified=False):
    """A race with one DONE entry, persisted, optionally with recipients."""
    race = race_mod.start_race("split-x", [race_mod.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())],
                               DecodingParams(), "20260630-3", notify_emails=notify_emails)
    race.entries[0].state = race_mod.DONE
    race.notified = notified
    race_mod._save(race)
    return race


def test_notify_fires_once_when_all_terminal(race_mod, monkeypatch):
    sent = []
    monkeypatch.setattr("app.notify.send_race_complete_email",
                        lambda race, ranked, **kw: sent.append(race.race_id) or True)
    race = _terminal_race(race_mod, ["me@example.com"])
    # First reconcile on a fully-terminal race → sends + flips notified.
    race_mod.reconcile_race(race.race_id)
    assert sent == [race.race_id]
    assert race_mod._load(race.race_id).notified is True
    # Second reconcile → notified guard means NO second send.
    race_mod.reconcile_race(race.race_id)
    assert sent == [race.race_id]


def test_notify_does_not_fire_while_race_in_progress(race_mod, monkeypatch):
    sent = []
    monkeypatch.setattr("app.notify.send_race_complete_email",
                        lambda race, ranked, **kw: sent.append(1) or True)
    # An entry still TRAINING → not all-terminal → no email.
    race = race_mod.start_race("split-x", [race_mod.RaceModel(model_id="qwen3-1.7b", hp=Hyperparams())],
                               DecodingParams(), "20260630-4", notify_emails=["me@example.com"])
    assert race.entries[0].state == race_mod.TRAINING
    race_mod._maybe_notify_complete(race_mod._load(race.race_id))
    assert sent == []
    assert race_mod._load(race.race_id).notified is False


def test_notify_marks_done_with_no_recipients_to_stop_rechecking(race_mod):
    """A terminal race with NO recipients flips notified=True (so we stop checking)
    without sending anything."""
    race = _terminal_race(race_mod, [])
    race_mod._maybe_notify_complete(race_mod._load(race.race_id))
    assert race_mod._load(race.race_id).notified is True


def test_notify_send_failure_does_not_set_flag_so_it_retries(race_mod, monkeypatch):
    """A transient send failure must NOT flip notified, so the next reconcile retries."""
    monkeypatch.setattr("app.notify.send_race_complete_email", lambda race, ranked, **kw: False)
    race = _terminal_race(race_mod, ["me@example.com"])
    race_mod._maybe_notify_complete(race_mod._load(race.race_id))
    assert race_mod._load(race.race_id).notified is False


# --- email body (pure) ------------------------------------------------------ #

def test_build_email_body_has_winner_status_and_score():
    from app.notify import build_race_complete_email
    race = types.SimpleNamespace(
        name="my run", race_id="r1", notify_emails=["a@b.co"],
        entries=[types.SimpleNamespace(state="done", model_display="Qwen3 1.7B", model_id="q", error=None),
                 types.SimpleNamespace(state="failed", model_display="Llama 1B", model_id="l", error="OOM")])
    ranked = [
        {"model_display": "Qwen3 1.7B", "model_id": "q", "state": "done",
         "rankScore": 0.83, "rankMetric": "token_f1", "isWinner": True, "error": None},
        {"model_display": "Llama 1B", "model_id": "l", "state": "failed",
         "rankScore": None, "rankMetric": "token_f1", "error": "OOM"},
    ]
    body = build_race_complete_email(race, ranked)
    assert "Qwen3 1.7B" in body["subject"] and "winner" in body["subject"].lower()
    assert "83.0%" in body["text"] and "OOM" in body["text"]
    assert "<table" in body["html"] and "Qwen3 1.7B" in body["html"]


def test_build_email_body_all_failed():
    from app.notify import build_race_complete_email
    race = types.SimpleNamespace(
        name="r", race_id="r2", notify_emails=[],
        entries=[types.SimpleNamespace(state="failed", model_display="M", model_id="m", error="boom")])
    ranked = [{"model_display": "M", "model_id": "m", "state": "failed",
               "rankScore": None, "rankMetric": "token_f1", "error": "boom"}]
    body = build_race_complete_email(race, ranked)
    assert "failed" in body["subject"].lower()


# --- recipient verification (SES stubbed) ----------------------------------- #

class _StubSes:
    def __init__(self, verified=(), raise_get=()):
        self._verified = set(verified)
        self._raise_get = set(raise_get)
        self.created = []

    def get_email_identity(self, EmailIdentity):
        if EmailIdentity in self._raise_get:
            raise RuntimeError("NotFoundException")
        return {"VerifiedForSendingStatus": EmailIdentity in self._verified}

    def create_email_identity(self, EmailIdentity):
        self.created.append(EmailIdentity)
        return {}


def test_ensure_verified_reports_and_requests():
    from app.notify import ensure_notify_recipients_verified
    stub = _StubSes(verified={"good@x.co"}, raise_get={"new@x.co"})
    out = ensure_notify_recipients_verified(["good@x.co", "pending@x.co", "new@x.co"], _client=stub)
    by = {r["email"]: r["status"] for r in out}
    assert by["good@x.co"] == "verified"
    assert by["pending@x.co"] == "pending"   # exists but not verified
    assert by["new@x.co"] == "pending"       # didn't exist → verification requested
    assert "new@x.co" in stub.created        # create_email_identity called → AWS sends link


def test_ensure_verified_empty_and_no_client(monkeypatch):
    from app import notify
    assert notify.ensure_notify_recipients_verified([]) == []
    # SES client unavailable → all "unknown", never raises, never touches AWS.
    monkeypatch.setattr(notify, "_ses_client", lambda: None)
    out = notify.ensure_notify_recipients_verified(["a@b.co"])
    assert out == [{"email": "a@b.co", "status": "unknown"}]


def test_send_skips_without_sender(monkeypatch):
    from app.notify import send_race_complete_email
    monkeypatch.delenv("SLM_NOTIFY_FROM_EMAIL", raising=False)
    race = types.SimpleNamespace(race_id="r", notify_emails=["a@b.co"], name="n", entries=[])
    # No sender configured → returns False (skipped), never raises.
    assert send_race_complete_email(race, [], _client=_StubSes()) is False
