# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for the feedback board (feedback.py + routes).

The global registry round-trips through the temp store; the only AWS is the
attachment copy/presign/delete, which we mock via a fake boto session (mirrors
test_curves/test_export). Covers validation, attachment caps (count/size/type),
status triage, and delete-own authorization.
"""
from __future__ import annotations

import pytest

from app.uploads import UPLOAD_PREFIX


class _FakeS3:
    """Minimal S3 stand-in: head_object reports a size/type, copy/get/delete no-op,
    generate_presigned_url returns a fake url."""

    def __init__(self, size=1234, content_type="image/png"):
        self._size = size
        self._ct = content_type
        self.copied: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    def head_object(self, Bucket, Key):  # noqa: N803
        return {"ContentLength": self._size, "ContentType": self._ct}

    def copy_object(self, Bucket, Key, CopySource, **kw):  # noqa: N803
        self.copied.append((CopySource["Key"], Key))

    def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
        return f"https://signed/{Params['Key']}"

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deleted.append(Key)


def _patch_s3(monkeypatch, fake):
    from app import feedback

    monkeypatch.setattr(feedback, "load_aws_config", lambda: type("C", (), {"bucket": "b"})())
    # feedback imports _session lazily from .orchestrate inside each fn.
    import app.orchestrate as orch
    monkeypatch.setattr(orch, "_session", lambda cfg: (None, type("B", (), {"client": lambda self, n, **k: fake})()))


def _upload_id(name="shot.png"):
    return f"{UPLOAD_PREFIX}/20260618-1-{name}"


def test_create_and_list_roundtrip(temp_store, monkeypatch):
    from app import feedback

    fake = _FakeS3()
    _patch_s3(monkeypatch, fake)
    e = feedback.create_feedback(
        type="issue", title="Verify spins forever", body="clicked verify, nothing happened",
        author="alice", attachment_upload_ids=[_upload_id()], stamp="s1")
    assert e["type"] == "issue" and e["author"] == "alice" and e["status"] == "open"
    assert len(e["attachments"]) == 1
    # The attachment was copied from uploads/ into the feedback/ prefix + presigned.
    assert fake.copied and fake.copied[0][1].startswith("slm-platform/feedback/")
    assert e["attachments"][0]["url"].startswith("https://signed/")

    board = feedback.list_feedback()
    assert any(x["id"] == e["id"] for x in board)


def test_type_and_title_validation(temp_store, monkeypatch):
    from app import feedback

    _patch_s3(monkeypatch, _FakeS3())
    with pytest.raises(feedback.FeedbackError):
        feedback.create_feedback(type="bogus", title="x", body="", author="a", stamp="s")
    with pytest.raises(feedback.FeedbackError):
        feedback.create_feedback(type="idea", title="   ", body="", author="a", stamp="s")


def test_attachment_caps(temp_store, monkeypatch):
    from app import feedback

    # Too many.
    _patch_s3(monkeypatch, _FakeS3())
    with pytest.raises(feedback.FeedbackError):
        feedback.create_feedback(
            type="issue", title="t", body="", author="a", stamp="s",
            attachment_upload_ids=[_upload_id(f"{i}.png") for i in range(feedback.MAX_ATTACHMENTS + 1)])

    # Non-image extension.
    with pytest.raises(feedback.FeedbackError):
        feedback.create_feedback(type="issue", title="t", body="", author="a", stamp="s2",
                                 attachment_upload_ids=[_upload_id("evil.exe")])

    # Oversized image.
    _patch_s3(monkeypatch, _FakeS3(size=feedback.MAX_ATTACHMENT_BYTES + 1))
    with pytest.raises(feedback.FeedbackError):
        feedback.create_feedback(type="issue", title="t", body="", author="a", stamp="s3",
                                 attachment_upload_ids=[_upload_id("big.png")])

    # Upload id outside the uploads prefix (can't copy arbitrary objects).
    _patch_s3(monkeypatch, _FakeS3())
    with pytest.raises(feedback.FeedbackError):
        feedback.create_feedback(type="issue", title="t", body="", author="a", stamp="s4",
                                 attachment_upload_ids=["some/other/key.png"])


def test_status_triage(temp_store, monkeypatch):
    from app import feedback

    _patch_s3(monkeypatch, _FakeS3())
    e = feedback.create_feedback(type="idea", title="dark mode", body="", author="bob", stamp="s")
    updated = feedback.set_feedback_status(e["id"], "planned")
    assert updated["status"] == "planned"
    with pytest.raises(feedback.FeedbackError):
        feedback.set_feedback_status(e["id"], "nope")
    with pytest.raises(feedback.FeedbackError):
        feedback.set_feedback_status("no-such-id", "open")


def test_delete_is_author_only(temp_store, monkeypatch):
    from app import feedback

    fake = _FakeS3()
    _patch_s3(monkeypatch, fake)
    e = feedback.create_feedback(type="issue", title="t", body="", author="alice", stamp="s",
                                 attachment_upload_ids=[_upload_id()])
    # A non-author cannot delete.
    with pytest.raises(feedback.FeedbackError):
        feedback.delete_feedback(e["id"], "mallory")
    # The author can; attachments are cleaned up.
    assert feedback.delete_feedback(e["id"], "alice") is True
    assert fake.deleted  # the attachment object was deleted
    assert feedback.get_feedback(e["id"]) is None
    # Deleting a missing id returns False (not an error).
    assert feedback.delete_feedback("gone", "alice") is False


def test_feedback_routes_registered():
    import app.main as m

    paths = {r.path for r in m.app.routes}
    for p in ["/api/feedback", "/api/feedback/{feedback_id}", "/api/feedback/{feedback_id}/status"]:
        assert p in paths, f"missing route {p}"


def test_create_endpoint_stamps_author(temp_store, monkeypatch):
    """The POST endpoint stamps the current user as author (not client-supplied)."""
    import app.main as m
    from app import feedback

    _patch_s3(monkeypatch, _FakeS3())
    monkeypatch.setattr(m, "_current_author", lambda: "carol")
    out = m.create_feedback_ep(m.FeedbackRequest(type="praise", title="love it", body="great tool"))
    assert out["author"] == "carol" and out["type"] == "praise"
    # The board reports `me` so the UI can gate Delete.
    monkeypatch.setattr(m, "_current_author", lambda: "carol")
    board = m.list_feedback_ep()
    assert board["me"] == "carol"
    assert "issue" in board["types"] and "open" in board["statuses"]
