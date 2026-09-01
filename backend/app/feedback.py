# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""User feedback board — issues, ideas, and praise, with screenshot attachments.

A lightweight shared feedback channel so users can report issues, request
features, or share what worked, and everyone can see the running list. Entries
are GLOBAL (a shared board, not per-user) — stored as a single root JSON doc
`feedback.json` ({id: entry}), the same global-root-doc pattern as
custom_models.json (TenantStore.read_root_json/write_root_json bypass tenant
scoping on purpose). The submitter's stable identity (alias/sub via
current_tenant) is stamped as the author so a global board still attributes.

Attachments (screenshots, images only) are uploaded browser→S3 via the SAME
presigned-PUT flow datasets use (uploads.make_upload_url → uploadId), then
COPIED here into a permanent per-entry prefix on create (the uploads prefix is
transient staging that may be lifecycle-expired). They're served back via
presigned GET urls.

Deterministic core; AWS only for the attachment copy/presign/delete. Validation
(type, title, image extension, count/size caps) happens before any S3 work.
"""

from __future__ import annotations

import json
from typing import Any

from .aws_config import load_aws_config
from .store import get_store
from .uploads import UPLOAD_PREFIX

_FEEDBACK_FILE = "feedback.json"  # global root doc {id: entry}
# Permanent home for attachments (distinct from the transient uploads prefix).
FEEDBACK_PREFIX = "slm-platform/feedback"

# Submission types + lifecycle states. Kept as explicit allowlists so a bad value
# can never reach the store (and the UI's pickers stay in sync).
FEEDBACK_TYPES = ("issue", "idea", "praise")
FEEDBACK_STATUSES = ("open", "planned", "done", "wont_do")
_DEFAULT_STATUS = "open"

# Attachments: images only, capped count + size — covers "screenshot for proof"
# without opening an arbitrary-file-upload surface.
_ALLOWED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
MAX_ATTACHMENTS = 5
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024  # 10 MB each
_PRESIGN_GET_EXPIRY_S = 3600


class FeedbackError(ValueError):
    """A feedback entry is invalid or an attachment can't be processed."""


def _registry() -> dict[str, Any]:
    return get_store().read_root_json(_FEEDBACK_FILE)


def _save_registry(reg: dict[str, Any]) -> None:
    get_store().write_root_json(_FEEDBACK_FILE, reg)


def _slug(title: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in title.strip().lower())
    safe = "-".join(p for p in safe.split("-") if p)
    return (safe[:32] or "feedback")


def feedback_id(title: str, stamp: str) -> str:
    """Stable, unique id from a title slug + the caller-supplied stamp (no time in
    lib code). The stamp keeps same-title entries distinct."""
    return f"{_slug(title)}-{stamp}"


def _is_image_key(key: str) -> bool:
    return key.lower().endswith(_ALLOWED_IMAGE_EXTS)


def _copy_attachment(upload_id: str, dest_key: str) -> dict[str, Any]:
    """Copy a presigned-uploaded object from the transient uploads prefix to the
    permanent feedback prefix, after enforcing the image-only + size caps. Returns
    {key, name, size, contentType}. Raises FeedbackError on a bad/oversized object."""
    if not isinstance(upload_id, str) or not upload_id.startswith(UPLOAD_PREFIX + "/"):
        raise FeedbackError("invalid attachment upload id")
    if not _is_image_key(upload_id):
        raise FeedbackError(
            "attachments must be images (png, jpg, jpeg, gif, or webp)")
    cfg = load_aws_config()
    from .orchestrate import _session

    _, boto_sess = _session(cfg)
    s3 = boto_sess.client("s3")
    head = s3.head_object(Bucket=cfg.bucket, Key=upload_id)
    size = head.get("ContentLength", 0)
    if size > MAX_ATTACHMENT_BYTES:
        raise FeedbackError(
            f"attachment is {size} bytes, exceeds the {MAX_ATTACHMENT_BYTES}-byte limit")
    content_type = head.get("ContentType", "application/octet-stream")
    if not str(content_type).lower().startswith("image/"):
        # The key extension already passed; this catches a mislabeled upload.
        content_type = "image/" + upload_id.rsplit(".", 1)[-1].lower().replace("jpg", "jpeg")
    s3.copy_object(
        Bucket=cfg.bucket, Key=dest_key,
        CopySource={"Bucket": cfg.bucket, "Key": upload_id},
        ContentType=content_type, MetadataDirective="REPLACE",
    )
    name = upload_id.rsplit("/", 1)[-1]
    # Strip the stamp prefix the uploads flow prepends, for a cleaner display name.
    if "-" in name:
        name = name.split("-", 1)[-1] if name.split("-", 1)[0].replace(".", "").isdigit() else name
    return {"key": dest_key, "name": name, "size": size, "contentType": content_type}


def _presign_get(key: str) -> str:
    cfg = load_aws_config()
    from .orchestrate import _session

    _, boto_sess = _session(cfg)
    return boto_sess.client("s3").generate_presigned_url(
        "get_object", Params={"Bucket": cfg.bucket, "Key": key},
        ExpiresIn=_PRESIGN_GET_EXPIRY_S,
    )


def _with_attachment_urls(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the entry with a fresh presigned GET url on each attachment
    (presigned urls expire, so they're minted at read time, not stored)."""
    atts = []
    for a in entry.get("attachments", []) or []:
        try:
            atts.append({**a, "url": _presign_get(a["key"])})
        except Exception:  # noqa: BLE001 — a missing object shouldn't hide the entry
            atts.append({**a, "url": ""})
    return {**entry, "attachments": atts}


def create_feedback(
    *, type: str, title: str, body: str, author: str,
    attachment_upload_ids: list[str] | None = None, stamp: str = "",
) -> dict[str, Any]:
    """Validate + persist a feedback entry, copying any image attachments from the
    uploads prefix into the entry's permanent prefix. Returns the stored entry
    (with presigned attachment urls). Raises FeedbackError on a bad type/title or a
    non-image/oversized/too-many attachment."""
    t = (type or "").strip().lower()
    if t not in FEEDBACK_TYPES:
        raise FeedbackError(f"type must be one of {list(FEEDBACK_TYPES)}")
    title = (title or "").strip()
    if not title:
        raise FeedbackError("a title is required")
    if len(title) > 200:
        raise FeedbackError("title is too long (max 200 chars)")
    body = (body or "").strip()[:10000]
    upload_ids = list(attachment_upload_ids or [])
    if len(upload_ids) > MAX_ATTACHMENTS:
        raise FeedbackError(f"at most {MAX_ATTACHMENTS} attachments are allowed")

    fid = feedback_id(title, stamp)
    attachments: list[dict[str, Any]] = []
    for i, up in enumerate(upload_ids):
        ext = up.rsplit(".", 1)[-1].lower() if "." in up else "png"
        dest = f"{FEEDBACK_PREFIX}/{fid}/{i}.{ext}"
        attachments.append(_copy_attachment(up, dest))

    entry = {
        "id": fid, "type": t, "title": title, "body": body,
        "author": author or "anonymous", "status": _DEFAULT_STATUS,
        "createdStamp": stamp,
        "attachments": [{k: a[k] for k in ("key", "name", "size", "contentType")}
                        for a in attachments],
    }
    reg = _registry()
    reg[fid] = entry
    _save_registry(reg)
    return _with_attachment_urls(entry)


def list_feedback() -> list[dict[str, Any]]:
    """All feedback entries, newest-first (by createdStamp), each with fresh
    presigned attachment urls."""
    entries = list(_registry().values())
    entries.sort(key=lambda e: e.get("createdStamp", ""), reverse=True)
    return [_with_attachment_urls(e) for e in entries]


def get_feedback(fid: str) -> dict[str, Any] | None:
    e = _registry().get(fid)
    return _with_attachment_urls(e) if e else None


def set_feedback_status(fid: str, status: str) -> dict[str, Any]:
    """Update an entry's lifecycle status (anyone can triage). Raises if unknown."""
    s = (status or "").strip().lower()
    if s not in FEEDBACK_STATUSES:
        raise FeedbackError(f"status must be one of {list(FEEDBACK_STATUSES)}")
    reg = _registry()
    if fid not in reg:
        raise FeedbackError(f"feedback not found: {fid}")
    reg[fid]["status"] = s
    _save_registry(reg)
    return _with_attachment_urls(reg[fid])


def delete_feedback(fid: str, requester: str) -> bool:
    """Delete an entry — ONLY by its author (delete-own). Removes its S3
    attachments best-effort. Returns False if not found; raises FeedbackError if
    the requester isn't the author."""
    reg = _registry()
    entry = reg.get(fid)
    if entry is None:
        return False
    if entry.get("author") != requester:
        raise FeedbackError("only the author can delete this feedback")
    # Best-effort attachment cleanup.
    try:
        cfg = load_aws_config()
        from .orchestrate import _session

        _, boto_sess = _session(cfg)
        s3 = boto_sess.client("s3")
        for a in entry.get("attachments", []) or []:
            try:
                s3.delete_object(Bucket=cfg.bucket, Key=a["key"])
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 — cleanup is best-effort; still drop the record
        pass
    del reg[fid]
    _save_registry(reg)
    return True
