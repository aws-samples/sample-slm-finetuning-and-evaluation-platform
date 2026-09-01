# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Direct-to-S3 uploads for large dataset files.

API Gateway caps request bodies at 10 MB (non-raisable), so big dataset JSONL
can't be POSTed through it. Instead the browser uploads each file DIRECTLY to S3
via a presigned PUT URL (bypassing API Gateway entirely — S3 allows up to 5 GB),
then calls the dataset endpoints referencing the uploaded object by `uploadId`.

Flow:
  1. POST /api/datasets/upload-url {filename} -> {uploadId, url}   (presigned PUT)
  2. browser PUTs the file bytes to `url`                          (straight to S3)
  3. POST /api/datasets/split/* with upload_id=<uploadId>          (server reads S3)

Uploads land under a dedicated prefix and are content addressed by a caller-less
random-free id derived from the stamp the endpoint supplies (no time/RNG here).
The objects are transient staging data; they can be lifecycle-expired.
"""

from __future__ import annotations

import re
import uuid

from .aws_config import load_aws_config
from .orchestrate import _session

# Where browser uploads land (kept separate from job inputs + state).
UPLOAD_PREFIX = "slm-platform/uploads"

# Presigned URL lifetime — long enough to upload a large file on a slow link.
PRESIGN_EXPIRY_SECONDS = 3600

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_name(filename: str) -> str:
    """Sanitise a client filename for use in an S3 key (no path traversal)."""
    base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = _SAFE.sub("_", base).strip("._") or "upload"
    return base[:120]


def _tenant_segment() -> str:
    """The current tenant as a key-safe path segment. Uploads are TENANT-SCOPED so
    one user can't read another's staged dataset. With multi-tenancy off this is the
    DEFAULT_TENANT sentinel, so keys are stable + behaviour is unchanged."""
    from .tenancy import current_tenant

    return _SAFE.sub("_", current_tenant() or "__default__")[:80]


def make_upload_url(filename: str, stamp: str) -> dict[str, str]:
    """Create a presigned S3 PUT URL for a browser to upload `filename` to.

    The key is TENANT-SCOPED and carries a random suffix:
        {UPLOAD_PREFIX}/{tenant}/{stamp}-{rand}-{safename}
    so (a) a user can only read uploads under their own tenant (enforced in
    fetch_upload_bytes) and (b) keys aren't guessable from the filename+time alone.
    `stamp` is caller-supplied (no time in lib code); the uuid is local, not RNG-
    seeded state, so it's fine here.

    Signed with SigV4 (signature_version="s3v4"): a SigV2 presigned PUT folds the
    request's Content-Type into the signature, so a browser that sets a
    `Content-Type` header gets a 403 SignatureDoesNotMatch. SigV4 doesn't sign an
    unsigned header, so the PUT succeeds whether or not the client sets one.
    """
    from botocore.config import Config

    from .aws_clients import botocore_config

    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    s3 = boto_sess.client(
        "s3", config=botocore_config(Config(signature_version="s3v4"))
    )
    key = f"{UPLOAD_PREFIX}/{_tenant_segment()}/{stamp}-{uuid.uuid4().hex[:8]}-{_safe_name(filename)}"
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": cfg.bucket, "Key": key},
        ExpiresIn=PRESIGN_EXPIRY_SECONDS,
    )
    return {"uploadId": key, "url": url}


def _owns_upload(upload_id: str) -> bool:
    """Whether `upload_id` is a well-formed uploads key OWNED by the current tenant.
    Rejects keys outside the uploads prefix AND keys under a different tenant segment
    — so a user can't read/delete another tenant's staged dataset by passing its id.
    Legacy keys (pre-tenant-scoping: `{UPLOAD_PREFIX}/{stamp}-...`, no tenant segment)
    are honored ONLY for the default tenant so existing single-tenant uploads keep
    working."""
    prefix = UPLOAD_PREFIX + "/"
    if not upload_id.startswith(prefix):
        return False
    rest = upload_id[len(prefix):]
    seg = rest.split("/", 1)
    if len(seg) == 2:
        return seg[0] == _tenant_segment()      # tenant-scoped key → must match
    # No tenant segment (legacy key) → only the default tenant may read it.
    from .tenancy import DEFAULT_TENANT, current_tenant

    return current_tenant() == DEFAULT_TENANT


def fetch_upload_bytes(upload_id: str, max_bytes: int) -> bytes:
    """Download a previously-uploaded object's bytes, enforcing a size cap AND that
    the requesting tenant owns the key (so this can't read another user's upload or
    an arbitrary object)."""
    if not _owns_upload(upload_id):
        raise ValueError("invalid upload id")
    cfg = load_aws_config()
    _, boto_sess = _session(cfg)
    s3 = boto_sess.client("s3")
    head = s3.head_object(Bucket=cfg.bucket, Key=upload_id)
    size = head.get("ContentLength", 0)
    if size > max_bytes:
        raise ValueError(f"uploaded file is {size} bytes, exceeds {max_bytes} limit")
    obj = s3.get_object(Bucket=cfg.bucket, Key=upload_id)
    return obj["Body"].read()


def delete_upload(upload_id: str) -> None:
    """Best-effort cleanup of a staged upload after it's been consumed (own tenant only)."""
    if not _owns_upload(upload_id):
        return
    try:
        cfg = load_aws_config()
        _, boto_sess = _session(cfg)
        boto_sess.client("s3").delete_object(Bucket=cfg.bucket, Key=upload_id)
    except Exception:  # noqa: BLE001 — cleanup is best-effort
        pass
