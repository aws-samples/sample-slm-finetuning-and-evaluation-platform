# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Build the deploy bundle: a SMALL zip the browser downloads, holding only the
scripts + the manifest. The multi-GB weights are NOT in here — deploy.sh fetches
them from S3 via the presigned URL embedded in manifest.json (see export.py).

The static templates live in export_templates/; we copy them verbatim and add a
manifest.json filled in for the specific (race, model). Keeping the zip tiny is
the whole point — it sidesteps API Gateway/Lambda response-size + timeout limits
that a weights-laden zip would hit.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from .export import export_info

_TEMPLATE_DIR = Path(__file__).resolve().parent / "export_templates"

# Files copied verbatim into every bundle.
_TEMPLATE_FILES = ("deploy.sh", "inference.py", "requirements.txt", "Dockerfile", "README.md")


def build_bundle(race_id: str, model_id: str, license_accepted: bool = False) -> tuple[str, bytes]:
    """Return (filename, zip_bytes) for the deploy bundle. Raises ExportError
    (from export_info) if the model can't be exported.

    For a gated full/freeze fine-tune the weights URL is license-gated: pass
    license_accepted=True so the embedded manifest carries a usable weightsUrl.
    Without it, export_info returns a license-gated manifest (no URL) and the
    bundle's deploy.sh would have nothing to download — the endpoint refuses in
    that case (see main.export_model_bundle)."""
    info = export_info(race_id, model_id, license_accepted=license_accepted)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in _TEMPLATE_FILES:
            src = _TEMPLATE_DIR / name
            if src.exists():
                zf.writestr(name, src.read_text(encoding="utf-8"))
        zf.writestr("manifest.json", json.dumps(info, indent=2))

    safe_model = "".join(c if c.isalnum() or c in "-_" else "-" for c in model_id)
    filename = f"slm-deploy-{safe_model}.zip"
    return filename, buf.getvalue()
