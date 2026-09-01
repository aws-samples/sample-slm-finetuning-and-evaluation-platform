#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Merge CDK stack outputs into the backend's data/config.json.

The backend resolves AWS config as: data/config.json > env var > built-in
default (see backend/app/aws_config.py). Writing the freshly-created resource
names here points the running app at the CDK-managed infra so the Settings
page reflects them and preflight goes green — with zero manual copy/paste.

Usage (run from infra/ after a deploy):
    cdk deploy --outputs-file cdk-outputs.json
    python3 apply_outputs.py cdk-outputs.json

Only the five Settings fields (region, account, bucket, roleArn, imageUri) are
merged, each read from its corresponding CfnOutput (see FIELD_BY_OUTPUT). Any
existing keys (e.g. resetCutoff, profile) are preserved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STACK_NAME = "SlmPlatformInfra"
# CDK output key (as declared by CfnOutput in stack.py) -> config.json Settings
# field. The two names differ: outputs are prefixed "Out", config fields are not,
# so this mapping is what makes the merge work. Keep it in sync with stack.py's
# CfnOutput ids — a key that is not actually exported silently merges nothing.
FIELD_BY_OUTPUT = {
    "OutRegion": "region",
    "OutAccount": "account",
    "OutDataBucket": "bucket",
    "OutSageMakerRoleArn": "roleArn",
    "OutTrainingImageUri": "imageUri",
}

# infra/ -> repo root -> backend/data/config.json
CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "backend" / "data" / "config.json"
)


def main() -> int:
    outputs_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cdk-outputs.json")
    if not outputs_file.exists():
        print(f"error: outputs file not found: {outputs_file}", file=sys.stderr)
        print("hint: cdk deploy --outputs-file cdk-outputs.json", file=sys.stderr)
        return 1

    raw = json.loads(outputs_file.read_text(encoding="utf-8"))
    outputs = raw.get(STACK_NAME)
    if not outputs:
        print(
            f"error: no outputs for stack {STACK_NAME} in {outputs_file} "
            f"(found: {list(raw)})",
            file=sys.stderr,
        )
        return 1

    # Load existing config (preserve resetCutoff, profile, anything else).
    current = {}
    if CONFIG_PATH.exists():
        try:
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            current = {}

    applied = {}
    for output_key, field in FIELD_BY_OUTPUT.items():
        value = outputs.get(output_key)
        if value:
            current[field] = value
            applied[field] = value

    if not applied:
        print(
            "warning: none of the expected outputs were present; nothing written.\n"
            f"  looked for: {', '.join(FIELD_BY_OUTPUT)}\n"
            f"  found:      {', '.join(outputs) or '(none)'}",
            file=sys.stderr,
        )
        return 1

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(applied)} field(s) to {CONFIG_PATH}:")
    for k, v in applied.items():
        print(f"  {k} = {v}")
    print("\nRestart the backend (or it picks this up on next config read), then")
    print("click 'Check environment' on the Settings page to confirm green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
