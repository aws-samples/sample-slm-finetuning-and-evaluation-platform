# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Persist a validated, disjoint split to disk so later steps can reference it.

A successful split is written under data/runs/<split_id>/:
  train.jsonl          the training rows (chat-template JSONL)
  eval.jsonl           the held-out eval rows
  dataset_info.json    LLaMA-Factory dataset registry pointing at the two files
  meta.json            split provenance (mode, counts, seed/ratio)

`dataset_info.json` is what `llamafactory-cli` reads to resolve a `dataset:`
name to files + format. We register the rows as `sharegpt` formatting with the
OpenAI-style `messages` column, which matches our chat-template schema.

The split_id is a content hash of the rows, so re-persisting the same split is
idempotent (same id, same dir) — deterministic, no timestamps/RNG.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .store import get_store

# Collection name within the state store (was the data/runs directory).
RUNS = "runs"
# Snapshotted training curves, keyed by training-job name. CloudWatch metrics
# down-sample after ~15 days and vanish after ~15 months, and short jobs lose
# their shape fast — so we snapshot the curve to durable storage when a job
# reaches a terminal state, and serve that instead of re-querying CloudWatch.
CURVES = "curves"
CURVES_FILE = "curves.json"

# Dataset names registered inside dataset_info.json (referenced by train YAML).
TRAIN_DATASET_NAME = "train_split"
EVAL_DATASET_NAME = "eval_split"
VAL_DATASET_NAME = "val_split"


def _split_id(train: list[dict], eval_: list[dict], val: list[dict] | None = None) -> str:
    h = hashlib.sha256()
    sections = [("train", train), ("eval", eval_)]
    if val:
        sections.append(("val", val))
    for tag, rows in sections:
        h.update(tag.encode())
        for r in rows:
            h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()[:12]


# Shared sharegpt tag block — role/content key names for the messages column.
_SHAREGPT_TAGS = {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system",
}


def _sft_entry() -> dict[str, Any]:
    """LLaMA-Factory registry entry for an SFT (messages-column) dataset."""
    return {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": dict(_SHAREGPT_TAGS),
    }


def _preference_entry() -> dict[str, Any]:
    """LLaMA-Factory registry entry for a PREFERENCE (ranking) dataset — DPO/KTO.

    Ranking turns on `"ranking": true` and points the `chosen`/`rejected` columns
    at the two competing final assistant turns; `messages` holds the shared prompt
    (every turn up to but excluding the response). Same sharegpt tags as SFT so the
    chat template renders both halves identically — only the loss differs (stage).
    """
    return {
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "messages",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "tags": dict(_SHAREGPT_TAGS),
    }


def _kto_entry() -> dict[str, Any]:
    """LLaMA-Factory registry entry for a KTO dataset — binary-labelled completions.

    KTO labels each completion independently as desirable/undesirable via a boolean
    `kto_tag` column (no chosen/rejected pairing). `messages` holds the FULL
    conversation including the assistant response being judged. Same sharegpt tags
    as SFT so the chat template renders identically — only the loss + the kto_tag
    differ.
    """
    return {
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "kto_tag": "kto_tag",
        },
        "tags": dict(_SHAREGPT_TAGS),
    }


def _rlvr_entry() -> dict[str, Any]:
    """LLaMA-Factory registry entry for an RLVR dataset — prompt + verifiable
    ground_truth. RLVR runs on the SERVERLESS engine (reshaped to VERL at launch),
    never on the LLaMA-Factory engine, so this entry is just a faithful record of
    the on-disk columns (messages = the prompt; ground_truth = the verifiable
    target). Same sharegpt tags as SFT for the prompt rendering."""
    return {
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "ground_truth": "ground_truth",
        },
        "tags": dict(_SHAREGPT_TAGS),
    }


def _rlaif_entry() -> dict[str, Any]:
    """LLaMA-Factory registry entry for an RLAIF dataset — PROMPT-ONLY (no
    ground_truth; the AI judge scores subjectively). The columns are just the
    prompt messages, identical in shape to SFT, but kept a distinct entry for
    clarity. RLAIF runs on the SERVERLESS engine (reshaped to the recipe at
    launch), never on the LLaMA-Factory engine."""
    return {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": dict(_SHAREGPT_TAGS),
    }


def _dataset_info(train_file: str, eval_file: str, val_file: str | None = None,
                  *, kind: str = "sft") -> dict[str, Any]:
    """LLaMA-Factory dataset registry for our split.

    `kind`: "sft" (messages), "preference" (ranking chosen/rejected), "kto"
    (messages + kto_tag), "rlvr" (prompt messages + ground_truth), or "rlaif"
    (prompt messages only). The TRAIN (and VAL) entries follow `kind`, but the
    held-out EVAL entry is ALWAYS plain messages — the shared generation eval
    scores a single gold answer (for DPO/KTO/RLVR that gold is derived at persist
    time; RLAIF has no gold, so its eval rows are prompt-only), so one eval
    harness stays fair across SFT, DPO, KTO, RLVR and RLAIF models alike.

    Registers a `val_split` entry too when a validation file is present (used by
    in-training eval / early stopping); the held-out `eval_split` is separate and
    never used for stopping.
    """
    train_entry = {"preference": _preference_entry, "kto": _kto_entry,
                   "rlvr": _rlvr_entry, "rlaif": _rlaif_entry}.get(kind, _sft_entry)()
    info = {
        TRAIN_DATASET_NAME: {"file_name": train_file, **train_entry},
        EVAL_DATASET_NAME: {"file_name": eval_file, **_sft_entry()},
    }
    if val_file is not None:
        info[VAL_DATASET_NAME] = {"file_name": val_file, **train_entry}
    return info


def persist_split(
    train: list[dict],
    eval_: list[dict],
    meta: dict[str, Any],
    val: list[dict] | None = None,
) -> tuple[str, Path]:
    """Write the split + dataset_info.json. Returns (split_id, run_dir).

    When `val` is provided, also writes val.jsonl and registers it in
    dataset_info.json. The held-out eval set is always written and is never used
    for in-training stopping (see split.py)."""
    split_id = _split_id(train, eval_, val)
    store = get_store()
    run_dir = store.workdir(RUNS, split_id)

    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    (run_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_) + "\n", encoding="utf-8"
    )
    val_file = None
    if val:
        (run_dir / "val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8"
        )
        val_file = "val.jsonl"
    (run_dir / "dataset_info.json").write_text(
        json.dumps(_dataset_info("train.jsonl", "eval.jsonl", val_file), indent=2), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"splitId": split_id, **meta}, indent=2), encoding="utf-8"
    )
    store.commit(RUNS, split_id)
    return split_id, run_dir


def _preference_eval_row(pref_row: dict) -> dict:
    """Derive a messages-shaped eval row from a preference (ranking) row.

    The shared generation eval scores ONE gold answer per prompt, so we use the
    `chosen` response as the gold: prompt = the row's `messages`, plus the chosen
    assistant turn appended. This is what makes a DPO model comparable to an SFT
    model on the SAME held-out set.

    We ALSO carry the raw `chosen_ref` + `rejected_ref` strings as extra top-level
    keys (ignored by everything except eval.py) so the eval can compute the
    preference-native `chosen_win_rate` — does the model's answer look more like the
    chosen than the rejected? `messages[-1]` stays the chosen gold, so every existing
    gold-overlap metric + the train/eval disjointness invariant are unchanged.
    """
    prompt = list(pref_row.get("messages", []))
    chosen = pref_row.get("chosen")
    # `chosen` is a single assistant turn {role, content}; tolerate a bare string.
    if isinstance(chosen, dict):
        chosen_turn = {"role": chosen.get("role", "assistant"),
                       "content": chosen.get("content", "")}
    else:
        chosen_turn = {"role": "assistant", "content": str(chosen)}
    rejected = pref_row.get("rejected")
    rejected_content = (rejected.get("content", "") if isinstance(rejected, dict)
                        else str(rejected) if rejected is not None else "")
    row = {"messages": prompt + [chosen_turn], "chosen_ref": chosen_turn["content"]}
    if rejected_content:
        row["rejected_ref"] = rejected_content
    return row


def persist_preference_split(
    train: list[dict],
    meta: dict[str, Any],
    val: list[dict] | None = None,
    eval_: list[dict] | None = None,
) -> tuple[str, Path]:
    """Write a PREFERENCE split (DPO/KTO) + a ranking dataset_info.json.

    `train`/`val` rows are ranking-format: {messages:[...prompt], chosen:{...},
    rejected:{...}}. The held-out `eval_` is messages-shaped (the shared eval);
    when not supplied it is DERIVED from the train rows' `chosen` responses so the
    same generation eval + leaderboard works unchanged. Returns (split_id, dir).
    """
    # Eval defaults to the chosen-as-gold projection of the train rows so a
    # preference dataset is immediately runnable end-to-end (the leaderboard needs
    # a messages-shaped held-out set). Callers can pass an explicit eval set.
    eval_rows = eval_ if eval_ is not None else [_preference_eval_row(r) for r in train]
    split_id = _split_id(train, eval_rows, val)
    store = get_store()
    run_dir = store.workdir(RUNS, split_id)

    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    (run_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n", encoding="utf-8"
    )
    val_file = None
    if val:
        (run_dir / "val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8"
        )
        val_file = "val.jsonl"
    (run_dir / "dataset_info.json").write_text(
        json.dumps(_dataset_info("train.jsonl", "eval.jsonl", val_file, kind="preference"), indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"splitId": split_id, "shape": "preference", **meta}, indent=2),
        encoding="utf-8",
    )
    store.commit(RUNS, split_id)
    return split_id, run_dir


def _kto_eval_row(kto_row: dict) -> dict | None:
    """Derive a messages-shaped eval row from a KTO row, or None to skip it.

    A KTO row is {messages:[...,assistant], kto_tag: bool}. Only the DESIRABLE
    completions (kto_tag=true) are meaningful gold for the shared generation eval —
    an undesirable response is exactly what we DON'T want the model to produce, so
    it must never become eval gold. The messages already end in the assistant turn,
    so a desirable row is used as-is; undesirable rows are dropped (return None).
    """
    if not kto_row.get("kto_tag"):
        return None
    msgs = kto_row.get("messages")
    if not isinstance(msgs, list) or not any(m.get("role") == "assistant" for m in msgs):
        return None
    return {"messages": list(msgs)}


def preference_eval_rows(pref_rows: list[dict]) -> list[dict]:
    """Messages-shaped eval gold from a list of preference rows (the chosen
    responses). Used to build a HELD-OUT test set from a test portion."""
    return [_preference_eval_row(r) for r in pref_rows]


def kto_eval_rows(kto_rows: list[dict]) -> list[dict]:
    """Messages-shaped eval gold from a list of KTO rows — only the DESIRABLE
    completions (undesirable rows are dropped). Used to build a HELD-OUT test set."""
    return [r for r in (_kto_eval_row(x) for x in kto_rows) if r is not None]


def _rlvr_eval_row(rlvr_row: dict) -> dict:
    """Derive a messages-shaped eval row from an RLVR row.

    An RLVR row is {messages:[...prompt turns], ground_truth:"..."} — prompt-only,
    with the verifiable answer in `ground_truth`. The shared generation eval scores
    ONE gold answer per prompt, so the gold is the ground_truth appended as the
    assistant turn. This keeps an RLVR-trained model comparable to SFT/DPO/KTO
    models on the SAME held-out set (the eval never sees the reward function)."""
    prompt = list(rlvr_row.get("messages", []))
    gt = str(rlvr_row.get("ground_truth", "")).strip()
    return {"messages": prompt + [{"role": "assistant", "content": gt}]}


def rlvr_eval_rows(rlvr_rows: list[dict]) -> list[dict]:
    """Messages-shaped eval gold from RLVR rows (ground_truth as the gold answer).
    Used to build a HELD-OUT test set from a test portion."""
    return [_rlvr_eval_row(r) for r in rlvr_rows]


def _rlaif_eval_row(rlaif_row: dict) -> dict:
    """Eval row for an RLAIF prompt-only row: the prompt itself (NO gold appended).

    RLAIF has no verifiable ground_truth, so unlike _rlvr_eval_row there is no gold
    answer to score against. The held-out eval set is the prompts; the leaderboard
    judges generated responses via the LLM-as-judge path (reference-free), not
    reference-overlap metrics."""
    return {"messages": list(rlaif_row.get("messages", []))}


def rlaif_eval_rows(rlaif_rows: list[dict]) -> list[dict]:
    """Prompt-only held-out eval rows from RLAIF rows (no gold answer)."""
    return [_rlaif_eval_row(r) for r in rlaif_rows]


def persist_rlaif_split(
    train: list[dict],
    meta: dict[str, Any],
    val: list[dict] | None = None,
    eval_: list[dict] | None = None,
) -> tuple[str, Path]:
    """Write an RLAIF split (prompt-only; no ground_truth) + dataset_info.json.

    `train`/`val` rows are rlaif-format: {messages:[...prompt]}. The held-out
    `eval_` is prompt-only too (no gold to derive — the AI judge scores subjectively
    at eval time). The serverless engine reshapes the train rows to the RLAIF recipe
    at launch (see engines/serverless_data); the LF registry entry is only a record
    (RLAIF never runs on the LLaMA-Factory engine). Returns (split_id, dir)."""
    eval_rows = eval_ if eval_ is not None else [_rlaif_eval_row(r) for r in train]
    split_id = _split_id(train, eval_rows, val)
    store = get_store()
    run_dir = store.workdir(RUNS, split_id)

    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    (run_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n", encoding="utf-8"
    )
    val_file = None
    if val:
        (run_dir / "val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8"
        )
        val_file = "val.jsonl"
    (run_dir / "dataset_info.json").write_text(
        json.dumps(_dataset_info("train.jsonl", "eval.jsonl", val_file, kind="rlaif"), indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"splitId": split_id, "shape": "rlaif", **meta}, indent=2),
        encoding="utf-8",
    )
    store.commit(RUNS, split_id)
    return split_id, run_dir


def persist_rlvr_split(
    train: list[dict],
    meta: dict[str, Any],
    val: list[dict] | None = None,
    eval_: list[dict] | None = None,
) -> tuple[str, Path]:
    """Write an RLVR split (prompt + verifiable ground_truth) + dataset_info.json.

    `train`/`val` rows are rlvr-format: {messages:[...prompt], ground_truth:"..."}.
    The held-out `eval_` is messages-shaped (the shared eval); when not supplied it
    is DERIVED from the train rows' ground_truth so the same generation eval +
    leaderboard work unchanged. The serverless engine reshapes the train rows to
    VERL at launch (see engines/serverless_data.rlvr_to_verl); the LF registry
    entry below is only a record (RLVR never runs on the LLaMA-Factory engine).
    Returns (split_id, dir)."""
    eval_rows = eval_ if eval_ is not None else [_rlvr_eval_row(r) for r in train]
    split_id = _split_id(train, eval_rows, val)
    store = get_store()
    run_dir = store.workdir(RUNS, split_id)

    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    (run_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n", encoding="utf-8"
    )
    val_file = None
    if val:
        (run_dir / "val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8"
        )
        val_file = "val.jsonl"
    (run_dir / "dataset_info.json").write_text(
        json.dumps(_dataset_info("train.jsonl", "eval.jsonl", val_file, kind="rlvr"), indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"splitId": split_id, "shape": "rlvr", **meta}, indent=2),
        encoding="utf-8",
    )
    store.commit(RUNS, split_id)
    return split_id, run_dir


def persist_kto_split(
    train: list[dict],
    meta: dict[str, Any],
    val: list[dict] | None = None,
    eval_: list[dict] | None = None,
) -> tuple[str, Path]:
    """Write a KTO split (binary-labelled completions) + a kto dataset_info.json.

    `train`/`val` rows are KTO-format: {messages:[...,assistant], kto_tag: bool}.
    The held-out `eval_` is messages-shaped (the shared eval); when not supplied it
    is DERIVED from the DESIRABLE train rows (kto_tag=true) so the same generation
    eval + leaderboard work unchanged. Returns (split_id, dir).
    """
    eval_rows = eval_ if eval_ is not None else [
        r for r in (_kto_eval_row(x) for x in train) if r is not None
    ]
    split_id = _split_id(train, eval_rows, val)
    store = get_store()
    run_dir = store.workdir(RUNS, split_id)

    (run_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8"
    )
    (run_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_rows) + "\n", encoding="utf-8"
    )
    val_file = None
    if val:
        (run_dir / "val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8"
        )
        val_file = "val.jsonl"
    (run_dir / "dataset_info.json").write_text(
        json.dumps(_dataset_info("train.jsonl", "eval.jsonl", val_file, kind="kto"), indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"splitId": split_id, "shape": "kto", **meta}, indent=2),
        encoding="utf-8",
    )
    store.commit(RUNS, split_id)
    return split_id, run_dir


def _eval_id(eval_: list[dict]) -> str:
    """Content hash for an eval-only dataset (no train half)."""
    h = hashlib.sha256()
    h.update(b"eval-only")
    for r in eval_:
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()[:12]


def persist_eval_only(eval_: list[dict], meta: dict[str, Any]) -> tuple[str, Path]:
    """Persist an EVAL-ONLY dataset (no training rows) for standalone evaluation.

    Standalone evaluate scores a fine-tuned model on held-out rows; it never
    needs a training split (the eval job only consumes eval.jsonl). This writes
    eval.jsonl + a dataset_info.json registering only the eval set + meta.json,
    in the same data/runs/<id>/ layout so the eval orchestrator and the dataset
    library treat it like any other dataset.
    """
    split_id = _eval_id(eval_)
    store = get_store()
    run_dir = store.workdir(RUNS, split_id)

    (run_dir / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_) + "\n", encoding="utf-8"
    )
    # Register only the eval set; eval.jsonl is what the container reads.
    info = {EVAL_DATASET_NAME: {"file_name": "eval.jsonl", "formatting": "sharegpt",
                                "columns": {"messages": "messages"},
                                "tags": {"role_tag": "role", "content_tag": "content",
                                         "user_tag": "user", "assistant_tag": "assistant",
                                         "system_tag": "system"}}}
    (run_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps({"splitId": split_id, "evalOnly": True, "trainRows": 0,
                    "evalRows": len(eval_), **meta}, indent=2),
        encoding="utf-8",
    )
    store.commit(RUNS, split_id)
    return split_id, run_dir


def persist_curves(job_name: str, curves: dict[str, Any]) -> None:
    """Snapshot a finished job's training curves to durable storage so they
    survive CloudWatch retention/down-sampling. Keyed by job name."""
    store = get_store()
    wd = store.workdir(CURVES, job_name)
    (wd / CURVES_FILE).write_text(json.dumps(curves), encoding="utf-8")
    store.commit(CURVES, job_name)


def load_curves(job_name: str) -> dict[str, Any] | None:
    """Return a job's snapshotted curves, or None if no snapshot exists."""
    raw = get_store().read_file(CURVES, job_name, CURVES_FILE)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def has_curves(job_name: str) -> bool:
    return get_store().file_exists(CURVES, job_name, CURVES_FILE)


def split_dir(split_id: str) -> Path | None:
    """Return a local dir holding the split's files, or None if no such split.

    The store materializes the split (a no-op for local disk, an S3→/tmp sync in
    the cloud). Existence is keyed on dataset_info.json, as before.
    """
    store = get_store()
    if not store.file_exists(RUNS, split_id, "dataset_info.json"):
        return None
    return store.workdir(RUNS, split_id)


def split_meta(split_id: str) -> dict[str, Any]:
    """Return the persisted meta.json for a split (incl. its human name), or {}."""
    store = get_store()
    raw = store.read_file(RUNS, split_id, "meta.json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}


def split_name(split_id: str) -> str | None:
    """The human-friendly dataset name for a split, if one was given."""
    return split_meta(split_id).get("name")


def is_dataset_archived(split_id: str) -> bool:
    """Whether a dataset is archived (hidden from pickers + leaderboard)."""
    return bool(split_meta(split_id).get("archived"))


def list_datasets(include_archived: bool = False) -> list[dict[str, Any]]:
    """Persisted datasets (splits) — the dataset library. Newest first by key
    mtime (no creation timestamp in meta; mtime is a proxy).

    By default ARCHIVED datasets are excluded, so every consumer (Fine-tune +
    Eval pickers, leaderboard) hides them automatically — the Datasets page is
    the single master that manages availability. Pass include_archived=True
    (the Datasets page does) to see archived ones too."""
    store = get_store()
    out: list[dict[str, Any]] = []
    for key in store.list_keys(RUNS):
        if not store.file_exists(RUNS, key, "dataset_info.json"):
            continue
        meta = split_meta(key)
        if meta.get("archived") and not include_archived:
            continue
        out.append(
            {
                "splitId": key,
                "name": meta.get("name") or None,
                "trainRows": meta.get("trainRows"),
                "evalRows": meta.get("evalRows"),
                "mode": meta.get("mode"),
                # Data shape: "preference" (DPO chosen/rejected) vs SFT messages
                # (absent/"sft"). Lets the picker filter datasets to the objective.
                "shape": meta.get("shape", "sft"),
                "evalOnly": meta.get("evalOnly", False),
                "archived": meta.get("archived", False),
                # Provenance: "huggingface" drives the HF badge in the UI; the
                # source dataset id is shown alongside.
                "source": meta.get("source"),
                "hfDataset": meta.get("hfDataset"),
                # Dataset-investigation recommendation (if run) — lets the
                # leaderboard default its 'Rank by' to the advised metric, and the
                # FineTune RLVR step pre-offer the verifiable reward that mirrors it.
                "recommendedRankMetric": meta.get("recommendedRankMetric"),
                "recommendedRewardMetric": meta.get("recommendedRewardMetric"),
                # KTO class-balance loss-weight recommendation (λ_D / λ_U) from the
                # profiler, so the FineTune KTO step can one-click pre-fill them.
                "recommendedChosenWeight": meta.get("recommendedChosenWeight"),
                "recommendedRejectedWeight": meta.get("recommendedRejectedWeight"),
                # Optional validation set — gates early stopping.
                "hasVal": meta.get("hasVal", False),
                "valRows": meta.get("valRows", 0),
                "valMode": meta.get("valMode", ""),
                "hasBaseline": store.file_exists(RUNS, key, "sonnet_baseline.json"),
                "mtime": store.key_mtime(RUNS, key) or 0.0,
            }
        )
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def set_dataset_archived(split_id: str, archived: bool) -> bool:
    """Archive (hide) or restore a dataset. Soft display state stored in
    meta.json — never touches the data files (datasets back races + the
    leaderboard). Returns False if the dataset doesn't exist."""
    return _update_dataset_meta(split_id, {"archived": archived})


def _update_dataset_meta(split_id: str, fields: dict[str, Any]) -> bool:
    """Merge `fields` into a dataset's meta.json (never touches data files).
    Returns False if the dataset doesn't exist."""
    store = get_store()
    if not store.file_exists(RUNS, split_id, "dataset_info.json"):
        return False
    meta = split_meta(split_id)
    meta.update(fields)
    wd = store.workdir(RUNS, split_id)
    (wd / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    store.commit(RUNS, split_id)
    return True


def set_recommended_metric(
    split_id: str,
    rank_metric: str,
    also_watch: list[str],
    reward_metric: str | None = None,
) -> bool:
    """Record the dataset-investigation recommended ranking metric on the dataset
    so the leaderboard can default its 'Rank by' to it (user can still override).
    `reward_metric` (the verifiable RLVR reward that mirrors the rank metric, or
    None) closes the reward↔metric loop — the FineTune RLVR step pre-offers it."""
    return _update_dataset_meta(
        split_id,
        {
            "recommendedRankMetric": rank_metric,
            "recommendedAlsoWatch": also_watch,
            "recommendedRewardMetric": reward_metric,
        },
    )


def set_recommended_kto_weights(
    split_id: str, chosen_weight: float, rejected_weight: float
) -> bool:
    """Record the profiler's KTO class-balance loss-weight recommendation (λ_D / λ_U)
    on the dataset so the FineTune KTO step can one-click pre-fill them — the KTO
    analog of the RLVR reward recommendation. Persisted only when non-default
    (the caller skips a balanced 1.0/1.0); user can still override in the form."""
    return _update_dataset_meta(
        split_id,
        {
            "recommendedChosenWeight": chosen_weight,
            "recommendedRejectedWeight": rejected_weight,
        },
    )
