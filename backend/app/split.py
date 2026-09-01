# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Train/eval(/val) split + leakage assertion.

The platform's core methodology rule: the eval set MUST be disjoint from the
training set (eval ∩ train = ∅). Evaluating on rows seen in training inflates
scores via memorization. This module enforces that two ways:

  - assert mode  : caller supplies separate train + eval JSONL; we verify
                   disjointness and report any overlapping rows.
  - auto mode    : caller supplies one JSONL; we deterministically split into
                   train/eval (disjoint by construction) using a seed.

OPTIONAL VALIDATION SET: a third "val" set can be added two ways —
an explicit val file (3-file mode) OR auto-carving X% of train. Val is what
in-training evaluation (and early stopping) uses; the HELD-OUT EVAL SET STAYS
UNTOUCHED so leaderboard comparison stays fair across models that stop at
different points. When a val set is present we assert FULL PAIRWISE disjointness
(train ∩ val, train ∩ eval, val ∩ eval all = ∅). Val is optional — datasets with
no val keep working exactly as before.

The HARD pass/fail key is the full `messages` array (exact duplicate). We also
compute a SOFT prompt-only collision count (same user turns, different
assistant answer) and surface it as a leakage warning — same input appearing in
two splits is a smell even when the full row differs.

Rows are validated with the row validator first; only valid rows are split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .validation import validate_jsonl

# How many sample rows / overlap examples to return to the UI.
PREVIEW_LIMIT = 5
OVERLAP_EXAMPLE_LIMIT = 10


def _canonical(obj: Any) -> str:
    """Stable JSON encoding for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def full_row_key(row: dict) -> str:
    """HARD key: hash of the entire `messages` array (exact-duplicate match)."""
    return hashlib.sha256(_canonical(row["messages"]).encode("utf-8")).hexdigest()


def prompt_key(row: dict) -> str:
    """SOFT key: hash of the non-assistant turns (the input/prompt side).

    Used only for the leakage warning, not the pass/fail assertion.
    """
    prompt_turns = [t for t in row["messages"] if t.get("role") != "assistant"]
    return hashlib.sha256(_canonical(prompt_turns).encode("utf-8")).hexdigest()


@dataclass
class SplitReport:
    ok: bool  # True only if all files valid AND no full-row overlap (pairwise)
    train_rows: int
    eval_rows: int
    # Hard overlap: identical full rows present in two splits.
    overlap_count: int
    overlap_examples: list[dict] = field(default_factory=list)
    # Soft leakage: same prompt (input turns) in two splits, regardless of answer.
    prompt_collision_count: int = 0
    # Per-file validation problems (bad rows are excluded from the split).
    train_invalid_rows: int = 0
    eval_invalid_rows: int = 0
    # Per-row {line, message} errors (folded-in validation detail for the UI).
    train_errors: list[dict] = field(default_factory=list)
    eval_errors: list[dict] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)  # human-readable notes
    train_preview: list[dict] = field(default_factory=list)
    eval_preview: list[dict] = field(default_factory=list)
    # Echo of how the split was produced, for the UI.
    mode: str = "assert"
    seed: int | None = None
    eval_ratio: float | None = None
    # --- Optional validation set ---
    has_val: bool = False
    # How the val set was produced: "" (none) | "file" (explicit) | "carve" (% of train).
    val_mode: str = ""
    val_rows: int = 0
    val_ratio: float | None = None  # carve fraction when val_mode == "carve"
    val_invalid_rows: int = 0
    val_errors: list[dict] = field(default_factory=list)
    val_preview: list[dict] = field(default_factory=list)
    # Full row lists (NOT serialized to the client) — used by storage to persist
    # the split so later steps can reference it by id.
    train_rows_full: list[dict] = field(default_factory=list, repr=False)
    eval_rows_full: list[dict] = field(default_factory=list, repr=False)
    val_rows_full: list[dict] = field(default_factory=list, repr=False)
    # Set by the persistence layer once the split is written to disk.
    split_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "seed": self.seed,
            "evalRatio": self.eval_ratio,
            "splitId": self.split_id,
            "trainRows": self.train_rows,
            "evalRows": self.eval_rows,
            "overlapCount": self.overlap_count,
            "overlapExamples": self.overlap_examples,
            "promptCollisionCount": self.prompt_collision_count,
            "trainInvalidRows": self.train_invalid_rows,
            "evalInvalidRows": self.eval_invalid_rows,
            "trainErrors": self.train_errors,
            "evalErrors": self.eval_errors,
            "messages": self.messages,
            "trainPreview": self.train_preview,
            "evalPreview": self.eval_preview,
            "hasVal": self.has_val,
            "valMode": self.val_mode,
            "valRows": self.val_rows,
            "valRatio": self.val_ratio,
            "valInvalidRows": self.val_invalid_rows,
            "valErrors": self.val_errors,
            "valPreview": self.val_preview,
        }


def _valid_rows(text: str) -> tuple[list[dict], int, list[dict]]:
    """Return (valid rows, invalid_row_count, per-row errors).

    The row validator gives the authoritative invalid-row count + the
    {line, message} errors (folded into the split result so the UI can show
    exactly which rows are bad — no separate Validation page needed). We re-parse
    here to recover the full set of valid row objects for splitting.
    """
    report = validate_jsonl(text)
    errors = [{"line": e.line, "message": e.message} for e in report.errors]
    good: list[dict] = []
    for raw in text.splitlines():
        if raw.strip() == "":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and _row_is_valid(obj):
            good.append(obj)
    return good, report.invalid_rows, errors


def _row_is_valid(row: dict) -> bool:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    saw_assistant = False
    for t in msgs:
        if not isinstance(t, dict):
            return False
        if t.get("role") not in {"system", "user", "assistant", "tool"}:
            return False
        c = t.get("content")
        if not isinstance(c, str) or c.strip() == "":
            return False
        if t["role"] == "assistant":
            saw_assistant = True
    return saw_assistant


def _overlap(train: list[dict], eval_: list[dict]) -> tuple[list[dict], int]:
    """Return (full-row overlap examples, prompt-collision count)."""
    train_full = {full_row_key(r) for r in train}
    overlaps = [r for r in eval_ if full_row_key(r) in train_full]

    train_prompts = {prompt_key(r) for r in train}
    prompt_collisions = sum(1 for r in eval_ if prompt_key(r) in train_prompts)

    return overlaps, prompt_collisions


def _gold_answer(row: dict) -> str:
    """The row's last assistant turn (the target/label) — the stratum for
    stratified sampling. Normalized (lower/trim) so trivial casing differences
    don't fragment a class."""
    msgs = row.get("messages", [])
    golds = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    return (golds[-1] if golds else "").strip().lower()


def _is_label_dataset(rows: list[dict], sample: int = 200) -> bool:
    """True if the dataset looks like a CLASSIFICATION task — the only case where
    stratified sampling is meaningful (a small set of discrete labels shared
    across rows). Reuses the profiler's detect_task on a sample of gold answers,
    and additionally requires the label cardinality to be low relative to the row
    count (so all-unique freeform answers never qualify)."""
    if len(rows) < 4:
        return False
    try:
        from .profiler import detect_task
    except Exception:  # noqa: BLE001 — if profiler import fails, just don't stratify
        return False
    sampled = rows[:sample]
    golds = [_gold_answer(r) for r in sampled]
    label_like = sum(1 for g in golds if detect_task(g) == "label")
    if label_like < len(sampled) * 0.9:  # ≥90% of rows must be label-shaped
        return False
    # Guard against "short but all-unique" (e.g. short freeform) — require the
    # distinct-label count to be a small fraction of the rows.
    distinct = len(set(golds))
    return distinct <= max(2, len(sampled) // 5)


def _stratified_indices(
    rows: list[dict], ratio: float, salt: str
) -> tuple[list[dict], list[dict]]:
    """Proportional (mirror) stratified split: take `ratio` from EACH class so the
    two halves share the source's class distribution. Deterministic (hash ordering
    within each class, keyed by `salt` — no RNG). Disjoint by construction (each
    row goes to exactly one half). Returns (majority_remainder, held_out)."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_gold_answer(r)].append(r)

    def order_key(r: dict) -> str:
        return hashlib.sha256(f"{salt}:{_canonical(r['messages'])}".encode("utf-8")).hexdigest()

    remainder: list[dict] = []
    held: list[dict] = []
    # Stable class iteration order (sorted) so the whole split is reproducible.
    for cls in sorted(groups):
        members = sorted(groups[cls], key=order_key)
        n = len(members)
        # Round per class; ensure at least 1 held-out row for classes with ≥2
        # members (so every class is represented), but never take the whole class.
        n_held = round(n * ratio)
        if n >= 2:
            n_held = max(1, min(n_held, n - 1))
        else:
            n_held = 0  # singleton class → keep it in the remainder (can't split)
        held.extend(members[:n_held])
        remainder.extend(members[n_held:])
    return remainder, held


def _carve_val(train: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministically hold out val_ratio of train as a validation set.

    Same hash-ordering as auto_split (no RNG, reproducible). Returns
    (remaining_train, val). The carve happens AFTER train is already disjoint
    from eval, and val is a subset of train, so val stays disjoint from eval too.
    Always leaves at least one training row.
    """
    def order_key(r: dict) -> str:
        return hashlib.sha256(f"val:{seed}:{_canonical(r['messages'])}".encode("utf-8")).hexdigest()

    ordered = sorted(train, key=order_key)
    n = len(ordered)
    n_val = round(n * val_ratio)
    n_val = max(1, min(n_val, n - 1)) if n >= 2 else 0  # keep ≥1 train row
    val = ordered[:n_val]
    remaining = ordered[n_val:]
    return remaining, val


def assert_disjoint(
    train_text: str,
    eval_text: str,
    val_text: str | None = None,
    val_ratio: float | None = None,
    seed: int = 42,
) -> SplitReport:
    """Two-file (or three-file) mode: validate, assert pairwise disjointness.

    Always asserts eval ∩ train = ∅. Optionally adds a validation set:
      - val_text given (3-file mode): assert train/val/eval pairwise disjoint.
      - val_ratio given (auto-carve): hold out that fraction of TRAIN as val
        (val ⊂ train, so it inherits train's disjointness from eval).
    val_text takes precedence if both are given. Val is optional — neither given
    means the historical train+eval behaviour, unchanged.
    """
    train, train_bad, train_errs = _valid_rows(train_text)
    eval_, eval_bad, eval_errs = _valid_rows(eval_text)

    val: list[dict] = []
    val_bad = 0
    val_errs: list[dict] = []
    val_mode = ""
    used_ratio: float | None = None

    if val_text is not None:
        val, val_bad, val_errs = _valid_rows(val_text)
        val_mode = "file"

    has_invalid = train_bad > 0 or eval_bad > 0 or val_bad > 0

    # Carve happens only when no explicit val file AND a ratio is requested AND
    # the inputs are clean (we won't carve from a partially-invalid train set).
    if not val_mode and val_ratio is not None and not has_invalid and len(train) >= 2:
        train, val = _carve_val(train, val_ratio, seed)
        val_mode = "carve"
        used_ratio = val_ratio

    # Pairwise full-row overlaps. train↔eval is the methodology-critical one.
    te_overlaps, te_prompts = _overlap(train, eval_)
    overlaps = list(te_overlaps)
    prompt_collisions = te_prompts
    tv_overlaps = ve_overlaps = []
    if val:
        tv_overlaps, tv_prompts = _overlap(train, val)
        ve_overlaps, ve_prompts = _overlap(val, eval_)
        overlaps += tv_overlaps + ve_overlaps
        prompt_collisions += tv_prompts + ve_prompts

    messages: list[str] = []
    if train_bad:
        messages.append(f"{train_bad} invalid row(s) in train — fix these before splitting")
    if eval_bad:
        messages.append(f"{eval_bad} invalid row(s) in eval — fix these before splitting")
    if val_bad:
        messages.append(f"{val_bad} invalid row(s) in val — fix these before splitting")
    if has_invalid:
        messages.append("split blocked: validate and fix the dataset(s) first")
    elif overlaps:
        if te_overlaps:
            messages.append(
                f"{len(te_overlaps)} eval row(s) are identical to training rows — eval set is NOT disjoint"
            )
        if tv_overlaps:
            messages.append(f"{len(tv_overlaps)} val row(s) are identical to training rows")
        if ve_overlaps:
            messages.append(f"{len(ve_overlaps)} val row(s) are identical to eval rows")
    elif prompt_collisions:
        messages.append(
            f"{prompt_collisions} prompt(s) appear in more than one split with a "
            f"different answer (possible leakage)"
        )
    else:
        if val:
            messages.append(
                f"train ∩ val ∩ eval all disjoint ✓ "
                f"({len(train)} train / {len(val)} val / {len(eval_)} eval"
                + (f", carved {val_mode == 'carve' and f'{used_ratio:g}' or ''} of train" if val_mode == "carve" else "")
                + ")"
            )
        else:
            messages.append("eval ∩ train = ∅ (disjoint) ✓")

    return SplitReport(
        ok=(len(train) > 0 and len(eval_) > 0 and len(overlaps) == 0 and not has_invalid),
        mode="assert",
        train_rows=len(train),
        eval_rows=len(eval_),
        overlap_count=len(overlaps),
        overlap_examples=overlaps[:OVERLAP_EXAMPLE_LIMIT],
        prompt_collision_count=prompt_collisions,
        train_invalid_rows=train_bad,
        eval_invalid_rows=eval_bad,
        train_errors=train_errs,
        eval_errors=eval_errs,
        messages=messages,
        train_preview=train[:PREVIEW_LIMIT],
        eval_preview=eval_[:PREVIEW_LIMIT],
        train_rows_full=train,
        eval_rows_full=eval_,
        has_val=bool(val),
        val_mode=val_mode if val else "",
        val_rows=len(val),
        val_ratio=used_ratio,
        val_invalid_rows=val_bad,
        val_errors=val_errs,
        val_preview=val[:PREVIEW_LIMIT],
        val_rows_full=val,
    )


def auto_split(
    text: str,
    eval_ratio: float = 0.2,
    seed: int = 42,
    val_ratio: float | None = None,
    stratify: bool = False,
) -> SplitReport:
    """One-file mode: deterministic shuffle + split, disjoint by construction.

    Dedupes exact-duplicate full rows BEFORE splitting so the same row can't land
    in two halves (which would defeat the disjointness guarantee). When
    val_ratio is given, additionally carves that fraction of the TRAIN portion as
    a validation set (val ⊂ train ⇒ disjoint from eval by construction).

    stratify=True keeps each CLASS proportionally represented across train/val/
    eval (so eval mirrors train's class distribution) — but ONLY when the dataset
    is detected as a classification/label task. For non-label data (freeform/JSON,
    where there's no discrete class) it transparently falls back to the normal
    random split and notes that in the report.
    """
    rows, bad, errs = _valid_rows(text)

    # Block on any invalid row — same integrity bar as the assert mode. We won't
    # auto-split a partially-dropped dataset.
    if bad > 0:
        return SplitReport(
            ok=False,
            mode="auto",
            seed=seed,
            eval_ratio=eval_ratio,
            train_rows=0,
            eval_rows=0,
            overlap_count=0,
            prompt_collision_count=0,
            train_invalid_rows=bad,
            eval_invalid_rows=0,
            train_errors=errs,
            messages=[
                f"{bad} invalid row(s) — fix these before splitting",
                "split blocked: validate and fix the dataset first",
            ],
        )

    # Drop exact duplicate full rows so a dupe can't straddle the split.
    seen: set[str] = set()
    unique: list[dict] = []
    dup_count = 0
    for r in rows:
        k = full_row_key(r)
        if k in seen:
            dup_count += 1
            continue
        seen.add(k)
        unique.append(r)

    # Choose split strategy. Stratified only applies to label/classification data
    # (needs discrete classes); otherwise fall back to the plain random split.
    stratified_applied = False
    if stratify and _is_label_dataset(unique):
        train, eval_ = _stratified_indices(unique, eval_ratio, salt=str(seed))
        stratified_applied = True
        n = len(unique)
    else:
        # Deterministic shuffle: order by hash of (seed, canonical row). No RNG, so
        # results are reproducible across runs and machines for a given seed.
        def order_key(r: dict) -> str:
            return hashlib.sha256(f"{seed}:{_canonical(r['messages'])}".encode("utf-8")).hexdigest()

        ordered = sorted(unique, key=order_key)
        n = len(ordered)
        n_eval = max(1, round(n * eval_ratio)) if n >= 2 else 0
        eval_ = ordered[:n_eval]
        train = ordered[n_eval:]

    # Optionally carve a validation set out of train (val ⊂ train ⇒ disjoint
    # from eval by construction). Stratified carve when stratify is active so val
    # mirrors the class distribution too.
    #
    # `val_ratio` is a fraction of the WHOLE dataset (so train/val/test sum to 1,
    # matching the UI's intuitive 3-way split + the DPO/KTO paths). Since the carve
    # happens on the post-eval `train` slice, convert: a whole-fraction v over a
    # train slice of size (1 - eval_ratio) is v / (1 - eval_ratio) of that slice.
    val: list[dict] = []
    used_val_ratio: float | None = None
    if val_ratio is not None and len(train) >= 2:
        denom = max(1e-9, 1.0 - eval_ratio)
        val_of_train = min(0.95, val_ratio / denom)  # clamp so train never empties
        if stratified_applied:
            train, val = _stratified_indices(train, val_of_train, salt=f"val:{seed}")
        else:
            train, val = _carve_val(train, val_of_train, seed)
        used_val_ratio = val_ratio

    messages: list[str] = []
    if bad:
        messages.append(f"{bad} invalid row(s) excluded before split")
    if dup_count:
        messages.append(f"{dup_count} exact-duplicate row(s) removed before split")
    if stratify and not stratified_applied:
        messages.append(
            "stratified sampling requested but not applied — dataset isn't a "
            "classification/label task (no discrete classes to stratify on); used random split"
        )
    elif stratified_applied:
        messages.append("stratified sampling: each class proportionally represented across splits ✓")
    if val:
        messages.append(
            f"split {n} unique rows → {len(train)} train / {len(val)} val / {len(eval_)} eval "
            f"(eval {eval_ratio:g}, val {used_val_ratio:g} of post-eval train, seed={seed}); "
            f"disjoint by construction ✓"
        )
    else:
        messages.append(
            f"split {n} unique rows → {len(train)} train / {len(eval_)} eval "
            f"(ratio={eval_ratio}, seed={seed}); disjoint by construction ✓"
        )

    return SplitReport(
        ok=(len(train) > 0 and len(eval_) > 0),
        mode="auto",
        seed=seed,
        eval_ratio=eval_ratio,
        train_rows=len(train),
        eval_rows=len(eval_),
        overlap_count=0,  # guaranteed disjoint by construction
        overlap_examples=[],
        prompt_collision_count=0,
        train_invalid_rows=bad,
        eval_invalid_rows=0,
        messages=messages,
        train_preview=train[:PREVIEW_LIMIT],
        eval_preview=eval_[:PREVIEW_LIMIT],
        train_rows_full=train,
        eval_rows_full=eval_,
        has_val=bool(val),
        val_mode="carve" if val else "",
        val_rows=len(val),
        val_ratio=used_val_ratio,
        val_preview=val[:PREVIEW_LIMIT],
        val_rows_full=val,
    )
