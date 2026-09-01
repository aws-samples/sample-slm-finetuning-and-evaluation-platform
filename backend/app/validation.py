# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Chat-template JSONL validation.

The platform's training format is one JSON object per line, each with a
`messages` array of `{role, content}` turns (OpenAI/LLaMA-Factory ShareGPT-style
chat template). This module parses + validates that format and returns a
structured report the UI can render, without raising on bad input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Roles we accept in a chat-template turn. `system` is optional and, when
# present, should be the first turn; `tool` is allowed for tool-calling datasets.
VALID_ROLES = {"system", "user", "assistant", "tool"}

# How many parsed rows to send back as a preview to the UI.
PREVIEW_LIMIT = 5


@dataclass
class RowError:
    line: int  # 1-based line number in the uploaded file
    message: str


@dataclass
class ValidationReport:
    valid: bool
    total_lines: int  # non-empty lines seen
    valid_rows: int
    invalid_rows: int
    errors: list[RowError] = field(default_factory=list)
    preview: list[dict[str, Any]] = field(default_factory=list)
    # Aggregate stats useful for a quick dataset sanity check.
    role_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "totalLines": self.total_lines,
            "validRows": self.valid_rows,
            "invalidRows": self.invalid_rows,
            "errors": [{"line": e.line, "message": e.message} for e in self.errors],
            "preview": self.preview,
            "roleCounts": self.role_counts,
        }


def _validate_messages(messages: Any) -> str | None:
    """Return an error string if `messages` is malformed, else None."""
    if not isinstance(messages, list):
        return "`messages` must be a list"
    if len(messages) == 0:
        return "`messages` is empty"

    saw_assistant = False
    for i, turn in enumerate(messages):
        if not isinstance(turn, dict):
            return f"turn {i} is not an object"
        role = turn.get("role")
        content = turn.get("content")
        if role not in VALID_ROLES:
            return f"turn {i} has invalid role {role!r} (allowed: {sorted(VALID_ROLES)})"
        if not isinstance(content, str) or content.strip() == "":
            return f"turn {i} ({role}) has empty or non-string content"
        if role == "assistant":
            saw_assistant = True

    if not saw_assistant:
        return "no `assistant` turn (need at least one target response to train on)"
    return None


def validate_jsonl(text: str, max_errors: int = 50) -> ValidationReport:
    """Validate raw JSONL text and return a structured report.

    Never raises on bad data — malformed lines are recorded as errors.
    """
    total = 0
    valid_rows = 0
    errors: list[RowError] = []
    preview: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue  # skip blank lines, don't count them
        total += 1

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, f"invalid JSON: {e.msg}"))
            continue

        if not isinstance(obj, dict):
            if len(errors) < max_errors:
                errors.append(RowError(lineno, "line is not a JSON object"))
            continue

        if "messages" not in obj:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, "missing `messages` field"))
            continue

        err = _validate_messages(obj["messages"])
        if err is not None:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, err))
            continue

        # Row is valid.
        valid_rows += 1
        for turn in obj["messages"]:
            role_counts[turn["role"]] = role_counts.get(turn["role"], 0) + 1
        if len(preview) < PREVIEW_LIMIT:
            preview.append(obj)

    invalid_rows = total - valid_rows
    return ValidationReport(
        valid=(total > 0 and invalid_rows == 0),
        total_lines=total,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        errors=errors,
        preview=preview,
        role_counts=role_counts,
    )


# --- Preference (DPO/KTO) JSONL --------------------------------------------- #
# A preference row pairs a prompt with a CHOSEN and a REJECTED response. We accept
# a few common shapes and normalize to the canonical ranking row the storage layer
# writes: {messages: [...prompt turns], chosen: {role, content}, rejected: {...}}.
#   - prompt: either a `messages` array (preferred) OR a bare `prompt` string
#             (wrapped as a single user turn).
#   - chosen/rejected: either a string (wrapped as an assistant turn) OR an object
#             {role, content} (role defaults to assistant).


def _as_assistant_turn(v: Any) -> dict[str, str] | None:
    """Normalize a chosen/rejected value to an {role:'assistant', content} turn, or
    None if it isn't a usable non-empty response."""
    if isinstance(v, str):
        return {"role": "assistant", "content": v} if v.strip() else None
    if isinstance(v, dict):
        content = v.get("content")
        if isinstance(content, str) and content.strip():
            return {"role": v.get("role", "assistant"), "content": content}
    return None


def _normalize_preference_row(obj: dict) -> tuple[dict | None, str | None]:
    """(canonical_row, error). Accepts messages-or-prompt + chosen/rejected."""
    # Prompt turns.
    if "messages" in obj:
        msgs = obj["messages"]
        err = _validate_prompt_messages(msgs)
        if err:
            return None, err
        prompt = list(msgs)
    elif isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
        prompt = [{"role": "user", "content": obj["prompt"]}]
    else:
        return None, "row needs a `messages` array or a non-empty `prompt` string"

    chosen = _as_assistant_turn(obj.get("chosen"))
    rejected = _as_assistant_turn(obj.get("rejected"))
    if chosen is None:
        return None, "missing or empty `chosen` response"
    if rejected is None:
        return None, "missing or empty `rejected` response"
    return {"messages": prompt, "chosen": chosen, "rejected": rejected}, None


def _validate_prompt_messages(messages: Any) -> str | None:
    """Like _validate_messages but for a PROMPT (no trailing assistant required —
    the responses live in chosen/rejected). Must be a non-empty list of valid
    {role, content} turns."""
    if not isinstance(messages, list) or not messages:
        return "`messages` must be a non-empty list"
    for i, turn in enumerate(messages):
        if not isinstance(turn, dict):
            return f"turn {i} is not an object"
        if turn.get("role") not in VALID_ROLES:
            return f"turn {i} has invalid role {turn.get('role')!r}"
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            return f"turn {i} has empty or non-string content"
    # The prompt must NOT already contain the response: a trailing assistant turn
    # means the user put prompt+answer in `messages` plus chosen/rejected, which
    # would train DPO on a double assistant turn. The response belongs in
    # chosen/rejected, not the prompt.
    if messages[-1].get("role") == "assistant":
        return "`messages` (the prompt) must not end with an assistant turn — the response goes in chosen/rejected"
    return None


def parse_preference_jsonl(text: str, max_errors: int = 50) -> tuple[list[dict], ValidationReport]:
    """Parse + validate preference JSONL. Returns (canonical_rows, report). Never
    raises — malformed lines are recorded as errors and excluded from the rows."""
    rows: list[dict] = []
    errors: list[RowError] = []
    preview: list[dict[str, Any]] = []
    total = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue
        total += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, f"invalid JSON: {e.msg}"))
            continue
        if not isinstance(obj, dict):
            if len(errors) < max_errors:
                errors.append(RowError(lineno, "line is not a JSON object"))
            continue
        row, err = _normalize_preference_row(obj)
        if err is not None:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, err))
            continue
        rows.append(row)
        if len(preview) < PREVIEW_LIMIT:
            preview.append(row)
    invalid = total - len(rows)
    report = ValidationReport(
        valid=(total > 0 and invalid == 0),
        total_lines=total,
        valid_rows=len(rows),
        invalid_rows=invalid,
        errors=errors,
        preview=preview,
        role_counts={},
    )
    return rows, report


# --- KTO (binary-feedback) JSONL -------------------------------------------- #
# A KTO row is a full conversation ending in an assistant response, plus a boolean
# label saying whether that response is DESIRABLE. We accept a few shapes and
# normalize to {messages:[...,assistant], kto_tag: bool}:
#   - messages: a chat list ending in an assistant turn (preferred), OR a
#     prompt/completion pair (prompt → user turn, completion → assistant turn).
#   - the label: `kto_tag` (bool), or `label` (bool / "good"/"bad" / 1/0).

_TRUE_LABELS = {"true", "good", "desirable", "1", "yes", "chosen", "positive"}
_FALSE_LABELS = {"false", "bad", "undesirable", "0", "no", "rejected", "negative", "-1"}


def _coerce_kto_tag(v: Any) -> bool | None:
    """Normalize a KTO label to a bool, or None if it isn't a usable label."""
    if isinstance(v, bool):
        return v
    # Numeric labels: 1/0 and the common -1 (bad) / 1 (good) convention.
    if isinstance(v, (int, float)) and v in (0, 1, -1):
        return v == 1
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE_LABELS:
            return True
        if s in _FALSE_LABELS:
            return False
    return None


def _normalize_kto_row(obj: dict) -> tuple[dict | None, str | None]:
    """(canonical_row, error). Accepts messages-or-prompt/completion + a label."""
    # Conversation ending in the assistant response being judged.
    if "messages" in obj:
        msgs = obj["messages"]
        if not isinstance(msgs, list) or not msgs:
            return None, "`messages` must be a non-empty list"
        last = msgs[-1]
        if not isinstance(last, dict) or last.get("role") != "assistant":
            return None, "`messages` must end with an assistant turn (the judged response)"
        err = _validate_messages(msgs)
        if err:
            return None, err
        messages = list(msgs)
    elif isinstance(obj.get("prompt"), str) and isinstance(obj.get("completion"), str):
        if not obj["prompt"].strip() or not obj["completion"].strip():
            return None, "`prompt` and `completion` must both be non-empty"
        messages = [
            {"role": "user", "content": obj["prompt"]},
            {"role": "assistant", "content": obj["completion"]},
        ]
    else:
        return None, "row needs `messages` (ending in assistant) or `prompt`+`completion`"

    raw_label = obj.get("kto_tag", obj.get("label"))
    tag = _coerce_kto_tag(raw_label)
    if tag is None:
        return None, "missing or unrecognized label (need kto_tag/label = good|bad / true|false)"
    return {"messages": messages, "kto_tag": tag}, None


def parse_kto_jsonl(text: str, max_errors: int = 50) -> tuple[list[dict], ValidationReport]:
    """Parse + validate KTO JSONL. Returns (canonical_rows, report). Never raises."""
    rows: list[dict] = []
    errors: list[RowError] = []
    preview: list[dict[str, Any]] = []
    total = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue
        total += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, f"invalid JSON: {e.msg}"))
            continue
        if not isinstance(obj, dict):
            if len(errors) < max_errors:
                errors.append(RowError(lineno, "line is not a JSON object"))
            continue
        row, err = _normalize_kto_row(obj)
        if err is not None:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, err))
            continue
        rows.append(row)
        if len(preview) < PREVIEW_LIMIT:
            preview.append(row)
    invalid = total - len(rows)
    report = ValidationReport(
        valid=(total > 0 and invalid == 0),
        total_lines=total,
        valid_rows=len(rows),
        invalid_rows=invalid,
        errors=errors,
        preview=preview,
        role_counts={},
    )
    return rows, report


def _normalize_rlvr_row(obj: dict) -> tuple[dict | None, str | None]:
    """(canonical_row, error) for an RLVR row.

    An RLVR row is a PROMPT (no answer) + an explicit verifiable target:
        {messages:[...prompt turns], ground_truth:"..."}
    Also accepts a bare `prompt` string. The `ground_truth` is the answer the
    reward fn (a gsm8k/prime_math preset or a custom reward) checks against — it is REQUIRED
    and kept separate from any worked solution, so 'correct' is unambiguous and the
    dataset can't be silently misused as plain SFT data."""
    if "messages" in obj:
        msgs = obj["messages"]
        err = _validate_prompt_messages(msgs)  # non-empty, valid turns, no trailing assistant
        if err:
            return None, err
        prompt = list(msgs)
    elif isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
        prompt = [{"role": "user", "content": obj["prompt"]}]
    else:
        return None, "row needs a `messages` array or a non-empty `prompt` string"

    gt = obj.get("ground_truth", obj.get("answer"))
    if not isinstance(gt, str) or not gt.strip():
        return None, "missing or empty `ground_truth` (the verifiable target the reward function checks)"
    return {"messages": prompt, "ground_truth": gt.strip()}, None


# --- Shape autodetection (for the Guided Fine-tuning agent) ------------------ #
# The guided agent hides the platform's jargon-y 7-way dataset-type control: when a
# user drops a raw JSONL it must DECIDE which shape the data is, in plain language,
# rather than ask "is this SFT/DPO/KTO/RLVR/RLAIF?". This sniffs a sample of rows
# and classifies them by KEY SIGNATURE, ordered most-specific-first so an ambiguous
# row (e.g. a preference row also has `messages`) lands in the most specific bucket.
#
# It reuses the same row normalizers the real importers use (_normalize_preference_row
# etc.), so detection can't drift from what each dataset endpoint will actually
# accept. Returns the detected shape + a plain-language label + per-shape match
# rates, so the agent can confirm ("I see a good and a worse answer per prompt — I'll
# teach the model to prefer the good ones") and fall back to asking on a tie.

# Plain-language, jargon-free descriptions per shape (what the agent SAYS).
SHAPE_PLAIN = {
    "sft": "examples of the task done right (a prompt and the ideal answer)",
    "preference": "pairs showing a better and a worse answer for each prompt",
    "kto": "answers each marked simply good or bad",
    "rlvr": "prompts with a checkable correct answer (e.g. a number)",
    "rlaif": "prompts only, with no example answers",
    "unknown": "an unrecognized format",
}

# Detection order: most specific signature first. SFT (a plain assistant reply) is
# the LEAST specific, so it's the final fallback — many shapes also contain messages.
_DETECT_ORDER = ("preference", "kto", "rlvr", "rlaif", "sft")


def _sample_objects(text: str, limit: int) -> list[dict]:
    """First `limit` parseable JSON objects from JSONL text (skips blanks/garbage)."""
    out: list[dict] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
        if len(out) >= limit:
            break
    return out


def detect_shape(text: str, sample: int = 200) -> dict[str, Any]:
    """Classify raw JSONL into a dataset SHAPE without it being split/persisted yet.

    Returns {shape, label, confidence, rows, matchRates:{shape:rate}, ambiguous}.
    `shape` is one of sft|preference|kto|rlvr|rlaif|unknown. Deterministic + total
    (never raises) — a file we can't classify yields shape="unknown" so the agent
    asks the user instead of guessing. The per-row classifiers are the SAME
    normalizers the importers use, so detection matches what will actually import."""
    objs = _sample_objects(text, sample)
    n = len(objs)
    if n == 0:
        return {"shape": "unknown", "label": SHAPE_PLAIN["unknown"], "confidence": 0.0,
                "rows": 0, "matchRates": {}, "ambiguous": False}

    # Per-shape acceptance test, reusing the canonical row normalizers.
    def _ok_sft(o: dict) -> bool:
        msgs = o.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return False
        return _validate_messages(msgs) is None  # ends in assistant, all turns valid

    classifiers = {
        "preference": lambda o: _normalize_preference_row(o)[1] is None,
        "kto": lambda o: _normalize_kto_row(o)[1] is None,
        "rlvr": lambda o: _normalize_rlvr_row(o)[1] is None,
        "rlaif": lambda o: _normalize_rlaif_row(o)[1] is None,
        "sft": _ok_sft,
    }
    match_rates: dict[str, float] = {}
    for shape, fn in classifiers.items():
        hits = 0
        for o in objs:
            try:
                if fn(o):
                    hits += 1
            except Exception:  # noqa: BLE001 — a weird row never breaks detection
                pass
        match_rates[shape] = round(hits / n, 4)

    # Pick the most-specific shape that the bulk of rows satisfy. A shape "wins" if
    # ≥80% of sampled rows match it; ties resolve by _DETECT_ORDER (specific first).
    winner = "unknown"
    for shape in _DETECT_ORDER:
        if match_rates.get(shape, 0.0) >= 0.8:
            winner = shape
            break

    # Ambiguity flag: several shapes' signatures overlap (RLAIF prompt-only matches
    # almost anything; a preference/KTO row also carries `messages`). If a shape MORE
    # specific than the winner ALSO matched strongly, the agent should confirm rather
    # than silently pick. The winner stays the most-specific qualifying shape.
    strong = [s for s in _DETECT_ORDER if match_rates.get(s, 0.0) >= 0.8]
    ambiguous = winner != "unknown" and len(strong) > 1

    return {
        "shape": winner,
        "label": SHAPE_PLAIN.get(winner, SHAPE_PLAIN["unknown"]),
        "confidence": match_rates.get(winner, 0.0) if winner != "unknown" else 0.0,
        "rows": n,
        "matchRates": match_rates,
        "ambiguous": ambiguous,
    }


def _normalize_rlaif_row(obj: dict) -> tuple[dict | None, str | None]:
    """(canonical_row, error) for an RLAIF row.

    An RLAIF row is PROMPT-ONLY — the model generates a response and an AI judge
    (the reward prompt) scores it subjectively, so there is NO ground_truth:
        {messages:[...prompt turns]}
    Also accepts a bare `prompt` string. Reuses _validate_prompt_messages, which
    already forbids a trailing assistant turn — exactly the prompt-only constraint.
    Any `ground_truth`/`answer` present is ignored (RLAIF doesn't use it)."""
    if "messages" in obj:
        msgs = obj["messages"]
        err = _validate_prompt_messages(msgs)  # non-empty, valid turns, no trailing assistant
        if err:
            return None, err
        prompt = list(msgs)
    elif isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
        prompt = [{"role": "user", "content": obj["prompt"]}]
    else:
        return None, "row needs a `messages` array or a non-empty `prompt` string"
    return {"messages": prompt}, None


def parse_rlaif_jsonl(text: str, max_errors: int = 50) -> tuple[list[dict], ValidationReport]:
    """Parse + validate RLAIF JSONL (prompt-only). Returns (canonical_rows, report).
    Never raises — malformed lines are recorded as errors and excluded."""
    rows: list[dict] = []
    errors: list[RowError] = []
    preview: list[dict[str, Any]] = []
    total = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue
        total += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, f"invalid JSON: {e.msg}"))
            continue
        if not isinstance(obj, dict):
            if len(errors) < max_errors:
                errors.append(RowError(lineno, "line is not a JSON object"))
            continue
        row, err = _normalize_rlaif_row(obj)
        if err is not None:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, err))
            continue
        rows.append(row)
        if len(preview) < PREVIEW_LIMIT:
            preview.append(row)
    invalid = total - len(rows)
    report = ValidationReport(
        valid=(total > 0 and invalid == 0),
        total_lines=total,
        valid_rows=len(rows),
        invalid_rows=invalid,
        errors=errors,
        preview=preview,
        role_counts={},
    )
    return rows, report


def parse_rlvr_jsonl(text: str, max_errors: int = 50) -> tuple[list[dict], ValidationReport]:
    """Parse + validate RLVR JSONL. Returns (canonical_rows, report). Never raises —
    malformed lines are recorded as errors and excluded from the rows."""
    rows: list[dict] = []
    errors: list[RowError] = []
    preview: list[dict[str, Any]] = []
    total = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() == "":
            continue
        total += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, f"invalid JSON: {e.msg}"))
            continue
        if not isinstance(obj, dict):
            if len(errors) < max_errors:
                errors.append(RowError(lineno, "line is not a JSON object"))
            continue
        row, err = _normalize_rlvr_row(obj)
        if err is not None:
            if len(errors) < max_errors:
                errors.append(RowError(lineno, err))
            continue
        rows.append(row)
        if len(preview) < PREVIEW_LIMIT:
            preview.append(row)
    invalid = total - len(rows)
    report = ValidationReport(
        valid=(total > 0 and invalid == 0),
        total_lines=total,
        valid_rows=len(rows),
        invalid_rows=invalid,
        errors=errors,
        preview=preview,
        role_counts={},
    )
    return rows, report
