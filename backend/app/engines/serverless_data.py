# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Dataset reshaping for the SageMaker serverless engine.

Our platform persists splits in LLaMA-Factory shapes (messages JSONL for SFT;
ranking {messages, chosen, rejected} for DPO). The serverless recipe converters
want DIFFERENT shapes (determined empirically against the live converters):

  SFT  -> hf_prompt_completion, with prompt+completion as PLAIN STRINGS:
          {"prompt": "<system+user rendered to text>", "completion": "<final assistant>"}
          (bare `messages` failed KeyError:'prompt'; prompt-as-array failed
           'list has no startswith' — only strings work for the SFT converter.)
  DPO  -> {"prompt": "<system+user text>", "chosen": "<chosen text>",
           "rejected": "<rejected text>"} (hf_preference / dpo string form).
  RLVR -> VERL format (differs again!): {"id": "<n>", "prompt": [<messages except
          final assistant>], "reward_model": {"ground_truth": "<final assistant>"}}.
          Here `prompt` IS a message ARRAY (not a string), and the verifiable
          target lives in reward_model.ground_truth (the gold answer the preset
          reward fn checks against).

These converters are pure functions over the on-disk JSONL so they're trivially
unit-testable without AWS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _render_prompt(messages: list[dict]) -> str:
    """Render all non-final-assistant turns to a single prompt string.

    System + user content joined by blank lines, in order. The recipe tokenizes
    with the model's own chat template internally; we just provide the text.
    """
    parts = [str(m.get("content", "")) for m in messages]
    return "\n\n".join(p for p in parts if p)


class DataConversionError(ValueError):
    """A row can't be converted to the serverless recipe shape (bad/missing data).
    Raised with a clear message so a launch fails fast with an actionable error
    instead of crashing cryptically or silently corrupting the training set."""


def _final_assistant_idx(messages: list[dict]) -> int:
    idxs = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not idxs:
        raise DataConversionError(
            "row has no assistant turn — SFT needs a final assistant response as "
            "the completion. Check the dataset shape (this looks non-messages or "
            "prompt-only)."
        )
    return max(idxs)


def _turn_text(turn: Any) -> str:
    """A chosen/rejected entry is either an assistant turn dict or a bare string.
    None/missing → "" (NOT the literal "None") so the caller's empty-check catches
    it instead of training on a garbage "None" completion."""
    if turn is None:
        return ""
    if isinstance(turn, dict):
        return str(turn.get("content", "") or "")
    return str(turn)


def messages_to_prompt_completion(messages: list[dict]) -> dict[str, str]:
    """One SFT row: prompt = everything before the final assistant turn (as a
    string), completion = the final assistant content (as a string)."""
    if not isinstance(messages, list) or not messages:
        raise DataConversionError("row has empty/invalid `messages`")
    la = _final_assistant_idx(messages)
    completion = str(messages[la].get("content", "")).strip()
    if not completion:
        raise DataConversionError("row's final assistant turn has empty content")
    return {"prompt": _render_prompt(messages[:la]), "completion": completion}


def ranking_to_dpo(row: dict) -> dict[str, str]:
    """One DPO row: prompt (the shared `messages` rendered to text) + chosen +
    rejected completion strings. Missing/empty chosen|rejected is an ERROR (not a
    silent "None"/"" string that would train the model on garbage)."""
    chosen = _turn_text(row.get("chosen")).strip()
    rejected = _turn_text(row.get("rejected")).strip()
    if not chosen or not rejected:
        raise DataConversionError(
            "DPO row needs non-empty `chosen` AND `rejected` completions "
            "(one was missing/empty)."
        )
    return {
        "prompt": _render_prompt(row.get("messages", [])),
        "chosen": chosen,
        "rejected": rejected,
    }


def rlvr_to_verl(row: dict, row_id: str) -> dict[str, Any]:
    """One RLVR row in VERL format from our rlvr shape.

    Our rlvr row is {messages:[...prompt turns], ground_truth:"..."} — the prompt
    turns (NO trailing assistant answer) plus an EXPLICIT verifiable target. VERL
    wants {id, prompt:[...messages...], reward_model:{ground_truth}} with prompt as
    an ARRAY (unlike SFT's string). The ground_truth is the answer the reward fn
    (a gsm8k/prime_math preset or a custom reward) checks against — kept separate
    from any worked solution so it's unambiguous what "correct" means."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise DataConversionError("RLVR row has empty/invalid `messages`")
    ground_truth = str(row.get("ground_truth", "")).strip()
    if not ground_truth:
        raise DataConversionError(
            "RLVR row has empty `ground_truth` — the verifiable target the reward "
            "function checks against is required (it's NOT derived from the prompt)."
        )
    # Keep only role/content so the recipe's converter gets a clean OpenAI shape.
    prompt = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))}
              for m in messages]
    if not prompt:
        raise DataConversionError("RLVR row has no prompt turns")
    return {"id": row_id, "prompt": prompt, "reward_model": {"ground_truth": ground_truth}}


def rlaif_to_verl(row: dict, row_id: str) -> dict[str, Any]:
    """One RLAIF row in VERL format from our rlaif (prompt-only) shape.

    Our rlaif row is {messages:[...prompt turns]} — PROMPT-ONLY, no verifiable
    target (the AI judge scores the response subjectively via the reward prompt).
    BUT the RLAIF recipe runs on the same VERL/GRPO stack as RLVR and its
    server-side converter REQUIRES a `reward_model` field on every row — proven by
    a real run that failed with "Dataset format is incompatible with RLVR training.
    Missing required field: 'reward_model'". The judge supplies the actual reward,
    so we emit reward_model with an EMPTY ground_truth placeholder: present to pass
    the format check, ignored by the judge. `prompt` is a message ARRAY (like RLVR)."""
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise DataConversionError("RLAIF row has empty/invalid `messages`")
    prompt = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))}
              for m in messages]
    if not prompt:
        raise DataConversionError("RLAIF row has no prompt turns")
    return {"id": row_id, "prompt": prompt, "reward_model": {"ground_truth": ""}}


def convert_file(src: Path, dst: Path, stage: str) -> int:
    """Convert an on-disk split file (messages / ranking / rlvr / rlaif JSONL) to
    the serverless shape for `stage` (sft|dpo|rlvr|rlaif). Returns rows written."""
    n = 0
    with dst.open("w", encoding="utf-8") as w:
        for lineno, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if stage == "dpo":
                    out = ranking_to_dpo(row)
                elif stage == "rlvr":
                    out = rlvr_to_verl(row, str(n))
                elif stage == "rlaif":
                    out = rlaif_to_verl(row, str(n))
                else:  # sft (and any messages-shaped fallback)
                    if "messages" not in row:
                        raise DataConversionError("row has no `messages` key")
                    out = messages_to_prompt_completion(row["messages"])
            except (DataConversionError, json.JSONDecodeError, KeyError, TypeError) as e:
                # Fail fast with the offending row so the user can fix the dataset,
                # rather than uploading a partially-converted (corrupt) train set.
                raise DataConversionError(f"{src.name} line {lineno}: {e}") from e
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise DataConversionError(f"{src.name} produced 0 usable rows")
    return n
