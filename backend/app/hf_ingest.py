# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Ingest a sample of a public Hugging Face dataset as chat-template rows.

Why this exists: the platform needs REAL task data to prove the leaderboard
discriminates between models (synthetic data gives near-identical metrics). A
public HF dataset with genuine task structure does that — without needing a
customer dataset. We only ever take a SAMPLE (a fraction / row-cap), since the
goal is to validate the pipeline, not chase SOTA.

Design:
  - Fetch via the HF **datasets-server HTTP API** (plain JSON: /splits, /rows),
    NOT the heavy `datasets` library — keeps the Lambda lean and is ideal for
    sampling a few hundred rows.
  - Convert the dataset's native columns into the platform's CANONICAL format:
    a `messages` array of {role, content}. "messages" is the model-agnostic
    middle layer; the per-model chat TEMPLATE is applied later by LLaMA-Factory.
    So one converted dataset trains every model unchanged.
  - The converted rows are emitted as JSONL text and fed through the EXISTING
    auto_split → _finalize_split path. Nothing downstream changes.

The conversion is a small, deterministic mapping layer. We auto-detect a few
common shapes and also accept an explicit column mapping from the caller.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import requests

# HF datasets-server base. Public, no auth needed for public datasets; a token
# can be supplied for gated ones (resolved by the caller, passed as a header).
DATASETS_SERVER = "https://datasets-server.huggingface.co"

# HF Hub API base — serves dataset METADATA (license, gated flag) which the
# datasets-server (rows/splits) does NOT expose. Used to surface a license
# advisory at import time.
HUB_API = "https://huggingface.co/api"

# Hard cap on how many rows we will ever pull in one import — this is a
# *sampling* tool, not a bulk loader. Keeps fetch time + training cost bounded.
MAX_SAMPLE_ROWS = 2000

# The datasets-server /rows endpoint pages at 100 rows max per call.
_PAGE = 100

# Network timeout per HTTP call (seconds).
_TIMEOUT = 30


class HFIngestError(Exception):
    """Raised on any unrecoverable problem fetching/converting an HF dataset."""


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _get(url: str, token: str | None, _retries: int = 3) -> dict:
    last_status = 0
    for attempt in range(_retries):
        try:
            r = requests.get(url, headers=_headers(token), timeout=_TIMEOUT)
        except requests.RequestException as e:  # network-level failure
            raise HFIngestError(f"request to HF failed: {e}") from e
        if r.status_code == 404:
            raise HFIngestError("dataset/config/split not found (check the id)")
        if r.status_code in (401, 403):
            raise HFIngestError("dataset is gated/private — a HF token is required")
        if r.status_code == 429:
            # Rate limited — back off and retry (no time/RNG: fixed growing sleep).
            last_status = 429
            import time

            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise HFIngestError(f"HF datasets-server returned HTTP {r.status_code}")
        try:
            return r.json()
        except ValueError as e:
            raise HFIngestError("HF returned a non-JSON response") from e
    raise HFIngestError(
        f"HF datasets-server rate-limited the request (HTTP {last_status}); try fewer rows"
    )


# --- Discovery -------------------------------------------------------------


def list_splits(dataset: str, token: str | None = None) -> list[dict[str, str]]:
    """Return [{config, split}, ...] available for a dataset."""
    d = _get(f"{DATASETS_SERVER}/splits?dataset={dataset}", token)
    return [
        {"config": s["config"], "split": s["split"]}
        for s in d.get("splits", [])
    ]


# --- License advisory ------------------------------------------------------
# Surfacing the dataset's license at import time is a COMPLIANCE AID, not a
# gate: HF license tags are self-reported and often "unknown", so we INFORM the
# user (and prompt them to verify) rather than auto-allow/deny. The bucket drives
# the UI banner severity; the raw slug is always shown.

# Clearly permissive / public-domain licenses → an "ok, attribute" info banner.
# Deliberately conservative: a slug only belongs here if it imposes no copyleft,
# no field-of-use restriction, and no non-commercial clause. Anything short of
# that goes in _RESTRICTIVE_LICENSES so the user gets the amber "read it" banner.
_PERMISSIVE_LICENSES = {
    "mit", "mit-0", "apache-2.0", "apache", "bsd", "bsd-2-clause", "bsd-3-clause",
    "isc", "zlib", "cc0-1.0", "cc-by-4.0", "cc-by-3.0", "cc-by-2.0",
    "unlicense", "wtfpl", "pddl", "odc-by", "artistic-2.0",
}
# Licenses that need a careful read before training/redistributing → amber warning.
# Non-commercial, copyleft, and known use-restricted / non-OSI families.
#
# The RAIL/OpenRAIL family (openrail, bigscience-openrail-m, creativeml-openrail-m)
# is here — not in the permissive set — because those licenses attach BEHAVIORAL-USE
# restrictions (an appendix of prohibited uses that must be passed downstream) and
# are not OSI-approved, so "permissive" would misinform someone deciding whether
# they may train on and redistribute the data. GFDL is likewise copyleft (share-alike
# plus invariant-section/document obligations), not permissive.
#
# ShareAlike (cc-by-sa-*) and EPL-2.0 are here for the same reason: both impose a
# reciprocal obligation on DERIVATIVE works — precisely the question someone picking
# a fine-tuning dataset is asking, since a fine-tuned model is plausibly a derivative
# of its training data. CC-BY-SA requires derivatives be licensed alike; EPL-2.0 is
# weak file-level copyleft. Neither is "attribution and you're done", so neither gets
# the green banner.
_RESTRICTIVE_LICENSES = {
    "cc-by-nc-4.0", "cc-by-nc-3.0", "cc-by-nc-2.0", "cc-by-nc-sa-4.0", "cc-by-nc-sa-3.0",
    "cc-by-nc-nd-4.0", "cc-by-nd-4.0", "gpl", "gpl-2.0", "gpl-3.0", "agpl-3.0",
    "lgpl-2.1", "lgpl-3.0", "osl-3.0", "ms-pl", "llama2", "llama3", "llama3.1",
    "llama3.2", "gemma", "other", "bsl-1.0", "cc", "afl-3.0",
    "openrail", "bigscience-openrail-m", "creativeml-openrail-m",
    "gfdl", "gfdl-1.3", "fdl-1.3",
    "cc-by-sa-4.0", "cc-by-sa-3.0", "epl-2.0",
}


def classify_license(license_slug: str | None) -> str:
    """Bucket a HF license slug → "permissive" | "restrictive" | "unknown".
    Drives the import banner's severity. Anything unrecognized / absent / literally
    "unknown" → "unknown" (an amber "verify it yourself" prompt, never a block)."""
    if not license_slug:
        return "unknown"
    slug = str(license_slug).strip().lower()
    if slug in ("", "unknown", "unlicensed"):
        return "unknown"
    if slug in _PERMISSIVE_LICENSES:
        return "permissive"
    if slug in _RESTRICTIVE_LICENSES:
        return "restrictive"
    # Heuristics for slugs not in the explicit sets (HF tags drift): an "nc"
    # (non-commercial) or "sa" (ShareAlike copyleft) token, "noncommercial" anywhere,
    # or a gpl/agpl family, is restrictive. Both tokens are matched against the
    # "-"-delimited parts, never as substrings — a bare `"sa" in slug` would fire on
    # any name that happens to contain those two letters in sequence.
    tokens = slug.split("-")
    if ("nc" in tokens or "sa" in tokens or "noncommercial" in slug
            or slug.startswith(("gpl", "agpl", "lgpl"))):
        return "restrictive"
    # Unlisted RAIL/OpenRAIL and GFDL variants (openrail++, bigscience-bloom-rail-1.0,
    # gfdl-1.3-or-later, …). These would otherwise fall through to "unknown", which
    # also warns — but naming the reason is more useful than "not confirmed".
    if "rail" in slug or slug.startswith(("gfdl", "fdl-")):
        return "restrictive"
    # A cc-by* slug is permissive only when it carries neither an "nc" nor an "sa"
    # token; both are caught by the token test above, so anything reaching here is
    # plain attribution (cc-by-1.0, …). Otherwise we don't know.
    if slug.startswith("cc-by") and "nc" not in slug:
        return "permissive"
    return "unknown"


def fetch_license(dataset: str, token: str | None = None) -> dict[str, Any]:
    """Best-effort lookup of a dataset's license + gated flag via the HF Hub API
    (a DIFFERENT endpoint than the datasets-server). Returns
    {license, bucket, gated} — license None when undeclared. NEVER raises: a
    license lookup must not break the preview/import (it's advisory), so any
    network/parse error degrades to bucket="unknown"."""
    try:
        r = requests.get(f"{HUB_API}/datasets/{dataset}", headers=_headers(token), timeout=_TIMEOUT)
        if r.status_code != 200:
            return {"license": None, "bucket": "unknown", "gated": None}
        d = r.json()
        card = d.get("cardData") or {}
        lic = card.get("license")
        # cardData.license may be a string OR a list (HF allows multiple) — take the
        # first; if absent, fall back to a `license:<x>` tag.
        if isinstance(lic, list):
            lic = lic[0] if lic else None
        if not lic:
            for t in d.get("tags", []) or []:
                if isinstance(t, str) and t.startswith("license:"):
                    lic = t.split(":", 1)[1]
                    break
        gated = d.get("gated")  # False | "auto" | "manual"
        return {"license": lic, "bucket": classify_license(lic), "gated": gated}
    except Exception:  # noqa: BLE001 — advisory only; never fail the preview
        return {"license": None, "bucket": "unknown", "gated": None}


def _fetch_rows(
    dataset: str, config: str, split: str, offset: int, length: int, token: str | None
) -> dict:
    url = (
        f"{DATASETS_SERVER}/rows?dataset={dataset}&config={config}"
        f"&split={split}&offset={offset}&length={length}"
    )
    return _get(url, token)


def _class_label_names(features: list[dict]) -> dict[str, list[str]]:
    """Map column-name -> class names, for ClassLabel features (so an int label
    like 2 becomes the human-readable class 'Business' the model can learn)."""
    out: dict[str, list[str]] = {}
    for f in features or []:
        t = f.get("type") or {}
        if t.get("_type") == "ClassLabel" and isinstance(t.get("names"), list):
            out[f["name"]] = t["names"]
    return out


# --- Conversion ------------------------------------------------------------

# Known column-name groups for auto-detection, in priority order. Each entry is
# (user_field_candidates, assistant_field_candidates, optional system_field).
_AUTODETECT = [
    (["instruction"], ["output", "response", "completion"], "input"),
    (["prompt"], ["completion", "response", "output", "chosen"], None),
    (["question"], ["answer", "answers"], "context"),
    # Summarization: article/document → highlights/summary (e.g. cnn_dailymail
    # uses 'article'/'highlights', xsum 'document'/'summary').
    (["article", "document", "body", "content", "story"], ["highlights", "summary", "abstract", "tldr"], None),
    (["text", "sentence", "document", "dialogue"], ["label", "summary", "target"], None),
    (["input"], ["output", "target"], None),
]


@dataclass
class ColumnMapping:
    """How to build a chat row from a dataset's columns.

    user_field   : column whose value becomes the user turn (required).
    target_field : column whose value becomes the assistant turn (required).
    system_field : optional column whose value becomes a per-row system turn.
    context_field: optional column appended to the user turn (e.g. QA context).
    instruction  : optional FIXED instruction applied to every row. Because it
                   is identical for all rows (e.g. "Classify the topic:"), it
                   maps to the SYSTEM turn — that's what a system prompt is for,
                   and our models' chat templates attend to it. This keeps the
                   user turn = the actual input data. If a per-row system_field
                   is also given, the fixed instruction is prepended to it.
    """

    user_field: str
    target_field: str
    system_field: str | None = None
    context_field: str | None = None
    instruction: str = ""


def autodetect_mapping(column_names: list[str]) -> ColumnMapping | None:
    """Best-effort guess of a ColumnMapping from available columns."""
    cols = {c.lower(): c for c in column_names}
    for users, targets, sys_field in _AUTODETECT:
        u = next((cols[c] for c in users if c in cols), None)
        t = next((cols[c] for c in targets if c in cols), None)
        if u and t:
            sf = cols.get(sys_field) if sys_field else None
            return ColumnMapping(user_field=u, target_field=t, context_field=sf)
    return None


# --- Preference (DPO) column mapping + detection --------------------------- #
# Preference datasets pair a prompt with a CHOSEN and a REJECTED response. The
# common public layouts we detect:
#   prompt + chosen + rejected         (UltraFeedback-binarized, many RLHF sets)
#   chosen + rejected (no prompt col)  (Anthropic/hh-rlhf — the prompt is embedded
#                                       as the shared conversational prefix of each)
# The chosen/rejected cells are usually a bare answer STRING, but some sets store
# a full chat list ([{role,content},...]); we handle both (see _last_assistant).

# Candidate column names, in priority order.
_PREF_PROMPT_COLS = ["prompt", "question", "instruction", "query", "input"]
_PREF_CHOSEN_COLS = ["chosen", "chosen_response", "response_chosen", "chosen_text", "j"]
_PREF_REJECTED_COLS = ["rejected", "rejected_response", "response_rejected", "rejected_text", "k"]


@dataclass
class PreferenceMapping:
    """How to build a preference (ranking) row from a dataset's columns.

    chosen_field / rejected_field : columns holding the two competing responses
        (required). Values may be a string OR a chat list.
    prompt_field  : optional column holding the prompt (string or chat list). When
        absent (e.g. hh-rlhf), the shared prefix of chosen vs rejected is the prompt.
    system_field  : optional per-row system column.
    instruction   : optional fixed instruction applied to every row (→ system turn).
    """

    chosen_field: str
    rejected_field: str
    prompt_field: str | None = None
    system_field: str | None = None
    instruction: str = ""


def autodetect_preference_mapping(column_names: list[str]) -> PreferenceMapping | None:
    """Best-effort guess of a PreferenceMapping. Requires chosen + rejected; the
    prompt column is optional (some sets embed it in the responses)."""
    cols = {c.lower(): c for c in column_names}
    chosen = next((cols[c] for c in _PREF_CHOSEN_COLS if c in cols), None)
    rejected = next((cols[c] for c in _PREF_REJECTED_COLS if c in cols), None)
    if not chosen or not rejected:
        return None
    prompt = next((cols[c] for c in _PREF_PROMPT_COLS if c in cols), None)
    system = cols.get("system")
    return PreferenceMapping(chosen_field=chosen, rejected_field=rejected,
                             prompt_field=prompt, system_field=system)


def _stringify(value: Any, label_names: list[str] | None) -> str:
    """Render a cell value as a string. Integers that index a ClassLabel are
    converted to the class name; lists/dicts are JSON-encoded; None -> ''."""
    if value is None:
        return ""
    if label_names is not None and isinstance(value, int) and 0 <= value < len(label_names):
        return label_names[value]
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def row_to_messages(
    row: dict,
    mapping: ColumnMapping,
    class_labels: dict[str, list[str]] | None = None,
) -> dict | None:
    """Convert one dataset row to a {messages:[...]} chat row, or None if the
    required user/target values are empty (such rows are skipped, not errored)."""
    class_labels = class_labels or {}

    user_val = _stringify(row.get(mapping.user_field), class_labels.get(mapping.user_field))
    target_val = _stringify(row.get(mapping.target_field), class_labels.get(mapping.target_field))

    if mapping.context_field:
        ctx = _stringify(row.get(mapping.context_field), class_labels.get(mapping.context_field))
        if ctx:
            user_val = f"{user_val}\n\n{ctx}" if user_val else ctx

    if not user_val.strip() or not target_val.strip():
        return None

    # Build the system turn from the fixed instruction and/or a per-row system
    # column. The fixed instruction (identical for all rows) belongs in system.
    system_parts: list[str] = []
    if mapping.instruction.strip():
        system_parts.append(mapping.instruction.strip())
    if mapping.system_field:
        sys_val = _stringify(row.get(mapping.system_field), None)
        if sys_val.strip():
            system_parts.append(sys_val.strip())

    messages: list[dict] = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.append({"role": "user", "content": user_val})
    messages.append({"role": "assistant", "content": target_val})
    return {"messages": messages}


# --- Preference (ranking) row conversion ----------------------------------- #


def _as_chat_list(value: Any) -> list[dict] | None:
    """If `value` is a chat list ([{role,content},...]), return it normalized;
    else None. Tolerates HF's occasional {'from','value'} sharegpt-native keys."""
    if not isinstance(value, list) or not value:
        return None
    out: list[dict] = []
    for turn in value:
        if not isinstance(turn, dict):
            return None
        role = turn.get("role") or turn.get("from")
        content = turn.get("content") or turn.get("value")
        if role is None or not isinstance(content, str):
            return None
        # sharegpt roles → our roles
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        out.append({"role": role, "content": content})
    return out


def _final_assistant_content(chat: list[dict]) -> str | None:
    """The content of the last assistant turn in a chat list (the response)."""
    for turn in reversed(chat):
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


# Transcript layout (Anthropic/hh-rlhf and friends): chosen/rejected are ONE
# string holding the whole conversation, with turns delimited by "\n\nHuman:" and
# "\n\nAssistant:". We split it back into a chat list so the shared-prefix logic
# (Layout 2) can treat it like any other chat — the prompt is the common prefix,
# the diverging final assistant turn is the response.
# Match a role marker either after a "\n\n" delimiter OR at the very start of the
# string (some transcripts omit the leading blank line — without `^` the first
# turn would be silently dropped).
_TRANSCRIPT_RE = re.compile(r"(?:^|\n\n)(Human|Assistant):\s?", re.IGNORECASE)


def _parse_transcript(value: Any) -> list[dict] | None:
    """Parse a '\\n\\nHuman: …\\n\\nAssistant: …' transcript string into a chat
    list, or None if it doesn't look like one (no Human/Assistant markers)."""
    if not isinstance(value, str) or "Assistant:" not in value:
        return None
    # Split keeping the role markers (the regex captures the role group).
    parts = _TRANSCRIPT_RE.split(value)
    # parts = [pre, role1, text1, role2, text2, ...]; leading pre is usually "".
    if len(parts) < 3:
        return None
    out: list[dict] = []
    i = 1
    while i + 1 < len(parts) + 1 and i < len(parts):
        role_raw = parts[i].lower()
        text = parts[i + 1] if i + 1 < len(parts) else ""
        role = "user" if role_raw == "human" else "assistant"
        content = (text or "").strip()
        if content:
            out.append({"role": role, "content": content})
        i += 2
    return out or None


def _common_prefix_len(a: list[dict], b: list[dict]) -> int:
    """How many leading turns chosen & rejected chat lists share (the prompt)."""
    n = 0
    for ta, tb in zip(a, b):
        if ta == tb:
            n += 1
        else:
            break
    return n


def preference_row_from_hf(
    row: dict,
    mapping: PreferenceMapping,
    class_labels: dict[str, list[str]] | None = None,
) -> dict | None:
    """Convert one HF row to a canonical ranking row {messages, chosen, rejected},
    or None if it can't be formed (missing/empty fields). Handles four layouts:

      1. chosen/rejected are STRINGS + an explicit prompt column.
      2. chosen/rejected are CHAT LISTS sharing a prompt prefix (no prompt column,
         e.g. hh-rlhf in list form) — the shared leading turns become `messages`,
         the diverging final assistant turns become chosen/rejected.
      3. chosen/rejected are chat lists AND a prompt column is given — use the
         prompt column for `messages`, the final assistant turns for the responses.
      4. chosen/rejected are TRANSCRIPT STRINGS ("\\n\\nHuman: …\\n\\nAssistant: …",
         the Anthropic/hh-rlhf shipping format) — parsed into chat lists, then
         treated like Layout 2 (shared prefix = prompt).
    """
    class_labels = class_labels or {}
    raw_chosen = row.get(mapping.chosen_field)
    raw_rejected = row.get(mapping.rejected_field)

    # Layout 4: transcript strings → chat lists (so the shared-prefix path applies).
    chosen_chat = _as_chat_list(raw_chosen) or _parse_transcript(raw_chosen)
    rejected_chat = _as_chat_list(raw_rejected) or _parse_transcript(raw_rejected)

    prompt_msgs: list[dict] | None = None
    chosen_text: str | None = None
    rejected_text: str | None = None

    if chosen_chat and rejected_chat:
        if mapping.prompt_field:
            # An explicit prompt column supplies the prompt; the responses are the
            # final assistant turn of each chat.
            chosen_text = _final_assistant_content(chosen_chat)
            rejected_text = _final_assistant_content(rejected_chat)
            pv = row.get(mapping.prompt_field)
            pchat = _as_chat_list(pv)
            prompt_msgs = pchat if pchat else [
                {"role": "user", "content": _stringify(pv, None)}]
        else:
            # Layout 2 (hh-rlhf): the two chats share a prompt prefix and diverge
            # at the response. The prompt is the shared leading turns; the
            # chosen/rejected responses are the FIRST diverging turn — which must
            # be an assistant turn. Do NOT use the final assistant turn: for
            # multi-turn data the divergence can be earlier, and the final turn
            # would then sit AFTER the truncated prompt → a misaligned pair. (For
            # the common single-divergence case, k IS the final turn, so this also
            # covers it.) If the divergence isn't an assistant/assistant pair, the
            # row can't form a clean preference pair → skip it.
            k = _common_prefix_len(chosen_chat, rejected_chat)
            c_turn = chosen_chat[k] if k < len(chosen_chat) else None
            r_turn = rejected_chat[k] if k < len(rejected_chat) else None
            if (
                isinstance(c_turn, dict) and c_turn.get("role") == "assistant"
                and isinstance(r_turn, dict) and r_turn.get("role") == "assistant"
            ):
                prompt_msgs = chosen_chat[:k] or None
                chosen_text = c_turn.get("content")
                rejected_text = r_turn.get("content")
            # else: prompt_msgs stays None → row skipped below.
    else:
        # Layout 1: string responses; need an explicit prompt column.
        chosen_text = _stringify(raw_chosen, class_labels.get(mapping.chosen_field))
        rejected_text = _stringify(raw_rejected, class_labels.get(mapping.rejected_field))
        if mapping.prompt_field:
            pv = row.get(mapping.prompt_field)
            pchat = _as_chat_list(pv)
            prompt_msgs = pchat if pchat else [
                {"role": "user", "content": _stringify(pv, class_labels.get(mapping.prompt_field))}]

    if not prompt_msgs:
        return None
    # The prompt must have real content — a configured prompt_field that's empty
    # for some rows would otherwise leak an empty-user-turn pair into training
    # (the SFT + KTO paths already guard this; keep them consistent).
    if not any(isinstance(t.get("content"), str) and t["content"].strip() for t in prompt_msgs):
        return None
    if not chosen_text or not chosen_text.strip():
        return None
    if not rejected_text or not rejected_text.strip():
        return None
    # Identical responses carry no preference signal — skip them.
    if chosen_text.strip() == rejected_text.strip():
        return None

    # Prepend a fixed instruction / per-row system column as a system turn.
    system_parts: list[str] = []
    if mapping.instruction.strip():
        system_parts.append(mapping.instruction.strip())
    if mapping.system_field:
        sv = _stringify(row.get(mapping.system_field), None)
        if sv.strip():
            system_parts.append(sv.strip())
    messages = list(prompt_msgs)
    if system_parts and not any(t.get("role") == "system" for t in messages):
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + messages

    return {
        "messages": messages,
        "chosen": {"role": "assistant", "content": chosen_text},
        "rejected": {"role": "assistant", "content": rejected_text},
    }


# --- Sampling --------------------------------------------------------------


@dataclass
class HFPreview:
    dataset: str
    config: str
    split: str
    num_rows_total: int | None  # split size if HF reports it, else None
    column_names: list[str] = field(default_factory=list)
    suggested_mapping: ColumnMapping | None = None
    sample_rows: list[dict] = field(default_factory=list)  # raw HF rows
    converted_preview: list[dict] = field(default_factory=list)  # messages rows
    # column -> class names, for ClassLabel columns (so the client can render an
    # int label as its human-readable class name in the live preview).
    class_labels: dict[str, list[str]] = field(default_factory=dict)
    # Preference (DPO) detection: when chosen/rejected columns are present, a
    # suggested PreferenceMapping + a ranking-shaped converted preview. The UI uses
    # `detectedShape` to offer "import as preference (DPO)" vs the SFT mapping.
    suggested_preference: "PreferenceMapping | None" = None
    preference_preview: list[dict] = field(default_factory=list)
    # KTO detection: when a completion + binary-label column are present.
    suggested_kto: "KtoMapping | None" = None
    kto_preview: list[dict] = field(default_factory=list)
    # RLVR suggestion: a prompt + a verifiable ground-truth column (e.g. gsm8k's
    # question + answer). Does NOT change detected_shape (a Q&A set is usually SFT;
    # RLVR is a deliberate pick) — it only pre-fills the columns when the user
    # chooses the RLVR objective.
    suggested_rlvr: "RlvrMapping | None" = None
    rlvr_preview: list[dict] = field(default_factory=list)
    # RLAIF suggestion: a prompt column is enough (prompt-only; the AI judge scores
    # the response). Like RLVR it does NOT change detected_shape (RLAIF is a
    # deliberate pick) — it only pre-fills the prompt column for the RLAIF objective.
    suggested_rlaif: "RlaifMapping | None" = None
    rlaif_preview: list[dict] = field(default_factory=list)
    # License advisory (compliance aid): {license, bucket, gated} from the HF Hub
    # API. Surfaced as a banner at import; never blocks. None-ish when undeclared.
    license_info: dict[str, Any] = field(default_factory=lambda: {"license": None, "bucket": "unknown", "gated": None})

    @property
    def detected_shape(self) -> str:
        # Preference (DPO) wins over KTO when both look possible (a chosen/rejected
        # pair is a stronger, less ambiguous signal than a single labelled column).
        if self.suggested_preference is not None:
            return "preference"
        if self.suggested_kto is not None:
            return "kto"
        return "sft"

    def to_dict(self) -> dict[str, Any]:
        m = self.suggested_mapping
        pm = self.suggested_preference
        return {
            "dataset": self.dataset,
            "config": self.config,
            "split": self.split,
            "numRowsTotal": self.num_rows_total,
            "columnNames": self.column_names,
            "classLabels": self.class_labels,
            "suggestedMapping": (
                {
                    "userField": m.user_field,
                    "targetField": m.target_field,
                    "systemField": m.system_field,
                    "contextField": m.context_field,
                    "instruction": m.instruction,
                }
                if m
                else None
            ),
            "detectedShape": self.detected_shape,
            "suggestedPreference": (
                {
                    "chosenField": pm.chosen_field,
                    "rejectedField": pm.rejected_field,
                    "promptField": pm.prompt_field,
                    "systemField": pm.system_field,
                    "instruction": pm.instruction,
                }
                if pm
                else None
            ),
            "preferencePreview": self.preference_preview,
            "suggestedKto": (
                {
                    "completionField": self.suggested_kto.completion_field,
                    "labelField": self.suggested_kto.label_field,
                    "promptField": self.suggested_kto.prompt_field,
                    "systemField": self.suggested_kto.system_field,
                    "instruction": self.suggested_kto.instruction,
                }
                if self.suggested_kto
                else None
            ),
            "ktoPreview": self.kto_preview,
            "suggestedRlvr": (
                {
                    "promptField": self.suggested_rlvr.prompt_field,
                    "groundTruthField": self.suggested_rlvr.ground_truth_field,
                    "systemField": self.suggested_rlvr.system_field,
                    "instruction": self.suggested_rlvr.instruction,
                }
                if self.suggested_rlvr
                else None
            ),
            "rlvrPreview": self.rlvr_preview,
            "suggestedRlaif": (
                {
                    "promptField": self.suggested_rlaif.prompt_field,
                    "systemField": self.suggested_rlaif.system_field,
                    "instruction": self.suggested_rlaif.instruction,
                }
                if self.suggested_rlaif
                else None
            ),
            "rlaifPreview": self.rlaif_preview,
            "sampleRows": self.sample_rows,
            "convertedPreview": self.converted_preview,
            "licenseInfo": self.license_info,
        }


def preview(
    dataset: str,
    config: str | None = None,
    split: str | None = None,
    token: str | None = None,
) -> HFPreview:
    """Inspect a dataset: resolve config/split, list columns, suggest a mapping,
    and show a few raw + converted rows. Powers the UI's mapping step."""
    splits = list_splits(dataset, token)
    if not splits:
        raise HFIngestError("dataset exposes no splits")
    # Resolve config/split: honour caller's choice, else first available.
    chosen = None
    for s in splits:
        if (config is None or s["config"] == config) and (split is None or s["split"] == split):
            chosen = s
            break
    if chosen is None:
        chosen = splits[0]
    cfg, sp = chosen["config"], chosen["split"]

    data = _fetch_rows(dataset, cfg, sp, 0, 5, token)
    features = data.get("features", [])
    column_names = [f["name"] for f in features]
    class_labels = _class_label_names(features)
    raw_rows = [r.get("row", {}) for r in data.get("rows", [])]
    num_total = data.get("num_rows_total")

    mapping = autodetect_mapping(column_names)
    converted: list[dict] = []
    if mapping:
        for r in raw_rows:
            row = row_to_messages(r, mapping, class_labels)
            if row:
                converted.append(row)

    # Preference (DPO) detection — independent of the SFT mapping. When the
    # columns look like chosen/rejected pairs, also build a ranking-shaped preview
    # so the UI can offer importing it as a preference dataset.
    pref_mapping = autodetect_preference_mapping(column_names)
    pref_preview: list[dict] = []
    if pref_mapping:
        for r in raw_rows:
            prow = preference_row_from_hf(r, pref_mapping, class_labels)
            if prow:
                pref_preview.append(prow)
        # If nothing converted, the columns matched by name but not by content —
        # don't claim a preference shape we can't actually build.
        if not pref_preview:
            pref_mapping = None

    # KTO detection — only when it's NOT already a preference set (a chosen/rejected
    # pair is the stronger signal). Build a labelled preview so the UI can offer it.
    kto_mapping = None if pref_mapping else autodetect_kto_mapping(column_names)
    kto_preview: list[dict] = []
    if kto_mapping:
        for r in raw_rows:
            krow = kto_row_from_hf(r, kto_mapping, class_labels)
            if krow:
                kto_preview.append(krow)
        if not kto_preview:
            kto_mapping = None

    # RLVR suggestion — a prompt + a verifiable answer column (e.g. gsm8k). Built
    # whenever such columns exist; it does NOT override detected_shape (the user
    # opts into RLVR), it just pre-fills the mapping + shows a converted preview.
    rlvr_mapping = autodetect_rlvr_mapping(column_names)
    rlvr_preview: list[dict] = []
    if rlvr_mapping:
        for r in raw_rows:
            rrow = rlvr_row_from_hf(r, rlvr_mapping, class_labels)
            if rrow:
                rlvr_preview.append(rrow)
        if not rlvr_preview:
            rlvr_mapping = None

    # RLAIF (prompt-only) detection — only needs a prompt column. Pre-fills the
    # RLAIF objective; never changes detected_shape (RLAIF is a deliberate pick).
    rlaif_mapping = autodetect_rlaif_mapping(column_names)
    rlaif_preview: list[dict] = []
    if rlaif_mapping:
        for r in raw_rows:
            arow = rlaif_row_from_hf(r, rlaif_mapping, class_labels)
            if arow:
                rlaif_preview.append(arow)
        if not rlaif_preview:
            rlaif_mapping = None

    return HFPreview(
        dataset=dataset,
        config=cfg,
        split=sp,
        num_rows_total=num_total,
        column_names=column_names,
        suggested_mapping=mapping,
        sample_rows=raw_rows,
        converted_preview=converted,
        class_labels=class_labels,
        suggested_preference=pref_mapping,
        preference_preview=pref_preview,
        suggested_kto=kto_mapping,
        kto_preview=kto_preview,
        suggested_rlvr=rlvr_mapping,
        rlvr_preview=rlvr_preview,
        suggested_rlaif=rlaif_mapping,
        rlaif_preview=rlaif_preview,
        # Best-effort license advisory (never raises; degrades to "unknown").
        license_info=fetch_license(dataset, token),
    )


def _ordered_pages(num_total: int, n_want: int, seed: int) -> list[int]:
    """Deterministically pick which PAGE offsets to fetch to gather ~n_want rows.

    We sample at PAGE granularity (not per-row) so the number of HTTP calls is
    bounded by ceil(n_want / _PAGE), not by how scattered the rows are — fetching
    individual rows spread across a 120k-row dataset would page the whole thing
    and hit the rate limit. Pages are chosen by a seeded hash ordering (no RNG,
    reproducible). Returns a list of page-aligned offsets [0, _PAGE, 2*_PAGE, …].
    """
    import hashlib

    n_pages_total = max(1, (num_total + _PAGE - 1) // _PAGE)
    n_pages_want = max(1, (n_want + _PAGE - 1) // _PAGE)
    if n_pages_want >= n_pages_total:
        return [p * _PAGE for p in range(n_pages_total)]

    def key(p: int) -> str:
        return hashlib.sha256(f"{seed}:page:{p}".encode()).hexdigest()

    chosen = sorted(range(n_pages_total), key=key)[:n_pages_want]
    return [p * _PAGE for p in sorted(chosen)]


def _sample_converted(
    dataset: str,
    config: str,
    split: str,
    max_rows: int,
    convert,
    seed: int = 42,
    token: str | None = None,
) -> tuple[list[dict], dict]:
    """Deterministically sample rows and apply `convert(raw_row, class_labels)`.

    Shared engine for SFT + preference sampling: identical paging/rate-limit
    handling, only the per-row converter differs. `convert` returns a converted
    dict or None (skipped). Returns (converted_rows, stats).
    """
    n_want = max(1, min(max_rows, MAX_SAMPLE_ROWS))

    # Probe to learn split size + class labels.
    head = _fetch_rows(dataset, config, split, 0, 1, token)
    features = head.get("features", [])
    class_labels = _class_label_names(features)
    num_total = head.get("num_rows_total")

    out: list[dict] = []
    skipped = 0

    def convert_batch(rows: list[dict]) -> None:
        nonlocal skipped
        for r in rows:
            if len(out) >= n_want:
                return
            row = convert(r, class_labels)
            if row is None:
                skipped += 1
                continue
            out.append(row)

    if num_total is None:
        # HF didn't report a total; page sequentially up to n_want.
        offset = 0
        while len(out) < n_want:
            batch = _fetch_rows(dataset, config, split, offset, _PAGE, token)
            rows = [r.get("row", {}) for r in batch.get("rows", [])]
            if not rows:
                break
            convert_batch(rows)
            offset += len(rows)
    else:
        # Fetch whole pages chosen deterministically — bounded # of HTTP calls.
        for page_offset in _ordered_pages(num_total, n_want, seed):
            if len(out) >= n_want:
                break
            batch = _fetch_rows(dataset, config, split, page_offset, _PAGE, token)
            rows = [r.get("row", {}) for r in batch.get("rows", [])]
            convert_batch(rows)

    stats = {
        "requested": n_want,
        "converted": len(out),
        "skipped": skipped,
        "numRowsTotal": num_total,
        "seed": seed,
    }
    return out, stats


def sample_to_jsonl(
    dataset: str,
    mapping: ColumnMapping,
    config: str,
    split: str,
    max_rows: int,
    seed: int = 42,
    token: str | None = None,
) -> tuple[str, dict]:
    """Fetch a deterministic sample and convert it to chat-template JSONL text.

    Returns (jsonl_text, stats). `max_rows` is capped at MAX_SAMPLE_ROWS. Rows
    that can't be converted (empty user/target) are skipped and counted.
    """
    rows, stats = _sample_converted(
        dataset, config, split, max_rows,
        lambda r, cl: row_to_messages(r, mapping, cl),
        seed=seed, token=token,
    )
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), stats


def sample_preference_to_jsonl(
    dataset: str,
    mapping: PreferenceMapping,
    config: str,
    split: str,
    max_rows: int,
    seed: int = 42,
    token: str | None = None,
) -> tuple[list[dict], dict]:
    """Fetch a deterministic sample and convert it to canonical RANKING rows
    ({messages, chosen, rejected}). Returns (rows, stats) — rows (not JSONL text)
    because the preference persist path takes a list, deriving the eval set itself.
    Rows that can't form a valid preference pair are skipped and counted."""
    return _sample_converted(
        dataset, config, split, max_rows,
        lambda r, cl: preference_row_from_hf(r, mapping, cl),
        seed=seed, token=token,
    )


# --- KTO (binary good/bad) column mapping + conversion --------------------- #
# A KTO dataset has a prompt + a completion + a binary label (good/bad). Common
# layouts: prompt/completion/label, or messages + a label column. The label may
# be bool, 0/1, -1/1, or "good"/"bad" strings (see validation._coerce_kto_tag).

_KTO_COMPLETION_COLS = ["completion", "response", "output", "answer", "target", "chosen"]
_KTO_LABEL_COLS = ["label", "kto_tag", "is_desirable", "desirable", "good", "rating", "score", "reward"]


@dataclass
class KtoMapping:
    """How to build a KTO row from a dataset's columns.

    completion_field : column holding the assistant response being judged (req).
    label_field      : column holding the binary good/bad signal (req).
    prompt_field     : optional column holding the prompt (string or chat list).
    system_field     : optional per-row system column.
    instruction      : optional fixed instruction applied to every row.
    """

    completion_field: str
    label_field: str
    prompt_field: str | None = None
    system_field: str | None = None
    instruction: str = ""


def autodetect_kto_mapping(column_names: list[str]) -> KtoMapping | None:
    """Best-effort guess of a KtoMapping. Requires a completion + a label column."""
    cols = {c.lower(): c for c in column_names}
    completion = next((cols[c] for c in _KTO_COMPLETION_COLS if c in cols), None)
    label = next((cols[c] for c in _KTO_LABEL_COLS if c in cols), None)
    if not completion or not label:
        return None
    prompt = next((cols[c] for c in _PREF_PROMPT_COLS if c in cols), None)
    return KtoMapping(completion_field=completion, label_field=label,
                      prompt_field=prompt, system_field=cols.get("system"))


def kto_row_from_hf(
    row: dict,
    mapping: KtoMapping,
    class_labels: dict[str, list[str]] | None = None,
) -> dict | None:
    """Convert one HF row to a canonical KTO row {messages:[...,assistant], kto_tag:bool},
    or None if it can't be formed (missing prompt/completion, or an unrecognized
    label). Reuses validation._coerce_kto_tag for the flexible label parsing."""
    from .validation import _coerce_kto_tag

    class_labels = class_labels or {}
    completion = _stringify(row.get(mapping.completion_field),
                            class_labels.get(mapping.completion_field))
    if not completion.strip():
        return None
    tag = _coerce_kto_tag(row.get(mapping.label_field))
    if tag is None:
        return None

    # Prompt → leading turns (chat list or a single user turn).
    prompt_msgs: list[dict]
    if mapping.prompt_field:
        pv = row.get(mapping.prompt_field)
        pchat = _as_chat_list(pv)
        prompt_msgs = pchat if pchat else [
            {"role": "user", "content": _stringify(pv, class_labels.get(mapping.prompt_field))}]
    else:
        prompt_msgs = []
    if not any(t.get("content", "").strip() for t in prompt_msgs):
        return None  # need a non-empty prompt

    # Optional fixed instruction / system column → leading system turn.
    system_parts: list[str] = []
    if mapping.instruction.strip():
        system_parts.append(mapping.instruction.strip())
    if mapping.system_field:
        sv = _stringify(row.get(mapping.system_field), None)
        if sv.strip():
            system_parts.append(sv.strip())
    messages = list(prompt_msgs)
    if system_parts and not any(t.get("role") == "system" for t in messages):
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + messages
    messages.append({"role": "assistant", "content": completion})
    return {"messages": messages, "kto_tag": tag}


def sample_kto_to_jsonl(
    dataset: str,
    mapping: KtoMapping,
    config: str,
    split: str,
    max_rows: int,
    seed: int = 42,
    token: str | None = None,
) -> tuple[list[dict], dict]:
    """Fetch a deterministic sample and convert it to canonical KTO rows
    ({messages, kto_tag}). Returns (rows, stats); unconvertible rows are skipped."""
    return _sample_converted(
        dataset, config, split, max_rows,
        lambda r, cl: kto_row_from_hf(r, mapping, cl),
        seed=seed, token=token,
    )


# --- RLVR import (prompt + verifiable ground_truth) ------------------------- #
# RLVR needs a PROMPT and a VERIFIABLE target the reward fn checks (NOT a worked
# solution to imitate). The classic fit is gsm8k (question + answer with a `####
# <final>` convention) — `answer`/`ground_truth`/`solution`/`target` all map to
# the ground_truth field.
_RLVR_GROUND_TRUTH_COLS = ["ground_truth", "answer", "solution", "target", "final_answer", "label"]


@dataclass
class RlvrMapping:
    """How to build an RLVR row from a dataset's columns.

    prompt_field       : column holding the prompt (string or chat list) (req).
    ground_truth_field : column holding the verifiable target answer (req).
    system_field       : optional per-row system column.
    instruction        : optional fixed instruction applied to every row.
    """

    prompt_field: str
    ground_truth_field: str
    system_field: str | None = None
    instruction: str = ""


def autodetect_rlvr_mapping(column_names: list[str]) -> RlvrMapping | None:
    """Best-effort guess of an RlvrMapping. Requires a prompt + a ground-truth
    column (e.g. gsm8k's question + answer)."""
    cols = {c.lower(): c for c in column_names}
    prompt = next((cols[c] for c in _PREF_PROMPT_COLS if c in cols), None)
    gt = next((cols[c] for c in _RLVR_GROUND_TRUTH_COLS if c in cols), None)
    if not prompt or not gt or prompt == gt:
        return None
    return RlvrMapping(prompt_field=prompt, ground_truth_field=gt,
                       system_field=cols.get("system"))


def rlvr_row_from_hf(
    row: dict,
    mapping: RlvrMapping,
    class_labels: dict[str, list[str]] | None = None,
) -> dict | None:
    """Convert one HF row to a canonical RLVR row {messages:[...prompt], ground_truth:"..."},
    or None if it can't be formed (missing prompt or ground_truth). The prompt is
    turns-only (no answer); the verifiable target goes in ground_truth."""
    class_labels = class_labels or {}
    ground_truth = _stringify(row.get(mapping.ground_truth_field),
                              class_labels.get(mapping.ground_truth_field))
    if not ground_truth.strip():
        return None

    pv = row.get(mapping.prompt_field)
    pchat = _as_chat_list(pv)
    prompt_msgs = pchat if pchat else [
        {"role": "user", "content": _stringify(pv, class_labels.get(mapping.prompt_field))}]
    if not any(t.get("content", "").strip() for t in prompt_msgs):
        return None  # need a non-empty prompt
    # A chat-list prompt must not already contain the answer (no trailing assistant).
    if prompt_msgs and prompt_msgs[-1].get("role") == "assistant":
        prompt_msgs = prompt_msgs[:-1] or prompt_msgs

    system_parts: list[str] = []
    if mapping.instruction.strip():
        system_parts.append(mapping.instruction.strip())
    if mapping.system_field:
        sv = _stringify(row.get(mapping.system_field), None)
        if sv.strip():
            system_parts.append(sv.strip())
    messages = list(prompt_msgs)
    if system_parts and not any(t.get("role") == "system" for t in messages):
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + messages
    return {"messages": messages, "ground_truth": ground_truth.strip()}


def sample_rlvr_to_jsonl(
    dataset: str,
    mapping: RlvrMapping,
    config: str,
    split: str,
    max_rows: int,
    seed: int = 42,
    token: str | None = None,
) -> tuple[list[dict], dict]:
    """Fetch a deterministic sample and convert it to canonical RLVR rows
    ({messages, ground_truth}). Returns (rows, stats); unconvertible rows skipped."""
    return _sample_converted(
        dataset, config, split, max_rows,
        lambda r, cl: rlvr_row_from_hf(r, mapping, cl),
        seed=seed, token=token,
    )


# --- RLAIF (prompt-only; an AI judge scores the response) -------------------
#
# RLAIF is the simplest shape to ingest: it needs ONLY a prompt column (no answer,
# no ground_truth, no label — the AI judge reward-prompt scores a freshly generated
# response at training time). It reuses the same prompt-column candidates as the
# other RL shapes.

@dataclass
class RlaifMapping:
    """How to build an RLAIF (prompt-only) row from a dataset's columns.

    prompt_field  : column holding the prompt (string or chat list) (req).
    system_field  : optional per-row system column.
    instruction   : optional fixed instruction applied to every row.
    """

    prompt_field: str
    system_field: str | None = None
    instruction: str = ""


def autodetect_rlaif_mapping(column_names: list[str]) -> RlaifMapping | None:
    """Best-effort guess of an RlaifMapping. Requires only a prompt column (RLAIF
    has no answer/ground_truth — the judge scores the response subjectively)."""
    cols = {c.lower(): c for c in column_names}
    prompt = next((cols[c] for c in _PREF_PROMPT_COLS if c in cols), None)
    if not prompt:
        return None
    return RlaifMapping(prompt_field=prompt, system_field=cols.get("system"))


def rlaif_row_from_hf(
    row: dict,
    mapping: RlaifMapping,
    class_labels: dict[str, list[str]] | None = None,
) -> dict | None:
    """Convert one HF row to a canonical RLAIF row {messages:[...prompt]} (PROMPT
    ONLY — no answer), or None if no non-empty prompt can be formed. A chat-list
    prompt that already ends in an assistant turn has it dropped (RLAIF trains on
    the prompt; the judge scores a fresh generation, so no gold answer is kept)."""
    class_labels = class_labels or {}
    pv = row.get(mapping.prompt_field)
    pchat = _as_chat_list(pv)
    prompt_msgs = pchat if pchat else [
        {"role": "user", "content": _stringify(pv, class_labels.get(mapping.prompt_field))}]
    if not any(t.get("content", "").strip() for t in prompt_msgs):
        return None  # need a non-empty prompt
    if prompt_msgs and prompt_msgs[-1].get("role") == "assistant":
        prompt_msgs = prompt_msgs[:-1] or prompt_msgs

    system_parts: list[str] = []
    if mapping.instruction.strip():
        system_parts.append(mapping.instruction.strip())
    if mapping.system_field:
        sv = _stringify(row.get(mapping.system_field), None)
        if sv.strip():
            system_parts.append(sv.strip())
    messages = list(prompt_msgs)
    if system_parts and not any(t.get("role") == "system" for t in messages):
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}] + messages
    return {"messages": messages}


def sample_rlaif_to_jsonl(
    dataset: str,
    mapping: RlaifMapping,
    config: str,
    split: str,
    max_rows: int,
    seed: int = 42,
    token: str | None = None,
) -> tuple[list[dict], dict]:
    """Fetch a deterministic sample and convert it to canonical RLAIF rows
    ({messages} prompt-only). Returns (rows, stats); unconvertible rows skipped."""
    return _sample_converted(
        dataset, config, split, max_rows,
        lambda r, cl: rlaif_row_from_hf(r, mapping, cl),
        seed=seed, token=token,
    )
