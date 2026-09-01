# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic dataset profiler + recommended eval strategy (advisory).

The platform validates that a dataset is well-FORMED (chat-template correct) and
that train/eval are disjoint — but never asks "what IS this dataset, what should
the model learn, and which metrics should judge it?". This profiler answers the
data-derivable half of that, cheaply and without an LLM, so the "Investigate
dataset" wizard can recommend an eval strategy + flag problems BEFORE a billable
run. (The agentic follow-up Q&A — business context the data can't reveal — layers
on later via AgentCore/Strands; this module is deterministic + advisory only.)

It mirrors the PdM "explore → classify → recommend" pattern, adapted to
instruction-tuning chat JSONL:
  * structure   — roles, single vs multi-turn, fixed vs varied system prompt
  * task type   — classification / json / numeric / free-text (+ how consistent)
  * outputs     — label balance, JSON-schema consistency, answer-length dist,
                  reasoning(<think>) prefix rate
  * quality     — empty/duplicate/malformed outputs, truncation risk vs cutoff
  * leakage     — train↔eval exact overlap (beyond the disjointness assertion)
  * strategy    — which of our task-aware metrics to rank on + a one-line why

Everything here is read-only over the persisted split files; large datasets are
sampled (MAX_PROFILE_ROWS) so it stays cheap in a Lambda.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from typing import Any

from .storage import split_dir

# Cap rows actually parsed for the profile — enough to characterize a dataset
# without pulling tens of MB into memory. Counts are reported as "sampled".
MAX_PROFILE_ROWS = 5000

_NUM_RE = re.compile(r"^[+-]?\$?\s*\d[\d,]*\.?\d*\s*%?$")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# General "scaffolding wrapper" detection: modern instruct/reasoning models (and
# some datasets) wrap the real answer in markup the task never asked for. A fixed
# <think> check only catches Qwen3; this set covers the common families so the
# signal generalizes. Each entry is (label, compiled-regex). Detection is on the
# GOLD answer here (a heads-up that the harness will need to extract before
# scoring); eval.py does the actual stripping at inference time.
_SCAFFOLD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("think", re.compile(r"<think>|<thinking>", re.IGNORECASE)),
    ("reasoning", re.compile(r"<reasoning>|<reason>|<scratchpad>", re.IGNORECASE)),
    ("gpt_oss_channel", re.compile(r"<\|channel\|>|<\|start\|>|<\|message\|>")),
    ("code_fence", re.compile(r"```")),
    ("answer_tag", re.compile(r"<answer>|\[/?INST\]", re.IGNORECASE)),
]


def detect_scaffold(text: str) -> list[str]:
    """Which scaffolding-wrapper patterns appear in `text` (may be several)."""
    return [label for label, rx in _SCAFFOLD_PATTERNS if rx.search(text or "")]


# --- task detection (kept in sync with container/eval.py detect_task) -------

def _first_json(s: str) -> Any | None:
    s = s.strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def detect_task(gold: str) -> str:
    """json | numeric | label | text — from a single gold answer (see eval.py).
    Strips a <think>…</think> reasoning block first so reasoning-model answers are
    classified by their real payload, not the scaffolding."""
    g = _strip_think((gold or "").strip())
    if g[:1] in "{[" and _first_json(g) is not None:
        return "json"
    if _NUM_RE.match(g):
        return "numeric"
    if len(g.split()) <= 4 and not re.search(r"[.!?]", g):
        return "label"
    return "text"


def _word_len(s: str) -> int:
    return len(s.split())


def _strip_think(s: str) -> str:
    return _THINK_RE.sub("", s).strip()


def _read_rows(split_id: str, file_name: str) -> tuple[list[dict], int, bool]:
    """Parse up to MAX_PROFILE_ROWS message-objects from a split file.
    Returns (rows, total_lines, sampled?). rows is [{prompt_msgs, gold}]."""
    run_dir = split_dir(split_id)
    if run_dir is None:
        raise ValueError(f"split {split_id} not found")
    path = run_dir / file_name
    if not path.exists():
        return [], 0, False
    rows: list[dict] = []
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        if len(rows) >= MAX_PROFILE_ROWS:
            continue  # keep counting total, stop parsing
        try:
            msgs = json.loads(line)["messages"]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            rows.append({"malformed": True})
            continue
        assistant_idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        if not assistant_idx:
            rows.append({"malformed": True})
            continue
        last = max(assistant_idx)
        rows.append({"prompt_msgs": msgs[:last], "gold": msgs[last].get("content", ""), "all": msgs})
    return rows, total, total > len(rows)


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": round(s[0], 2),
        "p50": round(statistics.median(s), 2),
        "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 2),
        "max": round(s[-1], 2),
        "mean": round(statistics.fmean(s), 2),
    }


def _profile_file(split_id: str, file_name: str, cutoff_len: int | None) -> dict[str, Any]:
    rows, total, sampled = _read_rows(split_id, file_name)
    parsed = [r for r in rows if not r.get("malformed")]
    malformed = sum(1 for r in rows if r.get("malformed"))
    if not parsed:
        return {"rows": total, "sampled": sampled, "parsed": 0, "malformed": malformed}

    golds = [r["gold"] for r in parsed]
    tasks = [detect_task(g) for g in golds]
    task_mix = dict(Counter(tasks))
    dominant = max(task_mix, key=task_mix.get)
    consistency = round(task_mix[dominant] / len(tasks), 4)

    # output characteristics
    gold_wordlens = [_word_len(g) for g in golds]
    empty = sum(1 for g in golds if not g.strip())
    dup = len(golds) - len(set(golds))

    # Scaffolding-wrapper detection (generalized — see detect_scaffold). Reports
    # the fraction of gold answers wrapped in ANY known scaffold + a breakdown by
    # pattern, so the wizard can warn "answers are wrapped; the harness extracts
    # before scoring" for more than just Qwen3's <think>.
    scaffold_hits = [detect_scaffold(g) for g in golds]
    wrapped = sum(1 for h in scaffold_hits if h)
    pattern_counts: Counter = Counter()
    for h in scaffold_hits:
        pattern_counts.update(h)
    scaffold = {
        "rate": round(wrapped / len(golds), 4),
        "patterns": {k: round(v / len(golds), 4) for k, v in pattern_counts.most_common()},
    }

    # JSON-specific: validity (raw + after stripping <think>) + schema consistency
    json_block: dict[str, Any] = {}
    if task_mix.get("json"):
        json_golds = [g for g, t in zip(golds, tasks) if t == "json"]
        raw_valid = sum(1 for g in json_golds if _first_json(g) is not None)
        stripped_valid = sum(1 for g in json_golds if _first_json(_strip_think(g)) is not None)
        keysets = Counter()
        for g in json_golds:
            obj = _first_json(_strip_think(g))
            if isinstance(obj, dict):
                keysets[tuple(sorted(obj.keys()))] += 1
        top_schema = keysets.most_common(1)[0] if keysets else (None, 0)
        json_block = {
            "jsonRows": len(json_golds),
            "goldValidRaw": round(raw_valid / len(json_golds), 4),
            "goldValidStripped": round(stripped_valid / len(json_golds), 4),
            "schemaConsistency": round(top_schema[1] / len(json_golds), 4) if json_golds else None,
            "dominantKeys": list(top_schema[0]) if top_schema[0] else [],
            "distinctSchemas": len(keysets),
        }

    # label-specific: class balance
    label_block: dict[str, Any] = {}
    if task_mix.get("label"):
        label_golds = [g.strip() for g, t in zip(golds, tasks) if t == "label"]
        counts = Counter(label_golds)
        n = sum(counts.values())
        dist_pct = {k: round(v / n, 4) for k, v in counts.most_common(20)}
        rarest = min(dist_pct.values()) if dist_pct else None
        label_block = {
            "labelRows": n,
            "numClasses": len(counts),
            "distribution": dist_pct,
            # Full class list (sorted by frequency), so the wizard can show every
            # class — not just the top-N in `distribution`.
            "classes": [k for k, _ in counts.most_common()],
            "minorityRate": rarest,
            "imbalanced": bool(rarest is not None and rarest < 0.05),
        }

    # truncation risk: rough token estimate (~1.3 words/token) of prompt+gold vs cutoff
    trunc = {}
    if cutoff_len:
        approx_tokens = [round((_word_len(r["gold"]) +
                                sum(_word_len(m.get("content", "")) for m in r.get("prompt_msgs", []))) * 1.3)
                         for r in parsed]
        over = sum(1 for t in approx_tokens if t > cutoff_len)
        trunc = {
            "cutoffLen": cutoff_len,
            "approxTokenP95": _dist(approx_tokens).get("p95"),
            "approxTokenMax": _dist(approx_tokens).get("max"),
            "estTruncatedRows": round(over / len(approx_tokens), 4),
        }

    return {
        "rows": total,
        "sampled": sampled,
        "parsed": len(parsed),
        "malformed": malformed,
        "emptyOutputs": empty,
        "duplicateOutputs": dup,
        "taskMix": task_mix,
        "dominantTask": dominant,
        "taskConsistency": consistency,
        "scaffold": scaffold,
        "goldWordLen": _dist(gold_wordlens),
        "json": json_block,
        "labels": label_block,
        "truncation": trunc,
    }


def _structure(split_id: str) -> dict[str, Any]:
    """Conversation structure from the train file: roles, turns, system-prompt variety."""
    rows, _, _ = _read_rows(split_id, "train.jsonl")
    parsed = [r for r in rows if not r.get("malformed")]
    if not parsed:
        return {}
    turn_counts = [sum(1 for m in r["all"] if m.get("role") in ("user", "assistant")) for r in parsed]
    has_system = sum(1 for r in parsed if any(m.get("role") == "system" for m in r["all"]))
    systems = {next((m.get("content", "") for m in r["all"] if m.get("role") == "system"), "") for r in parsed}
    systems.discard("")
    multi_turn = sum(1 for tc in turn_counts if tc > 2)
    # When the system prompt is fixed (one distinct value), surface its text so
    # the wizard can show WHAT the task instruction is — important now that HF
    # imports put the instruction in the system turn.
    fixed_system = next(iter(systems)) if len(systems) == 1 else None
    return {
        "hasSystemPromptRate": round(has_system / len(parsed), 4),
        "systemPromptFixed": len(systems) <= 1,
        "distinctSystemPrompts": len(systems),
        "fixedSystemPrompt": fixed_system,
        "multiTurnRate": round(multi_turn / len(parsed), 4),
        "turnsPerExample": _dist([float(t) for t in turn_counts]),
    }


def _leakage(split_id: str) -> dict[str, Any]:
    """Exact (prompt,gold) overlap between train and eval — beyond the disjointness
    assertion done at split time (this re-checks on the persisted files)."""
    tr, _, _ = _read_rows(split_id, "train.jsonl")
    ev, _, _ = _read_rows(split_id, "eval.jsonl")
    def keys(rows):
        out = set()
        for r in rows:
            if r.get("malformed"):
                continue
            p = json.dumps(r.get("prompt_msgs"), sort_keys=True)
            out.add((p, r.get("gold", "")))
        return out
    tk, ek = keys(tr), keys(ev)
    if not ek:
        return {"checked": False}
    overlap = len(tk & ek)
    return {
        "checked": True,
        "exactOverlapRows": overlap,
        "evalLeakedRate": round(overlap / len(ek), 4),
    }


def recommend_eval_strategy(prof: dict[str, Any]) -> dict[str, Any]:
    """Deterministic recommendation: which metric to RANK on + which to watch +
    a plain-language rationale, derived from the eval/train profile. Advisory."""
    f = prof.get("eval") or prof.get("train") or {}
    task = f.get("dominantTask", "text")
    consistency = f.get("taskConsistency", 1.0)
    rank, watch, notes = "token_f1", [], []

    if task == "label":
        rank = "label_accuracy"
        watch = ["per_class_accuracy"]
        notes.append("Classification task (short labels) → rank on label accuracy; watch per-class for imbalance.")
        if f.get("labels", {}).get("imbalanced"):
            notes.append(
                f"Imbalanced: minority class ~{round((f['labels']['minorityRate'] or 0)*100,1)}% — "
                "accuracy can mislead; also watch minority-class recall."
            )
    elif task == "json":
        rank = "json_structural"
        watch = ["json_valid", "json_key_recall", "llm_judge:faithfulness"]
        notes.append("JSON-generation task → rank on JSON structural match; gate on json_valid.")
        j = f.get("json", {})
        scaffold_rate = f.get("scaffold", {}).get("rate", 0)
        if scaffold_rate > 0.2:
            pats = ", ".join(f.get("scaffold", {}).get("patterns", {}).keys()) or "scaffolding"
            notes.append(
                f"{round(scaffold_rate*100)}% of outputs wrap the JSON in {pats} — "
                "the harness extracts the answer; trust json_valid over raw exact match."
            )
        if j.get("distinctSchemas", 0) > 1:
            notes.append(f"{j['distinctSchemas']} distinct JSON schemas — outputs aren't a single fixed shape.")
        notes.append("Open-ended narrative JSON → add the LLM judge (faithfulness) to catch hallucination.")
    elif task == "numeric":
        rank = "numeric_match"
        watch = ["token_f1"]
        notes.append("Numeric-answer task → rank on numeric match (tolerant of $/comma/%).")
    else:
        rank = "token_f1"
        watch = ["rouge_l", "char_f1", "llm_judge:overall"]
        notes.append("Free-text generation → rank on Token-F1; reference-overlap understates quality, "
                     "so add the LLM judge for the real signal.")

    if consistency < 0.9:
        notes.append(
            f"Mixed task types (dominant only {round(consistency*100)}%) — metrics are reported per "
            "task automatically; consider splitting the dataset by task if intentional."
        )
    return {"rankMetric": rank, "alsoWatch": watch, "rationale": notes, "detectedTask": task}


def warnings_from(prof: dict[str, Any]) -> list[dict[str, str]]:
    """Actionable, severity-tagged flags surfaced at the top of the wizard."""
    out: list[dict[str, str]] = []

    def add(sev: str, msg: str) -> None:
        out.append({"severity": sev, "message": msg})

    for which in ("train", "eval"):
        f = prof.get(which) or {}
        if not f.get("parsed"):
            continue
        if f.get("malformed"):
            add("error", f"{which}: {f['malformed']} malformed row(s) couldn't be parsed.")
        if f.get("emptyOutputs"):
            add("warning", f"{which}: {f['emptyOutputs']} empty assistant answer(s).")
        # Duplicate-output diversity check — but NOT for classification, where a
        # small fixed label set means duplicates are inherent, not a problem.
        if (
            f.get("duplicateOutputs", 0)
            and which == "train"
            and f["parsed"]
            and f.get("dominantTask") != "label"
        ):
            rate = f["duplicateOutputs"] / f["parsed"]
            if rate > 0.3:
                add("warning", f"train: {round(rate*100)}% of outputs are duplicates — low output diversity.")
        t = f.get("truncation", {})
        if t.get("estTruncatedRows", 0) > 0.05:
            add("warning",
                f"{which}: ~{round(t['estTruncatedRows']*100)}% of rows likely exceed cutoff_len "
                f"({t.get('cutoffLen')}) and will be truncated — raise cutoff or shorten data.")
        j = f.get("json", {})
        if j.get("goldValidRaw") is not None and j["goldValidRaw"] < 0.95:
            add("warning",
                f"{which}: only {round(j['goldValidRaw']*100)}% of JSON gold answers parse as raw JSON "
                f"({round((j.get('goldValidStripped') or 0)*100)}% after stripping <think>).")
        lab = f.get("labels", {})
        if lab.get("imbalanced"):
            add("info", f"{which}: class imbalance — rarest label ~{round((lab['minorityRate'] or 0)*100,1)}%.")

    lk = prof.get("leakage", {})
    if lk.get("checked") and lk.get("exactOverlapRows"):
        add("error", f"Leakage: {lk['exactOverlapRows']} eval row(s) "
                     f"({round(lk['evalLeakedRate']*100,1)}%) also appear in train.")

    # --- Preference (DPO) data-quality checks (grounded in Rafailov et al. 2023) ---
    pref = prof.get("preference") or {}
    if pref.get("pairs"):
        n = pref["pairs"]
        if pref.get("malformed"):
            add("error", f"DPO: {pref['malformed']} malformed preference row(s) couldn't be parsed.")
        # Identical chosen==rejected carry ZERO preference gradient (DPO Eq. 1–3).
        ident = pref.get("identicalPairs", 0)
        if ident:
            sev = "error" if ident / n > 0.2 else "warning"
            add(sev, f"DPO: {ident} pair(s) ({round(ident/n*100)}%) have identical chosen/rejected "
                     f"— these give no preference signal; remove them.")
        # Length bias: if `chosen` is consistently longer, DPO learns VERBOSITY, not
        # quality — the most-cited DPO failure (Rafailov et al. 2023, App. D.2).
        cw = (pref.get("chosenWordLen") or {}).get("p50")
        rw = (pref.get("rejectedWordLen") or {}).get("p50")
        if cw and rw and rw > 0:
            ratio = cw / rw
            if ratio >= 1.5:
                add("warning",
                    f"DPO: chosen responses are ~{ratio:.1f}× longer than rejected (median {cw:g} vs "
                    f"{rw:g} words) — the model may learn verbosity, not quality. Confirm length is a "
                    f"real signal, or balance the lengths.")

    # --- KTO data-quality checks (grounded in Ethayarajh et al. 2024) ---
    kto = prof.get("kto") or {}
    if kto.get("rows"):
        nd, nu = kto.get("desirable", 0), kto.get("undesirable", 0)
        if kto.get("malformed"):
            add("error", f"KTO: {kto['malformed']} malformed row(s) couldn't be parsed.")
        if nd == 0 or nu == 0:
            add("error", "KTO needs BOTH desirable and undesirable examples to contrast — "
                         f"this set has only {'desirable' if nd else 'undesirable'} ones.")
        else:
            # KTO §4.2 / Eq. 9: keep (λD·nD)/(λU·nU) ∈ [1, 4/3]. With default λ=1 a
            # skewed ratio silently degrades training; recommend re-weighting.
            ratio = max(nd, nu) / min(nd, nu)
            if ratio >= 1.5:
                bigger, smaller = ("desirable", "undesirable") if nd > nu else ("undesirable", "desirable")
                # Cite the CONCRETE recommended weights (same numbers the profile
                # card prefills) so the warning is actionable, not just directional.
                cw = kto.get("recommendedChosenWeight", 1.0)
                rw = kto.get("recommendedRejectedWeight", 1.0)
                add("warning",
                    f"KTO: class imbalance {nd}:{nu} desirable:undesirable (~{ratio:.1f}× more {bigger}). "
                    f"Per the KTO paper (§4.2), up-weight the {smaller} class to keep "
                    f"λD·nD/λU·nU ≈ 1 — set kto_chosen_weight={cw:g}, kto_rejected_weight={rw:g} "
                    f"(or add more {smaller} examples).")
    return out


def _profile_preference(split_id: str) -> dict[str, Any]:
    """Profile the ranking pairs of a PREFERENCE (DPO) train file: pair count, how
    distinct chosen vs rejected are, and prompt/response lengths. Reads the raw
    {messages, chosen, rejected} rows directly (NOT via the messages parser, which
    expects a final assistant turn this shape doesn't have)."""
    run_dir = split_dir(split_id)
    if run_dir is None:
        return {"pairs": 0}
    path = run_dir / "train.jsonl"
    if not path.exists():
        return {"pairs": 0}
    pairs = 0
    malformed = 0
    identical = 0
    chosen_lens: list[float] = []
    rejected_lens: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ch = row["chosen"]
            rj = row["rejected"]
            ch_txt = ch.get("content", "") if isinstance(ch, dict) else str(ch)
            rj_txt = rj.get("content", "") if isinstance(rj, dict) else str(rj)
            assert isinstance(row.get("messages"), list)
        except (json.JSONDecodeError, KeyError, TypeError, AssertionError):
            malformed += 1
            continue
        pairs += 1
        if ch_txt.strip() == rj_txt.strip():
            identical += 1
        chosen_lens.append(_word_len(ch_txt))
        rejected_lens.append(_word_len(rj_txt))
    cd = _dist(chosen_lens)
    rd = _dist(rejected_lens)
    # Concrete length-bias ratio (median chosen ÷ median rejected words). >~1.5
    # means DPO may learn verbosity rather than quality (Rafailov et al. 2023,
    # App. D.2) — surfaced as a first-class field so the DPO profile card can show
    # the number, not just the prose warning. None when rejected median is 0.
    cw, rw = cd.get("p50"), rd.get("p50")
    length_bias_ratio = round(cw / rw, 2) if cw and rw and rw > 0 else None
    return {
        "pairs": pairs,
        "malformed": malformed,
        "identicalPairs": identical,
        "chosenWordLen": cd,
        "rejectedWordLen": rd,
        "lengthBiasRatio": length_bias_ratio,
    }


def _profile_kto(split_id: str) -> dict[str, Any]:
    """Profile a KTO train file: row count + the desirable/undesirable balance.
    Reads raw {messages, kto_tag} rows (the messages parser can't — there's no
    pairing, just a per-row label)."""
    run_dir = split_dir(split_id)
    if run_dir is None:
        return {"rows": 0}
    path = run_dir / "train.jsonl"
    if not path.exists():
        return {"rows": 0}
    rows = malformed = desirable = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            tag = bool(row["kto_tag"])
            assert isinstance(row.get("messages"), list)
        except (json.JSONDecodeError, KeyError, TypeError, AssertionError):
            malformed += 1
            continue
        rows += 1
        desirable += int(tag)
    undesirable = rows - desirable
    out: dict[str, Any] = {
        "rows": rows,
        "malformed": malformed,
        "desirable": desirable,
        "undesirable": undesirable,
        "desirableRate": round(desirable / rows, 4) if rows else None,
    }
    out.update(_kto_weight_recommendation(desirable, undesirable))
    return out


def _kto_weight_recommendation(desirable: int, undesirable: int) -> dict[str, Any]:
    """Concrete KTO loss-weight recommendation (LF kto_chosen_weight /
    kto_rejected_weight = the KTO paper's λ_D / λ_U).

    Per Ethayarajh et al. 2024 §4.2 / Eq. 9, KTO is stable when the *effective*
    contribution of each class is balanced, i.e. keep (λ_D·n_D)/(λ_U·n_U) in the
    range [1, 4/3]. With the default λ=1 a skewed n_D:n_U silently degrades
    training. We raise λ on the MINORITY class to bring the product ratio toward 1,
    capped at the platform's weight bound. Returns a ready-to-use pair the UI can
    prefill into the FineTune KTO weight inputs.

    Always returns all fields so the UI can render unconditionally:
      imbalanceRatio: max(nD,nU)/min(nD,nU) (None if a class is missing)
      recommendedChosenWeight / recommendedRejectedWeight: λ_D / λ_U to apply
      weightsBalanced: True when no re-weighting is needed (ratio < 1.5)
    """
    if desirable <= 0 or undesirable <= 0:
        # Degenerate (a missing class) — warnings_from emits a hard error; no
        # meaningful weighting exists, so recommend the neutral default.
        return {
            "imbalanceRatio": None,
            "recommendedChosenWeight": 1.0,
            "recommendedRejectedWeight": 1.0,
            "weightsBalanced": True,
        }
    ratio = max(desirable, undesirable) / min(desirable, undesirable)
    chosen_w, rejected_w = 1.0, 1.0
    if ratio >= 1.5:
        # Up-weight the minority class so λ_minority·n_minority ≈ n_majority
        # (product ratio → 1). Cap at the catalog's KTO weight bound.
        from .catalog import _KTO_WEIGHT_MAX

        boost = round(min(ratio, _KTO_WEIGHT_MAX), 2)
        if desirable < undesirable:
            chosen_w = boost  # fewer desirable → up-weight the desirable (chosen) loss
        else:
            rejected_w = boost
    return {
        "imbalanceRatio": round(ratio, 2),
        "recommendedChosenWeight": chosen_w,
        "recommendedRejectedWeight": rejected_w,
        "weightsBalanced": ratio < 1.5,
    }


def _profile_rlvr(split_id: str) -> dict[str, Any]:
    """Profile an RLVR train file: row count + prompt/ground_truth length dists +
    how machine-verifiable the ground_truth looks (numeric / short-token / json —
    the kinds a preset/custom reward can actually check). Reads raw
    {messages:[prompt], ground_truth} rows directly — the messages parser would
    flag them malformed since there's no final assistant turn."""
    run_dir = split_dir(split_id)
    if run_dir is None:
        return {"rows": 0}
    path = run_dir / "train.jsonl"
    if not path.exists():
        return {"rows": 0}
    rows = malformed = empty_gt = 0
    numeric_gt = short_gt = code_gt = 0
    prompt_lens: list[float] = []
    gt_lens: list[float] = []
    task_counts: dict[str, int] = {}
    num_re = re.compile(r"^[+-]?\$?\s*\d[\d,]*\.?\d*\s*%?$")
    code_re = re.compile(r"```|def |class |function |import |#include|=>|;\s*$", re.MULTILINE)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            msgs = row["messages"]
            gt = row["ground_truth"]
            assert isinstance(msgs, list) and isinstance(gt, str)
        except (json.JSONDecodeError, KeyError, TypeError, AssertionError):
            malformed += 1
            continue
        rows += 1
        gt_s = gt.strip()
        if not gt_s:
            empty_gt += 1
        # Verifiability heuristics: is the gold a CHECKABLE target (the thing a
        # reward function needs)? A bare number (gsm8k/prime_math) or a short
        # token answer scores well; long prose ground_truth is a yellow flag.
        # Look for a number anywhere (handles the gsm8k "#### 72" convention).
        if num_re.match(gt_s) or re.search(r"\d", gt_s):
            numeric_gt += 1
        if len(gt_s.split()) <= 6:
            short_gt += 1
        if code_re.search(gt_s):
            code_gt += 1
        # Per-row task classification (reuse the eval task detector) so the launch
        # guard can check the picked reward's domain against the data shape.
        t = detect_task(gt_s)
        task_counts[t] = task_counts.get(t, 0) + 1
        prompt_text = " ".join(str(m.get("content", "")) for m in msgs)
        prompt_lens.append(_word_len(prompt_text))
        gt_lens.append(_word_len(gt_s))
    # Dominant ground_truth task — the single best descriptor of what a reward
    # must grade (numeric/json/label/text). Used by the reward-domain guard.
    dominant = max(task_counts, key=task_counts.get) if task_counts else None
    return {
        "rows": rows,
        "malformed": malformed,
        "emptyGroundTruth": empty_gt,
        # Fraction of ground_truth values that look machine-verifiable (contain a
        # number) or are short tokens — i.e. checkable by a preset/custom reward.
        "numericGroundTruthRate": round(numeric_gt / rows, 4) if rows else None,
        "shortGroundTruthRate": round(short_gt / rows, 4) if rows else None,
        "codeGroundTruthRate": round(code_gt / rows, 4) if rows else None,
        # Auto-detected ground_truth task type + the mix, so the launch guard can
        # warn when the chosen reward can't grade this shape (e.g. gsm8k on prose).
        "groundTruthTask": dominant,
        "groundTruthTaskMix": task_counts,
        "promptWordLen": _dist(prompt_lens),
        "groundTruthWordLen": _dist(gt_lens),
    }


def _profile_rlaif(split_id: str, filename: str = "train.jsonl") -> dict[str, Any]:
    """Profile an RLAIF file: row count + prompt length dist + the dominant prompt
    task. RLAIF rows are PROMPT-ONLY ({messages:[prompt]}, no ground_truth — the AI
    judge scores subjectively), so unlike _profile_rlvr there are no verifiability
    rates. Reads raw rows directly (the messages parser would flag a prompt-only
    row malformed since it has no final assistant turn)."""
    run_dir = split_dir(split_id)
    if run_dir is None:
        return {"rows": 0}
    path = run_dir / filename
    if not path.exists():
        return {"rows": 0}
    rows = malformed = empty_prompt = 0
    prompt_lens: list[float] = []
    task_counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            msgs = row["messages"]
            assert isinstance(msgs, list) and msgs
        except (json.JSONDecodeError, KeyError, TypeError, AssertionError):
            malformed += 1
            continue
        rows += 1
        prompt_text = " ".join(str(m.get("content", "")) for m in msgs)
        if not prompt_text.strip():
            empty_prompt += 1
        prompt_lens.append(_word_len(prompt_text))
        # Descriptive only — what KIND of prompt this is (no verifiable target).
        t = detect_task(prompt_text)
        task_counts[t] = task_counts.get(t, 0) + 1
    dominant = max(task_counts, key=task_counts.get) if task_counts else None
    return {
        "rows": rows,
        "malformed": malformed,
        "emptyPrompt": empty_prompt,
        "promptTask": dominant,
        "promptTaskMix": task_counts,
        "promptWordLen": _dist(prompt_lens),
    }


def recommend_objective(prof: dict[str, Any]) -> dict[str, Any]:
    """Recommend the training OBJECTIVE (LLaMA-Factory stage) from the data shape.

    A preference dataset (chosen/rejected pairs) → DPO; a KTO dataset (labelled
    good/bad completions) → KTO; a messages dataset → SFT. Advisory, mirroring
    recommend_eval_strategy: the FineTunePage already gates the objective on the
    dataset shape, but the investigate wizard surfaces the WHY."""
    shape = prof.get("shape")
    if shape == "preference":
        p = prof.get("preference", {}) or {}
        notes = [
            "Preference dataset (chosen/rejected pairs) → train with DPO: the model "
            "learns to prefer the chosen response over the rejected one.",
            "Evaluation is unchanged — the held-out set scores the chosen answer as "
            "gold, so a DPO model lands on the same leaderboard as an SFT one.",
        ]
        if p.get("identicalPairs"):
            notes.append(
                f"{p['identicalPairs']} pair(s) have identical chosen/rejected text — "
                "those carry no preference signal (dropped at import / ignored)."
            )
        return {"objective": "dpo", "rationale": notes}
    if shape == "kto":
        k = prof.get("kto", {}) or {}
        notes = [
            "KTO dataset (completions labelled good/bad) → train with KTO: binary "
            "feedback, no pairing needed — cheaper to collect than DPO pairs.",
            "Evaluation is unchanged — the held-out set scores the DESIRABLE "
            "completions as gold, so a KTO model lands on the same leaderboard.",
        ]
        if k.get("desirableRate") is not None:
            notes.append(
                f"Balance: {k.get('desirable')} desirable / {k.get('undesirable')} undesirable "
                f"({round(k['desirableRate']*100)}% good). KTO tolerates imbalance but needs both."
            )
        return {"objective": "kto", "rationale": notes}
    if shape == "rlvr":
        r = prof.get("rlvr", {}) or {}
        notes = [
            "RLVR dataset (prompt + verifiable ground_truth) → train with RLVR "
            "(GRPO): the model is REWARDED for producing answers a reward function "
            "marks correct, not for imitating a solution. Serverless engine only.",
            "Pick a reward at launch — a preset (gsm8k / prime_math, both reward a "
            "numeric answer) or a custom reward function. The ground_truth must be "
            "checkable in that reward's domain.",
        ]
        nr = r.get("numericGroundTruthRate")
        if nr is not None:
            notes.append(
                f"{round(nr*100)}% of ground_truth values contain a number "
                f"(machine-verifiable for the math presets). "
                + ("Good fit for gsm8k/prime_math." if nr >= 0.8
                   else "If these aren't math answers, use a CUSTOM reward whose logic matches your task.")
            )
        if r.get("emptyGroundTruth"):
            notes.append(
                f"⚠ {r['emptyGroundTruth']} row(s) have an EMPTY ground_truth — those can't be "
                "rewarded and should be removed."
            )
        return {"objective": "rlvr", "rationale": notes}
    if shape == "rlaif":
        r = prof.get("rlaif", {}) or {}
        notes = [
            "RLAIF dataset (prompts only) → train with RLAIF (GRPO from AI "
            "feedback): an AI JUDGE scores each generated response against a reward "
            "PROMPT you define, so the model improves on SUBJECTIVE qualities "
            "(tone, helpfulness, style) with no verifiable answer. Serverless only.",
            "Pick a reward PROMPT at launch (defined on the Reward functions page) — "
            "the judge reads {{prompt}}/{{response}} and returns a 0..1 score. No "
            "ground_truth is needed; the judge replaces the reward function.",
        ]
        if r.get("rows"):
            notes.append(f"{r['rows']} prompt(s) for the model to generate from.")
        if r.get("emptyPrompt"):
            notes.append(
                f"⚠ {r['emptyPrompt']} row(s) have an EMPTY prompt — those give the model "
                "nothing to respond to and should be removed."
            )
        return {"objective": "rlaif", "rationale": notes}
    return {
        "objective": "sft",
        "rationale": ["Messages dataset (prompt → response) → standard SFT."],
    }


def profile_dataset(split_id: str, cutoff_len: int | None = None) -> dict[str, Any]:
    """Full deterministic profile of a dataset + a recommended eval strategy.

    cutoff_len (optional) enables truncation-risk estimation against a model's
    sequence length. Advisory only — computes + recommends, never mutates data."""
    from .storage import split_meta

    meta = split_meta(split_id)
    shape = meta.get("shape", "sft")
    prof: dict[str, Any] = {
        "splitId": split_id,
        "name": meta.get("name"),
        # Data shape: "preference" (DPO chosen/rejected) vs SFT messages. Drives the
        # recommended OBJECTIVE below; the train rows of a preference set are
        # ranking-shaped (not messages), so train-file profiling is skipped for them.
        "shape": shape,
        "hasVal": meta.get("hasVal", False),
        "evalOnly": meta.get("evalOnly", False),
        # Provenance — where the dataset came from. For HF imports this carries
        # the source dataset id + the sampling parameters, so the wizard can show
        # exactly what was loaded (and that it's a sample, not the full set).
        "provenance": {
            "source": meta.get("source"),
            "hfDataset": meta.get("hfDataset"),
            "hfConfig": meta.get("hfConfig"),
            "hfSplit": meta.get("hfSplit"),
            "hfSampleSeed": meta.get("hfSampleSeed"),
        },
        "structure": _structure(split_id),
        # The held-out eval is ALWAYS messages-shaped (for a preference set it's the
        # chosen-as-gold projection), so eval profiling + the eval-metric
        # recommendation work uniformly — EXCEPT RLAIF, whose eval rows are
        # prompt-only (no gold), profiled below. For SFT we also profile train; for a
        # preference set the train rows are ranking-shaped (chosen/rejected), which
        # the messages parser would flag as malformed — so we profile the
        # preference pairs separately instead (see _profile_preference).
        "eval": (_profile_rlaif(split_id, "eval.jsonl") if shape == "rlaif"
                 else _profile_file(split_id, "eval.jsonl", cutoff_len)),
    }
    if shape == "preference":
        prof["preference"] = _profile_preference(split_id)
    elif shape == "kto":
        prof["kto"] = _profile_kto(split_id)
    elif shape == "rlvr":
        # RLVR train rows are {messages:[prompt], ground_truth} — prompt-only, no
        # final assistant turn, so the messages parser can't profile them. Profile
        # the prompt + verifiable-target shape separately (also surfaces row counts
        # so the investigate window isn't empty for RLVR).
        prof["rlvr"] = _profile_rlvr(split_id)
    elif shape == "rlaif":
        # RLAIF train rows are prompt-only {messages:[prompt]} (no ground_truth) —
        # same prompt-only profiling, no verifiability rates.
        prof["rlaif"] = _profile_rlaif(split_id)
    else:
        prof["train"] = _profile_file(split_id, "train.jsonl", cutoff_len)
        if meta.get("hasVal"):
            prof["val"] = _profile_file(split_id, "val.jsonl", cutoff_len)
    prof["leakage"] = _leakage(split_id)
    prof["recommendation"] = recommend_eval_strategy(prof)
    prof["objective"] = recommend_objective(prof)
    prof["warnings"] = warnings_from(prof)
    return prof
