# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Guided Fine-tuning agent — a conversational front door for non-ML users.

This is the "wizard with a personality" front door: an Engagement Manager or
Solutions Architect describes a goal in plain words, brings a dataset, and the
agent profiles it, PROPOSES a fine-tuning RACE (compare N models, pick a winner),
and — only after the human APPROVES — launches it and emails when done. It never
auto-launches a billable race.

ARCHITECTURE (decision locked + research-validated): a BACKEND orchestrator (NOT a
new AgentCore/Strands agent). It drives a small DETERMINISTIC state machine over
conversation turns and calls Bedrock Converse for NARRATION ONLY — the same
advisor.py/judge.py pattern. Every load-bearing decision (objective, models,
hyperparameters, gates, the launch) is rule-based, composed from existing verified
pieces (profiler / recommend_objective / race_planner / race.start_race / notify).
The LLM only writes the plain-language prose around an already-final plan; its
output is never parsed back into config, and any LLM/parse error degrades to a
deterministic templated string.

STATE MACHINE (server-authoritative; each user turn carries {action, payload}):
  GREET → COLLECT_GOAL → AWAIT_DATA → PROFILING → CONFIRM_TASK
        → CHOOSE_EFFORT → BUILDING_PLAN → REVIEW_PLAN → LAUNCHED → DONE
REVIEW_PLAN NEVER auto-advances to a launch — the user must explicitly approve.

PERSISTENCE: one JSON doc per session in the per-tenant "pitcrew" store collection,
so sessions are isolated per user and the ChatGPT-style history sidebar is just
list_keys()+key_mtime(). No DB; an optimistic `version` guards against a second tab
clobbering the phase.

V1 SCOPE: SFT / DPO / KTO. RLVR/RLAIF (reward-based) are detected and politely
declined (they need reward setup the guided flow doesn't do yet). Synthetic data
generation (the "no usable data" path) is not implemented.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .obs import log_event
from .store import get_store

# Per-tenant store collection (one doc per session). Mirrors races/investigations.
PITCREW = "pitcrew"
SESSION_FILE = "session.json"

# --- Phases ------------------------------------------------------------------
GREET = "greet"
COLLECT_GOAL = "collect_goal"
AWAIT_DATA = "await_data"
PROFILING = "profiling"
CONFIRM_TASK = "confirm_task"
CHOOSE_EFFORT = "choose_effort"
BUILDING_PLAN = "building_plan"
REVIEW_PLAN = "review_plan"
LAUNCHED = "launched"
DONE = "done"

# Appetite CEILINGS ("up to N"), not fixed counts — the planner fills only as many arms
# as add real signal, so a small/simple dataset yields fewer (and says why) rather than
# padding a race with redundant billable jobs. Keys match race_planner.EFFORT_JOBS.
EFFORT_JOBS_LABEL = {"quick": "up to 4 models", "balanced": "up to 8 models",
                     "thorough": "up to 16 models"}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Minimum usable training examples for the guided flow to build a plan. Below this,
# fine-tuning can't learn anything reliable, so we block at the DATA step (in plain
# language) rather than letting the user reach the billable approve click. A soft,
# generous floor — the real per-objective gates still run at launch.
MIN_USABLE_ROWS = 20


# =============================================================================
# Session persistence
# =============================================================================
def _friendly_stamp(stamp: str) -> str:
    """A short human label from a "YYYYMMDD-HHMMSS" stamp (e.g. "30 Jun, 14:30"),
    used to give each brand-new session a DISTINGUISHABLE default title. Pure: parses
    the caller-supplied stamp string, never reads the clock. Falls back to the raw
    stamp if it isn't the expected shape."""
    _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    try:
        date, time = stamp.split("-", 1)
        mon = _MONTHS[int(date[4:6]) - 1]
        return f"{int(date[6:8])} {mon}, {time[0:2]}:{time[2:4]}"
    except (ValueError, IndexError):
        return stamp


def _new_session(session_id: str, stamp: str) -> dict[str, Any]:
    """A fresh session doc. `stamp` is caller-supplied (no time in lib code)."""
    return {
        "sessionId": session_id,
        "createdAt": stamp,
        "updatedAt": stamp,
        "version": 0,
        "phase": GREET,
        # Distinguishable default (the time it was started) so a list of not-yet-named
        # sessions isn't a wall of identical "New session" rows. Becomes the goal text
        # once the user describes their task, and is user-renamable any time.
        "title": f"New session · {_friendly_stamp(stamp)}",
        # True once the user renames the session themselves — then the goal step
        # won't overwrite their chosen title.
        "titleManual": False,
        # Collected inputs.
        "goal": "",
        "splitId": "",
        "datasetName": "",
        "effort": "",
        # Derived artifacts (filled as the machine advances).
        "profile": None,          # the deterministic dataset profile
        "shape": "",              # detected/known data shape
        "objective": "",          # data-fixed objective (sft/dpo/kto)
        "rankMetric": "",         # recommended leaderboard metric
        "plan": None,             # the view dict shown on the review screen
        # The editable source of truth for the race: one spec per model
        # (modelId/method/variant/prefLoss). The review screen edits this SET and
        # the launch rebuilds gate-valid models from it (never trusting client hp).
        "planSpecs": [],
        "estimate": None,         # cost_estimate.estimate_race()
        "raceId": "",
        "notifyEmail": "",
        # The conversation transcript (assistant + user messages) for the UI thread.
        "messages": [],
    }


def _load(session_id: str) -> dict[str, Any] | None:
    raw = get_store().read_file(PITCREW, session_id, SESSION_FILE)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _save(session: dict[str, Any]) -> None:
    store = get_store()
    sid = session["sessionId"]
    wd = store.workdir(PITCREW, sid)
    (wd / SESSION_FILE).write_text(json.dumps(session, indent=2), encoding="utf-8")
    store.commit(PITCREW, sid)


def list_sessions() -> list[dict[str, Any]]:
    """Session summaries for the history sidebar, newest-first (by store mtime).
    Archived sessions are hidden (soft-delete) — the store has no hard-delete
    primitive, and a session backs a launched race we shouldn't orphan."""
    store = get_store()
    out: list[dict[str, Any]] = []
    for key in store.list_keys(PITCREW):
        s = _load(key)
        if not s or s.get("archived"):
            continue
        out.append({
            "sessionId": s.get("sessionId", key),
            "title": s.get("title") or "Fine-tuning session",
            "phase": s.get("phase", GREET),
            "raceId": s.get("raceId", ""),
            "createdAt": s.get("createdAt", ""),
            "mtime": store.key_mtime(PITCREW, key) or 0.0,
        })
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def archive_session(session_id: str, archived: bool = True) -> bool:
    """Hide (or restore) a session from the sidebar. Soft state in the session doc —
    never deletes the race a finished session launched. Returns False if unknown."""
    session = _load(session_id)
    if session is None:
        return False
    session["archived"] = archived
    _save(session)
    return True


# =============================================================================
# Message helpers
# =============================================================================
def _msg(text: str, **extra: Any) -> dict[str, Any]:
    """Build an assistant message dict (structured UI payload in `extra`). Split out
    from _say so a caller can build a message to REPLACE one in place, not just append."""
    msg = {"role": "assistant", "text": text}
    msg.update(extra)
    return msg


def _say(session: dict[str, Any], text: str, **extra: Any) -> None:
    """Append an assistant message to the transcript. `extra` carries structured UI
    payload (e.g. a plan, an estimate, a dataset list) the frontend renders inline."""
    session["messages"].append(_msg(text, **extra))


def _user_said(session: dict[str, Any], text: str, *, edit_kind: str = "") -> None:
    """Append a user message. `edit_kind` (e.g. "goal", "correction") marks a
    FREE-TEXT message the user can later edit/rewind; button-echoes leave it empty
    so they aren't editable (rewinding a "Effort: medium" echo is meaningless)."""
    if text:
        msg = {"role": "user", "text": text}
        if edit_kind:
            msg["editable"] = True
            msg["editKind"] = edit_kind
        session["messages"].append(msg)


# =============================================================================
# LLM narration (Bedrock Converse) — NARRATION ONLY, never load-bearing.
# =============================================================================
def _narrate(prompt: str, fallback: str) -> str:
    """One Bedrock Converse call (role 'pitcrew') to phrase something warmly + in
    plain language. Returns `fallback` verbatim on ANY error — the wizard must work
    with the LLM down. The model NEVER returns config; it only rewrites prose."""
    try:
        from .advisor import _session  # reuse the same boto session helper
        from .agent_models import resolve_model_id
        from .aws_config import load_aws_config

        cfg = load_aws_config()
        _, boto_sess = _session(cfg)
        client = boto_sess.client("bedrock-runtime", region_name=cfg.region)
        resp = client.converse(
            modelId=resolve_model_id("pitcrew"),
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 600, "temperature": 0.3},
        )
        out_msg = resp.get("output", {}).get("message", {})
        text = "".join(b.get("text", "") for b in out_msg.get("content", [])).strip()
        return text or fallback
    except Exception as e:  # noqa: BLE001 — narration is best-effort
        log_event("pitcrew.narrate.fallback", level="INFO", error=str(e))
        return fallback


# =============================================================================
# Public API — session lifecycle
# =============================================================================
def start_session(session_id: str, stamp: str) -> dict[str, Any]:
    """Create + persist a new guided session and emit the greeting. `session_id` and
    `stamp` are caller-supplied (no RNG/time in lib code)."""
    session = _new_session(session_id, stamp)
    _say(session,
         "Hi! I'm your fine-tuning guide. Tell me in plain words what you'd like a "
         "model to do — for example, \"sort support emails into categories\" or "
         "\"answer questions about our product docs\" — and I'll handle the technical "
         "parts and propose a plan for you to approve.",
         # Flag so the UI renders the goal text box. Without this the greeting had no
         # input and the conversation couldn't start.
         collectGoal=True)
    session["phase"] = COLLECT_GOAL
    _save(session)
    log_event("pitcrew.session.start", sessionId=session_id)
    return session


def get_session(session_id: str, stamp: str = "") -> dict[str, Any] | None:
    """Load a session, reconciling its race if it has launched one (so the in-thread
    finish update reflects live status). `stamp` timestamps a DONE transition."""
    session = _load(session_id)
    if session is None:
        return None
    if session.get("phase") in (LAUNCHED, DONE) and session.get("raceId"):
        _refresh_race(session, stamp or session.get("updatedAt", ""))
    return session


def rename_session(session_id: str, title: str) -> dict[str, Any] | None:
    """Set a user-chosen session title (sidebar label). Trimmed + capped; a blank
    title is ignored (keeps the current one). Returns the updated session, or None
    if it doesn't exist. Sets `titleManual` so the goal step won't overwrite it."""
    session = _load(session_id)
    if session is None:
        return None
    clean = (title or "").strip()[:80]
    if clean:
        session["title"] = clean
        session["titleManual"] = True
        _save(session)
    return session


# --- fields cleared when the conversation is rewound past a given phase ---------
# Ordered by the phase in which each is SET, so rewinding to an earlier point wipes
# exactly the state produced after it. The created dataset (splitId/datasetName) is
# UNLINKED here but NEVER deleted from disk — it may back a launched race and is
# reusable from the user's library (decision: keep-and-unlink).
def _reset_downstream(session: dict[str, Any], from_kind: str) -> None:
    """Clear session state produced AFTER the edited message. `from_kind` is the
    editKind of the message being changed ("goal" or "correction")."""
    # Editing the GOAL invalidates everything downstream (data choice included);
    # editing the CORRECTION (which happens at confirm, after data) keeps the data
    # but re-opens the task confirmation.
    if from_kind == "goal":
        session["splitId"] = ""          # unlink (dataset stays on disk)
        session["datasetName"] = ""
        session["profile"] = None
        session["shape"] = ""
    # Common to both: anything from confirm/effort/plan onward is stale.
    session["objective"] = ""
    session["rankMetric"] = ""
    session["effort"] = ""
    session["plan"] = None
    session["planSpecs"] = []
    session["estimate"] = None


def edit_message(session_id: str, message_index: int, new_text: str,
                 stamp: str, *, expected_version: int | None = None) -> dict[str, Any] | None:
    """ChatGPT-style edit/rewind of an earlier FREE-TEXT user message.

    Truncates the transcript at `message_index`, resets the downstream state (the
    created dataset is UNLINKED but kept on disk — it may back a launched race and is
    reusable), and REPLAYS the edited turn so the conversation continues from there.

    Guardrails:
      * a LAUNCHED/DONE session is never editable — its race is live + billable, so
        rewinding its inputs would be incoherent (returns the session unchanged);
      * only messages tagged editable (goal / correction) can be edited;
      * an out-of-range index or a non-user/uneditable target is a no-op.
    Returns the updated session, or None if the session is unknown."""
    session = _load(session_id)
    if session is None:
        return None
    # Same optimistic-concurrency guard as advance(): editing is destructive
    # (truncates the transcript + resets downstream state), so a stale message_index
    # from a second tab must be rejected, not applied.
    if expected_version is not None and expected_version != session.get("version", 0):
        raise ValueError("This session changed in another tab — reload to continue.")
    if session.get("phase") in (LAUNCHED, DONE):
        # Don't touch a launched race's inputs. Surface a gentle message.
        _say(session, "This run has already launched, so I can't change earlier steps. "
                      "Start a new session to try something different.")
        session["version"] = session.get("version", 0) + 1
        session["updatedAt"] = stamp
        _save(session)
        return session
    msgs = session.get("messages", [])
    if not (0 <= message_index < len(msgs)):
        return session
    target = msgs[message_index]
    if target.get("role") != "user" or not target.get("editable"):
        return session
    kind = target.get("editKind", "")
    text = (new_text or "").strip()[:1000]
    if not text:
        return session

    # Truncate the transcript to BEFORE the edited message, reset downstream state,
    # then replay the edited turn through the normal handler so all the usual
    # validation/gating applies.
    session["messages"] = msgs[:message_index]
    _reset_downstream(session, kind)
    if kind == "goal":
        session["phase"] = COLLECT_GOAL
        _h_collect_goal(session, {"goal": text}, stamp)
    elif kind == "correction":
        # A correction lives at the confirm step; re-open confirm on the SAME dataset
        # (kept/linked) and replay the correction.
        session["phase"] = CONFIRM_TASK
        _h_confirm_task(session, {"correction": text}, stamp)
    else:
        # Unknown editable kind — just re-prompt the current phase defensively.
        _reprompt_current_phase(session)

    session["version"] = session.get("version", 0) + 1
    session["updatedAt"] = stamp
    _save(session)
    log_event("pitcrew.edit_message", sessionId=session_id, index=message_index, kind=kind)
    return session


# =============================================================================
# The state machine — advance(session_id, action, payload, stamp)
# =============================================================================
def advance(session_id: str, action: str, payload: dict[str, Any], stamp: str,
            *, expected_version: int | None = None) -> dict[str, Any]:
    """Advance the session one user turn. `action` names the transition; `payload`
    carries its data. Returns the updated session. Raises ValueError for an unknown
    session or a stale version (optimistic concurrency — a second tab can't clobber)."""
    session = _load(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    if expected_version is not None and expected_version != session.get("version", 0):
        raise ValueError("This session changed in another tab — reload to continue.")

    phase = session.get("phase", GREET)
    handler = _HANDLERS.get((phase, action)) or _HANDLERS.get((phase, "*"))
    if handler is None:
        # An action that doesn't apply to the current phase (e.g. a stale button from
        # a a race between the UI and the server, or an out-of-order click). Rather
        # than a SILENT no-op — where the user clicks and nothing visibly happens —
        # re-emit the current step's guidance so they always see what to do next.
        log_event("pitcrew.advance.noop", sessionId=session_id, phase=phase, action=action)
        _reprompt_current_phase(session)
    else:
        handler(session, payload or {}, stamp)

    session["version"] = session.get("version", 0) + 1
    session["updatedAt"] = stamp
    _save(session)
    return session


def _reprompt_current_phase(session: dict[str, Any]) -> None:
    """Re-emit the prompt/affordance for whatever phase the session is in, so an
    out-of-phase action never leaves the user staring at an unchanged screen."""
    phase = session.get("phase", GREET)
    if phase == COLLECT_GOAL:
        _say(session, "Just tell me in plain words what you'd like the model to do, and I'll "
                      "take it from there.", collectGoal=True)
    elif phase == AWAIT_DATA:
        _say(session, "Bring some examples to continue — upload a file, pick an existing dataset, "
                      "or import one from Hugging Face.", datasetsHint=True)
    elif phase == CONFIRM_TASK:
        _say(session, "Does my read of your task look right? Confirm to continue, or type a "
                      "correction.", confirmTask=True)
    elif phase == CHOOSE_EFFORT:
        _ask_effort(session)
    elif phase == REVIEW_PLAN:
        _present_plan(session, intro=False)
    elif phase in (LAUNCHED, DONE):
        _say(session, "Your race has already launched — follow it on the Races page, or start a "
                      "new session to train something else.")
    else:
        _say(session, "Let's keep going — use the options above to continue.")


# --- handlers (each mutates session in place) --------------------------------
def _h_collect_goal(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    goal = str(payload.get("goal", "")).strip()[:1000]
    if not goal:
        _say(session, "No problem — just describe the task in your own words whenever you're ready.")
        return
    _user_said(session, goal, edit_kind="goal")
    session["goal"] = goal
    # Use the goal as the sidebar title — UNLESS the user already named the session
    # themselves (titleManual), in which case their choice wins.
    if not session.get("titleManual"):
        session["title"] = goal[:60]
    _say(session,
         "Got it. Now I need some examples of this task — a dataset. You can:\n"
         "• Upload a file (a .jsonl of examples), or\n"
         "• Pick one of your existing datasets, or\n"
         "• Bring one from Hugging Face.\n\n"
         "Each example should show the input and the ideal output. Even a few hundred "
         "good examples go a long way.",
         datasetsHint=True)
    session["phase"] = AWAIT_DATA


def _h_use_dataset(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    """User picked/uploaded a dataset that's already a persisted split (split_id)."""
    split_id = str(payload.get("splitId", "")).strip()
    if not split_id:
        _say(session, "I didn't catch which dataset — pick one or upload a file to continue.")
        return
    from .storage import split_dir, split_meta

    if split_dir(split_id) is None:
        _say(session, "I couldn't find that dataset. Try another, or upload a file.")
        return
    session["splitId"] = split_id
    meta = split_meta(split_id) or {}
    session["datasetName"] = meta.get("name") or split_id
    _user_said(session, f"Use dataset: {session['datasetName']}")
    _profile_and_confirm(session, stamp)


def _h_choose_effort(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    effort = str(payload.get("effort", "")).strip().lower()
    # Accept the legacy easy/medium/huge aliases too (old clients / cached payloads):
    # race_planner.EFFORT_JOBS maps them to the same ceilings.
    _ALIAS = {"easy": "quick", "medium": "balanced", "huge": "thorough"}
    effort = _ALIAS.get(effort, effort)
    if effort not in EFFORT_JOBS_LABEL:
        _say(session, "Pick how thorough to be — Quick, Balanced, or Thorough — and I'll build the plan.")
        return
    session["effort"] = effort
    _user_said(session, f"Effort: {effort}")
    _build_plan(session, stamp)


def _h_confirm_task(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    """User confirms (or corrects) the detected task, then we ask for effort."""
    if payload.get("correction"):
        # The user disagreed with the detected shape. v1 keeps this simple: explain
        # they can use the full Fine-tune page for an unusual setup, but proceed with
        # the detected objective (the planner only supports SFT/DPO/KTO anyway).
        _user_said(session, str(payload.get("correction"))[:300], edit_kind="correction")
        _say(session,
             "Thanks — I'll go with what the data looks like for now. If that's not right, "
             "the full Fine-tune page lets you choose the setup manually.")
    _ask_effort(session)


def _h_approve(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    """The ONLY billable transition — launch the approved race.

    IDEMPOTENT: if this session already launched a race, do NOT launch again. This is
    the guard against the duplicate-race footgun where a slow launch (many models = many
    SageMaker CreateTrainingJob calls) blew the 29s API-Gateway timeout, returned a 500
    to the browser BEFORE the session's raceId was persisted, and the user's retry
    launched a second identical race. Now: (a) a set raceId short-circuits to the
    launched view; (b) the launch runs OFF the request path (worker Lambda) so it can't
    time out; (c) the race_id is derived from a session-STABLE stamp so even a
    concurrent double-approve resolves to the same deterministic race_id."""
    if session.get("raceId"):
        # Already launched — re-affirm the launched state instead of double-launching.
        _say(session,
             "This race has already been launched — you can follow it here or on the "
             "Races page. Start a new session to train something else.",
             launched=True, raceId=session["raceId"])
        session["phase"] = LAUNCHED
        return
    notify_email = str(payload.get("notifyEmail", "")).strip().lower()
    if notify_email and _EMAIL_RE.match(notify_email):
        session["notifyEmail"] = notify_email
    _user_said(session, "Approved — launch the race.")
    # Pin a session-stable launch stamp the FIRST time we approve, so race_id
    # (race-{split}-{stamp}) is identical across any retry → the same race, never a dup.
    if not session.get("launchStamp"):
        session["launchStamp"] = stamp
    _launch(session, session["launchStamp"])


def _h_edit_effort(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    """User wants a different size of race before approving — go back to effort."""
    _ask_effort(session)


def _spec_entry_key(s: dict[str, Any]) -> str:
    """Stable per-arm key for a stored spec — MUST match PlannedModel.entry_key so
    the UI (which sends entryKeys) targets the exact arm."""
    return "::".join([s.get("modelId", ""), s.get("method", "lora"),
                      s.get("variant", "lora"), s.get("prefLoss", "sigmoid")])


def _h_edit_models(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    """User edited the model set on the review screen:
      {remove:[entryKey,...], add:[modelId,...]}.
    REMOVE targets a specific ARM by its stable entryKey (so removing the DoRA arm of
    a model doesn't also drop that model's plain-LoRA arm — they share a modelId).
    ADD introduces a fresh plain-LoRA arm for a modelId (deduped by entryKey). The
    plan is rebuilt + re-validated + re-estimated. Stays in REVIEW_PLAN."""
    remove = set(payload.get("remove", []) or [])
    add = [str(m) for m in (payload.get("add", []) or [])]
    specs = [s for s in session.get("planSpecs", []) if _spec_entry_key(s) not in remove]
    have = {_spec_entry_key(s) for s in specs}
    for mid in add:
        new = {"modelId": mid, "method": "lora", "variant": "lora",
               "prefLoss": "sigmoid", "role": "you added this model"}
        if mid and _spec_entry_key(new) not in have:
            specs.append(new)
            have.add(_spec_entry_key(new))
    session["planSpecs"] = specs
    # Re-present the (rebuilt, re-validated) plan IN PLACE (no new bubble — see
    # _present_plan), so the review card refines live instead of "reloading". If the
    # user emptied it, the review card itself blocks approval (handled client-side +
    # the launch guard), so we just rebuild with whatever remains.
    if not session["planSpecs"]:
        # Nothing left — keep the last review card but flag it empty via a nudge.
        _say(session, "There are no models left in the plan — add at least one to continue.")
        return
    _present_plan(session, intro=False)


def _h_cancel(session: dict[str, Any], payload: dict[str, Any], stamp: str) -> None:
    _user_said(session, "Cancel.")
    _say(session, "No problem — nothing was launched. Let's reconsider how thorough to be.")
    # Re-render the effort chooser as the LAST message so the user has something to
    # act on. Without this the phase moves to choose_effort but the last bubble is
    # plain text (no chooser) and the prompt bar is disabled here → a dead end.
    _ask_effort(session)


# (phase, action) → handler. "*" matches any action in that phase.
_HANDLERS = {
    (COLLECT_GOAL, "goal"): _h_collect_goal,
    (AWAIT_DATA, "use_dataset"): _h_use_dataset,
    (CONFIRM_TASK, "confirm"): _h_confirm_task,
    (CHOOSE_EFFORT, "effort"): _h_choose_effort,
    (REVIEW_PLAN, "approve"): _h_approve,
    (REVIEW_PLAN, "edit_effort"): _h_edit_effort,
    (REVIEW_PLAN, "edit_models"): _h_edit_models,
    (REVIEW_PLAN, "cancel"): _h_cancel,
}


# =============================================================================
# Internal steps (server-driven phases)
# =============================================================================
def _profile_and_confirm(session: dict[str, Any], stamp: str) -> None:
    """PROFILING → CONFIRM_TASK: profile the chosen split, decide the objective, and
    explain in plain language what we found + whether we can handle it."""
    from .profiler import profile_dataset
    from .race_planner import SUPPORTED_OBJECTIVES

    session["phase"] = PROFILING
    try:
        prof = profile_dataset(session["splitId"])
    except Exception as e:  # noqa: BLE001
        log_event("pitcrew.profile.error", level="WARNING",
                  sessionId=session["sessionId"], error=str(e))
        _say(session, "I had trouble reading that dataset. Make sure it's a .jsonl of "
                      "examples, then try again or pick a different one.")
        session["phase"] = AWAIT_DATA
        return

    session["profile"] = prof
    objective = (prof.get("objective") or {}).get("objective", "sft")
    session["shape"] = prof.get("shape", "sft")

    if objective not in SUPPORTED_OBJECTIVES:
        _say(session,
             "This dataset looks like a reward-based (reinforcement-learning) task. "
             "I don't handle those in the guided flow yet — the full Fine-tune page "
             "can set up the reward and run it. Want to try a different dataset?")
        session["phase"] = AWAIT_DATA
        return

    # Too-few-usable-examples gate, UP FRONT (not at the billable approve click):
    # a dataset that's empty after filtering — or below a sane learning floor —
    # would build an approvable, cost-estimated plan and only fail in start_race's
    # quality gate. Catch it here in plain language, at the step the user can fix.
    usable = _row_count(prof, objective)
    if usable < MIN_USABLE_ROWS:
        _say(session,
             f"This dataset only has {usable} usable example{'s' if usable != 1 else ''}, "
             f"but fine-tuning needs at least about {MIN_USABLE_ROWS} to learn anything "
             "reliable. Add more examples and re-upload, or pick a different dataset.",
             datasetsHint=True)
        session["phase"] = AWAIT_DATA
        return

    # Plain-language recap of the task, narrated (deterministic fallback below).
    plain = _OBJECTIVE_PLAIN.get(objective, "learn from your examples")
    task = (prof.get("recommendation") or {}).get("detectedTask", "")
    rows = _row_count(prof, objective)
    fallback = (
        f"I looked at \"{session['datasetName']}\". It has about {rows} examples, and it "
        f"looks like a task where the model will {plain}. Does that match what you want? "
        "If so, tell me how thorough you'd like the search to be."
    )
    narration = _narrate(
        "You are a friendly, jargon-free fine-tuning guide talking to a non-technical "
        "user (no ML background). In 2-3 short sentences, confirm what their dataset is "
        "for and ask them to confirm. Do NOT use jargon (never say SFT, DPO, LoRA, epochs). "
        f"Facts: the user's goal is \"{session['goal']}\". The dataset is named "
        f"\"{session['datasetName']}\" with about {rows} examples. In plain terms the model "
        f"will {plain}. End by asking them to confirm it looks right.",
        fallback)
    _say(session, narration, confirmTask=True,
         taskSummary={"objective": objective, "detectedTask": task, "rows": rows,
                      "plain": plain})
    session["phase"] = CONFIRM_TASK


def _ask_effort(session: dict[str, Any]) -> None:
    _say(session,
         "How thorough should I be? More models means a better chance of a great "
         "result, but it costs more and takes longer. These are ceilings — I'll only "
         "race as many as actually add something for your data, and tell you if fewer "
         "is enough:\n"
         "• Quick — a fast look (up to 4 models)\n"
         "• Balanced — a solid comparison (up to 8 models)\n"
         "• Thorough — a broad search (up to 16 models)",
         chooseEffort=True,
         efforts=[{"key": k, "label": v} for k, v in EFFORT_JOBS_LABEL.items()])
    session["phase"] = CHOOSE_EFFORT


def _build_plan(session: dict[str, Any], stamp: str) -> None:
    """BUILDING_PLAN → REVIEW_PLAN: turn the profile + effort into an exact race plan,
    then present it for approval (with the editable model set). Never auto-launches."""
    from .race_planner import plan_race, specs_from_plan
    from .secrets import get_hf_token

    session["phase"] = BUILDING_PLAN
    prof = session.get("profile") or {}
    plan = plan_race(prof, session["effort"], hf_token_ok=bool(get_hf_token()))

    if not plan.supported or not plan.planned:
        _say(session, plan.reason or "I couldn't build a plan for this dataset.",
             planUnsupported=True)
        session["phase"] = CHOOSE_EFFORT
        return

    # Persist the editable specs (the source of truth the review screen edits + the
    # launch rebuilds from) and the fixed objective/rankMetric. Also persist the effort
    # CEILING + detected task so the review screen can show "raced M of up-to-N" and the
    # eval-strategy label without recomputing.
    session["planSpecs"] = specs_from_plan(plan)
    session["objective"] = plan.objective
    session["rankMetric"] = plan.rank_metric
    session["ceiling"] = plan.job_budget
    session["detectedTask"] = plan.detected_task
    session["initialCount"] = len(plan.planned)  # how many the planner filled (pre-edit)
    _present_plan(session, intro=True)
    session["phase"] = REVIEW_PLAN


def _present_plan(session: dict[str, Any], *, intro: bool, note: str = "") -> None:
    """Build gate-valid models from the stored specs, estimate cost/time, and emit the
    review message (with the editable model list + the pool of models the user can
    add). Shared by the initial plan + every model edit so they stay consistent.
    `note` (optional) is prefixed to the card text — e.g. a launch-failure reason —
    so the actionable card carries the message instead of a separate dead-end bubble.
    When a note is given the card is always APPENDED (so it's the last, interactive
    message), not updated in place."""
    from .cost_estimate import estimate_race
    from .race_planner import build_models_from_specs, eligible_models
    from .secrets import get_hf_token

    prof = session.get("profile") or {}
    hf_token_ok = bool(get_hf_token())
    planned, skipped = build_models_from_specs(prof, session.get("planSpecs", []),
                                               hf_token_ok=hf_token_ok)
    # Keep the stored specs in sync with what actually built (drops any skipped).
    session["planSpecs"] = [{
        "modelId": p.race_model.model_id, "method": p.method, "variant": p.variant,
        "prefLoss": p.race_model.hp.pref_loss, "role": p.role,
    } for p in planned]

    objective = session.get("objective", "sft")
    eval_rows = _eval_rows(prof)
    train_rows = _row_count(prof, objective)
    entries = [{
        "instanceType": p.race_model.instance_type, "paramsB": p.params_b,
        "epochs": float(p.race_model.hp.num_train_epochs), "method": p.method,
        "baseEval": p.race_model.hp.stage != "rlaif",
    } for p in planned]
    estimate = estimate_race(entries, train_rows=train_rows, eval_rows=eval_rows,
                             use_spot=False, seq_tokens=_seq_tokens(prof))
    session["estimate"] = estimate

    ceiling = int(session.get("ceiling", len(planned)) or len(planned))
    plan_view = {
        "supported": True, "objective": objective, "engine": "llama_factory",
        "effort": session.get("effort", ""), "jobBudget": len(planned),
        "rankMetric": session.get("rankMetric", "token_f1"),
        "detectedTask": session.get("detectedTask", ""),
        "models": [p.to_view() for p in planned], "gatesApplied": [], "reason": "",
        # "up to N" transparency: how many arms are in the plan vs the ceiling the user
        # picked, and whether the planner deliberately stopped short (no more useful arms
        # for this dataset). The UI shows "raced M of up-to-N — more wouldn't add signal".
        "meaningfulCount": len(planned),
        "ceiling": ceiling,
        "capped": len(planned) < ceiling,
    }
    session["plan"] = plan_view

    n = len(planned)
    ceiling = int(session.get("ceiling", n) or n)
    capped = n < ceiling
    lo, hi = estimate["totalUsd"]["lo"], estimate["totalUsd"]["hi"]
    wlo, whi = estimate["wallClockMin"]["lo"], estimate["wallClockMin"]["hi"]
    if intro:
        # Be honest when we filled fewer than the ceiling: it's a feature (no redundant
        # billable jobs), not a shortfall.
        cap_note = (f" You picked up to {ceiling}, but {n} is all that adds anything for "
                    "this dataset — more would just be near-duplicates, so I stopped there."
                    if capped else "")
        fallback = (
            f"Here's my plan: I'll train and compare {n} model"
            f"{'s' if n != 1 else ''} on your data, then pick the winner automatically.{cap_note} "
            f"Estimated cost is about ${lo}–${hi}, taking roughly {wlo}–{whi} minutes. "
            "Review the models below — you can remove any or add more, and when you're "
            "happy, approve. Nothing is charged until you approve."
        )
        text = _narrate(
            "You are a friendly, jargon-free fine-tuning guide. In 2-3 short sentences, "
            "tell a non-technical user you've prepared a plan to train and compare "
            f"{n} models on their data and automatically pick the best one. Mention they "
            "can remove or add models, the estimated cost is about "
            f"${lo} to ${hi} and it takes roughly {wlo} to {whi} minutes, and that nothing "
            "is charged until they approve. No jargon.", fallback)
    else:
        text = (f"Updated — {n} model{'s' if n != 1 else ''} now, about ${lo}–${hi} "
                f"and roughly {wlo}–{whi} minutes. Adjust further or approve when ready.")
    if note:
        text = f"{note}\n\n{text}"

    # The pool of models the user can ADD (eligible for this dataset), minus the
    # ones already in the plan.
    chosen_ids = {p.race_model.model_id for p in planned}
    pool = [m for m in eligible_models(prof, hf_token_ok=hf_token_ok)
            if m["modelId"] not in chosen_ids]

    review_msg = {
        "role": "assistant", "text": text, "reviewPlan": True, "plan": plan_view,
        "estimate": estimate, "addPool": pool,
        "notifyEmailPrefill": session.get("notifyEmail", ""),
    }
    if intro or note:
        # intro → first plan; note → launch-failure/other message must be the LAST
        # (interactive) bubble, so append rather than update-in-place.
        session["messages"].append(review_msg)
    else:
        # UPDATE the existing review bubble IN PLACE rather than appending a new one,
        # so editing the model set refines the same card instead of spawning a fresh
        # message each time (which read as a jarring "reload" + hid the controls).
        for i in range(len(session["messages"]) - 1, -1, -1):
            if session["messages"][i].get("reviewPlan"):
                session["messages"][i] = review_msg
                return
        session["messages"].append(review_msg)


def _launch(session: dict[str, Any], stamp: str) -> None:
    """Approve → launch. Runs the launch OFF the 29s request path when a worker Lambda
    is configured (the deployed stack), else inline (local dev + tests).

    WHY: start_race makes ~2 synchronous SageMaker CreateTrainingJob calls PER model
    (train + base-eval). An 8-16 model guided race is ~16-32 calls ≈ 25-40s, which blew
    API Gateway's 29s timeout → 500 to the browser AFTER jobs were already created →
    the user's retry launched a duplicate race. Dispatching to the worker (15-min
    budget) makes approve return instantly; the frontend's launched-phase poll then
    surfaces the real entries once the worker finishes. dispatch_worker returns False
    with no worker configured (local dev), so we fall back to the inline path — keeping
    behavior + the simulation/tests byte-identical there."""
    from .dispatch import dispatch_worker

    if dispatch_worker({"task": "pitcrew_launch",
                        "sessionId": session["sessionId"], "stamp": stamp}):
        # Off the request path. Show an immediate "launching" state; the poll will
        # replace it with the real race once the worker calls _run_launch.
        session["phase"] = LAUNCHED
        session["launching"] = True
        _say(session,
             "🏁 Starting your race now — I'm spinning up the training jobs. This takes "
             "a moment; the models will appear here and on the Races page shortly.",
             launched=True, raceId="")
        return
    # No worker (local dev) → run inline, exactly as before.
    _run_launch(session, stamp)


def _run_launch(session: dict[str, Any], stamp: str) -> None:
    """The actual launch body — rebuilds launchable models from the stored (possibly
    EDITED) specs (never client hyperparameters, so gates are re-applied), starts the
    race, and moves the session to LAUNCHED. Called INLINE for local dev, or by the
    worker Lambda (task=pitcrew_launch) on the deployed stack. Safe to re-run: it
    reloads the session so a stale in-memory copy can't clobber a concurrent update."""
    from .catalog import DecodingParams, reasoning_eval_floor
    from .limits import LimitExceeded
    from .race import start_race
    from .race_planner import build_models_from_specs
    from .secrets import get_hf_token

    prof = session.get("profile") or {}
    planned, _skipped = build_models_from_specs(prof, session.get("planSpecs", []),
                                                hf_token_ok=bool(get_hf_token()))
    if not planned:
        _say(session, "Something changed and I can't launch this plan. Let's rebuild it.")
        session["phase"] = CHOOSE_EFFORT
        return

    race_models = [p.race_model for p in planned]
    model_ids = [rm.model_id for rm in race_models]
    # Raise the eval token floor if EITHER a reasoning-family base is racing OR the DATA
    # carries a <think>/reasoning scaffold (needs_raised_eval_floor) — the two are now
    # decoupled, so an instruct base fine-tuned on scaffolded data still gets enough
    # tokens for its emitted <think> block to CLOSE (an unclosed block zeros the answer).
    from .catalog import REASONING_EVAL_MAX_NEW_TOKENS
    from .race_planner import needs_raised_eval_floor

    eff_max_new = reasoning_eval_floor(model_ids, 256)
    if (session.get("profile") and eff_max_new < REASONING_EVAL_MAX_NEW_TOKENS
            and needs_raised_eval_floor(prof, session.get("objective", "sft"))):
        eff_max_new = REASONING_EVAL_MAX_NEW_TOKENS
    decoding = DecodingParams(max_new_tokens=eff_max_new, temperature=0.0)
    notify = [session["notifyEmail"]] if session.get("notifyEmail") else []
    race_name = _race_name(session)

    try:
        race = start_race(session["splitId"], race_models, decoding, stamp,
                          name=race_name, notify_emails=notify)
    except LimitExceeded as e:
        session["phase"] = REVIEW_PLAN
        # Re-render the plan card (carrying the busy note) so Approve/Edit/Cancel come
        # back — otherwise the last message is bare text and the user is stranded (no
        # buttons, prompt bar disabled at review_plan). The cap is a shipped limit, so
        # this path fires in prod.
        _present_plan(session, intro=False,
                      note=("The platform is busy right now, so I couldn't start this race "
                            f"({e}). Try again, or choose a smaller effort level."))
        return
    except Exception as e:  # noqa: BLE001
        log_event("pitcrew.launch.error", level="ERROR",
                  sessionId=session["sessionId"], error=str(e))
        session["phase"] = REVIEW_PLAN
        _present_plan(session, intro=False,
                      note=f"I hit a problem launching the race: {e}. Let's try again.")
        return

    session["raceId"] = race.race_id
    session.pop("launching", None)  # clear the provisional "starting…" state
    # Kick off SES verification for the notify recipient (best-effort).
    if notify:
        try:
            from .notify import ensure_notify_recipients_verified
            ensure_notify_recipients_verified(notify)
        except Exception:  # noqa: BLE001
            pass
    log_event("pitcrew.launch", sessionId=session["sessionId"], raceId=race.race_id,
              models=model_ids)
    email_note = (f" I'll email {session['notifyEmail']} when it's done."
                  if session.get("notifyEmail") else "")
    launched_msg = _msg(
        f"🏁 Your race is underway — {len(race_models)} models are training and will "
        f"be compared automatically.{email_note} You can follow progress here or on the "
        "Races page. This usually takes a little while.",
        launched=True, raceId=race.race_id)
    # Replace the provisional "starting your race now" bubble (worker path) in place so
    # the thread shows ONE launched message; otherwise just append (inline path).
    for i in range(len(session["messages"]) - 1, -1, -1):
        mi = session["messages"][i]
        if mi.get("launched") and not mi.get("raceId"):
            session["messages"][i] = launched_msg
            break
    else:
        session["messages"].append(launched_msg)
    session["phase"] = LAUNCHED


def run_pitcrew_launch(session_id: str, stamp: str) -> None:
    """Worker-Lambda entry for task=pitcrew_launch: load the session, run the launch
    body OFF the request path, and persist. Idempotent — if the session already has a
    raceId (a retry raced us), do nothing. The worker binds the tenant before calling
    this (worker_handler), so _load/_save hit the right user's partition."""
    session = _load(session_id)
    if session is None:
        log_event("pitcrew.launch.worker.missing", level="ERROR", sessionId=session_id)
        return
    if session.get("raceId"):
        return  # already launched (idempotent) — never double-launch
    _run_launch(session, stamp)
    session["version"] = session.get("version", 0) + 1
    session["updatedAt"] = stamp
    _save(session)


def _refresh_race(session: dict[str, Any], stamp: str) -> None:
    """Reconcile the launched race + reflect terminal status in the thread (once).
    `stamp` timestamps the DONE transition (caller-supplied; no time in lib code)."""
    from .race import rank_entries, reconcile_race

    race = reconcile_race(session["raceId"])
    if race is None:
        return
    from .race import TERMINAL

    if session.get("phase") == LAUNCHED and race.entries and all(
            e.state in TERMINAL for e in race.entries):
        ranked = rank_entries(race)
        winner = next((r for r in ranked if r.get("isWinner")), None)
        if winner:
            _say(session,
                 f"✅ All done! The winner is {winner.get('model_display', winner.get('model_id'))}. "
                 "Open the run to see the full comparison, then deploy or export the winner.",
                 finished=True, raceId=session["raceId"], winner=winner.get("model_id"))
        else:
            _say(session,
                 "Your race finished, but none of the models produced a usable result. "
                 "Open the run for details — you may want to check the dataset.",
                 finished=True, raceId=session["raceId"])
        session["phase"] = DONE
        # Bump version + stamp updatedAt so any client still holding the pre-DONE
        # version becomes stale — its next advance()/edit() correctly 409s instead
        # of silently clobbering the reconciled finish state (last-writer-wins on the
        # unlocked per-tenant S3 doc otherwise lets a stale tab overwrite it).
        session["version"] = session.get("version", 0) + 1
        session["updatedAt"] = stamp
        _save(session)


# Filler openers users type ("I want to…", "help me…") that make a lousy race name.
_GOAL_FILLER_RE = re.compile(
    r"^(i\s+(want|would like|need|wish)\s+to\s+|i'?d\s+like\s+to\s+|please\s+|help\s+me\s+(to\s+)?|"
    r"can\s+you\s+|could\s+you\s+|let'?s\s+|build\s+(a\s+)?model\s+to\s+|train\s+(a\s+)?model\s+to\s+|"
    r"make\s+(a\s+)?model\s+(that\s+|to\s+)?)+",
    re.IGNORECASE,
)


def _race_name(session: dict[str, Any]) -> str:
    """A clean, human race name for a Guided launch, derived from the user's GOAL. The
    old code sliced the raw goal to 60 chars, producing mid-word junk like "i want to
    classify the new". This strips filler openers ("i want to", "please", …), trims to a
    WORD boundary, and Capitalizes → e.g. 'Classify support tickets'.

    The dataset is deliberately NOT appended — the Races table already has a dataset
    column, so repeating it in the title is redundant. Falls back to a stamped generic
    only when there's no usable goal text."""
    goal = (session.get("goal") or "").strip()
    core = _GOAL_FILLER_RE.sub("", goal).strip().rstrip(".!?,")
    # Trim to ~60 chars on a word boundary (no mid-word cut).
    if len(core) > 60:
        core = core[:60].rsplit(" ", 1)[0].rstrip(".!?,") + "…"
    if core:
        return (core[0].upper() + core[1:])[:80]
    return f"Guided fine-tune · {_friendly_stamp(session.get('createdAt', ''))}"[:80]


# --- plain-language helpers --------------------------------------------------
_OBJECTIVE_PLAIN = {
    "sft": "learn to produce the right answer from your examples",
    "dpo": "learn to prefer the better answer of each pair you provided",
    "kto": "learn from answers you've marked as good or bad",
}


def _row_count(prof: dict[str, Any], objective: str) -> int:
    """TRUE training-example count for the plain-language message. For SFT the
    profiler samples at most MAX_PROFILE_ROWS (5000) for its analysis, so `parsed`
    would UNDER-report a large file as "~5000". `rows` is the uncapped total line
    count, so prefer it — a 109MB / 80k-row upload reports 80k, not 5000. (The
    preference/KTO profilers already count the whole file, so those are exact.)"""
    if objective == "dpo":
        return int((prof.get("preference") or {}).get("pairs", 0) or 0)
    if objective == "kto":
        return int((prof.get("kto") or {}).get("rows", 0) or 0)
    train = prof.get("train") or {}
    return int(train.get("rows") or train.get("parsed", 0) or 0)


def _eval_rows(prof: dict[str, Any]) -> int:
    ev = prof.get("eval") or {}
    return int(ev.get("parsed") or ev.get("rows", 0) or 0) or 50  # sane default for the estimate


def _seq_tokens(prof: dict[str, Any]) -> float | None:
    """A p95 tokens-per-example signal for the cost estimate, so long-context data
    isn't badly under-quoted. The profiler reports p95 GOLD word-length; approximate
    tokens ≈ words × 1.3 (the same factor profiler uses) and roughly double to
    account for the prompt as well as the answer. Prefer the eval file's answer
    length (always messages-shaped); fall back to train, then preference `chosen`.
    None when unavailable → the cost model uses factor 1.0."""
    for key in ("eval", "train"):
        wl = ((prof.get(key) or {}).get("goldWordLen") or {}).get("p95")
        if wl:
            return float(wl) * 1.3 * 2.0
    ch = ((prof.get("preference") or {}).get("chosenWordLen") or {}).get("p95")
    if ch:
        return float(ch) * 1.3 * 2.0
    return None
