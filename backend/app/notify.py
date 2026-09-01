# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Email notifications: tell a user (via SES) when their fine-tuning run finishes.

Two operations, both deliberately BEST-EFFORT — a notification must NEVER break a
launch or the reconcile loop:

  * ensure_notify_recipients_verified(emails) — called at launch. SES is in the
    sandbox for this account, so it can only deliver to VERIFIED addresses. For each
    recipient that isn't already verified, kick off SES email-identity verification
    (AWS emails them a confirmation link). Returns a per-address status list the UI
    uses to tell the user "click the link AWS just sent you".

  * send_race_complete_email(race, ...) — called from the reconcile loop the first
    time a race is fully terminal. Builds an HTML+text summary (per-model final
    status + the winning model & score) and sends it from the configured sender.

The email BODY building (build_race_complete_email) is a PURE function with no AWS,
so it's unit-testable; the SES calls are isolated in thin wrappers that swallow
errors and log, so a notify failure degrades to "no email", never an exception in
the caller.

Config (env):
  SLM_NOTIFY_FROM_EMAIL  — verified SES sender address (required to send; if unset,
                           sending is skipped with a log — the rest still works).
  SLM_NOTIFY_FROM_NAME   — friendly From display name (default "SLM Fine-tuning Platform").
  SLM_APP_URL            — app base URL for a run-detail deep link (optional).
"""
from __future__ import annotations

import os
from typing import Any

from .obs import log_event


def _from_email() -> str:
    return os.environ.get("SLM_NOTIFY_FROM_EMAIL", "").strip()


def _from_name() -> str:
    return os.environ.get("SLM_NOTIFY_FROM_NAME", "SLM Fine-tuning Platform").strip()


def _from_header() -> str:
    """RFC-5322 From with a friendly display name over the verified address:
    `SLM Fine-tuning Platform <addr@domain>`. Falls back to the bare address."""
    addr = _from_email()
    name = _from_name()
    return f"{name} <{addr}>" if name and addr else addr


def _app_url() -> str:
    return os.environ.get("SLM_APP_URL", "").strip().rstrip("/")


def _ses_client():
    """Build an SESv2 client the same way the rest of the app builds AWS clients
    (profile/region from aws_config). Returns None if it can't be built."""
    try:
        from .aws_clients import get_session
        from .aws_config import load_aws_config

        cfg = load_aws_config()
        return get_session(
            profile_name=cfg.profile or None, region_name=cfg.region
        ).client("sesv2")
    except Exception as e:  # noqa: BLE001 — notify is best-effort
        log_event("notify.ses_client_failed", level="WARNING", error=str(e))
        return None


# --- recipient verification (SES sandbox) ----------------------------------- #


def ensure_notify_recipients_verified(emails: list[str], *, _client: Any = None) -> list[dict]:
    """For each recipient, report whether it's SES-verified; for any that isn't,
    request verification (AWS sends a confirmation link). Returns a list of
    {email, status: "verified"|"pending"|"unknown"} the UI surfaces. Never raises.

    `_client` injects a stub for tests."""
    if not emails:
        return []
    client = _client or _ses_client()
    if client is None:
        # Can't reach SES — report unknown so the UI shows a soft message rather than
        # implying the address is good.
        return [{"email": e, "status": "unknown"} for e in emails]

    out: list[dict] = []
    for email in emails:
        status = "unknown"
        try:
            info = client.get_email_identity(EmailIdentity=email)
            status = "verified" if info.get("VerifiedForSendingStatus") else "pending"
        except Exception:  # noqa: BLE001 — most commonly NotFoundException (never verified)
            # Not an identity yet → create it, which sends the AWS verification email.
            try:
                client.create_email_identity(EmailIdentity=email)
                status = "pending"
                log_event("notify.verification_requested", email=email)
            except Exception as e:  # noqa: BLE001 — already-exists race / throttle / perms
                log_event("notify.verification_request_failed", level="WARNING",
                          email=email, error=str(e))
                status = "pending"
        out.append({"email": email, "status": status})
    return out


# --- email body (PURE — no AWS, unit-testable) ------------------------------ #


def _pct(score: Any) -> str:
    try:
        return f"{float(score) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


# Playful ML-nerd one-liners, themed to the outcome. Picked DETERMINISTICALLY from
# the race id (stable per-race, not random each render) so re-sends match. Kept
# tasteful: the "fail" set is gently encouraging, never mocking a failed run.
_QUOTES_WIN = (
    "The gradients have descended. The winner has ascended.",
    "Your GPUs earned their coffee break. ☕",
    "No overfitting was harmed in the making of this run.",
    "Loss went down, vibes went up.",
    "Backprop did its thing. You did your thing. Teamwork.",
    "Somewhere, a loss curve is finally getting some sleep.",
    "That's a wrap — the weights have spoken.",
    "Fine-tuned to perfection (or at least to a higher token_f1).",
)
_QUOTES_FAIL = (
    "Every failed run is just a data point for the next one. 💪",
    "The model didn't fail — it just found a way that doesn't work yet.",
    "Even GPT started with a few NaNs. Tweak and re-run.",
    "Plot twist: this is where the debugging montage begins.",
    "The loss didn't converge, but your persistence will.",
)


def _pick(seq: tuple, seed: str) -> Any:
    """Deterministic choice from `seq` keyed on `seed` (no RNG, stable across
    re-renders/re-sends)."""
    if not seq:
        return ""
    return seq[sum(ord(c) for c in (seed or "x")) % len(seq)]


def build_race_complete_email(race: Any, ranked: list[dict]) -> dict[str, str]:
    """Build the completion email from a (terminal) race + its ranked entries.
    PURE: returns {"subject", "html", "text"}. `ranked` is rank_entries(race)
    output (already best-first with isWinner set on the top scored entry).

    Bulletproof transactional-email construction (per Litmus + Amazon Shopbop/
    Goodreads guidance): full HTML doc, table-based 600px centered layout, inline
    CSS, emoji instead of hosted images, a dark-mode @media block, a hidden
    preheader, and an MSO/VML bulletproof button for Outlook. A plain-text part is
    always included for accessibility + non-HTML clients."""
    import html as _html

    def esc(s: Any) -> str:
        return _html.escape(str(s if s is not None else ""))

    name = (getattr(race, "name", "") or "").strip() or getattr(race, "race_id", "run")
    race_id = getattr(race, "race_id", "")
    entries = list(getattr(race, "entries", []) or [])
    total = len(entries)
    n_done = sum(1 for e in entries if getattr(e, "state", "") == "done")
    n_failed = total - n_done
    winner = next((r for r in ranked if r.get("isWinner")), None)
    all_failed = total > 0 and n_done == 0
    # The metric the winner was chosen by (a race is ranked by one metric — its
    # per-objective default). The live "Rank by" dropdown on the run page lets the
    # user re-rank by any other metric; we surface the metric here so the choice of
    # winner is transparent, and point at that dropdown.
    rank_metric = (winner or {}).get("rankMetric") or (ranked[0].get("rankMetric") if ranked else "") or "score"

    # Outcome-aware headline + quote (humor reads the room — no jokes on all-failed).
    if winner:
        headline = "Your models crossed the finish line! \U0001f3c1"
        quote = _pick(_QUOTES_WIN, race_id)
    elif n_done:
        headline = "Your run wrapped up \U0001f3c1"
        quote = _pick(_QUOTES_WIN, race_id)
    else:
        headline = "Your run finished — no winners this lap"
        quote = _pick(_QUOTES_FAIL, race_id)

    # --- subject (keeps the words "winner"/"failed" for scanability + tests) ---
    if winner:
        subject = f"\U0001f3c1 Run '{name}' finished — winner: {winner.get('model_display') or winner.get('model_id')}"
    elif n_done:
        subject = f"\U0001f3c1 Run '{name}' finished ({n_done}/{total} succeeded)"
    else:
        subject = f"\U0001f6a9 Run '{name}' finished — all {total} job(s) failed"

    link = ""
    app = _app_url()
    if app:
        link = f"{app}/#runs/{race_id}"

    rows = ranked if ranked else [
        {"model_display": getattr(e, "model_display", ""), "model_id": getattr(e, "model_id", ""),
         "state": getattr(e, "state", ""), "error": getattr(e, "error", None),
         "rankScore": None, "rankMetric": ""}
        for e in entries
    ]

    def _label(r: dict) -> str:
        return r.get("model_display") or r.get("model_id") or "model"

    # Medal/marker per row: 🥇🥈🥉 to the top-3 DONE entries (ranked order), 💥 to
    # failed, ▸ to any further done rows.
    def _markers() -> list[str]:
        out, done_seen = [], 0
        for r in rows:
            if r.get("state") == "done":
                done_seen += 1
                out.append({1: "\U0001f947", 2: "\U0001f948", 3: "\U0001f949"}.get(done_seen, "▸"))
            elif r.get("state") == "failed":
                out.append("\U0001f4a5")
            else:
                out.append("▸")
        return out

    markers = _markers()

    # --- plain-text part ---
    text_lines = [
        headline.encode("ascii", "ignore").decode().strip() or "Your fine-tuning run has finished.",
        "",
        f"  {n_done}/{total} job(s) succeeded" + (f", {n_failed} failed." if n_failed else "."),
        "",
    ]
    if winner:
        text_lines += [
            f"Best model: {_label(winner)} "
            f"({winner.get('rankMetric') or 'score'} = {_pct(winner.get('rankScore'))})",
            "",
        ]
    text_lines.append("Results:")
    for r in rows:
        st = r.get("state", "")
        mark = "* " if r.get("isWinner") else "  "
        line = f"{mark}{_label(r)} — {st}"
        if st == "done" and r.get("rankScore") is not None:
            line += f" ({r.get('rankMetric') or 'score'} {_pct(r.get('rankScore'))})"
        if st == "failed" and r.get("error"):
            line += f" — {str(r.get('error'))[:160]}"
        text_lines.append(line)
    if quote:
        text_lines += ["", f'"{quote}"']
    text_lines += ["", f"Ranked by {rank_metric}."
                   + (" Open the run to re-rank by a different metric." if link else "")]
    if link:
        text_lines += ["", f"View the full leaderboard: {link}"]
    text_lines += ["", "— SLM Fine-tuning Platform"]
    text = "\n".join(text_lines)

    # --- HTML part (table-based, inline CSS, dark-mode aware, Outlook-safe) ---
    accent = "#0969da"
    hero_bg = "#0b3d2e" if all_failed else "#0d2440"  # calmer green-ish on all-fail
    # Result table rows.
    trs = []
    for r, mk in zip(rows, markers):
        st = r.get("state", "")
        is_win = bool(r.get("isWinner"))
        result = (
            f'<span style="color:#1a7f37;font-weight:600">{esc(r.get("rankMetric") or "score")} '
            f'{_pct(r.get("rankScore"))}</span>'
            if st == "done" and r.get("rankScore") is not None else
            (f'<span style="color:#cf222e">{esc(str(r.get("error"))[:200])}</span>'
             if st == "failed" and r.get("error") else '<span style="color:#8c959f">—</span>')
        )
        status_color = "#1a7f37" if st == "done" else ("#cf222e" if st == "failed" else "#57606a")
        row_bg = "#f3fbf5" if is_win else "#ffffff"
        trs.append(
            f'<tr style="background:{row_bg}">'
            f'<td style="padding:10px 12px;border-bottom:1px solid #eaeef2;font-size:18px;width:34px">{mk}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #eaeef2;font-weight:{"700" if is_win else "500"};color:#1f2328">{esc(_label(r))}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #eaeef2;color:{status_color};text-transform:capitalize">{esc(st)}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #eaeef2;text-align:right">{result}</td>'
            f'</tr>'
        )
    winner_band = (
        f'<tr><td style="padding:0 28px 4px">'
        f'<div style="background:#f3fbf5;border:1px solid #b7e4c7;border-radius:8px;padding:14px 16px">'
        f'<span style="font-size:20px">\U0001f3c6</span> '
        f'<b style="color:#1f2328">Best model:</b> '
        f'<span style="color:#1f2328">{esc(_label(winner))}</span> '
        f'<span style="color:#57606a">({esc(winner.get("rankMetric") or "score")} = {_pct(winner.get("rankScore"))})</span>'
        f'</div></td></tr>' if winner else ""
    )
    # Bulletproof CTA button (MSO/VML for Outlook + a normal anchor for everyone else).
    button = ""
    if link:
        button = (
            f'<tr><td align="center" style="padding:22px 28px 8px">'
            f'<!--[if mso]>'
            f'<v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" xmlns:w="urn:schemas-microsoft-com:office:word" '
            f'href="{esc(link)}" style="height:46px;v-text-anchor:middle;width:260px;" arcsize="14%" '
            f'strokecolor="{accent}" fillcolor="{accent}">'
            f'<w:anchorlock/><center style="color:#ffffff;font-family:sans-serif;font-size:15px;font-weight:bold;">'
            f'See the full leaderboard &rarr;</center></v:roundrect>'
            f'<![endif]-->'
            f'<!--[if !mso]><!-->'
            f'<a href="{esc(link)}" style="background:{accent};color:#ffffff;font-size:15px;font-weight:600;'
            f'text-decoration:none;padding:13px 28px;border-radius:8px;display:inline-block">'
            f'See the full leaderboard &rarr;</a>'
            f'<!--<![endif]-->'
            f'</td></tr>'
        )
    summary_line = f"{n_done} of {total} job(s) succeeded" + (f" · {n_failed} failed" if n_failed else "")
    preheader = (
        f"{('Winner: ' + esc(_label(winner))) if winner else summary_line} — your fine-tuning run is done."
    )

    html_body = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light dark"><meta name="supported-color-schemes" content="light dark">'
        '<style>'
        '@media (prefers-color-scheme: dark){'
        '  .bg{background:#0d1117!important} .card{background:#161b22!important}'
        '  .ink{color:#e6edf3!important} .muted{color:#9da7b3!important}'
        '  .rowlight{background:#10241a!important}'
        '}'
        '@media only screen and (max-width:600px){ .container{width:100%!important} }'
        '</style></head>'
        f'<body class="bg" style="margin:0;padding:0;background:#eef1f5;">'
        # hidden preheader (preview text)
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0">{preheader}</div>'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef1f5">'
        '<tr><td align="center" style="padding:24px 12px">'
        '<table role="presentation" class="container" width="600" cellpadding="0" cellspacing="0" '
        'style="width:600px;max-width:600px;background:#ffffff;border-radius:14px;overflow:hidden;'
        'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'
        'box-shadow:0 1px 3px rgba(27,31,36,0.12)" class="card">'
        # hero band
        f'<tr><td style="background:{hero_bg};padding:30px 28px;text-align:center">'
        f'<div style="font-size:40px;line-height:1">{"\U0001f6a9" if all_failed else "\U0001f3c1"}</div>'
        f'<div style="color:#ffffff;font-size:22px;font-weight:700;margin-top:10px">{esc(headline)}</div>'
        f'<div style="color:#b9c4d0;font-size:14px;margin-top:6px">Run &lsquo;{esc(name)}&rsquo; &middot; {summary_line}</div>'
        f'</td></tr>'
        # quote band
        f'<tr><td style="padding:20px 28px 6px"><div style="border-left:3px solid {accent};'
        f'padding:6px 0 6px 14px;color:#57606a;font-style:italic;font-size:15px" class="muted">'
        f'“{esc(quote)}”</div></td></tr>'
        # winner band
        f'{winner_band}'
        # results table
        '<tr><td style="padding:14px 28px 4px">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;border:1px solid #eaeef2;border-radius:8px;overflow:hidden;font-size:14px">'
        '<tr style="background:#f6f8fa"><th style="padding:9px 12px;text-align:left;color:#57606a;font-weight:600" colspan="2">Model</th>'
        '<th style="padding:9px 12px;text-align:left;color:#57606a;font-weight:600">Status</th>'
        '<th style="padding:9px 12px;text-align:right;color:#57606a;font-weight:600">Result</th></tr>'
        f'{"".join(trs)}'
        '</table></td></tr>'
        # CTA
        f'{button}'
        # footer — surface the ranking metric + point at the live re-rank dropdown
        '<tr><td style="padding:14px 28px 26px;text-align:center">'
        f'<div style="color:#8c959f;font-size:12px;margin-bottom:4px" class="muted">'
        f'Ranked by <b>{esc(rank_metric)}</b>'
        + (' &middot; open the run to re-rank by a different metric (Token F1, ROUGE-L, exact match…)' if link else '')
        + '</div>'
        '<div style="color:#8c959f;font-size:12px" class="muted">You asked to be notified when this run finished. '
        '— SLM Fine-tuning Platform</div></td></tr>'
        '</table></td></tr></table></body></html>'
    )
    return {"subject": subject, "html": html_body, "text": text}


# --- send (SES; never raises) ----------------------------------------------- #


def send_race_complete_email(race: Any, ranked: list[dict], *, _client: Any = None) -> bool:
    """Send the completion email to race.notify_emails. Returns True if SES accepted
    a send (for at least the call), False if skipped/failed. NEVER raises — the
    reconcile loop must not break on a notification problem."""
    recipients = list(getattr(race, "notify_emails", []) or [])
    if not recipients:
        return False
    sender = _from_header()
    if not _from_email():
        log_event("notify.send_skipped_no_sender", raceId=getattr(race, "race_id", ""))
        return False
    client = _client or _ses_client()
    if client is None:
        return False
    body = build_race_complete_email(race, ranked)
    try:
        client.send_email(
            FromEmailAddress=sender,
            Destination={"ToAddresses": recipients},
            Content={
                "Simple": {
                    "Subject": {"Data": body["subject"], "Charset": "UTF-8"},
                    "Body": {
                        "Html": {"Data": body["html"], "Charset": "UTF-8"},
                        "Text": {"Data": body["text"], "Charset": "UTF-8"},
                    },
                }
            },
        )
        log_event("notify.race_complete_sent", raceId=getattr(race, "race_id", ""),
                  recipients=len(recipients))
        return True
    except Exception as e:  # noqa: BLE001 — sandbox/unverified/throttle/perms all degrade to no-email
        log_event("notify.send_failed", level="WARNING",
                  raceId=getattr(race, "race_id", ""), error=str(e))
        return False
