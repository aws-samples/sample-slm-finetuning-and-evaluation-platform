// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useRef, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Container from "@cloudscape-design/components/container";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import ButtonDropdown from "@cloudscape-design/components/button-dropdown";
import Input from "@cloudscape-design/components/input";
import Textarea from "@cloudscape-design/components/textarea";
import Grid from "@cloudscape-design/components/grid";
import Table from "@cloudscape-design/components/table";
import Alert from "@cloudscape-design/components/alert";
import Spinner from "@cloudscape-design/components/spinner";
import Checkbox from "@cloudscape-design/components/checkbox";
import Multiselect, { type MultiselectProps } from "@cloudscape-design/components/multiselect";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import PromptInput from "@cloudscape-design/components/prompt-input";
import LiveRegion from "@cloudscape-design/components/live-region";
import ChatBubble from "@cloudscape-design/chat-components/chat-bubble";
import Avatar from "@cloudscape-design/chat-components/avatar";

import {
  advancePitcrew,
  archivePitcrewSession,
  editPitcrewMessage,
  getPitcrewSession,
  listPitcrewSessions,
  newPitcrewSession,
  renamePitcrewSession,
  type CurrentSplit,
  type PitcrewMessage,
  type PitcrewSession,
  type PitcrewSessionSummary,
} from "./api";
import { getCurrentUser } from "./auth";
import { useNotify } from "./notifications";
import { DatasetPicker } from "./DatasetPicker";

// The Guided Fine-tuning agent page ("Pit Crew"): a professional ChatGPT/Claude-
// style chat built on the official AWS GenAI-chat components — ChatBubble + Avatar
// for the transcript and a persistent PromptInput (with FileInput drag-and-drop
// attach) pinned at the bottom. All judgement is server-side (pitcrew.py): the
// agent PROPOSES the best models, the user can swap/remove/add, and nothing is
// charged until they approve. Structured steps (effort, plan review, dataset pick)
// render as inline controls inside the agent's bubbles.

interface Props {
  // Jump to the Runs page once a race launches (so the user can watch it).
  onLaunched: (raceId: string) => void;
}

// Phases where the persistent prompt bar accepts free text: describing the goal,
// and correcting the agent's read of the task at the confirm step (so the user can
// type a correction instead of only clicking "Yes, that's right"). Other phases are
// driven by inline controls, so the bar becomes a contextual hint.
const TEXT_PHASES = new Set(["collect_goal", "confirm_task"]);
// Phases where data is brought in (the DatasetPicker renders inline).
const DATA_PHASES = new Set(["await_data"]);

export function GuidedFineTunePage({ onLaunched }: Props) {
  const { notify } = useNotify();
  const [sessions, setSessions] = useState<PitcrewSessionSummary[]>([]);
  const [session, setSession] = useState<PitcrewSession | null>(null);
  const [busy, setBusy] = useState(false);
  // Collapsible session history (ChatGPT-style). Persisted so the choice sticks.
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(
    () => (typeof localStorage !== "undefined" ? localStorage.getItem("pitcrew.sidebar") !== "closed" : true)
  );
  const toggleSidebar = () => {
    setSidebarOpen((o) => {
      const next = !o;
      try {
        localStorage.setItem("pitcrew.sidebar", next ? "open" : "closed");
      } catch {
        /* non-fatal */
      }
      return next;
    });
  };
  const threadEnd = useRef<HTMLDivElement | null>(null);

  // Persistent prompt-bar state (the goal text).
  const [prompt, setPrompt] = useState("");

  useEffect(() => {
    void (async () => {
      try {
        const list = await listPitcrewSessions();
        setSessions(list);
        if (list.length > 0) await openSession(list[0].sessionId);
        else await startNew();
      } catch (e) {
        notify({ type: "error", content: `Couldn't load guided sessions: ${String(e)}` });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [session?.messages.length]);

  // While a race is running, poll the session so the transcript updates and the
  // "✅ winner is X" bubble appears without the user touching anything. The backend
  // GET reconciles the race + flips phase→done (which stops this poll). Without it
  // the page is frozen for the whole 30–60 min run — the "follow progress here"
  // promise the launch message makes.
  const sid = session?.sessionId;
  const isRunning = session?.phase === "launched";
  useEffect(() => {
    if (!isRunning || !sid) return;
    const timer = setInterval(() => {
      getPitcrewSession(sid).then(setSession).catch(() => {/* transient; keep polling */});
    }, 20000);
    return () => clearInterval(timer);
  }, [isRunning, sid]);

  async function refreshSessions() {
    try {
      setSessions(await listPitcrewSessions());
    } catch {
      /* sidebar is best-effort */
    }
  }

  function resetPromptBar() {
    setPrompt("");
  }

  async function startNew() {
    setBusy(true);
    try {
      const s = await newPitcrewSession();
      setSession(s);
      resetPromptBar();
      await refreshSessions();
    } catch (e) {
      notify({ type: "error", content: `Couldn't start a session: ${String(e)}` });
    } finally {
      setBusy(false);
    }
  }

  async function openSession(id: string) {
    setBusy(true);
    try {
      setSession(await getPitcrewSession(id));
      resetPromptBar();
    } catch (e) {
      notify({ type: "error", content: `Couldn't open that session: ${String(e)}` });
    } finally {
      setBusy(false);
    }
  }

  async function renameSessionById(id: string, title: string) {
    const clean = title.trim();
    if (!clean) return;
    try {
      const updated = await renamePitcrewSession(id, clean);
      if (session?.sessionId === id) setSession(updated);
      await refreshSessions();
    } catch (e) {
      notify({ type: "error", content: `Couldn't rename: ${String(e)}` });
    }
  }

  async function archiveSessionById(id: string) {
    try {
      await archivePitcrewSession(id, true);
      const list = await listPitcrewSessions();
      setSessions(list);
      // If we archived the open session, switch to another (or start fresh).
      if (session?.sessionId === id) {
        if (list.length > 0) await openSession(list[0].sessionId);
        else await startNew();
      }
    } catch (e) {
      notify({ type: "error", content: `Couldn't archive: ${String(e)}` });
    }
  }

  // Advance the conversation one turn. Carries the version we last saw so a second
  // tab can't silently clobber the phase.
  async function advance(action: string, payload: Record<string, unknown> = {}) {
    if (!session) return;
    setBusy(true);
    try {
      const updated = await advancePitcrew(session.sessionId, action, payload, session.version);
      setSession(updated);
      void refreshSessions();
    } catch (e) {
      const msg = String(e);
      if (msg.includes("another tab")) {
        notify({ type: "warning", content: "This session changed elsewhere — reloading." });
        await openSession(session.sessionId);
      } else {
        notify({ type: "error", content: msg });
      }
    } finally {
      setBusy(false);
    }
  }

  // Edit/rewind an earlier free-text message. The backend truncates the thread,
  // unlinks any downstream dataset (kept on disk), and replays.
  async function editMessage(index: number, text: string) {
    if (!session) return;
    setBusy(true);
    try {
      setSession(await editPitcrewMessage(session.sessionId, index, text, session.version));
      void refreshSessions();
    } catch (e) {
      const msg = String(e);
      if (msg.includes("another tab")) {
        notify({ type: "warning", content: "This session changed elsewhere — reloading." });
        await openSession(session.sessionId);
      } else {
        notify({ type: "error", content: `Couldn't edit: ${msg}` });
      }
    } finally {
      setBusy(false);
    }
  }

  // The persistent prompt bar is for FREE TEXT (the goal description). Bringing
  // data — upload / Hugging Face / pick existing — is handled by the full
  // DatasetPicker rendered inline at the data step (same interface as Fine-tune
  // Step 1), so the bar only submits the goal.
  async function submitPrompt() {
    if (!session || busy) return;
    const text = prompt.trim();
    if (!text) return;
    if (session.phase === "collect_goal") {
      resetPromptBar();
      await advance("goal", { goal: text });
    } else if (session.phase === "confirm_task") {
      // Free text at the confirm step is a CORRECTION to the agent's read of the
      // task (the backend _h_confirm_task accepts a `correction`), so the user isn't
      // limited to the "Yes, that's right" button.
      resetPromptBar();
      await advance("confirm", { correction: text });
    }
  }

  const phase = session?.phase ?? "";
  const promptPlaceholder =
    phase === "collect_goal"
      ? "Describe what you want the model to do…"
      : phase === "confirm_task"
      ? "Looks right? Click the button — or type a correction here"
      : DATA_PHASES.has(phase)
      ? "Bring your data using the options above"
      : "Use the options above to continue";
  const promptDisabled = busy || !TEXT_PHASES.has(phase);

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Describe what you want a model to do and bring some examples. I'll propose a plan to train and compare several models — you can swap any, and you approve before anything runs."
          actions={<Button iconName="add-plus" onClick={startNew} disabled={busy}>New session</Button>}
        >
          Guided fine-tuning
        </Header>
      }
    >
      <Grid
        gridDefinition={
          sidebarOpen
            ? [{ colspan: { default: 12, s: 3 } }, { colspan: { default: 12, s: 9 } }]
            : [{ colspan: 12 }]
        }
      >
        {/* History sidebar — collapsible (ChatGPT-style). Each row is a self-
            contained session "bubble" with inline rename + archive. */}
        {sidebarOpen && (
          <Container
            header={
              <Header
                variant="h3"
                actions={
                  <Button
                    iconName="angle-left"
                    variant="icon"
                    ariaLabel="Collapse session history"
                    onClick={toggleSidebar}
                  />
                }
              >
                Your sessions
              </Header>
            }
          >
            <SpaceBetween size="xs">
              {sessions.length === 0 && <Box color="text-status-inactive">No sessions yet.</Box>}
              {sessions.map((s) => (
                <SessionRow
                  key={s.sessionId}
                  summary={s}
                  selected={session?.sessionId === s.sessionId}
                  busy={busy}
                  onOpen={() => openSession(s.sessionId)}
                  onRename={(title) => renameSessionById(s.sessionId, title)}
                  onArchive={() => archiveSessionById(s.sessionId)}
                />
              ))}
            </SpaceBetween>
          </Container>
        )}

        {/* Chat panel: scrollable transcript + a pinned prompt bar. When the sidebar
            is collapsed, a "show sessions" button appears in this panel's header. */}
        <Container
          disableContentPaddings
          header={
            !sidebarOpen ? (
              <Box padding={{ left: "s", top: "xs", bottom: "xs" }}>
                <Button
                  iconName="menu"
                  variant="normal"
                  onClick={toggleSidebar}
                >
                  Sessions
                </Button>
              </Box>
            ) : undefined
          }
        >
          {!session ? (
            <Box textAlign="center" padding="xxl"><Spinner size="large" /></Box>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 320px)", minHeight: 420 }}>
              {/* Scrollable transcript */}
              <div role="region" aria-label="Conversation" style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
                <SpaceBetween size="l">
                  {session.messages.map((m, i) => (
                    <ChatTurn
                      key={i}
                      message={m}
                      index={i}
                      active={i === session.messages.length - 1 && !busy}
                      busy={busy}
                      // Editing is disabled once a race has launched (its inputs are live).
                      canEdit={session.phase !== "launched" && session.phase !== "done"}
                      onAdvance={advance}
                      onEdit={editMessage}
                      onLaunched={onLaunched}
                    />
                  ))}
                  {busy && (
                    <ChatBubble
                      type="incoming"
                      ariaLabel="Guide is working"
                      showLoadingBar
                      avatar={<Avatar loading color="gen-ai" iconName="gen-ai" ariaLabel="Guide" />}
                    >
                      <Box color="text-status-inactive">Working…</Box>
                    </ChatBubble>
                  )}
                  <div ref={threadEnd} />
                </SpaceBetween>
              </div>

              {/* Pinned prompt bar (ChatGPT/Claude-style) for the goal description.
                  Data input (upload / HF / existing) is handled by the full
                  DatasetPicker rendered inline at the data step. */}
              <div style={{ borderTop: "1px solid var(--color-border-divider-default, #e9ebed)", padding: "12px 16px" }}>
                <PromptInput
                  value={prompt}
                  onChange={({ detail }) => setPrompt(detail.value)}
                  onAction={() => void submitPrompt()}
                  actionButtonAriaLabel="Send"
                  actionButtonIconName="send"
                  placeholder={promptPlaceholder}
                  disabled={promptDisabled}
                  maxRows={6}
                  minRows={1}
                />
                <LiveRegion hidden>{session.messages[session.messages.length - 1]?.text}</LiveRegion>
              </div>
            </div>
          )}
        </Container>
      </Grid>
    </ContentLayout>
  );
}

// One sidebar session "bubble": click to open; rename INLINE on the row (double-
// click or the ⋯ menu → edit); archive from the ⋯ menu. No separate text box.
function SessionRow({
  summary,
  selected,
  busy,
  onOpen,
  onRename,
  onArchive,
}: {
  summary: PitcrewSessionSummary;
  selected: boolean;
  busy: boolean;
  onOpen: () => void;
  onRename: (title: string) => Promise<void>;
  onArchive: () => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(summary.title);
  useEffect(() => setDraft(summary.title), [summary.title]);

  async function save() {
    const clean = draft.trim();
    if (clean && clean !== summary.title) await onRename(clean);
    setEditing(false);
  }

  if (editing) {
    return (
      <Input
        value={draft}
        onChange={({ detail }) => setDraft(detail.value)}
        onKeyDown={({ detail }) => {
          if (detail.key === "Enter") void save();
          if (detail.key === "Escape") { setDraft(summary.title); setEditing(false); }
        }}
        onBlur={() => void save()}
        autoFocus
        placeholder="Session name"
      />
    );
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: "8px 10px",
        borderRadius: 8,
        cursor: "pointer",
        background: selected ? "var(--color-background-item-selected, #f0fbff)" : "transparent",
        border: selected
          ? "1px solid var(--color-border-item-selected, #006ce0)"
          : "1px solid var(--color-border-divider-default, #e9ebed)",
      }}
      onClick={onOpen}
      onDoubleClick={() => setEditing(true)}
      title="Click to open · double-click to rename"
    >
      <span style={{ flex: 1, minWidth: 0, textAlign: "left" }}>
        <span style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {summary.title || "Fine-tuning session"}
        </span>
        <small style={{ color: "var(--color-text-status-inactive, #5f6b7a)" }}>
          {PHASE_LABEL[summary.phase] ?? summary.phase}
        </small>
      </span>
      <span onClick={(e) => e.stopPropagation()}>
        <ButtonDropdown
          variant="icon"
          disabled={busy}
          ariaLabel={`Actions for ${summary.title}`}
          items={[
            { id: "rename", text: "Rename", iconName: "edit" },
            { id: "archive", text: "Delete", iconName: "remove" },
          ]}
          onItemClick={({ detail }) => {
            if (detail.id === "rename") setEditing(true);
            else if (detail.id === "archive") void onArchive();
          }}
        />
      </span>
    </div>
  );
}

const PHASE_LABEL: Record<string, string> = {
  greet: "New",
  collect_goal: "Describing the task",
  await_data: "Choosing data",
  profiling: "Reading data",
  confirm_task: "Confirming",
  choose_effort: "Choosing effort",
  building_plan: "Building plan",
  review_plan: "Awaiting your approval",
  launched: "Running",
  done: "Finished",
};

// One message in the transcript: a Cloudscape ChatBubble with the right avatar.
// Assistant bubbles may carry structured inline controls (effort buttons, the plan
// review, the existing-dataset picker) which render only on the ACTIVE message.
function ChatTurn({
  message,
  index,
  active,
  busy,
  canEdit,
  onAdvance,
  onEdit,
  onLaunched,
}: {
  message: PitcrewMessage;
  index: number;
  active: boolean;
  busy: boolean;
  canEdit: boolean;
  onAdvance: (action: string, payload?: Record<string, unknown>) => Promise<void>;
  onEdit: (index: number, text: string) => Promise<void>;
  onLaunched: (raceId: string) => void;
}) {
  const user = getCurrentUser();
  const isUser = message.role === "user";
  const initials =
    (user?.firstName?.[0] ?? "") + (user?.name?.split(" ")?.[1]?.[0] ?? "") || "You";
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.text);
  const editable = isUser && !!message.editable && canEdit;

  async function saveEdit() {
    const t = draft.trim();
    setEditing(false);
    if (t && t !== message.text) await onEdit(index, t);
  }

  return (
    <ChatBubble
      type={isUser ? "outgoing" : "incoming"}
      ariaLabel={isUser ? "You" : "Guide"}
      avatar={
        isUser ? (
          <Avatar ariaLabel="You" tooltipText="You" initials={initials.slice(0, 2).toUpperCase()} />
        ) : (
          <Avatar color="gen-ai" iconName="gen-ai" ariaLabel="Guide" tooltipText="Fine-tuning guide" />
        )
      }
      // Edit affordance on an editable user bubble (goal / correction). Editing an
      // earlier turn rewinds the conversation there; the created dataset is kept.
      actions={
        editable && !editing ? (
          <Button
            variant="inline-icon"
            iconName="edit"
            ariaLabel="Edit this message"
            disabled={busy}
            onClick={() => { setDraft(message.text); setEditing(true); }}
          />
        ) : undefined
      }
    >
      {editing ? (
        <SpaceBetween size="xs">
          <Textarea
            value={draft}
            onChange={({ detail }) => setDraft(detail.value)}
            autoFocus
            rows={2}
          />
          <SpaceBetween size="xs" direction="horizontal">
            <Button variant="primary" disabled={busy || !draft.trim()} onClick={saveEdit}>
              Save &amp; rewind
            </Button>
            <Button variant="link" onClick={() => setEditing(false)}>Cancel</Button>
          </SpaceBetween>
          <Box variant="small" color="text-status-inactive">
            Editing this rewinds the conversation to here. Any dataset you created stays saved.
          </Box>
        </SpaceBetween>
      ) : (
        <SpaceBetween size="s">
          <Box variant="span">{renderText(message.text)}</Box>
          {!isUser && active && (
            <InlineControls message={message} busy={busy} onAdvance={onAdvance} onLaunched={onLaunched} />
          )}
        </SpaceBetween>
      )}
    </ChatBubble>
  );
}

// Render newlines as line breaks (the backend uses \n bullets in its prose).
function renderText(text: string) {
  return text.split("\n").map((line, i) => (
    <span key={i}>
      {line}
      <br />
    </span>
  ));
}

// Inline controls for the active step. Free-text (goal) + file attach live in the
// persistent prompt bar; everything else (effort, plan review, existing-dataset
// pick, launch/finish jumps) renders here in the bubble.
function InlineControls({
  message,
  busy,
  onAdvance,
  onLaunched,
}: {
  message: PitcrewMessage;
  busy: boolean;
  onAdvance: (action: string, payload?: Record<string, unknown>) => Promise<void>;
  onLaunched: (raceId: string) => void;
}) {
  if (message.datasetsHint) return <DataStep busy={busy} onAdvance={onAdvance} />;
  if (message.confirmTask) {
    return (
      <Button variant="primary" disabled={busy} onClick={() => onAdvance("confirm")}>
        Yes, that's right
      </Button>
    );
  }
  if (message.chooseEffort) {
    return (
      <SpaceBetween size="xs" direction="horizontal">
        {(message.efforts ?? []).map((e) => (
          <Button key={e.key} disabled={busy} onClick={() => onAdvance("effort", { effort: e.key })}>
            {e.key[0].toUpperCase() + e.key.slice(1)} — {e.label}
          </Button>
        ))}
      </SpaceBetween>
    );
  }
  if (message.reviewPlan && message.plan && message.estimate) {
    return (
      <PlanReview
        plan={message.plan}
        estimate={message.estimate}
        addPool={message.addPool ?? []}
        notifyEmailPrefill={message.notifyEmailPrefill ?? ""}
        busy={busy}
        onAdvance={onAdvance}
      />
    );
  }
  if (message.launched && message.raceId) {
    return (
      <Button iconName="external" onClick={() => onLaunched(message.raceId!)}>
        Watch progress on the Races page
      </Button>
    );
  }
  if (message.finished && message.raceId) {
    return (
      <Button variant="primary" iconName="external" onClick={() => onLaunched(message.raceId!)}>
        See the full comparison
      </Button>
    );
  }
  return null;
}

// The data step reuses the FULL Fine-tune Step-1 dataset interface (DatasetPicker):
// pick an existing dataset, upload a new file, import from Hugging Face, or create a
// preference/KTO/RLVR/RLAIF set — every path the platform supports. DatasetPicker
// emits a CurrentSplit once a dataset is chosen/created; we hand its splitId to the
// state machine via use_dataset (only on an explicit Continue, so a stray selection
// doesn't trigger profiling). RL-shaped datasets are still declined server-side.
function DataStep({
  busy,
  onAdvance,
}: {
  busy: boolean;
  onAdvance: (action: string, payload?: Record<string, unknown>) => Promise<void>;
}) {
  const [picked, setPicked] = useState<CurrentSplit | null>(null);
  return (
    <SpaceBetween size="m">
      <DatasetPicker selected={picked} onSelect={setPicked} />
      <Button
        variant="primary"
        disabled={busy || !picked}
        onClick={() => picked && onAdvance("use_dataset", { splitId: picked.splitId })}
      >
        {picked ? `Use "${picked.name || picked.splitId}"` : "Choose a dataset to continue"}
      </Button>
    </SpaceBetween>
  );
}

// The plan review + approval screen — the one place a billable launch is triggered.
// Shows the exact models (each removable), a model to add, an estimated cost/time
// range, and an email opt-in. Never advances past here without an explicit approve.
function PlanReview({
  plan,
  estimate,
  addPool,
  notifyEmailPrefill,
  busy,
  onAdvance,
}: {
  plan: NonNullable<PitcrewMessage["plan"]>;
  estimate: NonNullable<PitcrewMessage["estimate"]>;
  addPool: NonNullable<PitcrewMessage["addPool"]>;
  notifyEmailPrefill: string;
  busy: boolean;
  onAdvance: (action: string, payload?: Record<string, unknown>) => Promise<void>;
}) {
  const user = getCurrentUser();
  // Default ON (opt-out): most users want the finish email. Prefill from the Cognito
  // user-pool email; on localhost there's no token, so the box starts empty for the
  // user to type one manually (the checkbox still defaults on).
  const [notifyOn, setNotifyOn] = useState<boolean>(true);
  const [email, setEmail] = useState<string>(notifyEmailPrefill || user?.email || "");
  // Multi-select: the user can add SEVERAL models in one go (the backend edit_models
  // handler already accepts add:[modelId,...]). Selecting fewer clicks + reads nicer.
  const [addChoices, setAddChoices] = useState<readonly MultiselectProps.Option[]>([]);

  const addOptions = addPool.map((m) => ({ label: `${m.displayName} (${m.paramsB}B)`, value: m.modelId }));

  // "up to N" transparency: the effort was a CEILING, not a quota. When the planner
  // filled fewer than the ceiling it's because more arms would be redundant for this
  // dataset — surface that as a feature, never a shortfall.
  const ceiling = plan.ceiling ?? plan.jobBudget;
  const capped = plan.capped === true || (ceiling > 0 && plan.models.length < ceiling);

  return (
    <SpaceBetween size="m">
      {capped && (
        <Alert type="info" header={`Raced ${plan.models.length} of up to ${ceiling}`}>
          You chose up to {ceiling} models, but {plan.models.length} is all that adds
          anything for this dataset — more would just be near-duplicates, so I stopped
          there to save you cost and time. You can still add specific models below.
        </Alert>
      )}
      <Table
        variant="embedded"
        items={plan.models}
        // entryKey is unique PER ARM (a model can appear as plain LoRA + DoRA +
        // full-FT, all sharing modelId), so keying/removing by it targets the exact
        // arm instead of collapsing/clobbering every arm of a model.
        trackBy="entryKey"
        header={<Header variant="h3" counter={capped ? `(${plan.models.length} of up to ${ceiling})` : `(${plan.models.length})`}>Models I'll train and compare</Header>}
        columnDefinitions={[
          { id: "name", header: "Model", cell: (m) => m.label ?? m.displayName },
          { id: "size", header: "Size", cell: (m) => `${m.paramsB}B` },
          { id: "why", header: "Why this one", cell: (m) => m.role },
          {
            id: "remove",
            header: "",
            cell: (m) => (
              <Button
                variant="inline-link"
                iconName="close"
                disabled={busy || plan.models.length <= 1}
                onClick={() => onAdvance("edit_models", { remove: [m.entryKey] })}
              >
                Remove
              </Button>
            ),
          },
        ]}
      />

      {addOptions.length > 0 && (
        <SpaceBetween size="xs" direction="horizontal">
          <Box variant="span">Want specific models too?</Box>
          <Multiselect
            selectedOptions={addChoices}
            onChange={({ detail }) => setAddChoices(detail.selectedOptions)}
            options={addOptions}
            placeholder="Add one or more models…"
            filteringType="auto"
            tokenLimit={4}
            disabled={busy}
          />
          <Button
            disabled={busy || addChoices.length === 0}
            onClick={() => {
              const ids = addChoices.map((o) => o.value).filter((v): v is string => !!v);
              if (ids.length) onAdvance("edit_models", { add: ids });
              setAddChoices([]);
            }}
          >
            {addChoices.length > 1 ? `Add ${addChoices.length}` : "Add"}
          </Button>
        </SpaceBetween>
      )}

      <Alert type="info" header="Estimated cost and time">
        About <b>${estimate.totalUsd.lo}–${estimate.totalUsd.hi}</b> total, taking roughly{" "}
        <b>{estimate.wallClockMin.lo}–{estimate.wallClockMin.hi} minutes</b> ({estimate.jobs} jobs across{" "}
        {plan.models.length} models). <Box variant="span" color="text-status-inactive">{estimate.disclaimer}</Box>
      </Alert>

      <SpaceBetween size="xs">
        <Checkbox checked={notifyOn} onChange={({ detail }) => setNotifyOn(detail.checked)}>
          Email me when the comparison finishes
        </Checkbox>
        {notifyOn && (
          <Input value={email} onChange={({ detail }) => setEmail(detail.value)} placeholder="you@example.com" type="email" />
        )}
      </SpaceBetween>

      <SpaceBetween size="xs" direction="horizontal">
        <Button
          variant="primary"
          disabled={busy}
          onClick={() => onAdvance("approve", notifyOn && email ? { notifyEmail: email } : {})}
        >
          Approve &amp; launch
        </Button>
        <Button disabled={busy} onClick={() => onAdvance("edit_effort")}>Change effort level</Button>
        <Button variant="link" disabled={busy} onClick={() => onAdvance("cancel")}>Cancel</Button>
      </SpaceBetween>
      <StatusIndicator type="info">Nothing is charged until you approve.</StatusIndicator>
    </SpaceBetween>
  );
}
