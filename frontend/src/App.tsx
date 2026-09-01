// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useMemo, useState, type ReactNode } from "react";
import AppLayout from "@cloudscape-design/components/app-layout";
import SideNavigation from "@cloudscape-design/components/side-navigation";
import TopNavigation, {
  type TopNavigationProps,
} from "@cloudscape-design/components/top-navigation";

import { HomePage } from "./HomePage";
import { GuidedFineTunePage } from "./GuidedFineTunePage";
import { FineTunePage } from "./FineTunePage";
import { DatasetsPage } from "./DatasetsPage";
import { CatalogPage } from "./CatalogPage";
import { ImagesPage } from "./ImagesPage";
import { EvalPage } from "./EvalPage";
import { RacesPage } from "./RacesPage";
import { LeaderboardPage } from "./LeaderboardPage";
import { SettingsPage } from "./SettingsPage";
import { FeedbackPage } from "./FeedbackPage";
import { RewardFunctionsPage } from "./RewardFunctionsPage";
import { getCurrentUser, logout } from "./auth";
import { cloneRaceConfig, getHfTokenStatus, type CurrentSplit, type Dataset, type RaceRequest } from "./api";
import BreadcrumbGroup from "@cloudscape-design/components/breadcrumb-group";
import HelpPanel from "@cloudscape-design/components/help-panel";
import Link from "@cloudscape-design/components/link";
import { NotificationProvider, AppNotifications, useNotify } from "./notifications";
import { storedMode, toggleMode, isDark } from "./theme";
import { Mode } from "@cloudscape-design/global-styles";

// Per-page breadcrumb label + a short contextual help blurb (shown in the
// AppLayout tools drawer). Keeps the domain explanations (RLVR vs RLAIF, image
// tiers, verify-before-trust) out of cramped page descriptions.
const PAGE_META: Record<string, { label: string; help: ReactNode }> = {
  home: { label: "Home", help: <p>Overview and quick links to start a fine-tune, manage datasets, or review runs.</p> },
  guided: {
    label: "Guided fine-tuning",
    help: (
      <>
        <p>A step-by-step guide for non-experts: describe what you want, bring some examples, and it proposes a plan to train and compare several models — you approve before anything runs.</p>
        <p>Nothing is charged until you approve. It shows the exact models, an estimated cost and time, and emails you when the comparison finishes.</p>
      </>
    ),
  },
  finetune: {
    label: "Submit fine-tune jobs",
    help: (
      <>
        <p>Build a run: pick a dataset, one or more models, a method (LoRA/QLoRA) and an objective.</p>
        <p>
          <b>Objective</b> follows the dataset shape — messages→SFT, preference→DPO, KTO, or a
          prompt-only set→RLVR/RLAIF on the serverless engine. RLVR rewards a verifiable answer;
          RLAIF rewards a subjective one via an AI-judge prompt.
        </p>
      </>
    ),
  },
  datasets: { label: "Manage datasets", help: <p>Upload, import from Hugging Face, split, and investigate datasets. The investigator recommends an objective + eval metric.</p> },
  catalog: {
    label: "Model catalog",
    help: (
      <>
        <p>Curated open-weight models. A <b>⚡ serverless</b> badge means the model can also be fine-tuned with the managed SageMaker Serverless engine.</p>
        <p><b>Verify-before-trust</b>: a model is "untested" on a surface (image tier / method / serverless) until a tiny smoke-test job proves it — then it's verified ✓.</p>
        <p>Use <b>Find serverless-customizable models</b> to browse every model SageMaker Serverless supports and add/enable them.</p>
      </>
    ),
  },
  images: { label: "Docker images", help: <p>The LLaMA-Factory image tiers (stable/latest). Check for new releases, build them, and see which models each tier supports.</p> },
  races: { label: "View races", help: <p>Every fine-tuning race, in progress and finished. Select one to watch its models go train→eval and see the winner.</p> },
  leaderboard: { label: "View leaderboard", help: <p>Compare fine-tuned models against base models and frontier baselines, ranked by the dataset's recommended metric (or an LLM judge).</p> },
  evaluate: { label: "Evaluate a fine-tuned model", help: <p>Run a held-out evaluation on a completed model, with task-aware metrics and an optional LLM-as-judge pass.</p> },
  rewards: { label: "Reward functions (RLVR / RLAIF)", help: <p>Define how reinforcement fine-tuning rewards answers: a verifiable metric/Python (RLVR), or an AI-judge prompt (RLAIF). "Draft with AI" calibrates a judge rubric for you.</p> },
  feedback: { label: "Feedback", help: <p>Report issues, request features, or share what worked — with screenshots. Everyone sees the board.</p> },
  settings: { label: "Settings", help: <p>Platform configuration (AWS identity, region, image, judge model) — informational; set at deploy time.</p> },
};

type Page =
  | "home"
  | "guided"
  | "finetune"
  | "datasets"
  | "catalog"
  | "images"
  | "races"
  | "leaderboard"
  | "evaluate"
  | "rewards"
  | "feedback"
  | "settings";

// Shown at most once per page load (module flag — survives StrictMode's double
// effect run, which would defeat the Flashbar's string-only dedupe for JSX content).
let hfNagShown = false;

function AppShell() {
  const [page, setPage] = useState<Page>("home");
  const { notify } = useNotify();
  // Light/dark mode (applied to <body> by main.tsx on load; toggled here). Kept in
  // state only to flip the toggle button's icon/label — applyMode does the work.
  const [mode, setMode] = useState<Mode>(() => storedMode());
  // Contextual help drawer (AppLayout tools slot).
  const [helpOpen, setHelpOpen] = useState(false);
  // A dataset pre-selected when the user clicks "Fine-tune on this" from Datasets.
  const [initialSplit, setInitialSplit] = useState<CurrentSplit | null>(null);
  // The most recently launched race, so the Races page auto-opens it.
  const [activeRaceId, setActiveRaceId] = useState<string | null>(null);
  // "Clone & edit": a prior run's launch config, passed to the Fine-Tune builder
  // to pre-fill it (same dataset + models/hp) so the user edits → submits a NEW run.
  const [cloneConfig, setCloneConfig] = useState<RaceRequest | null>(null);
  async function cloneRun(raceId: string) {
    try {
      const cfg = await cloneRaceConfig(raceId);
      setInitialSplit(null); // the builder resolves the dataset from the clone payload
      setCloneConfig(cfg);
      setPage("finetune");
    } catch {
      /* surfaced in the Fine-Tune page if the cart ends up empty */
    }
  }
  function fineTuneOnDataset(d: Dataset) {
    setInitialSplit({
      splitId: d.splitId,
      name: d.name || undefined,
      trainRows: d.trainRows ?? 0,
      evalRows: d.evalRows ?? 0,
      // Carry the dataset SHAPE so the Fine-Tune page selects the right objective
      // (preference→DPO, kto→KTO, rlvr→RLVR). Without this every non-SFT dataset
      // silently defaulted to SFT (FineTunePage reads currentSplit?.shape ?? "sft").
      shape: d.shape,
      // Carry validation metadata so early stopping is offered when available.
      hasVal: d.hasVal,
      valRows: d.valRows,
      // Carry the investigation's recommended reward so the RLVR step can offer
      // "reward on the metric you're ranked on" (the reward↔metric loop).
      recommendedRewardMetric: d.recommendedRewardMetric,
      // Carry the KTO loss-weight recommendation so the KTO step can one-click it.
      recommendedChosenWeight: d.recommendedChosenWeight,
      recommendedRejectedWeight: d.recommendedRejectedWeight,
      origin: "existing dataset",
    });
    setPage("finetune");
  }

  function onRaceLaunched(raceId: string) {
    setActiveRaceId(raceId);
    setCloneConfig(null); // consumed — a fresh Fine-Tune visit should start empty
    setPage("races");
  }

  // Logged-in user (decoded from the Cognito id_token) for the top-right menu.
  // null in local/open mode — we show a "Local" indicator + no sign-out there.
  const user = getCurrentUser();

  // One-time nag for users who haven't stored their OWN HF token. They still
  // work (the backend lends the platform's shared token) but downloads run
  // under the owner's HF account + license approvals, so steer them to
  // Settings. Best-effort: on fetch error say nothing.
  useEffect(() => {
    if (hfNagShown) return;
    getHfTokenStatus()
      .then((s) => {
        if (s.isSet || hfNagShown) return;
        hfNagShown = true;
        notify({
          type: "warning",
          header: "Set your Hugging Face token",
          content: (
            <span>
              You haven't saved a Hugging Face token yet
              {s.usingSharedFallback
                ? " — you're temporarily borrowing the platform's shared token, which uses the owner's model-license approvals"
                : " — gated models (Llama, Mistral, Gemma) and HF imports won't work without one"}
              . Please set your own on the{" "}
              <Link
                href="#settings"
                onFollow={(e) => {
                  e.preventDefault();
                  setPage("settings");
                }}
              >
                Settings page
              </Link>
              .
            </span>
          ),
          autoDismissMs: 30000,
        });
      })
      .catch(() => {/* status is a nicety — never block or error the shell */});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // A friendly multilingual greeting, rotated once per mount (useMemo so it stays
  // stable across re-renders rather than flickering). "Hey", "Hi", "Namaste", …
  const GREETINGS = ["Hey", "Hello", "Hi", "Namaste", "Hola", "Bonjour", "Ciao", "Hallo", "Olá", "こんにちは"];
  const greeting = useMemo(
    () => GREETINGS[Math.floor(Math.random() * GREETINGS.length)],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const userLabel = user
    ? `${greeting} ${user.firstName || user.name || user.shortUsername || "there"}`
    : "Local session";
  // Light/dark toggle in the top nav (left of the user menu). Shows the icon for
  // the mode you'd switch TO, like most apps.
  const themeUtility: TopNavigationProps.Utility = {
    type: "button",
    iconName: isDark(mode) ? "star" : "star-filled",
    text: isDark(mode) ? "Light" : "Dark",
    ariaLabel: isDark(mode) ? "Switch to light mode" : "Switch to dark mode",
    onClick: () => setMode((m) => toggleMode(m)),
  };
  // Opens the contextual help drawer for the current page.
  const helpUtility: TopNavigationProps.Utility = {
    type: "button",
    iconName: "status-info",
    text: "Help",
    ariaLabel: "Open help for this page",
    onClick: () => setHelpOpen(true),
  };
  const utilities: TopNavigationProps.Utility[] = user
    ? [
        helpUtility,
        themeUtility,
        {
          type: "menu-dropdown",
          text: userLabel,
          iconName: "user-profile",
          description: user.email,
          items: [
            ...(user.shortUsername
              ? [{ id: "username", text: `Signed in as ${user.shortUsername}`, disabled: true }]
              : []),
            { id: "signout", text: "Sign out" },
          ],
          onItemClick: ({ detail }) => {
            if (detail.id === "signout") logout();
          },
        },
      ]
    : [
        helpUtility,
        themeUtility,
        {
          type: "menu-dropdown",
          text: userLabel,
          iconName: "user-profile",
          description: "Not signed in (local dev)",
          items: [{ id: "info", text: "No authentication configured", disabled: true }],
        },
      ];

  return (
    <>
      {/* Pin the top bar while the page scrolls. position:sticky keeps it in normal
          flow (no content overlap). We deliberately do NOT set AppLayout's
          headerSelector — that changes AppLayout's internal sticky-offset math and
          broke the Runs table's own sticky header. Plain sticky positioning pins the
          bar without touching AppLayout's layout calculations. */}
      <div style={{ position: "sticky", top: 0, zIndex: 1002 }}>
        <TopNavigation
          identity={{ href: "#", title: "SLM Fine-tune Platform" }}
          utilities={utilities}
        />
      </div>
      <AppLayout
        notifications={<AppNotifications />}
        breadcrumbs={
          <BreadcrumbGroup
            items={[
              { text: "SLM Fine-tune Platform", href: "#home" },
              { text: PAGE_META[page]?.label ?? page, href: `#${page}` },
            ]}
            onFollow={(e) => {
              e.preventDefault();
              const target = e.detail.href.replace("#", "") as Page;
              if (target === "finetune") setCloneConfig(null);
              setPage(target);
            }}
          />
        }
        tools={<HelpPanel header={<h2>{PAGE_META[page]?.label ?? "Help"}</h2>}>{PAGE_META[page]?.help}</HelpPanel>}
        toolsOpen={helpOpen}
        onToolsChange={(e) => setHelpOpen(e.detail.open)}
        navigation={
          <SideNavigation
            activeHref={`#${page}`}
            header={{ href: "#home", text: "SLM Fine-tune Platform" }}
            onFollow={(e) => {
              e.preventDefault();
              const target = e.detail.href.replace("#", "") as Page;
              // Navigating to Fine-Tune via the nav (not via Clone) starts fresh —
              // drop any pending clone payload so the builder isn't re-pre-filled.
              if (target === "finetune") setCloneConfig(null);
              setPage(target);
            }}
            items={[
              { type: "link", text: "Home", href: "#home" },
              {
                type: "section-group",
                title: "Build",
                items: [
                  { type: "link", text: "Guided fine-tuning", href: "#guided" },
                  { type: "link", text: "Submit fine-tune jobs", href: "#finetune" },
                  { type: "link", text: "Manage datasets", href: "#datasets" },
                ],
              },
              {
                type: "section-group",
                title: "Monitor",
                items: [
                  { type: "link", text: "View races", href: "#races" },
                  { type: "link", text: "View leaderboard", href: "#leaderboard" },
                ],
              },
              {
                type: "section-group",
                title: "Library",
                items: [
                  { type: "link", text: "Model catalog", href: "#catalog" },
                  { type: "link", text: "Docker images", href: "#images" },
                  { type: "link", text: "Reward functions (RLVR)", href: "#rewards" },
                ],
              },
              { type: "divider" },
              { type: "link", text: "Evaluate a fine-tuned model", href: "#evaluate" },
              { type: "link", text: "Settings", href: "#settings" },
              { type: "link", text: "Feedback", href: "#feedback" },
            ]}
          />
        }
        content={
          page === "home" ? (
            <HomePage go={(p) => setPage(p as Page)} />
          ) : page === "guided" ? (
            <GuidedFineTunePage onLaunched={onRaceLaunched} />
          ) : page === "finetune" ? (
            <FineTunePage initialSplit={initialSplit} initialClone={cloneConfig} onLaunched={onRaceLaunched} />
          ) : page === "datasets" ? (
            <DatasetsPage onUseDataset={fineTuneOnDataset} />
          ) : page === "catalog" ? (
            <CatalogPage />
          ) : page === "images" ? (
            <ImagesPage />
          ) : page === "races" ? (
            <RacesPage activeRaceId={activeRaceId} onCloneRun={cloneRun} />
          ) : page === "leaderboard" ? (
            <LeaderboardPage currentSplit={initialSplit} />
          ) : page === "evaluate" ? (
            <EvalPage currentSplit={initialSplit} />
          ) : page === "rewards" ? (
            <RewardFunctionsPage />
          ) : page === "feedback" ? (
            <FeedbackPage />
          ) : (
            <SettingsPage />
          )
        }
      />
    </>
  );
}

// Wrap the shell in the notification provider so any page can call useNotify()
// to push toasts into the single app-level Flashbar (AppLayout notifications slot).
export default function App() {
  return (
    <NotificationProvider>
      <AppShell />
    </NotificationProvider>
  );
}
