// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import Container from "@cloudscape-design/components/container";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Grid from "@cloudscape-design/components/grid";
import Cards from "@cloudscape-design/components/cards";
import Badge from "@cloudscape-design/components/badge";
import Icon from "@cloudscape-design/components/icon";
import Spinner from "@cloudscape-design/components/spinner";
import ExpandableSection from "@cloudscape-design/components/expandable-section";
import { ProviderIcon } from "./providerIcon";
import { getModels, getDatasets, listRaces } from "./api";

// Live, data-driven landing page. Reuses Cloudscape (Cards/Grid/Badge) + the
// lobehub provider logos already in the app — no new dependency. It still teaches
// the newcomer (4-step flow + glossary), but now leads with what the platform can
// DO (live stats + the breadth of methods/agents/providers) so it reflects how
// much has been built. `go` navigates; everything else is presentational.

// Each stat resolves INDEPENDENTLY (null = still loading) so a fast call (races,
// datasets) paints immediately instead of blocking on the slow catalog endpoint
// (/api/models enriches every model with its verification status — ~10s+).
interface LiveStats {
  models: number | null;
  providers: number | null;
  datasets: number | null;
  runs: number | null;
}

// The training objectives the platform supports (the breadth showcase). Each
// carries a "use when" so a newcomer can pick the right LEARNING SIGNAL for their
// data — the objective is dictated mostly by the SHAPE of the data you have.
const METHODS: { key: string; name: string; blurb: string; tag: string; useWhen: string }[] = [
  { key: "sft", name: "SFT", blurb: "Supervised fine-tuning — learn from prompt → answer pairs.", tag: "LoRA / QLoRA",
    useWhen: "You have examples of the right answer (prompt → answer). The default starting point for almost every task." },
  { key: "dpo", name: "DPO", blurb: "Direct Preference Optimization — learn from chosen vs. rejected.", tag: "preference",
    useWhen: "You have pairs where one answer is better than another, and want to push the model toward the preferred style/tone." },
  { key: "kto", name: "KTO", blurb: "Kahneman-Tversky — learn from good/bad labelled answers.", tag: "binary signal",
    useWhen: "You only have a thumbs-up / thumbs-down on each answer (no paired comparison) — cheaper to label than DPO pairs." },
  { key: "rlvr", name: "RLVR", blurb: "RL from a Verifiable Reward — GRPO against a checkable answer.", tag: "verifiable",
    useWhen: "Correctness is mechanically checkable — math, code, exact labels, JSON. The reward is the check itself." },
  { key: "rlaif", name: "RLAIF", blurb: "RL from AI Feedback — GRPO against an AI-judge reward prompt.", tag: "AI judge",
    useWhen: "Quality is judgable but not checkable (helpfulness, faithfulness, tone) — an AI judge scores each answer against your rubric." },
];

// The parameterization choice (HOW MUCH of the model you update) — orthogonal to
// the objective above. LoRA is the default; the rest are situational. Kept honest
// to the platform's real limits: full/freeze are SFT-only, ≤2B, on the bigger g6e.
const PARAMETERIZATIONS: { key: string; name: string; blurb: string; tag: string; useWhen: string }[] = [
  { key: "lora", name: "LoRA", blurb: "Trains a small adapter on top of the frozen base — the proven default.", tag: "default",
    useWhen: "Start here. Cheap, fast, and strong for most tasks; the adapter merges back to full weights at export." },
  { key: "qlora", name: "QLoRA", blurb: "LoRA on a 4-bit-quantized base — same adapter idea, less GPU memory.", tag: "memory-saver",
    useWhen: "The model is too big to LoRA-tune on the available GPU. Fits a larger model on the same card; merges to full precision." },
  { key: "full", name: "Full", blurb: "Updates every weight — a standalone model, highest capacity and cost.", tag: "≤2B · SFT",
    useWhen: "You need maximum capacity and an adapter isn't enough. Small models only (≤2B) on the bigger g6e card; SFT only." },
  { key: "freeze", name: "Freeze", blurb: "Trains only the top transformer layers — a lighter full-weight option.", tag: "≤2B · SFT",
    useWhen: "You want full-weight training but cheaper than Full — trains the top layers only. Small models (≤2B); SFT only." },
];

// A THIRD axis: optional LoRA "variants" — richer adapter recipes that ride the
// SAME cheap LoRA/QLoRA path (same rank/alpha, same merge-to-full-weights export),
// so they cost about the same to try. They aren't new methods — each is a modifier
// you turn on for a LoRA/QLoRA run. The honest framing: start plain, then race a
// variant against plain LoRA on YOUR data to see if it actually wins. `lift` is the
// one-liner on what it changes; `useWhen` is when to reach for it.
const LORA_VARIANTS: { key: string; name: string; blurb: string; tag: string; useWhen: string }[] = [
  { key: "dora", name: "DoRA", blurb: "Weight-Decomposed LoRA — adapts each weight's magnitude and direction separately.", tag: "recommended",
    useWhen: "The recommended variant to try first: best shot at closing the gap to full fine-tuning while staying on LoRA. Merges to full weights like LoRA." },
  { key: "rslora", name: "rsLoRA", blurb: "Rank-Stabilized LoRA — rescales the adapter by 1/√rank instead of 1/rank.", tag: "high rank",
    useWhen: "You're training at a higher LoRA rank (≥32) and want steadier, better-behaved training. One flag; merge-identical to LoRA, so zero export risk." },
  { key: "pissa", name: "PiSSA", blurb: "Initializes the adapter from an SVD of the base weights instead of from zero.", tag: "fast start",
    useWhen: "You want faster, higher convergence from a better starting point — useful on smaller datasets. Heaviest to set up (SVD init); still merges cleanly." },
  { key: "loraplus", name: "LoRA+", blurb: "Trains the adapter's B matrix at a higher learning rate than A (set the ratio).", tag: "simple win",
    useWhen: "A near-free speedup: one ratio knob (default 16×) that often converges faster than plain LoRA. Merge-identical to LoRA — the lowest-risk variant to try." },
];

// Preference-OBJECTIVE family: on a preference (chosen/rejected) dataset, you can
// train DPO or its reference-free cousins ORPO/SimPO. Same data, different loss —
// race them to see which wins. (These map to stage=dpo + a pref_loss in the engine.)
const PREFERENCE_OBJECTIVES: { key: string; name: string; blurb: string; tag: string; useWhen: string }[] = [
  { key: "dpo", name: "DPO", blurb: "Direct Preference Optimization — the proven standard, learns vs. a frozen reference model.", tag: "recommended",
    useWhen: "The default for preference data. Stable and well-understood; start here, then race the reference-free cousins to see if they beat it." },
  { key: "orpo", name: "ORPO", blurb: "Odds-Ratio Preference Optimization — folds preference into one SFT-style stage, no reference model.", tag: "reference-free",
    useWhen: "You want a cheaper run (no second resident model) and often-competitive quality. Great first alternative to race against DPO on your data." },
  { key: "simpo", name: "SimPO", blurb: "Simple Preference Optimization — reference-free with a length-normalized reward + target margin (γ).", tag: "reference-free",
    useWhen: "You want reference-free training with a tunable margin; can curb length bias. Tune γ if it under/over-shoots." },
];

// The three AI agents (Strands on Bedrock AgentCore) at the judgment boundaries.
// `pageLabel` + `findIt` make the jump-off honest: the agents aren't pages, they're
// actions embedded INSIDE a page, so we name the page AND the exact control to click.
const AGENTS: {
  name: string; blurb: string; where: string; page: string; pageLabel: string; findIt: string;
}[] = [
  { name: "Dataset investigator", blurb: "Asks only what the data can't reveal, then recommends a locked eval config.",
    where: "entry", page: "datasets", pageLabel: "Datasets",
    findIt: "Open a dataset, then click “Investigate this dataset”." },
  { name: "Failure triage", blurb: "Reads a failed job's logs, diagnoses the root cause, proposes a concrete fix.",
    where: "failure", page: "races", pageLabel: "Races",
    findIt: "On a failed race, click “Diagnose”." },
  { name: "Results interpreter", blurb: "Reads the leaderboard and recommends which model to ship for your priorities.",
    where: "exit", page: "leaderboard", pageLabel: "Leaderboard",
    findIt: "On the leaderboard, click “Which model should I ship?”" },
];

// Provider logos to show in the hero strip (matches the lobehub set in providerIcon).
const HERO_PROVIDERS = ["Qwen", "Meta", "DeepSeek", "Mistral", "Google", "Microsoft", "InternLM", "Nvidia"];

export function HomePage({ go }: { go: (page: string) => void }) {
  const [stats, setStats] = useState<LiveStats>({ models: null, providers: null, datasets: null, runs: null });

  // Pull real numbers so the home page reflects the live platform. Each call
  // updates ITS OWN tiles as it returns — the fast ones (runs ~2s, datasets ~5s)
  // don't wait on the slow catalog (~10s+). Best-effort: a failed call leaves its
  // tiles at 0 rather than spinning forever. Cancellation-safe via `alive`.
  useEffect(() => {
    let alive = true;
    listRaces()
      .then((runs) => alive && setStats((s) => ({ ...s, runs: runs.length })))
      .catch(() => alive && setStats((s) => ({ ...s, runs: 0 })));
    getDatasets(false)
      .then((ds) => alive && setStats((s) => ({ ...s, datasets: ds.length })))
      .catch(() => alive && setStats((s) => ({ ...s, datasets: 0 })));
    getModels()
      .then((c) =>
        alive &&
        setStats((s) => ({
          ...s,
          models: c.models.length,
          providers: new Set(c.models.map((m) => m.provider)).size,
        }))
      )
      .catch(() => alive && setStats((s) => ({ ...s, models: 0, providers: 0 })));
    return () => { alive = false; };
  }, []);

  const steps = [
    { n: 1, title: "Add your data", body: "Upload prompt → answer examples (or import from Hugging Face / a demo). The app validates the format and an AI agent can investigate it to recommend how to score.", cta: "Manage datasets", page: "datasets" },
    { n: 2, title: "Submit a fine-tune", body: "Pick one or more models and a method (SFT, DPO, KTO, RLVR, RLAIF). One model is a single-entry race; several race together so you can compare.", cta: "Submit fine-tune jobs", page: "finetune" },
    { n: 3, title: "Watch it train", body: "Each submission becomes a race with live training curves, then automatic scoring on a held-out set the model never saw.", cta: "View races", page: "races" },
    { n: 4, title: "Compare and ship", body: "The leaderboard ranks every finished model on quality, speed, and cost — and an AI agent recommends which to ship.", cta: "View leaderboard", page: "leaderboard" },
  ];

  const glossary = [
    { term: "Fine-tune", def: "Teaching a general small model your specific task by training on your examples." },
    { term: "Race", def: "One submission. Trains + scores the model(s); add several models to compete head-to-head in the same race." },
    { term: "Method", def: "How the model learns: SFT (imitate), DPO/KTO (preferences), RLVR/RLAIF (reinforcement against a reward)." },
    { term: "LoRA variant", def: "An optional upgrade to a LoRA/QLoRA run — DoRA, rsLoRA, PiSSA, or LoRA+ — that can raise quality on the same cheap path. Plain LoRA is the default." },
    { term: "Evaluation", def: "How well a model did on held-out examples — task-aware metrics, plus an AI judge for open-ended answers." },
    { term: "Leaderboard", def: "A ranked comparison of finished models — quality vs. speed vs. cost — to help you choose." },
    { term: "Agent", def: "An AI helper (Strands on Bedrock AgentCore) at key decision points: investigate data, triage failures, interpret results." },
  ];

  // A stat shown INSIDE the gradient hero. Color is set EXPLICITLY on each text
  // node (#fff): Cloudscape's global styles set text color on descendant elements,
  // which overrides a `color` inherited from the parent div — so we can't rely on
  // inheritance for white-on-gradient.
  const heroStat = (label: string, value: number | null) => (
    <div style={{ textAlign: "center", minWidth: 92 }}>
      <div style={{ fontSize: 30, fontWeight: 700, lineHeight: 1.1, color: "#ffffff" }}>
        {value === null ? <Spinner /> : value}
      </div>
      <div style={{ fontSize: 12, color: "rgba(255,255,255,0.82)", marginTop: 2 }}>{label}</div>
    </div>
  );

  // One reusable "concept showcase" — the 4 educational card grids (objectives /
  // methods / variants / preference) are identical in shape, so render them from a
  // single helper instead of copy-pasting <Cards> four times.
  const conceptCards = (
    items: { key: string; name: string; blurb: string; tag: string; useWhen: string }[],
    accentKey: string,
    perRow: { cards: number }[],
  ) => (
    <Cards
      cardDefinition={{
        header: (p) => (
          <SpaceBetween direction="horizontal" size="xs">
            <span>{p.name}</span>
            <Badge color={p.key === accentKey ? "green" : "blue"}>{p.tag}</Badge>
          </SpaceBetween>
        ),
        sections: [
          { id: "blurb", content: (p) => <Box color="text-body-secondary">{p.blurb}</Box> },
          {
            id: "useWhen",
            content: (p) => (
              <Box>
                <Box variant="awsui-key-label">Use when</Box>
                <Box color="text-body-secondary">{p.useWhen}</Box>
              </Box>
            ),
          },
        ],
      }}
      cardsPerRow={perRow}
      items={items}
      trackBy="key"
    />
  );

  return (
    <ContentLayout disableOverlap>
      <SpaceBetween size="l">
        {/* ============================ HERO ============================ */}
        {/* A real focal point: brand gradient band, value prop, inline live stats,
            provider strip, and the primary CTA. White text on its own gradient, so
            it reads the same in light + dark mode. */}
        <div
          style={{
            background: "linear-gradient(120deg,#0d2440 0%,#13315c 45%,#2a4a7f 100%)",
            borderRadius: 16,
            padding: "40px 36px",
            color: "#ffffff",
            boxShadow: "0 1px 4px rgba(0,0,0,0.18)",
          }}
        >
          <Grid gridDefinition={[{ colspan: { default: 12, s: 7 } }, { colspan: { default: 12, s: 5 } }]}>
            <div>
              <span style={{
                display: "inline-block", background: "rgba(255,255,255,0.18)", borderRadius: 20,
                padding: "3px 12px", fontSize: 12, fontWeight: 600, marginBottom: 14, color: "#ffffff",
              }}>
                ✨ AI agents built in
              </span>
              <div style={{ fontSize: 30, fontWeight: 700, lineHeight: 1.2, color: "#ffffff" }}>
                Fine-tune small language models on your own data
              </div>
              <div style={{ fontSize: 16, color: "rgba(255,255,255,0.9)", marginTop: 12, maxWidth: 560, lineHeight: 1.5 }}>
                Bring your data, race open-source models head-to-head on a fair leaderboard,
                and ship the winner — across two training engines (self-hosted and managed
                serverless) behind one comparison layer, deployed with a single command.
              </div>
              <div style={{ marginTop: 22 }}>
                <SpaceBetween direction="horizontal" size="xs">
                  <Button variant="primary" iconName="gen-ai" onClick={() => go("guided")}>
                    Start guided
                  </Button>
                  <Button iconName="add-plus" onClick={() => go("datasets")}>
                    Add a dataset
                  </Button>
                  <Button onClick={() => go("finetune")}>Submit a fine-tune</Button>
                  <Button onClick={() => go("leaderboard")}>View leaderboard</Button>
                </SpaceBetween>
              </div>
            </div>
            <div>
              {/* live stats, right-aligned on wide screens */}
              <div style={{ display: "flex", gap: 24, flexWrap: "wrap", justifyContent: "flex-start" }}>
                {heroStat("Models", stats.models)}
                {heroStat("Providers", stats.providers)}
                {heroStat("Your datasets", stats.datasets)}
                {heroStat("Races launched", stats.runs)}
              </div>
              <div style={{ marginTop: 22, fontSize: 12, color: "rgba(255,255,255,0.82)" }}>Fine-tune models from</div>
              <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 8, flexWrap: "wrap" }}>
                {HERO_PROVIDERS.map((p) => (
                  <span key={p} title={p} style={{
                    background: "#ffffff", borderRadius: 8, padding: 5, display: "inline-flex",
                    alignItems: "center", lineHeight: 0,
                  }}>
                    <ProviderIcon provider={p} size={24} />
                  </span>
                ))}
              </div>
            </div>
          </Grid>
        </div>

        {/* ===================== GET STARTED IN 4 STEPS ===================== */}
        {/* Action-first: the newcomer immediately sees the path. A guided lead-in sits
            ABOVE the manual 4-step flow — for non-ML users, the chat agent is the
            recommended on-ramp (it does steps 1–2 for you). Numbered cards follow for
            those who want the full manual control. */}
        <Container header={<Header variant="h2" description="From raw data to a shipped model in four moves.">Get started</Header>}>
          <SpaceBetween size="l">
            <div style={{
              display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap",
              background: "rgba(19,49,92,0.06)", border: "1px solid rgba(19,49,92,0.15)",
              borderRadius: 12, padding: "16px 20px",
            }}>
              <Icon name="gen-ai" size="medium" />
              <div style={{ flex: 1, minWidth: 260 }}>
                <Box variant="h3" padding={{ top: "n" }}>New here? Let the guided agent do it for you</Box>
                <Box variant="p" color="text-body-secondary" fontSize="body-s">
                  Describe your goal in plain words and bring a dataset — the guided agent
                  profiles it, proposes a ready-to-run race (models + method chosen for you),
                  and shows the cost before anything runs. No ML knowledge needed.
                </Box>
              </div>
              <Button variant="primary" iconName="gen-ai" onClick={() => go("guided")}>
                Start guided fine-tuning
              </Button>
            </div>
            <Box variant="p" color="text-body-secondary" fontSize="body-s">
              Prefer full control? The manual path is four moves:
            </Box>
          <ColumnLayout columns={4} variant="text-grid">
            {steps.map((s) => (
              <div key={s.n}>
                <SpaceBetween size="xs">
                  <span style={{
                    display: "inline-flex", alignItems: "center", justifyContent: "center",
                    width: 30, height: 30, borderRadius: "50%", background: "#13315c",
                    color: "#fff", fontWeight: 700, fontSize: 14,
                  }}>{s.n}</span>
                  <Box variant="h3" padding={{ top: "xxs" }}>{s.title}</Box>
                  <Box variant="p" color="text-body-secondary" fontSize="body-s">{s.body}</Box>
                  <Button variant="inline-link" iconAlign="right" iconName="arrow-right" onClick={() => go(s.page)}>
                    {s.cta}
                  </Button>
                </SpaceBetween>
              </div>
            ))}
          </ColumnLayout>
          </SpaceBetween>
        </Container>

        {/* ============ HEADLINE CAPABILITIES (2-up) ============ */}
        {/* Left: the breadth of training objectives (5 stacked cards — the taller
            column). Right: the AI agents panel, with "Learn the options" expandables
            stacked beneath it so the right column fills the same height (no dead
            space) AND the deep reference stays scannable/collapsed by default. */}
        <Grid gridDefinition={[{ colspan: { default: 12, m: 7 } }, { colspan: { default: 12, m: 5 } }]}>
          <Container header={<Header variant="h2" description="Pick the learning signal by the SHAPE of the data you have — on either training engine.">Five ways to train</Header>}>
            {conceptCards(METHODS, "sft", [{ cards: 1 }, { cards: 2 }])}
          </Container>

          <SpaceBetween size="l">
            <Container
              header={
                <Header variant="h2" description="Strands agents on Bedrock AgentCore at the entry / failure / exit decision points.">
                  <SpaceBetween direction="horizontal" size="xs">
                    <span>AI agents, where judgment matters</span>
                    <Icon name="gen-ai" />
                  </SpaceBetween>
                </Header>
              }
            >
              <SpaceBetween size="m">
                {AGENTS.map((a) => (
                  <div key={a.name}>
                    <SpaceBetween size="xxs">
                      <SpaceBetween direction="horizontal" size="xs">
                        <Icon name="gen-ai" size="small" />
                        <Box variant="awsui-key-label">{a.name}</Box>
                        <Badge>{a.where}</Badge>
                      </SpaceBetween>
                      <Box color="text-body-secondary" fontSize="body-s">{a.blurb}</Box>
                      <Button variant="inline-link" iconAlign="right" iconName="arrow-right" onClick={() => go(a.page)}>
                        Go to {a.pageLabel}
                      </Button>
                    </SpaceBetween>
                  </div>
                ))}
              </SpaceBetween>
            </Container>

            {/* Deep reference, tucked into collapsible panels — sits UNDER the agents
                panel to fill the right column + keep the page scannable. */}
            <Container header={<Header variant="h2" description="Expand what you need.">Learn the options</Header>}>
              <SpaceBetween size="xs">
                <ExpandableSection
                  headerText="Fine-tuning methods"
                  headerDescription="How much of the model you update — LoRA is the default."
                >
                  {conceptCards(PARAMETERIZATIONS, "lora", [{ cards: 1 }, { cards: 2 }])}
                </ExpandableSection>
                <ExpandableSection
                  headerText="LoRA variants"
                  headerDescription="Optional richer adapter recipes (LoRA/QLoRA only) on the same cheap path."
                >
                  {conceptCards(LORA_VARIANTS, "dora", [{ cards: 1 }, { cards: 2 }])}
                </ExpandableSection>
                <ExpandableSection
                  headerText="Preference objectives — DPO, ORPO, SimPO"
                  headerDescription="Same preference data, different loss; ORPO/SimPO are reference-free."
                >
                  {conceptCards(PREFERENCE_OBJECTIVES, "dpo", [{ cards: 1 }, { cards: 2 }])}
                </ExpandableSection>
                <ExpandableSection
                  headerText="Glossary"
                  headerDescription="The words used around the app, in plain terms."
                >
                  <ColumnLayout columns={2} variant="text-grid">
                    {glossary.map((g) => (
                      <div key={g.term}>
                        <Box variant="awsui-key-label">{g.term}</Box>
                        <Box variant="p" color="text-body-secondary">{g.def}</Box>
                      </div>
                    ))}
                  </ColumnLayout>
                </ExpandableSection>
              </SpaceBetween>
            </Container>
          </SpaceBetween>
        </Grid>
      </SpaceBetween>
    </ContentLayout>
  );
}
