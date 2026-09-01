// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Table from "@cloudscape-design/components/table";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Badge from "@cloudscape-design/components/badge";
import Container from "@cloudscape-design/components/container";
import FormField from "@cloudscape-design/components/form-field";
import Input from "@cloudscape-design/components/input";
import Select from "@cloudscape-design/components/select";
import SegmentedControl from "@cloudscape-design/components/segmented-control";
import Textarea from "@cloudscape-design/components/textarea";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Modal from "@cloudscape-design/components/modal";

import ExpandableSection from "@cloudscape-design/components/expandable-section";

import {
  authorRewardPrompt,
  createRewardFunction,
  deleteRewardFunction,
  getDatasets,
  listRewardFunctions,
  tryRewardFunction,
  tryRewardPrompt,
  validateRewardPrompt,
  validateRewardSnippet,
  type Dataset,
  type RewardAuthorResult,
  type RewardFunction,
  type RewardPromptDryRun,
} from "./api";
import { useNotify, errText } from "./notifications";

const SNIPPET_TEMPLATE = `import scoring

def reward(response, ground_truth):
    # Return a float in 0..1. \`scoring\` mirrors the platform's eval metrics:
    #   scoring.score("token_f1"|"label_accuracy"|"json_valid"|"numeric_match"|..., response, ground_truth)
    #   scoring.extract_answer(response)  # strips <think>/fences first
    # Example: reward exact numeric correctness, with partial credit for closeness.
    return scoring.score("numeric_match", response, ground_truth)
`;

// An RLAIF reward PROMPT (an AI judge). MUST contain {{prompt}} and {{response}}
// — the judge fills them with the rollout's prompt + the model's response — and
// should ask for a JSON {score 0..1, reasoning}. Used by the RLAIF objective.
const PROMPT_TEMPLATE = `You are a strict evaluator. Rate how HELPFUL and well-written the response is.

User prompt:
{{prompt}}

Model response:
{{response}}

Score from 0.0 (poor) to 1.0 (excellent) on naturalness, helpfulness, and tone.
Reply with ONLY JSON: {"score": <0..1>, "reasoning": "<one sentence>"}`;

function statusIndicator(rf: RewardFunction) {
  if (rf.status === "deployed") return <StatusIndicator type="success">deployed</StatusIndicator>;
  if (rf.status === "deploying") return <StatusIndicator type="in-progress">deploying</StatusIndicator>;
  if (rf.status === "failed") return <StatusIndicator type="error">failed</StatusIndicator>;
  return <StatusIndicator type="pending">draft</StatusIndicator>;
}

export function RewardFunctionsPage() {
  const { notify } = useNotify();
  const [rows, setRows] = useState<RewardFunction[]>([]);
  const [metrics, setMetrics] = useState<string[]>([]);
  const [judgeModels, setJudgeModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  // Create form
  const [mode, setMode] = useState<"metric" | "snippet" | "prompt">("metric");
  const [name, setName] = useState("");
  const [metric, setMetric] = useState("token_f1");
  const [snippet, setSnippet] = useState(SNIPPET_TEMPLATE);
  const [snippetError, setSnippetError] = useState<string | null>(null);
  // RLAIF reward-prompt mode: the judge prompt + an optional judge model id.
  const [prompt, setPrompt] = useState(PROMPT_TEMPLATE);
  const [promptError, setPromptError] = useState<string | null>(null);
  const [rewardModelId, setRewardModelId] = useState("");
  const [creating, setCreating] = useState(false);
  // Rows whose Python snippet is expanded in the table (a created reward fn's
  // snippet was otherwise unviewable after creation).
  const [expanded, setExpanded] = useState<RewardFunction[]>([]);
  // Delete confirmation: the reward fn the user is about to delete (a deployed
  // reward fn owns a real Lambda + Evaluator, so deletion shouldn't be one click).
  const [removeTarget, setRemoveTarget] = useState<RewardFunction | null>(null);
  const [removing, setRemoving] = useState(false);

  // Dry-run ("Test reward"): score one sample (response, ground_truth) with the
  // current draft IN-PROCESS (no AWS, no billable run) so the user sees what the
  // deployed reward would return before launching GRPO.
  const [tryResponse, setTryResponse] = useState("The answer is 42.");
  const [tryGroundTruth, setTryGroundTruth] = useState("42");
  const [trying, setTrying] = useState(false);
  const [tryScore, setTryScore] = useState<number | null>(null);
  const [tryError, setTryError] = useState<string | null>(null);

  async function doTry() {
    setTrying(true);
    setTryError(null);
    setTryScore(null);
    try {
      const req =
        mode === "metric"
          ? { response: tryResponse, groundTruth: tryGroundTruth, metric }
          : { response: tryResponse, groundTruth: tryGroundTruth, snippet };
      const { score } = await tryRewardFunction(req);
      setTryScore(score);
    } catch (e) {
      setTryError(e instanceof Error ? e.message : String(e));
    } finally {
      setTrying(false);
    }
  }

  // Prompt-mode dry-run: score a GOOD and a BAD candidate with the rubric + judge,
  // and show whether the rubric DISCRIMINATES (good mean − bad mean) before deploy.
  const [goodResp, setGoodResp] = useState("Sure! Here's a clear, friendly answer in one line.");
  const [badResp, setBadResp] = useState("idk figure it out yourself.");
  const [dryPrompt, setDryPrompt] = useState("How do I reset my password?");
  const [promptTrying, setPromptTrying] = useState(false);
  const [promptDryRun, setPromptDryRun] = useState<RewardPromptDryRun | null>(null);
  const [promptTryError, setPromptTryError] = useState<string | null>(null);

  async function doPromptTry() {
    setPromptTrying(true);
    setPromptTryError(null);
    setPromptDryRun(null);
    try {
      const out = await tryRewardPrompt({
        prompt,
        rewardModelId: rewardModelId.trim() || undefined,
        samples: [
          { prompt: dryPrompt, response: goodResp, intendedLabel: "good" },
          { prompt: dryPrompt, response: badResp, intendedLabel: "bad" },
        ],
      });
      setPromptDryRun(out);
    } catch (e) {
      setPromptTryError(e instanceof Error ? e.message : String(e));
    } finally {
      setPromptTrying(false);
    }
  }

  // "Draft with AI": the reward-author agent writes a calibrated {{prompt}}/
  // {{response}} rubric for a plain-English goal, grounded in an RLAIF dataset, and
  // proves it separates good from bad (real judge scores) before you deploy. The
  // result pre-fills the rubric Textarea + judge Select (still hand-editable).
  const [rlaifDatasets, setRlaifDatasets] = useState<Dataset[]>([]);
  const [authorSplitId, setAuthorSplitId] = useState("");
  const [authorGoal, setAuthorGoal] = useState("reward concise, friendly, helpful answers");
  const [authoring, setAuthoring] = useState(false);
  const [authorResult, setAuthorResult] = useState<RewardAuthorResult | null>(null);
  const [authorError, setAuthorError] = useState<string | null>(null);

  async function doAuthor() {
    if (!authorSplitId || !authorGoal.trim()) return;
    setAuthoring(true);
    setAuthorError(null);
    setAuthorResult(null);
    try {
      const out = await authorRewardPrompt({
        splitId: authorSplitId,
        goal: authorGoal.trim(),
        // Regenerate-with-feedback: feed the prior draft back so a re-run refines it.
        priorResult: authorResult,
      });
      setAuthorResult(out);
      // Pre-fill the authoring form with the calibrated draft (still editable).
      if (out.draftPrompt) {
        setPrompt(out.draftPrompt);
        setPromptError(null);
      }
      setRewardModelId(out.rewardModelId || "");
    } catch (e) {
      setAuthorError(e instanceof Error ? e.message : String(e));
    } finally {
      setAuthoring(false);
    }
  }

  function load() {
    setLoading(true);
    listRewardFunctions()
      .then((r) => {
        setRows(r.rewardFunctions);
        setMetrics(r.metrics);
        setJudgeModels(r.judgeModels ?? []);
      })
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoading(false));
  }
  useEffect(load, []);

  // Load the user's RLAIF (prompt-only) datasets once, so "Draft with AI" can
  // ground the agent in a real dataset's prompt mix. Failing is non-fatal — the
  // picker just shows empty + a hint to create an RLAIF dataset first.
  useEffect(() => {
    getDatasets()
      .then((ds) => {
        const rlaif = ds.filter((d) => d.shape === "rlaif");
        setRlaifDatasets(rlaif);
        if (rlaif.length && !authorSplitId) setAuthorSplitId(rlaif[0].splitId);
      })
      .catch(() => setRlaifDatasets([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll while anything is still deploying (the Lambda + Evaluator build runs in
  // the worker), so the status flips to deployed/failed without a manual refresh.
  useEffect(() => {
    if (!rows.some((r) => r.status === "deploying")) return;
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [rows]);

  async function create() {
    if (!name.trim()) return;
    setCreating(true);
    try {
      if (mode === "snippet") {
        const v = await validateRewardSnippet(snippet);
        if (!v.ok) {
          setSnippetError(v.error ?? "invalid snippet");
          setCreating(false);
          return;
        }
      } else if (mode === "prompt") {
        const v = await validateRewardPrompt(prompt);
        if (!v.ok) {
          setPromptError(v.error ?? "invalid reward prompt");
          setCreating(false);
          return;
        }
      }
      const req =
        mode === "metric" ? { name: name.trim(), metric }
        : mode === "snippet" ? { name: name.trim(), snippet }
        : { name: name.trim(), prompt, rewardModelId: rewardModelId.trim() || undefined };
      const created = await createRewardFunction(req);
      notify({
        type: "success",
        content: `Created reward function "${created.name}".${
          created.kind === "reward_prompt" ? "" : " Deploying its scoring Lambda…"
        }`,
      });
      setName("");
      load();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setCreating(false);
    }
  }

  async function remove(rf: RewardFunction) {
    setRemoving(true);
    try {
      await deleteRewardFunction(rf.id);
      setRemoveTarget(null);
      notify({ type: "success", content: `Deleted reward function "${rf.name}".` });
      load();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setRemoving(false);
    }
  }

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Define how reinforcement fine-tuning rewards a model's answers. RLVR rewards a VERIFIABLE answer — pick a leaderboard metric or write Python. RLAIF rewards a SUBJECTIVE answer via an AI-judge prompt. Used by the RLVR/RLAIF objectives on the serverless engine."
        >
          Reward functions (RLVR / RLAIF)
        </Header>
      }
    >
      <SpaceBetween size="l">
        <Container header={<Header variant="h2">Create a reward function</Header>}>
          <SpaceBetween size="m">
            <SegmentedControl
              selectedId={mode}
              onChange={({ detail }) => setMode(detail.selectedId as "metric" | "snippet" | "prompt")}
              options={[
                { id: "metric", text: "From a metric (RLVR)" },
                { id: "snippet", text: "Custom Python (RLVR)" },
                { id: "prompt", text: "AI-judge prompt (RLAIF)" },
              ]}
            />
            <FormField label={<span>Name <i>- required</i></span>}>
              <Input value={name} placeholder="numeric-correctness" onChange={({ detail }) => setName(detail.value)} />
            </FormField>

            {mode === "metric" ? (
              <FormField
                label="Reward = leaderboard metric"
                description="The reward is the SAME metric the leaderboard ranks on, computed per rollout against ground_truth. 'Train against the metric you ship on.'"
              >
                <Select
                  selectedOption={{ value: metric, label: metric }}
                  onChange={({ detail }) => setMetric(detail.selectedOption.value!)}
                  options={metrics.map((m) => ({ value: m, label: m }))}
                  // Until the metric list loads, the options are empty; disable so
                  // the shown default can't be submitted as an option absent from
                  // its own list (and a future metrics change can't silently 400).
                  disabled={metrics.length === 0}
                  loadingText="Loading metrics…"
                  statusType={metrics.length === 0 ? "loading" : "finished"}
                />
              </FormField>
            ) : mode === "snippet" ? (
              <FormField
                label="reward(response, ground_truth) -> float"
                description="Python returning a float in 0..1. `scoring` (the platform's eval metrics + extract_answer) is importable. Runs in a Lambda in your account; a raising/NaN reward scores 0 (never crashes the run)."
                errorText={snippetError ?? undefined}
              >
                <Textarea
                  value={snippet}
                  onChange={({ detail }) => {
                    setSnippet(detail.value);
                    setSnippetError(null);
                  }}
                  rows={12}
                  spellcheck={false}
                />
              </FormField>
            ) : (
              <SpaceBetween size="m">
                {/* "Draft with AI": the reward-author agent writes a calibrated
                    rubric for a goal + dataset and proves it discriminates good vs
                    bad before you deploy — so you start from a tested rubric, not a
                    blank box. Advisory: it pre-fills the form; deploy stays your click. */}
                <ExpandableSection
                  defaultExpanded
                  headerText="Draft with AI (calibrated rubric)"
                  headerDescription="Describe the goal; the agent writes a {{prompt}}/{{response}} rubric, scores good vs bad candidates with a real judge, and iterates until they separate. The draft fills the editor below — review + edit before deploying."
                >
                  <SpaceBetween size="s">
                    <FormField
                      label="Training goal (plain English)"
                      description="What should the reward encourage? e.g. 'reward concise, friendly tone' or 'reward accurate, well-cited summaries'."
                    >
                      <Input
                        value={authorGoal}
                        placeholder="reward concise, friendly, helpful answers"
                        onChange={({ detail }) => setAuthorGoal(detail.value)}
                      />
                    </FormField>
                    <FormField
                      label="Ground in an RLAIF dataset"
                      description="The agent draws realistic candidate responses from this prompt-only dataset's prompt mix. Create an RLAIF (prompt-only) dataset first if the list is empty."
                    >
                      <Select
                        selectedOption={
                          authorSplitId
                            ? {
                                value: authorSplitId,
                                label:
                                  rlaifDatasets.find((d) => d.splitId === authorSplitId)?.name ||
                                  authorSplitId,
                              }
                            : null
                        }
                        placeholder={
                          rlaifDatasets.length ? "Choose an RLAIF dataset" : "No RLAIF datasets yet"
                        }
                        onChange={({ detail }) => setAuthorSplitId(detail.selectedOption.value ?? "")}
                        options={rlaifDatasets.map((d) => ({
                          value: d.splitId,
                          label: d.name || d.splitId,
                          description: `${d.trainRows ?? "?"} prompts`,
                        }))}
                        disabled={rlaifDatasets.length === 0}
                        empty="No RLAIF (prompt-only) datasets found."
                      />
                    </FormField>
                    <Box>
                      <Button
                        loading={authoring}
                        disabled={!authorSplitId || !authorGoal.trim()}
                        onClick={doAuthor}
                        iconName="gen-ai"
                      >
                        {authorResult ? "Regenerate with feedback" : "Draft rubric with AI"}
                      </Button>
                    </Box>
                    {authoring && (
                      <Box variant="small" color="text-body-secondary">
                        The agent is drafting + calibrating against a real judge — this scores several
                        candidates and can take up to a couple of minutes.
                      </Box>
                    )}
                    {authorResult?.scoreSpread && authorResult.scoreSpread.goodMean != null && (
                      <Alert type={authorResult.scoreSpread.discriminates ? "success" : "warning"}>
                        Drafted rubric scored good{" "}
                        <b>{((authorResult.scoreSpread.goodMean ?? 0) * 100).toFixed(0)}%</b> vs bad{" "}
                        <b>{((authorResult.scoreSpread.badMean ?? 0) * 100).toFixed(0)}%</b> —
                        separation{" "}
                        <b>{((authorResult.scoreSpread.separation ?? 0) * 100).toFixed(0)} pts</b>
                        {" "}over {authorResult.judgeCalls ?? authorResult.samples?.length ?? 0} judge
                        calls in {authorResult.iterations} round
                        {authorResult.iterations === 1 ? "" : "s"}.{" "}
                        {authorResult.scoreSpread.discriminates
                          ? "It discriminates well — review the draft below and deploy."
                          : "Weak separation (<30 pts) — refine the goal or regenerate before deploying."}
                      </Alert>
                    )}
                    {authorResult?.rationale && authorResult.rationale.length > 0 && (
                      <Box variant="small" color="text-body-secondary">
                        <b>Why this rubric:</b>
                        <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                          {authorResult.rationale.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                      </Box>
                    )}
                    {authorResult?.warnings && authorResult.warnings.length > 0 && (
                      <Alert type="info">
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                          {authorResult.warnings.map((w, i) => (
                            <li key={i}>{w}</li>
                          ))}
                        </ul>
                      </Alert>
                    )}
                    {authorResult?.samples && authorResult.samples.length > 0 && (
                      <ExpandableSection headerText={`Scored candidates (${authorResult.samples.length})`}>
                        <SpaceBetween size="xxs">
                          {authorResult.samples.map((s, i) => (
                            <Box key={i} variant="small" color="text-body-secondary">
                              <Badge color={s.intendedLabel === "good" ? "green" : "grey"}>
                                {s.intendedLabel}
                              </Badge>{" "}
                              {s.error ? (
                                <Box color="text-status-error" display="inline">{s.error}</Box>
                              ) : (
                                <>
                                  <b>{(s.score * 100).toFixed(0)}%</b> — {s.response}
                                </>
                              )}
                            </Box>
                          ))}
                        </SpaceBetween>
                      </ExpandableSection>
                    )}
                    {authorError && (
                      <Alert type="error" dismissible onDismiss={() => setAuthorError(null)}>
                        {authorError}
                      </Alert>
                    )}
                  </SpaceBetween>
                </ExpandableSection>

                <FormField
                  label="AI-judge reward prompt"
                  description="For RLAIF (subjective tasks). An AI judge reads this prompt with {{prompt}} and {{response}} filled in, and returns a 0..1 score. Both placeholders are required. No ground_truth — the judge IS the reward (no Lambda)."
                  errorText={promptError ?? undefined}
                >
                  <Textarea
                    value={prompt}
                    onChange={({ detail }) => {
                      setPrompt(detail.value);
                      setPromptError(null);
                    }}
                    rows={12}
                    spellcheck={false}
                  />
                </FormField>
                <FormField
                  label="Judge model (optional)"
                  description="The model that scores responses. Leave as the recipe default, or pick a supported judge."
                >
                  <Select
                    selectedOption={{ value: rewardModelId, label: rewardModelId || "(recipe default)" }}
                    onChange={({ detail }) => setRewardModelId(detail.selectedOption.value ?? "")}
                    options={[
                      { value: "", label: "(recipe default)" },
                      ...judgeModels.map((m) => ({ value: m, label: m })),
                    ]}
                  />
                </FormField>

                {/* Prompt-mode dry-run: prove the rubric DISCRIMINATES good vs bad
                    BEFORE a billable GRPO run (RLAIF has no cheap small run). */}
                <ExpandableSection
                  headerText="Test rubric — good vs bad (dry-run)"
                  headerDescription="Score a sample good + bad response with this rubric to confirm it separates them. Indicative — uses the chosen judge (or a default preview), not a billable run."
                >
                  <SpaceBetween size="s">
                    <FormField label="Sample prompt">
                      <Input value={dryPrompt} onChange={({ detail }) => { setDryPrompt(detail.value); setPromptDryRun(null); }} />
                    </FormField>
                    <FormField label="A GOOD response (should score high)">
                      <Textarea value={goodResp} rows={2} spellcheck={false}
                        onChange={({ detail }) => { setGoodResp(detail.value); setPromptDryRun(null); }} />
                    </FormField>
                    <FormField label="A BAD response (should score low)">
                      <Textarea value={badResp} rows={2} spellcheck={false}
                        onChange={({ detail }) => { setBadResp(detail.value); setPromptDryRun(null); }} />
                    </FormField>
                    <Box>
                      <Button loading={promptTrying} onClick={doPromptTry} iconName="caret-right-filled">
                        Score good vs bad
                      </Button>
                    </Box>
                    {promptDryRun?.scoreSpread && (
                      <Alert type={promptDryRun.scoreSpread.discriminates ? "success" : "warning"}>
                        Good <b>{(promptDryRun.scoreSpread.goodMean * 100).toFixed(0)}%</b> vs
                        Bad <b>{(promptDryRun.scoreSpread.badMean * 100).toFixed(0)}%</b> —
                        separation <b>{(promptDryRun.scoreSpread.separation * 100).toFixed(0)} pts</b>.{" "}
                        {promptDryRun.scoreSpread.discriminates
                          ? "The rubric discriminates well — safe to deploy."
                          : "Weak separation (<30 pts) — the rubric may reward everything similarly; tighten the criteria before deploying."}
                        {" "}<i>Indicative score (preview judge), not the exact training-time reward.</i>
                      </Alert>
                    )}
                    {promptDryRun?.samples?.map((s, i) => (
                      <Box key={i} variant="small" color="text-body-secondary">
                        <Badge color={s.intendedLabel === "good" ? "green" : "grey"}>{s.intendedLabel}</Badge>{" "}
                        {s.error ? <Box color="text-status-error" display="inline">{s.error}</Box>
                          : <>score <b>{(s.score * 100).toFixed(0)}%</b> — {s.reasoning}</>}
                      </Box>
                    ))}
                    {promptTryError && (
                      <Alert type="error" dismissible onDismiss={() => setPromptTryError(null)}>{promptTryError}</Alert>
                    )}
                  </SpaceBetween>
                </ExpandableSection>
              </SpaceBetween>
            )}

            {mode !== "prompt" && (
            <ExpandableSection
              headerText="Test reward (dry-run)"
              headerDescription="Score one sample before deploying — no AWS, no billable run."
            >
              <SpaceBetween size="s">
                <FormField label="Model response" description="A sample rollout. Reasoning scaffolding (<think>, fences) is stripped, same as training.">
                  <Textarea
                    value={tryResponse}
                    onChange={({ detail }) => { setTryResponse(detail.value); setTryScore(null); setTryError(null); }}
                    rows={3}
                    spellcheck={false}
                  />
                </FormField>
                <FormField label="Ground truth" description="The dataset row's verifiable target the reward grades against.">
                  <Input
                    value={tryGroundTruth}
                    onChange={({ detail }) => { setTryGroundTruth(detail.value); setTryScore(null); setTryError(null); }}
                  />
                </FormField>
                <Box>
                  <Button loading={trying} onClick={doTry} iconName="caret-right-filled">
                    Score this sample
                  </Button>
                </Box>
                {tryScore !== null && (
                  <Alert type={tryScore > 0 ? "success" : "warning"}>
                    Reward = <b>{tryScore.toFixed(3)}</b>
                    {tryScore === 0 ? " — this sample scores 0; check the metric/logic matches your ground_truth shape." : ""}
                  </Alert>
                )}
                {tryError && (
                  <Alert type="error" dismissible onDismiss={() => setTryError(null)}>
                    {tryError}
                  </Alert>
                )}
              </SpaceBetween>
            </ExpandableSection>
            )}

            <Box>
              <Button variant="primary" loading={creating} disabled={!name.trim()} onClick={create}>
                {mode === "prompt" ? "Create reward prompt" : "Create & deploy"}
              </Button>
            </Box>
            <Box variant="small" color="text-status-inactive">
              {mode === "prompt"
                ? "Create registers this prompt as a SageMaker REWARD_PROMPT Evaluator (no Lambda; ~quick). RLAIF launch requires that registered Evaluator — deploy it here first."
                : "Deploy creates a scoring Lambda + registers it as a SageMaker Evaluator in your account (takes ~30s). Identical functions are reused, so re-creating one is cheap."}
            </Box>
          </SpaceBetween>
        </Container>

        <Table
          header={<Header variant="h2" counter={`(${rows.length})`}>Your reward functions</Header>}
          loading={loading}
          items={rows}
          trackBy={(r) => r.id}
          // Expand a row to view its (generated or authored) Python snippet — the
          // snippet is otherwise only visible in the editor before submit.
          expandableRows={{
            getItemChildren: () => [],
            isItemExpandable: (r) => Boolean(r.snippet),
            expandedItems: expanded,
            onExpandableItemToggle: ({ detail }) =>
              setExpanded((prev) =>
                detail.expanded
                  ? [...prev, detail.item]
                  : prev.filter((x) => x.id !== detail.item.id)
              ),
          }}
          empty={<Box textAlign="center" color="inherit">No reward functions yet — create one above.</Box>}
          columnDefinitions={[
            {
              id: "name",
              header: "Name",
              // When a row is expanded, render its snippet beneath the name so the
              // user can read exactly what scores their rollouts.
              cell: (r) =>
                expanded.some((x) => x.id === r.id) ? (
                  <SpaceBetween size="xs">
                    <span>{r.name}</span>
                    <Box variant="code" fontSize="body-s">
                      <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{r.snippet}</pre>
                    </Box>
                  </SpaceBetween>
                ) : (
                  r.name
                ),
            },
            {
              id: "kind",
              header: "Kind",
              cell: (r) =>
                r.kind === "metric" ? <Badge color="blue">metric: {r.metric}</Badge>
                : r.kind === "reward_prompt" ? <Badge color="green">AI-judge prompt</Badge>
                : <Badge>custom Python</Badge>,
            },
            { id: "status", header: "Status", cell: statusIndicator },
            {
              id: "detail",
              header: "Detail",
              cell: (r) =>
                r.status === "failed" ? (
                  <Box color="text-status-error" fontSize="body-s">{r.error}</Box>
                ) : r.evaluatorArn ? (
                  <Box fontSize="body-s" color="text-status-inactive">Evaluator ready</Box>
                ) : (
                  ""
                ),
            },
            {
              id: "actions",
              header: "",
              cell: (r) => (
                <Button variant="inline-link" onClick={() => setRemoveTarget(r)}>
                  Delete
                </Button>
              ),
            },
          ]}
        />
      </SpaceBetween>

      <Modal
        visible={removeTarget !== null}
        onDismiss={() => !removing && setRemoveTarget(null)}
        header="Delete reward function"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="link" onClick={() => setRemoveTarget(null)} disabled={removing}>
                Cancel
              </Button>
              <Button
                variant="primary"
                loading={removing}
                onClick={() => removeTarget && remove(removeTarget)}
              >
                Delete
              </Button>
            </SpaceBetween>
          </Box>
        }
      >
        Delete reward function <b>{removeTarget?.name}</b>?
        {removeTarget?.deployed
          ? " Its deployed scoring Lambda + SageMaker Evaluator will be removed from your account."
          : ""}{" "}
        Any RLVR runs already launched with it are unaffected. This can't be undone.
      </Modal>
    </ContentLayout>
  );
}
