// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import Modal from "@cloudscape-design/components/modal";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Box from "@cloudscape-design/components/box";
import Alert from "@cloudscape-design/components/alert";
import Badge from "@cloudscape-design/components/badge";
import Spinner from "@cloudscape-design/components/spinner";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Header from "@cloudscape-design/components/header";
import Container from "@cloudscape-design/components/container";
import Button from "@cloudscape-design/components/button";
import FormField from "@cloudscape-design/components/form-field";
import Textarea from "@cloudscape-design/components/textarea";
import Link from "@cloudscape-design/components/link";
import { AgentBadge } from "./AgentBadge";
import { useNotify, errText } from "./notifications";

import {
  profileDataset,
  investigateQuestions,
  investigateProposal,
  getInvestigation,
  type DatasetProfile,
  type FileProfile,
  type InvestigateQuestion,
  type InvestigateProposal,
} from "./api";

const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

function FileProfileCard({ title, f }: { title: string; f: FileProfile }) {
  if (!f || !f.parsed) {
    return (
      <Container header={<Header variant="h3">{title}</Header>}>
        <Box color="text-status-inactive">No rows.</Box>
      </Container>
    );
  }
  return (
    <Container header={<Header variant="h3">{title}</Header>}>
      <SpaceBetween size="s">
        <ColumnLayout columns={2} variant="text-grid">
          <div>
            <Box variant="awsui-key-label">Rows</Box>
            <Box>
              {f.rows}
              {f.sampled ? ` (sampled ${f.parsed})` : ""}
            </Box>
          </div>
          <div>
            <Box variant="awsui-key-label">Dominant task</Box>
            <Box>
              <Badge color="blue">{f.dominantTask}</Badge>{" "}
              {f.taskConsistency != null && f.taskConsistency < 1
                ? `(${pct(f.taskConsistency)} consistent)`
                : ""}
            </Box>
          </div>
          <div>
            <Box variant="awsui-key-label">Output length (words)</Box>
            <Box>
              p50 {f.goldWordLen?.p50 ?? "—"} · p95 {f.goldWordLen?.p95 ?? "—"} · max{" "}
              {f.goldWordLen?.max ?? "—"}
            </Box>
          </div>
          {f.scaffold?.rate ? (
            <div>
              <Box variant="awsui-key-label">Scaffold-wrapped answers</Box>
              <Box>
                {pct(f.scaffold.rate)}
                {f.scaffold.patterns && Object.keys(f.scaffold.patterns).length
                  ? ` (${Object.keys(f.scaffold.patterns).join(", ")})`
                  : ""}
              </Box>
            </div>
          ) : null}
          {f.malformed ? (
            <div>
              <Box variant="awsui-key-label">Malformed</Box>
              <Box color="text-status-error">{f.malformed}</Box>
            </div>
          ) : null}
          {f.emptyOutputs ? (
            <div>
              <Box variant="awsui-key-label">Empty outputs</Box>
              <Box>{f.emptyOutputs}</Box>
            </div>
          ) : null}
        </ColumnLayout>

        {f.taskMix && Object.keys(f.taskMix).length > 1 && (
          <Box fontSize="body-s">
            Task mix:{" "}
            {Object.entries(f.taskMix)
              .map(([k, v]) => `${k} ${v}`)
              .join(" · ")}
          </Box>
        )}

        {f.json?.jsonRows ? (
          <Box fontSize="body-s">
            JSON: valid {pct(f.json.goldValidRaw)} raw / {pct(f.json.goldValidStripped)} after
            stripping &lt;think&gt; · schema consistency {pct(f.json.schemaConsistency)} ·{" "}
            {f.json.distinctSchemas} distinct schema(s)
            {f.json.dominantKeys?.length ? ` · keys: ${f.json.dominantKeys.join(", ")}` : ""}
          </Box>
        ) : null}

        {f.labels?.numClasses ? (
          <SpaceBetween size="xxs">
            <Box fontSize="body-s">
              Labels: {f.labels.numClasses} classes · minority {pct(f.labels.minorityRate)}
              {f.labels.imbalanced ? " · imbalanced" : ""}
            </Box>
            {f.labels.classes?.length ? (
              <Box fontSize="body-s">
                Classes:{" "}
                {f.labels.classes.map((c) => (
                  <Badge key={c} color="grey">
                    {c}
                    {f.labels?.distribution?.[c] != null ? ` ${pct(f.labels.distribution[c])}` : ""}
                  </Badge>
                ))}
              </Box>
            ) : null}
          </SpaceBetween>
        ) : null}

        {f.truncation?.cutoffLen ? (
          <Box fontSize="body-s">
            Length vs cutoff {f.truncation.cutoffLen}: ~{pct(f.truncation.estTruncatedRows)} rows
            likely truncated (p95 ≈ {f.truncation.approxTokenP95} tokens)
          </Box>
        ) : null}
      </SpaceBetween>
    </Container>
  );
}

const facetColor = (facet: string) =>
  facet === "human" ? "severity-medium" : "blue";

/** The agentic follow-up: facet-gated questions → answers → config proposal.
 * Layers on top of the deterministic profile via the Strands agent on AgentCore. */
function AgenticInvestigation({ splitId, onFineTune }: { splitId: string; onFineTune?: () => void }) {
  const [questions, setQuestions] = useState<InvestigateQuestion[] | null>(null);
  const [summary, setSummary] = useState("");
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [proposal, setProposal] = useState<InvestigateProposal | null>(null);
  const [loadingQ, setLoadingQ] = useState(false);
  const [loadingP, setLoadingP] = useState(false);
  const { notify } = useNotify();

  // Restore any persisted investigation when the dataset changes.
  useEffect(() => {
    setQuestions(null);
    setSummary("");
    setAnswers({});
    setProposal(null);
    getInvestigation(splitId)
      .then((st) => {
        if (st.questions) {
          setQuestions(st.questions.questions);
          setSummary(st.questions.summary);
        }
        // Repopulate the user's previously-typed answers so they can review,
        // edit, and regenerate a new recommendation from them.
        if (st.answers && Object.keys(st.answers).length) setAnswers(st.answers);
        if (st.proposal) setProposal(st.proposal);
      })
      .catch(() => {});
  }, [splitId]);

  const ask = () => {
    setLoadingQ(true);
    investigateQuestions(splitId)
      .then((r) => {
        setQuestions(r.questions);
        setSummary(r.summary);
      })
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoadingQ(false));
  };

  const propose = () => {
    setLoadingP(true);
    investigateProposal(splitId, answers)
      .then(setProposal)
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoadingP(false));
  };

  const allAnswered =
    questions != null && questions.every((q) => (answers[q.id] || "").trim().length > 0);

  return (
    <Container
      header={
        <Header
          variant="h3"
          description={
            <>
              An AI agent (Strands on Bedrock AgentCore) asks only what the data can't reveal — the
              business context behind the numbers — then recommends a locked eval config. The
              question-selection gate ("ask only what you can't derive") follows the{" "}
              <Link
                external
                href="https://arxiv.org/abs/2403.00526"
                variant="primary"
              >
                Five Facets of Data Quality (arXiv 2403.00526)
              </Link>
              : only <i>task</i> and <i>human</i> facets are asked; <i>data</i>-facet dimensions are
              derived from the profile above.
            </>
          }
        >
          <SpaceBetween direction="horizontal" size="xs">
            <span>Follow-up questions</span>
            <AgentBadge />
          </SpaceBetween>
        </Header>
      }
    >
      <SpaceBetween size="m">
        {!questions && (
          <Button onClick={ask} loading={loadingQ} iconName="gen-ai" variant="primary">
            Ask follow-up questions
          </Button>
        )}

        {summary && (
          <Box variant="p" color="text-body-secondary">
            <i>{summary}</i>
          </Box>
        )}

        {questions && questions.length > 0 && (
          <SpaceBetween size="l">
            {questions.map((q) => (
              <FormField
                key={q.id}
                label={
                  <SpaceBetween direction="horizontal" size="xs">
                    <Badge color={facetColor(q.facet)}>{q.facet}</Badge>
                    <span>{q.question}</span>
                  </SpaceBetween>
                }
                description={`Why: ${q.why} · Affects: ${q.affects}`}
              >
                <Textarea
                  value={answers[q.id] || ""}
                  onChange={({ detail }) =>
                    setAnswers((a) => ({ ...a, [q.id]: detail.value }))
                  }
                  placeholder="Your answer (the agent uses this to tune the eval config)…"
                  rows={2}
                />
              </FormField>
            ))}
            <Button
              onClick={propose}
              loading={loadingP}
              disabled={!allAnswered}
              variant={proposal ? "normal" : "primary"}
              iconName={proposal ? "refresh" : undefined}
            >
              {proposal ? "Regenerate recommendation" : "Get recommendation"}
            </Button>
          </SpaceBetween>
        )}

        {/* The agent can decide NO follow-up questions are needed ("nothing the
            data can't reveal"). Without this branch the panel would dead-end —
            the Ask button is hidden (questions != null) and the questions block
            above is hidden (length === 0) — with no way to get a recommendation. */}
        {questions && questions.length === 0 && (
          <SpaceBetween size="s">
            <Box variant="p" color="text-body-secondary">
              No follow-up questions needed — the dataset is clear enough to
              recommend an eval configuration directly.
            </Box>
            <Button
              onClick={propose}
              loading={loadingP}
              variant={proposal ? "normal" : "primary"}
              iconName={proposal ? "refresh" : "gen-ai"}
            >
              {proposal ? "Regenerate recommendation" : "Get recommendation"}
            </Button>
          </SpaceBetween>
        )}

        {proposal && (
          <Alert
            type="success"
            header={
              <SpaceBetween direction="horizontal" size="xs">
                <span>Recommended configuration</span>
                <AgentBadge />
              </SpaceBetween>
            }
          >
            <SpaceBetween size="s">
              <Box>
                Task <Badge color="blue">{proposal.taskType}</Badge> · rank on{" "}
                <Badge color="green">{proposal.rankMetric}</Badge>
                {proposal.alsoWatch.length > 0 && (
                  <>
                    {" "}
                    · also watch{" "}
                    {proposal.alsoWatch.map((m) => (
                      <Badge key={m} color="grey">
                        {m}
                      </Badge>
                    ))}
                  </>
                )}
              </Box>
              {proposal.recommendedRewardMetric && (
                <Box fontSize="body-s">
                  RLVR reward: <Badge color="green">{proposal.recommendedRewardMetric}</Badge> — if you
                  fine-tune this as RLVR, the builder offers a reward on this exact metric (train on
                  what you're ranked on).
                </Box>
              )}
              {proposal.cutoffGuidance && (
                <Box fontSize="body-s">Cutoff: {proposal.cutoffGuidance}</Box>
              )}
              {proposal.flaggedIssues?.length > 0 && (
                <Box>
                  <Box variant="awsui-key-label">Flagged issues</Box>
                  <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                    {proposal.flaggedIssues.map((f, i) => (
                      <li key={i}>
                        <Box variant="span" fontSize="body-s">
                          {f}
                        </Box>
                      </li>
                    ))}
                  </ul>
                </Box>
              )}
              {proposal.rationale?.length > 0 && (
                <Box>
                  <Box variant="awsui-key-label">Why</Box>
                  <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                    {proposal.rationale.map((r, i) => (
                      <li key={i}>
                        <Box variant="span" fontSize="body-s">
                          {r}
                        </Box>
                      </li>
                    ))}
                  </ul>
                </Box>
              )}
              <Alert type="success" header="Applied">
                <SpaceBetween size="s">
                  <Box fontSize="body-s">
                    This dataset's leaderboard will default its <b>Rank by</b> metric to{" "}
                    <Badge color="green">{proposal.rankMetric}</Badge> the next time it loads. Fine-tune
                    this dataset, and the leaderboard will rank the winner on this metric (you can
                    still override it there). An already-open leaderboard won't pick this up until you
                    reopen or reload it.
                  </Box>
                  {onFineTune && (
                    <Button variant="primary" iconName="arrow-right" onClick={onFineTune}>
                      Fine-tune this dataset
                    </Button>
                  )}
                </SpaceBetween>
              </Alert>
            </SpaceBetween>
          </Alert>
        )}
      </SpaceBetween>
    </Container>
  );
}

export function InvestigateDataset({
  splitId,
  name,
  onClose,
  onFineTune,
  detailsOnly = false,
}: {
  splitId: string;
  name?: string | null;
  onClose: () => void;
  onFineTune?: () => void; // when provided, the proposal shows a "Fine-tune this dataset" button
  // When true, show the deterministic dataset DETAILS only (stats, structure,
  // eval strategy, warnings) and hide the agentic AI follow-up Q&A. Used by the
  // deep-link from Runs/Leaderboard, where the user wants "what's in this
  // dataset", not the full investigation wizard.
  detailsOnly?: boolean;
}) {
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const { notify } = useNotify();

  useEffect(() => {
    setProfile(null);
    profileDataset(splitId)
      .then(setProfile)
      .catch((e) => notify({ type: "error", content: errText(e) }));
  }, [splitId, notify]);

  return (
    <Modal
      visible
      onDismiss={onClose}
      size="large"
      header={detailsOnly ? `Dataset details — ${name || splitId}` : `Investigate dataset — ${name || splitId}`}
      footer={
        <Box float="right">
          <Button variant="link" onClick={onClose}>
            Close
          </Button>
        </Box>
      }
    >
      {!profile && (
        <Box textAlign="center" padding="l">
          <Spinner size="large" /> <Box variant="span">Profiling dataset…</Box>
        </Box>
      )}
      {profile && (
        <SpaceBetween size="l">
          <Box variant="p" color="text-body-secondary">
            {detailsOnly
              ? "What's in this dataset and how it will be evaluated. Advisory only — nothing is changed."
              : "A deterministic look at what's in this dataset and how to evaluate it, plus AI follow-up questions for the business context the data can't reveal. Advisory only — nothing is changed."}
          </Box>

          {/* Provenance — where the dataset came from (esp. Hugging Face imports) */}
          {profile.provenance?.source === "huggingface" && (
            <Alert type="info">
              Imported from Hugging Face:{" "}
              <b>{profile.provenance.hfDataset}</b>
              {profile.provenance.hfConfig ? ` · config ${profile.provenance.hfConfig}` : ""}
              {profile.provenance.hfSplit ? ` · split ${profile.provenance.hfSplit}` : ""}
              {profile.provenance.hfSampleSeed != null
                ? ` · sampled (seed ${profile.provenance.hfSampleSeed})`
                : ""}
              . This is a sample for pipeline testing, not the full dataset.
            </Alert>
          )}

          {/* Warnings first — the actionable stuff */}
          {profile.warnings.length > 0 && (
            <SpaceBetween size="xs">
              {profile.warnings.map((w, i) => (
                <Alert
                  key={i}
                  type={w.severity === "error" ? "error" : w.severity === "warning" ? "warning" : "info"}
                >
                  {w.message}
                </Alert>
              ))}
            </SpaceBetween>
          )}
          {profile.warnings.length === 0 && (
            <Alert type="success">No data-quality issues detected.</Alert>
          )}

          {/* Recommended eval strategy — the headline output */}
          <Container header={<Header variant="h3">Recommended eval strategy</Header>}>
            <SpaceBetween size="s">
              <Box>
                Detected task: <Badge color="blue">{profile.recommendation.detectedTask}</Badge> · rank
                the run on{" "}
                <Badge color="green">{profile.recommendation.rankMetric}</Badge>
                {profile.recommendation.alsoWatch.length > 0 && (
                  <>
                    {" "}
                    · also watch{" "}
                    {profile.recommendation.alsoWatch.map((m) => (
                      <Badge key={m} color="grey">
                        {m}
                      </Badge>
                    ))}
                  </>
                )}
              </Box>
              <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                {profile.recommendation.rationale.map((r, i) => (
                  <li key={i}>
                    <Box variant="span" fontSize="body-s">
                      {r}
                    </Box>
                  </li>
                ))}
              </ul>
            </SpaceBetween>
          </Container>

          {/* Recommended training objective (from the data SHAPE: preference→DPO,
              messages→SFT). Surfaced when DPO so the user sees WHY the run will use
              it (the Fine-tune page already gates the objective on the shape). */}
          {profile.objective?.objective === "dpo" && (
            <Alert type="info" header="Preference dataset → train with DPO">
              <SpaceBetween size="xs">
                <Box>
                  This dataset is <Badge color="green">preference</Badge> (chosen/rejected pairs)
                  {profile.preference ? <> · {profile.preference.pairs} pairs</> : null} — fine-tuning
                  it uses the <b>DPO</b> objective automatically.
                </Box>
                <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                  {profile.objective.rationale.map((r, i) => (
                    <li key={i}>
                      <Box variant="span" fontSize="body-s">
                        {r}
                      </Box>
                    </li>
                  ))}
                </ul>
              </SpaceBetween>
            </Alert>
          )}
          {profile.objective?.objective === "kto" && (
            <Alert type="info" header="KTO dataset → train with KTO">
              <SpaceBetween size="xs">
                <Box>
                  This dataset is <Badge color="green">KTO</Badge> (completions labelled good/bad)
                  {profile.kto ? <> · {profile.kto.rows} rows ({profile.kto.desirable} good)</> : null} —
                  fine-tuning it uses the <b>KTO</b> objective automatically.
                </Box>
                <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                  {profile.objective.rationale.map((r, i) => (
                    <li key={i}>
                      <Box variant="span" fontSize="body-s">{r}</Box>
                    </li>
                  ))}
                </ul>
              </SpaceBetween>
            </Alert>
          )}
          {profile.objective?.objective === "rlvr" && (
            <Alert type="info" header="RLVR dataset → train with a verifiable reward (GRPO)">
              <SpaceBetween size="xs">
                <Box>
                  This dataset is <Badge color="severity-high">RLVR</Badge> (prompt + verifiable
                  ground_truth)
                  {profile.rlvr ? <> · {profile.rlvr.rows} rows</> : null} — fine-tuning it uses the{" "}
                  <b>RLVR</b> objective on the serverless engine. You pick the reward (a preset or a
                  custom reward function) at launch.
                </Box>
                <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                  {profile.objective.rationale.map((r, i) => (
                    <li key={i}>
                      <Box variant="span" fontSize="body-s">{r}</Box>
                    </li>
                  ))}
                </ul>
              </SpaceBetween>
            </Alert>
          )}

          {/* Structure */}
          <Container header={<Header variant="h3">Conversation structure</Header>}>
            <SpaceBetween size="s">
              <ColumnLayout columns={3} variant="text-grid">
                <div>
                  <Box variant="awsui-key-label">System prompt</Box>
                  <Box>
                    {pct(profile.structure.hasSystemPromptRate)} of rows
                    {profile.structure.systemPromptFixed ? " · fixed" : " · varies"}
                  </Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Multi-turn</Box>
                  <Box>{pct(profile.structure.multiTurnRate)} of rows</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Validation split</Box>
                  <Box>{profile.hasVal ? "yes" : "no"}</Box>
                </div>
              </ColumnLayout>
              {profile.structure.fixedSystemPrompt ? (
                <div>
                  <Box variant="awsui-key-label">System prompt text</Box>
                  <Box variant="code" fontSize="body-s">
                    {profile.structure.fixedSystemPrompt}
                  </Box>
                </div>
              ) : profile.structure.distinctSystemPrompts &&
                profile.structure.distinctSystemPrompts > 1 ? (
                <Box fontSize="body-s" color="text-status-inactive">
                  {profile.structure.distinctSystemPrompts} distinct system prompts (varies per row).
                </Box>
              ) : null}
            </SpaceBetween>
          </Container>

          {/* Agentic follow-up — facet-gated questions + config proposal. Hidden in
              details-only mode (the deep-link from Runs/Leaderboard wants stats, not
              the full investigation wizard). */}
          {!detailsOnly && !profile.evalOnly && (
            <AgenticInvestigation splitId={splitId} onFineTune={onFineTune} />
          )}

          {/* Per-file profiles. Preference datasets have no messages train file —
              show a ranking-pair summary instead. */}
          {profile.train ? (
            <FileProfileCard title="Train" f={profile.train} />
          ) : profile.preference ? (
            <Container header={<Header variant="h3">Preference pairs (train)</Header>}>
              <ColumnLayout columns={4} variant="text-grid">
                <div>
                  <Box variant="awsui-key-label">Pairs</Box>
                  <Box>{profile.preference.pairs}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Chosen length (words)</Box>
                  <Box>{profile.preference.chosenWordLen?.p50 ?? "—"} median</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Rejected length (words)</Box>
                  <Box>{profile.preference.rejectedWordLen?.p50 ?? "—"} median</Box>
                </div>
                {/* Length bias: chosen ÷ rejected median words. ≥1.5 → DPO may
                    learn verbosity, not quality (Rafailov et al. 2023, App. D.2). */}
                <div>
                  <Box variant="awsui-key-label">Length bias (chosen ÷ rejected)</Box>
                  {profile.preference.lengthBiasRatio == null ? (
                    <Box>—</Box>
                  ) : profile.preference.lengthBiasRatio >= 1.5 ? (
                    <Box color="text-status-warning" fontWeight="bold">
                      {profile.preference.lengthBiasRatio}× — chosen much longer
                    </Box>
                  ) : (
                    <Box color="text-status-success">
                      {profile.preference.lengthBiasRatio}× (balanced)
                    </Box>
                  )}
                </div>
              </ColumnLayout>
            </Container>
          ) : profile.kto ? (
            <Container header={<Header variant="h3">KTO completions (train)</Header>}>
              <ColumnLayout columns={3} variant="text-grid">
                <div>
                  <Box variant="awsui-key-label">Rows</Box>
                  <Box>{profile.kto.rows}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Desirable / undesirable</Box>
                  <Box>
                    {profile.kto.desirable ?? "—"} / {profile.kto.undesirable ?? "—"}
                    {profile.kto.desirableRate != null
                      ? ` (${Math.round(profile.kto.desirableRate * 100)}% good)`
                      : ""}
                  </Box>
                </div>
                {/* Concrete λ recommendation (KTO paper §4.2): when the classes
                    are skewed, raise the minority class's loss weight so
                    λD·nD ≈ λU·nU. These are the exact values to enter in the
                    FineTune "KTO loss weights" inputs. */}
                <div>
                  <Box variant="awsui-key-label">Recommended loss weights (λD / λU)</Box>
                  {/* imbalanceRatio===null ⇒ a class is entirely missing (the set
                      is unusable for KTO — a hard error shows above); don't paint a
                      reassuring green "balanced" for it. */}
                  {profile.kto.imbalanceRatio == null ? (
                    <Box color="text-status-error">n/a — needs both classes</Box>
                  ) : profile.kto.weightsBalanced !== false ? (
                    <Box color="text-status-success">
                      1.0 / 1.0 — balanced ({profile.kto.imbalanceRatio}× ratio)
                    </Box>
                  ) : (
                    <Box color="text-status-warning" fontWeight="bold">
                      {profile.kto.recommendedChosenWeight ?? 1} / {profile.kto.recommendedRejectedWeight ?? 1}
                      {profile.kto.imbalanceRatio != null ? ` (${profile.kto.imbalanceRatio}× imbalance)` : ""}
                    </Box>
                  )}
                </div>
              </ColumnLayout>
            </Container>
          ) : profile.rlvr ? (
            <Container header={<Header variant="h3">RLVR prompts + verifiable target (train)</Header>}>
              <SpaceBetween size="s">
                <ColumnLayout columns={3} variant="text-grid">
                  <div>
                    <Box variant="awsui-key-label">Rows</Box>
                    <Box>{profile.rlvr.rows}</Box>
                  </div>
                  <div>
                    <Box variant="awsui-key-label">Prompt length (words)</Box>
                    <Box>{profile.rlvr.promptWordLen?.p50 ?? "—"} median</Box>
                  </div>
                  <div>
                    <Box variant="awsui-key-label">Ground-truth length (words)</Box>
                    <Box>{profile.rlvr.groundTruthWordLen?.p50 ?? "—"} median</Box>
                  </div>
                </ColumnLayout>
                <ColumnLayout columns={2} variant="text-grid">
                  <div>
                    <Box variant="awsui-key-label">Verifiable (numeric) ground-truth</Box>
                    <Box>
                      {profile.rlvr.numericGroundTruthRate != null
                        ? `${Math.round(profile.rlvr.numericGroundTruthRate * 100)}% contain a number`
                        : "—"}
                    </Box>
                  </div>
                  <div>
                    <Box variant="awsui-key-label">Empty ground-truth</Box>
                    <Box>
                      {profile.rlvr.emptyGroundTruth
                        ? `${profile.rlvr.emptyGroundTruth} row(s) — remove these`
                        : "none"}
                    </Box>
                  </div>
                </ColumnLayout>
                <Box fontSize="body-s" color="text-status-inactive">
                  The reward function checks each answer against ground_truth. Math presets
                  (gsm8k / prime_math) need a numeric/extractable target; for other tasks use a
                  custom reward whose logic matches your ground_truth.
                </Box>
              </SpaceBetween>
            </Container>
          ) : null}
          {profile.val && <FileProfileCard title="Validation" f={profile.val} />}
          <FileProfileCard title="Test" f={profile.eval} />

          <Box fontSize="body-s" color="text-status-inactive">
            Profiles sample up to 5,000 rows per file. Leakage check:{" "}
            {profile.leakage.checked
              ? `${profile.leakage.exactOverlapRows ?? 0} exact train↔eval overlap`
              : "n/a"}
            .
          </Box>
        </SpaceBetween>
      )}
    </Modal>
  );
}
