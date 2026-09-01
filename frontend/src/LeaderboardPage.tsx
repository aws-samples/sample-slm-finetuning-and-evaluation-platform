// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useMemo, useRef, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Popover from "@cloudscape-design/components/popover";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Table from "@cloudscape-design/components/table";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Badge from "@cloudscape-design/components/badge";
import Box from "@cloudscape-design/components/box";
import Select from "@cloudscape-design/components/select";
import FormField from "@cloudscape-design/components/form-field";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import TextFilter from "@cloudscape-design/components/text-filter";
import Modal from "@cloudscape-design/components/modal";
import Textarea from "@cloudscape-design/components/textarea";
import Spinner from "@cloudscape-design/components/spinner";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import { colHeader } from "./metricDefs";
import { QualityCostScatter } from "./QualityCostScatter";
import { JudgeRadar, type JudgeRadarModel } from "./JudgeRadar";
import { wilsonInterval, isSmallSample, marginLabel, SMALL_SAMPLE_THRESHOLD } from "./stats";
import { ProviderIcon } from "./providerIcon";
import { AgentBadge, AgentCaption } from "./AgentBadge";
import { useNotify, errText } from "./notifications";

import {
  getBaselineStatus,
  getBaselineModels,
  getDatasets,
  getEvalSplits,
  getLeaderboard,
  type Dataset,
  runSonnetBaseline,
  interpretLeaderboard,
  loadLastInterpret,
  getJudge,
  startJudge,
  type JudgeStatus,
  type BaselineModel,
  type CurrentSplit,
  type EvalSplit,
  type InterpretResult,
  type LeaderboardRow,
  type SonnetBaselineMetrics,
} from "./api";

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
function usd(v: number | null | undefined): string {
  return v == null ? "—" : `$${v.toFixed(4)}`;
}
function numOrDash(v: number | null | undefined): string {
  return v == null ? "—" : `${v}`;
}
function ms(v: number | null | undefined): string {
  // p50 latency in ms; show whole ms, or seconds once it's large enough to read better.
  if (v == null) return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
}

// Quality metrics the user can rank the leaderboard by (higher = better).
const RANK_OPTIONS = [
  { value: "tokenF1", label: "Token F1" },
  { value: "rougeL", label: "ROUGE-L" },
  { value: "charF1", label: "Char F1" },
  { value: "containsGold", label: "Contains gold" },
  { value: "normalizedMatch", label: "Normalized match" },
  { value: "exactMatch", label: "Exact match" },
  // RLAIF judge reward (reference-free) — only RLAIF rows carry it; others show "—".
  { value: "rewardMean", label: "Judge reward" },
] as const;

type RankKey = (typeof RANK_OPTIONS)[number]["value"];

// Map a dataset-investigation recommended metric (backend metric names like
// "label_accuracy", "json_structural") onto the leaderboard's rank keys. For
// single-token label tasks, label_accuracy == exact_match (the recommendation
// itself notes this), so we rank on exactMatch. Metrics with no leaderboard
// column fall through to undefined (we keep the current default then).
const RECOMMENDED_TO_RANKKEY: Record<string, RankKey> = {
  label_accuracy: "exactMatch",
  exact_match: "exactMatch",
  normalized_match: "normalizedMatch",
  contains_gold: "containsGold",
  token_f1: "tokenF1",
  rouge_l: "rougeL",
  char_f1: "charF1",
  json_structural: "tokenF1", // no dedicated column; F1 is the closest comparator
};

export function LeaderboardPage({ currentSplit }: { currentSplit: CurrentSplit | null }) {
  const { notify } = useNotify();
  const [splits, setSplits] = useState<EvalSplit[]>([]);
  const [splitId, setSplitId] = useState<string | null>(currentSplit?.splitId ?? null);
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [baselineRows, setBaselineRows] = useState<LeaderboardRow[]>([]);
  const [rankBy, setRankBy] = useState<RankKey>("tokenF1");
  const [filterText, setFilterText] = useState("");
  const [loading, setLoading] = useState(false);

  // Baseline model picker — a modal grouped by provider (Anthropic / Amazon /
  // Meta / Mistral / Cohere; more can be added backend-side, all via Converse).
  const [baselineModels, setBaselineModels] = useState<BaselineModel[]>([]);
  const [baselineModalOpen, setBaselineModalOpen] = useState(false);
  // Multi-baseline "cart": queue several models, run them all, track per-row state.
  const [cartProvider, setCartProvider] = useState<string | null>(null);
  const [cartModelKey, setCartModelKey] = useState<string | null>(null);
  // key → {label, provider, status: 'queued'|'running'|'done'|'error', detail?}
  const [cart, setCart] = useState<
    Record<string, { label: string; provider: string; status: string; detail?: string }>
  >({});
  const [runningAll, setRunningAll] = useState(false);

  // Results-interpreter agent.
  const [interpretOpen, setInterpretOpen] = useState(false);
  const [priorities, setPriorities] = useState("");
  const [interpretLoading, setInterpretLoading] = useState(false);
  const [interpretResult, setInterpretResult] = useState<InterpretResult | null>(null);

  // LLM-judge per row (keyed by evalJob), mirroring RacesPage: fetch known results
  // for the displayed non-baseline rows, poll while any is running, judge on demand.
  const [judges, setJudges] = useState<Record<string, JudgeStatus>>({});
  const judgePollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  async function runJudgeFor(evalJob: string) {
    setJudges((m) => ({ ...m, [evalJob]: { evalJob, status: "running", result: null } }));
    try {
      const r = await startJudge(evalJob);
      if (r.status === "done" && r.result) {
        setJudges((m) => ({ ...m, [evalJob]: { evalJob, status: "done", result: r.result! } }));
      }
    } catch (e) {
      notify({ type: "error", content: errText(e) });
      setJudges((m) => ({ ...m, [evalJob]: { evalJob, status: "failed", result: null } }));
    }
  }

  // Dataset library (rows/val/source/HF id) to enrich the selected-split stats
  // panel — getEvalSplits only carries name + recommended metric.
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  useEffect(() => {
    getDatasets(true).then(setDatasets).catch(() => {});
  }, []);
  const selectedDataset = datasets.find((d) => d.splitId === splitId) ?? null;

  useEffect(() => {
    getEvalSplits()
      .then((s) => {
        setSplits(s);
        setSplitId((cur) => cur ?? currentSplit?.splitId ?? s[0]?.splitId ?? null);
      })
      .catch((e) => notify({ type: "error", content: errText(e) }));
    getBaselineModels()
      .then((r) => setBaselineModels(r.models))
      .catch(() => {});
  }, []);

  // Build a leaderboard row from one baseline model's metrics.
  function baselineRowFromMetrics(sid: string, m: SonnetBaselineMetrics): LeaderboardRow {
    const label = m.baseline.label || "Frontier baseline";
    const key = m.baseline.key || "baseline";
    return {
      evalJob: `baseline-${key}`,
      sourceJob: null,
      model: `${label} (baseline)`,
      splitId: sid,
      count: m.count,
      exactMatch: m.exact_match,
      normalizedMatch: m.normalized_match,
      containsGold: m.contains_gold ?? null,
      tokenF1: m.token_f1,
      rougeL: m.rouge_l ?? null,
      charF1: m.char_f1 ?? null,
      lengthRatio: m.length_ratio ?? null,
      jsonStructural: m.json_structural,
      rewardMean: null, // frontier baselines have no reference-free reward
      backend: "bedrock",
      tokensPerSec: null,
      p50LatencyMs: null,
      p90LatencyMs: null, // API baseline has no self-host timing (explicit null)
      p99LatencyMs: null,
      trainCostUsd: null,
      trainInstance: null,
      projectedServeCostPer1k: null,
      evalInstance: "—",
      isBaseline: true,
      creationTime: "",
      apiCostPer1k: m.baseline.apiCostPer1kRows,
    };
  }

  function refresh(sid = splitId) {
    if (!sid) return;
    setLoading(true);
    getLeaderboard(sid)
      .then((res) => {
        setRows(res.rows);
        // Prefer the full baselines[] list; fall back to the single baseline.
        const bs = res.baselines && res.baselines.length ? res.baselines : res.baseline ? [res.baseline] : [];
        setBaselineRows(bs.map((m) => baselineRowFromMetrics(sid, m)));
      })
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoading(false));
  }

  // The selected split's dataset-investigation recommendation, if any.
  const selectedSplit = splits.find((s) => s.splitId === splitId) ?? null;
  const recommendedRankKey =
    selectedSplit?.recommendedRankMetric
      ? RECOMMENDED_TO_RANKKEY[selectedSplit.recommendedRankMetric]
      : undefined;

  useEffect(() => {
    if (splitId) refresh(splitId);
    // Load the LAST persisted recommendation for this split (survives reloads), so
    // the user sees "what you ran last time" + when. Clear first to avoid showing
    // the previous split's result while the fetch is in flight.
    setInterpretResult(null);
    if (splitId) {
      loadLastInterpret(splitId)
        .then((last) => {
          if (last) {
            setInterpretResult(last);
            if (last.priorities) setPriorities(last.priorities);
          }
        })
        .catch(() => {});
    }
    // Clear the baseline cart too: 'done' rows are scored against the OLD split,
    // and runCart skips 'done' models — so a leaked cart would never re-score the
    // baseline on the new split. Reset the picker selection alongside it.
    setCart({});
    setCartProvider(null);
    setCartModelKey(null);
    // Default 'Rank by' to the dataset-investigation recommendation for this
    // split (the user can still change it). Re-defaults per split on switch.
    if (recommendedRankKey) setRankBy(recommendedRankKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [splitId]);

  // Run ONE baseline model to terminal state; resolves with {ok, detail}. Updates
  // the leaderboard on success. Used by the cart's sequential "Run all".
  function runOneBaseline(sid: string, key: string): Promise<{ ok: boolean; detail?: string }> {
    return new Promise((resolve) => {
      runSonnetBaseline(sid, 256, 0.0, key)
        .then((r) => {
          if (r.status === "done" && r.metrics) {
            upsertBaseline(baselineRowFromMetrics(sid, r.metrics));
            resolve({ ok: true });
            return;
          }
          // Bound the poll so a baseline that never reaches a terminal state
          // (worker died, Bedrock access pending, status stuck at running/none)
          // can't hang runCart forever with the buttons locked. ~20 min @ 5s.
          let attempts = 0;
          const MAX_ATTEMPTS = 240;
          const poll = setInterval(async () => {
            attempts += 1;
            try {
              const st = await getBaselineStatus(sid, key);
              if (st.status === "done" && st.metrics) {
                clearInterval(poll);
                upsertBaseline(baselineRowFromMetrics(sid, st.metrics));
                resolve({ ok: true });
              } else if (st.status === "failed") {
                clearInterval(poll);
                resolve({ ok: false, detail: st.detail ?? "failed" });
              } else if (attempts >= MAX_ATTEMPTS) {
                clearInterval(poll);
                resolve({ ok: false, detail: "timed out waiting for the baseline to finish" });
              }
            } catch (e) {
              clearInterval(poll);
              resolve({ ok: false, detail: e instanceof Error ? e.message : String(e) });
            }
          }, 5000);
        })
        .catch((e) => resolve({ ok: false, detail: e instanceof Error ? e.message : String(e) }));
    });
  }

  function addToCart() {
    if (!cartModelKey) return;
    const m = baselineModels.find((x) => x.key === cartModelKey);
    if (!m) return;
    // "not_started" until Run all is clicked — it's only truly queued once a run
    // is dispatched (showing "queued" before that misled users into thinking it
    // was already running).
    setCart((prev) => ({ ...prev, [m.key]: { label: m.label, provider: m.provider, status: "not_started" } }));
    setCartModelKey(null);
  }

  function removeFromCart(key: string) {
    setCart((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }

  // Reset a finished cart entry so the next "Run all" re-scores it on the current
  // split (runCart skips 'done' rows, so without this a baseline can't be re-run).
  function rerunCartEntry(key: string) {
    setCart((prev) =>
      prev[key] ? { ...prev, [key]: { ...prev[key], status: "not_started", detail: undefined } } : prev,
    );
  }

  // Run every queued model SEQUENTIALLY; each row goes running → done/error
  // independently (one failure never blocks the rest).
  async function runCart() {
    if (!splitId) return;
    const sid = splitId;
    const keys = Object.keys(cart);
    if (keys.length === 0) return;
    setRunningAll(true);
    // Mark everything not-yet-done as queued up front, so closing the dialog and
    // reopening shows the right picture (the loop below advances each to
    // running → done/error; the poll updates `cart` regardless of the dialog).
    setCart((p) => {
      const next = { ...p };
      for (const key of keys) if (next[key].status !== "done") next[key] = { ...next[key], status: "queued" };
      return next;
    });
    for (const key of keys) {
      if (cart[key]?.status === "done") continue;  // don't re-run finished ones
      setCart((p) => ({ ...p, [key]: { ...p[key], status: "running", detail: undefined } }));
      const res = await runOneBaseline(sid, key);
      setCart((p) => ({
        ...p,
        [key]: { ...p[key], status: res.ok ? "done" : "error", detail: res.ok ? undefined : res.detail },
      }));
    }
    setRunningAll(false);
  }

  // Add/replace a baseline row by model (so re-running one updates in place).
  function upsertBaseline(row: LeaderboardRow) {
    setBaselineRows((prev) => {
      const without = prev.filter((b) => b.evalJob !== row.evalJob);
      return [...without, row];
    });
  }

  async function doInterpret() {
    if (!splitId) return;
    setInterpretLoading(true);
    setInterpretResult(null);
    try {
      const r = await interpretLeaderboard(splitId, priorities);
      setInterpretResult(r);
    } catch (e) {
      setInterpretResult({ recommendation: "", reasoning: "", error: e instanceof Error ? e.message : String(e) });
    } finally {
      setInterpretLoading(false);
    }
  }

  // Stable colour index per model, keyed on model name in a FIXED order (rows as
  // they arrive from the server, baselines first) — NOT the sorted display order.
  // So a model keeps its colour when you re-rank, and the judge radar (which is
  // metric-independent) never recolours on a Rank-by change. (Was: index into the
  // re-sorted `displayed`, which shifted every colour on each sort — the bug.)
  const modelColorIdx = useMemo(() => {
    const m = new Map<string, number>();
    let n = 0;
    for (const r of [...baselineRows, ...rows]) {
      if (!m.has(r.model)) m.set(r.model, n++);
    }
    return m;
  }, [rows, baselineRows]);

  // Rank: baselines pinned on top, model rows sorted by the chosen metric (desc).
  const ranked = useMemo(() => {
    const sorted = [...rows].sort((a, b) => {
      const av = (a[rankBy] as number | null) ?? -Infinity;
      const bv = (b[rankBy] as number | null) ?? -Infinity;
      return bv - av;
    });
    sorted.forEach((r, i) => ((r as LeaderboardRow & { _winner?: boolean })._winner = i === 0));
    return [...baselineRows, ...sorted];
  }, [rows, rankBy, baselineRows]);

  const displayed = useMemo(() => {
    const t = filterText.trim().toLowerCase();
    if (!t) return ranked;
    return ranked.filter((r) => r.isBaseline || r.model.toLowerCase().includes(t));
  }, [ranked, filterText]);

  // Fetch known LLM-judge results for the displayed real (non-baseline) rows — a
  // baseline's synthetic evalJob has no predictions in S3, so skip it.
  const judgeableJobs = displayed.filter((r) => !r.isBaseline && r.evalJob).map((r) => r.evalJob);
  useEffect(() => {
    judgeableJobs.forEach((ej) => {
      getJudge(ej)
        .then((js) => setJudges((m) => ({ ...m, [ej]: js })))
        .catch(() => {});
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [judgeableJobs.join(",")]);

  // Poll any running judges until they settle (same cadence as RacesPage).
  useEffect(() => {
    const running = Object.values(judges).filter((j) => j.status === "running");
    if (running.length === 0) {
      if (judgePollRef.current) clearInterval(judgePollRef.current);
      return;
    }
    judgePollRef.current = setInterval(() => {
      running.forEach((j) =>
        getJudge(j.evalJob)
          .then((js) => setJudges((m) => ({ ...m, [js.evalJob]: js })))
          .catch(() => {})
      );
    }, 5000);
    return () => {
      if (judgePollRef.current) clearInterval(judgePollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.entries(judges).map(([k, v]) => `${k}:${v.status}`).join(",")]);

  const winnerModel = useMemo(() => {
    const top = [...rows].sort((a, b) => {
      const av = (a[rankBy] as number | null) ?? -Infinity;
      const bv = (b[rankBy] as number | null) ?? -Infinity;
      return bv - av;
    })[0];
    return top ?? null;
  }, [rows, rankBy]);

  // Models with a completed judge + per-dimension breakdown → radar inputs. colorIdx
  // is the STABLE per-model index (modelColorIdx), so a model is the same colour in
  // the scatter, radar, and table — and re-ranking never recolours the radar.
  const judgeRadarModels: JudgeRadarModel[] = useMemo(() => {
    const out: JudgeRadarModel[] = [];
    displayed.forEach((r) => {
      if (r.isBaseline || !r.evalJob) return;
      const j = judges[r.evalJob];
      const dims = j?.status === "done" ? j.result?.dimensions : undefined;
      if (dims && Object.keys(dims).length > 0) {
        out.push({ model: r.model, colorIdx: modelColorIdx.get(r.model) ?? 0, dimensions: dims });
      }
    });
    return out;
  }, [displayed, judges, modelColorIdx]);

  const splitOptions = splits.map((s) => ({
    value: s.splitId,
    label: s.name || s.splitId,
    description: `${s.name ? s.splitId + " · " : ""}${s.evalJobs} eval job(s) · latest ${s.latest.slice(0, 16)}`,
  }));

  const rankLabel = RANK_OPTIONS.find((o) => o.value === rankBy)?.label;

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Compare models on ONE held-out test set (same rows + decoding for every candidate). Add frontier-model baselines (Haiku/Sonnet/Opus), then ask the AI advisor which model to ship."
        >
          Leaderboard
        </Header>
      }
    >
      <SpaceBetween size="l">
        <Container>
          <ColumnLayout columns={2}>
            <FormField
              label="Test set (split)"
              description="A leaderboard is only valid within one test set. Only splits that have completed evaluations are listed."
            >
              <Select
                selectedOption={
                  splitId
                    ? splitOptions.find((o) => o.value === splitId) ?? {
                        value: splitId,
                        label: splits.find((s) => s.splitId === splitId)?.name || splitId,
                      }
                    : null
                }
                onChange={({ detail }) => setSplitId(detail.selectedOption.value ?? null)}
                options={splitOptions}
                placeholder="Choose a test set"
                empty="No evals yet — run a fine-tune + eval first"
              />
            </FormField>
            <FormField
              label="Rank by"
              description="Which metric decides the winner + sort order."
              constraintText={
                recommendedRankKey && rankBy === recommendedRankKey
                  ? `Defaulted from dataset investigation (recommended ${selectedSplit?.recommendedRankMetric}).`
                  : recommendedRankKey
                  ? `Dataset investigation recommended ${selectedSplit?.recommendedRankMetric}.`
                  : selectedSplit?.recommendedRankMetric
                  ? `Dataset investigation recommended ${selectedSplit.recommendedRankMetric}, which has no leaderboard column — pick the closest metric.`
                  : undefined
              }
            >
              <Select
                selectedOption={RANK_OPTIONS.find((o) => o.value === rankBy) ?? null}
                onChange={({ detail }) => setRankBy(detail.selectedOption.value as RankKey)}
                options={RANK_OPTIONS.map((o) => ({
                  value: o.value,
                  label: o.label,
                  description: o.value === recommendedRankKey ? "recommended" : undefined,
                }))}
              />
            </FormField>
          </ColumnLayout>
        </Container>

        {/* Stats for the selected test set — at-a-glance context for the leaderboard. */}
        {splitId && (
          <Container
            header={
              <Header variant="h3">
                Dataset · {selectedSplit?.name || selectedDataset?.name || splitId}
              </Header>
            }
          >
            <ColumnLayout columns={4} variant="text-grid">
              <div>
                <Box variant="awsui-key-label">Test rows (eval)</Box>
                <Box>{selectedDataset?.evalRows ?? "—"}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Train rows</Box>
                <Box>{selectedDataset?.evalOnly ? "eval-only" : selectedDataset?.trainRows ?? "—"}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Validation</Box>
                <Box>{selectedDataset?.hasVal ? `${selectedDataset?.valRows ?? "?"} rows` : "none"}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Eval runs</Box>
                <Box>{selectedSplit?.evalJobs ?? "—"}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Source</Box>
                <Box>
                  {selectedDataset?.source === "huggingface" && selectedDataset?.hfDataset ? (
                    <>🤗 {selectedDataset.hfDataset}</>
                  ) : (
                    selectedDataset?.source || "—"
                  )}
                </Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Recommended metric</Box>
                <Box>
                  {selectedSplit?.recommendedRankMetric ? (
                    <Badge color="blue">{selectedSplit.recommendedRankMetric}</Badge>
                  ) : (
                    "—"
                  )}
                </Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Split id</Box>
                <Box fontSize="body-s" color="text-status-inactive">{splitId}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Frontier baseline</Box>
                <Box>{baselineRows.length > 0 ? `${baselineRows.length} run` : "none yet"}</Box>
              </div>
            </ColumnLayout>
          </Container>
        )}

        {winnerModel && (() => {
          // 95% Wilson margin on the winner's ranking metric, from its eval row count.
          const ci = wilsonInterval(winnerModel[rankBy] as number | null, winnerModel.count);
          return (
            <Alert type="success" header={`Best model: ${winnerModel.model}`}>
              Top {rankLabel} on split {splitId}: <b>{pct(winnerModel[rankBy] as number | null)}</b>
              {ci && <> ({marginLabel(ci.half)})</>}
              {baselineRows.map((b) => (
                <span key={b.evalJob}>
                  {" "}· {b.model.replace(" (baseline)", "")}:{" "}
                  <b>{pct(b[rankBy] as number | null)}</b>
                </span>
              ))}
              .
            </Alert>
          );
        })()}

        {/* Small-sample honesty: when the test set is small, the scores carry a real
            margin of error, so close rankings shouldn't be over-read. Surfaced once
            above the visuals/table. evalN is the per-row eval count (same for every
            model on this split). */}
        {(() => {
          const evalN = displayed.find((r) => !r.isBaseline)?.count ?? null;
          if (!isSmallSample(evalN)) return null;
          const ci = wilsonInterval(winnerModel?.[rankBy] as number | null, evalN);
          return (
            <Alert type="warning" header="Small test set — read rankings with care">
              This test set has <b>{evalN}</b> eval rows (under {SMALL_SAMPLE_THRESHOLD}), so each
              score carries a ~{ci ? marginLabel(ci.half) : "wide"} 95% margin of error. Treat models
              within that margin of each other as <b>tied</b>, not ranked. Add more eval rows to
              tighten the comparison.
            </Alert>
          );
        })()}

        {/* Decision visuals above the numbers table — quality-vs-cost scatter, and
            (once a model is judged) the per-dimension judge radar beside it. Both use
            the SAME ranked rows + palette so a model reads as one colour everywhere. */}
        {displayed.length > 0 && (
          <ColumnLayout columns={judgeRadarModels.length > 0 ? 2 : 1}>
            <QualityCostScatter
              rows={displayed}
              rankBy={rankBy}
              rankLabel={rankLabel ?? rankBy}
              colorIdxFor={(m) => modelColorIdx.get(m) ?? 0}
            />
            {judgeRadarModels.length > 0 && <JudgeRadar models={judgeRadarModels} />}
          </ColumnLayout>
        )}

        <Table
          variant="container"
          loading={loading}
          loadingText="Loading eval results…"
          resizableColumns
          stickyColumns={{ first: 1 }}
          header={
            <Header
              counter={`(${rows.length} model${rows.length === 1 ? "" : "s"})`}
              actions={
                <SpaceBetween direction="horizontal" size="xs">
                  <Button iconName="refresh" onClick={() => refresh()} loading={loading}>
                    Refresh
                  </Button>
                  <Button
                    iconName="gen-ai"
                    onClick={() => setInterpretOpen(true)}
                    disabled={!splitId || rows.length === 0}
                  >
                    Which model to ship?
                  </Button>
                  {/* Run baseline → opens the provider-grouped picker modal */}
                  <Button
                    variant="primary"
                    iconName="add-plus"
                    onClick={() => setBaselineModalOpen(true)}
                    loading={runningAll}
                    disabled={!splitId}
                  >
                    Run baselines
                  </Button>
                </SpaceBetween>
              }
              description={
                splitId
                  ? `Latest evaluation per model on split ${splitId}. Baselines run on this same test set; add several to compare frontier tiers.`
                  : "Select a test set above."
              }
            >
              Model comparison
            </Header>
          }
          filter={
            <TextFilter
              filteringText={filterText}
              filteringPlaceholder="Find a model"
              onChange={({ detail }) => setFilterText(detail.filteringText)}
              countText={`${displayed.filter((r) => !r.isBaseline).length} model${
                displayed.filter((r) => !r.isBaseline).length === 1 ? "" : "s"
              }`}
            />
          }
          columnDefinitions={[
            {
              id: "model",
              header: "Model",
              minWidth: 180,
              cell: (r) => (
                <SpaceBetween direction="horizontal" size="xs">
                  {r.isBaseline ? <Badge color="blue">{r.model}</Badge> : <span>{r.model}</span>}
                  {(r as LeaderboardRow & { _winner?: boolean })._winner && (
                    <Badge color="green">best</Badge>
                  )}
                </SpaceBetween>
              ),
            },
            {
              id: "engine",
              header: "Engine",
              minWidth: 130,
              // Engine isn't a first-class leaderboard field; derive it from the
              // signals the backend already sends: serverless train cost carries
              // trainInstance="serverless"/trainServerless, and the model label
              // encodes "-serverless" (job-name convention). Baselines have no engine.
              cell: (r) => {
                if (r.isBaseline) return "—";
                const serverless =
                  r.trainInstance === "serverless" ||
                  (r as LeaderboardRow & { trainServerless?: boolean }).trainServerless === true ||
                  /(?:^|-)serverless(?:$|-)/.test(r.model);
                return serverless ? (
                  <Badge color="severity-neutral">Serverless</Badge>
                ) : (
                  <Badge>LLaMA-Factory</Badge>
                );
              },
            },
            { id: "exact", header: colHeader("Exact", "exact"), minWidth: 100, cell: (r) => pct(r.exactMatch) },
            { id: "norm", header: colHeader("Normalized", "norm"), minWidth: 110, cell: (r) => pct(r.normalizedMatch) },
            { id: "contains", header: colHeader("Contains", "contains"), minWidth: 105, cell: (r) => pct(r.containsGold) },
            { id: "f1", header: colHeader("Token F1", "f1"), minWidth: 100, cell: (r) => pct(r.tokenF1) },
            { id: "rouge", header: colHeader("ROUGE-L", "rouge"), minWidth: 100, cell: (r) => pct(r.rougeL) },
            { id: "charf1", header: colHeader("Char F1", "charf1"), minWidth: 100, cell: (r) => pct(r.charF1) },
            // RLAIF judge reward (reference-free) — "—" for gold-overlap rows + baselines.
            { id: "reward", header: colHeader("Judge reward", "reward"), minWidth: 120, cell: (r) => pct(r.rewardMean) },
            // LLM-as-judge (1-5), fetched per real row's eval job; on-demand for the rest.
            {
              id: "judge",
              header: colHeader("LLM judge (1-5)", "judge"),
              minWidth: 120,
              cell: (r) => {
                if (r.isBaseline || !r.evalJob) return "—";
                const j = judges[r.evalJob];
                if (j?.status === "done" && j.result && j.result.judgeScore != null) {
                  const dims = j.result.dimensions;
                  const el = <span><b>{j.result.judgeScore.toFixed(2)}</b> / 5</span>;
                  return dims && Object.keys(dims).length > 0 ? (
                    <Popover header="Judge breakdown (1-5)" triggerType="custom" dismissButton={false} position="top"
                      content={<SpaceBetween size="xxs">{Object.entries(dims).map(([d, v]) => (
                        <div key={d}>{d}: <b>{(v as number).toFixed(2)}</b></div>))}</SpaceBetween>}>
                      {el}
                    </Popover>
                  ) : el;
                }
                if (j?.status === "running") return <Spinner size="normal" />;
                if (j?.status === "failed") return <Box color="text-status-error">failed</Box>;
                return <Button variant="inline-link" iconName="gen-ai" onClick={() => runJudgeFor(r.evalJob)}>Judge</Button>;
              },
            },
            { id: "tps", header: colHeader("Tokens/s", "tps"), minWidth: 100, cell: (r) => numOrDash(r.tokensPerSec) },
            // p50 latency — MEASURED at eval, sent by the backend, previously unshown.
            // The third axis of the quality×cost×latency pitch.
            {
              id: "latency",
              header: colHeader("Latency p50/p90/p99", "latency"),
              minWidth: 150,
              cell: (r) => {
                if (r.p50LatencyMs == null) return "—";
                // Show the tail when we have it; flag a "tail blowup" (p99 ≫ p50) in
                // red — the production-relevant signal a median alone would hide.
                const hasTail = r.p90LatencyMs != null || r.p99LatencyMs != null;
                if (!hasTail) return ms(r.p50LatencyMs); // older eval run: p50 only
                const blowup = r.p99LatencyMs != null && r.p50LatencyMs > 0 && r.p99LatencyMs > r.p50LatencyMs * 5;
                return (
                  <span>
                    {ms(r.p50LatencyMs)} / {ms(r.p90LatencyMs)} /{" "}
                    <Box variant="span" color={blowup ? "text-status-error" : undefined}>{ms(r.p99LatencyMs)}</Box>
                  </span>
                );
              },
            },
            {
              id: "train",
              header: colHeader("Train cost", "train"),
              minWidth: 130,
              cell: (r) =>
                r.isBaseline ? (
                  "n/a"
                ) : r.trainCostIsEstimate ? (
                  <span>
                    {usd(r.trainCostUsd)} <Badge color="blue">spot (est.)</Badge>
                  </span>
                ) : (
                  usd(r.trainCostUsd)
                ),
            },
            {
              id: "serve",
              header: colHeader("Cost / 1k rows", "serve"),
              minWidth: 140,
              cell: (r) =>
                r.isBaseline ? (
                  <span>
                    {usd(r.apiCostPer1k)} <Box variant="small" display="inline">API actual</Box>
                  </span>
                ) : (
                  <span>
                    {usd(r.projectedServeCostPer1k)} <Box variant="small" display="inline">projected</Box>
                  </span>
                ),
            },
          ]}
          items={displayed}
          empty={
            <Box textAlign="center" padding="l">
              {rows.length === 0
                ? "No eval results for this split yet. Run a fine-tune + eval first."
                : "No models match the filter."}
            </Box>
          }
        />

        <Alert type="info" header="How to read this">
          All rows share the <b>same held-out test set</b> (the selected split) and the same
          decoding — that's what makes the comparison fair. <b>Train cost</b> and{" "}
          <b>tokens/s</b> are measured; <b>cost / 1k rows</b> is a <i>projected self-host</i>{" "}
          estimate for fine-tuned models vs. each frontier baseline's <i>actual API</i> price.
          Baseline rows (Haiku/Sonnet/Opus) are foundation-model references for this test set.
        </Alert>
      </SpaceBetween>

      {/* Results-interpreter agent modal */}
      <Modal
        visible={interpretOpen}
        onDismiss={() => setInterpretOpen(false)}
        size="medium"
        header={
          <SpaceBetween direction="horizontal" size="xs">
            <span>Which model should I ship?</span>
            <AgentBadge />
          </SpaceBetween>
        }
      >
        <SpaceBetween size="m">
          <AgentCaption>
            An AI advisor reads this leaderboard (quality × cost × latency, including any frontier
            baselines) and recommends which model to ship for your priorities — advisory, not a deploy.
          </AgentCaption>
          <FormField
            label="Your priorities (optional)"
            description="e.g. 'cost matters most, latency under 200ms, quality within 5% of best'."
          >
            <Textarea
              value={priorities}
              onChange={({ detail }) => setPriorities(detail.value)}
              placeholder="What matters for this deployment?"
              rows={2}
            />
          </FormField>
          <Button variant="primary" onClick={doInterpret} loading={interpretLoading} iconName="gen-ai">
            {interpretResult && !interpretResult.error ? "Re-run recommendation" : "Get recommendation"}
          </Button>

          {interpretLoading && (
            <Box textAlign="center">
              <Spinner /> Analyzing the leaderboard…
            </Box>
          )}

          {interpretResult?.error && <Alert type="error">{interpretResult.error}</Alert>}

          {interpretResult && !interpretResult.error && (
            <Alert type="success" header={`Recommended: ${interpretResult.recommendation}`}>
              <SpaceBetween size="s">
                {interpretResult.ranAt && (
                  <Box variant="small" color="text-body-secondary">
                    Last run {new Date(interpretResult.ranAt).toLocaleString()}
                    {interpretResult.priorities
                      ? ` · priorities: "${interpretResult.priorities}"`
                      : " · no priorities given"}
                  </Box>
                )}
                <Box>{interpretResult.reasoning}</Box>
                {interpretResult.vsBaseline && (
                  <Box>
                    <Box variant="awsui-key-label">vs. baseline</Box>
                    {interpretResult.vsBaseline}
                  </Box>
                )}
                {interpretResult.runnerUp && (
                  <Box>
                    <Box variant="awsui-key-label">Runner-up</Box>
                    {interpretResult.runnerUp}
                  </Box>
                )}
                {interpretResult.caveats && interpretResult.caveats.length > 0 && (
                  <Box>
                    <Box variant="awsui-key-label">Caveats</Box>
                    <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                      {interpretResult.caveats.map((c, i) => (
                        <li key={i}>
                          <Box variant="span" fontSize="body-s">
                            {c}
                          </Box>
                        </li>
                      ))}
                    </ul>
                  </Box>
                )}
                <Box fontSize="body-s" color="text-status-inactive">
                  Advisory — your call. Based on the current leaderboard for this split.
                </Box>
              </SpaceBetween>
            </Alert>
          )}
        </SpaceBetween>
      </Modal>

      {/* Multi-baseline cart: pick provider + model, Add to a queue, then Run all.
          Each runs as its own job (Bedrock Converse) sequentially; the table shows
          per-model status + any API error (e.g. Bedrock model-access denied). */}
      <Modal
        visible={baselineModalOpen}
        // Always dismissable — the baselines run server-side and the poll keeps
        // updating the cart even with the dialog closed, so the user can leave and
        // reopen anytime to check progress (it no longer freezes during a run).
        onDismiss={() => setBaselineModalOpen(false)}
        header="Run frontier baselines"
        size="large"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="link" onClick={() => setBaselineModalOpen(false)}>
                Close
              </Button>
              <Button
                variant="primary"
                onClick={runCart}
                loading={runningAll}
                disabled={Object.keys(cart).length === 0 || !splitId || runningAll}
              >
                {runningAll ? "Running…" : `Run all (${Object.keys(cart).length})`}
              </Button>
            </SpaceBetween>
          </Box>
        }
      >
        <SpaceBetween size="m">
          <Box variant="p" color="text-status-inactive">
            Queue one or more frontier models to score on this test set — the
            "buy-instead-of-fine-tune" comparison. Each runs via Bedrock and needs model
            access enabled in your account (failures show in the table).
          </Box>
          {runningAll && (
            <Alert type="info">
              Baselines are running server-side — you can close this dialog and come back; progress
              keeps updating here and the leaderboard fills in as each finishes.
            </Alert>
          )}

          {/* provider → model → Add */}
          <ColumnLayout columns={3}>
            <FormField label="Provider">
              <Select
                selectedOption={
                  cartProvider
                    ? { value: cartProvider, label: cartProvider, iconSvg: <ProviderIcon provider={cartProvider} size={16} /> }
                    : null
                }
                onChange={({ detail }) => {
                  setCartProvider(detail.selectedOption.value!);
                  setCartModelKey(null);
                }}
                placeholder="Choose provider"
                options={[...new Set(baselineModels.map((m) => m.provider))].map((p) => ({
                  value: p,
                  label: p,
                  iconSvg: <ProviderIcon provider={p} size={16} />,
                }))}
              />
            </FormField>
            <FormField label="Model">
              <Select
                selectedOption={
                  cartModelKey
                    ? {
                        value: cartModelKey,
                        label: baselineModels.find((m) => m.key === cartModelKey)?.label ?? cartModelKey,
                        iconSvg: cartProvider ? <ProviderIcon provider={cartProvider} size={16} /> : undefined,
                      }
                    : null
                }
                onChange={({ detail }) => setCartModelKey(detail.selectedOption.value!)}
                placeholder={cartProvider ? "Choose model" : "Pick a provider first"}
                disabled={!cartProvider}
                options={baselineModels
                  .filter((m) => m.provider === cartProvider)
                  .map((m) => ({
                    value: m.key,
                    label: m.label,
                    iconSvg: <ProviderIcon provider={m.provider} size={16} />,
                    // Bedrock list price, shown per MILLION tokens (values stored per-1K → ×1000).
                    description: `$${(m.inPer1k * 1000).toFixed(2)} in / $${(m.outPer1k * 1000).toFixed(2)} out per 1M tok`,
                  }))}
              />
            </FormField>
            <FormField label=" ">
              <Button iconName="add-plus" onClick={addToCart} disabled={!cartModelKey}>
                Add
              </Button>
            </FormField>
          </ColumnLayout>

          {/* the queued-models table with per-row status/error + remove */}
          <Table
            variant="embedded"
            items={Object.entries(cart).map(([key, v]) => ({ key, ...v }))}
            empty={<Box textAlign="center" color="text-status-inactive">No models queued — add one above.</Box>}
            columnDefinitions={[
              {
                id: "provider",
                header: "Provider",
                cell: (r) => (
                  <SpaceBetween direction="horizontal" size="xs" alignItems="center">
                    <ProviderIcon provider={r.provider} size={18} />
                    <span>{r.provider}</span>
                  </SpaceBetween>
                ),
              },
              { id: "model", header: "Model", cell: (r) => r.label },
              {
                id: "status",
                header: "Status",
                cell: (r) =>
                  r.status === "done" ? (
                    <StatusIndicator type="success">done</StatusIndicator>
                  ) : r.status === "running" ? (
                    <StatusIndicator type="in-progress">running…</StatusIndicator>
                  ) : r.status === "error" ? (
                    <StatusIndicator type="error">error</StatusIndicator>
                  ) : r.status === "queued" ? (
                    <StatusIndicator type="pending">queued</StatusIndicator>
                  ) : (
                    <StatusIndicator type="stopped">not started</StatusIndicator>
                  ),
              },
              {
                id: "detail",
                header: "Error / note",
                cell: (r) =>
                  r.detail ? (
                    <Box variant="small" color="text-status-error">{r.detail}</Box>
                  ) : (
                    "—"
                  ),
              },
              {
                id: "actions",
                header: "",
                cell: (r) => (
                  <SpaceBetween direction="horizontal" size="xs">
                    {r.status === "done" && (
                      <Button
                        variant="inline-link"
                        iconName="refresh"
                        disabled={runningAll}
                        onClick={() => rerunCartEntry(r.key)}
                      >
                        Re-run
                      </Button>
                    )}
                    <Button
                      variant="inline-link"
                      iconName="remove"
                      disabled={runningAll}
                      onClick={() => removeFromCart(r.key)}
                    >
                      Remove
                    </Button>
                  </SpaceBetween>
                ),
              },
            ]}
          />
        </SpaceBetween>
      </Modal>
    </ContentLayout>
  );
}
