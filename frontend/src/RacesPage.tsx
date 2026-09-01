// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useMemo, useRef, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Spinner from "@cloudscape-design/components/spinner";
import Popover from "@cloudscape-design/components/popover";
import Header from "@cloudscape-design/components/header";
import Container from "@cloudscape-design/components/container";
import Tabs from "@cloudscape-design/components/tabs";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Table, { type TableProps } from "@cloudscape-design/components/table";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Badge from "@cloudscape-design/components/badge";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Select from "@cloudscape-design/components/select";
import FormField from "@cloudscape-design/components/form-field";
import Toggle from "@cloudscape-design/components/toggle";
import TextFilter from "@cloudscape-design/components/text-filter";
import { useCollection } from "@cloudscape-design/collection-hooks";
import Modal from "@cloudscape-design/components/modal";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Link from "@cloudscape-design/components/link";
import { colHeader } from "./metricDefs";
import { AgentBadge, AgentCaption } from "./AgentBadge";
import { LossChart } from "./LossChart";
import { RewardCurve } from "./RewardCurve";
import { RaceResultBars } from "./RaceResultBars";
import { InvestigateDataset } from "./InvestigateDataset";
import { useNotify, errText } from "./notifications";

import {
  archiveRace,
  exportBundleUrl,
  exportModelInfo,
  getJudge,
  getRace,
  getDatasets,
  listRaces,
  retryRaceEntry,
  startJudge,
  triageRaceEntry,
  type Dataset,
  type ExportInfo,
  type JudgeStatus,
  type RaceEntry,
  type RaceResult,
  type RaceSummary,
  type TriageResult,
} from "./api";

// Derived per-race fields the table sorts/filters on (so the collection has
// flat, comparable values rather than nested states/models).
type RaceRow = RaceSummary & {
  _modelCount: number;
  _doneCount: number;
  _failedCount: number;
  _overall: string; // running | done | failed | mixed
};

function overallStatus(states: Record<string, string>): string {
  const vals = Object.values(states);
  if (vals.length === 0) return "—";
  const done = vals.filter((s) => s === "done").length;
  const failed = vals.filter((s) => s === "failed").length;
  const active = vals.filter((s) => s !== "done" && s !== "failed").length;
  if (active > 0) return "running";
  if (failed === vals.length) return "failed";
  if (done === vals.length) return "done";
  return "mixed";
}

const STATUS_FILTER_OPTIONS = [
  { value: "all", label: "All statuses" },
  { value: "running", label: "Running" },
  { value: "done", label: "Done" },
  { value: "failed", label: "Failed" },
  { value: "mixed", label: "Mixed (some failed)" },
];

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

// The untrained base model's score on a given metric (for the base→fine-tuned
// lift columns). base_metrics is null until the base eval completes, {} if it
// failed, so this returns null in both cases.
function baseScore(e: RaceEntry, metric: string): number | null {
  const m = e.base_metrics as Record<string, number | null> | null | undefined;
  if (!m) return null;
  const v = m[metric];
  return typeof v === "number" ? v : null;
}

function stateIndicator(state: RaceEntry["state"]) {
  switch (state) {
    case "done":
      return <StatusIndicator type="success">Done</StatusIndicator>;
    case "failed":
      return <StatusIndicator type="error">Failed</StatusIndicator>;
    case "launching":
      return <StatusIndicator type="in-progress">Launching…</StatusIndicator>;
    case "training":
      return <StatusIndicator type="in-progress">Training</StatusIndicator>;
    case "evaluating":
      return <StatusIndicator type="in-progress">Evaluating</StatusIndicator>;
    default:
      return <StatusIndicator type="pending">{state}</StatusIndicator>;
  }
}

function isTerminal(s: string) {
  return s === "done" || s === "failed";
}

// Ranking metrics the user can pick to decide the winner (higher = better).
const RANK_METRIC_OPTIONS = [
  { value: "token_f1", label: "Token F1" },
  { value: "rouge_l", label: "ROUGE-L" },
  { value: "char_f1", label: "Char F1" },
  { value: "contains_gold", label: "Contains gold" },
  { value: "normalized_match", label: "Normalized match" },
  { value: "exact_match", label: "Exact match" },
  // RLAIF only: the AI-judge reward (reference-free). The server forces this for
  // an RLAIF race since its held-out set is prompt-only (no gold to overlap).
  { value: "reward_mean", label: "Judge reward" },
];

export function RacesPage({
  activeRaceId,
  onCloneRun,
}: {
  activeRaceId: string | null;
  // "Clone & edit": open the Fine-Tune builder pre-filled with this run's config.
  onCloneRun?: (raceId: string) => void;
}) {
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [selected, setSelected] = useState<string | null>(activeRaceId);
  const [detail, setDetail] = useState<RaceResult | null>(null);
  // Full dataset list (loaded once) → resolve splitId → name/shape for the table,
  // and back the in-page "dataset details" modal opened from a Dataset link.
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoaded, setDatasetsLoaded] = useState(false);
  // The dataset whose details modal is open (opened in-page, no navigation).
  const [datasetDetail, setDatasetDetail] = useState<Dataset | null>(null);
  const [rankMetric, setRankMetric] = useState("token_f1");
  // Human-readable name of the current ranking metric (for the "Fine-tune lift"
  // header, so the points are unambiguously tied to a metric).
  const rankMetricLabel =
    RANK_METRIC_OPTIONS.find((o) => o.value === rankMetric)?.label ?? rankMetric;
  const [showArchived, setShowArchived] = useState(false);
  const { notify } = useNotify();
  // Starts true so the first paint is a spinner, not "No runs yet" — the list
  // fetch hasn't returned, so an empty table would falsely read as "no runs".
  const [loading, setLoading] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Bumped after a retry/resume to restart the SINGLE self-stopping poll effect
  // (instead of spawning a second interval that never checks for terminal state).
  const [pollNonce, setPollNonce] = useState(0);

  function refreshList() {
    setLoading(true);  // show the spinner on the Refresh button + list while refetching
    listRaces()
      .then(setRaces)
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoading(false));
  }

  async function toggleArchive(raceId: string, archived: boolean) {
    try {
      await archiveRace(raceId, archived);
      // If we just archived the open race, close the detail pane.
      if (archived && selected === raceId) setSelected(null);
      notify({ type: "success", content: archived ? "Race archived." : "Race restored." });
      refreshList();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    }
  }

  useEffect(refreshList, []);

  // Load datasets once to resolve splitId → name (include archived so a run on a
  // since-archived dataset still shows its name).
  useEffect(() => {
    getDatasets(true)
      .then((ds) => setDatasets(ds))
      .catch(() => {})
      .finally(() => setDatasetsLoaded(true));
  }, []);

  const datasetById = useMemo(() => {
    const map: Record<string, { name: string; shape: string }> = {};
    for (const d of datasets) map[d.splitId] = { name: d.name || d.splitId, shape: d.shape || "sft" };
    return map;
  }, [datasets]);

  // Friendly dataset label. Until the dataset list has loaded, show a neutral
  // placeholder rather than flashing the raw splitId hash (then the real name).
  const datasetLabel = (splitId: string) =>
    datasetById[splitId]?.name ?? (datasetsLoaded ? splitId : "…");

  // Open the dataset DETAILS modal IN-PAGE (no navigation away from Runs). Falls
  // back to a minimal stub if the dataset list hasn't loaded yet.
  function openDatasetDetails(splitId: string) {
    const d = datasets.find((x) => x.splitId === splitId) ?? ({ splitId } as Dataset);
    setDatasetDetail(d);
  }

  // When a race is launched elsewhere, auto-select it.
  useEffect(() => {
    if (activeRaceId) {
      setSelected(activeRaceId);
      refreshList();
    }
  }, [activeRaceId]);

  // Current rank metric in a ref so the poll can send it WITHOUT re-subscribing the
  // effect on every dropdown change (which would refetch + feel sluggish). Changing
  // "Rank by" now re-ranks client-side (see rankedDetail); the server is only hit on
  // the initial load + the 15s poll while a run is live.
  const rankMetricRef = useRef(rankMetric);
  rankMetricRef.current = rankMetric;

  // Poll the selected race's detail. NOT keyed on rankMetric — the re-rank is
  // client-side. The server still resolves the objective default / RLAIF override on
  // the FIRST fetch, which we adopt into the dropdown.
  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    let first = true;
    const tick = () =>
      getRace(selected, rankMetricRef.current)
        .then((r) => {
          setDetail(r);
          // On the FIRST load, adopt the server's resolved metric (objective default,
          // or the forced RLAIF judge reward) so the control matches what's shown.
          // Later user changes are client-side, so we don't override them here.
          if (first && r.rankMetric && r.rankMetric !== rankMetricRef.current) {
            setRankMetric(r.rankMetric);
          }
          first = false;
          if (r.entries.every((e) => isTerminal(e.state)) && pollRef.current) {
            clearInterval(pollRef.current);
          }
        })
        .catch((e) => notify({ type: "error", content: errText(e) }));
    tick();
    pollRef.current = setInterval(tick, 15000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [selected, pollNonce]);

  // Keyed on `${entryId}:${action}` so only the clicked button shows a spinner —
  // a 'Resume from checkpoint' click must not also spin the 'Retry fresh' button.
  const [retrying, setRetrying] = useState<string | null>(null);
  async function retry(modelId: string, resume = false) {
    if (!selected) return;
    setRetrying(`${modelId}:${resume ? "resume" : "fresh"}`);
    try {
      const r = await retryRaceEntry(selected, modelId, resume);
      setDetail(r);
      // Restart the SINGLE self-stopping poll (the retried entry is in progress
      // again) by bumping the nonce the poll effect depends on — don't spawn a
      // second interval (the old code's interval had no terminal check → polled
      // forever after the run finished).
      setPollNonce((n) => n + 1);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setRetrying(null);
    }
  }

  // Failure-triage agent: diagnose a FAILED entry + propose a concrete fix.
  const [triaging, setTriaging] = useState<string | null>(null);
  const [triage, setTriage] = useState<{ model: string; result: TriageResult } | null>(null);
  async function diagnose(modelId: string) {
    if (!selected) return;
    setTriaging(modelId);
    try {
      const r = await triageRaceEntry(selected, modelId);
      setTriage({ model: modelId, result: r });
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setTriaging(null);
    }
  }

  // Export-to-your-own-account: fetch the manifest (license-driven adapter vs
  // merged + a presigned weights URL) and show it in a modal with a download.
  const [exporting, setExporting] = useState<string | null>(null);
  const [exportInfo, setExportInfo] = useState<ExportInfo | null>(null);
  const [downloading, setDownloading] = useState(false);
  // License acceptance for a gated full/freeze fine-tune (Option B): the merged
  // weights embed the gated base, so the download is withheld until the user
  // accepts the base license. Tracks the modelId the user has accepted for.
  const [licenseAccepted, setLicenseAccepted] = useState<string | null>(null);
  async function openExport(modelId: string, accepted = false) {
    if (!selected) return;
    setExporting(modelId);
    try {
      setExportInfo(await exportModelInfo(selected, modelId, accepted));
      setLicenseAccepted(accepted ? modelId : null);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setExporting(null);
    }
  }

  // Download the deploy bundle via fetch→blob (not a plain link) so the auth
  // Bearer token — attached by the global fetch wrapper — rides along in prod.
  async function downloadBundle(modelId: string) {
    if (!selected) return;
    setDownloading(true);
    try {
      const res = await fetch(exportBundleUrl(selected, modelId, licenseAccepted === modelId));
      if (!res.ok) throw new Error(`bundle download failed (${res.status})`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `slm-deploy-${modelId}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setDownloading(false);
    }
  }

  // LLM-as-judge state, keyed by eval job. Winner is auto-judged server-side;
  // other models get an on-demand "Judge" button. We fetch known results when
  // the detail loads, and poll while any judge is running.
  const [judges, setJudges] = useState<Record<string, JudgeStatus>>({});
  const judgePollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // On detail load, fetch judge status for every entry that has an eval job.
    if (!detail) return;
    detail.entries
      .filter((e) => e.eval_job)
      .forEach((e) => {
        getJudge(e.eval_job as string)
          .then((js) => setJudges((m) => ({ ...m, [e.eval_job as string]: js })))
          .catch(() => {});
      });
  }, [detail?.raceId, detail?.entries.map((e) => e.eval_job).join(",")]);

  // Poll any running judges until they settle.
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
  }, [Object.entries(judges).map(([k, v]) => `${k}:${v.status}`).join(",")]);

  async function runJudgeFor(evalJob: string) {
    setJudges((m) => ({ ...m, [evalJob]: { evalJob, status: "running", result: null } }));
    try {
      const r = await startJudge(evalJob);
      if (r.status === "done" && r.result) {
        setJudges((m) => ({ ...m, [evalJob]: { evalJob, status: "done", result: r.result! } }));
      }
      // else: status=running → the poll effect takes over.
    } catch (e) {
      notify({ type: "error", content: errText(e) });
      setJudges((m) => ({ ...m, [evalJob]: { evalJob, status: "failed", result: null } }));
    }
  }

  // CLIENT-SIDE re-rank: when the user changes "Rank by", recompute rankScore +
  // isWinner from the already-loaded per-entry metrics instead of refetching the
  // race (the old behaviour — a full server round-trip per dropdown change, which
  // felt sluggish). Mirrors the backend rank_entries: rankScore = metrics[metric];
  // winner = highest among DONE entries. The server still owns the FIRST load +
  // polling (and the objective default / RLAIF override) — this only re-sorts what's
  // already here. reward_mean lives on metrics too, so it works for every metric.
  const rankedDetail = useMemo(() => {
    if (!detail) return null;
    const entries = detail.entries.map((e) => {
      const score = (e.metrics as unknown as Record<string, unknown> | null)?.[rankMetric];
      return { ...e, rankScore: typeof score === "number" ? score : null, isWinner: false };
    });
    // Winner = best score among done entries (stable, deterministic order).
    let bestI = -1;
    entries.forEach((e, i) => {
      if (e.state === "done" && e.rankScore != null &&
          (bestI < 0 || (e.rankScore as number) > (entries[bestI].rankScore as number))) {
        bestI = i;
      }
    });
    if (bestI >= 0) entries[bestI].isWinner = true;
    return { ...detail, entries, rankMetric };
  }, [detail, rankMetric]);

  const winner = rankedDetail?.entries.find((e) => e.isWinner) ?? null;
  const doneCount = rankedDetail?.entries.filter((e) => e.state === "done").length ?? 0;
  const allTerminal =
    rankedDetail != null && rankedDetail.entries.every((e) => isTerminal(e.state));

  // The run's fine-tuning TECHNIQUE. A race is single-objective (the dataset shape
  // gates the stage across all entries), so the detail table can show a metric set
  // tailored to the technique — unlike the cross-model leaderboard, which must keep
  // columns stable for comparison. Prefer the entries' persisted hp.stage; fall
  // back to the dataset shape. (sft|dpo|kto|rlvr|rlaif)
  const technique = useMemo<"sft" | "dpo" | "kto" | "rlvr" | "rlaif">(() => {
    const stage = detail?.entries.map((e) => e.hp?.stage as string | undefined).find(Boolean);
    const shape = detail ? datasetById[detail.splitId]?.shape : undefined;
    const v = (stage || (shape === "preference" ? "dpo" : shape) || "sft") as string;
    return (["sft", "dpo", "kto", "rlvr", "rlaif"].includes(v) ? v : "sft") as
      "sft" | "dpo" | "kto" | "rlvr" | "rlaif";
  }, [detail, datasetById]);

  // Which metric COLUMNS to show per technique (Runs detail only). Gold-overlap
  // metrics are honest for SFT (constrained tasks) + RLVR (real verifiable gold),
  // misleading for DPO/KTO (scored vs ONE acceptable answer), and N/A for RLAIF
  // (no gold). The LLM judge is the real signal for preference/open-ended runs.
  const VISIBLE_COLS: Record<string, Set<string>> = {
    // SFT: the full gold-overlap + task-aware set (task-aware cells self-blank).
    // base/tuned/lift form the before→after→delta triplet on the ranking metric.
    sft: new Set(["base", "tuned", "lift", "exact", "norm", "contains", "f1", "rouge", "charf1",
                  "labelAcc", "numeric", "jsonValid", "json", "jsonKeys", "scaffold", "len", "judge"]),
    // RLVR: a real verifiable gold — numeric/extracted overlap is legit; keep judge.
    rlvr: new Set(["base", "tuned", "lift", "exact", "norm", "contains", "f1", "numeric",
                   "jsonValid", "json", "jsonKeys", "scaffold", "len", "judge"]),
    // DPO/KTO: lead with the judge; overlap kept but de-emphasized (it's vs ONE
    // acceptable answer). Drop the task-aware JSON/numeric/label columns (rarely
    // apply to free-text preference data) to cut noise. DPO also gets the
    // preference-native win-rate (KTO has no rejected pair → N/A).
    dpo: new Set(["judge", "winrate", "f1", "rouge", "charf1", "contains", "scaffold", "len"]),
    kto: new Set(["judge", "f1", "rouge", "charf1", "contains", "scaffold", "len"]),
    // RLAIF: no gold → only the judge reward is meaningful; hide all 13 overlap/lift
    // columns (they're structurally always "—").
    rlaif: new Set(["reward", "judge"]),
  };
  const showCol = (id: string) => VISIBLE_COLS[technique].has(id);

  // Default view hides archived races; the toggle reveals them.
  const archivedCount = races.filter((r) => r.archived).length;
  const [statusFilter, setStatusFilter] = useState("all");

  // Build sortable/filterable rows (flatten nested states/models into scalars).
  const raceRows: RaceRow[] = useMemo(
    () =>
      races
        .filter((r) => (showArchived ? true : !r.archived))
        .filter((r) => statusFilter === "all" || overallStatus(r.states) === statusFilter)
        .map((r) => {
          const states = Object.values(r.states);
          return {
            ...r,
            _modelCount: r.models.length,
            _doneCount: states.filter((s) => s === "done").length,
            _failedCount: states.filter((s) => s === "failed").length,
            _overall: overallStatus(r.states),
          };
        }),
    [races, showArchived, statusFilter]
  );

  const { items, collectionProps, filterProps, filteredItemsCount } = useCollection(raceRows, {
    filtering: {
      // Match against name, id, split, and model ids.
      filteringFunction: (item, text) => {
        const hay = `${item.name ?? ""} ${item.raceId} ${item.splitId} ${item.models.join(" ")}`.toLowerCase();
        return hay.includes(text.toLowerCase());
      },
    },
    sorting: { defaultState: { sortingColumn: { sortingField: "stamp" }, isDescending: true } },
  });

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="All fine-tuning races — in progress and finished. Select one to watch its models progress (train → eval) and see the winner."
        >
          Races
        </Header>
      }
    >
      <SpaceBetween size="l">
        {/* When a run is selected, cap the list to ~half the viewport (scrollable,
            sticky header) so the detail panel below is visible without scrolling
            past a long history — EC2-console style. Unselected: full height. */}
        <div style={detail ? { maxHeight: "45vh", overflow: "auto" } : undefined}>
        <Table
          {...collectionProps}
          variant="container"
          stickyHeader
          resizableColumns
          loading={loading}
          loadingText="Loading races…"
          header={
            <Header
              counter={
                showArchived
                  ? `(${raceRows.length})`
                  : `(${raceRows.length} of ${races.length})`
              }
              actions={
                <SpaceBetween direction="horizontal" size="m">
                  <Toggle
                    checked={showArchived}
                    onChange={({ detail }) => setShowArchived(detail.checked)}
                  >
                    Show archived{archivedCount ? ` (${archivedCount})` : ""}
                  </Toggle>
                  <Button iconName="refresh" onClick={refreshList} loading={loading}>
                    Refresh
                  </Button>
                </SpaceBetween>
              }
            >
              Race history
            </Header>
          }
          filter={
            <SpaceBetween direction="horizontal" size="xs">
              <TextFilter
                {...filterProps}
                filteringPlaceholder="Find by name, id, split, or model"
                countText={`${filteredItemsCount ?? 0} match${filteredItemsCount === 1 ? "" : "es"}`}
              />
              <Select
                selectedOption={
                  STATUS_FILTER_OPTIONS.find((o) => o.value === statusFilter) ?? STATUS_FILTER_OPTIONS[0]
                }
                onChange={({ detail }) => setStatusFilter(detail.selectedOption.value!)}
                options={STATUS_FILTER_OPTIONS}
              />
            </SpaceBetween>
          }
          columnDefinitions={[
            {
              id: "race",
              header: "Race",
              sortingField: "name",
              // Show the friendly name only — the raw run id is noise in the table
              // (it's available in the detail panel below when a row is selected).
              cell: (r) => (
                <SpaceBetween direction="horizontal" size="xs">
                  <span>{r.name || r.raceId}</span>
                  {r.isSample && <Badge color="blue">sample</Badge>}
                  {r.archived && <Badge color="grey">archived</Badge>}
                </SpaceBetween>
              ),
            },
            {
              id: "status",
              header: "Status",
              sortingField: "_overall",
              cell: (r) =>
                r._overall === "running" ? (
                  <StatusIndicator type="in-progress">Running</StatusIndicator>
                ) : r._overall === "done" ? (
                  <StatusIndicator type="success">Done</StatusIndicator>
                ) : r._overall === "failed" ? (
                  <StatusIndicator type="error">Failed</StatusIndicator>
                ) : r._overall === "mixed" ? (
                  <StatusIndicator type="warning">Mixed</StatusIndicator>
                ) : (
                  <StatusIndicator type="pending">—</StatusIndicator>
                ),
            },
            {
              id: "split",
              header: "Dataset",
              sortingField: "splitId",
              // Friendly dataset name → opens that dataset's stats on the Datasets
              // page (deep-link), not just the page.
              cell: (r) => (
                <Link
                  onFollow={(e) => {
                    e.preventDefault();
                    openDatasetDetails(r.splitId);
                  }}
                  href="#datasets"
                >
                  {datasetLabel(r.splitId)}
                </Link>
              ),
            },
            {
              id: "models",
              header: "Models",
              sortingField: "_modelCount",
              // Vertical badge stack (capped) keeps the row compact instead of a
              // wide comma-joined string that pushes the table horizontal.
              cell: (r) => (
                <SpaceBetween size="xxs">
                  {r.models.slice(0, 3).map((m, i) => (
                    <Badge key={`${m}-${i}`}>{m}</Badge>
                  ))}
                  {r.models.length > 3 && (
                    <Box variant="small" color="text-status-inactive">
                      +{r.models.length - 3} more
                    </Box>
                  )}
                </SpaceBetween>
              ),
            },
            {
              id: "progress",
              header: "Progress",
              sortingField: "_doneCount",
              cell: (r) =>
                `${r._doneCount} done${r._failedCount ? `, ${r._failedCount} failed` : ""} / ${r._modelCount}`,
            },
            {
              id: "open",
              header: "Actions",
              // No "View" button — selecting the row opens the detail panel below
              // (EC2-console style). Only Archive/Restore lives here.
              cell: (r) => (
                <SpaceBetween direction="horizontal" size="xs">
                  {onCloneRun && (
                    <Button
                      variant="inline-link"
                      iconName="copy"
                      onClick={() => onCloneRun(r.raceId)}
                    >
                      Clone &amp; edit
                    </Button>
                  )}
                  {r.archived ? (
                    <Button
                      variant="inline-link"
                      iconName="undo"
                      onClick={() => toggleArchive(r.raceId, false)}
                    >
                      Restore
                    </Button>
                  ) : (
                    <Button
                      variant="inline-link"
                      iconName="close"
                      onClick={() => toggleArchive(r.raceId, true)}
                    >
                      Archive
                    </Button>
                  )}
                </SpaceBetween>
              ),
            },
          ]}
          items={items}
          selectionType="single"
          selectedItems={items.filter((r) => r.raceId === selected)}
          onSelectionChange={({ detail: d }) => setSelected(d.selectedItems[0]?.raceId ?? null)}
          trackBy="raceId"
          empty={
            <Box textAlign="center" padding="m">
              {races.length === 0 ? (
                <>No runs yet. Launch one from <b>Fine-tune</b>.</>
              ) : (
                <>No runs match. Adjust the filter{!showArchived ? " or enable Show archived" : ""}.</>
              )}
            </Box>
          }
        />
        </div>

        {detail && (
          <Container
            header={
              <Header
                variant="h2"
                description={detail.name ? `${detail.name} · ${detail.raceId}` : detail.raceId}
                counter={`(${doneCount}/${detail.entries.length} done)`}
                actions={
                  <SpaceBetween direction="horizontal" size="xs">
                    {onCloneRun && (
                      <Button iconName="copy" onClick={() => onCloneRun(detail.raceId)}>
                        Clone &amp; edit
                      </Button>
                    )}
                    <FormField label="Rank by">
                      <Select
                        selectedOption={
                          RANK_METRIC_OPTIONS.find((o) => o.value === rankMetric) ?? null
                        }
                        onChange={({ detail: d }) => setRankMetric(d.selectedOption.value!)}
                        // Technique-aware options: RLAIF ranks ONLY by judge reward
                        // (no gold); gold-overlap objectives hide reward_mean (always
                        // blank for them). The server's effective_rank_metric enforces
                        // the same, so the dropdown just mirrors what's meaningful.
                        options={RANK_METRIC_OPTIONS.filter((o) =>
                          technique === "rlaif" ? o.value === "reward_mean" : o.value !== "reward_mean"
                        )}
                      />
                    </FormField>
                  </SpaceBetween>
                }
              >
                {allTerminal ? "Race finished" : "Race in progress"}
              </Header>
            }
          >
            <SpaceBetween size="m">
              {/* EC2-console-style summary: key facts about the selected run. */}
              <ColumnLayout columns={4} variant="text-grid">
                <div>
                  <Box variant="awsui-key-label">Race name</Box>
                  <div>{detail.name || <Box color="text-status-inactive">(unnamed)</Box>}</div>
                </div>
                <div>
                  <Box variant="awsui-key-label">Race id</Box>
                  <Box variant="samp" fontSize="body-s">{detail.raceId}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Status</Box>
                  <div>
                    {allTerminal ? (
                      detail.entries.some((e) => e.state === "failed") &&
                      !detail.entries.some((e) => e.state === "done") ? (
                        <StatusIndicator type="error">Failed</StatusIndicator>
                      ) : (
                        <StatusIndicator type="success">Finished</StatusIndicator>
                      )
                    ) : (
                      <StatusIndicator type="in-progress">In progress</StatusIndicator>
                    )}
                  </div>
                </div>
                <div>
                  <Box variant="awsui-key-label">Dataset</Box>
                  <Link
                    href="#datasets"
                    onFollow={(e) => {
                      e.preventDefault();
                      openDatasetDetails(detail.splitId);
                    }}
                  >
                    {datasetLabel(detail.splitId)}
                  </Link>
                  <Box variant="small" color="text-status-inactive">
                    {{
                      preference: "DPO",
                      kto: "KTO",
                      rlvr: "RLVR",
                      rlaif: "RLAIF",
                    }[datasetById[detail.splitId]?.shape ?? "sft"] ?? "SFT"}
                  </Box>
                </div>
              </ColumnLayout>

              {/* EC2-console style: a tabbed lower panel. "Detail" holds the
                  evaluation table (+ winner/status); "Metrics" holds the training
                  curves. Keeps the dense eval grid and the charts separated. */}
              <Tabs
                tabs={[
                  {
                    id: "detail",
                    label: "Detail",
                    content: (
                      <SpaceBetween size="m">
                        {detail.useSpot && (
                          <Box>
                            <Badge color="blue">spot</Badge>{" "}
                            <Box variant="small" display="inline" color="text-status-inactive">
                              Trained on managed spot with checkpoint/resume — cheaper, interruptible.
                            </Box>
                          </Box>
                        )}
                        {detail.entries?.some((e) => e.spot_fell_back) && (
                          <Box>
                            <Badge color="severity-medium">spot → on-demand</Badge>{" "}
                            <Box variant="small" display="inline" color="text-status-inactive">
                              One or more entries couldn't get spot capacity in time and were
                              auto-converted to on-demand (resumed from checkpoint) — these bill at
                              the on-demand rate.
                            </Box>
                          </Box>
                        )}
                        {winner && rankMetric === "reward_mean" ? (
                          <Alert type="success" header={`Winner: ${winner.model_display}`}>
                            Highest <b>judge reward</b>{" "}
                            {winner.metrics?.reward_source === "training" ? "(final training reward)" : "on the held-out prompts"}:{" "}
                            <b>{pct(winner.rankScore)}</b>. RLAIF has no reference answer, so models
                            are ranked by the AI judge's reward — not gold-overlap metrics.
                          </Alert>
                        ) : winner ? (
                          <Alert type="success" header={`Winner: ${winner.model_display}`}>
                            Best <b>{RANK_METRIC_OPTIONS.find((o) => o.value === rankMetric)?.label ?? rankMetric}</b>{" "}
                            on the held-out test set: <b>{pct(winner.rankScore)}</b>. Change
                            "Rank by" to pick the winner on a different metric.
                          </Alert>
                        ) : null}
                        {/* Per-technique metric-lens note: the held-out eval scores
                            pred-vs-ONE-gold, which is the wrong lens for preference
                            objectives. Spell that out so a low overlap number on a
                            good DPO/KTO model doesn't read as failure. */}
                        {(technique === "dpo" || technique === "kto") && (
                          <Alert type="info">
                            <b>{technique.toUpperCase()} is a preference objective.</b> The overlap
                            metrics below (Token F1, ROUGE-L, …) are scored against{" "}
                            <b>one acceptable answer</b> (the {technique === "dpo" ? "chosen" : "desirable"}{" "}
                            response), so a different-but-good answer scores low — a low number here
                            isn't necessarily a bad model. Use the <b>LLM judge</b> column for the real
                            quality signal.
                          </Alert>
                        )}
                        {technique === "rlvr" && (
                          <Alert type="info">
                            <b>RLVR has a verifiable answer</b>, so <b>Numeric</b> / extracted-exact are
                            the metrics to trust. Token-overlap columns can read low when the model adds
                            reasoning before the final answer — that's expected.
                          </Alert>
                        )}
                        {!allTerminal && (
                          <Box variant="small">
                            Refreshing every 15s. The run advances server-side too — safe to leave.
                          </Box>
                        )}
                        {/* Ranked-bar comparison — the head-to-head "who won" visual,
                            here on the run itself (not just the leaderboard page). Only
                            worth showing once ≥2 models have finished. */}
                        {(rankedDetail ?? detail).entries.filter((e) => e.state === "done" && e.rankScore != null).length >= 2 && (
                          <RaceResultBars entries={(rankedDetail ?? detail).entries} rankMetric={rankMetric} rankLabel={rankMetricLabel} />
                        )}
              <Table
                variant="embedded"
                wrapLines
                resizableColumns
                stickyColumns={{ first: 1 }}
                columnDefinitions={([
                  {
                    id: "model",
                    header: "Model",
                    minWidth: 180,
                    cell: (e) => (
                      <SpaceBetween direction="horizontal" size="xs">
                        <span>{e.model_display}</span>
                        {e.isWinner && <Badge color="green">winner</Badge>}
                      </SpaceBetween>
                    ),
                  },
                  {
                    id: "engine",
                    header: "Engine",
                    minWidth: 130,
                    // Per-entry engine (a run can mix engines across its models).
                    // Read from the entry's persisted hp; default llama_factory.
                    cell: (e) =>
                      (e.hp?.engine ?? "llama_factory") === "sagemaker_serverless" ? (
                        <Badge color="severity-neutral">Serverless</Badge>
                      ) : (
                        <Badge>LLaMA-Factory</Badge>
                      ),
                  },
                  { id: "state", header: colHeader("State", "state"), minWidth: 110, cell: (e) => stateIndicator(e.state) },
                  // Base-model control + lift on the CURRENT ranking metric — quantifies
                  // how much fine-tuning helped (base → fine-tuned). Empty if base eval
                  // hasn't completed or doesn't apply.
                  {
                    id: "base",
                    header: colHeader("Base (untrained)", "base"),
                    minWidth: 130,
                    cell: (e) => baseScore(e, rankMetric) != null ? pct(baseScore(e, rankMetric)) : "—",
                  },
                  {
                    id: "tuned",
                    minWidth: 150,
                    // The fine-tuned model's score on the SAME (ranking) metric as
                    // Base + Lift, so base → tuned → lift reads as a before→after→delta
                    // triplet on one metric (the score is also in its own metric column,
                    // but this column always tracks the "Rank by" selector).
                    header: colHeader(`Fine-tuned (${rankMetricLabel})`, "tuned"),
                    cell: (e) => {
                      const f = e.metrics?.[rankMetric as keyof typeof e.metrics] as number | null | undefined;
                      return f != null ? pct(f as number) : "—";
                    },
                  },
                  {
                    id: "lift",
                    // Name the metric the lift is measured on, inline in the header,
                    // so it's unambiguous which "Rank by" metric the points reflect.
                    header: colHeader(`Fine-tune lift (${rankMetricLabel})`, "lift"),
                    minWidth: 150,
                    cell: (e) => {
                      const b = baseScore(e, rankMetric);
                      const f = e.metrics?.[rankMetric as keyof typeof e.metrics] as number | null | undefined;
                      if (b == null || f == null) return "—";
                      const d = (f as number) - b;
                      const sign = d >= 0 ? "+" : "";
                      return (
                        <Badge color={d > 0.001 ? "green" : d < -0.001 ? "red" : "grey"}>
                          {sign}{(d * 100).toFixed(1)} pts
                        </Badge>
                      );
                    },
                  },
                  // RLAIF judge reward (reference-free). "—" for gold-overlap
                  // objectives, which don't produce a reward. The popover notes
                  // whether it's the held-out or final training reward.
                  {
                    id: "reward",
                    header: colHeader("Judge reward", "reward"),
                    minWidth: 130,
                    cell: (e) =>
                      e.metrics?.reward_mean != null ? (
                        <SpaceBetween direction="horizontal" size="xs">
                          <span>{pct(e.metrics.reward_mean)}</span>
                          {e.metrics.reward_source === "held_out" ? (
                            <Badge color="blue">held-out</Badge>
                          ) : e.metrics.reward_source === "training" ? (
                            <Badge color="grey">training</Badge>
                          ) : null}
                        </SpaceBetween>
                      ) : (
                        "—"
                      ),
                  },
                  // DPO preference win-rate (closer to chosen than rejected) — the
                  // metric DPO actually optimizes. "—" for non-DPO objectives.
                  {
                    id: "winrate",
                    header: colHeader("Prefers chosen", "winrate"),
                    minWidth: 130,
                    cell: (e) => pct(e.metrics?.chosen_win_rate),
                  },
                  {
                    id: "exact",
                    header: colHeader("Exact", "exact"),
                    minWidth: 110,
                    // Show the EXTRACTED exact-match alongside raw when they differ
                    // (model wrapped its answer in <think>/fences/prose) — the raw
                    // number understates a scaffolded-but-correct answer.
                    cell: (e) => {
                      const raw = e.metrics?.exact_match;
                      const ex = e.metrics?.exact_match_extracted;
                      if (raw == null) return "—";
                      const showBoth = ex != null && Math.abs(ex - raw) > 0.0005;
                      return showBoth ? (
                        <SpaceBetween direction="horizontal" size="xs">
                          <span>{pct(raw)}</span>
                          <Badge color="blue">{pct(ex)} extr</Badge>
                        </SpaceBetween>
                      ) : (
                        pct(raw)
                      );
                    },
                  },
                  { id: "norm", header: colHeader("Normalized", "norm"), minWidth: 110, cell: (e) => pct(e.metrics?.normalized_match) },
                  { id: "contains", header: colHeader("Contains gold", "contains"), minWidth: 120, cell: (e) => pct(e.metrics?.contains_gold) },
                  { id: "f1", header: colHeader("Token F1", "f1"), minWidth: 100, cell: (e) => pct(e.metrics?.token_f1) },
                  { id: "rouge", header: colHeader("ROUGE-L", "rouge"), minWidth: 100, cell: (e) => pct(e.metrics?.rouge_l) },
                  { id: "charf1", header: colHeader("Char F1", "charf1"), minWidth: 100, cell: (e) => pct(e.metrics?.char_f1) },
                  // Task-aware columns — show "—" on rows whose task type doesn't apply.
                  { id: "labelAcc", header: colHeader("Label acc", "labelAcc"), minWidth: 100, cell: (e) => pct(e.metrics?.label_accuracy) },
                  { id: "numeric", header: colHeader("Numeric", "numeric"), minWidth: 100, cell: (e) => pct(e.metrics?.numeric_match) },
                  { id: "jsonValid", header: colHeader("JSON valid", "jsonValid"), minWidth: 100, cell: (e) => pct(e.metrics?.json_valid) },
                  { id: "json", header: colHeader("JSON struct", "json"), minWidth: 105, cell: (e) => pct(e.metrics?.json_structural) },
                  { id: "jsonKeys", header: colHeader("JSON keys", "jsonKeys"), minWidth: 100, cell: (e) => pct(e.metrics?.json_key_recall) },
                  { id: "scaffold", header: colHeader("Scaffold", "scaffold"), minWidth: 100, cell: (e) => pct(e.metrics?.scaffold_rate) },
                  {
                    id: "len",
                    header: colHeader("Len ratio", "lenratio"),
                    minWidth: 95,
                    cell: (e) => (e.metrics?.length_ratio != null ? e.metrics.length_ratio.toFixed(2) : "—"),
                  },
                  {
                    id: "tps",
                    header: colHeader("Tokens/s", "tps"),
                    minWidth: 100,
                    cell: (e) => e.metrics?.timing?.tokens_per_sec ?? "—",
                  },
                  { id: "instance", header: colHeader("Instance", "instance"), minWidth: 130, cell: (e) => e.instance_type },
                  {
                    id: "judge",
                    header: colHeader("LLM judge (1-5)", "judge"),
                    minWidth: 120,
                    cell: (e) => {
                      if (!e.eval_job || e.state !== "done") return "—";
                      const j = judges[e.eval_job];
                      if (j?.status === "done" && j.result) {
                        const dims = j.result.dimensions;
                        const scoreEl = (
                          <span>
                            <b>{j.result.judgeScore.toFixed(2)}</b> / 5{" "}
                            {e.isWinner && <Badge color="green">auto</Badge>}
                          </span>
                        );
                        // When the rubric returned per-dimension scores, show them
                        // in a popover (faithfulness/format/completeness/…).
                        if (dims && Object.keys(dims).length > 0) {
                          return (
                            <Popover
                              header="Judge breakdown (1-5)"
                              triggerType="custom"
                              dismissButton={false}
                              position="top"
                              content={
                                <SpaceBetween size="xxs">
                                  {Object.entries(dims).map(([d, v]) => (
                                    <div key={d}>
                                      {d}: <b>{v.toFixed(2)}</b>
                                    </div>
                                  ))}
                                </SpaceBetween>
                              }
                            >
                              {scoreEl}
                            </Popover>
                          );
                        }
                        return scoreEl;
                      }
                      if (j?.status === "running") {
                        return (
                          <SpaceBetween direction="horizontal" size="xs">
                            <Spinner size="normal" />
                            <Box variant="small">judging…</Box>
                          </SpaceBetween>
                        );
                      }
                      if (j?.status === "failed") return <Box color="text-status-error">failed</Box>;
                      return (
                        <Button
                          variant="inline-link"
                          iconName="gen-ai"
                          onClick={() => runJudgeFor(e.eval_job as string)}
                        >
                          Judge
                        </Button>
                      );
                    },
                  },
                  {
                    id: "note",
                    header: "Note / action",
                    // A generous min-width so the error text + action buttons get a
                    // real column — without it (and with wrapLines on), the long
                    // SageMaker error wrapped CHARACTER-BY-CHARACTER into a thin
                    // vertical ribbon. minWidth gives it room to wrap by word.
                    minWidth: 300,
                    cell: (e) =>
                      e.state === "failed" ? (
                        // Error text on its own line (wraps by word, capped width)
                        // ABOVE the actions — a long SageMaker error used to run on
                        // one line and stretch the column off-screen.
                        <SpaceBetween size="xs">
                          <Box variant="small" color="text-status-error">
                            <div style={{ maxWidth: 460, whiteSpace: "normal", overflowWrap: "break-word" }}>
                              {e.error ?? "failed"}
                            </div>
                          </Box>
                          <SpaceBetween direction="horizontal" size="xs">
                            <Button
                              variant="inline-link"
                              iconName="gen-ai"
                              loading={triaging === (e.entryKey ?? e.model_id)}
                              onClick={() => diagnose(e.entryKey ?? e.model_id)}
                            >
                              Diagnose
                            </Button>
                            {e.canResume && (
                              <Button
                                variant="inline-link"
                                iconName="redo"
                                loading={retrying === `${e.entryKey ?? e.model_id}:resume`}
                                onClick={() => retry(e.entryKey ?? e.model_id, true)}
                              >
                                Resume from checkpoint
                              </Button>
                            )}
                            <Button
                              variant="inline-link"
                              iconName="refresh"
                              loading={retrying === `${e.entryKey ?? e.model_id}:fresh`}
                              onClick={() => retry(e.entryKey ?? e.model_id)}
                            >
                              {e.canResume ? "Retry fresh" : "Retry"}
                            </Button>
                          </SpaceBetween>
                        </SpaceBetween>
                      ) : e.state === "training" ? (
                        "training…"
                      ) : e.state === "evaluating" ? (
                        "evaluating…"
                      ) : e.state === "done" ? (
                        <Button
                          variant="inline-link"
                          iconName="download"
                          loading={exporting === (e.entryKey ?? e.model_id)}
                          onClick={() => openExport(e.entryKey ?? e.model_id)}
                        >
                          Export &amp; deploy
                        </Button>
                      ) : (
                        ""
                      ),
                  },
                ] as TableProps.ColumnDefinition<RaceEntry>[]).filter((c) =>
                  // Structural columns always show; metric columns only when they're
                  // meaningful for THIS run's technique (Runs detail is single-objective).
                  ["model", "engine", "state", "instance", "tps", "note"].includes(c.id ?? "") ||
                  showCol(c.id ?? "")
                )}
                items={(rankedDetail ?? detail).entries}
                trackBy={(e) => e.entryKey ?? e.model_id}
                empty="No entries"
              />
                      </SpaceBetween>
                    ),
                  },
                  {
                    id: "metrics",
                    label: "Metrics",
                    content: (
                      <SpaceBetween size="m">
                        <Box variant="small" color="text-status-inactive">
                          Training &amp; validation loss per model (from CloudWatch). Validation
                          loss is logged when the dataset has a validation split.
                        </Box>
                        <LossChart entries={detail.entries} />
                        {detail.entries.some((e) => (e.hp?.stage as string) === "rlvr") && (
                          <RewardCurve entries={detail.entries} />
                        )}
                      </SpaceBetween>
                    ),
                  },
                ]}
              />
            </SpaceBetween>
          </Container>
        )}
      </SpaceBetween>

      {/* Dataset details — opened IN-PAGE from a Dataset link (no navigation away
          from Runs). Details-only: stats + eval strategy, not the agentic wizard. */}
      {datasetDetail && (
        <InvestigateDataset
          splitId={datasetDetail.splitId}
          name={datasetDetail.name}
          detailsOnly
          onClose={() => setDatasetDetail(null)}
        />
      )}

      {/* Failure-triage agent result */}
      <Modal
        visible={triage !== null}
        onDismiss={() => setTriage(null)}
        size="medium"
        header={
          <SpaceBetween direction="horizontal" size="xs">
            <span>Diagnosis — {triage?.model ?? ""}</span>
            <AgentBadge />
          </SpaceBetween>
        }
      >
        {triage && (
          <SpaceBetween size="m">
            <AgentCaption>
              An AI agent read this job's logs and diagnosed the failure — advisory; review before retrying.
            </AgentCaption>
            <Alert
              type={triage.result.retryable ? "info" : "warning"}
              header={triage.result.summary}
            >
              {triage.result.retryable
                ? "Looks retryable (transient) — retrying may just work."
                : "Not a transient failure — apply the fix below before retrying."}
            </Alert>
            <Box>
              <Box variant="awsui-key-label">Root cause</Box>
              {triage.result.rootCause}
            </Box>
            <Box>
              <Box variant="awsui-key-label">Recommended fix</Box>
              {triage.result.fix}
            </Box>
            {triage.result.configChanges && Object.keys(triage.result.configChanges).length > 0 && (
              <Box>
                <Box variant="awsui-key-label">Suggested config changes</Box>
                <ul style={{ margin: 0, paddingLeft: "1.2em" }}>
                  {Object.entries(triage.result.configChanges).map(([k, v]) => (
                    <li key={k}>
                      <Box variant="span" fontSize="body-s">
                        <code>{k}</code> → <b>{v}</b>
                      </Box>
                    </li>
                  ))}
                </ul>
              </Box>
            )}
            <SpaceBetween direction="horizontal" size="xs">
              {triage.result.confidence && (
                <Badge color={triage.result.confidence === "high" ? "green" : "grey"}>
                  confidence: {triage.result.confidence}
                </Badge>
              )}
              {triage.result._context?.classification?.category && (
                <Badge color="blue">{triage.result._context.classification.category}</Badge>
              )}
            </SpaceBetween>
            <Box fontSize="body-s" color="text-status-inactive">
              Advisory — apply the change on the Fine-tune page (or click Retry for a transient
              issue). The agent reads the job's log + config; it doesn't change anything itself.
            </Box>
          </SpaceBetween>
        )}
      </Modal>

      {/* Export & deploy to the user's own AWS account */}
      <Modal
        visible={exportInfo !== null}
        onDismiss={() => setExportInfo(null)}
        size="medium"
        header={`Export & deploy — ${exportInfo?.modelDisplay ?? ""}`}
        footer={
          exportInfo && (
            <SpaceBetween direction="horizontal" size="xs">
              {exportInfo.licenseRequired ? (
                // Gated full/freeze: the download is withheld until the user accepts
                // the base license. Accepting re-fetches export_info with the flag,
                // which mints the weights URL and flips the button to Download.
                <Button
                  variant="primary"
                  iconName="status-positive"
                  loading={exporting === exportInfo.modelId}
                  onClick={() => openExport(exportInfo.modelId, true)}
                >
                  Accept {exportInfo.licenseModel} license & enable download
                </Button>
              ) : (
                <Button
                  variant="primary"
                  iconName="download"
                  loading={downloading}
                  onClick={() => downloadBundle(exportInfo.modelId)}
                >
                  Download deploy bundle
                </Button>
              )}
              <Button variant="link" onClick={() => setExportInfo(null)}>
                Close
              </Button>
            </SpaceBetween>
          )
        }
      >
        {exportInfo && (
          <SpaceBetween size="m">
            <Box>
              Deploy this fine-tune to a SageMaker endpoint in <b>your own AWS account</b>. The
              bundle is a small zip (deploy script + inference code + manifest) — the weights are
              downloaded separately via a time-limited link, so nothing multi-GB goes through your
              browser.
            </Box>
            {exportInfo.licenseRequired ? (
              <Alert type="warning" header="Gated full fine-tune → license required">
                This is a full/freeze fine-tune of the gated base <b>{exportInfo.licenseModel}</b>.
                It has no adapter, so the artifact is the <b>merged standalone weights</b> — which
                embed the gated base. Downloading therefore redistributes those weights, so you must
                accept <b>{exportInfo.licenseModel}</b>'s license first. (You can still evaluate and
                compare this model here without downloading.) Click the button below to accept and
                enable the download.
              </Alert>
            ) : exportInfo.deployMode === "adapter" ? (
              <Alert type="warning" header="Gated base model → adapter-only bundle">
                <b>{exportInfo.hfBaseModel}</b> is license-gated, so the bundle ships the LoRA{" "}
                <b>adapter</b> only. <code>deploy.sh</code> pulls the base from Hugging Face at
                deploy time — set <code>HF_TOKEN</code> in your env first (and accept the model's
                license on HF). Redistributing merged gated weights would breach the license.
              </Alert>
            ) : (
              <Alert type="success" header="Standalone merged model">
                The bundle deploys a self-contained merged model — no Hugging Face token or
                dependency needed at deploy time.
                {exportInfo.requiresLicenseAcceptance && (
                  <> (You accepted the {exportInfo.licenseModel} license to enable this download.)</>
                )}
              </Alert>
            )}
            <ColumnLayout columns={2} variant="text-grid">
              <div>
                <Box variant="awsui-key-label">Base model</Box>
                <Box>{exportInfo.hfBaseModel}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Deploy mode</Box>
                <Badge color={exportInfo.deployMode === "merged" ? "green" : "blue"}>
                  {exportInfo.deployMode}
                </Badge>
              </div>
              <div>
                <Box variant="awsui-key-label">Engine</Box>
                <Badge color={exportInfo.engine === "sagemaker_serverless" ? "blue" : "grey"}>
                  {exportInfo.engine === "sagemaker_serverless"
                    ? "⚡ serverless"
                    : "LLaMA-Factory"}
                </Badge>
              </div>
              <div>
                <Box variant="awsui-key-label">Suggested instance</Box>
                <Box>{exportInfo.suggestedInstance}</Box>
              </div>
              {exportInfo.weightsTtlSeconds != null && (
                <div>
                  <Box variant="awsui-key-label">Weights link valid for</Box>
                  <Box>{Math.round(exportInfo.weightsTtlSeconds / 3600)} h</Box>
                </div>
              )}
              {exportInfo.weightsFiles && (
                <div>
                  <Box variant="awsui-key-label">Weight files</Box>
                  <Box>{exportInfo.weightsFiles.length} files (loose, no tarball)</Box>
                </div>
              )}
            </ColumnLayout>
            {exportInfo.weightsTtlSeconds != null && (
              <Box fontSize="body-s" color="text-status-inactive">
                After downloading: <code>unzip</code>, then{" "}
                <code>./deploy.sh --profile &lt;you&gt; --region &lt;region&gt;</code>. The README has
                the full steps. The link expires in{" "}
                {Math.round(exportInfo.weightsTtlSeconds / 3600)} h — re-export for a fresh one.
              </Box>
            )}
          </SpaceBetween>
        )}
      </Modal>
    </ContentLayout>
  );
}
