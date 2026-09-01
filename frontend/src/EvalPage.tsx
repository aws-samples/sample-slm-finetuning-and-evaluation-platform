// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useMemo, useRef, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import Container from "@cloudscape-design/components/container";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Select from "@cloudscape-design/components/select";
import Input from "@cloudscape-design/components/input";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import Spinner from "@cloudscape-design/components/spinner";
import Table from "@cloudscape-design/components/table";

import {
  getEvalStatus,
  getRace,
  launchEval,
  listRaces,
  type CurrentSplit,
  type EvalLaunchResponse,
  type EvalStatus,
  type RaceEntry,
  type RaceSummary,
} from "./api";
import { EvalDatasetPicker } from "./EvalDatasetPicker";
import { useNotify, errText } from "./notifications";

function num(s: string): number | undefined {
  if (s.trim() === "") return undefined;
  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}

function statusType(s: string): "in-progress" | "success" | "error" | "stopped" | "pending" {
  switch (s) {
    case "Completed":
      return "success";
    case "Failed":
      return "error";
    case "Stopped":
      return "stopped";
    case "InProgress":
      return "in-progress";
    default:
      return "pending";
  }
}

export function EvalPage({ currentSplit: initial }: { currentSplit: CurrentSplit | null }) {
  const { notify } = useNotify();
  // Standalone evaluate owns its own dataset selection (the test set to score on).
  const [currentSplit, setCurrentSplit] = useState<CurrentSplit | null>(initial);
  // Model selection: pick a race, then a trained model within it.
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [selectedRace, setSelectedRace] = useState<string | null>(null);
  const [raceEntries, setRaceEntries] = useState<RaceEntry[]>([]);
  const [loadingEntries, setLoadingEntries] = useState(false);
  // The chosen model's source training job (what the eval scores).
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [backend, setBackend] = useState("vllm");
  const [temperature, setTemperature] = useState("0.0");
  const [topP, setTopP] = useState("1.0");
  const [maxNewTokens, setMaxNewTokens] = useState("256");
  const [seed, setSeed] = useState("42");

  const [launching, setLaunching] = useState(false);
  const [launch, setLaunch] = useState<EvalLaunchResponse | null>(null);
  const [status, setStatus] = useState<EvalStatus | null>(null);
  const [loadingRaces, setLoadingRaces] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load the selected race's entries (so a model that finishes training after
  // the page loaded becomes selectable). `keepJob` keeps the current selection
  // when refreshing in place; the race-switch effect resets it.
  function loadEntries(raceId: string, keepJob = false) {
    setLoadingEntries(true);
    if (!keepJob) setSelectedJob(null);
    getRace(raceId)
      .then((r) => setRaceEntries(r.entries))
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoadingEntries(false));
  }

  function refreshRaces() {
    setLoadingRaces(true);
    listRaces()
      .then(setRaces)
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoadingRaces(false));
    // Also reload the selected race's entries — Refresh otherwise only updates
    // the race LIST, so a model that finished training never became selectable.
    if (selectedRace) loadEntries(selectedRace, true);
  }

  useEffect(refreshRaces, []);

  // When a race is picked, load its entries (to choose a trained model).
  useEffect(() => {
    if (!selectedRace) {
      setRaceEntries([]);
      return;
    }
    loadEntries(selectedRace);
  }, [selectedRace]);

  const raceOptions = useMemo(
    () =>
      races.map((r) => ({
        value: r.raceId,
        label: r.name || r.raceId,
        description: `${r.models.length} model(s) · ${r.splitId}`,
      })),
    [races]
  );

  // A model is evaluable only once its TRAINING has COMPLETED. In the race
  // state machine the entry moves training → eval_pending → evaluating → done;
  // so training is finished for any state past "training". We must NOT offer a
  // model whose training is still running (pending/training) or failed —
  // there's no model artifact to evaluate yet.
  const TRAINED_STATES = ["eval_pending", "evaluating", "done"];
  const trainingDone = (e: RaceEntry) => !!e.train_job && TRAINED_STATES.includes(e.state);

  const modelOptions = useMemo(
    () =>
      raceEntries.filter(trainingDone).map((e) => ({
        value: e.train_job as string,
        label: `${e.model_display}${e.isWinner ? " 🏆" : ""}`,
        description: `${e.state}${e.rankScore != null ? ` · score ${e.rankScore.toFixed(3)}` : ""}`,
      })),
    [raceEntries]
  );

  // Models in the race still training (shown as a hint so the user knows why
  // they're not yet selectable).
  const stillTraining = useMemo(
    () => raceEntries.filter((e) => e.state === "pending" || e.state === "training"),
    [raceEntries]
  );

  useEffect(() => {
    if (!launch) return;
    const tick = () =>
      getEvalStatus(launch.jobName)
        .then(setStatus)
        .catch((e) => notify({ type: "error", content: errText(e) }));
    tick();
    pollRef.current = setInterval(tick, 10000);
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [launch]);

  useEffect(() => {
    if (status && ["Completed", "Failed", "Stopped"].includes(status.status)) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;  // null after clearing so a re-launch can't double-poll
      }
    }
  }, [status]);

  async function doLaunch() {
    if (!selectedJob || !currentSplit) return;
    setLaunching(true);
    setStatus(null);
    try {
      const res = await launchEval({
        sourceJobName: selectedJob,
        splitId: currentSplit.splitId,
        backend,
        temperature: num(temperature),
        topP: num(topP),
        maxNewTokens: num(maxNewTokens),
        seed: num(seed),
      });
      setLaunch(res);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setLaunching(false);
    }
  }

  const metrics = status?.metrics ?? null;
  const perClassItems = metrics
    ? Object.entries(metrics.per_class_accuracy).map(([label, v]) => ({ label, ...v }))
    : [];

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Score a fine-tuned model on the held-out test set with deterministic decoding. Computes all metrics so you can compare candidates on whichever matters."
        >
          Evaluate
        </Header>
      }
    >
      <SpaceBetween size="l">
        <EvalDatasetPicker selected={currentSplit} onSelect={setCurrentSplit} />

        <Container
          header={
            <Header
              variant="h2"
              actions={
                <Button iconName="refresh" onClick={refreshRaces} loading={loadingRaces}>
                  Refresh
                </Button>
              }
            >
              Model & decoding
            </Header>
          }
        >
          <SpaceBetween size="m">
            <ColumnLayout columns={2}>
              <FormField
                label="Run"
                description="Pick a fine-tuning run, then choose a model trained in it."
              >
                <Select
                  selectedOption={
                    selectedRace
                      ? {
                          value: selectedRace,
                          label: raceOptions.find((o) => o.value === selectedRace)?.label ?? selectedRace,
                        }
                      : null
                  }
                  onChange={({ detail }) => setSelectedRace(detail.selectedOption.value ?? null)}
                  options={raceOptions}
                  placeholder={loadingRaces ? "Loading…" : "Choose a run"}
                  empty="No runs yet — submit a fine-tune first"
                  filteringType="auto"
                />
              </FormField>
              <FormField
                label="Fine-tuned model"
                description="Only models whose training has COMPLETED are selectable (🏆 = run winner)."
              >
                <Select
                  selectedOption={
                    selectedJob
                      ? {
                          value: selectedJob,
                          label:
                            modelOptions.find((o) => o.value === selectedJob)?.label ?? selectedJob,
                        }
                      : null
                  }
                  onChange={({ detail }) => setSelectedJob(detail.selectedOption.value ?? null)}
                  options={modelOptions}
                  placeholder={
                    !selectedRace
                      ? "Pick a run first"
                      : loadingEntries
                      ? "Loading…"
                      : "Choose a trained model"
                  }
                  empty={
                    stillTraining.length > 0
                      ? "Models still training — wait for one to finish"
                      : "No trained models in this run yet"
                  }
                  disabled={!selectedRace}
                  filteringType="auto"
                />
              </FormField>
            </ColumnLayout>

            {selectedRace && stillTraining.length > 0 && (
              <Alert type="info">
                {stillTraining.length} model(s) in this run are still training (
                {stillTraining.map((e) => e.model_display).join(", ")}) — they'll become
                selectable once their training job completes.
              </Alert>
            )}
            <ColumnLayout columns={5}>
              <FormField label="Backend">
                <Select
                  selectedOption={{ value: backend, label: backend }}
                  onChange={({ detail }) => setBackend(detail.selectedOption.value!)}
                  options={[
                    { value: "vllm", label: "vLLM" },
                    { value: "hf", label: "HF generate" },
                  ]}
                />
              </FormField>
              <FormField label="Temperature" description="0 = greedy (deterministic)">
                <Input value={temperature} type="number" onChange={({ detail }) => setTemperature(detail.value)} />
              </FormField>
              <FormField label="Top-p" description="Nucleus sampling; 1.0 = off (no truncation)">
                <Input value={topP} type="number" onChange={({ detail }) => setTopP(detail.value)} />
              </FormField>
              <FormField label="Max new tokens">
                <Input value={maxNewTokens} type="number" onChange={({ detail }) => setMaxNewTokens(detail.value)} />
              </FormField>
              <FormField label="Seed">
                <Input value={seed} type="number" onChange={({ detail }) => setSeed(detail.value)} />
              </FormField>
            </ColumnLayout>
            <Button
              variant="primary"
              // Only block while a launch is in-flight OR the current eval is still
              // running — once it's Completed/Failed/Stopped the user can run
              // another (different model/decoding). Previously `!!launch` disabled
              // the button forever after the first eval, forcing a page reload.
              disabled={
                !selectedJob ||
                !currentSplit ||
                launching ||
                (!!launch && !!status && !["Completed", "Failed", "Stopped"].includes(status.status)) ||
                (!!launch && !status)
              }
              loading={launching}
              onClick={doLaunch}
            >
              {launch ? "Run another eval (incurs cost)" : "Launch eval (incurs cost)"}
            </Button>
          </SpaceBetween>
        </Container>

        {launch && (
          <Container
            header={
              <Header
                variant="h2"
                description={launch.jobName}
                actions={
                  status && !["Completed", "Failed", "Stopped"].includes(status.status) ? (
                    <Spinner />
                  ) : undefined
                }
              >
                Eval status
              </Header>
            }
          >
            <SpaceBetween size="m">
              <ColumnLayout columns={4} variant="text-grid">
                <div>
                  <Box variant="awsui-key-label">Status</Box>
                  {status ? (
                    <StatusIndicator type={statusType(status.status)}>
                      {status.status}
                    </StatusIndicator>
                  ) : (
                    <Spinner />
                  )}
                </div>
                <div>
                  <Box variant="awsui-key-label">Stage</Box>
                  <Box>{status?.secondaryStatus ?? "—"}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Billable time</Box>
                  <Box>{status?.billableTimeSeconds != null ? `${status.billableTimeSeconds}s` : "—"}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Source model</Box>
                  <Box variant="small">{launch.sourceJob}</Box>
                </div>
              </ColumnLayout>

              {status?.failureReason && (
                <Alert type="error" header="Failure reason">
                  {status.failureReason}
                </Alert>
              )}
            </SpaceBetween>
          </Container>
        )}

        {metrics && (
          <Container
            header={
              <Header
                variant="h2"
                description={`${metrics.count} test rows · backend ${metrics.decoding.backend} · temp ${metrics.decoding.temperature}${metrics.decoding.top_p < 1 ? ` · top_p ${metrics.decoding.top_p}` : ""} · seed ${metrics.decoding.seed}`}
              >
                Metrics
              </Header>
            }
          >
            <SpaceBetween size="m">
              <ColumnLayout columns={4} variant="text-grid">
                <div>
                  <Box variant="awsui-key-label">Exact match</Box>
                  <Box variant="h2">{pct(metrics.exact_match)}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Normalized match</Box>
                  <Box variant="h2">{pct(metrics.normalized_match)}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">Token F1</Box>
                  <Box variant="h2">{pct(metrics.token_f1)}</Box>
                </div>
                <div>
                  <Box variant="awsui-key-label">JSON-structural</Box>
                  <Box variant="h2">{pct(metrics.json_structural)}</Box>
                  <Box variant="small">
                    {metrics.json_applicable_rows} applicable row(s)
                  </Box>
                </div>
              </ColumnLayout>

              <Box>
                <Box variant="awsui-key-label">Per-class accuracy (normalized)</Box>
                <Table
                  variant="embedded"
                  columnDefinitions={[
                    { id: "label", header: "Class (gold)", cell: (r) => r.label },
                    { id: "acc", header: "Accuracy", cell: (r) => pct(r.accuracy) },
                    { id: "correct", header: "Correct", cell: (r) => r.correct },
                    { id: "total", header: "Total", cell: (r) => r.total },
                  ]}
                  items={perClassItems}
                  empty="No rows"
                />
              </Box>
            </SpaceBetween>
          </Container>
        )}
      </SpaceBetween>
    </ContentLayout>
  );
}
