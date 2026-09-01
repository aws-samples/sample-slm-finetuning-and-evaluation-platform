// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useRef, useState } from "react";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import LineChart from "@cloudscape-design/components/line-chart";
import Box from "@cloudscape-design/components/box";
import Select from "@cloudscape-design/components/select";
import SegmentedControl from "@cloudscape-design/components/segmented-control";
import Slider from "@cloudscape-design/components/slider";
import Alert from "@cloudscape-design/components/alert";
import FormField from "@cloudscape-design/components/form-field";
import SpaceBetween from "@cloudscape-design/components/space-between";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Spinner from "@cloudscape-design/components/spinner";

import { getTrainingCurves, type RaceEntry, type TrainingCurves } from "./api";

// Which curve to plot in COMPARE mode. Loss curves answer "how many epochs?";
// LR is a sanity check on the schedule (warmup → decay).
const METRIC_OPTIONS = [
  { value: "trainLoss", label: "Training loss" },
  { value: "evalLoss", label: "Validation loss" },
  { value: "learningRate", label: "Learning rate" },
  { value: "gradNorm", label: "Gradient norm" },
] as const;

type MetricKey = "trainLoss" | "evalLoss" | "learningRate" | "gradNorm";

// Three ways to read the curves:
//  - dashboard : ALL metrics at once as small-multiples (one mini-chart per metric,
//                one line per model) — no dropdown-driving; the at-a-glance default.
//  - compare   : one metric, one line per model (zoom in on which model wins).
//  - detail    : one model, train + eval loss overlaid (is THIS model overfitting?
//                — the generalization gap the validation set exists to reveal).
type ViewMode = "dashboard" | "compare" | "detail";

// X axis: training progress as elapsed time, or as epochs (what scientists
// actually reason in — "overfits after epoch 2"). Both are already scraped;
// epoch values are interpolated onto each loss point by shared timestamp.
type XAxis = "epoch" | "minutes";

// Distinct colours so overlaid model lines stay readable (Cloudscape chart
// palette hexes).
const COLORS = ["#688ae8", "#c33d69", "#2ea597", "#e07941", "#8456ce", "#3184c2"];

// Fixed colours for the train/eval overlay in detail mode (consistent meaning).
const TRAIN_COLOR = "#688ae8"; // blue
const EVAL_COLOR = "#c33d69"; // red

function isTerminal(s: string) {
  return s === "done" || s === "failed";
}

type Pt = { x: number; y: number };

// Interpolate an epoch value for a given minutes-x against the epoch series
// (monotonic in time). Both series are stamped in the same minutes-since-start
// coordinate, so we find the surrounding epoch points and lerp.
function minutesToEpoch(minutes: number, epochSeries: Pt[]): number | null {
  if (epochSeries.length === 0) return null;
  if (minutes <= epochSeries[0].x) return epochSeries[0].y;
  if (minutes >= epochSeries[epochSeries.length - 1].x) return epochSeries[epochSeries.length - 1].y;
  for (let i = 1; i < epochSeries.length; i++) {
    const a = epochSeries[i - 1];
    const b = epochSeries[i];
    if (minutes <= b.x) {
      const t = b.x === a.x ? 0 : (minutes - a.x) / (b.x - a.x);
      return a.y + t * (b.y - a.y);
    }
  }
  return epochSeries[epochSeries.length - 1].y;
}

// The exported model is the MIN eval-loss checkpoint (load_best_model_at_end).
// Return that point (lowest y), or null if there's no eval data.
function bestEvalPoint(evalSeries: Pt[]): Pt | null {
  if (evalSeries.length === 0) return null;
  return evalSeries.reduce((best, p) => (p.y < best.y ? p : best), evalSeries[0]);
}

// Exponential moving average (TensorBoard's "smoothing"): each point is blended
// with the running average, weight = the smoothing factor in [0,1). Raw training
// loss is noisy step-to-step; smoothing reveals the underlying trend. 0 = raw.
// Debiased like TensorBoard so the early points aren't dragged toward 0.
function ema(data: Pt[], weight: number): Pt[] {
  if (weight <= 0 || data.length === 0) return data;
  let last = 0;
  let debiasWeight = 0;
  return data.map((p) => {
    last = last * weight + (1 - weight) * p.y;
    debiasWeight = debiasWeight * weight + (1 - weight);
    return { x: p.x, y: last / (debiasWeight || 1) };
  });
}

// Auto-detected training-health read on the focused model's loss curves, surfaced
// as a one-line insight so a non-expert knows what the curve is telling them.
type Health = { kind: "good" | "warn" | "info"; text: string } | null;
function trainingHealth(train: Pt[], evalL: Pt[], status: string): Health {
  if (train.length < 3) return null;
  const tail = (s: Pt[], n: number) => s.slice(Math.max(0, s.length - n));
  const mean = (s: Pt[]) => (s.length ? s.reduce((a, p) => a + p.y, 0) / s.length : 0);
  const first = train[0].y;
  const lastTrain = train[train.length - 1].y;
  // Instability: a late spike well above the recent floor (diverging / LR too high).
  const recent = tail(train, 8);
  const recentMin = Math.min(...recent.map((p) => p.y));
  if (lastTrain > recentMin * 2.5 && lastTrain > 0.05) {
    return { kind: "warn", text: "Training loss is spiking — possible instability (try a lower learning rate or check the data)." };
  }
  // Overfitting: validation rising while training keeps falling (the gap widening).
  if (evalL.length >= 4) {
    const eHalf = Math.floor(evalL.length / 2);
    const evalEarly = mean(evalL.slice(0, eHalf));
    const evalLate = mean(tail(evalL, Math.max(2, evalL.length - eHalf)));
    const trainStillFalling = lastTrain < mean(tail(train, 5)) * 1.05;
    if (evalLate > evalEarly * 1.05 && trainStillFalling) {
      return { kind: "warn", text: "Validation loss is rising while training loss falls — the model is starting to overfit (the exported best-checkpoint guards against this)." };
    }
  }
  // Barely moving: loss flat from the start (LR too low / wrong target / too few steps).
  if (Math.abs(first - lastTrain) < first * 0.05 && train.length >= 5) {
    return { kind: "info", text: "Training loss is barely moving — the model isn't learning much (consider a higher learning rate, more epochs, or check the dataset)." };
  }
  // Converging nicely.
  const dropped = first > 0 ? ((first - lastTrain) / first) * 100 : 0;
  if (status === "Completed" || isTerminalStatus(status)) {
    return { kind: "good", text: `Converged — training loss fell ${dropped.toFixed(0)}% from start to finish.` };
  }
  return { kind: "good", text: `Converging — training loss down ${dropped.toFixed(0)}% so far.` };
}

function isTerminalStatus(s: string): boolean {
  return s === "Completed" || s === "Failed" || s === "Stopped";
}

// Remap a metric series' x-axis from minutes to the chosen axis. For "minutes"
// it's identity; for "epoch" each point's x becomes its interpolated epoch.
function remapX(data: Pt[], xAxis: XAxis, epochSeries: Pt[]): Pt[] {
  if (xAxis === "minutes") return data;
  return data
    .map((p) => {
      const e = minutesToEpoch(p.x, epochSeries);
      return e == null ? null : { x: e, y: p.y };
    })
    .filter((p): p is Pt => p !== null);
}

/**
 * Live training-curve chart for a race: one line per model, fetched from
 * CloudWatch (the metrics SageMaker scrapes from the training logs). Polls
 * while any model is still training; stops once all entries are terminal.
 *
 * Only entries with a launched train_job are plotted. Empty series (job too new,
 * or launched before metric scraping existed) render as a clear empty state.
 */
export function LossChart({ entries }: { entries: RaceEntry[] }) {
  const [metric, setMetric] = useState<MetricKey>("trainLoss");
  const [mode, setMode] = useState<ViewMode>("dashboard");
  const [xAxis, setXAxis] = useState<XAxis>("epoch");
  // Log-scale Y reveals late-training behaviour (loss spans orders of magnitude;
  // the interesting tail is a flat line near zero on a linear scale).
  const [logY, setLogY] = useState(false);
  // EMA smoothing factor [0,1) — TensorBoard-style. Off by default so the raw
  // curve is the baseline; raise it to see the trend through noisy steps.
  const [smoothing, setSmoothing] = useState(0);
  const [detailModel, setDetailModel] = useState<string | null>(null);
  const [curves, setCurves] = useState<Record<string, TrainingCurves>>({});
  const [loading, setLoading] = useState(false);
  // Whether the FIRST curve fetch has resolved. Until it has, an empty `curves`
  // would wrongly render the "trained before live curves" copy — so we show a
  // loading state instead while the initial fetch is in flight.
  const [fetched, setFetched] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Entries that have a training job to chart. Stable join key for the effect so
  // we re-subscribe when the set of jobs (or their states) changes.
  const charted = entries.filter((e) => e.train_job);
  const anyTraining = charted.some((e) => !isTerminal(e.state));
  const jobsKey = charted.map((e) => `${e.train_job}:${e.state}`).join(",");

  // Default the detail-mode model to the first charted one; keep it valid as the
  // race evolves.
  useEffect(() => {
    if (charted.length === 0) {
      setDetailModel(null);
    } else if (!detailModel || !charted.some((e) => (e.entryKey ?? e.model_id) === detailModel)) {
      setDetailModel(charted[0].entryKey ?? charted[0].model_id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobsKey]);

  useEffect(() => {
    if (charted.length === 0) {
      setCurves({});
      return;
    }
    let cancelled = false;
    setFetched(false); // new job set → first fetch pending again
    const tick = async () => {
      setLoading(true);
      const results = await Promise.all(
        charted.map((e) =>
          getTrainingCurves(e.train_job as string)
            .then((c) => [e.entryKey ?? e.model_id, c] as const)
            .catch(() => null)
        )
      );
      if (cancelled) return;
      const next: Record<string, TrainingCurves> = {};
      for (const r of results) if (r) next[r[0]] = r[1];
      setCurves(next);
      setLoading(false);
      setFetched(true);
    };
    tick();
    // Poll while training is live; once everything is terminal the curve is
    // final, so we fetch once more (above) and don't set an interval.
    if (anyTraining) {
      pollRef.current = setInterval(tick, 20000);
    }
    return () => {
      cancelled = true;
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobsKey]);

  // LR is meaningless on an epoch axis interpolation? No — it's logged on the
  // same lines as loss/epoch, so the epoch remap is valid for it too.
  // Smoothing applies to loss / grad-norm (a trend through noise) but NOT learning
  // rate (a deterministic schedule — smoothing it is meaningless).
  // One line per model for a given metric (the compare-mode series). Factored out
  // so the dashboard view can build the same series for EVERY metric at once.
  const seriesForMetric = (mk: MetricKey) =>
    charted
      .map((e, i) => {
        const c = curves[e.entryKey ?? e.model_id];
        const data = c?.series[mk] ?? [];
        if (data.length === 0) return null;
        // LR is a deterministic schedule — never smooth it (mirrors `smoothable`).
        const sm = mk === "learningRate" ? data : ema(data, smoothing);
        return {
          title: e.model_display,
          type: "line" as const,
          color: COLORS[i % COLORS.length],
          data: remapX(sm, xAxis, c?.series.epoch ?? []),
        };
      })
      .filter((s): s is NonNullable<typeof s> => s !== null);

  const compareSeries = seriesForMetric(metric);

  // DETAIL mode: one model, train + eval loss overlaid (the generalization gap).
  const detailEntry = charted.find((e) => (e.entryKey ?? e.model_id) === detailModel) ?? null;
  const detailCurves = detailModel ? curves[detailModel] : undefined;
  const detailBest = detailCurves ? bestEvalPoint(detailCurves.series.evalLoss ?? []) : null;
  const detailSeries = (() => {
    if (!detailCurves) return [] as any[];
    const out: any[] = [];
    const epochSeries = detailCurves.series.epoch ?? [];
    const train = detailCurves.series.trainLoss ?? [];
    const evalL = detailCurves.series.evalLoss ?? [];
    if (train.length)
      out.push({ title: "Training loss", type: "line", color: TRAIN_COLOR, data: remapX(ema(train, smoothing), xAxis, epochSeries) });
    if (evalL.length)
      out.push({ title: "Validation loss", type: "line", color: EVAL_COLOR, data: remapX(ema(evalL, smoothing), xAxis, epochSeries) });
    // Mark the exported (best) checkpoint as a horizontal threshold at its
    // eval-loss value — this is the model that actually ships.
    if (detailBest) {
      out.push({
        title: `Best checkpoint (exported) · eval ${detailBest.y.toFixed(3)}`,
        type: "threshold",
        color: "#1d8102",
        y: detailBest.y,
      });
    }
    return out;
  })();

  const series = mode === "detail" ? detailSeries : compareSeries;

  // Numeric readout for the focused model (detail model, else first charted).
  // Scientists want the exact figures, not just the curve shape.
  const readoutModel = mode === "detail" ? detailModel : (charted[0] ? charted[0].entryKey ?? charted[0].model_id : null);
  const readoutCurves = readoutModel ? curves[readoutModel] : undefined;
  const readout = (() => {
    if (!readoutCurves) return null;
    const train = readoutCurves.series.trainLoss ?? [];
    const evalL = readoutCurves.series.evalLoss ?? [];
    const epochS = readoutCurves.series.epoch ?? [];
    const best = bestEvalPoint(evalL);
    const bestEpoch = best ? minutesToEpoch(best.x, epochS) : null;
    return {
      latestTrain: train.length ? train[train.length - 1].y : null,
      latestEval: evalL.length ? evalL[evalL.length - 1].y : null,
      bestEval: best ? best.y : null,
      bestEpoch,
      curEpoch: epochS.length ? epochS[epochS.length - 1].y : null,
      status: readoutCurves.status,
    };
  })();
  const fmtLoss = (v: number | null) => (v == null ? "—" : v.toFixed(4));

  // Auto-detected training-health insight on the focused model (RAW curves — the
  // read shouldn't change with the cosmetic smoothing slider).
  const health: Health = readoutCurves
    ? trainingHealth(
        readoutCurves.series.trainLoss ?? [],
        readoutCurves.series.evalLoss ?? [],
        readoutCurves.status
      )
    : null;

  const COMPARE_Y_LABEL: Record<MetricKey, string> = {
    trainLoss: "Training loss",
    evalLoss: "Validation loss",
    learningRate: "Learning rate",
    gradNorm: "Gradient norm",
  };
  const yLabel = mode === "detail" ? "Loss" : COMPARE_Y_LABEL[metric];

  // Learning rate spans tiny magnitudes (exp notation); everything else is a
  // friendlier 0–N range, but very small non-zero values (late-training loss, small
  // grad norms) also use exponential so a tick doesn't collapse to "0.000".
  const fmtY = (v: number) => {
    if (mode === "compare" && metric === "learningRate") return v.toExponential(1);
    if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(1);
    return v.toFixed(3);
  };

  const hasAny = series.length > 0;
  // Dashboard renders its own per-metric series, so `series` (the single-metric one)
  // doesn't reflect it — check whether ANY model has ANY of the dashboard metrics.
  const hasAnyDashboardData = charted.some((e) => {
    const c = curves[e.entryKey ?? e.model_id];
    return c && METRIC_OPTIONS.some((m) => (c.series[m.value as MetricKey]?.length ?? 0) > 0);
  });

  // Log scale applies to loss/grad-norm views (not LR) and needs positive y.
  // The log-Y toggle is offered for any loss-bearing view: detail, dashboard (its
  // loss mini-charts apply log per-metric via useLogHere), or compare on a non-LR
  // metric. `useLog` only drives the single compare/detail chart below.
  const isLossView = mode === "dashboard" || mode === "detail" || metric !== "learningRate";
  const useLog = logY && isLossView && mode !== "dashboard";

  // A log y-scale blanks/breaks the chart on any y <= 0 (grad-norm can be 0 early,
  // loss can underflow to 0 on tiny data). When log is on, drop non-positive points
  // (and threshold-line series with a non-positive value) so the plot still renders.
  const plotSeries = useLog
    ? series
        .map((s) =>
          s.type === "threshold"
            ? s
            : { ...s, data: (s.data ?? []).filter((p: { y: number }) => p.y > 0) }
        )
        .filter((s) => s.type === "threshold" ? (s.y as number) > 0 : (s.data?.length ?? 0) > 0)
    : series;

  return (
    <Container
      header={
        <Header
          variant="h3"
          description={
            mode === "detail"
              ? "Training vs validation loss for one model — watch the gap: if validation flattens or rises while training keeps falling, the model is overfitting."
              : mode === "dashboard"
              ? "Every metric at once, one line per model — pulled from CloudWatch. No need to switch metrics one by one."
              : "One metric across all models — pulled from CloudWatch. Use it to compare convergence and right-size epochs."
          }
          actions={
            <SpaceBetween direction="horizontal" size="xs" alignItems="end">
              <FormField label="X axis">
                <SegmentedControl
                  selectedId={xAxis}
                  onChange={({ detail }) => setXAxis(detail.selectedId as XAxis)}
                  options={[
                    { id: "epoch", text: "Epoch" },
                    { id: "minutes", text: "Time" },
                  ]}
                />
              </FormField>
              {isLossView && (
                <FormField label="Y scale">
                  <SegmentedControl
                    selectedId={logY ? "log" : "linear"}
                    onChange={({ detail }) => setLogY(detail.selectedId === "log")}
                    options={[
                      { id: "linear", text: "Linear" },
                      { id: "log", text: "Log" },
                    ]}
                  />
                </FormField>
              )}
              {/* TensorBoard-style smoothing — meaningful for the noisy loss/grad-norm
                  curves, not the deterministic LR schedule. Always available in
                  dashboard (it shows loss charts) + detail; in compare, only when the
                  selected metric isn't the LR schedule. */}
              {(mode === "dashboard" || mode === "detail" || metric !== "learningRate") && (
                <FormField label={`Smoothing ${smoothing ? `(${smoothing.toFixed(2)})` : ""}`}>
                  <div style={{ width: 130 }}>
                    <Slider
                      value={smoothing}
                      onChange={({ detail }) => setSmoothing(detail.value)}
                      min={0}
                      max={0.95}
                      step={0.05}
                      ariaLabel="EMA smoothing factor"
                    />
                  </div>
                </FormField>
              )}
            </SpaceBetween>
          }
        >
          Training curve
        </Header>
      }
    >
      {/* View-mode selector — LEFT-aligned as the primary control of this panel (not
          tucked in the right-aligned header actions). Choosing dashboard/compare/
          single-model is the first decision, so it leads. */}
      <Box padding={{ bottom: "s" }}>
        <SegmentedControl
          selectedId={mode}
          onChange={({ detail }) => setMode(detail.selectedId as ViewMode)}
          options={[
            { id: "dashboard", text: "Dashboard" },
            { id: "compare", text: "Compare models" },
            { id: "detail", text: "Single model" },
          ]}
        />
      </Box>
      {/* Compare mode: the single-metric picker, left-aligned under the modes.
          Capped width — a metric name is short, so a full-width Select looks unwieldy. */}
      {mode === "compare" && (
        <Box padding={{ bottom: "s" }}>
          <FormField label="Metric">
            <div style={{ maxWidth: 260 }}>
              <Select
                selectedOption={METRIC_OPTIONS.find((o) => o.value === metric) ?? null}
                onChange={({ detail }) => setMetric(detail.selectedOption.value as MetricKey)}
                options={METRIC_OPTIONS as unknown as { value: string; label: string }[]}
              />
            </div>
          </FormField>
        </Box>
      )}
      {/* Single-model picker — its OWN prominent row right under the modes so
          choosing which model to inspect is easy to find. Detail mode only.
          Capped width so a short model name doesn't stretch across the panel. */}
      {mode === "detail" && charted.length > 0 && (
        <Box padding={{ bottom: "s" }}>
          <FormField label="Model to inspect">
            <div style={{ maxWidth: 320 }}>
              <Select
                selectedOption={
                  detailEntry
                    ? { value: detailEntry.entryKey ?? detailEntry.model_id, label: detailEntry.model_display }
                    : null
                }
                onChange={({ detail }) => setDetailModel(detail.selectedOption.value ?? null)}
                options={charted.map((e) => ({ value: e.entryKey ?? e.model_id, label: e.model_display }))}
              />
            </div>
          </FormField>
        </Box>
      )}
      {/* Health insight + numeric readout describe ONE focused model, so they only
          show in single-model (detail) mode. In dashboard/compare they'd misleadingly
          read as if they summarize ALL models when they only reflect the first one. */}
      {mode === "detail" && health && hasAny && (
        <Box padding={{ bottom: "s" }}>
          <Alert type={health.kind === "warn" ? "warning" : health.kind === "good" ? "success" : "info"}>
            {health.text}
          </Alert>
        </Box>
      )}
      {mode === "detail" && readout && hasAny && (
        <Box padding={{ bottom: "s" }}>
          <ColumnLayout columns={4} variant="text-grid">
            <div>
              <Box variant="awsui-key-label">Latest train loss</Box>
              <Box variant="h3">{fmtLoss(readout.latestTrain)}</Box>
            </div>
            <div>
              <Box variant="awsui-key-label">Latest validation loss</Box>
              <Box variant="h3">{fmtLoss(readout.latestEval)}</Box>
            </div>
            <div>
              <Box variant="awsui-key-label">Best validation loss (exported)</Box>
              <Box variant="h3">
                {fmtLoss(readout.bestEval)}
                {readout.bestEpoch != null && (
                  <Box variant="small" color="text-status-inactive" display="inline">
                    {" "}
                    @ epoch {readout.bestEpoch.toFixed(2)}
                  </Box>
                )}
              </Box>
            </div>
            <div>
              <Box variant="awsui-key-label">Current epoch</Box>
              <Box variant="h3">{readout.curEpoch != null ? readout.curEpoch.toFixed(2) : "—"}</Box>
            </div>
          </ColumnLayout>
        </Box>
      )}
      {mode === "dashboard" ? (
        // Small-multiples: one mini-chart per metric, each with one line per model.
        // Everything at a glance — no metric dropdown to drive. Shares the x-axis,
        // smoothing, and (for loss/grad-norm) the log toggle with the other modes.
        !fetched ? (
          <Box textAlign="center" color="inherit" padding="m">
            <Spinner /> <span>Loading curves…</span>
          </Box>
        ) : !hasAnyDashboardData ? (
          <Box textAlign="center" color="inherit" padding="m">
            <b>No curve data yet.</b>
            <Box variant="p" color="inherit">
              {anyTraining
                ? "Training just started — curves appear within a minute or two as logs reach CloudWatch."
                : "These models were trained before live curves were enabled, or logged no metrics."}
            </Box>
          </Box>
        ) : (
          <>
            {/* When the per-chart legends are hidden (>4 models, to avoid clutter in
                each small-multiple), show ONE shared legend so line→model is still
                readable. Colours match COLORS, the same order seriesForMetric uses. */}
            {charted.length > 4 && (
              <Box padding={{ bottom: "s" }}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 16px" }}>
                  {charted.map((e, i) => (
                    <span
                      key={e.entryKey ?? e.model_id}
                      style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}
                    >
                      <span style={{ width: 14, height: 3, background: COLORS[i % COLORS.length], display: "inline-block", borderRadius: 1 }} />
                      {e.model_display}
                    </span>
                  ))}
                </div>
              </Box>
            )}
            <ColumnLayout columns={2}>
              {METRIC_OPTIONS.map((m) => {
                const isLossMetric = m.value !== "learningRate";
              const useLogHere = logY && isLossMetric;
              let s = seriesForMetric(m.value as MetricKey);
              if (useLogHere) {
                s = s.map((ln) => ({ ...ln, data: ln.data.filter((p: Pt) => p.y > 0) })).filter((ln) => ln.data.length > 0);
              }
              return (
                <div key={m.value}>
                  <Box variant="awsui-key-label" padding={{ bottom: "xxs" }}>{m.label}</Box>
                  <LineChart
                    series={s as any}
                    height={200}
                    xScaleType="linear"
                    yScaleType={useLogHere ? "log" : "linear"}
                    xTitle={xAxis === "epoch" ? "Epoch" : "Minutes"}
                    yTitle={m.label}
                    hideFilter
                    hideLegend={charted.length > 4}
                    i18nStrings={{
                      xTickFormatter: (v) => (xAxis === "epoch" ? (v as number).toFixed(1) : `${(v as number).toFixed(0)}m`),
                      // LR → exponential (tiny magnitudes); others → 3 sig-ish digits
                      // but switch to exponential for very small non-zero values so a
                      // grad-norm/loss tick doesn't collapse to a misleading "0.00".
                      yTickFormatter: (v) => {
                        const n = v as number;
                        if (m.value === "learningRate") return n.toExponential(0);
                        if (n !== 0 && Math.abs(n) < 0.01) return n.toExponential(1);
                        return n.toFixed(2);
                      },
                    }}
                    ariaLabel={`${m.label} across models`}
                    empty={<Box textAlign="center" color="text-status-inactive" padding="s">No {m.label.toLowerCase()} logged yet.</Box>}
                  />
                </div>
                );
              })}
            </ColumnLayout>
          </>
        )
      ) : (
        <LineChart
          series={plotSeries}
          height={260}
          xScaleType="linear"
          yScaleType={useLog ? "log" : "linear"}
          xTitle={xAxis === "epoch" ? "Epoch" : "Minutes since training start"}
          yTitle={useLog ? `${yLabel} (log)` : yLabel}
          i18nStrings={{
            xTickFormatter: (v) => (xAxis === "epoch" ? (v as number).toFixed(2) : `${(v as number).toFixed(0)}m`),
            yTickFormatter: (v) => fmtY(v as number),
          }}
          ariaLabel="Training loss chart"
          empty={
            !fetched ? (
              <Box textAlign="center" color="inherit" padding="m">
                <Spinner /> <span>Loading curves…</span>
              </Box>
            ) : (
              <Box textAlign="center" color="inherit" padding="m">
                <b>No curve data yet.</b>
                <Box variant="p" color="inherit">
                  {anyTraining
                    ? "Training just started — the curve appears within a minute or two as logs reach CloudWatch. Eval loss only appears once the first evaluation runs (every save/eval-steps)."
                    : "These models were trained before live curves were enabled, or logged no metrics."}
                </Box>
              </Box>
            )
          }
          noMatch={<Box textAlign="center">No data</Box>}
        />
      )}
      {hasAny && (anyTraining || loading) && (
        <Box variant="small" color="text-status-inactive" padding={{ top: "xs" }}>
          Refreshing every 20s while training is in progress.
        </Box>
      )}
    </Container>
  );
}
