// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Quality-vs-cost scatter for the leaderboard — the "is the fine-tuned SLM worth
// it vs. the frontier baseline?" decision in one picture, which the numbers table
// can't convey at a glance. Recharts (Cloudscape has no scatter).
//
//   x = cost per 1k rows  (fine-tuned: projected self-host; baseline: actual API)
//   y = the ACTIVE ranking metric (follows the leaderboard's "Rank by")
//   point = a model;  baselines drawn as a diamond, fine-tunes as a circle
//   the BEST quadrant is top-LEFT (high quality, low cost)
//
// Points with no cost or no metric are dropped (can't place them) and counted in a
// caption so nothing is silently hidden.
import { useMemo, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
  Cell,
} from "recharts";
import Box from "@cloudscape-design/components/box";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import FormField from "@cloudscape-design/components/form-field";
import Multiselect, { type MultiselectProps } from "@cloudscape-design/components/multiselect";

import type { LeaderboardRow } from "./api";
import { useIsDark, chartColors, seriesColor } from "./chartTheme";
import { wilsonInterval } from "./stats";

// Cost per 1k for a row: baselines carry the actual API price; fine-tunes the
// projected self-host estimate. One axis, honest label in the tooltip.
function rowCost(r: LeaderboardRow): number | null {
  return r.isBaseline ? r.apiCostPer1k ?? null : r.projectedServeCostPer1k ?? null;
}

interface Pt {
  x: number;          // cost/1k
  y: number;          // ranking-metric value (0..1)
  z: number;          // latency ms (point size), 0 when unknown
  yErr: number;       // 95% Wilson half-width on y (0 when n unknown) — error bar
  model: string;
  isBaseline: boolean;
  colorIdx: number;
  latency: number | null;
}

export function QualityCostScatter({
  rows,
  rankBy,
  rankLabel,
  colorIdxFor,
}: {
  rows: LeaderboardRow[];
  rankBy: keyof LeaderboardRow;
  rankLabel: string;
  // Stable per-MODEL colour index (keyed on model name), so a model keeps its hue
  // when the leaderboard is re-sorted by a different metric. Falls back to 0.
  colorIdxFor: (model: string) => number;
}) {
  const dark = useIsDark();
  const c = chartColors(dark);

  // Build ALL plottable points; drop any without BOTH a cost and a metric value.
  let dropped = 0;
  const allPts: Pt[] = [];
  rows.forEach((r) => {
    const cost = rowCost(r);
    const yv = r[rankBy] as number | null;
    if (cost == null || yv == null) {
      dropped += 1;
      return;
    }
    const ci = wilsonInterval(yv, r.count);
    allPts.push({
      x: cost,
      y: yv,
      z: r.p50LatencyMs ?? 0,
      yErr: ci?.half ?? 0, // 95% margin on the quality metric (small test sets = wide)
      model: r.model,
      isBaseline: r.isBaseline,
      colorIdx: colorIdxFor(r.model), // stable per model, not list position
      latency: r.p50LatencyMs ?? null,
    });
  });

  // Which models to plot. `selected === null` = default view: ALL fine-tunes, and
  // baselines OFF — a frontier API baseline costs orders of magnitude more, so
  // including it by default skews the log x-axis and shrinks the fine-tune cluster.
  // The baseline IS the "is the SLM worth it?" comparison though, so it stays one
  // click away in the picker. Once the user touches the picker we honour their exact
  // choice (including empty → an empty state, no snap-back).
  const allNames = allPts.map((p) => p.model);
  const namesKey = allNames.join("|");
  const [selected, setSelected] = useState<string[] | null>(null);
  const shownNames = useMemo(() => {
    if (selected === null) return new Set(allPts.filter((p) => !p.isBaseline).map((p) => p.model));
    return new Set(selected.filter((n) => allNames.includes(n)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namesKey, selected]);

  const pts = allPts.filter((p) => shownNames.has(p.model));
  const baselinePts = pts.filter((p) => p.isBaseline);
  const modelPts = pts.filter((p) => !p.isBaseline);
  const anyLatency = pts.some((p) => p.latency != null);
  const hiddenBaselines = allPts.filter((p) => p.isBaseline && !shownNames.has(p.model)).length;

  // Picker options: fine-tunes first, then baselines (labelled), so the baseline is
  // easy to find when you DO want the comparison.
  const msOptions: MultiselectProps.Option[] = [
    ...allPts.filter((p) => !p.isBaseline).map((p) => ({ value: p.model, label: p.model })),
    ...allPts.filter((p) => p.isBaseline).map((p) => ({ value: p.model, label: `${p.model} (baseline)` })),
  ];

  // All ranking metrics are normalized fractions (0–1) by the backend, so the y-axis
  // is [0, 1]. Guard defensively: if a value ever exceeds 1 (a future non-fraction
  // metric), grow the top so points aren't silently clipped to the axis edge.
  const yMax = Math.max(1, ...pts.map((p) => p.y));

  const fmtPct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const fmtUsd = (v: number) =>
    v >= 1 ? `$${v.toFixed(2)}` : v >= 0.01 ? `$${v.toFixed(3)}` : `$${v.toFixed(5)}`;

  // Costs span orders of magnitude (a self-host SLM ≈ $0.0001/1k vs. a frontier API
  // baseline ≈ $0.8/1k), so a LINEAR x-axis crushes every fine-tune into a single
  // blob at the left edge (the unreadable chart). A LOG x-axis spreads them out.
  // Log needs x>0: clamp any non-positive cost to a tiny floor so it still plots.
  // Domain is computed from the SHOWN points, so hiding the pricey baseline tightens
  // the cluster automatically. Falls back to a sane range when nothing is shown.
  const COST_FLOOR = 1e-5;
  pts.forEach((p) => { if (p.x <= 0) p.x = COST_FLOOR; });
  const costs = pts.map((p) => p.x);
  const xMin = costs.length ? Math.min(...costs) : COST_FLOOR;
  const xMax = costs.length ? Math.max(...costs) : 1;
  // Pad the log domain by a factor so edge points aren't on the axis line.
  const xDomain: [number, number] = [Math.max(COST_FLOOR, xMin / 2), xMax * 2];

  return (
    <Container
      header={
        <Header
          variant="h3"
          description="Each model as a point — quality (the ranking metric) against cost per 1k rows. Top-left is best (high quality, low cost). Baselines are diamonds (hidden by default — they cost far more and skew the axis; add them to compare); point size = p50 latency; vertical whiskers = the 95% margin on quality (overlap ≈ a tie)."
          actions={
            allPts.length > 1 ? (
              <FormField label="Models shown">
                <div style={{ minWidth: 240 }}>
                  <Multiselect
                    selectedOptions={msOptions.filter((o) => shownNames.has(o.value!))}
                    onChange={({ detail }) => setSelected(detail.selectedOptions.map((o) => o.value!).filter(Boolean))}
                    options={msOptions}
                    placeholder="Pick models to plot"
                    tokenLimit={3}
                  />
                </div>
              </FormField>
            ) : undefined
          }
        >
          Quality vs. cost
        </Header>
      }
    >
      {allPts.length === 0 ? (
        <Box textAlign="center" color="text-status-inactive" padding="l">
          No points to plot yet — needs at least one model with both a {rankLabel} score and a cost.
        </Box>
      ) : pts.length === 0 ? (
        <Box textAlign="center" color="text-status-inactive" padding="l">
          No models selected. Pick one or more in <b>Models shown</b> to plot them
          {hiddenBaselines > 0 ? " (baselines are available there too)" : ""}.
        </Box>
      ) : (
        <>
          <div style={{ width: "100%", height: 320 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 24, bottom: 36, left: 8 }}>
                <CartesianGrid stroke={c.grid} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Cost / 1k rows"
                  scale="log"
                  domain={xDomain}
                  allowDataOverflow
                  tickFormatter={fmtUsd}
                  stroke={c.axis}
                  tick={{ fill: c.text, fontSize: 12 }}
                  label={{ value: "Cost / 1k rows · log scale (lower = better →)", position: "bottom", offset: 18, fill: c.text, fontSize: 12 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name={rankLabel}
                  domain={[0, yMax]}
                  tickFormatter={fmtPct}
                  stroke={c.axis}
                  tick={{ fill: c.text, fontSize: 12 }}
                  label={{ value: `${rankLabel} (higher = better ↑)`, angle: -90, position: "insideLeft", fill: c.text, fontSize: 12, style: { textAnchor: "middle" } }}
                />
                {/* Point size ∝ latency when we have it; a fixed range keeps dots readable. */}
                <ZAxis type="number" dataKey="z" range={anyLatency ? [60, 360] : [120, 120]} name="p50 latency" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3", stroke: c.axis }}
                  // Custom content so the tooltip leads with the MODEL NAME (the
                  // default scatter tooltip only lists x/y/z values, not the label).
                  content={({ active, payload }) => {
                    if (!active || !payload || payload.length === 0) return null;
                    const p = payload[0].payload as Pt;
                    return (
                      <div style={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 8, color: c.text, padding: "8px 10px", fontSize: 12 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>
                          {p.model}{p.isBaseline ? " (baseline)" : ""}
                        </div>
                        <div>{rankLabel}: <b>{fmtPct(p.y)}</b></div>
                        <div>Cost / 1k: <b>{fmtUsd(p.x)}</b>{p.isBaseline ? " (API)" : " (projected)"}</div>
                        <div>p50 latency: <b>{p.latency != null ? `${Math.round(p.latency)} ms` : "—"}</b></div>
                      </div>
                    );
                  }}
                />
                {/* Fine-tuned models: circles, each its own series colour. The
                    vertical ErrorBar is the 95% margin on the quality metric — wide
                    whiskers (small test set) mean a close ranking is really a tie. */}
                <Scatter name="Fine-tuned" data={modelPts} fill={seriesColor(0)} shape="circle">
                  {modelPts.map((p) => (
                    <Cell key={p.model} fill={seriesColor(p.colorIdx)} />
                  ))}
                  <ErrorBar dataKey="yErr" direction="y" width={4} strokeWidth={1.5} stroke={c.axis} />
                </Scatter>
                {/* Baselines: diamonds, a steady reference hue. */}
                <Scatter name="Baseline" data={baselinePts} fill={c.reference} shape="diamond">
                  {baselinePts.map((p) => (
                    <Cell key={p.model} fill={c.reference} />
                  ))}
                  <ErrorBar dataKey="yErr" direction="y" width={4} strokeWidth={1.5} stroke={c.reference} />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          {/* A compact legend of which colour is which model (Recharts dots are by
              index; the table above is the canonical list, this just bridges them). */}
          <Box padding={{ top: "s" }}>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 16px" }}>
              {pts.map((p) => (
                <span key={p.model} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: c.text }}>
                  <span
                    style={{
                      width: 10, height: 10,
                      background: p.isBaseline ? c.reference : seriesColor(p.colorIdx),
                      transform: p.isBaseline ? "rotate(45deg)" : undefined,
                      borderRadius: p.isBaseline ? 0 : "50%",
                      display: "inline-block",
                    }}
                  />
                  {p.model}
                </span>
              ))}
            </div>
            {hiddenBaselines > 0 && (
              <Box variant="small" color="text-status-inactive" padding={{ top: "xs" }}>
                {hiddenBaselines} frontier baseline{hiddenBaselines === 1 ? "" : "s"} hidden (cost far more — they'd
                skew the axis). Add {hiddenBaselines === 1 ? "it" : "them"} in <b>Models shown</b> for the
                "is the SLM worth it?" comparison.
              </Box>
            )}
            {dropped > 0 && (
              <Box variant="small" color="text-status-inactive" padding={{ top: "xs" }}>
                {dropped} model{dropped === 1 ? "" : "s"} not plotted (missing a {rankLabel} score or a cost).
              </Box>
            )}
            {!anyLatency && (
              <Box variant="small" color="text-status-inactive" padding={{ top: "xxs" }}>
                Point size is uniform — no p50 latency recorded for these rows yet.
              </Box>
            )}
          </Box>
        </>
      )}
    </Container>
  );
}
