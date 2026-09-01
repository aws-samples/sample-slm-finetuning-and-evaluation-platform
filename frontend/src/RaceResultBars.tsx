// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Ranked-bar "race result" for the Runs detail pane — a head-to-head comparison of
// the run's models on the active ranking metric, right where you watch a race
// finish (the leaderboard's scatter is a separate page). Horizontal bars because on
// one run you usually have a few models and care about RANK, not the cost trade-off.
//
// Two modes:
//   - score : the fine-tuned model's ABSOLUTE metric (e.g. Token F1) — "who's best".
//   - lift  : fine-tuned MINUS the untrained base on the same metric, in percentage
//             POINTS — "what did fine-tuning actually ADD" (the Fine-tune lift column,
//             as a chart). Green = helped, red = hurt. Needs base_metrics.
//
// Everything is computed CLIENT-SIDE from the already-loaded entry metrics, so the
// metric/mode toggles are instant (no server round-trip). A faint error bar shows
// the 95% margin on the metric (small test set ⇒ wide ⇒ a close lead is a tie).
import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
  ReferenceLine,
  Cell,
} from "recharts";
import Box from "@cloudscape-design/components/box";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SegmentedControl from "@cloudscape-design/components/segmented-control";
import SpaceBetween from "@cloudscape-design/components/space-between";

import type { RaceEntry, EvalMetrics } from "./api";
import { useIsDark, chartColors, seriesColor } from "./chartTheme";
import { wilsonInterval } from "./stats";

type Mode = "score" | "lift";

interface Bar1 {
  model: string;
  value: number;        // score (0..1) or lift (signed points, -1..1)
  err: number;          // 95% Wilson half-width on the fine-tuned score
  colorIdx: number;
  isWinner: boolean;    // best by SCORE (the absolute winner, mode-independent)
  base: number | null;  // base score (for the tooltip in lift mode)
  tuned: number;        // fine-tuned score
}

// Pull a metric value out of an EvalMetrics blob by its rank key.
function metricVal(m: EvalMetrics | null | undefined, key: string): number | null {
  if (!m) return null;
  const v = (m as unknown as Record<string, unknown>)[key];
  return typeof v === "number" ? v : null;
}

export function RaceResultBars({
  entries,
  rankMetric,
  rankLabel,
}: {
  entries: RaceEntry[];
  rankMetric: string; // EvalMetrics key, e.g. "token_f1"
  rankLabel: string;
}) {
  const dark = useIsDark();
  const c = chartColors(dark);
  const [mode, setMode] = useState<Mode>("score");

  // Done entries with a real fine-tuned score, ranked best→worst BY SCORE (so the
  // winner is stable regardless of the score/lift toggle). Computed here from the
  // loaded metrics — switching metric or mode never hits the server.
  const ranked = entries
    .map((e, i) => {
      const tuned = metricVal(e.metrics, rankMetric);
      if (e.state !== "done" || tuned == null) return null;
      const base = metricVal(e.base_metrics, rankMetric);
      const ci = wilsonInterval(tuned, e.metrics?.count ?? null);
      return {
        model: e.model_display,
        tuned,
        base,
        err: ci?.half ?? 0,
        colorIdx: i, // stable colour vs. the model's training curve
      };
    })
    .filter((b): b is NonNullable<typeof b> => b !== null)
    .sort((a, b) => b.tuned - a.tuned);

  // Lift needs a base score; only offer the toggle when at least one entry has one.
  const anyBase = ranked.some((b) => b.base != null);

  const bars: Bar1[] = ranked.map((b, idx) => ({
    model: b.model,
    value: mode === "lift" ? (b.base != null ? b.tuned - b.base : 0) : b.tuned,
    err: b.err,
    colorIdx: b.colorIdx,
    isWinner: idx === 0, // top by score
    base: b.base,
    tuned: b.tuned,
  }));

  const fmtPct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const fmtPts = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)} pts`;

  // Lift can be negative; give the x-axis symmetric-ish room around 0. Score is [0,1+].
  const liftMax = Math.max(0.05, ...bars.map((b) => Math.abs(b.value) + b.err));
  const xDomain: [number, number] =
    mode === "lift" ? [-liftMax, liftMax] : [0, Math.max(1, ...bars.map((b) => b.value + b.err))];

  return (
    <Container
      header={
        <Header
          variant="h3"
          description={
            mode === "lift"
              ? `Fine-tune LIFT on ${rankLabel} — trained minus the untrained base, in points. Green = fine-tuning helped, red = it hurt. Whiskers = 95% margin.`
              : `Finished models ranked by ${rankLabel} on the held-out test set — same rows + decoding for each. Whiskers = 95% margin (overlap ≈ a tie).`
          }
          actions={
            anyBase ? (
              <SegmentedControl
                selectedId={mode}
                onChange={({ detail }) => setMode(detail.selectedId as Mode)}
                options={[
                  { id: "score", text: "Score" },
                  { id: "lift", text: "Fine-tune lift" },
                ]}
              />
            ) : undefined
          }
        >
          Race result
        </Header>
      }
    >
      {bars.length === 0 ? (
        <Box textAlign="center" color="text-status-inactive" padding="m">
          No finished models yet — bars appear as each model completes its evaluation.
        </Box>
      ) : (
        <SpaceBetween size="xs">
          {mode === "lift" && !anyBase && (
            <Box variant="small" color="text-status-inactive">
              No base-model scores recorded for this run, so lift isn't available.
            </Box>
          )}
          <div style={{ width: "100%", height: Math.max(120, 56 + bars.length * 44) }}>
            <ResponsiveContainer>
              <BarChart data={bars} layout="vertical" margin={{ top: 8, right: 56, bottom: 8, left: 8 }}>
                <CartesianGrid stroke={c.grid} strokeDasharray="3 3" horizontal={false} />
                <XAxis
                  type="number"
                  domain={xDomain}
                  tickFormatter={(v) => (mode === "lift" ? `${(v * 100).toFixed(0)}` : fmtPct(v as number))}
                  stroke={c.axis}
                  tick={{ fill: c.text, fontSize: 12 }}
                />
                <YAxis type="category" dataKey="model" width={150} stroke={c.axis} tick={{ fill: c.text, fontSize: 12 }} />
                {mode === "lift" && <ReferenceLine x={0} stroke={c.axis} />}
                <Tooltip
                  cursor={{ fill: dark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)" }}
                  content={({ active, payload }) => {
                    if (!active || !payload || payload.length === 0) return null;
                    const b = payload[0].payload as Bar1;
                    return (
                      <div style={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 8, color: c.text, padding: "8px 10px", fontSize: 12 }}>
                        <div style={{ fontWeight: 600, marginBottom: 4 }}>{b.model}{b.isWinner ? " ★" : ""}</div>
                        {mode === "lift" ? (
                          <>
                            <div>Lift: <b>{b.base != null ? fmtPts(b.tuned - b.base) : "—"}</b></div>
                            <div style={{ color: c.text, opacity: 0.8 }}>base {b.base != null ? fmtPct(b.base) : "—"} → tuned {fmtPct(b.tuned)}</div>
                          </>
                        ) : (
                          <div>{rankLabel}: <b>{fmtPct(b.tuned)}</b> {b.err > 0 && <>± {(b.err * 100).toFixed(0)}%</>}</div>
                        )}
                      </div>
                    );
                  }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
                  {bars.map((b) => {
                    // Score mode: the model's palette colour (winner full, others muted).
                    // Lift mode: green if it helped, red if it hurt — the at-a-glance read.
                    const fill =
                      mode === "lift"
                        ? b.value >= 0 ? "#1d8102" : "#c33d69"
                        : seriesColor(b.colorIdx);
                    return <Cell key={b.model} fill={fill} fillOpacity={mode === "lift" || b.isWinner ? 1 : 0.55} />;
                  })}
                  <ErrorBar dataKey="err" direction="x" width={4} strokeWidth={1.5} stroke={c.axis} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          {mode === "lift" && (
            <Box variant="small" color="text-status-inactive">
              Lift isolates what the fine-tuning ADDED (vs. the untrained base) — distinct from the bar
              height in "Score", which is the model's absolute quality. A model can rank #1 by score yet
              have small/negative lift (it was already good), or rank low yet have large lift.
            </Box>
          )}
        </SpaceBetween>
      )}
    </Container>
  );
}
