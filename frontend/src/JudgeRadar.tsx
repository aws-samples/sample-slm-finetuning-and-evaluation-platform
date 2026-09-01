// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// LLM-judge radar — the 5 quality dimensions (correctness / faithfulness / format /
// completeness / conciseness, each 1-5) the judge already scores, surfaced as an
// overlaid radar so you can SEE each model's shape instead of digging into a hover
// popover one model at a time. Recharts (Cloudscape has no radar).
//
// One coloured polygon per judged model (same series colour as the scatter/table).
// Only models with a completed judge result + per-dimension breakdown appear; the
// caller passes those in. An empty set renders a clear "run the judge" hint.
import { useMemo, useState } from "react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import Box from "@cloudscape-design/components/box";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import FormField from "@cloudscape-design/components/form-field";
import Multiselect, { type MultiselectProps } from "@cloudscape-design/components/multiselect";

import { useIsDark, chartColors, seriesColor } from "./chartTheme";

export interface JudgeRadarModel {
  model: string;
  colorIdx: number; // index into the shared palette (keeps colours stable vs. table)
  dimensions: Record<string, number>; // dimension → mean (1-5)
}

// Overlaying every judged model makes the radar an unreadable mush once you have
// more than a few, so default to showing at most this many; the rest are opt-in
// via the multi-select.
const DEFAULT_MAX_OVERLAY = 4;

// Title-case a snake/lower dimension key for the axis label.
function prettyDim(d: string): string {
  return d.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

export function JudgeRadar({ models }: { models: JudgeRadarModel[] }) {
  const dark = useIsDark();
  const c = chartColors(dark);

  // Which models to overlay. `selected === null` means "user hasn't chosen" → use
  // the default (first few, readable out of the box even with many judged models).
  // Once the user touches the picker we honour their EXACT choice — including an
  // empty selection (they can clear the chart; we show an empty state, no snap-back).
  const allNames = models.map((m) => m.model);
  const namesKey = allNames.join("|");
  const [selected, setSelected] = useState<string[] | null>(null);
  const validSelected = useMemo(() => {
    if (selected === null) return allNames.slice(0, DEFAULT_MAX_OVERLAY); // default view
    return selected.filter((n) => allNames.includes(n)); // honour user choice (may be [])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [namesKey, selected]);

  const shown = models.filter((m) => validSelected.includes(m.model));

  // Union of dimensions across the SHOWN models (stable order), so every polygon
  // shares the same axes even if one is missing a dimension (rare).
  const dimSet: string[] = [];
  for (const m of shown) {
    for (const d of Object.keys(m.dimensions)) if (!dimSet.includes(d)) dimSet.push(d);
  }

  // Recharts wants one row per axis: { dimension, <model>: value, ... }.
  const data = dimSet.map((d) => {
    const row: Record<string, number | string> = { dimension: prettyDim(d) };
    shown.forEach((m) => {
      const v = m.dimensions[d];
      if (v != null) row[m.model] = v;
    });
    return row;
  });

  const options: MultiselectProps.Option[] = models.map((m) => ({ value: m.model, label: m.model }));

  return (
    <Container
      header={
        <Header
          variant="h3"
          description="The LLM judge's per-dimension scores (1–5), overlaid so you can compare each model's profile — e.g. high correctness but weak conciseness. Judge a model from the table to add it here."
          actions={
            models.length > 1 ? (
              <FormField label="Models to overlay">
                <div style={{ minWidth: 240 }}>
                  <Multiselect
                    selectedOptions={options.filter((o) => validSelected.includes(o.value!))}
                    onChange={({ detail }) => setSelected(detail.selectedOptions.map((o) => o.value!).filter(Boolean))}
                    options={options}
                    placeholder="Pick models to compare"
                    tokenLimit={3}
                  />
                </div>
              </FormField>
            ) : undefined
          }
        >
          Judge profile (per dimension)
        </Header>
      }
    >
      {models.length === 0 ? (
        <Box textAlign="center" color="text-status-inactive" padding="l">
          No per-dimension judge scores yet. Click <b>Judge</b> on a model row (LLM judge column) — once
          it finishes, its 5-dimension profile appears here.
        </Box>
      ) : shown.length === 0 || dimSet.length === 0 ? (
        // User deselected everything — honour it (no snap-back); prompt to pick one.
        <Box textAlign="center" color="text-status-inactive" padding="l">
          No models selected. Pick one or more in <b>Models to overlay</b> to compare their judge profiles.
        </Box>
      ) : (
        <div style={{ width: "100%", height: 340 }}>
          <ResponsiveContainer>
            <RadarChart data={data} outerRadius="72%">
              <PolarGrid stroke={c.grid} />
              <PolarAngleAxis dataKey="dimension" tick={{ fill: c.text, fontSize: 12 }} />
              <PolarRadiusAxis domain={[0, 5]} tickCount={6} tick={{ fill: c.text, fontSize: 10 }} stroke={c.axis} />
              {shown.map((m) => (
                <Radar
                  key={m.model}
                  name={m.model}
                  dataKey={m.model}
                  stroke={seriesColor(m.colorIdx)}
                  fill={seriesColor(m.colorIdx)}
                  fillOpacity={shown.length > 1 ? 0.15 : 0.35}
                />
              ))}
              <Legend wrapperStyle={{ fontSize: 12, color: c.text }} />
              <Tooltip
                contentStyle={{ background: c.tooltipBg, border: `1px solid ${c.tooltipBorder}`, borderRadius: 8, color: c.text, fontSize: 12 }}
                formatter={(value: number | string, name: string) => [`${(value as number).toFixed(2)} / 5`, name]}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Container>
  );
}
