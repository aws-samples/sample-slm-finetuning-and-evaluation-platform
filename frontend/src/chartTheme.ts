// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Shared theming for the Recharts-based comparison visuals (scatter, radar).
//
// We keep Cloudscape's built-in LineChart for the training curves (it's themed +
// accessible out of the box), and use Recharts ONLY for the chart types Cloudscape
// lacks (scatter, radar). To keep those looking native, this module exposes:
//   - useIsDark()  — live light/dark flag, read off the <body> mode class Cloudscape
//                     toggles (applyMode), so charts re-render on a theme switch.
//   - chartColors()— axis/grid/text colors per mode (approximating Cloudscape tokens).
//   - SERIES_COLORS — the same categorical palette LossChart uses, so a model is the
//                     same colour across every chart.
import { useSyncExternalStore } from "react";

// Categorical palette — identical to LossChart's COLORS so one model reads as the
// same hue in the training curve, the scatter, and the radar.
export const SERIES_COLORS = [
  "#688ae8", // blue
  "#c33d69", // red
  "#2ea597", // teal
  "#e07941", // orange
  "#8456ce", // purple
  "#3184c2", // cyan-blue
  "#e3b505", // amber
  "#1d8102", // green
];

// Stable colour for a model index (wraps the palette). Callers assign the index
// once per model from a fixed order (see LeaderboardPage.modelColorIdx) so a model
// keeps its hue across re-sorts — the index, not this fn, is what's kept stable.
export function seriesColor(i: number): string {
  return SERIES_COLORS[i % SERIES_COLORS.length];
}

// Cloudscape applies `awsui-dark-mode` to <body> via applyMode(). We watch that
// class so Recharts (which can't read CSS design tokens directly) recolours on a
// theme toggle without threading `mode` through every page as a prop.
function bodyIsDark(): boolean {
  if (typeof document === "undefined") return false;
  return document.body.classList.contains("awsui-dark-mode");
}

// ONE shared MutationObserver for the whole app (not one per chart). useIsDark()
// subscribes via useSyncExternalStore, so N charts share a single observer that's
// created lazily on the first subscriber and disconnected when the last unmounts.
let _observer: MutationObserver | null = null;
const _listeners = new Set<() => void>();

function subscribeDark(cb: () => void): () => void {
  _listeners.add(cb);
  if (!_observer && typeof document !== "undefined") {
    _observer = new MutationObserver(() => _listeners.forEach((l) => l()));
    _observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });
  }
  return () => {
    _listeners.delete(cb);
    if (_listeners.size === 0 && _observer) {
      _observer.disconnect();
      _observer = null;
    }
  };
}

export function useIsDark(): boolean {
  // getSnapshot reads the live class; getServerSnapshot returns false (SSR-safe).
  return useSyncExternalStore(subscribeDark, bodyIsDark, () => false);
}

// Axis / grid / tooltip colors approximating Cloudscape's chart tokens per mode.
export interface ChartColors {
  axis: string;       // axis lines + ticks
  text: string;       // tick + label text
  grid: string;       // gridlines (subtle)
  tooltipBg: string;  // tooltip surface
  tooltipBorder: string;
  reference: string;  // reference/threshold lines (e.g. baseline marker)
}

export function chartColors(dark: boolean): ChartColors {
  return dark
    ? {
        axis: "#5f6b7a",
        text: "#d1d5db",
        grid: "#2f3742",
        tooltipBg: "#1b232d",
        tooltipBorder: "#414d5c",
        reference: "#8d99a8",
      }
    : {
        axis: "#8d99a8",
        text: "#414d5c",
        grid: "#e9ebed",
        tooltipBg: "#ffffff",
        tooltipBorder: "#c6c6cd",
        reference: "#5f6b7a",
      };
}
