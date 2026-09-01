// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import LineChart from "@cloudscape-design/components/line-chart";
import Box from "@cloudscape-design/components/box";
import Spinner from "@cloudscape-design/components/spinner";

import { fetchRewardCurve, type RaceEntry, type RewardCurve as RewardCurveData } from "./api";

// RLVR reward trajectory: the GRPO reward over training steps (analogous to the
// loss curve for SFT/DPO). Shown only for RLVR entries — for those, "is the model
// improving?" is answered by reward going UP (loss is not meaningful for GRPO).
const COLORS = ["#688ae8", "#c33d69", "#2ea597", "#e07941", "#8456ce", "#3184c2"];

function isRlvr(e: RaceEntry): boolean {
  return (e.hp?.stage as string) === "rlvr";
}

type Series = {
  title: string;
  type: "line";
  data: { x: number; y: number }[];
  color: string;
};

export function RewardCurve({ entries }: { entries: RaceEntry[] }) {
  const rlvr = entries.filter(isRlvr);
  const [curves, setCurves] = useState<Record<string, RewardCurveData>>({});
  const [loading, setLoading] = useState(false);
  // A STABLE key for the effect: the parent re-derives `entries` (new array
  // identity) on every poll, so depending on `[entries]` would tear down and
  // recreate the 8s interval + re-fetch on every render. Key on the RLVR jobs +
  // their states instead, so the poll has a steady cadence.
  const jobsKey = rlvr.map((e) => `${e.train_job}:${e.state}`).join(",");

  // Fetch each RLVR entry's reward curve; poll while any is still training (so
  // the curve grows live). Keyed by train_job.
  useEffect(() => {
    const jobs = rlvr.map((e) => e.train_job).filter((j): j is string => !!j);
    if (jobs.length === 0) return;
    let cancelled = false;
    const anyTraining = rlvr.some((e) => e.state === "training" || e.state === "evaluating");

    function load() {
      setLoading(true);
      Promise.all(
        jobs.map((j) =>
          fetchRewardCurve(j)
            .then((c) => [j, c] as const)
            .catch(() => [j, null] as const)
        )
      )
        .then((pairs) => {
          if (cancelled) return;
          const next: Record<string, RewardCurveData> = {};
          for (const [j, c] of pairs) if (c) next[j] = c;
          setCurves(next);
        })
        .finally(() => !cancelled && setLoading(false));
    }
    load();
    const t = anyTraining ? setInterval(load, 8000) : null;
    return () => {
      cancelled = true;
      if (t) clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobsKey]);

  if (rlvr.length === 0) {
    return (
      <Box variant="small" color="text-status-inactive">
        No RLVR entries in this run — the reward curve applies to RLVR (GRPO) training only.
      </Box>
    );
  }

  // One line per RLVR entry: its mean training reward over steps. (Min/max are
  // available per-step too but we keep the compare view to one line per model for
  // readability.) A held-out val-reward line is added when present.
  const series: Series[] = [];
  rlvr.forEach((e, i) => {
    const c = e.train_job ? curves[e.train_job] : undefined;
    if (!c || !c.hasData) return;
    const color = COLORS[i % COLORS.length];
    const data = c.steps
      .map((s, idx) => ({ x: s, y: c.rewardMean[idx] }))
      .filter((p): p is { x: number; y: number } => p.y != null);
    if (data.length) series.push({ title: `${e.model_display} · train reward`, type: "line", data, color });
    if (c.valReward.length) {
      // Give the val-reward line a DISTINCT color (next slot in the palette) so it
      // doesn't read as the same series as train reward — previously both used the
      // entry's single color and were indistinguishable when overlaid.
      series.push({
        title: `${e.model_display} · val reward`,
        type: "line",
        data: c.valReward.map((v) => ({ x: v.step, y: v.value })),
        color: COLORS[(i + rlvr.length) % COLORS.length],
      });
    }
  });

  const hasAny = series.length > 0;

  return (
    <Container
      header={
        <Header
          variant="h2"
          description="GRPO reward over training steps — for RLVR, the model is improving when reward goes UP (loss isn't meaningful for reinforcement training). Held-out val reward is shown when the recipe logs it."
        >
          Reward curve (RLVR){loading ? <> <Spinner size="normal" /></> : null}
        </Header>
      }
    >
      <LineChart
        series={series as never}
        xTitle="Training step"
        yTitle="Reward"
        height={280}
        hideFilter
        empty={
          <Box textAlign="center" color="inherit">
            {hasAny ? "No reward data." : "No reward data yet — it appears once the RLVR job logs its first steps."}
          </Box>
        }
        noMatch={<Box textAlign="center">No data</Box>}
      />
    </Container>
  );
}
