// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Small-sample honesty for eval metrics. A "65%" on a 30-row test set is far less
// certain than the same 65% on 5000 rows, but the leaderboard renders them
// identically. These helpers quantify that uncertainty so the UI can show error
// bars + a small-sample caption instead of implying false precision.
//
// We use the WILSON score interval for a binomial proportion — well-behaved for
// small n and near 0/1 (unlike the naive normal approximation, which can give
// intervals outside [0,1] or zero-width at the extremes). Most of our ranking
// metrics (exact/normalized/contains/label-accuracy, and the F1/ROUGE family in
// aggregate) are means of per-row scores in [0,1], so a proportion CI is a sound,
// honest approximation of "how much could this score move on a different sample?".

// z for a 95% two-sided interval.
const Z_95 = 1.96;

export interface Interval {
  lo: number; // lower bound, clamped to [0,1]
  hi: number; // upper bound, clamped to [0,1]
  half: number; // (hi-lo)/2 — the "± margin" for compact display
}

// Wilson score interval for proportion `p` (in [0,1]) over `n` trials. Returns null
// when n is missing/zero (no basis for an interval). z defaults to 95%.
export function wilsonInterval(p: number | null | undefined, n: number | null | undefined, z = Z_95): Interval | null {
  if (p == null || n == null || n <= 0) return null;
  const pp = Math.min(1, Math.max(0, p));
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (pp + z2 / (2 * n)) / denom;
  const margin = (z * Math.sqrt((pp * (1 - pp)) / n + z2 / (4 * n * n))) / denom;
  const lo = Math.max(0, center - margin);
  const hi = Math.min(1, center + margin);
  return { lo, hi, half: (hi - lo) / 2 };
}

// A test set is "small" (worth a margin-of-error warning) below this row count.
// At n=100 a 50% score has a ±~10% Wilson half-width — still loose; below ~200 the
// ranking of close models shouldn't be over-read. 200 is a pragmatic threshold.
export const SMALL_SAMPLE_THRESHOLD = 200;

export function isSmallSample(n: number | null | undefined): boolean {
  return n != null && n > 0 && n < SMALL_SAMPLE_THRESHOLD;
}

// "± 9%" style margin string from a half-width fraction; "" when no interval.
export function marginLabel(half: number | null | undefined): string {
  if (half == null) return "";
  return `±${(half * 100).toFixed(0)}%`;
}
