// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import Popover from "@cloudscape-design/components/popover";
import Link from "@cloudscape-design/components/link";
import SpaceBetween from "@cloudscape-design/components/space-between";

// Shared metric/column definitions used by BOTH the Leaderboard and the Race
// detail table, so the info popovers stay in sync across the app.
export const METRIC_DEFS: Record<string, { title: string; body: string }> = {
  state: {
    title: "State",
    body: "Where this model is in the run lifecycle: pending → training → eval_pending → evaluating → done (or failed). Metrics appear once it reaches done.",
  },
  exact: {
    title: "Exact match",
    body: "Fraction of eval rows where the model's RAW output equals the gold answer character-for-character (after trimming). Strictest metric — 0 if the model adds any extra text.\n\nWhen the model wraps its answer in <think>/fences/prose (see Scaffold rate), raw exact understates quality, so the cell also shows the 'extr' (extracted) exact match — the fair number scored after stripping that scaffolding. Both appear together only when they differ.",
  },
  norm: {
    title: "Normalized match",
    body: "Exact match after normalizing both sides (lowercase, strip punctuation/articles/extra whitespace). More forgiving than exact, but still 0 if the answer is wrapped in prose.",
  },
  contains: {
    title: "Contains gold",
    body: "Fraction of rows where the (normalized) gold answer appears somewhere inside the model output. Credits a correct answer even when surrounded by extra words.",
  },
  f1: {
    title: "Token F1",
    body: "Word-overlap F1 between output and gold (harmonic mean of token precision & recall). Partial credit for partly-correct answers. Robust default for short free-text.",
  },
  rouge: {
    title: "ROUGE-L",
    body: "F-measure based on the Longest Common Subsequence of words — rewards in-order overlap with the gold answer. Standard summarization/generation metric.",
  },
  charf1: {
    title: "Char F1",
    body: "Character-level overlap F1 (bag of characters). Forgiving of tokenization/formatting differences; catches near-misses that word-level F1 misses.",
  },
  jsonValid: {
    title: "JSON valid",
    body: "For rows whose gold is JSON: fraction of model outputs that PARSE as valid JSON (after stripping any <think> reasoning block / code fences). The strict format gate — answers this first: does the model emit usable JSON at all?",
  },
  json: {
    title: "JSON structural",
    body: "For JSON-gold rows: fraction where the output is valid JSON with the SAME structure (key set + value types) as the gold. Stricter than 'JSON valid'. Blank when the gold isn't JSON.",
  },
  jsonKeys: {
    title: "JSON key recall",
    body: "For JSON-gold rows whose gold is an object: average fraction of the gold's top-level keys present in the model's JSON. Catches partial objects that parse but miss required fields.",
  },
  labelAcc: {
    title: "Label accuracy",
    body: "For classification rows (short label gold, e.g. 'urgency: high'): fraction where the model's (normalized) answer exactly matches the gold label. The right metric for classification tasks.",
  },
  numeric: {
    title: "Numeric match",
    body: "For rows whose gold is a number: fraction where the model's numeric answer equals the gold (tolerant of $, commas, %). The right metric for math / numeric-extraction tasks.",
  },
  scaffold: {
    title: "Scaffold rate",
    body: "Fraction of outputs where the model wrapped its answer in reasoning/markdown scaffolding (<think>…</think>, ```fences```, or prose). Diagnostic: high here means the raw strict metrics understate quality — trust the 'extracted' metrics instead.",
  },
  lenratio: {
    title: "Length ratio",
    body: "Output length ÷ gold length (in tokens, on the extracted answer). Diagnostic, not a score: ~1.0 means similar length; >>1 means padding/rambling; <<1 means too terse.",
  },
  tps: {
    title: "Tokens / sec",
    body: "MEASURED generation throughput during the eval run (output tokens ÷ generation seconds). Higher = faster inference. Drives the projected serving-cost estimate.",
  },
  latency: {
    title: "Latency p50 / p90 / p99",
    body: "MEASURED per-row generation latency during the eval run, in milliseconds, at three percentiles: p50 (median — the typical request), p90, and p99 (the TAIL — the slowest 10% / 1%). Lower = snappier. Production latency budgets are set on the tail, not the median: a good p50 with a bad p99 means most requests are fast but some are very slow. p99 turns red when it's more than 5× the p50 (a 'tail blowup'). Older eval runs show p50 only. Blank for the API baseline (no self-host timing).",
  },
  instance: {
    title: "Instance",
    body: "The SageMaker GPU instance type this model trained + evaluated on (auto-picked from the model's size). Affects both speed and cost.",
  },
  winrate: {
    title: "Prefers chosen (DPO win-rate)",
    body: "DPO only: the fraction of held-out prompts where the model's answer is closer (token-F1) to the CHOSEN response than to the REJECTED one — i.e. it reproduces the stored preference. This is the preference-native metric (what DPO actually optimizes), unlike gold-overlap which scores against the single chosen answer.\n\nBlank for SFT/KTO/RLVR (no rejected pair). Ties count as not-a-win, so it's a conservative fraction.",
  },
  reward: {
    title: "Judge reward (RLAIF)",
    body: "RLAIF only: the AI judge's reward for this model's responses (0-100%), reference-free — the judge scores the response against the reward PROMPT, with no gold answer to overlap. This is the ONLY ranking signal for an RLAIF run (its held-out set is prompt-only).\n\n'held-out' = reward on the held-out eval prompts; 'training' = final training-step reward (fallback when no held-out reward was logged). Blank for SFT/DPO/KTO/RLVR, which rank on the gold-overlap metrics.",
  },
  judge: {
    title: "LLM judge (1-5)",
    body: "Overall quality score (1-5) from an LLM-as-judge (Sonnet) that sees the original prompt, the gold, and the answer — judges meaning, not wording, and checks the answer is grounded in the prompt (catches hallucination).\n\nHover the score on a done row for the per-dimension breakdown (correctness / faithfulness / format / completeness / conciseness). Winner judged automatically; others on demand.",
  },
  base: {
    title: "Base (untrained)",
    body: "The original (un-fine-tuned) model's score on the same held-out set + ranking metric — the 'before' that makes Fine-tune lift meaningful.\n\nBlank until the base eval finishes (it runs alongside training), or where it doesn't apply (RLAIF).",
  },
  tuned: {
    title: "Fine-tuned (ranking metric)",
    body: "The fine-tuned model's score on the ranking metric — the 'after' to Base's 'before', so Base → Fine-tuned → Lift reads as before → after → gain. (This column follows the 'Rank by' selector; the same score is also in its own per-metric column.)",
  },
  lift: {
    title: "Fine-tune lift",
    body: "Fine-tuned score minus base (untrained) score on the ranking metric, in percentage POINTS (e.g. 0.47 → 0.65 = +18.0 pts — an absolute point gap, not a % change).\n\nGreen = helped · red = hurt (scored worse than the untrained base) · grey = no real change.\n\nBlank until the base eval finishes, or for RLAIF (no gold — use Judge reward).",
  },
  train: {
    title: "Train cost (USD)",
    body: "MEASURED: the training job's billable seconds × the instance's on-demand $/hr. The one-time cost to produce this fine-tuned model. 'n/a' for the API baseline.",
  },
  serve: {
    title: "Cost / 1k rows",
    body: "Fine-tuned models: DERIVED projected self-host cost = (instance $/hr ÷ 3600) ÷ tokens-per-sec × 1000. Sonnet baseline: ACTUAL API price for 1k eval rows. The core cost comparison.",
  },
};

// Render a tooltip body that may contain paragraph breaks. A raw string handed to
// Popover `content` collapses "\n" to a space (HTML whitespace), so author bodies
// with blank-line ("\n\n") separators and we split them into real <p> blocks. This
// works for EVERY tooltip uniformly — single-paragraph bodies render unchanged.
function TooltipBody({ body }: { body: string }) {
  const paras = body.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
  return (
    <SpaceBetween size="xs">
      {paras.map((p, i) => (
        <span key={i}>{p}</span>
      ))}
    </SpaceBetween>
  );
}

function HeaderInfo({ k }: { k: string }) {
  const d = METRIC_DEFS[k];
  if (!d) return null;
  return (
    <Popover
      header={d.title}
      content={<TooltipBody body={d.body} />}
      triggerType="custom"
      dismissButton={false}
      position="top"
    >
      <Link variant="info">info</Link>
    </Popover>
  );
}

// A column header label with an info popover beside it.
export function colHeader(label: string, k: string) {
  return (
    <SpaceBetween direction="horizontal" size="xxs">
      <span>{label}</span>
      <HeaderInfo k={k} />
    </SpaceBetween>
  );
}
