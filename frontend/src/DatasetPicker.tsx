// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Select from "@cloudscape-design/components/select";
import SegmentedControl from "@cloudscape-design/components/segmented-control";
import Tabs from "@cloudscape-design/components/tabs";
import FileUpload from "@cloudscape-design/components/file-upload";
import Input from "@cloudscape-design/components/input";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Checkbox from "@cloudscape-design/components/checkbox";
import Badge from "@cloudscape-design/components/badge";
import Link from "@cloudscape-design/components/link";
import ColumnLayout from "@cloudscape-design/components/column-layout";

import {
  autoSplit,
  getDatasets,
  hfImport,
  hfPreview,
  type CurrentSplit,
  type Dataset,
  type HFPreview,
  type SplitReport,
} from "./api";
import { demoFile, downloadDemo } from "./demoDatasets";
import { useNotify, errText } from "./notifications";
import { SplitResult } from "./SplitResult";
import { InvestigateDataset } from "./InvestigateDataset";
import { fileUploadI18n } from "./i18n";
import { MessagesPreview } from "./hfBits";

// A dataset's shape → the fine-tuning objective it feeds. Shared so the picker,
// the Datasets page, and anywhere else label it identically.
export function shapeLabel(shape?: string): string {
  if (shape === "preference") return "DPO";
  if (shape === "kto") return "KTO";
  if (shape === "rlvr") return "RLVR";
  if (shape === "rlaif") return "RLAIF";
  return "SFT"; // absent/"sft" → messages data
}

// Render the prompt (last user turn) of a canonical row for the previews below.
function promptText(row: Record<string, unknown>): string {
  const msgs = (row?.messages as { role: string; content: string }[]) ?? [];
  const lastUser = [...msgs].reverse().find((m) => m.role === "user");
  return lastUser?.content ?? (msgs[0]?.content ?? "");
}
function turnText(t: unknown): string {
  if (typeof t === "string") return t;
  if (t && typeof t === "object") return String((t as { content?: unknown }).content ?? "");
  return "";
}
const clip = (s: string, n = 240) => (s.length > n ? s.slice(0, n) + "…" : s);

// A test/val split (fractions of the whole) is valid when each is 0–<1 and they
// leave a positive Train remainder. Shared by SplitRatios + the create buttons.
function splitValid(testRatio: string, valRatio: string): boolean {
  const t = Number(testRatio);
  const v = Number(valRatio);
  const frac = (x: number) => Number.isFinite(x) && x >= 0 && x < 1;
  return frac(t) && frac(v) && 1 - t - v > 0;
}

// Preview for a PREFERENCE (DPO) import: prompt + the chosen (preferred) vs
// rejected responses, so the user can sanity-check the conversion before import.
function PreferencePreview({ rows }: { rows: Record<string, unknown>[] }) {
  return (
    <SpaceBetween size="s">
      {rows.slice(0, 2).map((r, i) => (
        <Box key={i} padding="xs">
          <Box variant="awsui-key-label">Prompt</Box>
          <Box variant="p">{clip(promptText(r))}</Box>
          <ColumnLayout columns={2}>
            <div>
              <Badge color="green">chosen</Badge>
              <Box variant="p" fontSize="body-s">{clip(turnText(r.chosen))}</Box>
            </div>
            <div>
              <Badge color="red">rejected</Badge>
              <Box variant="p" fontSize="body-s">{clip(turnText(r.rejected))}</Box>
            </div>
          </ColumnLayout>
        </Box>
      ))}
    </SpaceBetween>
  );
}

// Preview for a KTO import: prompt + the completion + its good/bad label.
function KtoPreview({ rows }: { rows: Record<string, unknown>[] }) {
  return (
    <SpaceBetween size="s">
      {rows.slice(0, 2).map((r, i) => {
        const msgs = (r.messages as { role: string; content: string }[]) ?? [];
        const completion = [...msgs].reverse().find((m) => m.role === "assistant")?.content ?? "";
        return (
          <Box key={i} padding="xs">
            <Box variant="awsui-key-label">Prompt</Box>
            <Box variant="p">{clip(promptText(r))}</Box>
            <SpaceBetween direction="horizontal" size="xs">
              <Badge color={r.kto_tag ? "green" : "red"}>{r.kto_tag ? "good" : "bad"}</Badge>
              <Box variant="p" fontSize="body-s">{clip(completion)}</Box>
            </SpaceBetween>
          </Box>
        );
      })}
    </SpaceBetween>
  );
}

// Intuitive 3-way split control: Test + Validation are entered as percentages;
// Train is auto-computed (100 − test − val) and shown read-only, so the three
// always sum to 100%. Includes the deterministic Seed. Used by the DPO / KTO /
// HF import flows so the split UI is consistent everywhere (replaces the two bare
// ratio boxes). Values are 0–1 fractions in state; shown as % here.
function SplitRatios({
  testRatio,
  valRatio,
  seed,
  onTest,
  onVal,
  onSeed,
}: {
  testRatio: string; // "0.1"
  valRatio: string;
  seed: string;
  onTest: (v: string) => void;
  onVal: (v: string) => void;
  onSeed: (v: string) => void;
}) {
  const t = Number(testRatio);
  const v = Number(valRatio);
  const trainFrac = 1 - t - v;
  const ok = splitValid(testRatio, valRatio);
  const pct = (x: number) => `${Math.round(x * 100)}%`;
  // Show fractions as whole-percent inputs without fighting the user's typing.
  const asPctInput = (frac: string) => {
    const n = Number(frac);
    return Number.isFinite(n) ? String(Math.round(n * 100)) : frac;
  };
  const fromPctInput = (s: string) => {
    const n = Number(s);
    // Round to whole-percent BEFORE storing, so the displayed value (which is
    // also whole-percent via asPctInput) always equals what gets submitted —
    // otherwise typing 12.5 would show 13% but submit 0.125.
    return Number.isFinite(n) ? String(Math.round(n) / 100) : s;
  };
  return (
    <FormField
      label="Split"
      description="One file → three sets. Train (the rest) learns the model; Validation drives the val-loss curve + early stopping; Test is the held-out benchmark scored on the leaderboard."
      errorText={!ok ? "Test + Validation must each be 0–99% and leave room for Train (they can't sum to ≥100%)." : undefined}
    >
      {/* Short numeric fields are width-capped so a 2–3 digit value (10, 42) isn't
          stretched across a full-width input. The three split boxes sit on one row;
          Seed gets the same narrow cap below. */}
      <SpaceBetween size="xs">
        <SpaceBetween direction="horizontal" size="m">
          <FormField label="Train" description="auto = 100 − test − val">
            <div style={{ width: 90 }}>
              <Input value={ok ? pct(trainFrac) : "—"} disabled />
            </div>
          </FormField>
          <FormField label="Validation %">
            <div style={{ width: 90 }}>
              <Input
                type="number"
                value={asPctInput(valRatio)}
                onChange={({ detail }) => onVal(fromPctInput(detail.value))}
              />
            </div>
          </FormField>
          <FormField label="Test %">
            <div style={{ width: 90 }}>
              <Input
                type="number"
                value={asPctInput(testRatio)}
                onChange={({ detail }) => onTest(fromPctInput(detail.value))}
              />
            </div>
          </FormField>
        </SpaceBetween>
        <FormField label="Seed" description="Same seed → same split (reproducible).">
          <div style={{ width: 120 }}>
            <Input type="number" value={seed} onChange={({ detail }) => onSeed(detail.value)} />
          </div>
        </FormField>
      </SpaceBetween>
    </FormField>
  );
}

// A self-contained dataset chooser: pick an existing dataset, or create a new
// one (upload + disjoint split). Emits the chosen dataset as a CurrentSplit.
export function DatasetPicker({
  selected,
  onSelect,
}: {
  selected: CurrentSplit | null;
  onSelect: (split: CurrentSplit | null) => void;
}) {
  const { notify } = useNotify();
  const [mode, setMode] = useState<"existing" | "new" | "hf" | "pref" | "kto" | "rlvr" | "rlaif">("existing");
  // Sensitive-data advisory: dismissed per session (re-shows next visit). Reads
  // sessionStorage lazily so a refresh within the session keeps it dismissed.
  const [dataWarningDismissed, setDataWarningDismissed] = useState<boolean>(() => {
    try { return sessionStorage.getItem("slm-data-warning-dismissed") === "1"; } catch { return false; }
  });
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  // Starts true so the picker shows "Loading…" on first paint instead of the
  // misleading "No datasets yet" before the fetch returns. Set false (and
  // loadError set) in loadDatasets so a slow/failed load is distinguishable
  // from a genuinely empty library.
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  // Whether an HF token is set — gates the prompt shown in the HF import tab
  // (public datasets work without one, but gated/rate-limited fetches need it).
  const [hfTokenSet, setHfTokenSet] = useState<boolean | null>(null);
  useEffect(() => {
    import("./api").then(({ getHfTokenStatus }) =>
      getHfTokenStatus().then((s) => setHfTokenSet(s.isSet || !!s.usingSharedFallback)).catch(() => setHfTokenSet(null))
    );
  }, []);

  // Hugging Face import state. Flow: enter id → Preview (inspect columns +
  // suggested mapping) → adjust mapping/sample size → Import (samples, converts
  // to messages, runs through the normal auto-split path).
  const [hfDataset, setHfDataset] = useState("");
  const [hfPrev, setHfPrev] = useState<HFPreview | null>(null);
  const [hfPreviewing, setHfPreviewing] = useState(false);
  const [hfUserField, setHfUserField] = useState("");
  const [hfTargetField, setHfTargetField] = useState("");
  const [hfSystemField, setHfSystemField] = useState<string>(""); // optional per-row system column
  const [hfContextField, setHfContextField] = useState<string>(""); // optional appended-to-user column
  const [hfInstruction, setHfInstruction] = useState("");
  const [hfMaxRows, setHfMaxRows] = useState("500");
  const hfMaxRowsNum = Number(hfMaxRows);
  const hfMaxRowsValid = Number.isInteger(hfMaxRowsNum) && hfMaxRowsNum > 0 && hfMaxRowsNum <= 2000;
  // Which objective to import the HF dataset as. Defaults to the detected shape
  // after Preview; the user can override (e.g. force SFT, or fix the columns).
  const [hfImportAs, setHfImportAs] = useState<"sft" | "preference" | "kto" | "rlvr" | "rlaif">("sft");
  // Manual column overrides for the DPO + KTO + RLVR import paths — seeded from the
  // backend's autodetected mapping on Preview, then editable so the user can
  // correct a wrong guess (the gap that made HF DPO autodetect-only before).
  const [hfChosenField, setHfChosenField] = useState("");
  const [hfRejectedField, setHfRejectedField] = useState("");
  const [hfPrefPromptField, setHfPrefPromptField] = useState("");
  const [hfCompletionField, setHfCompletionField] = useState("");
  const [hfLabelField, setHfLabelField] = useState("");
  const [hfKtoPromptField, setHfKtoPromptField] = useState("");
  // RLVR import: a prompt column + a verifiable ground-truth column (e.g. gsm8k).
  const [hfRlvrPromptField, setHfRlvrPromptField] = useState("");
  const [hfRlvrGroundTruthField, setHfRlvrGroundTruthField] = useState("");
  // RLAIF HF import: prompt-only (the AI judge scores the response — no answer col).
  const [hfRlaifPromptField, setHfRlaifPromptField] = useState("");
  // 3-way split for HF DPO/KTO imports (the SFT path uses its own test/val controls).
  const [hfTestRatio, setHfTestRatio] = useState("0.1");
  const [hfValRatio, setHfValRatio] = useState("0.1");

  // Client-side mirror of the backend converter, so the preview updates live as
  // the user changes the column mapping / system prompt (no extra round-trip).
  const hfLivePreview = (hfPrev?.sampleRows ?? []).slice(0, 2).map((raw) => {
    const cell = (f: string) => {
      const v = (raw as Record<string, unknown>)[f];
      if (v == null) return "";
      // Map an int that indexes a ClassLabel column to its class name, matching
      // the backend converter (so the preview equals what gets imported).
      const names = hfPrev?.classLabels?.[f];
      if (names && typeof v === "number" && v >= 0 && v < names.length) return names[v];
      return typeof v === "object" ? JSON.stringify(v) : String(v);
    };
    // system turn = fixed instruction + optional per-row system column (matches backend)
    const sysParts = [hfInstruction.trim(), hfSystemField ? cell(hfSystemField) : ""].filter(Boolean);
    // user turn = user column + optional context column appended (matches backend)
    let userVal = cell(hfUserField);
    if (hfContextField) {
      const ctx = cell(hfContextField);
      if (ctx) userVal = userVal ? `${userVal}\n\n${ctx}` : ctx;
    }
    const messages: { role: string; content: string }[] = [];
    if (sysParts.length) messages.push({ role: "system", content: sysParts.join("\n\n") });
    messages.push({ role: "user", content: userVal });
    messages.push({ role: "assistant", content: cell(hfTargetField) });
    return { messages };
  });

  // Set true when the user clicks a "Create …" button while a REQUIRED field
  // (name / file) is empty. Rather than silently disabling the button — which left
  // users guessing why nothing happened — the button stays clickable and this flag
  // turns on inline errorText that highlights the missing field(s). Cleared when
  // the mode changes so switching tabs starts clean.
  const [triedSubmit, setTriedSubmit] = useState(false);
  useEffect(() => setTriedSubmit(false), [mode]);

  // create-new state. SFT is single-file (auto-split), consistent with the DPO/
  // KTO/RLVR/RLAIF tabs — the backend carves train/val/test from the one upload.
  const [name, setName] = useState("");
  // True while `name` is the auto-suggested HF name (not hand-edited), so it stays
  // in sync when the user re-previews a DIFFERENT dataset or changes sample
  // size/seed. Cleared the moment the user edits the name field.
  const [autoName, setAutoName] = useState(true);
  const [autoFiles, setAutoFiles] = useState<File[]>([]);
  const [ratio, setRatio] = useState("0.2");
  const [seed, setSeed] = useState("42");
  // Stratified sampling: keep each class proportionally represented across
  // train/val/test (label tasks only; falls back to random otherwise). Shared
  // by the one-file auto-split + HF import flows.
  const [stratify, setStratify] = useState(false);
  const [report, setReport] = useState<SplitReport | null>(null);
  const [creating, setCreating] = useState(false);
  // "Investigate" wizard for the currently-selected dataset (advisory profiler).
  const [investigating, setInvestigating] = useState(false);

  // The auto-suggested HF dataset name, from the LIVE id / sample size / seed so
  // it always matches what actually gets imported. (Was captured at preview time
  // and only set when the name was blank → re-previewing a different dataset, or
  // changing sample size/seed after preview, left a stale/mismatched name.)
  // Fold the resolved config/split into the suffix so different subsets of the
  // SAME dataset (e.g. gsm8k main/train vs socratic/train) get DISTINCT names.
  const suggestedHfName = () => {
    const cs = hfPrev ? `-${hfPrev.config}-${hfPrev.split}` : "";
    return `hf:${hfDataset.trim()}${cs}-${hfMaxRows}r-s${seed}`;
  };
  // Keep the suggested name in sync while the user hasn't hand-edited it — but
  // only once a preview has run (hfPrev set), so the name reflects a real dataset.
  useEffect(() => {
    if (mode === "hf" && autoName && hfPrev) setName(suggestedHfName());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, autoName, hfPrev, hfDataset, hfMaxRows, seed]);

  // Optional validation set. Enables early stopping. The
  // held-out test set is never used for in-training stopping — kept fair.
  // Default ON: carve a validation set so every fine-tuning job gets a
  // validation-loss curve + early-stopping signal (consistent with DPO/KTO).
  const [addVal, setAddVal] = useState(true);
  const [valRatio, setValRatio] = useState("0.1");
  const valRatioNum = Number(valRatio);
  const valRatioValid = Number.isFinite(valRatioNum) && valRatioNum > 0 && valRatioNum < 1;

  // Preference (DPO) upload state. A JSONL of {messages|prompt, chosen, rejected}.
  // One file → 3-way split (test/val/train); val is always carved so every job
  // gets a validation-loss curve + early-stopping signal.
  const [prefFiles, setPrefFiles] = useState<File[]>([]);
  const [prefName, setPrefName] = useState("");
  const [prefTestRatio, setPrefTestRatio] = useState("0.1");
  const [prefValRatio, setPrefValRatio] = useState("0.1");
  const [prefBusy, setPrefBusy] = useState(false);

  async function createPreference() {
    if (!prefFiles[0] || !prefName.trim()) { setTriedSubmit(true); return; }
    setPrefBusy(true);
    try {
      const { createPreferenceDataset } = await import("./api");
      const res = await createPreferenceDataset(
        prefFiles[0], prefName.trim(), Number(prefTestRatio), Number(prefValRatio), seedNum);
      // Select it immediately (it's preference-shaped → the run uses DPO).
      onSelect({
        splitId: res.splitId,
        name: res.name,
        trainRows: res.trainRows,
        evalRows: res.testRows,
        hasVal: res.valRows > 0,
        valRows: res.valRows,
        shape: "preference",
        origin: `preference upload (${res.totalPairs} pairs)`,
      });
      setPrefFiles([]);
      setPrefName("");
      loadDatasets();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setPrefBusy(false);
    }
  }

  // KTO upload state. A JSONL of {messages|prompt+completion, kto_tag|label}.
  const [ktoFiles, setKtoFiles] = useState<File[]>([]);
  const [ktoName, setKtoName] = useState("");
  const [ktoTestRatio, setKtoTestRatio] = useState("0.1");
  const [ktoValRatio, setKtoValRatio] = useState("0.1");
  const [ktoBusy, setKtoBusy] = useState(false);

  async function createKto() {
    if (!ktoFiles[0] || !ktoName.trim()) { setTriedSubmit(true); return; }
    setKtoBusy(true);
    try {
      const { createKtoDataset } = await import("./api");
      const res = await createKtoDataset(
        ktoFiles[0], ktoName.trim(), Number(ktoTestRatio), Number(ktoValRatio), seedNum);
      onSelect({
        splitId: res.splitId,
        name: res.name,
        trainRows: res.trainRows,
        evalRows: res.testRows,
        hasVal: res.valRows > 0,
        valRows: res.valRows,
        shape: "kto",
        origin: `KTO upload (${res.totalRows} rows, ${res.desirable} good)`,
      });
      setKtoFiles([]);
      setKtoName("");
      loadDatasets();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setKtoBusy(false);
    }
  }

  // RLVR upload state. A JSONL of {messages|prompt, ground_truth}. The verifiable
  // answer goes in its own ground_truth field (NOT a worked solution to imitate);
  // the preset reward function is chosen at launch time.
  const [rlvrFiles, setRlvrFiles] = useState<File[]>([]);
  const [rlvrName, setRlvrName] = useState("");
  const [rlvrTestRatio, setRlvrTestRatio] = useState("0.1");
  const [rlvrValRatio, setRlvrValRatio] = useState("0.1");
  const [rlvrBusy, setRlvrBusy] = useState(false);

  async function createRlvr() {
    if (!rlvrFiles[0] || !rlvrName.trim()) { setTriedSubmit(true); return; }
    setRlvrBusy(true);
    try {
      const { createRlvrDataset } = await import("./api");
      const res = await createRlvrDataset(
        rlvrFiles[0], rlvrName.trim(), Number(rlvrTestRatio), Number(rlvrValRatio), seedNum);
      onSelect({
        splitId: res.splitId,
        name: res.name,
        trainRows: res.trainRows,
        evalRows: res.testRows,
        hasVal: res.valRows > 0,
        valRows: res.valRows,
        shape: "rlvr",
        origin: `RLVR upload (${res.totalRows} rows)`,
      });
      setRlvrFiles([]);
      setRlvrName("");
      loadDatasets();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setRlvrBusy(false);
    }
  }

  // RLAIF upload (prompt-only JSONL — no ground_truth; an AI judge scores at launch).
  const [rlaifFiles, setRlaifFiles] = useState<File[]>([]);
  const [rlaifName, setRlaifName] = useState("");
  const [rlaifTestRatio, setRlaifTestRatio] = useState("0.1");
  const [rlaifValRatio, setRlaifValRatio] = useState("0.1");
  const [rlaifBusy, setRlaifBusy] = useState(false);

  async function createRlaif() {
    if (!rlaifFiles[0] || !rlaifName.trim()) { setTriedSubmit(true); return; }
    setRlaifBusy(true);
    try {
      const { createRlaifDataset } = await import("./api");
      const res = await createRlaifDataset(
        rlaifFiles[0], rlaifName.trim(), Number(rlaifTestRatio), Number(rlaifValRatio), seedNum);
      onSelect({
        splitId: res.splitId,
        name: res.name,
        trainRows: res.trainRows,
        evalRows: res.testRows,
        hasVal: res.valRows > 0,
        valRows: res.valRows,
        shape: "rlaif",
        origin: `RLAIF upload (${res.totalRows} prompts)`,
      });
      setRlaifFiles([]);
      setRlaifName("");
      loadDatasets();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setRlaifBusy(false);
    }
  }

  function loadDatasets() {
    setDatasetsLoading(true);
    setLoadError(false);
    getDatasets()
      // Fine-tuning needs a TRAINING set, so eval-only datasets (no train rows)
      // are not selectable here — they're for the standalone Evaluate page.
      .then((ds) => setDatasets(ds.filter((d) => !d.evalOnly)))
      .catch((e) => {
        setLoadError(true);
        notify({ type: "error", content: errText(e) });
      })
      .finally(() => setDatasetsLoading(false));
  }
  useEffect(loadDatasets, []);

  function toCurrentSplit(r: SplitReport): CurrentSplit | null {
    if (!r.ok || !r.splitId) return null;
    return {
      splitId: r.splitId,
      name: r.name || undefined,
      trainRows: r.trainRows,
      evalRows: r.evalRows,
      hasVal: r.hasVal,
      valRows: r.valRows,
      origin: r.mode === "auto" ? `auto-split (ratio ${r.evalRatio})` : "two-file assert",
    };
  }

  function datasetToCurrent(d: Dataset): CurrentSplit {
    return {
      splitId: d.splitId,
      name: d.name || undefined,
      trainRows: d.trainRows ?? 0,
      evalRows: d.evalRows ?? 0,
      hasVal: d.hasVal ?? false,
      valRows: d.valRows ?? 0,
      shape: d.shape ?? "sft",
      recommendedRewardMetric: d.recommendedRewardMetric,
      origin: "existing dataset",
    };
  }

  async function create(fn: () => Promise<SplitReport>) {
    setCreating(true);
    setReport(null);
    try {
      const r = await fn();
      setReport(r);
      const cs = toCurrentSplit(r);
      if (cs) {
        onSelect(cs);
        loadDatasets(); // new dataset now appears in the existing list
      }
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setCreating(false);
    }
  }

  const ratioNum = Number(ratio);
  const seedNum = Number(seed);
  const ratioValid = Number.isFinite(ratioNum) && ratioNum > 0 && ratioNum < 1;
  const seedValid = Number.isInteger(seedNum);

  async function doHfPreview(cfg?: string, sp?: string) {
    setHfPreviewing(true);
    setHfPrev(null);
    try {
      const p = await hfPreview(hfDataset.trim(), cfg, sp);
      setHfPrev(p);
      // Seed the mapping fields from the auto-detected suggestion.
      setHfUserField(p.suggestedMapping?.userField ?? "");
      setHfTargetField(p.suggestedMapping?.targetField ?? "");
      setHfContextField(p.suggestedMapping?.contextField ?? "");
      setHfSystemField(p.suggestedMapping?.systemField ?? "");
      // Default the import objective to the detected shape; seed the per-objective
      // column overrides from the autodetected mappings (editable below).
      setHfImportAs(p.detectedShape);
      setHfChosenField(p.suggestedPreference?.chosenField ?? "");
      setHfRejectedField(p.suggestedPreference?.rejectedField ?? "");
      setHfPrefPromptField(p.suggestedPreference?.promptField ?? "");
      setHfCompletionField(p.suggestedKto?.completionField ?? "");
      setHfLabelField(p.suggestedKto?.labelField ?? "");
      setHfKtoPromptField(p.suggestedKto?.promptField ?? "");
      setHfRlvrPromptField(p.suggestedRlvr?.promptField ?? "");
      setHfRlvrGroundTruthField(p.suggestedRlvr?.groundTruthField ?? "");
      setHfRlaifPromptField(p.suggestedRlaif?.promptField ?? p.suggestedRlvr?.promptField ?? "");
      // Auto-name includes the sample size + seed so repeated imports of the
      // SAME HF dataset get DISTINCT names (avoids a pile of identical
      // "hf:fancyzhx/ag_news" entries you can't tell apart). Regenerated on every
      // preview while the user hasn't hand-edited it (autoName), so previewing a
      // DIFFERENT dataset doesn't leave a stale name; the sync effect above keeps
      // it fresh if sample size/seed then change. Editing the field clears autoName.
      if (autoName) setName(suggestedHfName());
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setHfPreviewing(false);
    }
  }

  async function doHfImport() {
    if (!hfPrev) return;
    const vr = addVal && valRatioValid ? valRatioNum : null;

    // Preference (DPO) + KTO take their own backend paths (ranking / labelled rows
    // → 3-way split); they don't return a SplitReport, so handled outside create().
    if (hfImportAs === "preference") {
      setCreating(true);
      try {
        const { hfImportPreference } = await import("./api");
        const res = await hfImportPreference({
          dataset: hfPrev.dataset, config: hfPrev.config, split: hfPrev.split,
          name: name.trim(),
          chosenField: hfChosenField,
          rejectedField: hfRejectedField,
          promptField: hfPrefPromptField || null,
          systemField: hfSystemField || null,
          instruction: hfInstruction.trim(),
          maxRows: hfMaxRowsNum, seed: seedNum,
          testRatio: Number(hfTestRatio), valRatio: Number(hfValRatio),
        });
        onSelect({
          splitId: res.splitId, name: res.name, trainRows: res.trainRows,
          evalRows: res.testRows, hasVal: res.valRows > 0, valRows: res.valRows,
          shape: "preference",
          origin: `HF preference: ${hfPrev.dataset} (${res.totalPairs} pairs)`,
        });
        loadDatasets();
      } catch (e) {
        notify({ type: "error", content: errText(e) });
      } finally {
        setCreating(false);
      }
      return;
    }
    if (hfImportAs === "kto") {
      setCreating(true);
      try {
        const { hfImportKto } = await import("./api");
        const res = await hfImportKto({
          dataset: hfPrev.dataset, config: hfPrev.config, split: hfPrev.split,
          name: name.trim(),
          completionField: hfCompletionField,
          labelField: hfLabelField,
          promptField: hfKtoPromptField || null,
          systemField: hfSystemField || null,
          instruction: hfInstruction.trim(),
          maxRows: hfMaxRowsNum, seed: seedNum,
          testRatio: Number(hfTestRatio), valRatio: Number(hfValRatio),
        });
        onSelect({
          splitId: res.splitId, name: res.name, trainRows: res.trainRows,
          evalRows: res.testRows, hasVal: res.valRows > 0, valRows: res.valRows,
          shape: "kto",
          origin: `HF KTO: ${hfPrev.dataset} (${res.totalRows} rows, ${res.desirable} good)`,
        });
        loadDatasets();
      } catch (e) {
        notify({ type: "error", content: errText(e) });
      } finally {
        setCreating(false);
      }
      return;
    }
    if (hfImportAs === "rlvr") {
      setCreating(true);
      try {
        const { hfImportRlvr } = await import("./api");
        const res = await hfImportRlvr({
          dataset: hfPrev.dataset, config: hfPrev.config, split: hfPrev.split,
          name: name.trim(),
          promptField: hfRlvrPromptField,
          groundTruthField: hfRlvrGroundTruthField,
          systemField: hfSystemField || null,
          instruction: hfInstruction.trim(),
          maxRows: hfMaxRowsNum, seed: seedNum,
          testRatio: Number(hfTestRatio), valRatio: Number(hfValRatio),
        });
        onSelect({
          splitId: res.splitId, name: res.name, trainRows: res.trainRows,
          evalRows: res.testRows, hasVal: res.valRows > 0, valRows: res.valRows,
          shape: "rlvr",
          origin: `HF RLVR: ${hfPrev.dataset} (${res.totalRows} rows)`,
        });
        loadDatasets();
      } catch (e) {
        notify({ type: "error", content: errText(e) });
      } finally {
        setCreating(false);
      }
      return;
    }
    if (hfImportAs === "rlaif") {
      setCreating(true);
      try {
        const { hfImportRlaif } = await import("./api");
        const res = await hfImportRlaif({
          dataset: hfPrev.dataset, config: hfPrev.config, split: hfPrev.split,
          name: name.trim(),
          promptField: hfRlaifPromptField,
          systemField: hfSystemField || null,
          instruction: hfInstruction.trim(),
          maxRows: hfMaxRowsNum, seed: seedNum,
          testRatio: Number(hfTestRatio), valRatio: Number(hfValRatio),
        });
        onSelect({
          splitId: res.splitId, name: res.name, trainRows: res.trainRows,
          evalRows: res.testRows, hasVal: res.valRows > 0, valRows: res.valRows,
          shape: "rlaif",
          origin: `HF RLAIF: ${hfPrev.dataset} (${res.totalRows} prompts)`,
        });
        loadDatasets();
      } catch (e) {
        notify({ type: "error", content: errText(e) });
      } finally {
        setCreating(false);
      }
      return;
    }
    // SFT path (auto-split → messages dataset).
    await create(() =>
      hfImport({
        dataset: hfPrev.dataset,
        config: hfPrev.config,
        split: hfPrev.split,
        name: name.trim(),
        userField: hfUserField,
        targetField: hfTargetField,
        systemField: hfSystemField || null,
        contextField: hfContextField || null,
        instruction: hfInstruction.trim(),
        maxRows: hfMaxRowsNum,
        seed: seedNum,
        evalRatio: ratioNum,
        valRatio: vr,
        stratify,
      })
    );
  }

  return (
    <Container
      header={
        <Header
          variant="h2"
          description="Choose an existing dataset or create a new one (upload + held-out split)."
        >
          Step 1 — Dataset
        </Header>
      }
    >
      <SpaceBetween size="m">
        {/* Sensitive-data advisory. Uploads/imports write to the PLATFORM's S3
            bucket (the deployment owner's AWS account) — which is NOT your account
            unless YOU cdk-deployed this. Warn before any data leaves the user's
            hands. Dismissible (sessionStorage) so it doesn't nag every visit, and
            hidden on "Use existing" (no upload happens there). */}
        {mode !== "existing" && !dataWarningDismissed && (
          <Alert
            type="warning"
            header="Heads up: don't upload sensitive data to a shared deployment"
            dismissible
            onDismiss={() => {
              setDataWarningDismissed(true);
              try { sessionStorage.setItem("slm-data-warning-dismissed", "1"); } catch { /* ignore */ }
            }}
          >
            Datasets you upload or import are stored in the <b>platform owner's AWS
            account</b> (the S3 bucket this app was deployed into) — that is{" "}
            <b>not your account</b> unless you deployed this platform yourself via CDK.
            For confidential or customer data, run your own deployment (one{" "}
            <code>cdk deploy</code> into your account) and upload there. For trying the
            platform, prefer the public Hugging Face / demo datasets.
          </Alert>
        )}

        {/* Mode selector. Tabs (not SegmentedControl) so the 7 options never wrap
            to a second line — Tabs handles overflow with horizontal scrolling. Used
            purely as a selector (no per-tab content); the shared body below the bar
            self-selects by `mode` (= activeTabId). disableContentPaddings keeps Tabs
            from drawing an empty panel under the header. Short labels keep the row
            tidy; full method names live in each tab's body copy. */}
        <Tabs
          activeTabId={mode}
          disableContentPaddings
          onChange={({ detail }) =>
            setMode(detail.activeTabId as "existing" | "new" | "hf" | "pref" | "kto" | "rlvr" | "rlaif")
          }
          tabs={[
            { id: "existing", label: "Use existing" },
            { id: "new", label: "SFT" },
            { id: "pref", label: "DPO / ORPO / SimPO" },
            { id: "kto", label: "KTO" },
            { id: "rlvr", label: "RLVR" },
            { id: "rlaif", label: "RLAIF" },
            { id: "hf", label: "Hugging Face" },
          ]}
        />

        {mode === "hf" ? (
          <SpaceBetween size="m">
            <Box variant="small" color="text-status-inactive">
              Load a <b>sample</b> of a public Hugging Face dataset as real task data —
              useful for testing the pipeline without a customer dataset. The columns are
              converted to the platform's chat format, then split like any other dataset.
            </Box>
            {hfTokenSet === false && (
              <Alert type="info" header="Set a Hugging Face token for gated or rate-limited datasets">
                Public datasets usually import without a token, but gated datasets (and
                heavy/rate-limited fetches) need one. Add a token under{" "}
                <Link href="#settings">Settings</Link>: create it at{" "}
                <Link external href="https://huggingface.co/settings/tokens">
                  huggingface.co/settings/tokens
                </Link>{" "}
                ("New token" → type <b>Read</b> → copy the <code>hf_…</code> value).
              </Alert>
            )}
            <FormField
              label="Hugging Face dataset id"
              description={
                <SpaceBetween size="xxs">
                  <Box variant="small">
                    Any public dataset id. The objective is auto-detected from the columns
                    (you can override it below). Examples:
                  </Box>
                  <Box variant="small">
                    <b>SFT</b> — <code>fancyzhx/ag_news</code> (classification),{" "}
                    <code>knkarthick/samsum</code> (summarization)
                  </Box>
                  <Box variant="small">
                    <b>DPO</b> — <code>Anthropic/hh-rlhf</code>,{" "}
                    <code>trl-lib/ultrafeedback_binarized</code>
                  </Box>
                  <Box variant="small">
                    <b>KTO</b> — <code>trl-lib/kto-mix-14k</code>
                  </Box>
                  <Box variant="small">
                    <b>RLVR</b> — <code>openai/gsm8k</code> (math; map <code>question</code> →
                    prompt, <code>answer</code> → ground_truth),{" "}
                    <code>openai/openai_humaneval</code> (code). Pick <b>RLVR</b> under "Import as".
                  </Box>
                </SpaceBetween>
              }
            >
              <SpaceBetween direction="horizontal" size="xs">
                <Input
                  value={hfDataset}
                  placeholder="org/dataset"
                  onChange={({ detail }) => setHfDataset(detail.value)}
                />
                <Button
                  loading={hfPreviewing}
                  disabled={hfDataset.trim() === ""}
                  onClick={() => doHfPreview()}
                >
                  Preview
                </Button>
              </SpaceBetween>
            </FormField>

            {hfPrev && (
              <SpaceBetween size="m">
                <Alert type="info">
                  <b>{hfPrev.dataset}</b> · config <b>{hfPrev.config}</b> / split{" "}
                  <b>{hfPrev.split}</b>
                  {hfPrev.numRowsTotal != null ? ` · ${hfPrev.numRowsTotal.toLocaleString()} rows` : ""}
                  {" · columns: "}
                  {hfPrev.columnNames.join(", ")}
                </Alert>

                {/* License advisory (compliance aid). Permissive → a quiet "ok,
                    attribute" line; restrictive / unknown / gated → an amber
                    "verify the terms yourself" warning. ADVISORY ONLY — it never
                    blocks the import (HF license tags are self-reported + often
                    "unknown", so an allow/deny would over-claim). */}
                {(() => {
                  const li = hfPrev.licenseInfo;
                  const bucket = li?.bucket ?? "unknown";
                  const lic = li?.license;
                  const gated = li?.gated;
                  const licLabel = lic ? lic : "not declared";
                  if (bucket === "permissive" && !gated) {
                    return (
                      <Alert type="success" header={`License: ${licLabel}`}>
                        This dataset declares a permissive license. You're still
                        responsible for honoring its attribution terms if you
                        redistribute anything derived from it.{" "}
                        <Link external href={`https://huggingface.co/datasets/${hfPrev.dataset}`}>
                          View the dataset card
                        </Link>
                        .
                      </Alert>
                    );
                  }
                  return (
                    <Alert
                      type="warning"
                      header={
                        gated
                          ? `Gated dataset · license: ${licLabel}`
                          : bucket === "restrictive"
                          ? `Review the license: ${licLabel}`
                          : `License not confirmed: ${licLabel}`
                      }
                    >
                      {gated ? (
                        <>This dataset is <b>gated</b> on Hugging Face — access requires
                        accepting its terms. </>
                      ) : bucket === "restrictive" ? (
                        <>This dataset's license (<b>{licLabel}</b>) may restrict commercial
                        use or redistribution (e.g. non-commercial or copyleft). </>
                      ) : (
                        <>Hugging Face doesn't report a clear license for this dataset
                        (<b>{licLabel}</b>). </>
                      )}
                      Confirm you're permitted to use this data for training — and to
                      redistribute anything derived from it — <b>before</b> importing.
                      Importing is allowed; this is an advisory check, not approval.{" "}
                      <Link external href={`https://huggingface.co/datasets/${hfPrev.dataset}`}>
                        Check the dataset card &amp; license
                      </Link>
                      .
                    </Alert>
                  );
                })()}

                {/* Config/split selector. Many HF datasets expose several configs
                    (e.g. gsm8k main/socratic) and splits (train/test); preview/import
                    default to the first. (config, split) pairs are COUPLED, so the
                    split list is filtered to the chosen config, and changing config
                    re-picks the first valid split under it. Each change re-previews
                    (columns differ per config, so the mapping must re-autodetect). */}
                {hfPrev.splits.length > 1 && (() => {
                  const configs = [...new Set(hfPrev.splits.map((s) => s.config))];
                  const splitsForConfig = hfPrev.splits
                    .filter((s) => s.config === hfPrev.config)
                    .map((s) => s.split);
                  return (
                    <SpaceBetween direction="horizontal" size="xs">
                      {configs.length > 1 && (
                        <FormField label="Config" description="Dataset subset/configuration.">
                          <Select
                            disabled={hfPreviewing}
                            selectedOption={{ value: hfPrev.config, label: hfPrev.config }}
                            onChange={({ detail }) => {
                              const nextConfig = detail.selectedOption.value!;
                              const firstSplit =
                                hfPrev.splits.find((s) => s.config === nextConfig)?.split;
                              doHfPreview(nextConfig, firstSplit);
                            }}
                            options={configs.map((c) => ({ value: c, label: c }))}
                          />
                        </FormField>
                      )}
                      <FormField label="Split" description="Which split to sample rows from.">
                        <Select
                          disabled={hfPreviewing}
                          selectedOption={{ value: hfPrev.split, label: hfPrev.split }}
                          onChange={({ detail }) =>
                            doHfPreview(hfPrev.config, detail.selectedOption.value!)
                          }
                          options={splitsForConfig.map((s) => ({ value: s, label: s }))}
                        />
                      </FormField>
                    </SpaceBetween>
                  );
                })()}

                {/* Import objective: defaults to the detected shape; the user can
                    override (force SFT, or fix the columns). DPO/KTO show manual
                    column pickers so an autodetect miss can be corrected. */}
                <FormField
                  label="Import as"
                  description={
                    hfPrev.detectedShape === "preference"
                      ? "Detected chosen/rejected columns → DPO. Override to SFT if you'd rather train on the chosen answers only."
                      : hfPrev.detectedShape === "kto"
                        ? "Detected a completion + good/bad label column → KTO. Override to SFT to ignore the labels."
                        : hfPrev.suggestedRlvr
                          ? "Looks like a question/answer set — defaulting to SFT, but it can also train RLVR (verifiable reward): pick RLVR to use the answer as the gold target instead of imitating it."
                          : "No preference/KTO columns detected — importing as SFT (messages)."
                  }
                >
                  <SegmentedControl
                    selectedId={hfImportAs}
                    onChange={({ detail }) => setHfImportAs(detail.selectedId as "sft" | "preference" | "kto" | "rlvr" | "rlaif")}
                    options={[
                      { id: "sft", text: "SFT" },
                      { id: "preference", text: "Preference (DPO / ORPO / SimPO)" },
                      { id: "kto", text: "KTO (Good/Bad)" },
                      { id: "rlvr", text: "RLVR (verifiable)" },
                      { id: "rlaif", text: "RLAIF (AI judge)" },
                    ]}
                  />
                </FormField>

                {/* DPO column mapping (overridable). */}
                {hfImportAs === "preference" && (
                  <SpaceBetween direction="horizontal" size="xs">
                    <FormField label="Chosen column" description="The preferred response.">
                      <Select
                        selectedOption={hfChosenField ? { value: hfChosenField, label: hfChosenField } : null}
                        onChange={({ detail }) => setHfChosenField(detail.selectedOption.value!)}
                        options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                        placeholder="Choose column"
                      />
                    </FormField>
                    <FormField label="Rejected column" description="The dispreferred response.">
                      <Select
                        selectedOption={hfRejectedField ? { value: hfRejectedField, label: hfRejectedField } : null}
                        onChange={({ detail }) => setHfRejectedField(detail.selectedOption.value!)}
                        options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                        placeholder="Choose column"
                      />
                    </FormField>
                    <FormField label="Prompt column (optional)" description="Blank = inferred from the shared prefix.">
                      <Select
                        selectedOption={hfPrefPromptField ? { value: hfPrefPromptField, label: hfPrefPromptField } : { value: "", label: "(infer)" }}
                        onChange={({ detail }) => setHfPrefPromptField(detail.selectedOption.value || "")}
                        options={[{ value: "", label: "(infer)" }, ...hfPrev.columnNames.map((c) => ({ value: c, label: c }))]}
                      />
                    </FormField>
                  </SpaceBetween>
                )}

                {/* KTO column mapping (overridable). */}
                {hfImportAs === "kto" && (
                  <SpaceBetween direction="horizontal" size="xs">
                    <FormField label="Completion column" description="The response being judged.">
                      <Select
                        selectedOption={hfCompletionField ? { value: hfCompletionField, label: hfCompletionField } : null}
                        onChange={({ detail }) => setHfCompletionField(detail.selectedOption.value!)}
                        options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                        placeholder="Choose column"
                      />
                    </FormField>
                    <FormField label="Label column" description="Good/bad signal (bool, 0/1, -1/1, or good/bad).">
                      <Select
                        selectedOption={hfLabelField ? { value: hfLabelField, label: hfLabelField } : null}
                        onChange={({ detail }) => setHfLabelField(detail.selectedOption.value!)}
                        options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                        placeholder="Choose column"
                      />
                    </FormField>
                    <FormField label="Prompt column (optional)">
                      <Select
                        selectedOption={hfKtoPromptField ? { value: hfKtoPromptField, label: hfKtoPromptField } : { value: "", label: "(none)" }}
                        onChange={({ detail }) => setHfKtoPromptField(detail.selectedOption.value || "")}
                        options={[{ value: "", label: "(none)" }, ...hfPrev.columnNames.map((c) => ({ value: c, label: c }))]}
                      />
                    </FormField>
                  </SpaceBetween>
                )}

                {/* RLVR column mapping (overridable). A prompt + a verifiable answer
                    — gsm8k's question + answer is the canonical case. */}
                {hfImportAs === "rlvr" && (
                  <SpaceBetween direction="horizontal" size="xs">
                    <FormField label="Prompt column" description="The question/task the model answers.">
                      <Select
                        selectedOption={hfRlvrPromptField ? { value: hfRlvrPromptField, label: hfRlvrPromptField } : null}
                        onChange={({ detail }) => setHfRlvrPromptField(detail.selectedOption.value!)}
                        options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                        placeholder="Choose column"
                      />
                    </FormField>
                    <FormField label="Ground-truth column" description="The verifiable answer the reward scorer checks (e.g. gsm8k's answer).">
                      <Select
                        selectedOption={hfRlvrGroundTruthField ? { value: hfRlvrGroundTruthField, label: hfRlvrGroundTruthField } : null}
                        onChange={({ detail }) => setHfRlvrGroundTruthField(detail.selectedOption.value!)}
                        options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                        placeholder="Choose column"
                      />
                    </FormField>
                  </SpaceBetween>
                )}

                {/* RLAIF column mapping (overridable). PROMPT ONLY — no answer column;
                    the AI judge (a reward prompt picked at launch) scores the response. */}
                {hfImportAs === "rlaif" && (
                  <SpaceBetween direction="horizontal" size="xs">
                    <FormField label="Prompt column" description="The task the model responds to. No answer is imported — the AI judge scores a fresh response at training time.">
                      <Select
                        selectedOption={hfRlaifPromptField ? { value: hfRlaifPromptField, label: hfRlaifPromptField } : null}
                        onChange={({ detail }) => setHfRlaifPromptField(detail.selectedOption.value!)}
                        options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                        placeholder="Choose column"
                      />
                    </FormField>
                  </SpaceBetween>
                )}

                <FormField
                  label={<span>Dataset name <i>- required</i></span>}
                  description="How you'll find this imported sample later."
                >
                  <Input
                    value={name}
                    onChange={({ detail }) => {
                      // A manual edit takes over — stop auto-syncing the name to
                      // the dataset id / sample size / seed.
                      setAutoName(false);
                      setName(detail.value);
                    }}
                  />
                </FormField>

                {hfImportAs === "sft" && (
                  <SpaceBetween size="m">
                    <SpaceBetween direction="horizontal" size="xs">
                      <FormField
                        label="User-turn column"
                        description="Becomes the prompt the model sees."
                      >
                        <Select
                          selectedOption={hfUserField ? { value: hfUserField, label: hfUserField } : null}
                          onChange={({ detail }) => setHfUserField(detail.selectedOption.value!)}
                          options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                          placeholder="Choose column"
                        />
                      </FormField>
                      <FormField
                        label="Assistant-turn column"
                        description="Becomes the target the model learns."
                      >
                        <Select
                          selectedOption={hfTargetField ? { value: hfTargetField, label: hfTargetField } : null}
                          onChange={({ detail }) => setHfTargetField(detail.selectedOption.value!)}
                          options={hfPrev.columnNames.map((c) => ({ value: c, label: c }))}
                          placeholder="Choose column"
                        />
                      </FormField>
                    </SpaceBetween>

                    <FormField
                      label="System prompt (optional)"
                      description="A fixed instruction applied to every row — becomes the `system` turn (e.g. 'Classify the news topic:'). The user turn stays the raw input."
                    >
                      <Input
                        value={hfInstruction}
                        placeholder="Classify the news topic:"
                        onChange={({ detail }) => setHfInstruction(detail.value)}
                      />
                    </FormField>

                    <SpaceBetween direction="horizontal" size="xs">
                      <FormField
                        label="Context column (optional)"
                        description="Appended to the user turn (e.g. a QA passage / extra field)."
                      >
                        <Select
                          selectedOption={
                            hfContextField ? { value: hfContextField, label: hfContextField } : { value: "", label: "(none)" }
                          }
                          onChange={({ detail }) => setHfContextField(detail.selectedOption.value || "")}
                          options={[
                            { value: "", label: "(none)" },
                            ...hfPrev.columnNames.map((c) => ({ value: c, label: c })),
                          ]}
                        />
                      </FormField>
                      <FormField
                        label="System column (optional)"
                        description="A per-row column to use as the system turn (combined with the fixed prompt above, if any)."
                      >
                        <Select
                          selectedOption={
                            hfSystemField ? { value: hfSystemField, label: hfSystemField } : { value: "", label: "(none)" }
                          }
                          onChange={({ detail }) => setHfSystemField(detail.selectedOption.value || "")}
                          options={[
                            { value: "", label: "(none)" },
                            ...hfPrev.columnNames.map((c) => ({ value: c, label: c })),
                          ]}
                        />
                      </FormField>
                    </SpaceBetween>
                  </SpaceBetween>
                )}

                <FormField
                  label="Sample size (rows)"
                  description="Max 2000. A few hundred is plenty to compare models."
                  errorText={!hfMaxRowsValid ? "1–2000" : undefined}
                >
                  <Input value={hfMaxRows} type="number" onChange={({ detail }) => setHfMaxRows(detail.value)} />
                </FormField>

                {/* DPO/KTO imports use the intuitive 3-way split control (incl. seed).
                    SFT keeps its own test-ratio + optional-val + stratify controls. */}
                {hfImportAs !== "sft" ? (
                  <SplitRatios
                    testRatio={hfTestRatio}
                    valRatio={hfValRatio}
                    seed={seed}
                    onTest={setHfTestRatio}
                    onVal={setHfValRatio}
                    onSeed={setSeed}
                  />
                ) : (
                  <>
                    <SpaceBetween direction="horizontal" size="xs">
                      <FormField label="Test ratio" errorText={!ratioValid ? "0–1" : undefined}>
                        <Input value={ratio} type="number" onChange={({ detail }) => setRatio(detail.value)} />
                      </FormField>
                      <FormField label="Seed" errorText={!seedValid ? "integer" : undefined}>
                        <Input value={seed} type="number" onChange={({ detail }) => setSeed(detail.value)} />
                      </FormField>
                    </SpaceBetween>
                    <Checkbox
                      checked={addVal}
                      onChange={({ detail }) => setAddVal(detail.checked)}
                    >
                      Add a validation set (carve from train) — enables early stopping
                    </Checkbox>
                    {addVal && (
                      <FormField label="Validation fraction (of train)" errorText={!valRatioValid ? "0–1" : undefined}>
                        <Input value={valRatio} type="number" onChange={({ detail }) => setValRatio(detail.value)} />
                      </FormField>
                    )}
                  </>
                )}

                {hfImportAs === "sft" && (
                  <Checkbox checked={stratify} onChange={({ detail }) => setStratify(detail.checked)}>
                    Use stratified sampling — keep each class proportionally represented across
                    train/val/test
                    <Box variant="small" color="text-status-inactive">
                      For classification datasets only; falls back to random for free-text/JSON.
                    </Box>
                  </Checkbox>
                )}

                {/* Objective-appropriate converted preview, so the mapping can be
                    sanity-checked before importing. */}
                {hfImportAs === "sft" && hfPrev.sampleRows.length > 0 && hfUserField && hfTargetField && (
                  <FormField
                    label="Converted preview"
                    description="How the first rows become chat messages with your current mapping."
                  >
                    <MessagesPreview rows={hfLivePreview} limit={2} />
                  </FormField>
                )}
                {hfImportAs === "preference" && (hfPrev.preferencePreview?.length ?? 0) > 0 && (
                  <FormField
                    label="Converted preview (DPO)"
                    description="How the first rows become preference pairs — the model learns to prefer 'chosen' over 'rejected'."
                  >
                    <PreferencePreview rows={hfPrev.preferencePreview as Record<string, unknown>[]} />
                  </FormField>
                )}
                {hfImportAs === "kto" && (hfPrev.ktoPreview?.length ?? 0) > 0 && (
                  <FormField
                    label="Converted preview (KTO)"
                    description="How the first rows become labelled completions — the model learns from the good/bad signal."
                  >
                    <KtoPreview rows={hfPrev.ktoPreview as Record<string, unknown>[]} />
                  </FormField>
                )}
                {hfImportAs === "rlvr" && (hfPrev.rlvrPreview?.length ?? 0) > 0 && (
                  <FormField
                    label="Converted preview (RLVR)"
                    description="How the first rows become prompt + verifiable ground_truth — the reward scorer checks the answer against ground_truth."
                  >
                    <MessagesPreview rows={hfPrev.rlvrPreview as Parameters<typeof MessagesPreview>[0]["rows"]} limit={2} />
                  </FormField>
                )}
                {hfImportAs === "rlaif" && (hfPrev.rlaifPreview?.length ?? 0) > 0 && (
                  <FormField
                    label="Converted preview (RLAIF)"
                    description="How the first rows become prompt-only messages — there's no gold answer; the AI judge scores a freshly generated response at training time."
                  >
                    <MessagesPreview rows={hfPrev.rlaifPreview as Parameters<typeof MessagesPreview>[0]["rows"]} limit={2} />
                  </FormField>
                )}

                <Button
                  variant="primary"
                  loading={creating}
                  disabled={
                    name.trim() === "" ||
                    !hfMaxRowsValid ||
                    !seedValid ||
                    // Per-objective required mapping + split. DPO/KTO/RLVR use the
                    // 3-way SplitRatios (hfTestRatio/hfValRatio); SFT uses its own
                    // test ratio + optional val (addVal/valRatio).
                    (hfImportAs === "preference"
                      ? hfChosenField === "" || hfRejectedField === "" ||
                        hfChosenField === hfRejectedField || !splitValid(hfTestRatio, hfValRatio)
                      : hfImportAs === "kto"
                        ? hfCompletionField === "" || hfLabelField === "" ||
                          !splitValid(hfTestRatio, hfValRatio)
                        : hfImportAs === "rlvr"
                          ? hfRlvrPromptField === "" || hfRlvrGroundTruthField === "" ||
                            hfRlvrPromptField === hfRlvrGroundTruthField ||
                            !splitValid(hfTestRatio, hfValRatio)
                          : hfImportAs === "rlaif"
                            ? hfRlaifPromptField === "" || !splitValid(hfTestRatio, hfValRatio)
                            : hfUserField === "" ||
                              hfTargetField === "" ||
                              hfUserField === hfTargetField ||
                              !ratioValid ||
                              (addVal && !valRatioValid))
                  }
                  onClick={doHfImport}
                >
                  {hfImportAs === "preference"
                    ? "Import as DPO dataset"
                    : hfImportAs === "kto"
                      ? "Import as KTO dataset"
                      : hfImportAs === "rlvr"
                        ? "Import as RLVR dataset"
                        : hfImportAs === "rlaif"
                          ? "Import as RLAIF dataset"
                          : "Import sample as dataset"}
                </Button>
                {report && <SplitResult report={report} />}
              </SpaceBetween>
            )}
          </SpaceBetween>
        ) : mode === "existing" ? (
          <FormField label="Dataset" description="Pick from your dataset library.">
            <Select
              selectedOption={
                selected
                  ? { value: selected.splitId, label: selected.name || selected.splitId }
                  : null
              }
              onChange={({ detail }) => {
                const d = datasets.find((x) => x.splitId === detail.selectedOption.value);
                onSelect(d ? datasetToCurrent(d) : null);
              }}
              options={datasets.map((d) => ({
                value: d.splitId,
                // Include the split id + val marker in the LABEL so same-named
                // datasets (e.g. several "hf:fancyzhx/ag_news") are distinguishable
                // at a glance — the #1 cause of "I picked the wrong one".
                label:
                  `[${shapeLabel(d.shape)}] ` +
                  (d.source === "huggingface" ? "🤗 " : "") +
                  (d.name || d.splitId) +
                  `  (${d.splitId.slice(0, 6)}` +
                  (d.hasVal ? ", val ✓" : ", no val") +
                  ")",
                description:
                  `${shapeLabel(d.shape)} · ` +
                  `${d.trainRows ?? "?"} train / ${d.evalRows ?? "?"} test` +
                  (d.hasVal ? ` / ${d.valRows ?? "?"} val` : " / no validation set") +
                  (d.source === "huggingface" && d.hfDataset ? ` · from ${d.hfDataset}` : ""),
              }))}
              placeholder="Choose a dataset"
              statusType={datasetsLoading ? "loading" : loadError ? "error" : "finished"}
              loadingText="Loading datasets…"
              errorText="Couldn't load datasets."
              recoveryText="Retry"
              onLoadItems={() => {
                if (loadError) loadDatasets();
              }}
              empty="No datasets yet — switch to 'LoRA SFT' to create one"
              filteringType="auto"
            />
          </FormField>
        ) : mode === "pref" ? (
          <SpaceBetween size="m">
            <Box variant="small" color="text-status-inactive">
              Upload a <b>preference</b> JSONL for <b>DPO / ORPO / SimPO</b> — one object per line with
              a prompt and two responses:{" "}
              <code>{`{"messages": [...], "chosen": {...}, "rejected": {...}}`}</code>{" "}
              (or a bare <code>prompt</code> string + string <code>chosen</code>/<code>rejected</code>).
              The model learns to prefer the chosen response; evaluation is unchanged (the chosen
              answer is the gold). <b>One preference dataset trains all three objectives</b> — pick
              DPO, ORPO, or SimPO per model on the Fine-tune page and race them.
            </Box>
            <Box variant="small">
              New to DPO?{" "}
              <Link
                variant="info"
                onFollow={() => {
                  setPrefFiles([demoFile("dpo")]);
                  if (!prefName.trim()) setPrefName("demo-dpo-support");
                }}
              >
                Use a demo dataset
              </Link>{" "}
              (120 sample preference pairs) ·{" "}
              <Link variant="info" onFollow={() => downloadDemo("dpo")}>
                Download the demo file
              </Link>
            </Box>
            <FormField
              label={<span>Dataset name <i>- required</i></span>}
              errorText={triedSubmit && prefName.trim() === "" ? "Enter a name for this dataset." : undefined}
            >
              <Input
                value={prefName}
                placeholder="preferences-v1"
                onChange={({ detail }) => setPrefName(detail.value)}
              />
            </FormField>
            <FormField
              label="Preference file (.jsonl)"
              errorText={triedSubmit && !prefFiles[0] ? "Choose a .jsonl file to upload." : undefined}
            >
              <FileUpload
                onChange={({ detail }) => setPrefFiles(detail.value)}
                value={prefFiles}
                accept=".jsonl,.json"
                i18nStrings={{
                  uploadButtonText: () => "Choose file",
                  removeFileAriaLabel: (i) => `Remove file ${i + 1}`,
                  dropzoneText: () => "Drop a .jsonl file",
                }}
                constraintText="One JSON object per line: prompt + chosen + rejected."
                showFileLastModified
                showFileSize
              />
            </FormField>
            <SplitRatios
              testRatio={prefTestRatio}
              valRatio={prefValRatio}
              seed={seed}
              onTest={setPrefTestRatio}
              onVal={setPrefValRatio}
              onSeed={setSeed}
            />
            <Button
              variant="primary"
              loading={prefBusy}
              disabled={!splitValid(prefTestRatio, prefValRatio)}
              onClick={createPreference}
            >
              Create preference dataset
            </Button>
          </SpaceBetween>
        ) : mode === "kto" ? (
          <SpaceBetween size="m">
            <Box variant="small" color="text-status-inactive">
              Upload a <b>KTO</b> JSONL — one object per line: a completion plus a good/bad label.{" "}
              <code>{`{"messages": [...,{"role":"assistant",...}], "kto_tag": true}`}</code> or{" "}
              <code>{`{"prompt": "...", "completion": "...", "label": "good"}`}</code>. Binary
              feedback, no pairing — cheaper to collect than DPO pairs. Needs both good AND bad
              examples. Evaluation is unchanged (the desirable answers are the gold).
            </Box>
            <Box variant="small">
              New to KTO?{" "}
              <Link
                variant="info"
                onFollow={() => {
                  setKtoFiles([demoFile("kto")]);
                  if (!ktoName.trim()) setKtoName("demo-kto-support");
                }}
              >
                Use a demo dataset
              </Link>{" "}
              (120 labelled examples, balanced good/bad) ·{" "}
              <Link variant="info" onFollow={() => downloadDemo("kto")}>
                Download the demo file
              </Link>
            </Box>
            <FormField
              label={<span>Dataset name <i>- required</i></span>}
              errorText={triedSubmit && ktoName.trim() === "" ? "Enter a name for this dataset." : undefined}
            >
              <Input
                value={ktoName}
                placeholder="kto-feedback-v1"
                onChange={({ detail }) => setKtoName(detail.value)}
              />
            </FormField>
            <FormField
              label="KTO file (.jsonl)"
              errorText={triedSubmit && !ktoFiles[0] ? "Choose a .jsonl file to upload." : undefined}
            >
              <FileUpload
                onChange={({ detail }) => setKtoFiles(detail.value)}
                value={ktoFiles}
                accept=".jsonl,.json"
                i18nStrings={{
                  uploadButtonText: () => "Choose file",
                  removeFileAriaLabel: (i) => `Remove file ${i + 1}`,
                  dropzoneText: () => "Drop a .jsonl file",
                }}
                constraintText="One JSON object per line: a completion + a good/bad label."
                showFileLastModified
                showFileSize
              />
            </FormField>
            <SplitRatios
              testRatio={ktoTestRatio}
              valRatio={ktoValRatio}
              seed={seed}
              onTest={setKtoTestRatio}
              onVal={setKtoValRatio}
              onSeed={setSeed}
            />
            <Box variant="small" color="text-status-inactive">
              The split is stratified so the good/bad balance is preserved across all three sets.
            </Box>
            <Button
              variant="primary"
              loading={ktoBusy}
              disabled={!splitValid(ktoTestRatio, ktoValRatio)}
              onClick={createKto}
            >
              Create KTO dataset
            </Button>
          </SpaceBetween>
        ) : mode === "rlvr" ? (
          <SpaceBetween size="m">
            <Box variant="small" color="text-status-inactive">
              Upload an <b>RLVR</b> JSONL — one object per line: a prompt plus a{" "}
              <b>verifiable</b> answer.{" "}
              <code>{`{"messages": [{"role":"user","content":"..."}], "ground_truth": "72"}`}</code> or{" "}
              <code>{`{"prompt": "...", "ground_truth": "72"}`}</code>. The model is trained with
              GRPO to maximize a <b>reward</b> for answers a preset scorer (gsm8k / prime_math)
              marks correct — not to imitate a worked solution. The{" "}
              <code>ground_truth</code> is what the scorer checks, so it's a field of its own.
              Runs on the <b>SageMaker Serverless</b> engine; you pick the reward function at launch.
            </Box>
            <Box variant="small">
              New to RLVR?{" "}
              <Link
                variant="info"
                onFollow={() => {
                  setRlvrFiles([demoFile("rlvr")]);
                  if (!rlvrName.trim()) setRlvrName("demo-rlvr-gsm8k");
                }}
              >
                Use a demo dataset
              </Link>{" "}
              (100 grade-school math problems → numeric answers; pair with the{" "}
              <code>gsm8k</code> reward) ·{" "}
              <Link variant="info" onFollow={() => downloadDemo("rlvr")}>
                Download the demo file
              </Link>
            </Box>
            <FormField
              label={<span>Dataset name <i>- required</i></span>}
              errorText={triedSubmit && rlvrName.trim() === "" ? "Enter a name for this dataset." : undefined}
            >
              <Input
                value={rlvrName}
                placeholder="rlvr-math-v1"
                onChange={({ detail }) => setRlvrName(detail.value)}
              />
            </FormField>
            <FormField
              label="RLVR file (.jsonl)"
              errorText={triedSubmit && !rlvrFiles[0] ? "Choose a .jsonl file to upload." : undefined}
            >
              <FileUpload
                onChange={({ detail }) => setRlvrFiles(detail.value)}
                value={rlvrFiles}
                accept=".jsonl,.json"
                i18nStrings={{
                  uploadButtonText: () => "Choose file",
                  removeFileAriaLabel: (i) => `Remove file ${i + 1}`,
                  dropzoneText: () => "Drop a .jsonl file",
                }}
                constraintText="One JSON object per line: a prompt + a verifiable ground_truth answer."
                showFileLastModified
                showFileSize
              />
            </FormField>
            <SplitRatios
              testRatio={rlvrTestRatio}
              valRatio={rlvrValRatio}
              seed={seed}
              onTest={setRlvrTestRatio}
              onVal={setRlvrValRatio}
              onSeed={setSeed}
            />
            <Box variant="small" color="text-status-inactive">
              The held-out test set scores the model on the leaderboard (its ground_truth answers
              become the gold), exactly like SFT/DPO/KTO datasets.
            </Box>
            <Button
              variant="primary"
              loading={rlvrBusy}
              disabled={!splitValid(rlvrTestRatio, rlvrValRatio)}
              onClick={createRlvr}
            >
              Create RLVR dataset
            </Button>
          </SpaceBetween>
        ) : mode === "rlaif" ? (
          <SpaceBetween size="m">
            <Box variant="small" color="text-status-inactive">
              Upload an <b>RLAIF</b> JSONL — one object per line, a prompt ONLY (no
              answer):{" "}
              <code>{`{"messages": [{"role":"user","content":"..."}]}`}</code> or{" "}
              <code>{`{"prompt": "..."}`}</code>. The model is trained with GRPO to maximize
              an <b>AI judge's</b> score — you define the judge as a <b>reward prompt</b> on
              the Reward functions page, and pick it at launch. Use RLAIF for SUBJECTIVE goals
              (tone, helpfulness, style) where there's no verifiable answer. Runs on the{" "}
              <b>SageMaker Serverless</b> engine.
            </Box>
            <Box variant="small">
              New to RLAIF?{" "}
              <Link
                variant="info"
                onFollow={() => {
                  setRlaifFiles([demoFile("rlaif")]);
                  if (!rlaifName.trim()) setRlaifName("demo-rlaif-notes");
                }}
              >
                Use a demo dataset
              </Link>{" "}
              (180 short writing prompts; the judge scores tone/helpfulness — pair with an
              AI-judge reward prompt at launch) ·{" "}
              <Link variant="info" onFollow={() => downloadDemo("rlaif")}>
                Download the demo file
              </Link>
            </Box>
            <FormField
              label={<span>Dataset name <i>- required</i></span>}
              errorText={triedSubmit && rlaifName.trim() === "" ? "Enter a name for this dataset." : undefined}
            >
              <Input
                value={rlaifName}
                placeholder="rlaif-helpfulness-v1"
                onChange={({ detail }) => setRlaifName(detail.value)}
              />
            </FormField>
            <FormField
              label="RLAIF file (.jsonl)"
              errorText={triedSubmit && !rlaifFiles[0] ? "Choose a .jsonl file to upload." : undefined}
            >
              <FileUpload
                onChange={({ detail }) => setRlaifFiles(detail.value)}
                value={rlaifFiles}
                accept=".jsonl,.json"
                i18nStrings={{
                  uploadButtonText: () => "Choose file",
                  removeFileAriaLabel: (i) => `Remove file ${i + 1}`,
                  dropzoneText: () => "Drop a .jsonl file",
                }}
                constraintText="One JSON object per line: a prompt only (no ground_truth — the AI judge scores the response)."
                showFileLastModified
                showFileSize
              />
            </FormField>
            <SplitRatios
              testRatio={rlaifTestRatio}
              valRatio={rlaifValRatio}
              seed={seed}
              onTest={setRlaifTestRatio}
              onVal={setRlaifValRatio}
              onSeed={setSeed}
            />
            <Box variant="small" color="text-status-inactive">
              GRPO needs ≥128 training prompts (≥143 raw without a separate validation file).
              The held-out test prompts feed the leaderboard's judge-based eval.
            </Box>
            <Button
              variant="primary"
              loading={rlaifBusy}
              disabled={!splitValid(rlaifTestRatio, rlaifValRatio)}
              onClick={createRlaif}
            >
              Create RLAIF dataset
            </Button>
          </SpaceBetween>
        ) : (
          <SpaceBetween size="m">
            <Box variant="small">
              New to fine-tuning?{" "}
              <Link
                variant="info"
                onFollow={() => {
                  setAutoFiles([demoFile("sft")]);
                  if (!name.trim()) setName("demo-news-topics");
                  setStratify(true); // it's a balanced label task — show stratified split
                }}
              >
                Use a demo dataset
              </Link>{" "}
              (60 news headlines → topic label: World / Sports / Business / Sci/Tech) ·{" "}
              <Link variant="info" onFollow={() => downloadDemo("sft")}>
                Download the demo file
              </Link>
            </Box>
            <FormField
              label={<span>Dataset name <i>- required</i></span>}
              description="Human-friendly label, e.g. support-tickets-v1. Required — it's how you'll find this dataset in the pickers and Datasets page."
              errorText={triedSubmit && name.trim() === "" ? "Enter a name for this dataset." : undefined}
            >
              <Input value={name} placeholder="support-tickets-v1" onChange={({ detail }) => setName(detail.value)} />
            </FormField>
            {/* SFT is single-file like every other tab: upload one dataset and the
                backend carves train/val/test (the old two-file "bring your own test"
                mode was removed for consistency — Test/Validation are set below). */}
            <FormField
              label="Dataset file (JSONL)"
              errorText={triedSubmit && autoFiles.length === 0 ? "Choose a .jsonl file to upload." : undefined}
            >
              <FileUpload
                onChange={({ detail }) => setAutoFiles(detail.value)}
                value={autoFiles}
                accept=".jsonl,.json,.txt"
                i18nStrings={fileUploadI18n}
                showFileSize
              />
            </FormField>
            {/* One file → intuitive Train/Val/Test (= 100%) + seed, same control
                as DPO/KTO. Val is a fraction of the WHOLE (backend converts to a
                train-slice carve), so the three sum to 1. */}
            <SplitRatios
              testRatio={ratio}
              valRatio={valRatio}
              seed={seed}
              onTest={setRatio}
              onVal={(v) => {
                setValRatio(v);
                setAddVal(Number(v) > 0); // a non-zero val carve enables early stopping
              }}
              onSeed={setSeed}
            />
            <Checkbox checked={stratify} onChange={({ detail }) => setStratify(detail.checked)}>
              Use stratified sampling — keep each class proportionally represented across
              train/val/test
              <Box variant="small" color="text-status-inactive">
                Applies to SFT classification datasets only; falls back to random for
                free-text/JSON. (Not used for DPO — every row is one chosen/rejected pair; KTO
                auto-balances its good/bad labels across the split.)
              </Box>
            </Checkbox>
            <Button
              variant="primary"
              loading={creating}
              // Only genuinely-invalid NUMERIC settings disable the button (the
              // split ratios can't produce a valid dataset). Missing name/file no
              // longer disable it — clicking highlights them via errorText instead,
              // so the user understands what's needed rather than facing a dead button.
              disabled={
                !ratioValid || !seedValid || (addVal && !valRatioValid) ||
                // Train must stay positive: test + val can't reach 100%.
                ratioNum + (addVal && valRatioValid ? valRatioNum : 0) >= 1
              }
              onClick={() => {
                if (name.trim() === "" || autoFiles.length === 0) {
                  setTriedSubmit(true);
                  return;
                }
                create(() =>
                  autoSplit(autoFiles[0], ratioNum, seedNum, name, addVal && valRatioValid ? valRatioNum : null, stratify)
                );
              }}
            >
              Create dataset
            </Button>
            {report && <SplitResult report={report} />}
          </SpaceBetween>
        )}

        {selected && (
          <Alert type="success">
            <SpaceBetween size="xs">
              <SpaceBetween direction="horizontal" size="xs">
                <span>
                  Selected dataset: <b>{selected.name || selected.splitId}</b> ·{" "}
                  {selected.trainRows} train / {selected.evalRows} test
                  {selected.hasVal ? ` / ${selected.valRows} val` : ""}
                </span>
                {selected.hasVal ? (
                  <Badge color="green">validation set ✓ early stopping available</Badge>
                ) : (
                  <Badge color="grey">no validation set</Badge>
                )}
              </SpaceBetween>
              <Button iconName="search" onClick={() => setInvestigating(true)}>
                Investigate this dataset
              </Button>
            </SpaceBetween>
          </Alert>
        )}
        {selected && investigating && (
          <InvestigateDataset
            splitId={selected.splitId}
            name={selected.name}
            onClose={() => setInvestigating(false)}
          />
        )}
        {mode === "new" && (
          <Box variant="small">
            Tip: create the dataset, then it's selectable under "Use existing" here and on other pages.
          </Box>
        )}
      </SpaceBetween>
    </Container>
  );
}
