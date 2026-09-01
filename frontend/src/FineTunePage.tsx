// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useMemo, useRef, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import Container from "@cloudscape-design/components/container";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Select, { type SelectProps } from "@cloudscape-design/components/select";
import Multiselect, { type MultiselectProps } from "@cloudscape-design/components/multiselect";
import Input from "@cloudscape-design/components/input";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Table from "@cloudscape-design/components/table";
import Badge from "@cloudscape-design/components/badge";
import Toggle from "@cloudscape-design/components/toggle";
import ExpandableSection from "@cloudscape-design/components/expandable-section";
import Popover from "@cloudscape-design/components/popover";
import Link from "@cloudscape-design/components/link";
import Modal from "@cloudscape-design/components/modal";
import Wizard from "@cloudscape-design/components/wizard";
import Tabs from "@cloudscape-design/components/tabs";
import Spinner from "@cloudscape-design/components/spinner";
import CodeView from "@cloudscape-design/code-view/code-view";

import {
  autofillRace,
  checkRewardDomain,
  createRewardFunction,
  getDatasets,
  getHfTokenStatus,
  getModels,
  launchRace,
  listRewardFunctions,
  recommendConfig,
  renderConfig,
  type AutofillArm,
  type CurrentSplit,
  type FinetuningType,
  type LoraVariant,
  type PrefLoss,
  type ModelSpec,
  type RaceModelConfig,
  type RaceRequest,
  type RecommendRationale,
  type RecommendResponse,
  type RenderResponse,
} from "./api";
import { DatasetPicker } from "./DatasetPicker";
import { getCurrentUser } from "./auth";

// Plain-language explanations for each hyperparameter (shown via info popovers).
const HP_HELP: Record<string, { title: string; body: string }> = {
  loraRank: {
    title: "LoRA rank (r)",
    body: "Size of the low-rank adapter matrices. Higher = more trainable capacity (can fit more, but more params, slower, more VRAM, higher overfit risk). Typical: 8–64. Generic across models.",
  },
  loraAlpha: {
    title: "LoRA alpha",
    body: "Scaling applied to the LoRA update (effective scale ≈ alpha / rank). Common practice sets alpha = 2 × rank. Leave blank to auto-use 2 × rank.",
  },
  loraplusLrRatio: {
    title: "LoRA+ LR ratio (λ)",
    body: "LoRA+ trains the adapter's B matrix at a higher learning rate than A; this is the B/A ratio. The paper finds ~16 a good default — higher speeds convergence but can destabilize. Only used when the LoRA variant is LoRA+.",
  },
  learningRate: {
    title: "Learning rate",
    body: "Step size for optimizer updates. LoRA fine-tuning typically uses 1e-4 to 2e-5. Too high → unstable/diverges; too low → barely learns.",
  },
  numTrainEpochs: {
    title: "Epochs",
    body: "How many times the trainer passes over the full training set. More epochs = more learning but more overfitting risk on small datasets. Often 1–3 for instruction tuning.",
  },
  perDeviceTrainBatchSize: {
    title: "Per-device batch size",
    body: "Training examples processed per GPU per step. Higher = faster but more VRAM. Combine with grad-accumulation to reach a larger effective batch without more memory.",
  },
  gradientAccumulationSteps: {
    title: "Gradient accumulation steps",
    body: "Number of forward/backward passes accumulated before an optimizer step. Effective batch = per-device batch × this. Lets you simulate large batches on small GPUs.",
  },
  cutoffLen: {
    title: "Cutoff length",
    body: "Max sequence length (tokens). Sequences longer than this are truncated. Higher = more context but more VRAM. Blank uses the model's catalog default (set per-model).",
  },
  saveSteps: {
    title: "Save steps",
    body: "Checkpoint frequency (every N training steps). Also enables resume from the latest checkpoint (used for spot-instance interruption recovery).",
  },
  maxSamples: {
    title: "Max samples",
    body: "Cap on training rows used (after shuffling). Blank = use all rows. Useful to cap cost for a quick smoke test. Does not affect the held-out test set.",
  },
  earlyStoppingPatience: {
    title: "Early stopping patience",
    body: "Number of consecutive evaluations with no improvement in validation loss before training stops. Higher = more tolerant of noise (trains longer); lower = stops sooner. Typical: 2–4.",
  },
  ktoChosenWeight: {
    title: "KTO desirable weight (λD)",
    body: "Loss weight on DESIRABLE (good) completions in KTO. 1.0 = neutral. Raise it when desirable examples are the MINORITY so each class contributes equally — the KTO paper recommends keeping λD·nD ≈ λU·nU (§4.2). Investigate the dataset for a concrete recommendation.",
  },
  ktoRejectedWeight: {
    title: "KTO undesirable weight (λU)",
    body: "Loss weight on UNDESIRABLE (bad) completions in KTO. 1.0 = neutral. Raise it when undesirable examples are the MINORITY. The KTO paper keeps λD·nD ≈ λU·nU so neither class dominates (§4.2); the Investigate card suggests exact values for your label balance.",
  },
};

function HpInfo({ k }: { k: string }) {
  const h = HP_HELP[k];
  if (!h) return null;
  return (
    <Popover header={h.title} content={h.body} triggerType="custom" dismissButton={false} position="top">
      <Link variant="info">Info</Link>
    </Popover>
  );
}

// A staged model with its own hyperparameters (the "batch cart").
interface Staged {
  // Stable per-CARD instance id. The cart can hold MULTIPLE cards for the same
  // modelId (e.g. to race DoRA vs rsLoRA of one model), so cards/removal/render
  // key on `iid`, not modelId or array index. Assigned once when a card is added
  // (picker or Duplicate) and never reused.
  iid: string;
  modelId: string;
  display: string;
  family: string;
  gated: boolean;
  engine: "llama_factory" | "sagemaker_serverless";
  stage: "sft" | "dpo" | "kto" | "rlvr" | "rlaif";
  // RLVR reward: EITHER a preset (presetRewardFunction) OR a custom reward function
  // (rewardFunctionId). RLAIF reward: a reward-PROMPT id in rewardFunctionId (no
  // preset) + an optional judge model in rewardModelId. "" for non-GRPO objectives.
  presetRewardFunction: string;
  rewardFunctionId: string;
  rewardModelId: string;
  finetuningType: FinetuningType;
  loraRank: string;
  loraAlpha: string;
  // LoRA adapter variant (rides lora/qlora; ignored for full/freeze). "lora" =
  // plain. loraplusLrRatio is only used when loraVariant === "loraplus".
  loraVariant: LoraVariant;
  loraplusLrRatio: string;
  freezeTrainableLayers: string; // freeze-only: # top layers to train
  learningRate: string;
  numTrainEpochs: string;
  perDeviceTrainBatchSize: string;
  gradientAccumulationSteps: string;
  cutoffLen: string;
  saveSteps: string;
  maxSamples: string;
  earlyStoppingEnabled: boolean;
  earlyStoppingPatience: string;
  // KTO-only per-class loss weights (λD/λU). "1" = neutral.
  ktoChosenWeight: string;
  ktoRejectedWeight: string;
  // Preference-loss family for a DPO-shaped dataset: "sigmoid" = standard DPO,
  // "orpo"/"simpo" = reference-free alternatives (race them against DPO). Only
  // meaningful when wantStage === "dpo". simpoGamma is SimPO's target margin.
  prefLoss: PrefLoss;
  simpoGamma: string;
  // Efficiency knobs (LLaMA-Factory engine only). NEFTune embedding noise (0=off),
  // Liger fused kernels, and SFT sequence packing. Defaults are no-ops.
  neftuneNoiseAlpha: string;
  enableLigerKernel: boolean;
  packing: boolean;
}

function num(s: string): number | undefined {
  if (s.trim() === "") return undefined;
  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}

// Adapter methods carry a LoRA adapter (so a variant applies); full/freeze are
// full-weight (no adapter — no variant). Mirrors render.py's is_full_weight gate.
function isAdapter(method: FinetuningType): boolean {
  return method === "lora" || method === "qlora";
}

// Monotonic per-card instance-id generator. A plain counter (not random) so ids
// are stable + collision-free within a session; the cart holds many cards for one
// model, each needing a unique key for React + removal.
let _iidSeq = 0;
function nextIid(): string {
  _iidSeq += 1;
  return `card-${_iidSeq}`;
}

// Mirrors the Render page's editable hyperparameters (full set). Blank fields
// fall back to backend defaults: loraAlpha→auto (2×rank), cutoffLen→model default,
// maxSamples→all rows.
const DEFAULTS = {
  // Training engine. "llama_factory" (default) is our frozen-image path —
  // every model + every method/objective. "sagemaker_serverless" routes to
  // SageMaker's managed serverless customization (no infra; SFT/DPO + LoRA only),
  // available only for models with a Public-Hub mapping (model.engines).
  engine: "llama_factory" as "llama_factory" | "sagemaker_serverless",
  // Objective (LLaMA-Factory stage). "sft" is the default; "dpo" needs a
  // preference-shaped dataset (the dataset gates this — see objectiveForSplit).
  // "rlvr" (GRPO vs a verifiable reward) is serverless-only + reuses sft data.
  stage: "sft" as "sft" | "dpo" | "kto" | "rlvr" | "rlaif",
  // RLVR-only: which preset reward fn drives GRPO (gsm8k|prime_math). Resolved at
  // launch to an auto-provisioned built-in reward function (Evaluator ARN).
  presetRewardFunction: "" as string,
  // RLVR: a custom reward function id (alt to a preset). RLAIF: a reward-PROMPT id.
  rewardFunctionId: "" as string,
  // RLAIF-only: the judge model id ("" = recipe default).
  rewardModelId: "" as string,
  // Parameterization method. "lora" keeps the original one-click path; "qlora"
  // is LoRA on a 4-bit base (fits bigger models cheaper); "full"/"freeze" are
  // full-weight (no adapter, standalone model — small models + SFT only). Not a
  // numeric field — excluded from NumKey below; rendered as a Select.
  finetuningType: "lora" as FinetuningType,
  loraRank: "8",
  loraAlpha: "",
  // LoRA adapter variant. "lora" = plain (byte-identical to before). DoRA/rsLoRA/
  // PiSSA/LoRA+ ride the same adapter path; only sent for adapter methods. The
  // LoRA+ ratio (B/A LR) is only meaningful when loraVariant === "loraplus".
  loraVariant: "lora" as LoraVariant,
  loraplusLrRatio: "16",
  freezeTrainableLayers: "2",
  learningRate: "0.0001",
  numTrainEpochs: "3",
  perDeviceTrainBatchSize: "1",
  gradientAccumulationSteps: "8",
  cutoffLen: "",
  saveSteps: "500",
  maxSamples: "",
  earlyStoppingEnabled: false,
  earlyStoppingPatience: "3",
  // KTO-only per-class loss weights (λD/λU). "1" = neutral; the Investigate
  // card recommends raising the minority class to fix label imbalance. Only
  // sent when stage=kto; backend ignores them otherwise + emits nothing at 1.0.
  ktoChosenWeight: "1",
  ktoRejectedWeight: "1",
  // Preference-loss family (DPO datasets). "sigmoid" = standard DPO (default);
  // "orpo"/"simpo" are reference-free. Only sent when wantStage === "dpo".
  prefLoss: "sigmoid" as PrefLoss,
  simpoGamma: "0.5",
  // Efficiency knobs (llama_factory only). No-op defaults keep configs byte-identical;
  // only sent to the backend when changed.
  neftuneNoiseAlpha: "0",
  enableLigerKernel: false,
  packing: false,
};

// Full-weight (full/freeze) learning-rate default. Full-FT at the LoRA-scale 1e-4
// diverges; LLaMA-Factory's full_sft example uses 1e-5. Switching the method picker
// into full/freeze auto-sets this (the backend also clamps it as a safety net).
const FULL_WEIGHT_LR = "0.00001";
// GRPO (RLVR/RLAIF) learning-rate default. GRPO is RL, not supervised — the live
// managed GRPO recipes specify learning_rate default 1e-5 (range [1e-7, 1e-3]); the
// SFT-scale 1e-4 is 10x too high and risks an unstable/wasted run. Switching the
// objective to RLVR/RLAIF auto-sets this when the LR is still the SFT default (the
// engine also snaps an inherited SFT LR down as an unbypassable safety net).
const GRPO_LR = "0.00001";

// Curated provider/family display order for the model picker (step 2): most-popular
// open-weight SLM providers first. Families NOT listed here (e.g. a newly-onboarded
// custom model) sort after these, in catalog order. Edit this list to re-rank — it's
// purely a display ordering, never gates which models are available. Values must
// match ModelInfo.family from the catalog.
const FAMILY_ORDER = [
  "Llama",
  "Qwen3",
  "Qwen2.5",
  "Gemma",
  "Mistral",
  "Phi",
  "DeepSeek-R1-Distill",
  "GPT-OSS",
  "GLM",
  "Granite",
  "InternLM",
  "MiniCPM",
  "LFM2",
];

const METHOD_LABEL: Record<FinetuningType, string> = {
  lora: "LoRA (default)",
  qlora: "QLoRA (4-bit base)",
  full: "Full fine-tuning",
  freeze: "Freeze (top layers)",
};

// The selected-method description shown under the picker.
const METHOD_HELP: Record<FinetuningType, string> = {
  lora: "LoRA: the proven default. Switch to QLoRA to fit a bigger model on the same GPU. Compare methods by launching one run of each on this dataset.",
  qlora: "QLoRA: LoRA on a 4-bit-quantized base — fits bigger models on the same GPU, merges to full-precision weights at export.",
  full: "Full fine-tuning: updates EVERY weight (no adapter — a standalone model). Highest capacity + cost; needs a low learning rate (auto-set). Small models + SFT only.",
  freeze: "Freeze: trains only the top N transformer layers (a lighter step up from LoRA, no adapter). Lower memory than full; needs a low learning rate (auto-set).",
};

// The per-option hint in the dropdown.
const METHOD_OPTION_HELP: Record<FinetuningType, string> = {
  lora: "The proven default — adapter on the full-precision base",
  qlora: "LoRA on a 4-bit-quantized base — smaller GPU footprint",
  full: "Full-weight training — standalone model, higher cost (small models only)",
  freeze: "Train only the top layers — lighter full-weight option",
};

// LoRA adapter VARIANT picker (a modifier on lora/qlora). All ride the same
// rank/alpha/target + merge/export, so they're cheap to race against plain LoRA
// — "does the fancier adapter beat LoRA on YOUR data?".
const VARIANT_LABEL: Record<LoraVariant, string> = {
  lora: "None (plain LoRA)",
  dora: "DoRA",
  rslora: "rsLoRA",
  pissa: "PiSSA",
  loraplus: "LoRA+",
};
// The per-option hint in the variant dropdown.
const VARIANT_OPTION_HELP: Record<LoraVariant, string> = {
  lora: "Standard LoRA — the proven default",
  dora: "Weight-decomposed LoRA — often closes more of the full-FT quality gap",
  rslora: "Rank-stabilized scaling — steadier training at higher ranks",
  pissa: "SVD-based init — can converge faster (slowest to start; heaviest)",
  loraplus: "Higher learning rate for the B matrix — faster convergence",
};
// Shown under the variant picker (selected-variant description).
const VARIANT_HELP: Record<LoraVariant, string> = {
  lora: "Plain LoRA. Pick a variant to try a richer adapter that can recover more of the full fine-tuning quality on the cheap LoRA path — race it against plain LoRA to see if it wins on your data.",
  dora: "DoRA decomposes each weight into magnitude + direction and adapts both — frequently beats plain LoRA at the same rank. Merges to standalone weights at export like LoRA.",
  rslora: "rsLoRA rescales the adapter by 1/√rank instead of 1/rank, which stabilizes training at higher ranks. Merge-identical to LoRA.",
  pissa: "PiSSA initializes the adapter from an SVD of the base weights (rather than zeros), which can converge faster and higher. Heaviest variant (SVD init); merges back onto the original base at export.",
  loraplus: "LoRA+ trains the B matrix with a higher learning rate than A (set the ratio below), which speeds convergence. Merge-identical to LoRA.",
};

// Variant cell for the staged-batch table: a small badge so a DoRA/PiSSA row
// reads as a DISTINCT leaderboard entry, not silently as plain LoRA.
function variantBadge(variant: LoraVariant) {
  if (variant === "lora") return null;
  return <Badge color="blue">{VARIANT_LABEL[variant]}</Badge>;
}

// Variants that need the FULL-PRECISION weight matrix and so CANNOT run on a 4-bit
// QLoRA base — DoRA (magnitude×direction decomposition) and PiSSA (SVD init). PEFT
// hard-rejects them on a quantized base at model-load, so we must not let the picker
// pair them with QLoRA (a real race burned 2 jobs on exactly this). Mirrors the
// backend's Hyperparams.QUANT_INCOMPATIBLE_VARIANTS — the source of truth + guard.
const QUANT_INCOMPATIBLE_VARIANTS: LoraVariant[] = ["dora", "pissa"];
function variantAllowedForMethod(variant: LoraVariant, method: FinetuningType): boolean {
  if (method === "qlora") return !QUANT_INCOMPATIBLE_VARIANTS.includes(variant);
  return true; // plain LoRA allows all; full/freeze hide the picker entirely
}

// Method cell for the staged-batch tables: a colored badge per method so full/
// freeze read as DISTINCT leaderboard rows, not silently as "LoRA".
const METHOD_BADGE_COLOR: Record<FinetuningType, "blue" | "green" | "red" | "grey"> = {
  lora: "grey", qlora: "blue", full: "green", freeze: "red",
};
function methodBadge(method: FinetuningType) {
  if (method === "lora") return "LoRA";
  return <Badge color={METHOD_BADGE_COLOR[method]}>{METHOD_LABEL[method].replace(/ \(.*\)$/, "")}</Badge>;
}

// Verification key (model, image-tier, METHOD): LoRA = bare tier; others under
// `<tier>::<method>`. Mirrors the backend so the card can show whether the chosen
// method is proven for this model.
function methodStatusFor(m: ModelSpec, method: FinetuningType): string {
  const key = method === "lora" ? m.imageTag : `${m.imageTag}::${method}`;
  return m.verifications?.[key]?.status ?? "untested";
}

// Verification key for a LoRA VARIANT (model, image-tier, method, variant): a
// non-plain variant lives under `<tier>::<method>::<variant>` (the method token is
// always present alongside it). Plain "lora" has no variant key — it falls back to
// the method's own status. Mirrors the backend _key() so the card shows whether
// the chosen variant (not just the base method) is proven for this model.
function variantStatusFor(m: ModelSpec, method: FinetuningType, variant: LoraVariant): string {
  if (variant === "lora") return methodStatusFor(m, method);
  return m.verifications?.[`${m.imageTag}::${method}::${variant}`]?.status ?? "untested";
}

type NumKey =
  | "loraRank" | "loraAlpha" | "loraplusLrRatio" | "freezeTrainableLayers" | "learningRate"
  | "numTrainEpochs" | "perDeviceTrainBatchSize" | "gradientAccumulationSteps"
  | "cutoffLen" | "saveSteps" | "maxSamples" | "earlyStoppingPatience"
  | "ktoChosenWeight" | "ktoRejectedWeight" | "simpoGamma" | "neftuneNoiseAlpha";

// Preference-loss family options (shown for a DPO-shaped dataset). DPO is the
// default; ORPO/SimPO are reference-free alternatives to race against it.
const PREF_LOSS_LABEL: Record<PrefLoss, string> = {
  sigmoid: "DPO (standard)",
  orpo: "ORPO (reference-free)",
  simpo: "SimPO (reference-free)",
};
const PREF_LOSS_HELP: Record<PrefLoss, string> = {
  sigmoid: "Standard DPO — log-prob ratio vs a frozen reference model. The proven default.",
  orpo: "ORPO folds preference into one SFT-style stage with NO reference model — cheaper, often competitive. Race it against DPO on your data.",
  simpo: "SimPO is reference-free with a length-normalized reward + target margin (γ). Cheaper than DPO; tune γ if it under/over-shoots.",
};

// Per-model config card — a faithful copy of the single-model hyperparameter form,
// bound to ONE staged entry (`entry`) instead of a shared draft. Every selected
// model gets its own card with the full set of HP + early-stopping options + both
// AI helpers, all editable. Edits flow up via onChange(patch); the card owns its
// own suggest/advise/rationale/sweep/error state so cards don't interfere.
function ModelConfigCard({
  entry,
  model,
  bounds,
  currentSplit,
  effectiveStage,
  wantStage,
  isRlvr,
  isRlaif,
  grpoTooSmall,
  grpoMinRows,
  rewardFns,
  onRewardFnsChange,
  onChange,
  onRemove,
  onDuplicate,
  onPreview,
}: {
  entry: Staged;
  model: ModelSpec;
  bounds: Record<string, { min: number; max: number }>;
  currentSplit: CurrentSplit | null;
  effectiveStage: Staged["stage"];
  wantStage: Staged["stage"];
  isRlvr: boolean;
  isRlaif: boolean;
  grpoTooSmall: boolean;
  grpoMinRows: number;
  rewardFns: import("./api").RewardFunction[];
  onRewardFnsChange: (fns: import("./api").RewardFunction[]) => void;
  onChange: (patch: Partial<Staged>) => void;
  onRemove: () => void;
  onDuplicate: () => void;
  onPreview: () => void;
}) {
  const [suggesting, setSuggesting] = useState(false);
  const [advising, setAdvising] = useState(false);
  const [rationale, setRationale] = useState<RecommendRationale[] | null>(null);
  const [sweep, setSweep] = useState<import("./api").AdviseResponse | null>(null);
  const [applyingReward, setApplyingReward] = useState(false);
  const [cardError, setCardError] = useState<string | null>(null);
  const [rewardDomainWarning, setRewardDomainWarning] = useState<string | null>(null);

  // RLVR reward-domain guard (advisory) — re-run on split/reward change.
  const splitIdForGuard = currentSplit?.splitId ?? null;
  useEffect(() => {
    if (!isRlvr || !splitIdForGuard) {
      setRewardDomainWarning(null);
      return;
    }
    let cancelled = false;
    checkRewardDomain({
      splitId: splitIdForGuard,
      presetRewardFunction: entry.rewardFunctionId ? "" : (entry.presetRewardFunction || "gsm8k"),
      rewardFunctionId: entry.rewardFunctionId || "",
    })
      .then((r) => { if (!cancelled) setRewardDomainWarning(r.warning); })
      .catch(() => { if (!cancelled) setRewardDomainWarning(null); });
    return () => { cancelled = true; };
  }, [isRlvr, splitIdForGuard, entry.presetRewardFunction, entry.rewardFunctionId]);

  const isFullWeight = entry.finetuningType === "full" || entry.finetuningType === "freeze";

  // Deterministic "Tune for my dataset" — fills THIS entry's numeric HP.
  async function suggestDefaults() {
    setSuggesting(true);
    setCardError(null);
    try {
      const rec = await recommendConfig({
        modelId: model.id,
        splitId: currentSplit?.splitId ?? null,
        trainRows: currentSplit?.trainRows ?? null,
        hasVal: currentSplit?.hasVal ?? null,
        finetuningType: entry.finetuningType,
      });
      const h = rec.hyperparams;
      onChange({
        freezeTrainableLayers: h.freezeTrainableLayers != null
          ? String(h.freezeTrainableLayers) : entry.freezeTrainableLayers,
        loraRank: isFullWeight ? entry.loraRank : String(h.loraRank),
        loraAlpha: isFullWeight ? entry.loraAlpha : (h.loraAlpha == null ? "" : String(h.loraAlpha)),
        learningRate: String(h.learningRate),
        numTrainEpochs: String(h.numTrainEpochs),
        perDeviceTrainBatchSize: String(h.perDeviceTrainBatchSize),
        gradientAccumulationSteps: String(h.gradientAccumulationSteps),
        cutoffLen: h.cutoffLen == null ? "" : String(h.cutoffLen),
        saveSteps: String(h.saveSteps),
        maxSamples: h.maxSamples == null ? "" : String(h.maxSamples),
        earlyStoppingEnabled: h.earlyStoppingEnabled && !!currentSplit?.hasVal,
        earlyStoppingPatience: String(h.earlyStoppingPatience),
      });
      setRationale(rec.rationale);
    } catch (e) {
      setCardError(e instanceof Error ? e.message : String(e));
    } finally {
      setSuggesting(false);
    }
  }

  // AI "Propose a sweep" — applying a config fills THIS entry's numeric HP.
  async function suggestSweep() {
    setAdvising(true);
    setCardError(null);
    try {
      const { adviseConfig } = await import("./api");
      const r = await adviseConfig({
        modelId: model.id,
        splitId: currentSplit?.splitId ?? null,
        trainRows: currentSplit?.trainRows ?? null,
        hasVal: currentSplit?.hasVal ?? null,
        finetuningType: entry.finetuningType,
      });
      setSweep(r);
    } catch (e) {
      setCardError(e instanceof Error ? e.message : String(e));
    } finally {
      setAdvising(false);
    }
  }

  function applySweepConfig(h: RecommendResponse["hyperparams"]) {
    onChange({
      freezeTrainableLayers: (h as { freezeTrainableLayers?: number }).freezeTrainableLayers != null
        ? String((h as { freezeTrainableLayers?: number }).freezeTrainableLayers) : entry.freezeTrainableLayers,
      loraRank: isFullWeight ? entry.loraRank : String(h.loraRank),
      loraAlpha: isFullWeight ? entry.loraAlpha : (h.loraAlpha == null ? "" : String(h.loraAlpha)),
      // Carry an advised LoRA variant (e.g. the DoRA arm) — adapter methods only, so
      // applying a sweep arm to a full/freeze entry never sets a variant.
      loraVariant: isFullWeight ? entry.loraVariant : ((h.loraVariant as LoraVariant) ?? "lora"),
      loraplusLrRatio: h.loraplusLrRatio != null ? String(h.loraplusLrRatio) : entry.loraplusLrRatio,
      learningRate: String(h.learningRate),
      numTrainEpochs: String(h.numTrainEpochs),
      perDeviceTrainBatchSize: String(h.perDeviceTrainBatchSize),
      gradientAccumulationSteps: String(h.gradientAccumulationSteps),
      cutoffLen: h.cutoffLen == null ? "" : String(h.cutoffLen),
      saveSteps: String(h.saveSteps),
      maxSamples: h.maxSamples == null ? "" : String(h.maxSamples),
      earlyStoppingEnabled: h.earlyStoppingEnabled && !!currentSplit?.hasVal,
      earlyStoppingPatience: String(h.earlyStoppingPatience),
    });
  }

  async function applyRecommendedReward(metric: string) {
    setApplyingReward(true);
    setCardError(null);
    try {
      const fresh = await listRewardFunctions();
      let rf = fresh.rewardFunctions.find(
        (r) => r.kind === "metric" && r.metric === metric && r.status !== "failed",
      );
      if (!rf) rf = await createRewardFunction({ name: `rank-${metric}`, metric });
      onRewardFnsChange(rewardFns.some((p) => p.id === rf!.id) ? rewardFns : [rf!, ...rewardFns]);
      onChange({ rewardFunctionId: rf!.id, presetRewardFunction: "" });
    } catch (e) {
      setCardError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplyingReward(false);
    }
  }

  // A numeric field bound to this entry (mirrors the single-model draftField).
  function field(label: string, key: NumKey, description?: string) {
    const b = bounds[key];
    const raw = entry[key];
    const n = Number(raw);
    const err =
      raw !== "" && b
        ? !Number.isFinite(n)
          ? "Must be a number."
          : n < b.min || n > b.max
            ? `Must be between ${b.min} and ${b.max}.`
            : undefined
        : undefined;
    return (
      <FormField label={label} description={description} info={<HpInfo k={key} />} errorText={err}>
        <Input value={entry[key]} type="number" onChange={({ detail }) => onChange({ [key]: detail.value } as Partial<Staged>)} />
      </FormField>
    );
  }

  const serverless = entry.engine === "sagemaker_serverless";
  const reasoningInstance = model.suggestedInstance;

  return (
    <Container
      header={
        <Header
          variant="h3"
          actions={
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="inline-link" iconName="file" onClick={onPreview}>Preview YAML</Button>
              {/* Duplicate this model's card as a new instance — clone now, then
                  change (typically) the LoRA variant so the same model races
                  head-to-head as distinct entries. */}
              <Button variant="inline-link" iconName="copy" onClick={onDuplicate}>Duplicate</Button>
              <Button variant="inline-link" iconName="remove" onClick={onRemove}>Remove</Button>
            </SpaceBetween>
          }
        >
          <SpaceBetween direction="horizontal" size="xs">
            <span>{entry.display}</span>
            <Badge>{entry.family}</Badge>
            {entry.gated && <Badge color="red">gated</Badge>}
            {methodBadge(entry.finetuningType)}
            {isAdapter(entry.finetuningType) && variantBadge(entry.loraVariant)}
          </SpaceBetween>
        </Header>
      }
    >
      <SpaceBetween size="m">
        {/* Per-model fixed facts from the catalog (not tunable). */}
        <ColumnLayout columns={4} variant="text-grid">
          <div><Box variant="awsui-key-label">Chat template</Box><Box>{model.template}</Box></div>
          <div><Box variant="awsui-key-label">LoRA targets</Box><Box>{model.loraTarget}</Box></div>
          <div><Box variant="awsui-key-label">Default cutoff</Box><Box>{model.defaultCutoffLen}</Box></div>
          <div><Box variant="awsui-key-label">Instance</Box><Box>{reasoningInstance}</Box></div>
        </ColumnLayout>

        {model.gated && (
          <Alert type="info" header="Gated model">
            <b>{entry.display}</b> is license-gated. Training needs a saved HF token (Settings) AND the
            token's account to have accepted the license:{" "}
            <Link external href={`https://huggingface.co/${model.hfModelId}`}>
              Request access to {model.hfModelId} ↗
            </Link>
          </Alert>
        )}

        {/* HP helpers (same two as the single-model form). */}
        <SpaceBetween direction="horizontal" size="xs" alignItems="center">
          <Box variant="awsui-key-label">Hyperparameters (pre-filled — edit any field)</Box>
          <Button iconName="suggestions" loading={suggesting} onClick={suggestDefaults}>
            Tune for my dataset
          </Button>
          <Button iconName="gen-ai" loading={advising} onClick={suggestSweep}>
            Propose a sweep to compare (AI)
          </Button>
        </SpaceBetween>
        {sweep && (
          <Alert type={sweep.source === "llm" ? "info" : "warning"} dismissible onDismiss={() => setSweep(null)}
            header={sweep.source === "llm" ? "AI-proposed sweep — Apply one to this model" : "Deterministic sweep (AI advisor unavailable)"}>
            <SpaceBetween size="xs">
              {sweep.note && <Box variant="small">{sweep.note}</Box>}
              {sweep.configs.map((c, i) => (
                <Box key={i}>
                  <SpaceBetween direction="horizontal" size="xs" alignItems="center">
                    <Badge color="blue">{c.label}</Badge>
                    <Box variant="small">rank {c.hyperparams.loraRank} · lr {c.hyperparams.learningRate} · epochs {c.hyperparams.numTrainEpochs}</Box>
                    <Button variant="inline-link" onClick={() => applySweepConfig(c.hyperparams)}>Apply</Button>
                  </SpaceBetween>
                  {c.reason && <Box variant="small" color="text-status-inactive">{c.reason}</Box>}
                </Box>
              ))}
            </SpaceBetween>
          </Alert>
        )}
        {rationale && (
          <Alert type="info" dismissible onDismiss={() => setRationale(null)}
            header="Tuned for your dataset — edit freely; the run decides the winner">
            <SpaceBetween size="xxs">
              {rationale.map((r) => (
                <Box key={r.field} variant="small"><b>{r.field}</b>: {r.value} — {r.reason}</Box>
              ))}
            </SpaceBetween>
          </Alert>
        )}

        {/* KTO per-class loss weights (KTO objective only). */}
        {wantStage === "kto" && (() => {
          // One-click pre-fill of the profiler's recommended λD/λU (the KTO analog
          // of the RLVR "Use this reward"). Shown only when the dataset carries a
          // recommendation that differs from what's already entered.
          const recCw = currentSplit?.recommendedChosenWeight;
          const recRw = currentSplit?.recommendedRejectedWeight;
          const hasRec = recCw != null && recRw != null;
          const recCwStr = hasRec ? String(recCw) : "";
          const recRwStr = hasRec ? String(recRw) : "";
          const alreadyApplied =
            hasRec && entry.ktoChosenWeight === recCwStr && entry.ktoRejectedWeight === recRwStr;
          return (
            <SpaceBetween size="m">
              {hasRec && !alreadyApplied && (
                <Alert type="info" header="Balance the class weights for this dataset">
                  Investigation found this KTO set is class-imbalanced and recommends{" "}
                  <b>λD={recCwStr}</b> / <b>λU={recRwStr}</b> to rebalance the loss.{" "}
                  <Button
                    variant="inline-link"
                    onClick={() =>
                      onChange({ ktoChosenWeight: recCwStr, ktoRejectedWeight: recRwStr })
                    }
                  >
                    Use recommended weights
                  </Button>
                </Alert>
              )}
              <FormField label="KTO loss weights" description="Per-class loss weights (λD/λU). 1.0 = balanced; raise the minority class for skewed sets (Investigate the dataset for exact values).">
                <ColumnLayout columns={2}>
                  {field("Desirable weight (λD)", "ktoChosenWeight")}
                  {field("Undesirable weight (λU)", "ktoRejectedWeight")}
                </ColumnLayout>
              </FormField>
            </SpaceBetween>
          );
        })()}

        {/* Preference-loss family (DPO-shaped datasets only). DPO/ORPO/SimPO race
            on the SAME preference data — only the loss differs. ORPO/SimPO are
            reference-free (cheaper) and LLaMA-Factory-only, so picking one forces
            the LF engine. SimPO exposes its target-margin γ. */}
        {wantStage === "dpo" && (
          <SpaceBetween size="m">
            <FormField
              label="Preference objective"
              description={PREF_LOSS_HELP[entry.prefLoss]}
            >
              <Select
                selectedOption={{ value: entry.prefLoss, label: PREF_LOSS_LABEL[entry.prefLoss] }}
                onChange={({ detail }) => {
                  const pl = (detail.selectedOption.value as PrefLoss) ?? "sigmoid";
                  // ORPO/SimPO are LLaMA-Factory-only — force the engine off serverless.
                  onChange(pl === "sigmoid"
                    ? { prefLoss: pl }
                    : { prefLoss: pl, engine: "llama_factory" });
                }}
                options={(["sigmoid", "orpo", "simpo"] as PrefLoss[]).map((v) => ({
                  value: v,
                  label: v === "sigmoid" ? `${PREF_LOSS_LABEL[v]} (recommended)` : PREF_LOSS_LABEL[v],
                  description: PREF_LOSS_HELP[v],
                }))}
              />
            </FormField>
            {entry.prefLoss === "simpo" && (
              <FormField label="SimPO target margin (γ)" description="Reward margin SimPO trains toward. Paper default 0.5; raise for a stronger chosen-vs-rejected gap, lower if it overshoots.">
                <ColumnLayout columns={3}>
                  {field("γ (simpo_gamma)", "simpoGamma")}
                </ColumnLayout>
              </FormField>
            )}
            {entry.prefLoss !== "sigmoid" && (
              <Alert type="info">
                {PREF_LOSS_LABEL[entry.prefLoss].replace(/ \(.*\)$/, "")} is <b>reference-free</b> (no
                frozen reference model) — cheaper than DPO and runs on the LLaMA-Factory engine. Race
                it against standard DPO on this dataset to see which wins; it's an unproven loss for
                this model until a run completes.
              </Alert>
            )}
          </SpaceBetween>
        )}

        {/* Efficiency knobs — orthogonal training accelerators (LLaMA-Factory only).
            NEFTune (embedding noise → quality), Liger kernel (speed/memory), and SFT
            sequence packing (throughput). Hidden on serverless (managed recipe) and
            for GRPO. Packing only for SFT (it concatenates samples). */}
        {!serverless && !isRlvr && !isRlaif && (
          <ExpandableSection headerText="Efficiency knobs (optional)" variant="footer">
            <SpaceBetween size="m">
              <FormField
                label="NEFTune noise (α)"
                description="Adds uniform noise to embeddings during training — often a free quality bump. 0 = off; the paper's sweet spot is ~5. Applies to any objective."
              >
                <ColumnLayout columns={3}>{field("α (neftune_noise_alpha)", "neftuneNoiseAlpha")}</ColumnLayout>
              </FormField>
              <Toggle
                checked={entry.enableLigerKernel}
                onChange={({ detail }) => onChange({ enableLigerKernel: detail.checked })}
              >
                Liger kernel — fused kernels for less memory + faster training (drop-in; verify-before-trust still applies)
              </Toggle>
              {wantStage === "sft" && (
                <Toggle
                  checked={entry.packing}
                  onChange={({ detail }) => onChange({ packing: detail.checked })}
                >
                  Sequence packing — concatenate short samples to one sequence (throughput win on short SFT data)
                </Toggle>
              )}
            </SpaceBetween>
          </ExpandableSection>
        )}

        {/* Engine picker — only when this model offers serverless + not GRPO. */}
        {!isRlvr && !isRlaif && (model.engines ?? ["llama_factory"]).includes("sagemaker_serverless") && (
          <FormField
            label="Engine"
            description={serverless
              ? "SageMaker Serverless: managed, pay per token, SFT/DPO with LoRA."
              : "LLaMA-Factory (default): self-hosted image — every method/objective."}
          >
            <Select
              selectedOption={serverless
                ? { value: "sagemaker_serverless", label: "SageMaker Serverless" }
                : { value: "llama_factory", label: "LLaMA-Factory (default)" }}
              onChange={({ detail }) => {
                const eng = (detail.selectedOption.value as Staged["engine"]) ?? "llama_factory";
                const toServerless = eng === "sagemaker_serverless";
                onChange({
                  engine: eng,
                  // Serverless is LoRA-only + can't honor a LoRA variant (managed
                  // recipe), so clear both when switching to it.
                  finetuningType: toServerless ? "lora" : entry.finetuningType,
                  loraVariant: toServerless ? "lora" : entry.loraVariant,
                  // ORPO/SimPO are LLaMA-Factory-only — reset to plain DPO when
                  // switching to serverless so the launch can't send a serverless+
                  // ORPO/SimPO combo the backend rejects (mirrors the variant reset).
                  prefLoss: toServerless ? "sigmoid" : entry.prefLoss,
                });
              }}
              options={[
                { value: "llama_factory", label: "LLaMA-Factory (default)" },
                { value: "sagemaker_serverless", label: "SageMaker Serverless", description: "Managed, no infra — SFT/DPO + LoRA" },
              ]}
            />
          </FormField>
        )}

        {/* RLVR reward source. */}
        {isRlvr && (() => {
          const deployed = rewardFns.filter((r) => r.status === "deployed");
          const PRESET = "__preset__";
          const selectedVal = entry.rewardFunctionId || PRESET;
          const opts = [
            { value: PRESET, label: "Preset scorer (built-in)", description: "gsm8k / prime_math" },
            ...deployed.map((r) => ({ value: r.id, label: `Custom: ${r.name}`, description: r.kind === "metric" ? `metric: ${r.metric}` : "custom Python reward" })),
          ];
          const recReward = currentSplit?.recommendedRewardMetric || "";
          const selectedRf = deployed.find((r) => r.id === entry.rewardFunctionId);
          const alreadyOnRecommended = !!recReward && selectedRf?.kind === "metric" && selectedRf?.metric === recReward;
          return (
            <SpaceBetween size="m">
              {recReward && !alreadyOnRecommended && (
                <Alert type="info" header="Reward on the metric you're ranked on">
                  Investigation recommends ranking on <b>{recReward}</b>.{" "}
                  <Button variant="inline-link" loading={applyingReward} onClick={() => applyRecommendedReward(recReward)}>Use this reward</Button>
                </Alert>
              )}
              <FormField label="Reward source" description="How RLVR scores answers against ground_truth.">
                <Select
                  selectedOption={opts.find((o) => o.value === selectedVal) ?? opts[0]}
                  onChange={({ detail }) => {
                    const v = detail.selectedOption.value!;
                    onChange(v === PRESET
                      ? { rewardFunctionId: "", presetRewardFunction: entry.presetRewardFunction || "gsm8k" }
                      : { rewardFunctionId: v, presetRewardFunction: "" });
                  }}
                  options={opts}
                />
              </FormField>
              {!entry.rewardFunctionId && (
                <FormField
                  label="Preset scorer"
                  description="Both reward a numeric answer (extract the final number, exact match). At launch this is auto-provisioned as a built-in reward function — it appears on the Reward functions page."
                >
                  <Select
                    selectedOption={{ value: entry.presetRewardFunction || "gsm8k", label: entry.presetRewardFunction || "gsm8k" }}
                    onChange={({ detail }) => onChange({ presetRewardFunction: detail.selectedOption.value ?? "gsm8k" })}
                    options={[
                      { value: "gsm8k", label: "gsm8k", description: "Grade-school math (numeric answer)" },
                      { value: "prime_math", label: "prime_math", description: "General math (numeric answer)" },
                    ]}
                  />
                </FormField>
              )}
              {grpoTooSmall && (
                <Alert type="error" header="Dataset too small for RLVR">
                  GRPO needs ≥{grpoMinRows} training examples; this dataset has {currentSplit?.trainRows ?? 0}.
                </Alert>
              )}
              {rewardDomainWarning && <Alert type="warning" header="This reward may not grade your dataset">{rewardDomainWarning}</Alert>}
            </SpaceBetween>
          );
        })()}

        {/* RLAIF reward prompt. */}
        {isRlaif && (() => {
          const prompts = rewardFns.filter((r) => r.kind === "reward_prompt" && r.status === "deployed");
          const opts = prompts.map((r) => ({ value: r.id, label: r.name, description: r.rewardModelId ? `judge: ${r.rewardModelId}` : "AI-judge reward prompt" }));
          return (
            <SpaceBetween size="m">
              {prompts.length === 0 ? (
                <Alert type="warning" header="No reward prompt yet">
                  RLAIF needs an AI-judge <b>reward prompt</b> — create one on the <Link href="#rewards">Reward functions</Link> page, then pick it here.
                </Alert>
              ) : (
                <FormField label="Reward prompt (AI judge)" description="The judge scores each response with this prompt (0..1).">
                  <Select
                    selectedOption={opts.find((o) => o.value === entry.rewardFunctionId) ?? null}
                    placeholder="Pick a reward prompt"
                    onChange={({ detail }) => onChange({ rewardFunctionId: detail.selectedOption.value! })}
                    options={opts}
                  />
                </FormField>
              )}
              {grpoTooSmall && (
                <Alert type="error" header="Dataset too small for RLAIF">
                  GRPO needs ≥{grpoMinRows} training examples; this dataset has {currentSplit?.trainRows ?? 0}.
                </Alert>
              )}
            </SpaceBetween>
          );
        })()}

        {/* Method picker (parameterization). */}
        <FormField label="Method" description={METHOD_HELP[entry.finetuningType]}>
          <Select
            selectedOption={{ value: entry.finetuningType, label: METHOD_LABEL[entry.finetuningType] }}
            onChange={({ detail }) => {
              const next = (detail.selectedOption.value as FinetuningType) ?? "lora";
              const wasFull = isFullWeight;
              const nowFull = next === "full" || next === "freeze";
              let learningRate = entry.learningRate;
              if (nowFull && !wasFull && (entry.learningRate === DEFAULTS.learningRate || entry.learningRate.trim() === "")) {
                learningRate = FULL_WEIGHT_LR;
              } else if (!nowFull && wasFull && entry.learningRate === FULL_WEIGHT_LR) {
                learningRate = DEFAULTS.learningRate;
              }
              // Keep the variant honest for the new method:
              //  - full/freeze have no adapter → clear to plain "lora" (picker hidden).
              //  - switching INTO qlora with DoRA/PiSSA selected → clear it, since
              //    those need full-precision weights (invalid on a 4-bit base). Else
              //    the disabled option would stay selected and launch a failing combo.
              // Adapter→adapter that stays valid keeps the chosen variant.
              const loraVariant = nowFull
                ? "lora"
                : variantAllowedForMethod(entry.loraVariant, next)
                  ? entry.loraVariant
                  : "lora";
              onChange({ finetuningType: next, learningRate, loraVariant });
            }}
            options={(["lora", "qlora", "full", "freeze"] as FinetuningType[]).map((mth) => {
              const allowed = (model.allowedMethods ?? ["lora", "qlora"]).includes(mth);
              const fullWeight = mth === "full" || mth === "freeze";
              const disabled = mth === "lora" ? false : serverless ? true : !allowed || (fullWeight && effectiveStage !== "sft");
              let description = METHOD_OPTION_HELP[mth];
              if (serverless && mth !== "lora") description = "Not available on the Serverless engine (LoRA only)";
              else if (fullWeight && !allowed) description = "Full/freeze need a small model (≤2B)";
              else if (fullWeight && effectiveStage !== "sft") description = "Full/freeze are SFT-only";
              return { value: mth, label: METHOD_LABEL[mth], disabled, description };
            })}
          />
        </FormField>
        {/* Verify-before-trust for non-LoRA methods. */}
        {entry.finetuningType !== "lora" && (() => {
          const name = METHOD_LABEL[entry.finetuningType].replace(/ \(.*\)$/, "");
          const st = methodStatusFor(model, entry.finetuningType);
          if (st === "verified") return <Alert type="success">{name} is verified for {entry.display} on image {model.imageTag}.</Alert>;
          if (st === "incompatible") return <Alert type="warning">{name} previously failed for {entry.display}. You can still launch (advisory) — smoke-test it from the Catalog to confirm.</Alert>;
          return <Alert type="info">{name} isn't verified yet for {entry.display} — launching also verifies it.</Alert>;
        })()}

        {/* LoRA variant picker — a modifier on the adapter methods (lora/qlora).
            Hidden for full/freeze (no adapter) and on serverless (LoRA-only, AWS
            controls the recipe — variants aren't exposable there). */}
        {!isFullWeight && !serverless && (
          <SpaceBetween size="xs">
            <FormField label="LoRA variant" description={VARIANT_HELP[entry.loraVariant]}>
              <Select
                selectedOption={{ value: entry.loraVariant, label: VARIANT_LABEL[entry.loraVariant] }}
                onChange={({ detail }) =>
                  onChange({ loraVariant: (detail.selectedOption.value as LoraVariant) ?? "lora" })
                }
                // DoRA is the recommended variant (best quality/effort), so it's
                // surfaced first (after plain) + labelled — mirrors how LoRA is the
                // recommended method. Order: plain, DoRA, then the situational rest.
                // DoRA + PiSSA are DISABLED on QLoRA: they need full-precision weights
                // a 4-bit base lacks, so PEFT rejects them at model-load (the picker
                // must not let you launch that invalid, billable combo).
                options={(["lora", "dora", "rslora", "pissa", "loraplus"] as LoraVariant[]).map((v) => {
                  const ok = variantAllowedForMethod(v, entry.finetuningType);
                  return {
                    value: v,
                    label: v === "dora" ? `${VARIANT_LABEL[v]} (recommended)` : VARIANT_LABEL[v],
                    description: ok
                      ? VARIANT_OPTION_HELP[v]
                      : "Not available on QLoRA — needs full-precision weights (use plain LoRA for this variant)",
                    disabled: !ok,
                  };
                })}
              />
            </FormField>
            {/* rsLoRA's sweet spot is high rank — nudge toward it when rank ≥ 32 and
                the user hasn't already picked it (purely advisory). */}
            {entry.loraVariant !== "rslora" && (num(entry.loraRank) ?? 8) >= 32 && (
              <Alert type="info">
                At LoRA rank {num(entry.loraRank)} (≥32), the <b>rsLoRA</b> variant rescales the
                adapter by 1/√rank, which often trains more stably at high ranks. Consider it for
                this run — it's merge-identical to LoRA, so there's no export risk.
              </Alert>
            )}
            {entry.loraVariant !== "lora" && (() => {
              // Variants are now verified PER-VARIANT (model, method, variant key),
              // so the badge is honest: a DoRA proof no longer rides plain LoRA's.
              const vst = variantStatusFor(model, entry.finetuningType, entry.loraVariant);
              const name = VARIANT_LABEL[entry.loraVariant];
              if (vst === "verified")
                return (
                  <Alert type="success">
                    {name} is verified for {entry.display} on image {model.imageTag} — a real run
                    of this exact variant completed.
                  </Alert>
                );
              if (vst === "incompatible")
                return (
                  <Alert type="warning">
                    {name} previously failed for {entry.display}. You can still launch (advisory) —
                    smoke-test it from the Catalog to re-confirm.
                  </Alert>
                );
              // Untested/pending: remind the variant itself is unproven on a launch,
              // since DoRA/PiSSA change training (+ DoRA the merge).
              return (
                <Alert type="info">
                  {name} isn't verified yet for {entry.display} — launching also verifies its
                  train→merge path. Compare it against a plain-LoRA run on the same dataset to see
                  if it wins.
                </Alert>
              );
            })()}
            {/* PiSSA does an SVD of the base at startup — set expectations that the
                job is slower to BEGIN and pays off most on smaller datasets. */}
            {entry.loraVariant === "pissa" && (
              <Alert type="warning">
                PiSSA computes an SVD of the base weights before training starts, so the job takes
                noticeably longer to <i>begin</i> (and merges via pissa_convert at export). It's
                best on smaller datasets, where the better initialization pays off most.
              </Alert>
            )}
            {entry.loraVariant === "loraplus" && (() => {
              const ratio = num(entry.loraplusLrRatio);
              const tooHigh = ratio != null && ratio > 32;
              const noop = ratio != null && ratio <= 1;
              return (
                <SpaceBetween size="xs">
                  <ColumnLayout columns={3}>
                    {field("LoRA+ LR ratio (λ)", "loraplusLrRatio",
                      "How much faster the B matrix trains than A. Paper default 16; range 1–128.")}
                  </ColumnLayout>
                  {tooHigh && (
                    <Alert type="warning">
                      A ratio above ~32 can destabilize training (the B matrix learns too fast).
                      16 is the paper default; raise it only if a lower value underfits.
                    </Alert>
                  )}
                  {noop && (
                    <Alert type="info">
                      A ratio of {ratio} means B and A train at the same rate — i.e. plain LoRA with
                      no LoRA+ benefit. Raise it (≈16) for the speedup, or switch the variant to None.
                    </Alert>
                  )}
                </SpaceBetween>
              );
            })()}
          </SpaceBetween>
        )}

        <ColumnLayout columns={3}>
          {entry.finetuningType !== "full" && entry.finetuningType !== "freeze" && (
            <>
              {field("LoRA rank", "loraRank")}
              {field("LoRA alpha", "loraAlpha", "Blank = auto (2 × rank)")}
            </>
          )}
          {entry.finetuningType === "freeze" && field("Trainable layers", "freezeTrainableLayers", "Top N transformer layers to train")}
          {field("Learning rate", "learningRate", isFullWeight ? "Full-weight default 1e-5 (low)" : undefined)}
          {field(entry.earlyStoppingEnabled ? "Max epochs" : "Epochs", "numTrainEpochs", entry.earlyStoppingEnabled ? "Ceiling — may stop earlier" : undefined)}
          {field("Per-device batch size", "perDeviceTrainBatchSize")}
          {field("Grad accumulation steps", "gradientAccumulationSteps")}
          {field("Cutoff len", "cutoffLen", `Blank = model default (${model.defaultCutoffLen})`)}
          {field("Save steps", "saveSteps")}
          {field("Max samples", "maxSamples", "Blank = all")}
        </ColumnLayout>

        {/* Early stopping (gated on a validation set). */}
        <Box variant="awsui-key-label">Early stopping</Box>
        {currentSplit && !currentSplit.hasVal ? (
          <Alert type="info">This dataset has no validation set — recreate it with a validation split (Step 1) to enable early stopping.</Alert>
        ) : !currentSplit ? (
          <Box variant="small" color="text-status-inactive">Pick a dataset to see early-stopping availability.</Box>
        ) : (
          <ColumnLayout columns={3}>
            <FormField label="Enable early stopping" description="Uses the validation set">
              <Toggle checked={entry.earlyStoppingEnabled} onChange={({ detail }) => onChange({ earlyStoppingEnabled: detail.checked })}>
                {entry.earlyStoppingEnabled ? "On" : "Off"}
              </Toggle>
            </FormField>
            {entry.earlyStoppingEnabled && field("Patience", "earlyStoppingPatience", "Evals w/o improvement before stop")}
          </ColumnLayout>
        )}

        {cardError && <Alert type="error" dismissible onDismiss={() => setCardError(null)}>{cardError}</Alert>}
      </SpaceBetween>
    </Container>
  );
}

export function FineTunePage({
  initialSplit,
  initialClone,
  onLaunched,
}: {
  initialSplit: CurrentSplit | null; // pre-selected when arriving from Datasets page
  // "Clone & edit": a prior run's launch config to pre-fill the builder (same
  // dataset + models/hp). The user then edits and submits a NEW run.
  initialClone?: RaceRequest | null;
  onLaunched: (raceId: string) => void;
}) {
  const [models, setModels] = useState<ModelSpec[]>([]);
  // Per-field hyperparameter bounds from the catalog (camelCase keys: loraRank,
  // learningRate, …) → client-side min/max + errorText so an out-of-range value
  // is caught in the form instead of only being silently clamped by the backend.
  const [bounds, setBounds] = useState<Record<string, { min: number; max: number }>>({});
  const [error, setError] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

  // The dataset this fine-tune run targets (chosen via the DatasetPicker).
  const [currentSplit, setCurrentSplit] = useState<CurrentSplit | null>(initialSplit);
  useEffect(() => {
    if (initialSplit) setCurrentSplit(initialSplit);
  }, [initialSplit]);

  // Multi-select: the models picked to fine-tune (each becomes a staged card).
  const [bulkSelected, setBulkSelected] = useState<readonly MultiselectProps.Option[]>([]);

  // The objective is gated by the dataset SHAPE: a preference dataset
  // (chosen/rejected) trains DPO; a KTO dataset (labelled good/bad) trains KTO; an
  // RLVR dataset (prompt + verifiable ground_truth) trains RLVR; a messages dataset
  // trains SFT. Keep the draft's stage in lockstep with the selected dataset so the
  // user can't stage an incompatible (objective, dataset) pair (backend rejects it too).
  const datasetShape = currentSplit?.shape ?? "sft";
  const wantStage: "sft" | "dpo" | "kto" | "rlvr" | "rlaif" =
    datasetShape === "preference" ? "dpo"
    : datasetShape === "kto" ? "kto"
    : datasetShape === "rlvr" ? "rlvr"
    : datasetShape === "rlaif" ? "rlaif"
    : "sft";
  // RLVR + RLAIF are serverless-only (the LLaMA-Factory image can't do GRPO). Such
  // a dataset FORCES the serverless engine + LoRA. RLVR defaults its reward to the
  // gsm8k preset; RLAIF needs an AI-judge reward PROMPT (rewardFunctionId) — there
  // is no preset, so it stays unset until the user picks one.
  const isRlvr = wantStage === "rlvr";
  const isRlaif = wantStage === "rlaif";
  const isGrpo = isRlvr || isRlaif;  // both force serverless + LoRA
  useEffect(() => {
    // Reconcile ALREADY-STAGED entries to the new objective so the cart badges
    // match what will actually launch (and so an SFT-staged entry can't submit
    // (llama_factory, rlvr) → a confusing backend 400). For RLVR/RLAIF force
    // serverless + LoRA; for non-GRPO clear the reward fields. (Early-stop is gated
    // backend-side on val-set presence, so it needs no reconcile here.) Only
    // rewrite when something actually changes.
    setBatch((prev) => {
      let changed = false;
      const next: Staged[] = prev.map((s): Staged => {
        if (isRlvr) {
          const hasReward = s.presetRewardFunction || s.rewardFunctionId;
          // GRPO needs a much lower LR than SFT — default to GRPO_LR unless the user
          // already set a non-SFT-default value (preserve a deliberate choice).
          const grpoLr = s.learningRate === DEFAULTS.learningRate ? GRPO_LR : s.learningRate;
          if (s.stage === "rlvr" && s.engine === "sagemaker_serverless"
              && s.finetuningType === "lora" && s.loraVariant === "lora" && hasReward
              && s.learningRate === grpoLr) return s;
          changed = true;
          // Force serverless + plain LoRA: the managed GRPO recipe can't honor a
          // LoRA variant, so clear it too (else the card badge shows a variant the
          // launch won't use — launch() also overrides it, but keep state honest).
          return {
            ...s, stage: "rlvr", engine: "sagemaker_serverless", finetuningType: "lora",
            loraVariant: "lora", learningRate: grpoLr,
            presetRewardFunction: hasReward ? s.presetRewardFunction : "gsm8k",
          };
        }
        if (isRlaif) {
          const grpoLr = s.learningRate === DEFAULTS.learningRate ? GRPO_LR : s.learningRate;
          if (s.stage === "rlaif" && s.engine === "sagemaker_serverless"
              && s.finetuningType === "lora" && s.loraVariant === "lora"
              && s.learningRate === grpoLr) return s;
          changed = true;
          return {
            ...s, stage: "rlaif", engine: "sagemaker_serverless", finetuningType: "lora",
            loraVariant: "lora", learningRate: grpoLr,
            presetRewardFunction: "",
          };
        }
        const needsStage = s.stage !== wantStage;
        const needsRewardClear = s.presetRewardFunction || s.rewardFunctionId;
        // full/freeze are SFT-only — if the dataset swap makes this a DPO/KTO run,
        // snap the method back to lora so we never submit an invalid (full, dpo)
        // payload (the backend rejects it; this avoids a confusing 400 after a swap).
        const needsMethodReset =
          wantStage !== "sft" && (s.finetuningType === "full" || s.finetuningType === "freeze");
        // Leaving GRPO for a supervised objective: restore the SFT LR default if the
        // card still carries the GRPO default (so a former-RLVR card doesn't keep its
        // low RL LR for SFT/DPO/KTO). Preserve any other (deliberate) value.
        const wasGrpo = s.stage === "rlvr" || s.stage === "rlaif";
        const needsLrRestore = wasGrpo && s.learningRate === GRPO_LR;
        if (!needsStage && !needsRewardClear && !needsMethodReset && !needsLrRestore) return s;
        changed = true;
        return {
          ...s, stage: wantStage, presetRewardFunction: "", rewardFunctionId: "",
          finetuningType: needsMethodReset ? "lora" : s.finetuningType,
          // When snapping full/freeze back to lora (SFT-only gate), start the
          // re-enabled variant picker from plain LoRA rather than a stale value.
          loraVariant: needsMethodReset ? "lora" : s.loraVariant,
          learningRate: needsLrRestore ? DEFAULTS.learningRate : s.learningRate,
        };
      });
      if (!changed) return prev;
      // A forced reconciliation (dataset/objective swap) can collapse two cards onto
      // one identity — e.g. LoRA+QLoRA of the same model both forced to RLVR-LoRA, or
      // two LoRA-variant duplicates both forced to plain LoRA. Those would become
      // truly identical backend entries (same entryKey, incl. variant), so keep the
      // first of each key. Distinct variants stay distinct (variant is in the key);
      // only genuine post-reconciliation duplicates are dropped.
      const seen = new Set<string>();
      const deduped = next.filter((s) => {
        const k = entryKey(s.modelId, s.finetuningType, s.engine, s.stage, s.loraVariant, s.prefLoss);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
      // Tell the user when this dropped a card — otherwise it's silent data loss
      // (their Duplicate'd variant vanishes and the Multiselect token stays checked).
      // Deferred out of the updater so we don't setState mid-render.
      const droppedCount = next.length - deduped.length;
      if (droppedCount > 0) {
        setTimeout(() => setError(
          `Switching to ${wantStage.toUpperCase()} forced ${droppedCount === 1 ? "a duplicate model card" : `${droppedCount} duplicate model cards`} ` +
          "to the same configuration (this objective doesn't support per-card LoRA variants/methods), so " +
          `${droppedCount === 1 ? "it was" : "they were"} merged to avoid an identical re-run. Re-add and re-configure if you still need ${droppedCount === 1 ? "it" : "them"}.`,
        ), 0);
      }
      return deduped;
    });
  }, [wantStage, isRlvr, isRlaif]);
  // The stage that will actually be launched (dataset-driven).
  const effectiveStage = wantStage;

  // The staged batch (one entry per selected model; each has its own card).
  const [batch, setBatch] = useState<Staged[]>([]);
  const [raceName, setRaceName] = useState("");
  // Spot is a launch-level cost choice (applies to all jobs in the batch).
  const [useSpot, setUseSpot] = useState(false);
  // Spot→on-demand fallback (opt-in, minutes). "" = off. Only meaningful with spot.
  const [spotFallbackMin, setSpotFallbackMin] = useState("");
  // Max wall-clock per training job, in HOURS (converted to seconds for the API).
  const [maxRunHours, setMaxRunHours] = useState("5");
  // Email-me-when-finished (launch-level). Prefill the address with the signed-in
  // user's Cognito email; default the toggle ON when we have an email to send to.
  const currentEmail = getCurrentUser()?.email || "";
  const [notify, setNotify] = useState<boolean>(!!currentEmail);
  const [notifyEmail, setNotifyEmail] = useState<string>(currentEmail);

  // Max-run-hours validity: gate the launch so display == submission.
  const maxHrsNum = num(maxRunHours);
  const maxHrsValid = maxHrsNum !== undefined && maxHrsNum >= 0.25 && maxHrsNum <= 24;

  // Notification emails: parse the comma-separated field → trimmed, non-empty list.
  // The backend re-normalizes/validates authoritatively; this is just for the UI
  // gate + payload. Valid when notify is off, or at least one address looks like one.
  const notifyEmailList = notifyEmail
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
  const notifyEmailValid =
    !notify || (notifyEmailList.length > 0 && notifyEmailList.every((e) => EMAIL_RE.test(e)));

  // GRPO (RLVR + RLAIF) dataset-size pre-check: global_batch_size floor is 128; the
  // recipe carves its own 10% val when there's no val file — so a GRPO split needs
  // ≥128 train rows with a val file, else ≥143 raw rows.
  const grpoMinRows = currentSplit?.hasVal ? 128 : 143;
  const grpoTooSmall =
    isGrpo && currentSplit != null && currentSplit.trainRows != null && currentSplit.trainRows < grpoMinRows;

  // YAML preview modal (folds in the old Render config page).
  const [previewFor, setPreviewFor] = useState<Staged | null>(null);
  const [previewYaml, setPreviewYaml] = useState<RenderResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const [hfTokenSet, setHfTokenSet] = useState(false);
  // Race-picker default filter: verified-by-default, opt-in to see untested.
  const [showUntested, setShowUntested] = useState(false);
  // Stepped wizard: Dataset → Models → Launch options → Review.
  const [activeStepIndex, setActiveStepIndex] = useState(0);
  // Deployed custom RLVR/RLAIF reward functions (for the per-card reward picker).
  const [rewardFns, setRewardFns] = useState<import("./api").RewardFunction[]>([]);

  // --- multi-model staging (each selected model is its own ModelConfigCard) ---

  // Reconcile the staged batch to EXACTLY the multiselect's picked model ids:
  // append newly-checked models (with default hyperparameters + the dataset's
  // objective/engine), drop unchecked ones, preserve order + any edits on kept
  // rows. The multiselect is the single source of truth for WHICH models; each
  // card owns its own hyperparameters.
  function syncBatchToSelection(ids: string[]) {
    const grpo = wantStage === "rlvr" || wantStage === "rlaif";
    const engine: Staged["engine"] = grpo ? "sagemaker_serverless" : "llama_factory";
    setBatch((prev) => {
      // The Multiselect picks which MODELS are staged; the cart may hold several
      // cards per model (Duplicate). So: keep EVERY card whose model is still
      // selected (preserving duplicates), drop cards whose model was unchecked,
      // and add ONE new card for a newly-checked model that has no card yet.
      const keep = prev.filter((s) => ids.includes(s.modelId));
      const present = new Set(keep.map((s) => s.modelId));
      const additions: Staged[] = [];
      for (const id of ids) {
        if (present.has(id)) continue;
        const m = models.find((x) => x.id === id);
        if (!m) continue;
        additions.push({
          iid: nextIid(),
          modelId: m.id,
          display: m.displayName,
          family: m.family,
          gated: m.gated,
          ...DEFAULTS,
          engine,
          stage: wantStage,
          earlyStoppingEnabled: !!currentSplit?.hasVal,
        });
      }
      // Preserve the kept cards in their current order (keeps duplicates together),
      // then append cards for any newly-checked models.
      return [...keep, ...additions];
    });
    setError(null);
  }

  // --- Auto-fill the race portfolio (the guided plan_race brain, one click) --------
  // Calls the research-backed planner for the CURRENT dataset + a job CEILING, then
  // stages the returned arms as cards. The promise is a PORTFOLIO to race (the
  // leaderboard picks the winner), never a single "best" — and it's an "up to N"
  // ceiling: the planner fills only as many arms as add signal (never pads).
  const [autofillBusy, setAutofillBusy] = useState(false);
  const [autofillCeiling, setAutofillCeiling] = useState<{ label: string; value: string }>(
    { label: "Balanced — up to 8", value: "8" });
  const [autofillNote, setAutofillNote] = useState<string | null>(null);
  // Auto-fill mirrors the guided planner, which supports SFT/DPO/KTO only.
  const autofillSupported = wantStage === "sft" || wantStage === "dpo" || wantStage === "kto";

  function armToStaged(a: AutofillArm): Staged {
    return {
      iid: nextIid(),
      modelId: a.modelId,
      display: a.displayName,
      family: a.family,
      gated: a.gated,
      ...DEFAULTS,
      // The planner's per-arm config overrides the DEFAULTS (objective-aware LR,
      // DoRA/full method, rank, freeze layers, ORPO pref-loss, etc.).
      engine: a.hp.engine,
      stage: a.hp.stage,
      finetuningType: a.hp.finetuningType,
      loraRank: a.hp.loraRank,
      loraAlpha: a.hp.loraAlpha,
      loraVariant: a.hp.loraVariant,
      loraplusLrRatio: a.hp.loraplusLrRatio,
      freezeTrainableLayers: a.hp.freezeTrainableLayers,
      learningRate: a.hp.learningRate,
      numTrainEpochs: a.hp.numTrainEpochs,
      perDeviceTrainBatchSize: a.hp.perDeviceTrainBatchSize,
      gradientAccumulationSteps: a.hp.gradientAccumulationSteps,
      cutoffLen: a.hp.cutoffLen,
      saveSteps: a.hp.saveSteps,
      maxSamples: a.hp.maxSamples,
      earlyStoppingEnabled: a.hp.earlyStoppingEnabled,
      earlyStoppingPatience: a.hp.earlyStoppingPatience,
      prefLoss: a.hp.prefLoss,
    };
  }

  async function handleAutofill() {
    if (!currentSplit?.splitId) {
      setError("Pick a dataset first, then auto-fill the race.");
      return;
    }
    setAutofillBusy(true);
    setAutofillNote(null);
    setError(null);
    try {
      const ceiling = Number(autofillCeiling.value) || 8;
      const res = await autofillRace({ splitId: currentSplit.splitId, ceiling });
      if (!res.supported || res.models.length === 0) {
        setError(res.reason || "Couldn't auto-fill a race for this dataset.");
        return;
      }
      const staged = res.models.map(armToStaged);
      setBatch(staged);
      // Keep the multiselect tokens in sync with the DISTINCT models staged (the batch
      // may hold several arms of one model, e.g. plain LoRA + DoRA — the picker shows
      // the model once).
      const distinct = Array.from(new Set(staged.map((s) => s.modelId)));
      setBulkSelected(
        distinct.map((id) => {
          const m = models.find((x) => x.id === id);
          return { label: m?.displayName ?? id, value: id };
        }),
      );
      // "up to N" transparency — say it plainly, don't pad.
      setAutofillNote(
        res.capped
          ? `Filled ${res.meaningfulCount} of up to ${res.ceiling} — that's all that adds anything for `
            + `this dataset; more would just be near-duplicates. Edit any card below, then review & launch.`
          : `Filled ${res.meaningfulCount} candidate${res.meaningfulCount === 1 ? "" : "s"} to race. `
            + `Edit any card below, then review & launch.`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Auto-fill failed.");
    } finally {
      setAutofillBusy(false);
    }
  }

  // Patch one staged card by its instance id (a card edited its hyperparameters).
  // Keyed by iid, not array index, so it's stable when the cart holds duplicates.
  function updateBatchByIid(iid: string, patch: Partial<Staged>) {
    setBatch((prev) => prev.map((s) => (s.iid === iid ? { ...s, ...patch } : s)));
  }

  // Remove one staged card by iid. Only UNTICK the Multiselect token when this was
  // the LAST card for that model — otherwise removing one duplicate would drop the
  // model entirely (and syncBatchToSelection would prune its siblings).
  function removeByIid(iid: string) {
    setBatch((prev) => {
      const removed = prev.find((s) => s.iid === iid);
      const next = prev.filter((s) => s.iid !== iid);
      if (removed && !next.some((s) => s.modelId === removed.modelId)) {
        setBulkSelected((sel) => sel.filter((o) => o.value !== removed.modelId));
      }
      return next;
    });
  }

  // Duplicate a staged card: clone its full config as a NEW instance (fresh iid),
  // inserted right after the original so the pair sits together. The clone starts
  // identical; the user then changes (typically) the LoRA variant so the two race
  // head-to-head as distinct entries (entryKey now includes the variant).
  function duplicateByIid(iid: string) {
    setBatch((prev) => {
      const i = prev.findIndex((s) => s.iid === iid);
      if (i < 0) return prev;
      const copy: Staged = { ...prev[i], iid: nextIid() };
      const next = [...prev];
      next.splice(i + 1, 0, copy);
      return next;
    });
  }
  useEffect(() => {
    getModels()
      .then((c) => {
        setModels(c.models);
        setBounds(c.bounds ?? {});
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    // Gated models are selectable when a token is EFFECTIVE for this user —
    // their own, or the platform's shared fallback (the app-level banner nags
    // fallback users to set their own).
    getHfTokenStatus()
      .then((s) => setHfTokenSet(s.isSet || !!s.usingSharedFallback))
      .catch(() => setHfTokenSet(false));
    // Custom reward functions (RLVR) — best-effort; the preset path works without.
    import("./api").then(({ listRewardFunctions }) =>
      listRewardFunctions()
        .then((r) => setRewardFns(r.rewardFunctions))
        .catch(() => setRewardFns([]))
    );
  }, []);


  // "Clone & edit": once the catalog has loaded, seed the batch cart + launch
  // options from a prior run's config. Runs once per distinct clone payload.
  const clonedRef = useRef<RaceRequest | null>(null);
  useEffect(() => {
    const c = initialClone;
    if (!c || models.length === 0 || clonedRef.current === c) return;
    clonedRef.current = c;
    const staged: Staged[] = (c.models ?? []).map((rm) => {
      const spec = models.find((mm) => mm.id === rm.modelId);
      return {
        iid: nextIid(),
        modelId: rm.modelId,
        display: spec?.displayName ?? rm.modelId,
        family: spec?.family ?? "",
        gated: spec?.gated ?? false,
        engine: (rm.engine as Staged["engine"]) ?? "llama_factory",
        stage: (rm.stage as Staged["stage"]) ?? "sft",
        presetRewardFunction: rm.presetRewardFunction ?? "",
        rewardFunctionId: rm.rewardFunctionId ?? "",
        rewardModelId: rm.rewardModelId ?? "",
        finetuningType: (rm.finetuningType as Staged["finetuningType"]) ?? "lora",
        loraRank: String(rm.loraRank ?? DEFAULTS.loraRank),
        loraAlpha: rm.loraAlpha == null ? "" : String(rm.loraAlpha),
        loraVariant: (rm.loraVariant as Staged["loraVariant"]) ?? "lora",
        loraplusLrRatio: String(rm.loraplusLrRatio ?? DEFAULTS.loraplusLrRatio),
        freezeTrainableLayers: String(rm.freezeTrainableLayers ?? DEFAULTS.freezeTrainableLayers),
        learningRate: String(rm.learningRate ?? DEFAULTS.learningRate),
        numTrainEpochs: String(rm.numTrainEpochs ?? DEFAULTS.numTrainEpochs),
        perDeviceTrainBatchSize: String(rm.perDeviceTrainBatchSize ?? DEFAULTS.perDeviceTrainBatchSize),
        gradientAccumulationSteps: String(rm.gradientAccumulationSteps ?? DEFAULTS.gradientAccumulationSteps),
        cutoffLen: rm.cutoffLen == null ? "" : String(rm.cutoffLen),
        saveSteps: String(rm.saveSteps ?? DEFAULTS.saveSteps),
        maxSamples: rm.maxSamples == null ? "" : String(rm.maxSamples),
        earlyStoppingEnabled: rm.earlyStoppingEnabled ?? false,
        earlyStoppingPatience: String(rm.earlyStoppingPatience ?? DEFAULTS.earlyStoppingPatience),
        ktoChosenWeight: String(rm.ktoChosenWeight ?? DEFAULTS.ktoChosenWeight),
        ktoRejectedWeight: String(rm.ktoRejectedWeight ?? DEFAULTS.ktoRejectedWeight),
        prefLoss: (rm.prefLoss as PrefLoss) ?? "sigmoid",
        simpoGamma: String(rm.simpoGamma ?? DEFAULTS.simpoGamma),
        neftuneNoiseAlpha: String(rm.neftuneNoiseAlpha ?? DEFAULTS.neftuneNoiseAlpha),
        enableLigerKernel: rm.enableLigerKernel ?? false,
        packing: rm.packing ?? false,
      };
    });
    setBatch(staged);
    // Seed the multiselect tokens from the cloned models. The Multiselect is a
    // CONTROLLED mirror of which MODELS are staged (one token per modelId, even if
    // several cards share it) — without this, a cloned run shows populated cards but
    // an EMPTY picker, and the next multiselect change would run
    // syncBatchToSelection([…]) and silently drop the cloned cards. Dedup by modelId
    // so duplicate cards don't make duplicate tokens.
    const seenTok = new Set<string>();
    setBulkSelected(
      staged
        .filter((s) => (seenTok.has(s.modelId) ? false : (seenTok.add(s.modelId), true)))
        .map((s) => ({ value: s.modelId, label: s.display })),
    );
    if (c.name) setRaceName(c.name);
    if (c.useSpot != null) setUseSpot(c.useSpot);
    if (c.maxRunSeconds) setMaxRunHours(String(Math.round((c.maxRunSeconds / 3600) * 100) / 100));
    if (c.spotFallbackMinutes != null) setSpotFallbackMin(String(c.spotFallbackMinutes));
    // Resolve the original dataset (same dataset, editable) so the cloned run is
    // ready to launch as-is; the user can swap it in the Dataset step. Best-effort:
    // if the split was archived/removed, leave the picker empty for the user.
    getDatasets(true)
      .then((ds) => {
        const d = ds.find((x) => x.splitId === c.splitId);
        if (d) {
          setCurrentSplit({
            splitId: d.splitId,
            name: d.name ?? undefined,
            trainRows: d.trainRows ?? 0,
            evalRows: d.evalRows ?? 0,
            hasVal: d.hasVal,
            shape: d.shape,
          } as CurrentSplit);
        }
      })
      .catch(() => {/* dataset resolution is best-effort */});
    // Jump to the Models step so the user sees the pre-filled cart immediately.
    setActiveStepIndex(1);
  }, [initialClone, models]);

  // Default early stopping ON for every staged model whenever the selected dataset
  // HAS a validation set (the eval-loss signal it needs) — strictly better in the
  // common case (stop at convergence + export the best checkpoint). Re-applies the
  // sensible default to all staged rows when the dataset (or its val-set presence)
  // changes; the user can still toggle any row off. No val set → forced off (the
  // launch guard strips it anyway).
  useEffect(() => {
    const hasVal = !!currentSplit?.hasVal;
    setBatch((prev) => {
      if (prev.every((s) => s.earlyStoppingEnabled === hasVal)) return prev;
      return prev.map((s) => (s.earlyStoppingEnabled === hasVal ? s : { ...s, earlyStoppingEnabled: hasVal }));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSplit?.splitId, currentSplit?.hasVal]);

  // A model is "verified" when its OWN image tier is proven (a real run on that
  // image completed). The picker defaults to verified-only — but this is a soft
  // filter, not a hard block: a transient failure shouldn't permanently hide a
  // good model, so "Show untested / unverified models" reveals the rest.
  const isVerified = (m: ModelSpec) =>
    m.verifications?.[m.imageTag]?.status === "verified";

  // Group models by family; flag gated + show per-model verification state.
  const options: SelectProps.OptionGroup[] = useMemo(() => {
    const visible = models.filter((m) => showUntested || isVerified(m));
    // Order provider groups by curated popularity (FAMILY_ORDER); any family not in
    // the list (e.g. a newly-onboarded custom model) sorts AFTER the known ones, in
    // first-seen catalog order. Models WITHIN a family keep their catalog order.
    const families = [...new Set(visible.map((m) => m.family))];
    const rank = (fam: string) => {
      const i = FAMILY_ORDER.indexOf(fam);
      return i === -1 ? FAMILY_ORDER.length : i;
    };
    families.sort((a, b) => rank(a) - rank(b)); // stable: ties keep first-seen order
    return families.map((fam) => ({
      label: fam,
      options: visible
        .filter((m) => m.family === fam)
        .map((m) => {
          const v = m.verifications?.[m.imageTag]?.status ?? "untested";
          const badge =
            v === "verified" ? " ✓" : v === "incompatible" ? " ⚠ incompatible" : " · untested";
          // Serverless marker: backend only includes "sagemaker_serverless" in
          // engines[] when the engine is enabled (the flag), so this label appears
          // ONLY when serverless is on AND the model supports it.
          const serverless = (m.engines ?? []).includes("sagemaker_serverless");
          // RLVR/RLAIF are serverless-only → a non-serverless model can't run them;
          // disable it in the picker (with a reason) for a GRPO-shaped dataset.
          const grpoBlocked = isGrpo && !serverless;
          return {
            value: m.id,
            label:
              m.displayName + (m.gated ? " (gated)" : "") + badge +
              (serverless ? " · ⚡serverless" : "") +
              (grpoBlocked ? ` · no serverless (${wantStage.toUpperCase()} n/a)` : ""),
            description:
              `${m.paramsB}B · template ${m.template} · ${m.suggestedInstance} · image ${m.imageTag}` +
              (serverless ? " · ⚡ SageMaker Serverless available" : ""),
            // Gated models need an HF token; enabled once one is set in Settings.
            // For RLVR/RLAIF, only serverless-mapped models are selectable.
            disabled: (m.gated && !hfTokenSet) || grpoBlocked,
          };
        }),
    }));
  }, [models, hfTokenSet, showUntested, isGrpo, wantStage]);

  // Stable per-entry key = model + method + engine (+ rlvr), so the SAME model
  // can be staged with both LoRA and QLoRA AND both engines AND SFT-vs-RLVR as
  // distinct entries. Mirrors the backend entry_key_for: method first
  // (back-compat), then engine, then the "rlvr" token (only for RLVR); default
  // (lora, llama_factory, sft) → bare model id.
  function entryKey(
    modelId: string,
    method: FinetuningType,
    engine: "llama_factory" | "sagemaker_serverless" = "llama_factory",
    stage: Staged["stage"] = "sft",
    variant: LoraVariant = "lora",
    prefLoss: PrefLoss = "sigmoid",
  ) {
    const parts = [modelId];
    if (method !== "lora") parts.push(method);
    if (engine !== "llama_factory") parts.push(engine);
    if (stage === "rlvr" || stage === "rlaif") parts.push(stage);
    // Variant (mirrors backend entry_key_for): a non-plain variant makes the
    // SAME (model, method) a distinct row, so DoRA vs rsLoRA of one model don't
    // collide. Plain "lora" adds nothing → pre-variant keys stay byte-identical.
    if (variant && variant !== "lora") parts.push(variant);
    // Preference-loss LAST (mirrors backend): DPO vs ORPO vs SimPO of the same
    // (model, method) are distinct rows. Plain DPO (sigmoid) adds nothing.
    if (stage === "dpo" && prefLoss && prefLoss !== "sigmoid") parts.push(`pref${prefLoss}`);
    return parts.join("::");
  }

  // Render + show the exact LLaMA-Factory train/export YAML for a staged model
  // (the inspection capability that used to be the separate Render config page).
  function openPreview(s: Staged) {
    if (!currentSplit) return;
    setPreviewFor(s);
    setPreviewYaml(null);
    setPreviewLoading(true);
    renderConfig({
      modelId: s.modelId,
      splitId: currentSplit.splitId,
      // RLVR/RLAIF run on the managed serverless recipe (no LLaMA-Factory YAML), so
      // the YAML preview falls back to SFT just to show the prompt formatting.
      stage: (s.stage === "rlvr" || s.stage === "rlaif") ? "sft" : s.stage,
      // Show the preference-loss family + SimPO margin in the YAML (DPO datasets).
      prefLoss: s.stage === "dpo" ? s.prefLoss : "sigmoid",
      simpoGamma: s.stage === "dpo" && s.prefLoss === "simpo" ? num(s.simpoGamma) : undefined,
      // Show efficiency knobs in the previewed YAML (non-default only; the backend
      // emits nothing for the no-op defaults). Packing only renders for SFT.
      neftuneNoiseAlpha: num(s.neftuneNoiseAlpha) || undefined,
      enableLigerKernel: s.enableLigerKernel || undefined,
      packing: s.stage === "sft" && s.packing ? true : undefined,
      finetuningType: s.finetuningType,
      loraRank: num(s.loraRank),
      loraAlpha: s.loraAlpha.trim() === "" ? null : num(s.loraAlpha),
      // Show the variant flags in the previewed YAML (adapter methods only).
      loraVariant: isAdapter(s.finetuningType) ? s.loraVariant : "lora",
      loraplusLrRatio: num(s.loraplusLrRatio),
      freezeTrainableLayers: num(s.freezeTrainableLayers),
      learningRate: num(s.learningRate),
      numTrainEpochs: num(s.numTrainEpochs),
      perDeviceTrainBatchSize: num(s.perDeviceTrainBatchSize),
      gradientAccumulationSteps: num(s.gradientAccumulationSteps),
      cutoffLen: s.cutoffLen.trim() === "" ? null : num(s.cutoffLen),
      saveSteps: num(s.saveSteps),
      maxSamples: s.maxSamples.trim() === "" ? null : num(s.maxSamples),
      earlyStoppingEnabled: s.earlyStoppingEnabled,
      earlyStoppingPatience: num(s.earlyStoppingPatience),
    })
      .then(setPreviewYaml)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setPreviewLoading(false));
  }

  function downloadYaml(filename: string, content: string) {
    const blob = new Blob([content], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function launch() {
    if (!currentSplit || batch.length === 0) return;
    if (!maxHrsValid) {
      setError("Max training time must be a number between 0.25 and 24 hours.");
      return;
    }
    if (!notifyEmailValid) {
      setError("Enter a valid notification email (or comma-separated addresses), or turn off email notification.");
      return;
    }
    if (grpoTooSmall) {
      setError(
        `${wantStage.toUpperCase()} (GRPO) needs at least ${grpoMinRows} training examples; this dataset has ` +
        `${currentSplit.trainRows}. Add more rows or pick a larger dataset.`,
      );
      return;
    }
    // RLAIF requires a reward PROMPT to be selected (no preset fallback).
    if (isRlaif && batch.some((s) => !s.rewardFunctionId)) {
      setError(
        "RLAIF needs a reward prompt (the AI judge). Pick one in the Reward source step — " +
        "create one on the Reward functions page if the list is empty.",
      );
      return;
    }
    // Duplicate guard: two cards that resolve to the SAME entry identity (model +
    // method + engine + stage + variant + pref_loss) would create colliding backend
    // entries (one would overwrite the other's state/leaderboard row). Caught here
    // with a clear message — the user duplicated a card but hasn't differentiated it.
    {
      const seen = new Set<string>();
      const dup = batch.find((s) => {
        const k = entryKey(s.modelId, s.finetuningType, s.engine, wantStage, s.loraVariant,
                           wantStage === "dpo" ? s.prefLoss : "sigmoid");
        if (seen.has(k)) return true;
        seen.add(k);
        return false;
      });
      if (dup) {
        const v = isAdapter(dup.finetuningType) && dup.loraVariant !== "lora"
          ? ` · ${VARIANT_LABEL[dup.loraVariant]}` : "";
        const pl = wantStage === "dpo" && dup.prefLoss !== "sigmoid"
          ? ` · ${PREF_LOSS_LABEL[dup.prefLoss].replace(/ \(.*\)$/, "")}` : "";
        setError(
          `Two staged cards are identical: ${dup.display} (${METHOD_LABEL[dup.finetuningType]}${v}${pl}). ` +
          "Change the method, LoRA variant, or preference objective on one (that's the point of Duplicate), or remove it.",
        );
        return;
      }
    }
    setLaunching(true);
    setError(null);
    try {
      const modelConfigs: RaceModelConfig[] = batch.map((s) => ({
        modelId: s.modelId,
        // Engine must agree with the dataset-derived objective: RLVR is
        // serverless-ONLY (the LLaMA-Factory engine rejects stage=rlvr → 400). A
        // model staged while an SFT/DPO/KTO dataset was selected carries
        // engine="llama_factory"; if the dataset is later swapped to RLVR we must
        // send the serverless engine, not the stale one. (The batch is also
        // reconciled on dataset change so the table badges match — see the effect.)
        engine: (wantStage === "rlvr" || wantStage === "rlaif") ? "sagemaker_serverless" : s.engine,
        // Objective is a property of the DATASET (shared across the race), so every
        // entry follows the dataset-derived wantStage (sft|dpo|kto|rlvr|rlaif). The
        // backend also rejects an entry whose objective mismatches the dataset shape.
        stage: wantStage,
        // RLVR reward (per-entry, so you can race gsm8k vs a custom reward): a custom
        // reward function id wins if set, else a preset (default gsm8k). RLAIF uses a
        // reward-PROMPT id (rewardFunctionId) + a judge model (rewardModelId), no
        // preset. All "" for non-GRPO objectives.
        presetRewardFunction:
          wantStage === "rlvr" && !s.rewardFunctionId ? (s.presetRewardFunction || "gsm8k") : "",
        rewardFunctionId:
          (wantStage === "rlvr" || wantStage === "rlaif") ? s.rewardFunctionId : "",
        rewardModelId: wantStage === "rlaif" ? s.rewardModelId : "",
        // DPO preference-loss family (sent only for DPO): sigmoid|orpo|simpo. ORPO/
        // SimPO are reference-free, LLaMA-Factory-only — so force plain DPO (sigmoid)
        // unless the resolved engine is llama_factory. Sent "sigmoid" for non-DPO so
        // a stale value can't leak (belt-and-suspenders behind the engine-switch reset).
        prefLoss:
          wantStage === "dpo" && s.engine === "llama_factory" ? s.prefLoss : "sigmoid",
        simpoGamma:
          wantStage === "dpo" && s.engine === "llama_factory" && s.prefLoss === "simpo"
            ? num(s.simpoGamma) : undefined,
        // KTO-only per-class loss weights (sent only for KTO; backend ignores them
        // for other objectives and emits nothing at the neutral 1.0 default).
        ktoChosenWeight: wantStage === "kto" ? num(s.ktoChosenWeight) : undefined,
        ktoRejectedWeight: wantStage === "kto" ? num(s.ktoRejectedWeight) : undefined,
        // Efficiency knobs (LLaMA-Factory engine only; orthogonal to objective).
        // Sent only when non-default + only on the LF engine (serverless/GRPO can't
        // honor them — the resolved engine is serverless for rlvr/rlaif or when the
        // card picked it); packing is SFT-only (backend rejects it on pref/KTO/RL).
        neftuneNoiseAlpha:
          !isGrpo && s.engine === "llama_factory" && num(s.neftuneNoiseAlpha)
            ? num(s.neftuneNoiseAlpha) : undefined,
        enableLigerKernel:
          !isGrpo && s.engine === "llama_factory" && s.enableLigerKernel ? true : undefined,
        packing:
          !isGrpo && s.engine === "llama_factory" && wantStage === "sft" && s.packing
            ? true : undefined,
        finetuningType: s.finetuningType,
        loraRank: num(s.loraRank),
        loraAlpha: s.loraAlpha.trim() === "" ? null : num(s.loraAlpha),
        // LoRA variant rides the LLaMA-Factory adapter path ONLY. Full/freeze have
        // no adapter, and serverless (incl. the GRPO objectives forced onto it) runs
        // AWS's managed recipe that doesn't expose variants — so send plain "lora"
        // for those, matching the picker (hidden on serverless/full/freeze). Also
        // force "lora" when the variant is invalid for the method (DoRA/PiSSA on
        // QLoRA need full-precision weights) — defense-in-depth behind the disabled
        // picker option + the backend guard, so a stale state can't launch a failing,
        // billable combo. Keeps the persisted/cloned config honest too.
        loraVariant:
          isAdapter(s.finetuningType) &&
          !(wantStage === "rlvr" || wantStage === "rlaif") &&
          s.engine !== "sagemaker_serverless" &&
          variantAllowedForMethod(s.loraVariant, s.finetuningType)
            ? s.loraVariant
            : "lora",
        loraplusLrRatio: num(s.loraplusLrRatio),
        freezeTrainableLayers: num(s.freezeTrainableLayers),
        learningRate: num(s.learningRate),
        numTrainEpochs: num(s.numTrainEpochs),
        perDeviceTrainBatchSize: num(s.perDeviceTrainBatchSize),
        gradientAccumulationSteps: num(s.gradientAccumulationSteps),
        cutoffLen: s.cutoffLen.trim() === "" ? null : num(s.cutoffLen),
        saveSteps: num(s.saveSteps),
        maxSamples: s.maxSamples.trim() === "" ? null : num(s.maxSamples),
        // ES only takes effect when the dataset has a val set (backend gates it
        // too); send the user's choice and let the backend be the source of truth.
        earlyStoppingEnabled: s.earlyStoppingEnabled,
        earlyStoppingPatience: num(s.earlyStoppingPatience),
      }));
      // Hours → seconds; clamp to a sane 0.25–24h window (a 0 or blank would
      // make SageMaker reject the job). Falls back to the 5h default.
      const hrs = Math.min(24, Math.max(0.25, num(maxRunHours) || 5));
      // Spot fallback only applies when spot is on; a blank/0/invalid value = off.
      const fbMin = useSpot ? num(spotFallbackMin) : undefined;
      const res = await launchRace({
        splitId: currentSplit.splitId,
        models: modelConfigs,
        name: raceName.trim() || undefined,
        useSpot,
        maxRunSeconds: Math.round(hrs * 3600),
        spotFallbackMinutes: fbMin && fbMin > 0 ? fbMin : undefined,
        notifyEmails: notify && notifyEmailList.length > 0 ? notifyEmailList : undefined,
      });
      setBatch([]);
      setRaceName("");
      setActiveStepIndex(0); // reset the wizard for the next submission
      onLaunched(res.raceId);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLaunching(false);
    }
  }

  // --- Wizard step content ---

  const datasetStep = (
    <SpaceBetween size="l">
      <DatasetPicker selected={currentSplit} onSelect={setCurrentSplit} />
      {!currentSplit && (
        <Alert type="info">
          Select an existing dataset or create one above to continue. Tip: use{" "}
          <b>Investigate this dataset</b> to see its task type + recommended eval metric before training.
        </Alert>
      )}
    </SpaceBetween>
  );

  const modelsStep = (
    <SpaceBetween size="l">
      <Container
        header={
          <Header
            variant="h2"
            description="Pick one or more models — each is staged with its own full hyperparameter form (pre-filled, editable). Add several to compare them in one run."
          >
            Select models
          </Header>
        }
      >
        <SpaceBetween size="m">
          {/* Multi-select picker — same options + Show-untested filter as before;
              full-width like the old single picker. Picking models stages each with
              its own ModelConfigCard below. */}
          <FormField
            label="Models"
            description={
              hfTokenSet
                ? "Choose the models to fine-tune. Each gets its own editable hyperparameters below. Gated models are enabled (HF token is set)."
                : "Choose the models to fine-tune. Each gets its own editable hyperparameters below. Gated models (Llama, Mistral, Gemma) are disabled — set an HF token in Settings to enable them."
            }
          >
            <Multiselect
              selectedOptions={bulkSelected}
              onChange={({ detail }) => {
                setBulkSelected(detail.selectedOptions);
                // Reconcile the staged batch to exactly the picked models: add the
                // newly-checked (with defaults), drop the unchecked. Keeps the cards
                // and the multiselect tokens in lockstep (single source of truth).
                syncBatchToSelection(detail.selectedOptions.map((o) => o.value!).filter(Boolean));
              }}
              options={options}
              placeholder="Choose models to fine-tune"
              filteringType="auto"
              tokenLimit={8}
            />
          </FormField>

          <Toggle checked={showUntested} onChange={({ detail }) => setShowUntested(detail.checked)}>
            Show untested / unverified models
            <Box variant="span" color="text-status-inactive" fontSize="body-s">
              {" "}— by default only models proven on their image (✓) are shown
            </Box>
          </Toggle>

          {/* Auto-fill: let the research-backed planner assemble the best race
              PORTFOLIO for this dataset + a job ceiling. Not a single "best" config —
              a diverse set for the leaderboard to decide, and an "up to N" ceiling it
              fills only as far as adds signal (never padding). SFT/DPO/KTO only. */}
          {autofillSupported && (
            <ExpandableSection
              headerText="Auto-optimize this race (recommended)"
              headerDescription="Let the platform pick a research-backed set of models + techniques to race for your data — you can still edit or remove any card below."
              defaultExpanded={batch.length === 0}
            >
              <SpaceBetween size="xs" direction="horizontal">
                <FormField label="How thorough?">
                  <Select
                    selectedOption={autofillCeiling}
                    onChange={({ detail }) =>
                      setAutofillCeiling(detail.selectedOption as { label: string; value: string })}
                    options={[
                      { label: "Quick — up to 4", value: "4" },
                      { label: "Balanced — up to 8", value: "8" },
                      { label: "Thorough — up to 16", value: "16" },
                    ]}
                    disabled={autofillBusy || !currentSplit?.splitId}
                  />
                </FormField>
                <FormField label=" ">
                  <Button
                    iconName="gen-ai"
                    loading={autofillBusy}
                    disabled={!currentSplit?.splitId}
                    onClick={handleAutofill}
                  >
                    {batch.length > 0 ? "Re-fill race" : "Auto-fill race"}
                  </Button>
                </FormField>
              </SpaceBetween>
              {!currentSplit?.splitId && (
                <Box variant="small" color="text-status-inactive">
                  Pick a dataset first (step 1), then auto-fill.
                </Box>
              )}
              {autofillNote && (
                <Box padding={{ top: "xs" }}>
                  <Alert type="success" dismissible onDismiss={() => setAutofillNote(null)}>
                    {autofillNote}
                  </Alert>
                </Box>
              )}
            </ExpandableSection>
          )}

          {error && (
            <Alert type="error" dismissible onDismiss={() => setError(null)}>
              {error}
            </Alert>
          )}
        </SpaceBetween>
      </Container>

      {batch.length === 0 ? (
        <Alert type="info">
          No models selected yet. Pick one or more above — each will appear here with its own
          editable hyperparameters.
        </Alert>
      ) : (
        <SpaceBetween size="l">
          <Box variant="h3">
            Configure each model{" "}
            <Box variant="small" display="inline">
              ({batch.length} staged — every field is pre-filled and editable)
            </Box>
          </Box>
          {batch.map((s) => {
            const m = models.find((x) => x.id === s.modelId);
            if (!m) return null;
            return (
              <ModelConfigCard
                key={s.iid}
                entry={s}
                model={m}
                bounds={bounds}
                currentSplit={currentSplit}
                effectiveStage={effectiveStage}
                wantStage={wantStage}
                isRlvr={isRlvr}
                isRlaif={isRlaif}
                grpoTooSmall={grpoTooSmall}
                grpoMinRows={grpoMinRows}
                rewardFns={rewardFns}
                onRewardFnsChange={setRewardFns}
                onChange={(patch) => updateBatchByIid(s.iid, patch)}
                onRemove={() => removeByIid(s.iid)}
                onDuplicate={() => duplicateByIid(s.iid)}
                onPreview={() => openPreview(s)}
              />
            );
          })}
        </SpaceBetween>
      )}
    </SpaceBetween>
  );

  const optionsStep = (
    <Container
      header={
        <Header variant="h2" description="Applies to every job in this submission.">
          Race name &amp; launch options
        </Header>
      }
    >
      <SpaceBetween size="m">
        <FormField label="Race name (optional)" description="A friendly label shown instead of the raw id on the Races page. Leave blank to use the generated id.">
          <Input
            value={raceName}
            placeholder={batch.length > 1 ? "e.g. qwen-vs-phi-epochs3" : "e.g. qwen3-1.7b-baseline"}
            onChange={({ detail }) => setRaceName(detail.value)}
          />
        </FormField>
        <FormField
          label="Use spot instances"
          description="Cheaper (~3× less) managed-spot capacity with checkpoint/resume. Interruptible, so wall-clock can vary."
        >
          <Toggle checked={useSpot} onChange={({ detail }) => setUseSpot(detail.checked)}>
            {useSpot ? "On — spot + checkpoint/resume" : "Off — on-demand"}
          </Toggle>
        </FormField>
        {useSpot && (
          <FormField
            label="Fall back to on-demand after (minutes)"
            description="If spot can't get capacity within this many minutes, automatically switch this run to on-demand (resuming from its checkpoint). Blank = wait for spot indefinitely. On-demand costs ~3× spot, so this trades cost for a wall-clock guarantee."
          >
            <Input
              type="number"
              value={spotFallbackMin}
              placeholder="e.g. 15 (blank = never)"
              step={5}
              onChange={({ detail }) => setSpotFallbackMin(detail.value)}
            />
          </FormField>
        )}
        <FormField
          label="Max training time (hours)"
          description="SageMaker stops a training job that runs longer than this — a safety cap on cost/wall-clock. Default 5h. Range 0.25–24h."
          errorText={
            maxRunHours.trim() !== "" && !maxHrsValid
              ? "Enter a number between 0.25 and 24 (hours)."
              : undefined
          }
        >
          <Input
            type="number"
            value={maxRunHours}
            placeholder="5"
            step={0.5}
            onChange={({ detail }) => setMaxRunHours(detail.value)}
          />
        </FormField>
        <FormField
          label="Email me when this run finishes"
          description="Get a summary email (per-model status + the winning model) once every job in this run reaches a final state."
        >
          <Toggle checked={notify} onChange={({ detail }) => setNotify(detail.checked)}>
            {notify ? "On — email a summary when done" : "Off"}
          </Toggle>
        </FormField>
        {notify && (
          <>
            <FormField
              label="Notification email"
              description="Comma-separate to notify more than one address."
              errorText={
                notifyEmail.trim() !== "" && !notifyEmailValid
                  ? "Enter a valid email address (or comma-separated addresses)."
                  : undefined
              }
            >
              <Input
                value={notifyEmail}
                placeholder="you@example.com"
                onChange={({ detail }) => setNotifyEmail(detail.value)}
              />
            </FormField>
            <Alert type="info">
              If an address hasn&rsquo;t been verified with SES yet, AWS will email it a
              one-time verification link &mdash; click that link to start receiving
              notifications. While this account is in the SES sandbox, only verified
              addresses receive mail.
            </Alert>
          </>
        )}
      </SpaceBetween>
    </Container>
  );

  const reviewRow = (label: string, value: React.ReactNode) => (
    <div>
      <Box variant="awsui-key-label">{label}</Box>
      <Box>{value}</Box>
    </div>
  );

  const reviewStep = (
    <SpaceBetween size="l">
      <Container
        header={
          <Header variant="h3" actions={<Button onClick={() => setActiveStepIndex(0)}>Edit</Button>}>
            Dataset
          </Header>
        }
      >
        {currentSplit ? (
          <ColumnLayout columns={3} variant="text-grid">
            {reviewRow("Name", currentSplit.name || currentSplit.splitId)}
            {reviewRow("Rows", `${currentSplit.trainRows} train / ${currentSplit.evalRows} test${currentSplit.hasVal ? ` / ${currentSplit.valRows} val` : ""}`)}
            {reviewRow("Validation set", currentSplit.hasVal ? "yes (early stopping available)" : "no")}
          </ColumnLayout>
        ) : (
          <Box color="text-status-error">No dataset selected.</Box>
        )}
      </Container>

      <Container
        header={
          <Header
            variant="h3"
            counter={`(${batch.length})`}
            actions={<Button onClick={() => setActiveStepIndex(1)}>Edit</Button>}
          >
            Models
          </Header>
        }
      >
        <Table
          variant="embedded"
          columnDefinitions={[
            { id: "model", header: "Model", cell: (s) => s.display },
            {
              id: "engine",
              header: "Engine",
              cell: (s) =>
                s.engine === "sagemaker_serverless" ? (
                  <Badge color="severity-neutral">Serverless</Badge>
                ) : (
                  "LLaMA-Factory"
                ),
            },
            {
              id: "objective",
              header: "Objective",
              cell: (s) =>
                s.stage === "dpo" ? <Badge color="green">DPO</Badge> :
                s.stage === "kto" ? <Badge color="green">KTO</Badge> :
                s.stage === "rlvr" ? (
                  <Badge color="severity-high">RLVR · {s.presetRewardFunction || "gsm8k"}</Badge>
                ) : s.stage === "rlaif" ? (
                  <Badge color="severity-high">RLAIF</Badge>
                ) : "SFT",
            },
            {
              id: "method",
              header: "Method",
              cell: (s) => {
                // Method badge + (for adapter methods) the LoRA variant badge, so a
                // DoRA/PiSSA/LoRA+ row reads distinctly from plain LoRA in the review.
                const vb = isAdapter(s.finetuningType) ? variantBadge(s.loraVariant) : null;
                return vb ? (
                  <SpaceBetween direction="horizontal" size="xxs">
                    {methodBadge(s.finetuningType)}
                    {vb}
                  </SpaceBetween>
                ) : (
                  methodBadge(s.finetuningType)
                );
              },
            },
            { id: "rank", header: "Rank", cell: (s) => s.loraRank },
            { id: "lr", header: "LR", cell: (s) => s.learningRate },
            { id: "epochs", header: "Epochs", cell: (s) => s.numTrainEpochs },
            { id: "cutoff", header: "Cutoff", cell: (s) => s.cutoffLen || "default" },
            {
              id: "es",
              header: "Early stop",
              cell: (s) =>
                s.earlyStoppingEnabled ? `patience ${s.earlyStoppingPatience}` : "off",
            },
          ]}
          items={batch}
          trackBy={(s) => s.iid}
          empty={<Box textAlign="center" padding="m">No models staged.</Box>}
        />
      </Container>

      <Container
        header={
          <Header variant="h3" actions={<Button onClick={() => setActiveStepIndex(2)}>Edit</Button>}>
            Launch options
          </Header>
        }
      >
        <ColumnLayout columns={3} variant="text-grid">
          {reviewRow("Race name", raceName.trim() || "(generated id)")}
          {reviewRow("Capacity", useSpot ? "spot (cheaper, interruptible)" : "on-demand")}
          {useSpot && num(spotFallbackMin) && num(spotFallbackMin)! > 0
            ? reviewRow("Spot fallback", `→ on-demand after ${spotFallbackMin} min waiting for capacity`)
            : null}
          {reviewRow("Max training time", `${maxRunHours} h`)}
          {reviewRow(
            "Email when finished",
            notify && notifyEmailList.length > 0 ? notifyEmailList.join(", ") : "off",
          )}
        </ColumnLayout>
      </Container>

      <Box>
        {batch.length <= 1
          ? "Launching will start a single fine-tune job."
          : `Launching will start a race comparing ${batch.length} models (parallel training, shared eval, ranked to a winner).`}{" "}
        After launch you'll jump to the Races page.
      </Box>
    </SpaceBetween>
  );

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Configure a fine-tune step by step: choose a dataset, stage one or more models, set launch options, then review and submit. One model runs a single fine-tune; two or more run together (parallel training, shared held-out test set, ranked to a winner)."
        >
          Submit fine-tune jobs
        </Header>
      }
    >
      <SpaceBetween size="l">
        {error && (
          <Alert type="error" header="Error" dismissible onDismiss={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Wizard
          activeStepIndex={activeStepIndex}
          onNavigate={({ detail }) => {
            // Block advancing past a step whose requirement isn't met.
            if (detail.requestedStepIndex > activeStepIndex) {
              if (activeStepIndex === 0 && !currentSplit) {
                setError("Select or create a dataset before continuing.");
                return;
              }
              if (activeStepIndex === 1 && batch.length === 0) {
                setError("Stage at least one model before continuing.");
                return;
              }
            }
            setError(null);
            setActiveStepIndex(detail.requestedStepIndex);
          }}
          onSubmit={launch}
          isLoadingNextStep={launching}
          submitButtonText={batch.length <= 1 ? "Launch fine-tune" : `Launch race (${batch.length} models)`}
          i18nStrings={{
            stepNumberLabel: (n) => `Step ${n}`,
            collapsedStepsLabel: (n, total) => `Step ${n} of ${total}`,
            cancelButton: "Cancel",
            previousButton: "Previous",
            nextButton: "Next",
            submitButton: batch.length <= 1 ? "Launch fine-tune" : `Launch race (${batch.length} models)`,
            optional: "optional",
          }}
          steps={[
            { title: "Dataset", description: "Choose or create the dataset to fine-tune on.", content: datasetStep },
            { title: "Models", description: "Stage one or more models with their hyperparameters.", content: modelsStep },
            { title: "Launch options", description: "Race name, spot, max training time.", content: optionsStep, isOptional: true },
            { title: "Review & submit", content: reviewStep },
          ]}
        />
      </SpaceBetween>

      <Modal
        visible={previewFor !== null}
        onDismiss={() => setPreviewFor(null)}
        size="large"
        header={previewFor ? `Generated config — ${previewFor.display}` : "Config"}
        footer={
          <Box float="right">
            <Button variant="link" onClick={() => setPreviewFor(null)}>
              Close
            </Button>
          </Box>
        }
      >
        {previewLoading || !previewYaml ? (
          <Box textAlign="center" padding="l">
            <Spinner /> Rendering LLaMA-Factory config…
          </Box>
        ) : (
          <Tabs
            tabs={[
              {
                id: "train",
                label: "train.yaml",
                content: (
                  <SpaceBetween size="s">
                    <Button
                      iconName="download"
                      onClick={() => downloadYaml("train.yaml", previewYaml.trainYaml)}
                    >
                      Download train.yaml
                    </Button>
                    <CodeView content={previewYaml.trainYaml} />
                  </SpaceBetween>
                ),
              },
              {
                id: "export",
                label: "export.yaml",
                content: (
                  <SpaceBetween size="s">
                    <Button
                      iconName="download"
                      onClick={() => downloadYaml("export.yaml", previewYaml.exportYaml)}
                    >
                      Download export.yaml
                    </Button>
                    <CodeView content={previewYaml.exportYaml} />
                  </SpaceBetween>
                ),
              },
            ]}
          />
        )}
      </Modal>
    </ContentLayout>
  );
}
