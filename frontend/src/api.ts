// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Typed client for the SLM platform backend. Mirrors the shapes returned by
// FastAPI in backend/app/main.py and validation.py.

export interface RowError {
  line: number;
  message: string;
}

export interface ChatTurn {
  role: string;
  content: string;
}

export interface PreviewRow {
  messages: ChatTurn[];
}

export interface ValidationReport {
  valid: boolean;
  totalLines: number;
  validRows: number;
  invalidRows: number;
  errors: RowError[];
  preview: PreviewRow[];
  roleCounts: Record<string, number>;
  filename?: string;
}

// API Gateway has a hard 10 MB request-body limit (not raisable). Dataset JSONL
// can exceed that, so we gzip files in the browser before upload (JSONL
// compresses ~10×) — the backend transparently decompresses via gzip magic
// bytes. CompressionStream is supported in all current evergreen browsers; if
// it's somehow unavailable we fall back to the raw file (works for < 10 MB).
async function gzipFile(file: File): Promise<Blob> {
  if (typeof CompressionStream === "undefined") return file;
  const stream = file.stream().pipeThrough(new CompressionStream("gzip"));
  const compressed = await new Response(stream).blob();
  return compressed;
}

// Append a (gzipped) file to a FormData under `field`, preserving the original
// filename so the server still sees e.g. train.jsonl.
async function appendGzipped(form: FormData, field: string, file: File): Promise<void> {
  const gz = await gzipFile(file);
  form.append(field, gz, file.name);
}

// Direct-to-S3 upload (bypasses API Gateway's hard 10 MB request cap — the file
// never traverses it). Get a presigned PUT URL, gzip the file, PUT it straight
// to S3, and return the uploadId the dataset endpoints read the object by.
// Handles arbitrarily large datasets (S3 single-PUT goes to 5 GB).
async function uploadToS3(file: File): Promise<string> {
  const res = await fetch("/api/datasets/upload-url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const { uploadId, url } = (await res.json()) as { uploadId: string; url: string };

  const gz = await gzipFile(file); // smaller transfer; server auto-gunzips
  const put = await fetch(url, { method: "PUT", body: gz });
  if (!put.ok) throw new Error(`S3 upload failed: ${put.status} ${put.statusText}`);
  return uploadId;
}

export async function validateDataset(file: File): Promise<ValidationReport> {
  const form = new FormData();
  await appendGzipped(form, "file", file);

  const res = await fetch("/api/datasets/validate", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(`Validation request failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as ValidationReport;
}

// --- Train/eval split + disjointness assertion ---

export interface SplitReport {
  ok: boolean;
  mode: "assert" | "auto";
  seed: number | null;
  evalRatio: number | null;
  splitId: string | null; // set when the split is persisted (ok === true)
  trainRows: number;
  evalRows: number;
  overlapCount: number;
  overlapExamples: PreviewRow[];
  promptCollisionCount: number;
  trainInvalidRows: number;
  evalInvalidRows: number;
  trainErrors?: RowError[];
  evalErrors?: RowError[];
  messages: string[];
  trainPreview: PreviewRow[];
  evalPreview: PreviewRow[];
  filename?: string;
  trainFilename?: string;
  evalFilename?: string;
  name?: string; // human-friendly dataset name
  // Optional validation set. Held-out eval set stays untouched.
  hasVal: boolean;
  valMode: "" | "file" | "carve";
  valRows: number;
  valRatio: number | null;
  valInvalidRows: number;
  valErrors?: RowError[];
  valPreview: PreviewRow[];
}

// Pull a readable error message out of FastAPI's {detail: ...} body when present.
async function errorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    /* fall through */
  }
  return `${res.status} ${res.statusText}`;
}

// Optional validation set: pass either a val File (3-file mode) OR a valRatio
// to auto-carve that fraction of train. The held-out eval set is never touched.
export async function assertSplit(
  train: File,
  evalFile: File,
  name = "",
  opts: { valFile?: File | null; valRatio?: number | null } = {}
): Promise<SplitReport> {
  // Upload each file directly to S3 (no API Gateway 10 MB limit), then send only
  // the small upload ids in the request.
  const [trainId, evalId] = await Promise.all([uploadToS3(train), uploadToS3(evalFile)]);
  const valId = opts.valFile ? await uploadToS3(opts.valFile) : null;

  const form = new FormData();
  form.append("train_upload_id", trainId);
  form.append("eval_upload_id", evalId);
  form.append("name", name);
  if (valId) form.append("val_upload_id", valId);
  if (opts.valRatio != null) form.append("val_ratio", String(opts.valRatio));

  const res = await fetch("/api/datasets/split/assert", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as SplitReport;
}

export async function autoSplit(
  file: File,
  evalRatio: number,
  seed: number,
  name = "",
  valRatio?: number | null,
  stratify = false
): Promise<SplitReport> {
  const uploadId = await uploadToS3(file);
  const form = new FormData();
  form.append("upload_id", uploadId);
  form.append("eval_ratio", String(evalRatio));
  form.append("seed", String(seed));
  form.append("name", name);
  if (valRatio != null) form.append("val_ratio", String(valRatio));
  if (stratify) form.append("stratify", "true");

  const res = await fetch("/api/datasets/split/auto", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as SplitReport;
}

// --- Preference (DPO) dataset upload ---

export interface PreferenceDatasetResult {
  splitId: string;
  name: string;
  shape: "preference";
  trainRows: number;
  valRows: number;
  testRows: number;
  totalPairs: number;
  preview: unknown[];
}

// Upload a JSONL of {messages|prompt, chosen, rejected} rows → a preference
// dataset usable with the DPO objective. The backend derives the shared eval set
// from the `chosen` responses so the leaderboard works unchanged.
export async function createPreferenceDataset(
  file: File,
  name: string,
  testRatio = 0.1,
  valRatio: number | null = 0.1,
  seed = 42
): Promise<PreferenceDatasetResult> {
  const uploadId = await uploadToS3(file);
  const form = new FormData();
  form.append("upload_id", uploadId);
  form.append("name", name);
  form.append("seed", String(seed));
  form.append("test_ratio", String(testRatio));
  if (valRatio != null) form.append("val_ratio", String(valRatio));

  const res = await fetch("/api/datasets/preference", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as PreferenceDatasetResult;
}

// --- KTO (binary-feedback) dataset upload ---

export interface KtoDatasetResult {
  splitId: string;
  name: string;
  shape: "kto";
  trainRows: number;
  valRows: number;
  testRows: number;
  totalRows: number;
  desirable: number;
  preview: unknown[];
}

// Upload a JSONL of {messages|prompt+completion, kto_tag|label} rows → a KTO
// dataset (binary good/bad feedback). The backend derives the shared eval set
// from the desirable completions so the leaderboard works unchanged.
export async function createKtoDataset(
  file: File,
  name: string,
  testRatio = 0.1,
  valRatio: number | null = 0.1,
  seed = 42
): Promise<KtoDatasetResult> {
  const uploadId = await uploadToS3(file);
  const form = new FormData();
  form.append("upload_id", uploadId);
  form.append("name", name);
  form.append("seed", String(seed));
  form.append("test_ratio", String(testRatio));
  if (valRatio != null) form.append("val_ratio", String(valRatio));

  const res = await fetch("/api/datasets/kto", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as KtoDatasetResult;
}

export interface RlvrDatasetResult {
  splitId: string;
  name: string;
  shape: "rlvr";
  trainRows: number;
  valRows: number;
  testRows: number;
  totalRows: number;
  preview: unknown[];
}

// Upload a JSONL of {messages|prompt, ground_truth} rows → an RLVR dataset
// (prompt + verifiable target). The held-out eval gold is derived from
// ground_truth so the leaderboard works unchanged. The preset reward function is
// chosen at launch, so the dataset itself stays reward-agnostic.
export async function createRlvrDataset(
  file: File,
  name: string,
  testRatio = 0.1,
  valRatio: number | null = 0.1,
  seed = 42
): Promise<RlvrDatasetResult> {
  const uploadId = await uploadToS3(file);
  const form = new FormData();
  form.append("upload_id", uploadId);
  form.append("name", name);
  form.append("seed", String(seed));
  form.append("test_ratio", String(testRatio));
  if (valRatio != null) form.append("val_ratio", String(valRatio));

  const res = await fetch("/api/datasets/rlvr", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RlvrDatasetResult;
}

export interface RlaifDatasetResult {
  splitId: string;
  name: string;
  shape: "rlaif";
  trainRows: number;
  valRows: number;
  testRows: number;
  totalRows: number;
  preview: unknown[];
}

// Create an RLAIF dataset from a JSONL of PROMPTS (no ground_truth — an AI judge
// scores generated responses against a reward prompt picked at launch).
export async function createRlaifDataset(
  file: File,
  name: string,
  testRatio = 0.1,
  valRatio: number | null = 0.1,
  seed = 42
): Promise<RlaifDatasetResult> {
  const uploadId = await uploadToS3(file);
  const form = new FormData();
  form.append("upload_id", uploadId);
  form.append("name", name);
  form.append("seed", String(seed));
  form.append("test_ratio", String(testRatio));
  if (valRatio != null) form.append("val_ratio", String(valRatio));

  const res = await fetch("/api/datasets/rlaif", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RlaifDatasetResult;
}

// --- Load a sample of a public Hugging Face dataset ---

export interface HFColumnMapping {
  userField: string;
  targetField: string;
  systemField: string | null;
  contextField: string | null;
  instruction: string;
}

export interface HFPreferenceMapping {
  chosenField: string;
  rejectedField: string;
  promptField: string | null;
  systemField: string | null;
  instruction: string;
}

export interface HFKtoMapping {
  completionField: string;
  labelField: string;
  promptField: string | null;
  systemField: string | null;
  instruction: string;
}

export interface HFRlvrMapping {
  promptField: string;
  groundTruthField: string;
  systemField: string | null;
  instruction: string;
}

export interface HFRlaifMapping {
  promptField: string;
  systemField: string | null;
  instruction: string;
}

export interface HFPreview {
  dataset: string;
  config: string;
  split: string;
  numRowsTotal: number | null;
  columnNames: string[];
  classLabels: Record<string, string[]>; // column -> class names (ClassLabel cols)
  suggestedMapping: HFColumnMapping | null;
  // Detected objective shape + the per-objective suggested column mappings.
  detectedShape: "sft" | "preference" | "kto";
  suggestedPreference: HFPreferenceMapping | null;
  preferencePreview: unknown[];
  suggestedKto: HFKtoMapping | null;
  ktoPreview: unknown[];
  // RLVR suggestion (prompt + verifiable ground_truth). Does NOT change
  // detectedShape — RLVR is a deliberate pick; this just pre-fills the columns.
  suggestedRlvr: HFRlvrMapping | null;
  rlvrPreview: unknown[];
  // RLAIF suggestion (prompt-only — the AI judge scores the response). Like RLVR
  // it does NOT change detectedShape; it just pre-fills the prompt column.
  suggestedRlaif: HFRlaifMapping | null;
  rlaifPreview: unknown[];
  sampleRows: Record<string, unknown>[];
  convertedPreview: PreviewRow[];
  splits: { config: string; split: string }[];
  // License advisory (compliance aid): the dataset's HF license + a severity
  // bucket + gated flag. Surfaced as a banner at import — informs, never blocks.
  licenseInfo?: {
    license: string | null; // HF license slug, e.g. "mit"; null if undeclared
    bucket: "permissive" | "restrictive" | "unknown";
    gated: boolean | string | null; // false | "auto" | "manual" | null
  };
}

export async function hfPreview(
  dataset: string,
  config?: string,
  split?: string
): Promise<HFPreview> {
  const res = await fetch("/api/datasets/hf/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, config: config ?? null, split: split ?? null }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as HFPreview;
}

export async function hfImport(req: {
  dataset: string;
  config: string;
  split: string;
  name: string;
  userField: string;
  targetField: string;
  systemField?: string | null;
  contextField?: string | null;
  instruction?: string;
  maxRows: number;
  seed: number;
  evalRatio: number;
  valRatio?: number | null;
  stratify?: boolean;
}): Promise<SplitReport & { hfStats?: Record<string, unknown> }> {
  const res = await fetch("/api/datasets/hf/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as SplitReport;
}

// Import a HF PREFERENCE dataset (chosen/rejected) → a DPO-ready preference
// dataset. The held-out eval is derived from the chosen responses server-side.
export async function hfImportPreference(req: {
  dataset: string;
  config: string;
  split: string;
  name: string;
  chosenField: string;
  rejectedField: string;
  promptField?: string | null;
  systemField?: string | null;
  instruction?: string;
  maxRows: number;
  seed: number;
  testRatio?: number;
  valRatio?: number | null;
}): Promise<PreferenceDatasetResult & { hfStats?: Record<string, unknown> }> {
  const res = await fetch("/api/datasets/hf/import-preference", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Import a HF dataset (completion + binary good/bad label) → a KTO dataset.
export async function hfImportKto(req: {
  dataset: string;
  config: string;
  split: string;
  name: string;
  completionField: string;
  labelField: string;
  promptField?: string | null;
  systemField?: string | null;
  instruction?: string;
  maxRows: number;
  seed: number;
  testRatio?: number;
  valRatio?: number | null;
}): Promise<KtoDatasetResult & { hfStats?: Record<string, unknown> }> {
  const res = await fetch("/api/datasets/hf/import-kto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Import a HF dataset (prompt + verifiable ground_truth, e.g. gsm8k) → an RLVR
// dataset. The preset reward function is chosen at launch, not import.
export async function hfImportRlvr(req: {
  dataset: string;
  config: string;
  split: string;
  name: string;
  promptField: string;
  groundTruthField: string;
  systemField?: string | null;
  instruction?: string;
  maxRows: number;
  seed: number;
  testRatio?: number;
  valRatio?: number | null;
}): Promise<RlvrDatasetResult & { hfStats?: Record<string, unknown> }> {
  const res = await fetch("/api/datasets/hf/import-rlvr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Import a HF dataset's PROMPT column → an RLAIF dataset (prompt-only; no answer).
// The AI-judge reward prompt is chosen at launch, not import — so only a prompt
// column is needed.
export async function hfImportRlaif(req: {
  dataset: string;
  config: string;
  split: string;
  name: string;
  promptField: string;
  systemField?: string | null;
  instruction?: string;
  maxRows: number;
  seed: number;
  testRatio?: number;
  valRatio?: number | null;
}): Promise<RlaifDatasetResult & { hfStats?: Record<string, unknown> }> {
  const res = await fetch("/api/datasets/hf/import-rlaif", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// --- Model catalog + LLaMA-Factory YAML render ---

// Per-image-tier verification status. A model proven on 0.9.4 is NOT
// auto-proven on 0.9.5 — verification is tied to the image tag, so an image
// bump resets everyone to "untested" until re-proved by a real run.
export type VerificationStatus =
  | "verified"
  | "incompatible"
  | "access_denied" // gated model: HF token lacks license approval (fixable on HF)
  | "untested"
  | "pending";

export interface VerificationRecord {
  status: VerificationStatus;
  jobName: string | null;
  reason: string | null;
  ts: string | null;
  seed?: boolean; // true = from the shipped known-good baseline, not a local run
}

export interface ModelSpec {
  id: string;
  displayName: string;
  hfModelId: string;
  provider: string; // friendly provider name derived from the HF org (Qwen→Alibaba…)
  template: string;
  tier: "tiny" | "small" | "mid" | "large";
  paramsB: number;
  defaultCutoffLen: number;
  suggestedInstance: string;
  family: string;
  gated: boolean;
  loraTarget: string;
  trustRemoteCode: boolean;
  notes: string;
  imageTag: string; // which Docker image tier this model runs on (stable|latest)
  allowedMethods: string[]; // parameterization hint, e.g. ["lora","qlora"]
  reasoning?: boolean; // emits a <think>/CoT prefix → eval uses a larger token budget
  engine?: string; // default engine for this model ("llama_factory")
  serverlessModelId?: string; // SageMaker Public Hub id; "" if no serverless equivalent
  engines?: string[]; // engines offered, e.g. ["llama_factory","sagemaker_serverless"]
  custom?: boolean;
  // Per-tier verification grid, e.g. { stable: {...}, latest: {...} }.
  verifications?: Record<string, VerificationRecord>;
}

export interface CatalogResponse {
  models: ModelSpec[];
  bounds: Record<string, { min: number; max: number }>;
  imageTiers?: Record<string, string>; // tier → ECR image tag (stable→0.9.4)
}

// Parameterization methods. lora/qlora = adapter; full/freeze = full-weight
// (standalone model, no adapter — llama_factory + SFT + small models only).
export type FinetuningType = "lora" | "qlora" | "full" | "freeze";

// LoRA adapter VARIANT — a modifier on top of the lora/qlora method (NOT a new
// method). All ride the same LoRA rank/alpha/target + merge/export path:
//   lora     = plain (default)        dora   = weight-decomposed LoRA
//   rslora   = rank-stabilized        pissa  = SVD-based init
//   loraplus = higher LR for B matrix (carries loraplusLrRatio)
// Ignored for full/freeze. Backend maps these to LLaMA-Factory flags (render.py).
export type LoraVariant = "lora" | "dora" | "rslora" | "pissa" | "loraplus";

// Preference-loss FAMILY for a DPO-shaped (chosen/rejected) dataset. The objective
// stays "dpo" — only the loss algorithm differs:
//   sigmoid = standard DPO (default)   orpo = ORPO (reference-free)
//   simpo   = SimPO (reference-free, carries simpoGamma)
// ORPO/SimPO are reference-free (cheaper — no frozen reference model). Surfaced in
// the UI as distinct "objectives" but map to stage=dpo + this pref_loss. LLaMA-
// Factory engine only. Ignored unless stage==="dpo".
export type PrefLoss = "sigmoid" | "orpo" | "simpo";

export interface RenderRequest {
  modelId: string;
  splitId: string;
  stage?: "sft" | "dpo" | "kto"; // objective; defaults to sft
  prefBeta?: number; // DPO preference-loss beta (ignored unless stage==="dpo")
  prefLoss?: PrefLoss; // DPO loss family: sigmoid|orpo|simpo (ignored unless stage==="dpo")
  simpoGamma?: number; // SimPO target reward margin γ (only when prefLoss==="simpo")
  ktoChosenWeight?: number; // KTO desirable-loss weight λD (ignored unless stage==="kto")
  ktoRejectedWeight?: number; // KTO undesirable-loss weight λU (ignored unless stage==="kto")
  neftuneNoiseAlpha?: number; // NEFTune embedding noise (0=off); llama_factory only
  enableLigerKernel?: boolean; // Liger fused kernels (speed/memory); llama_factory only
  packing?: boolean; // sequence packing (SFT-only, throughput); llama_factory only
  finetuningType?: FinetuningType; // parameterization; defaults to lora
  loraRank?: number;
  loraAlpha?: number | null;
  loraVariant?: LoraVariant; // adapter variant (rides lora/qlora); defaults to lora
  loraplusLrRatio?: number; // LoRA+ B/A learning-rate ratio (only when loraVariant==="loraplus")
  freezeTrainableLayers?: number; // freeze-only: # top layers to train
  learningRate?: number;
  numTrainEpochs?: number;
  perDeviceTrainBatchSize?: number;
  gradientAccumulationSteps?: number;
  cutoffLen?: number | null;
  saveSteps?: number;
  maxSamples?: number | null;
  earlyStoppingEnabled?: boolean;
  earlyStoppingPatience?: number;
}

export interface RenderResponse {
  model: ModelSpec;
  splitId: string;
  trainYaml: string;
  exportYaml: string;
}

// A split carried across pages (Split page → Render page).
export interface CurrentSplit {
  splitId: string;
  name?: string; // human-friendly dataset name
  trainRows: number;
  evalRows: number;
  hasVal?: boolean; // dataset carries a validation split (enables early stopping)
  valRows?: number;
  shape?: string; // "sft" (messages) | "preference" (DPO) — gates the objective
  // The verifiable RLVR reward metric the dataset investigation recommends (the
  // metric you're ranked on, mirrored as a reward). Carried into the FineTune
  // RLVR step so it can pre-offer that reward. null/absent when none applies.
  recommendedRewardMetric?: string | null;
  // KTO class-balance loss-weight recommendation (λ_D / λ_U) — carried into the
  // FineTune KTO step's one-click "Use recommended weights". Absent when balanced.
  recommendedChosenWeight?: number | null;
  recommendedRejectedWeight?: number | null;
  origin: string; // human-readable description of where it came from
}

export async function getModels(): Promise<CatalogResponse> {
  const res = await fetch("/api/models");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as CatalogResponse;
}

// Deterministic starting-point hyperparameters for a model + dataset, with a
// per-field rationale. The race remains the source of truth — these are an
// editable default, not a decision.
export interface RecommendRationale {
  field: string;
  value: string;
  reason: string;
}

export interface RecommendResponse {
  modelId: string;
  splitId: string | null;
  hyperparams: {
    finetuningType?: FinetuningType;
    loraRank: number;
    loraAlpha: number | null;
    loraVariant?: LoraVariant; // advisor may propose a DoRA arm
    loraplusLrRatio?: number;
    freezeTrainableLayers?: number;
    learningRate: number;
    numTrainEpochs: number;
    perDeviceTrainBatchSize: number;
    gradientAccumulationSteps: number;
    cutoffLen: number | null;
    saveSteps: number;
    maxSamples: number | null;
    earlyStoppingEnabled: boolean;
    earlyStoppingPatience: number;
  };
  rationale: RecommendRationale[];
}

export async function recommendConfig(req: {
  modelId: string;
  splitId?: string | null;
  trainRows?: number | null;
  hasVal?: boolean | null;
  finetuningType?: FinetuningType;
}): Promise<RecommendResponse> {
  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RecommendResponse;
}

// --- LLM advisor (Tier 3): propose a sweep of configs to race ---

export interface AdviseConfig {
  label: string;
  reason: string;
  hyperparams: RecommendResponse["hyperparams"];
}

export interface AdviseResponse {
  modelId: string;
  source: "llm" | "fallback";
  baseline: RecommendResponse["hyperparams"];
  configs: AdviseConfig[];
  note?: string;
}

export async function adviseConfig(req: {
  modelId: string;
  splitId?: string | null;
  trainRows?: number | null;
  hasVal?: boolean | null;
  finetuningType?: FinetuningType;
}): Promise<AdviseResponse> {
  const res = await fetch("/api/advise", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as AdviseResponse;
}

// --- Auto-fill a race portfolio (the guided plan_race brain, on the manual flow) ---

// Full staged-card hyperparameters for one auto-filled arm (camelCase strings, matching
// the manual Step-2 card fields). Blank string ⇒ use the card/engine default.
export interface AutofillArmHp {
  engine: "llama_factory" | "sagemaker_serverless";
  stage: "sft" | "dpo" | "kto" | "rlvr" | "rlaif";
  finetuningType: FinetuningType;
  loraRank: string;
  loraAlpha: string;
  loraVariant: LoraVariant;
  loraplusLrRatio: string;
  freezeTrainableLayers: string;
  learningRate: string;
  numTrainEpochs: string;
  perDeviceTrainBatchSize: string;
  gradientAccumulationSteps: string;
  cutoffLen: string;
  saveSteps: string;
  maxSamples: string;
  earlyStoppingEnabled: boolean;
  earlyStoppingPatience: string;
  prefLoss: PrefLoss;
}

export interface AutofillArm {
  modelId: string;
  displayName: string;
  label: string;
  family: string;
  gated: boolean;
  paramsB: number;
  role: string;
  hp: AutofillArmHp;
}

export interface AutofillResponse {
  supported: boolean;
  reason?: string;
  objective: string;
  rankMetric?: string;
  detectedTask?: string;
  models: AutofillArm[];
  // "up to N" transparency: how many arms were filled vs the ceiling the user picked,
  // and whether the planner deliberately stopped short (no more useful arms).
  meaningfulCount: number;
  ceiling: number;
  capped: boolean;
  gatesApplied?: string[];
}

export async function autofillRace(req: { splitId: string; ceiling: number }): Promise<AutofillResponse> {
  const res = await fetch("/api/autofill-race", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as AutofillResponse;
}

// --- Auto-onboard a model from a Hugging Face id (Tier 1) ---

export interface ProbedModel {
  id: string;
  hfModelId: string;
  displayName: string;
  family: string;
  template: string | null;
  templateMatch: string;
  templateKnown: boolean;
  knownTemplates: string[];
  paramsB: number;
  defaultCutoffLen: number;
  suggestedInstance: string;
  gated: boolean;
  imageTag?: string;  // optional tier pin (discovery flow sets it to the newest tier)
  serverlessModelId?: string;  // set when onboarding a serverless-customizable candidate
  architectures: string[];
  modelType: string;
}

export async function probeModel(repo: string): Promise<ProbedModel> {
  const res = await fetch("/api/models/probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as ProbedModel;
}

export async function addCustomModel(m: ProbedModel): Promise<{ ok: boolean; id: string }> {
  const res = await fetch("/api/models/custom", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: m.id,
      displayName: m.displayName,
      hfModelId: m.hfModelId,
      template: m.template,
      family: m.family,
      paramsB: m.paramsB,
      defaultCutoffLen: m.defaultCutoffLen,
      suggestedInstance: m.suggestedInstance,
      gated: m.gated,
      // The discovery flow pins a model to a specific image tier (e.g. "latest");
      // forward it so the backend doesn't silently default it to "stable".
      ...(m.imageTag ? { imageTag: m.imageTag } : {}),
      // Onboarding a serverless-customizable candidate carries its Public Hub id
      // so the serverless engine is enabled on the new row immediately.
      ...(m.serverlessModelId ? { serverlessModelId: m.serverlessModelId } : {}),
    }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export async function deleteCustomModel(id: string): Promise<void> {
  const res = await fetch(`/api/models/custom/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorDetail(res));
}

export interface SmokeTestResponse {
  jobName: string;
  modelId: string;
  splitId: string;
  imageTag: string;
  method: FinetuningType;
  loraVariant?: LoraVariant;
  imageUri: string;
}

// Smoke-test a model on an image tier with a given parameterization. `method`
// defaults to lora (back-compat); pass "qlora" to prove the 4-bit path, which is
// verified independently from LoRA. `loraVariant` defaults to plain "lora"; pass
// "dora"/"rslora"/"pissa"/"loraplus" to prove a non-plain adapter variant, which
// is verified independently (DoRA/PiSSA change training + the merge). `engine`
// defaults to llama_factory; pass "sagemaker_serverless" to prove the serverless
// surface (distinct from images).
export async function smokeTestModel(
  id: string,
  imageTag?: string,
  method: FinetuningType = "lora",
  engine: "llama_factory" | "sagemaker_serverless" = "llama_factory",
  loraVariant: LoraVariant = "lora",
): Promise<SmokeTestResponse> {
  const params = new URLSearchParams();
  if (imageTag) params.set("image_tag", imageTag);
  if (method !== "lora") params.set("method", method);
  if (engine !== "llama_factory") params.set("engine", engine);
  if (loraVariant !== "lora") params.set("lora_variant", loraVariant);
  const qs = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(`/api/models/${encodeURIComponent(id)}/smoke-test${qs}`, { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Poll a smoke-test/verification job; on a terminal state the backend records
// the (model, image_tag, method) result (verified | incompatible) and returns it.
export interface VerifyPollResponse {
  jobName: string;
  jobStatus: string;
  modelId: string;
  imageTag: string;
  method: FinetuningType;
  engine?: "llama_factory" | "sagemaker_serverless";
  loraVariant?: LoraVariant;
  verification: VerificationRecord;
}

export async function pollVerification(
  modelId: string,
  imageTag: string,
  jobName: string,
  method: FinetuningType = "lora",
  engine: "llama_factory" | "sagemaker_serverless" = "llama_factory",
  loraVariant: LoraVariant = "lora",
): Promise<VerifyPollResponse> {
  const params = new URLSearchParams();
  if (method !== "lora") params.set("method", method);
  if (engine !== "llama_factory") params.set("engine", engine);
  if (loraVariant !== "lora") params.set("lora_variant", loraVariant);
  const qs = params.toString() ? `?${params.toString()}` : "";
  const res = await fetch(
    `/api/verify/${encodeURIComponent(modelId)}/${encodeURIComponent(imageTag)}/${encodeURIComponent(jobName)}${qs}`,
  );
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export async function backfillVerifications(): Promise<{ scanned: number; promoted: number }> {
  const res = await fetch("/api/verifications/backfill", { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Self-healing triage: classify a model's failure and recommend a newer image
// tier if the model needs a newer stack.
export interface DiagnoseResult {
  modelId: string;
  currentTier: string;
  classification: {
    category: string;
    needsNewerStack: boolean;
    explanation: string;
    source: string;
  };
  recommendedTier: string | null;
  imageReady: boolean | null;
  action: "none" | "no_image_change" | "already_newest" | "smoke_test" | "build_then_smoke_test";
}

export async function diagnoseModel(modelId: string, reason?: string): Promise<DiagnoseResult> {
  const qs = reason ? `?reason=${encodeURIComponent(reason)}` : "";
  const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/diagnose${qs}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export async function buildImage(imageTag: string): Promise<{ buildId: string; project: string; status: string }> {
  const res = await fetch(`/api/images/${encodeURIComponent(imageTag)}/build`, { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// --- ECR image-tier management (Images page) ---
export interface ImageTierStatus {
  tier: string;
  tag: string;
  imageUri: string;
  existsInEcr: boolean;
  pushedAt: string | null;
  sizeMB: number | null;
  buildProject: string;
  lastBuild: { status: string | null; id: string | null; startTime: string | null } | null;
  verifiedModels: number;
}

export async function listImages(): Promise<{ tiers: ImageTierStatus[] }> {
  const res = await fetch("/api/images");
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export async function resetTierVerifications(tier: string): Promise<{ tier: string; cleared: number }> {
  const res = await fetch(`/api/images/${encodeURIComponent(tier)}/reset-verifications`, { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// --- LLaMA-Factory release tracking + new-model discovery ---
export interface UpdateCheck {
  upstream: string[];
  builtTags: string[];
  newest: string | null;
  haveNewest: boolean;
  newReleases: string[]; // newer than our newest built tag
}

export async function checkImageUpdates(): Promise<UpdateCheck> {
  const res = await fetch("/api/images/check-updates");
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export async function buildRelease(
  lfTag: string,
  vllmVersion?: string,
): Promise<{ buildId: string; project: string; tier: string; tag: string; tiers: Record<string, string> }> {
  const res = await fetch("/api/images/build-release", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lfTag, ...(vllmVersion ? { vllmVersion } : {}) }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export interface NewModelSuggestion {
  architecture: string;
  repos: string[];
}
export interface NewModelsResult {
  newTag: string;
  baseTag: string | null;
  transformers?: string;
  newArchitectures: string[];
  newTemplates: string[];
  suggestions: NewModelSuggestion[];
  note: string;
}

export async function newModelsForImage(imageTag: string, baseTag?: string): Promise<NewModelsResult> {
  const qs = baseTag ? `?base_tag=${encodeURIComponent(baseTag)}` : "";
  const res = await fetch(`/api/images/${encodeURIComponent(imageTag)}/new-models${qs}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// --- Serverless model discovery ("Find serverless-customizable models") ---
// Mirrors the image-diff "Find new models" flow, but for the serverless engine:
// queries the live SageMaker Public Hub for customizable models and classifies
// them against our catalog. Suggestions only — applying a tag is a separate call.
export interface ServerlessUntaggedMatch {
  id: string; // catalog model id we can light up the serverless engine for
  displayName: string;
  hfModelId: string;
  hubId: string; // SageMaker Public Hub id to tag it with
  recipes: string[]; // sft_lora / dpo_lora / rlvr_lora / rlaif_lora
  gated: boolean;
}
export interface ServerlessStaleTag {
  id: string;
  displayName: string;
  hubId: string; // the tag the hub no longer lists as customizable
}
export interface ServerlessNewCandidate {
  hubId: string; // customizable on the hub, no catalog row yet
  hf: string;
  recipes: string[];
  onboardable: boolean; // true → one-click Add + enable serverless (has HF repo, not VLM)
  reason: string; // why not onboardable (Nova has no HF repo / VLM unsupported), else ""
}
// A flat row in the browsable "all serverless-customizable models" list.
// state: enabled = already in catalog + tagged; addable = in catalog, untagged
// (one-click Enable); onboardable = not in catalog, has HF repo (Add + enable);
// unavailable = no HF repo / VLM (awareness only, `reason` explains).
export interface ServerlessModelRow {
  hubId: string;
  displayName: string;
  id: string; // catalog id (empty for not-yet-onboarded)
  hfModelId: string;
  recipes: string[];
  gated: boolean;
  state: "enabled" | "addable" | "onboardable" | "unavailable";
  verified: boolean; // serverless smoke-test passed (only meaningful when enabled)
  reason: string;
}
export interface ServerlessCandidates {
  customizableCount: number;
  allModels: ServerlessModelRow[];
  untaggedMatches: ServerlessUntaggedMatch[];
  staleTags: ServerlessStaleTag[];
  newCandidates: ServerlessNewCandidate[];
  note: string;
}

export async function serverlessCandidates(): Promise<ServerlessCandidates> {
  const res = await fetch("/api/models/serverless-candidates");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as ServerlessCandidates;
}

// Apply a discovered serverless tag (catalogId → Public Hub id) as a runtime
// overlay — the serverless engine becomes available for the model, no redeploy.
// Blank hubId clears a previously-applied overlay tag. The model stays UNVERIFIED
// on serverless until a smoke test proves it.
export async function setServerlessTag(modelId: string, hubId: string): Promise<{ ok: boolean }> {
  const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/serverless-tag`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hubId }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export async function renderConfig(req: RenderRequest): Promise<RenderResponse> {
  const res = await fetch("/api/render", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RenderResponse;
}

// --- SageMaker training launch + status ---

export interface TrainRequest extends RenderRequest {
  instanceType?: string;
  maxRunSeconds?: number;
}

export interface LaunchResponse {
  jobName: string;
  instanceType: string;
  imageUri: string;
  datasetS3: string;
  configS3: string;
  outputS3: string;
  region: string;
}

export interface JobStatus {
  jobName: string;
  status: "InProgress" | "Completed" | "Failed" | "Stopped" | string;
  secondaryStatus: string | null;
  failureReason: string | null;
  instanceType: string;
  billableTimeSeconds: number | null;
  trainingStartTime: string | null;
  trainingEndTime: string | null;
  modelArtifacts: string | null;
  region: string;
}

export async function launchTraining(req: TrainRequest): Promise<LaunchResponse> {
  const res = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as LaunchResponse;
}

export async function getJobStatus(jobName: string): Promise<JobStatus> {
  const res = await fetch(`/api/train/${encodeURIComponent(jobName)}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as JobStatus;
}

// Training curves: loss/lr/epoch time series scraped into CloudWatch. X is
// minutes elapsed since training started. Series are empty for jobs too new to
// have logged, or launched before metric scraping was added.
export interface CurvePoint {
  x: number; // minutes since training start
  y: number;
}

export interface TrainingCurves {
  jobName: string;
  status: string;
  startTime: string | null;
  series: {
    trainLoss: CurvePoint[];
    evalLoss: CurvePoint[];
    learningRate: CurvePoint[];
    epoch: CurvePoint[];
    gradNorm: CurvePoint[];
  };
}

export async function getTrainingCurves(jobName: string): Promise<TrainingCurves> {
  const res = await fetch(`/api/train/${encodeURIComponent(jobName)}/curves`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as TrainingCurves;
}

// --- Offline batch eval ---

export interface CompletedJob {
  jobName: string;
  modelArtifacts: string;
  instanceType: string;
  creationTime: string;
}

export interface PerClass {
  total: number;
  correct: number;
  accuracy: number;
}

export interface EvalMetrics {
  count: number;
  exact_match: number;
  exact_match_extracted?: number;
  normalized_match: number;
  contains_gold?: number;
  contains_gold_extracted?: number;
  token_f1: number;
  rouge_l?: number;
  char_f1?: number;
  length_ratio?: number;
  scaffold_rate?: number;
  // task-aware (null when no row of that task type)
  json_valid?: number | null;
  json_structural: number | null;
  json_key_recall?: number | null;
  json_rows?: number;
  numeric_match?: number | null;
  numeric_rows?: number;
  label_accuracy?: number | null;
  label_rows?: number;
  // DPO only: fraction of held-out prompts where the model's answer is closer to
  // the chosen than the rejected response — the preference-native metric (what DPO
  // optimizes). null for SFT/KTO/RLVR (no rejected ref).
  chosen_win_rate?: number | null;
  chosen_win_rate_rows?: number;
  // RLAIF only: the final AI-judge reward (0..1, reference-free) the entry is
  // ranked by — present INSTEAD of the gold-overlap metrics, which don't apply to
  // a prompt-only held-out set. reward_source flags held-out vs training reward.
  reward_mean?: number | null;
  reward_source?: "held_out" | "training";
  task_mix?: Record<string, number>;
  json_applicable_rows?: number;
  per_class_accuracy: Record<string, PerClass>;
  decoding: {
    backend: string;
    temperature: number;
    top_p: number;
    max_new_tokens: number;
    seed: number;
  };
  timing?: {
    gen_seconds: number;
    output_tokens: number;
    tokens_per_sec: number | null;
    p50_latency_ms: number | null;
  };
}

export interface EvalRequest {
  sourceJobName: string;
  splitId: string;
  backend?: string;
  temperature?: number;
  topP?: number;
  maxNewTokens?: number;
  seed?: number;
  instanceType?: string;
  maxRunSeconds?: number;
}

export interface EvalLaunchResponse {
  jobName: string;
  sourceJob: string;
  sourceModel: string;
  instanceType: string;
  datasetS3: string;
  outputS3: string;
  region: string;
}

export interface EvalStatus extends JobStatus {
  metrics: EvalMetrics | null;
  metricsError?: string;
}

export async function getCompletedJobs(): Promise<CompletedJob[]> {
  const res = await fetch("/api/jobs/completed");
  if (!res.ok) throw new Error(await errorDetail(res));
  return ((await res.json()) as { jobs: CompletedJob[] }).jobs;
}

export async function launchEval(req: EvalRequest): Promise<EvalLaunchResponse> {
  const res = await fetch("/api/eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as EvalLaunchResponse;
}

export async function getEvalStatus(jobName: string): Promise<EvalStatus> {
  const res = await fetch(`/api/eval/${encodeURIComponent(jobName)}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as EvalStatus;
}

// --- Leaderboard + Sonnet baseline ---

export interface LeaderboardRow {
  evalJob: string;
  sourceJob: string | null;
  model: string;
  splitId?: string;
  count: number | null;
  exactMatch: number | null;
  normalizedMatch: number | null;
  containsGold?: number | null;
  tokenF1: number | null;
  rougeL?: number | null;
  charF1?: number | null;
  lengthRatio?: number | null;
  jsonStructural: number | null;
  rewardMean?: number | null; // RLAIF judge reward (reference-free); null otherwise
  backend: string | null;
  tokensPerSec: number | null;
  p50LatencyMs: number | null;
  p90LatencyMs?: number | null; // tail latency (None on older eval runs)
  p99LatencyMs?: number | null;
  trainCostUsd: number | null;
  trainInstance: string | null;
  trainSpot?: boolean;
  trainServerless?: boolean; // true for SageMaker-serverless-trained rows (engine column)
  trainCostIsEstimate?: boolean;
  projectedServeCostPer1k: number | null;
  evalInstance: string;
  isBaseline: boolean;
  creationTime: string;
  // Baseline-only fields (merged client-side for the Sonnet row):
  apiCostPer1k?: number | null;
}

export interface EvalSplit {
  splitId: string;
  name: string | null;
  evalJobs: number;
  latest: string;
  recommendedRankMetric?: string | null; // from dataset investigation, if run
}

export async function getEvalSplits(): Promise<EvalSplit[]> {
  const res = await fetch("/api/leaderboard/splits");
  if (!res.ok) throw new Error(await errorDetail(res));
  return ((await res.json()) as { splits: EvalSplit[] }).splits;
}

// Metrics shape returned by the cached baseline (same as SonnetBaselineResult.metrics).
export type SonnetBaselineMetrics = SonnetBaselineResult["metrics"];

export interface LeaderboardResponse {
  rows: LeaderboardRow[];
  baseline: SonnetBaselineMetrics | null; // back-compat (Sonnet 4.5)
  baselines?: SonnetBaselineMetrics[]; // all frontier-model baselines run on this split
}

export async function getLeaderboard(splitId: string): Promise<LeaderboardResponse> {
  const res = await fetch(`/api/leaderboard?split_id=${encodeURIComponent(splitId)}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as LeaderboardResponse;
}

// Selectable frontier baseline models (Haiku/Sonnet/Opus).
export interface BaselineModel {
  key: string;
  provider: string;
  label: string;
  modelId: string;
  inPer1k: number;
  outPer1k: number;
}

export async function getBaselineModels(): Promise<{ default: string; models: BaselineModel[] }> {
  const res = await fetch("/api/baseline/models");
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export interface SonnetBaselineResult {
  splitId: string;
  metrics: {
    count: number;
    exact_match: number;
    normalized_match: number;
    contains_gold?: number;
    token_f1: number;
    rouge_l?: number;
    char_f1?: number;
    length_ratio?: number;
    json_structural: number | null;
    per_class_accuracy: Record<string, PerClass>;
    baseline: {
      key?: string;
      label?: string;
      model: string;
      inputTokens: number;
      outputTokens: number;
      apiCostUsd: number;
      apiCostPer1kRows: number | null;
    };
  };
}

export type BaselineMetrics = SonnetBaselineResult["metrics"];

// Start the baseline. Hosted: returns {status:"running"} (runs in a worker
// Lambda to avoid the 29s API GW limit) — poll getBaselineStatus. Local dev:
// returns {status:"done", metrics} inline.
export interface BaselineStartResult {
  splitId: string;
  status: "running" | "done";
  metrics?: BaselineMetrics;
}

export async function runSonnetBaseline(
  splitId: string,
  maxNewTokens = 256,
  temperature = 0.0,
  baselineKey = "sonnet-4-5"
): Promise<BaselineStartResult> {
  const res = await fetch("/api/baseline/sonnet", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ splitId, maxNewTokens, temperature, baselineKey }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as BaselineStartResult;
}

export interface BaselineStatus {
  splitId: string;
  status: "none" | "running" | "done" | "failed";
  detail?: string;
  metrics: BaselineMetrics | null;
}

export async function getBaselineStatus(
  splitId: string,
  baselineKey = "sonnet-4-5"
): Promise<BaselineStatus> {
  const res = await fetch(
    `/api/baseline/sonnet/${encodeURIComponent(splitId)}?baseline_key=${encodeURIComponent(baselineKey)}`
  );
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as BaselineStatus;
}

// --- Failure-triage + results-interpreter agents (AgentCore) ---

export interface TriageResult {
  summary: string;
  rootCause: string;
  fix: string;
  retryable: boolean;
  configChanges?: Record<string, string>;
  confidence?: string;
  _context?: { classification?: { category?: string; explanation?: string }; instance?: string };
}

export async function triageRaceEntry(raceId: string, modelId: string): Promise<TriageResult> {
  const res = await fetch(
    `/api/race/${encodeURIComponent(raceId)}/triage?model_id=${encodeURIComponent(modelId)}`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }
  );
  if (!res.ok) throw new Error(await errorDetail(res));
  const out = await res.json();
  if (out.result) return out.result as TriageResult; // inline (local dev)
  if (out.summary) return out as TriageResult; // legacy inline shape
  // running → poll the worker until done (gathers logs + agent call > 29s possible)
  for (let i = 0; i < 30; i++) {
    await sleep(3000);
    const st = await fetch(
      `/api/race/${encodeURIComponent(raceId)}/triage/${encodeURIComponent(modelId)}`
    ).then((r) => r.json());
    if (st.status === "failed") throw new Error(st.detail || "triage failed");
    if (st.result) return st.result as TriageResult;
  }
  throw new Error("timed out waiting for diagnosis");
}

export interface InterpretResult {
  recommendation: string;
  reasoning: string;
  runnerUp?: string;
  vsBaseline?: string;
  caveats?: string[];
  error?: string;
  // Set on a PERSISTED result so the UI can show "last run": when (UTC ISO) +
  // the priorities it was run with. Absent on a fresh inline result.
  ranAt?: string;
  priorities?: string;
}

// Fetch the LAST persisted recommendation for a split (no new run). Returns null
// when none has been run yet. Lets the leaderboard show "what you ran last time"
// (recommendation + when + priorities) on load, surviving reloads.
export async function loadLastInterpret(splitId: string): Promise<InterpretResult | null> {
  const res = await fetch(`/api/leaderboard/interpret/${encodeURIComponent(splitId)}`);
  if (!res.ok) return null;
  const st = await res.json();
  return (st.result as InterpretResult | null) ?? null;
}

export async function interpretLeaderboard(
  splitId: string,
  priorities: string
): Promise<InterpretResult> {
  const res = await fetch("/api/leaderboard/interpret", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ splitId, priorities }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const out = await res.json();
  if (out.result) return out.result as InterpretResult; // inline (local dev)
  if (out.recommendation || out.error) return out as InterpretResult; // legacy inline
  for (let i = 0; i < 30; i++) {
    await sleep(3000);
    const st = await fetch(`/api/leaderboard/interpret/${encodeURIComponent(splitId)}`).then((r) =>
      r.json()
    );
    if (st.status === "failed") throw new Error(st.detail || "interpret failed");
    if (st.result) return st.result as InterpretResult;
  }
  throw new Error("timed out waiting for recommendation");
}

// --- LLM-as-judge (quality scoring of an eval job's predictions) ---

export interface JudgeResult {
  judgeScore: number; // overall mean 1-5
  dimensions?: Record<string, number>; // per-dimension means (faithfulness/format/…)
  judgedRows: number;
  distribution: Record<string, number>; // "1".."5" -> count
  model: string;
  apiCostUsd: number;
  samples: {
    gold: string;
    pred: string;
    score: number;
    reason: string;
    dimensions?: Record<string, number>;
  }[];
}

export interface JudgeStatus {
  evalJob: string;
  status: "none" | "running" | "done" | "failed";
  detail?: string;
  result: JudgeResult | null;
}

// Start (or fetch cached) judge for an eval job. Hosted: {status:running} → poll.
export async function startJudge(
  evalJob: string
): Promise<{ evalJob: string; status: "running" | "done"; result?: JudgeResult }> {
  const res = await fetch(`/api/judge/${encodeURIComponent(evalJob)}`, { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

export async function getJudge(evalJob: string): Promise<JudgeStatus> {
  const res = await fetch(`/api/judge/${encodeURIComponent(evalJob)}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as JudgeStatus;
}

// --- Fine-tuning race ---

export interface RaceEntry {
  model_id: string;
  // Stable per-entry id (model_id, or model_id::method) — distinguishes a
  // same-model LoRA vs QLoRA entry. Used for retry/diagnose/export keying.
  entryKey?: string;
  model_display: string;
  instance_type: string;
  hp: Record<string, unknown>;
  state: "pending" | "launching" | "training" | "eval_pending" | "evaluating" | "done" | "failed";
  train_job: string | null;
  eval_job: string | null;
  metrics: EvalMetrics | null;
  error: string | null;
  rankScore: number | null;
  isWinner?: boolean;
  // Base-model control: the UNTRAINED model's metrics on the same test set, so
  // the UI can show base → fine-tuned lift (how much fine-tuning helped). Runs in
  // parallel with training; null until its eval completes ({} if it failed).
  base_eval_job?: string | null;
  base_metrics?: EvalMetrics | null;
  // True when a failed TRAINING run can RESUME from its last checkpoint (rather
  // than retrain from scratch) — LLaMA-Factory entries with a recorded checkpoint
  // prefix. Drives the "Resume from checkpoint" button.
  canResume?: boolean;
  // True once this entry was auto-converted spot → on-demand by the capacity
  // fallback. Surfaced as a badge so the cost/behavior change is visible.
  spot_fell_back?: boolean;
}

export interface RaceResult {
  raceId: string;
  name?: string;
  splitId: string;
  useSpot?: boolean;
  rankMetric?: string;
  rankMetrics?: string[];
  entries: RaceEntry[];
}

// Per-model config for a fine-tune launch (1 model = single job, 2+ = a race).
export interface RaceModelConfig {
  modelId: string;
  instanceType?: string | null;
  engine?: "llama_factory" | "sagemaker_serverless"; // training engine; defaults to llama_factory
  stage?: "sft" | "dpo" | "kto" | "rlvr" | "rlaif"; // objective; defaults to sft (rlvr/rlaif = serverless-only)
  prefBeta?: number; // DPO preference-loss beta (ignored unless stage==="dpo")
  prefLoss?: PrefLoss; // DPO loss family: sigmoid|orpo|simpo (ignored unless stage==="dpo")
  simpoGamma?: number; // SimPO target reward margin γ (only when prefLoss==="simpo")
  ktoChosenWeight?: number; // KTO desirable-loss weight λD (ignored unless stage==="kto")
  ktoRejectedWeight?: number; // KTO undesirable-loss weight λU (ignored unless stage==="kto")
  neftuneNoiseAlpha?: number; // NEFTune embedding noise (0=off); llama_factory only
  enableLigerKernel?: boolean; // Liger fused kernels (speed/memory); llama_factory only
  packing?: boolean; // sequence packing (SFT-only, throughput); llama_factory only
  presetRewardFunction?: string; // RLVR-only: gsm8k|prime_math (ignored otherwise)
  rewardFunctionId?: string; // RLVR: a custom reward id; RLAIF: a reward-prompt id
  rewardModelId?: string; // RLAIF-only: the judge model id ("" = recipe default)
  finetuningType?: FinetuningType; // parameterization; defaults to lora
  loraRank?: number;
  loraAlpha?: number | null;
  loraVariant?: LoraVariant; // adapter variant (rides lora/qlora); defaults to lora
  loraplusLrRatio?: number; // LoRA+ B/A learning-rate ratio (only when loraVariant==="loraplus")
  freezeTrainableLayers?: number; // freeze-only: # top layers to train
  learningRate?: number;
  numTrainEpochs?: number;
  perDeviceTrainBatchSize?: number;
  gradientAccumulationSteps?: number;
  cutoffLen?: number | null;
  saveSteps?: number;
  maxSamples?: number | null;
  earlyStoppingEnabled?: boolean;
  earlyStoppingPatience?: number;
}

export interface RaceRequest {
  splitId: string;
  models: RaceModelConfig[];
  name?: string;
  useSpot?: boolean; // race-level cost toggle: spot capacity + checkpoint/resume
  maxRunSeconds?: number; // max wall-clock per training job (SageMaker stop cap)
  // Spot→on-demand fallback (opt-in, minutes): auto-convert a spot job stuck
  // waiting for capacity past this to on-demand. Omit/0 = off. Only with useSpot.
  spotFallbackMinutes?: number;
  evalMaxNewTokens?: number;
  evalTemperature?: number;
  // Email addresses to notify when the whole run finishes. Omit = no notification.
  notifyEmails?: string[];
}

export async function launchRace(req: RaceRequest): Promise<RaceResult> {
  const res = await fetch("/api/race", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RaceResult;
}

// Reconstructed launch config for a run, used by "Clone & edit": the Fine-Tune
// builder pre-fills with this (same dataset + models/hp), the user edits, then
// submits a NEW run. Same shape as RaceRequest (+ a suggested clone name).
export async function cloneRaceConfig(raceId: string): Promise<RaceRequest> {
  const res = await fetch(`/api/race/${encodeURIComponent(raceId)}/clone-config`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RaceRequest;
}

export async function getRace(raceId: string, rankMetric?: string): Promise<RaceResult> {
  const q = rankMetric ? `?rank_metric=${encodeURIComponent(rankMetric)}` : "";
  const res = await fetch(`/api/race/${encodeURIComponent(raceId)}${q}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RaceResult;
}

export interface RaceSummary {
  raceId: string;
  name?: string;
  archived?: boolean;
  useSpot?: boolean;
  splitId: string;
  stamp: string;
  models: string[];
  states: Record<string, string>;
  // True for curated read-only showcase runs surfaced via the Settings "Import
  // sample runs" toggle (shared namespace; clone to make your own).
  isSample?: boolean;
}

export async function listRaces(): Promise<RaceSummary[]> {
  const res = await fetch("/api/races");
  if (!res.ok) throw new Error(await errorDetail(res));
  return ((await res.json()) as { races: RaceSummary[] }).races;
}

// Archive (hide) or restore a race. Display-only — never touches its jobs.
export async function archiveRace(raceId: string, archived: boolean): Promise<void> {
  const res = await fetch(
    `/api/race/${encodeURIComponent(raceId)}/archive?archived=${archived}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await errorDetail(res));
}

// --- Settings: AWS environment config + preflight ---

export interface AwsConfigView {
  region: string;
  account: string;
  bucket: string;
  roleArn: string;
  imageUri: string;
  profile: string;
  // Effective on/off state of the SageMaker serverless engine (env or saved
  // config). Set via the Settings toggle (putConfig) — flips the engine without
  // a redeploy. Undefined on older backends.
  enableSagemakerServerless?: boolean;
  // Set only when the backend could not determine the AWS account, in which case
  // account/bucket/roleArn/imageUri come back empty and this holds the reason.
  // Expected on a first run before an account is configured.
  configError?: string;
}

export interface PreflightCheck {
  check: string;
  ok: boolean;
  detail: string;
}

export interface PreflightResult {
  ok: boolean;
  config: AwsConfigView;
  checks: PreflightCheck[];
}

export async function getConfig(): Promise<AwsConfigView> {
  const res = await fetch("/api/config");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as AwsConfigView;
}

export async function putConfig(update: Partial<AwsConfigView>): Promise<AwsConfigView> {
  const res = await fetch("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as AwsConfigView;
}

export async function checkConfig(): Promise<PreflightResult> {
  const res = await fetch("/api/config/check", { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as PreflightResult;
}

// --- Per-agent model selection (Settings → AI agents) ---

// One selectable Bedrock model (from the shared baseline registry).
export interface AgentModelChoice {
  key: string;
  label: string;
  provider: string;
  modelId: string;
}

// One configurable agent role + its resolved selection. `deployTime` agents run
// on the AgentCore runtime, so a change only applies after the agent is redeployed.
export interface AgentRoleView {
  key: string;
  label: string;
  description: string;
  deployTime: boolean;
  selectedKey: string;
  selectedLabel: string;
  selectedModelId: string;
}

export interface AgentModelsView {
  roles: AgentRoleView[];
  models: AgentModelChoice[];
  default: string;
}

export async function getAgentModels(): Promise<AgentModelsView> {
  const res = await fetch("/api/agent-models");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as AgentModelsView;
}

// Persist per-role overrides ({roleKey: modelKey}). Setting a role to its default
// key clears the override. Returns the fresh view.
export async function putAgentModels(
  overrides: Record<string, string>,
): Promise<AgentModelsView> {
  const res = await fetch("/api/agent-models", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ overrides }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as AgentModelsView;
}

// Hide all currently-existing SageMaker jobs/runs from the UI by stamping a
// reset cutoff at NOW. SageMaker job records can't be deleted, so this is a soft
// "start blank": only jobs created AFTER the cutoff appear in listings.
export async function resetJobHistory(): Promise<{ resetCutoff: string }> {
  const res = await fetch("/api/config/reset-cutoff", { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as { resetCutoff: string };
}

// Sample ("golden") runs — a curated showcase a user can enable to explore a real
// dataset → fine-tune → leaderboard without launching anything. Toggling does NOT
// copy data; it flips a per-user flag that overlays the shared sample namespace.
export interface SamplesStatus {
  enabled: boolean;
  sampleCount: number;
}

export async function getSamplesStatus(): Promise<SamplesStatus> {
  const res = await fetch("/api/samples/status");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as SamplesStatus;
}

export async function setSamplesEnabled(enabled: boolean): Promise<SamplesStatus> {
  const res = await fetch("/api/samples/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as SamplesStatus;
}

export interface Limits {
  maxModelsPerRace: number;
  maxConcurrentRaces: number;
  // Optional platform-wide (cross-tenant) concurrent-run cap; 0 = disabled.
  maxGlobalConcurrentRaces?: number;
  allowedInstanceTypes: string[];
}

export async function getLimits(): Promise<Limits> {
  const res = await fetch("/api/limits");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as Limits;
}

// --- HuggingFace token (write-only; unlocks gated models) ---

// isSet = the caller's OWN token. usingSharedFallback = they have none, but the
// platform's shared token is covering for them (HF features work; the app nags
// them to set their own).
export interface HfTokenStatus {
  isSet: boolean;
  usingSharedFallback?: boolean;
}

export async function getHfTokenStatus(): Promise<HfTokenStatus> {
  const res = await fetch("/api/hf-token");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as HfTokenStatus;
}

export async function setHfToken(token: string): Promise<{ isSet: boolean }> {
  const res = await fetch("/api/hf-token", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as { isSet: boolean };
}

// --- Dataset library ---

export interface Dataset {
  splitId: string;
  name: string | null;
  trainRows: number | null;
  evalRows: number | null;
  mode: string | null;
  evalOnly?: boolean;
  archived?: boolean;
  source?: string | null; // "huggingface" | "auto" | "assert" | "eval-upload" | "preference"
  hfDataset?: string | null; // the source HF dataset id, when source==="huggingface"
  shape?: string; // "sft" (messages) | "preference" (DPO chosen/rejected)
  hasVal?: boolean; // has a validation split → early stopping available
  valRows?: number;
  valMode?: string;
  hasBaseline: boolean;
  mtime: number;
  // Dataset-investigation recommendations (if run): the leaderboard rank metric
  // and the verifiable RLVR reward that mirrors it (null when none applies).
  recommendedRankMetric?: string | null;
  recommendedRewardMetric?: string | null;
  // KTO class-balance loss-weight recommendation (λ_D / λ_U) from the profiler,
  // when the KTO dataset is imbalanced — the FineTune KTO step one-click pre-fills.
  recommendedChosenWeight?: number | null;
  recommendedRejectedWeight?: number | null;
}

// By default the backend EXCLUDES archived datasets (so the Fine-tune + Eval
// pickers only see available ones). The Datasets master page passes
// includeArchived=true to manage the full set.
export async function getDatasets(includeArchived = false): Promise<Dataset[]> {
  const q = includeArchived ? "?include_archived=true" : "";
  const res = await fetch(`/api/datasets${q}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return ((await res.json()) as { datasets: Dataset[] }).datasets;
}

// Archive (hide) or restore a dataset. Display-only — never deletes data.
export async function archiveDataset(splitId: string, archived: boolean): Promise<void> {
  const res = await fetch(
    `/api/datasets/${encodeURIComponent(splitId)}/archive?archived=${archived}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await errorDetail(res));
}

// --- Investigate dataset: deterministic profile + recommended eval strategy ---
export interface LenDist {
  min?: number;
  p50?: number;
  p95?: number;
  max?: number;
  mean?: number;
}
export interface FileProfile {
  rows: number;
  sampled: boolean;
  parsed: number;
  malformed: number;
  emptyOutputs?: number;
  duplicateOutputs?: number;
  taskMix?: Record<string, number>;
  dominantTask?: string;
  taskConsistency?: number;
  scaffold?: { rate?: number; patterns?: Record<string, number> };
  goldWordLen?: LenDist;
  json?: {
    jsonRows?: number;
    goldValidRaw?: number;
    goldValidStripped?: number;
    schemaConsistency?: number | null;
    dominantKeys?: string[];
    distinctSchemas?: number;
  };
  labels?: {
    labelRows?: number;
    numClasses?: number;
    distribution?: Record<string, number>;
    classes?: string[]; // full class list (freq-sorted), not just top-N
    minorityRate?: number | null;
    imbalanced?: boolean;
  };
  truncation?: {
    cutoffLen?: number;
    approxTokenP95?: number;
    approxTokenMax?: number;
    estTruncatedRows?: number;
  };
}
export interface DatasetProfile {
  splitId: string;
  name: string | null;
  shape?: string; // "sft" (messages) | "preference" (DPO chosen/rejected)
  hasVal: boolean;
  evalOnly: boolean;
  provenance?: {
    source?: string | null;
    hfDataset?: string | null;
    hfConfig?: string | null;
    hfSplit?: string | null;
    hfSampleSeed?: number | null;
  };
  structure: {
    hasSystemPromptRate?: number;
    systemPromptFixed?: boolean;
    distinctSystemPrompts?: number;
    fixedSystemPrompt?: string | null;
    multiTurnRate?: number;
    turnsPerExample?: LenDist;
  };
  train?: FileProfile; // absent for preference datasets (ranking train file)
  eval: FileProfile;
  val?: FileProfile;
  // Present for preference (DPO) datasets — the ranking-pair profile.
  preference?: {
    pairs: number;
    malformed?: number;
    identicalPairs?: number;
    chosenWordLen?: LenDist;
    rejectedWordLen?: LenDist;
    // median(chosen words) / median(rejected words). >~1.5 → DPO may learn
    // verbosity, not quality (Rafailov et al. 2023). null when rejected median 0.
    lengthBiasRatio?: number | null;
  };
  // Present for KTO datasets — the labelled-completion balance + λ recommendation.
  kto?: {
    rows: number;
    malformed?: number;
    desirable?: number;
    undesirable?: number;
    desirableRate?: number | null;
    // max(nD,nU)/min(nD,nU) — 1.0 = perfectly balanced; null if a class is missing.
    imbalanceRatio?: number | null;
    // Concrete KTO loss weights to apply — shown in the Investigate KTO card for
    // the user to enter in the FineTune "KTO loss weights" inputs (manual, not
    // auto-prefilled). Raises λ on the minority class so λD·nD/λU·nU ≈ 1 (KTO §4.2).
    recommendedChosenWeight?: number;
    recommendedRejectedWeight?: number;
    weightsBalanced?: boolean; // true when no re-weighting is needed (ratio < 1.5)
  };
  // Present for RLVR datasets — prompt + verifiable ground_truth profile.
  rlvr?: {
    rows: number;
    malformed?: number;
    emptyGroundTruth?: number;
    numericGroundTruthRate?: number | null;
    shortGroundTruthRate?: number | null;
    promptWordLen?: LenDist;
    groundTruthWordLen?: LenDist;
  };
  leakage: { checked: boolean; exactOverlapRows?: number; evalLeakedRate?: number };
  recommendation: {
    rankMetric: string;
    alsoWatch: string[];
    rationale: string[];
    detectedTask: string;
  };
  // Recommended training objective from the data shape + why.
  objective?: { objective: "sft" | "dpo" | "kto" | "rlvr" | "rlaif"; rationale: string[] };
  warnings: { severity: "error" | "warning" | "info"; message: string }[];
}

export async function profileDataset(splitId: string, cutoffLen?: number): Promise<DatasetProfile> {
  const qs = cutoffLen ? `?cutoff_len=${cutoffLen}` : "";
  const res = await fetch(`/api/datasets/${encodeURIComponent(splitId)}/profile${qs}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// --- Agentic dataset investigation (Strands agent on Bedrock AgentCore) -------
// Layers on the deterministic profile: facet-gated follow-up questions about the
// business context the data can't reveal, then a confirmed eval-config proposal.

export interface InvestigateQuestion {
  id: string;
  facet: "task" | "human" | string;
  question: string;
  why: string;
  affects: string;
}

export interface InvestigateQuestions {
  questions: InvestigateQuestion[];
  summary: string;
  profile: DatasetProfile;
}

export interface InvestigateProposal {
  splitId: string;
  taskType: string;
  rankMetric: string;
  alsoWatch: string[];
  // The verifiable RLVR reward metric that mirrors rankMetric, or null when the
  // rank metric isn't a per-row verifiable check (e.g. llm_judge:*). Closes the
  // reward↔leaderboard-metric loop: the FineTune RLVR step pre-offers this reward.
  recommendedRewardMetric?: string | null;
  cutoffGuidance: string;
  flaggedIssues: string[];
  rationale: string[];
  appliedAnswers: Record<string, string>;
}

export interface InvestigationState {
  status: { phase: string; status: string; detail: string };
  questions: InvestigateQuestions | null;
  proposal: InvestigateProposal | null;
  answers: Record<string, string>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Both agent steps run ~20-28s. The hosted backend returns {status:"running"}
// and runs them on the worker Lambda (API Gateway's 29s limit), so we poll the
// GET endpoint until the result lands. Local backend returns the result inline.

export async function investigateQuestions(
  splitId: string,
  cutoffLen?: number,
): Promise<InvestigateQuestions> {
  const qs = cutoffLen ? `?cutoff_len=${cutoffLen}` : "";
  const res = await fetch(
    `/api/datasets/${encodeURIComponent(splitId)}/investigate/questions${qs}`,
    { method: "POST" },
  );
  if (!res.ok) throw new Error(await errorDetail(res));
  const out = await res.json();
  if (out.questions) return out; // inline result
  // running → poll
  for (let i = 0; i < 40; i++) {
    await sleep(3000);
    const st = await getInvestigation(splitId);
    if (st.status.phase === "questions" && st.status.status === "failed")
      throw new Error(st.status.detail || "question generation failed");
    if (st.questions) return st.questions;
  }
  throw new Error("timed out waiting for follow-up questions");
}

export async function investigateProposal(
  splitId: string,
  answers: Record<string, string>,
): Promise<InvestigateProposal> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(splitId)}/investigate/proposal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const out = await res.json();
  if (out.rankMetric) return out; // inline result
  for (let i = 0; i < 40; i++) {
    await sleep(3000);
    const st = await getInvestigation(splitId);
    if (st.status.phase === "proposal" && st.status.status === "failed")
      throw new Error(st.status.detail || "proposal failed");
    if (st.proposal) return st.proposal;
  }
  throw new Error("timed out waiting for recommendation");
}

export async function getInvestigation(splitId: string): Promise<InvestigationState> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(splitId)}/investigate`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Eval-only dataset upload (standalone evaluate): a single JSONL of held-out
// rows to score on — no train split.
export interface EvalDatasetResult {
  ok: boolean;
  splitId?: string;
  evalRows: number;
  invalidRows: number;
  errors?: RowError[];
  filename?: string;
  name?: string;
}

export async function uploadEvalDataset(file: File, name = ""): Promise<EvalDatasetResult> {
  const uploadId = await uploadToS3(file);
  const form = new FormData();
  form.append("upload_id", uploadId);
  form.append("name", name);
  const res = await fetch("/api/datasets/eval-only", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as EvalDatasetResult;
}

// --- Race retry (per-stage) ---

export async function retryRaceEntry(
  raceId: string,
  modelId: string,
  resume = false,
): Promise<RaceResult> {
  // resume=true continues a failed TRAINING run from its last checkpoint (only
  // honoured server-side when the entry is resumable; else falls back to fresh).
  const res = await fetch(
    `/api/race/${encodeURIComponent(raceId)}/retry?model_id=${encodeURIComponent(modelId)}` +
      (resume ? "&resume=true" : ""),
    { method: "POST" }
  );
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RaceResult;
}

// --- Model export (deploy to your own AWS account) ---

// One presigned weight file (serverless engine: the output is an uncompressed
// prefix of loose files, so each is presigned individually).
export interface ExportWeightFile {
  name: string; // path relative to the weights prefix
  url: string;
  size: number;
}

export interface ExportInfo {
  modelId: string;
  modelDisplay: string;
  hfBaseModel: string;
  template: string;
  gated: boolean;
  deployMode: "merged" | "adapter";
  weightsSubdir: string;
  suggestedInstance: string;
  trainJob: string;
  artifactS3Uri: string;
  // Which engine produced the artifact. "llama_factory" → one weightsUrl tarball;
  // "sagemaker_serverless" → weightsFiles (per-file presigned loose files).
  engine: "llama_factory" | "sagemaker_serverless";
  weightsUrl?: string; // present for llama_factory only
  weightsFiles?: ExportWeightFile[]; // present for sagemaker_serverless only
  weightsTtlSeconds?: number;
  // Set for a GATED full/freeze fine-tune: the artifact is merged weights of a
  // gated base, so the download is withheld until the user accepts that base's
  // license. When licenseRequired is true, weightsUrl/weightsFiles are ABSENT.
  requiresLicenseAcceptance?: boolean;
  licenseRequired?: boolean;
  licenseModel?: string; // the gated base model whose license must be accepted
}

export async function exportModelInfo(
  raceId: string,
  modelId: string,
  licenseAccepted = false,
): Promise<ExportInfo> {
  const qs = licenseAccepted ? "?license_accepted=true" : "";
  const res = await fetch(
    `/api/race/${encodeURIComponent(raceId)}/export/${encodeURIComponent(modelId)}${qs}`
  );
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as ExportInfo;
}

// The deploy bundle is a file download — return the URL the browser navigates to.
// Pass licenseAccepted for a gated full/freeze fine-tune (else the backend 403s).
export function exportBundleUrl(raceId: string, modelId: string, licenseAccepted = false): string {
  const qs = licenseAccepted ? "?license_accepted=true" : "";
  return `/api/race/${encodeURIComponent(raceId)}/export/${encodeURIComponent(modelId)}/bundle${qs}`;
}

// --- RLVR custom reward functions ---

export interface RewardFunction {
  id: string;
  name: string;
  snippet: string;
  // "snippet"/"metric" = verifiable RLVR rewards (a Lambda). "reward_prompt" = an
  // RLAIF AI-judge prompt (no Lambda; passed inline to the trainer).
  kind: "snippet" | "metric" | "reward_prompt";
  metric: string | null;
  prompt?: string; // RLAIF reward_prompt: the judge prompt text
  rewardModelId?: string; // RLAIF reward_prompt: the judge model id ("" = recipe default)
  lambdaArn: string;
  evaluatorArn: string;
  status: "draft" | "deploying" | "deployed" | "failed";
  error: string;
  deployed: boolean;
  createdStamp: string;
}

export interface RewardFunctionsResponse {
  rewardFunctions: RewardFunction[];
  metrics: string[]; // eval.py metrics a 'metric' reward can mirror
  judgeModels?: string[]; // RLAIF judge models a reward prompt can use
}

export async function listRewardFunctions(): Promise<RewardFunctionsResponse> {
  const res = await fetch("/api/reward-functions");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RewardFunctionsResponse;
}

// Create a reward function from EXACTLY ONE source: a `metric` (generates a
// verifiable snippet), a user-authored Python `snippet`, or a `prompt` (an RLAIF
// AI-judge reward prompt). metric/snippet build a Lambda (status=deploying); a
// prompt needs no AWS and is usable immediately.
export async function createRewardFunction(
  req: { name: string; metric?: string; snippet?: string; prompt?: string; rewardModelId?: string }
): Promise<RewardFunction> {
  const res = await fetch("/api/reward-functions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RewardFunction;
}

export async function getRewardFunction(rewardId: string): Promise<RewardFunction> {
  const res = await fetch(`/api/reward-functions/${encodeURIComponent(rewardId)}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RewardFunction;
}

export async function deleteRewardFunction(rewardId: string): Promise<void> {
  const res = await fetch(`/api/reward-functions/${encodeURIComponent(rewardId)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorDetail(res));
}

// Parse-validate a snippet without saving (inline editor feedback).
export async function validateRewardSnippet(snippet: string): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch("/api/reward-functions/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ snippet }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Parse-validate an RLAIF reward PROMPT without saving (inline editor feedback):
// must contain the {{prompt}} and {{response}} placeholders.
export async function validateRewardPrompt(prompt: string): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch("/api/reward-functions/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return await res.json();
}

// Pre-launch advisory: would the chosen RLVR reward grade this split's
// ground_truth, or score ~0 every step? Returns a human-readable warning (or
// null). Surfaced inline on the FineTune RLVR step before a billable launch.
export async function checkRewardDomain(
  req: { splitId: string; presetRewardFunction?: string; rewardFunctionId?: string }
): Promise<{ warning: string | null; groundTruthTask: string | null }> {
  const res = await fetch("/api/reward-functions/domain-check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as { warning: string | null; groundTruthTask: string | null };
}

// Dry-run a reward against ONE sample (response, ground_truth) — in-process, no
// AWS — so the user sees the score the deployed Lambda would return before
// launching a billable GRPO run. Pass rewardId for a saved reward OR snippet for
// unsaved editor code. Throws (400 detail) on an invalid/raising snippet.
export async function tryRewardFunction(
  req: { response: string; groundTruth: string; rewardId?: string; snippet?: string; metric?: string }
): Promise<{ score: number }> {
  const res = await fetch("/api/reward-functions/try", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as { score: number };
}

// Dry-run an RLAIF judge RUBRIC on good/bad candidate responses → per-sample scores
// + a discrimination spread (does the rubric separate good from bad?), all before
// any billable GRPO run.
export interface RewardPromptSample {
  prompt: string;
  response: string;
  intendedLabel?: "good" | "bad" | null;
  score: number;
  reasoning: string;
  error: string | null;
}
export interface RewardPromptDryRun {
  samples: RewardPromptSample[];
  scoreSpread: {
    goodMean: number;
    badMean: number;
    separation: number;
    discriminates: boolean;
  } | null;
  indicative: boolean;
}
export async function tryRewardPrompt(req: {
  prompt: string;
  samples: { prompt: string; response: string; intendedLabel?: "good" | "bad" }[];
  rewardModelId?: string;
}): Promise<RewardPromptDryRun> {
  const res = await fetch("/api/reward-functions/try-prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RewardPromptDryRun;
}

// --- RLAIF reward-prompt authoring agent ("Draft with AI") ---
// The agent drafts a {{prompt}}/{{response}} judge rubric for a plain-English goal,
// scores fabricated good/bad candidates with a real judge, and iterates until they
// separate — so the user starts from a CALIBRATED rubric, not a blank box. The
// returned draft pre-fills the authoring form; deploy stays an explicit click.
export interface RewardAuthorResult {
  draftPrompt: string;
  rewardModelId: string;
  samples: RewardPromptSample[];
  scoreSpread: {
    goodMean: number | null;
    badMean: number | null;
    separation: number | null;
    discriminates: boolean;
  } | null;
  rationale: string[];
  iterations: number;
  judgeCalls?: number;
  warnings: string[];
  splitId?: string;
}

// Author a rubric for `goal` grounded in `splitId`'s prompt-only profile. The agent
// makes several judge calls + an AgentCore round-trip (>29s possible), so the
// endpoint dispatches to the worker; we poll until the draft is ready. Local dev
// returns the result inline. The poll cap is generous (≤3 rounds × ~8 judge calls
// + cold start) — wider than triage/interpret's 90s.
export async function authorRewardPrompt(req: {
  splitId: string;
  goal: string;
  priorResult?: RewardAuthorResult | null;
}): Promise<RewardAuthorResult> {
  const res = await fetch("/api/reward-functions/author", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const out = await res.json();
  if (out.result) return out.result as RewardAuthorResult; // inline (local dev)
  if (out.draftPrompt) return out as RewardAuthorResult; // legacy inline shape
  // running → poll the worker. ~60 × 3s = 180s, generous for the multi-call loop
  // + a possible cold start (triage/interpret's 90s isn't enough here).
  for (let i = 0; i < 60; i++) {
    await sleep(3000);
    const st = await fetch(
      `/api/reward-functions/author/${encodeURIComponent(req.splitId)}`
    ).then((r) => r.json());
    if (st.status === "failed") throw new Error(st.detail || "reward author failed");
    if (st.result) return st.result as RewardAuthorResult;
  }
  throw new Error("timed out waiting for the drafted rubric");
}

// --- RLVR reward curve (GRPO reward over training steps) ---

export interface RewardCurve {
  steps: number[];
  rewardMean: (number | null)[];
  rewardMax: (number | null)[];
  rewardMin: (number | null)[];
  valReward: { step: number; value: number }[];
  hasData: boolean;
}

export async function fetchRewardCurve(jobName: string): Promise<RewardCurve> {
  const res = await fetch(`/api/train/${encodeURIComponent(jobName)}/reward-curve`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as RewardCurve;
}

// --- Feedback board (issues / ideas / praise, with screenshot attachments) ---

export interface FeedbackAttachment {
  key: string;
  name: string;
  size: number;
  contentType: string;
  url: string; // fresh presigned GET url (minted per read; may be empty if expired)
}
export interface FeedbackEntry {
  id: string;
  type: "issue" | "idea" | "praise";
  title: string;
  body: string;
  author: string;
  status: "open" | "planned" | "done" | "wont_do";
  createdStamp: string;
  attachments: FeedbackAttachment[];
}
export interface FeedbackBoard {
  feedback: FeedbackEntry[];
  types: string[];
  statuses: string[];
  maxAttachments: number;
  me: string; // the current user (to show Delete only on own rows)
}

export async function listFeedback(): Promise<FeedbackBoard> {
  const res = await fetch("/api/feedback");
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as FeedbackBoard;
}

// Upload one image RAW (NOT gzipped — unlike uploadToS3, which gzips dataset
// JSONL) to S3 via a presigned PUT, returning its uploadId for the feedback POST.
export async function uploadImageToS3(file: File): Promise<string> {
  const res = await fetch("/api/datasets/upload-url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  const { uploadId, url } = (await res.json()) as { uploadId: string; url: string };
  const put = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": file.type || "image/png" },
    body: file,
  });
  if (!put.ok) throw new Error(`upload failed: ${put.status} ${put.statusText}`);
  return uploadId;
}

// Submit feedback. Uploads each screenshot to S3 first, then posts the entry with
// the resulting uploadIds (the backend copies them into the entry + serves them).
export async function createFeedback(req: {
  type: "issue" | "idea" | "praise";
  title: string;
  body: string;
  files: File[];
}): Promise<FeedbackEntry> {
  const attachmentUploadIds: string[] = [];
  for (const f of req.files) attachmentUploadIds.push(await uploadImageToS3(f));
  const res = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: req.type, title: req.title, body: req.body, attachmentUploadIds }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as FeedbackEntry;
}

export async function setFeedbackStatus(id: string, status: string): Promise<FeedbackEntry> {
  const res = await fetch(`/api/feedback/${encodeURIComponent(id)}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as FeedbackEntry;
}

export async function deleteFeedback(id: string): Promise<void> {
  const res = await fetch(`/api/feedback/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorDetail(res));
}

// --- Guided Fine-tuning agent ("Pit Crew") ---------------------------------
// A conversational front door for non-ML users: describe a goal → bring data →
// the agent profiles it + proposes a RACE → the user approves → it launches +
// emails on finish. The state machine lives in backend/app/pitcrew.py; these are
// thin wrappers over /api/pitcrew/*. The only billable action is `approve`.

// One model in a proposed race plan (the plain-language view the UI renders).
export interface PlannedModelView {
  entryKey: string; // stable per-ARM id (a model can appear as several arms)
  modelId: string;
  displayName: string;
  label: string; // display label that distinguishes arms of the same model
  paramsB: number;
  instanceType: string;
  method: string;
  variant: string;
  stage: string;
  learningRate: number;
  loraRank: number | null;
  role: string; // plain-language "why this one"
}

export interface RacePlanView {
  supported: boolean;
  objective: string;
  engine: string;
  effort: string;
  jobBudget: number;
  rankMetric: string;
  detectedTask: string;
  models: PlannedModelView[];
  gatesApplied: string[];
  reason: string;
  // "up to N" transparency: how many arms were actually filled vs the ceiling the
  // user picked, and whether the planner deliberately stopped short (no more useful
  // arms). Optional so older payloads still parse. See race_planner.RacePlan.
  meaningfulCount?: number;
  ceiling?: number;
  capped?: boolean;
}

export interface CostEstimateView {
  totalUsd: { lo: number; hi: number };
  wallClockMin: { lo: number; hi: number };
  jobs: number;
  useSpot: boolean;
  perModel: { instanceType: string; costUsd: number; minutes: number }[];
  disclaimer: string;
}

// A single chat message in the session thread. `role` is assistant|user; the
// optional structured fields drive inline UI (a plan card, an estimate, etc.).
export interface PitcrewMessage {
  role: "assistant" | "user";
  text: string;
  editable?: boolean; // a free-text user message the user can edit/rewind
  editKind?: string; // "goal" | "correction"
  collectGoal?: boolean; // render the goal text box (the opening turn)
  datasetsHint?: boolean;
  confirmTask?: boolean;
  taskSummary?: { objective: string; detectedTask: string; rows: number; plain: string };
  chooseEffort?: boolean;
  efforts?: { key: string; label: string }[];
  reviewPlan?: boolean;
  plan?: RacePlanView;
  estimate?: CostEstimateView;
  // Models the user may ADD to the race (eligible for this dataset, minus the ones
  // already proposed). Drives the "add a model" picker on the review screen.
  addPool?: { modelId: string; displayName: string; paramsB: number; family: string }[];
  notifyEmailPrefill?: string;
  planUnsupported?: boolean;
  launched?: boolean;
  launchBlocked?: boolean;
  finished?: boolean;
  raceId?: string;
  winner?: string;
}

export interface PitcrewSession {
  sessionId: string;
  createdAt: string;
  updatedAt: string;
  version: number;
  phase: string;
  title: string;
  titleManual?: boolean;
  goal: string;
  splitId: string;
  datasetName: string;
  effort: string;
  shape: string;
  plan: RacePlanView | null;
  estimate: CostEstimateView | null;
  raceId: string;
  notifyEmail: string;
  messages: PitcrewMessage[];
}

export interface PitcrewSessionSummary {
  sessionId: string;
  title: string;
  phase: string;
  raceId: string;
  createdAt: string;
  mtime: number;
}

export async function listPitcrewSessions(): Promise<PitcrewSessionSummary[]> {
  const res = await fetch("/api/pitcrew/sessions");
  if (!res.ok) throw new Error(await errorDetail(res));
  return ((await res.json()) as { sessions: PitcrewSessionSummary[] }).sessions;
}

export async function newPitcrewSession(): Promise<PitcrewSession> {
  const res = await fetch("/api/pitcrew/sessions", { method: "POST" });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as PitcrewSession;
}

export async function getPitcrewSession(sessionId: string): Promise<PitcrewSession> {
  const res = await fetch(`/api/pitcrew/sessions/${encodeURIComponent(sessionId)}`);
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as PitcrewSession;
}

// Rename a session (the sidebar label). A user-chosen title sticks — it won't be
// overwritten by the goal text later.
export async function renamePitcrewSession(sessionId: string, title: string): Promise<PitcrewSession> {
  const res = await fetch(`/api/pitcrew/sessions/${encodeURIComponent(sessionId)}/title`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as PitcrewSession;
}

// Archive (hide) a session from the sidebar. Soft-delete — the launched race is
// never orphaned. Pass archived=false to restore.
export async function archivePitcrewSession(sessionId: string, archived = true): Promise<void> {
  const res = await fetch(`/api/pitcrew/sessions/${encodeURIComponent(sessionId)}/archive?archived=${archived}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await errorDetail(res));
}

// Edit/rewind an earlier free-text user message (the goal, or a task correction).
// The conversation truncates there and replays; any downstream dataset is unlinked
// but kept on disk. Blocked once a race has launched.
export async function editPitcrewMessage(
  sessionId: string,
  messageIndex: number,
  text: string,
  expectedVersion?: number
): Promise<PitcrewSession> {
  const res = await fetch(`/api/pitcrew/sessions/${encodeURIComponent(sessionId)}/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messageIndex, text, expectedVersion }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as PitcrewSession;
}

// Advance the session one user turn. `action` names the transition (goal,
// use_dataset, confirm, effort, approve, edit_effort, cancel); payload carries
// its data. `expectedVersion` guards against a second tab clobbering the phase
// (a 409 means reload).
export async function advancePitcrew(
  sessionId: string,
  action: string,
  payload: Record<string, unknown> = {},
  expectedVersion?: number
): Promise<PitcrewSession> {
  const res = await fetch(`/api/pitcrew/sessions/${encodeURIComponent(sessionId)}/advance`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, payload, expectedVersion }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as PitcrewSession;
}

// Detect a raw upload's dataset shape (sft/preference/kto/rlvr/rlaif) so the
// guided agent can describe it in plain language without the user picking a type.
export interface ShapeDetection {
  shape: string;
  label: string;
  confidence: number;
  rows: number;
  matchRates: Record<string, number>;
  ambiguous: boolean;
}

export async function detectShape(text: string): Promise<ShapeDetection> {
  const res = await fetch("/api/datasets/detect-shape", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return (await res.json()) as ShapeDetection;
}
