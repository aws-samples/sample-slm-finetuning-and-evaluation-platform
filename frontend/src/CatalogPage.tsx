// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useMemo, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Table from "@cloudscape-design/components/table";
import Button from "@cloudscape-design/components/button";
import Box from "@cloudscape-design/components/box";
import Badge from "@cloudscape-design/components/badge";
import Icon from "@cloudscape-design/components/icon";
import Link from "@cloudscape-design/components/link";
import Popover from "@cloudscape-design/components/popover";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import TextFilter from "@cloudscape-design/components/text-filter";
import Select from "@cloudscape-design/components/select";
import { useCollection } from "@cloudscape-design/collection-hooks";
import { ProviderIcon } from "./providerIcon";

import Modal from "@cloudscape-design/components/modal";
import {
  backfillVerifications,
  buildImage,
  deleteCustomModel,
  diagnoseModel,
  getHfTokenStatus,
  getModels,
  smokeTestModel,
  type FinetuningType,
  type LoraVariant,
  type ModelSpec,
  type VerificationRecord,
  type VerificationStatus,
} from "./api";
import { AddFromHuggingFace } from "./AddFromHuggingFace";
import { ServerlessFinder } from "./ServerlessFinder";
import { useNotify, errText } from "./notifications";

// A catalog row is EITHER a provider group header OR a model under it. The table
// uses expandableRows, so providers are collapsible parents and models children.
type Row =
  | { kind: "provider"; key: string; provider: string; models: ModelSpec[] }
  | ({ kind: "model"; key: string; parentProvider: string } & ModelSpec);

const STATUS_FILTER_OPTIONS = [
  { value: "all", label: "All verification states" },
  { value: "verified", label: "Verified (on its image)" },
  { value: "pending", label: "Verifying (pending)" },
  { value: "incompatible", label: "Incompatible" },
  { value: "access_denied", label: "Access denied (gated)" },
  { value: "untested", label: "Untested" },
];

// Human labels for each parameterization, used in the verify notification + chips.
const METHOD_LABELS: Record<FinetuningType, string> = {
  lora: "LoRA",
  qlora: "QLoRA",
  full: "Full",
  freeze: "Freeze",
};

// One short description per method, shown in the chip's popover so the user knows
// what they're verifying (and the cost) before clicking a billable smoke-test.
const METHOD_BLURB: Record<FinetuningType, string> = {
  lora: "Adapter on the full-precision base — the proven default. Runs on the model's own image (g5).",
  qlora: "LoRA on a 4-bit-quantized base — verified independently (its 4-bit load can fail where LoRA works).",
  full: "Full-weight training (no adapter, standalone model). Full-weight runs use g6e.2xlarge (~$2.80/hr) and SFT only; ≤2B models.",
  freeze: "Trains only the top transformer layers (lighter full-weight, no adapter). Runs on g6e.2xlarge (~$2.80/hr); SFT only, ≤2B models.",
};

// Stable display order for the method chips (LoRA is the primary/default).
const METHOD_ORDER: FinetuningType[] = ["lora", "qlora", "full", "freeze"];

// full/freeze are full-weight runs that route to the pricier g6e card.
const isFullWeight = (m: FinetuningType) => m === "full" || m === "freeze";

// LoRA variants are MODIFIERS on the adapter methods (lora/qlora) — verified
// independently because DoRA/PiSSA change training (and the merge). Display order:
// DoRA first (recommended), then the situational rest. Plain "lora" isn't listed
// here — it IS the method chip's own status.
const VARIANT_ORDER: LoraVariant[] = ["dora", "rslora", "pissa", "loraplus"];
const VARIANT_LABELS: Record<LoraVariant, string> = {
  lora: "Plain LoRA",
  dora: "DoRA",
  rslora: "rsLoRA",
  pissa: "PiSSA",
  loraplus: "LoRA+",
};
// DoRA + PiSSA need full-precision weights a 4-bit QLoRA base lacks (PEFT rejects
// them at load) — mirrors the backend QUANT_INCOMPATIBLE_VARIANTS + Hyperparams
// guard, so the picker can't offer an invalid, billable combo.
const QUANT_INCOMPATIBLE_VARIANTS: LoraVariant[] = ["dora", "pissa"];
const variantAllowedForMethod = (variant: LoraVariant, method: FinetuningType) =>
  method === "qlora" ? !QUANT_INCOMPATIBLE_VARIANTS.includes(variant) : method === "lora";

// Verification key for a non-plain LoRA variant (model, tier, method, variant):
// `<tier>::<method>::<variant>`. Mirrors the backend _key(). Plain "lora" has no
// variant key (it's the bare method key), so this is only for the listed variants.
const variantKeyFor = (tier: string, method: FinetuningType, variant: LoraVariant) =>
  `${tier}::${method}::${variant}`;

// The per-(model, tier, method, variant) BUSY-spinner key. ONE helper so the verify
// button and verify() launcher can't drift (they did: the button embedded a `::lora`
// token the launcher omitted for plain LoRA, so the optimistic spinner never showed).
// Mirrors verifyKeyFor's back-compat rule: plain LoRA has no method token; a non-plain
// variant appends `::<variant>`.
const busyKeyFor = (
  modelId: string,
  tier: string,
  method: FinetuningType,
  variant: LoraVariant = "lora",
) => {
  const base = method === "lora" ? `${modelId}:${tier}` : `${modelId}:${tier}::${method}`;
  return variant === "lora" ? base : `${base}::${variant}`;
};

// Cloudscape badge color for a method chip given its verification status. Green =
// proven, red = incompatible, blue = a job is running, grey = untested/unknown.
function chipColor(status: VerificationStatus): "green" | "red" | "blue" | "grey" {
  if (status === "verified") return "green";
  if (status === "incompatible") return "red";
  if (status === "pending") return "blue";
  return "grey"; // untested | access_denied
}

// The verification status of a model ON ITS OWN image tier (what the race uses).
function ownStatus(m: ModelSpec): VerificationStatus {
  return m.verifications?.[m.imageTag]?.status ?? "untested";
}

// Format a stored verification timestamp (ISO-ish "2026-06-11 16:01:54.93+00:00")
// into a date-only "26-Jun-2026" (dd-month-year) for the verified/tested rows in
// the method-chip + serverless popovers. Falls back to the raw value if it won't
// parse (never throws on odd input).
function fmtVerifiedDate(ts: string | null): string | null {
  if (!ts) return null;
  const d = new Date(ts.includes("T") ? ts : ts.replace(" ", "T"));
  if (Number.isNaN(d.getTime())) return ts;
  const day = String(d.getUTCDate()).padStart(2, "0");
  const month = d.toLocaleString("en-US", { month: "short", timeZone: "UTC" });
  return `${day}-${month}-${d.getUTCFullYear()}`;
}

// A verification status rendered as a Cloudscape StatusIndicator (shared by the
// per-tier rows inside a method chip's popover).
function statusIndicator(status: VerificationStatus, seed?: boolean) {
  if (status === "verified")
    return <StatusIndicator type="success">{seed ? "verified (baseline)" : "verified"}</StatusIndicator>;
  if (status === "incompatible") return <StatusIndicator type="error">incompatible</StatusIndicator>;
  if (status === "access_denied") return <StatusIndicator type="warning">access denied</StatusIndicator>;
  if (status === "pending") return <StatusIndicator type="in-progress">verifying…</StatusIndicator>;
  return <StatusIndicator type="pending">untested</StatusIndicator>;
}

// Roll several per-tier verification records into the single status shown on a
// method's chip: verified if proven on ANY image, else incompatible/pending/
// untested by precedence. (The popover still breaks it down per tier.)
function rollupStatus(recs: VerificationRecord[]): VerificationStatus {
  if (recs.some((r) => r.status === "verified")) return "verified";
  if (recs.some((r) => r.status === "pending")) return "pending";
  if (recs.some((r) => r.status === "incompatible")) return "incompatible";
  if (recs.some((r) => r.status === "access_denied")) return "access_denied";
  return "untested";
}

// Verification key for (tier, method): LoRA lives under the bare tier key
// (back-compat), every other method under `<tier>::<method>`. Mirrors the backend.
const verifyKeyFor = (tier: string, method: FinetuningType) =>
  method === "lora" ? tier : `${tier}::${method}`;

// One method (LoRA/QLoRA/Full/Freeze) as a single colored chip. Green = verified
// (on at least one image), grey = untested, red = incompatible, blue = a job is
// running. Clicking opens a popover (serverless-badge style) with the per-image
// breakdown, provenance, the cost note, and a verify/re-test button per tier.
function MethodChip({
  model,
  method,
  tiers,
  tierImages,
  busy,
  blockedByGate,
  hfTokenSet,
  onVerify,
}: {
  model: ModelSpec;
  method: FinetuningType;
  tiers: string[];
  tierImages: Record<string, string>;
  busy: Record<string, boolean>;
  blockedByGate: boolean;
  hfTokenSet: boolean;
  onVerify: (tier: string, method: FinetuningType, variant?: LoraVariant) => void;
}) {
  const label = METHOD_LABELS[method];
  // Adapter methods (lora/qlora) carry VARIANT flavors; full/freeze don't.
  // Flavors = Plain ("lora") + the variants this method allows: LoRA → all 5
  // (Plain/DoRA/rsLoRA/PiSSA/LoRA+); QLoRA → 3 (Plain/rsLoRA/LoRA+ — DoRA & PiSSA
  // need full-precision weights a 4-bit base lacks). Each (tier, flavor) is proven
  // independently, so the chip popover is a tier × flavor grid.
  const isAdapterMethod = method === "lora" || method === "qlora";
  const flavors: LoraVariant[] = isAdapterMethod
    ? (["lora", ...VARIANT_ORDER] as LoraVariant[]).filter((v) => variantAllowedForMethod(v, method))
    : ["lora"];
  // The verification record for one (tier, flavor): Plain uses the method's own key
  // (bare tier for LoRA, `tier::qlora` for QLoRA); a variant uses `tier::method::variant`.
  const recFor = (tier: string, flavor: LoraVariant): VerificationRecord =>
    model.verifications?.[
      flavor === "lora" ? verifyKeyFor(tier, method) : variantKeyFor(tier, method, flavor)
    ] ?? { status: "untested", jobName: null, reason: null, ts: null };
  const flavorLabel = (f: LoraVariant) => (f === "lora" ? "Plain" : VARIANT_LABELS[f]);
  // Per-tier plain records — kept for the full/freeze branch (no flavors).
  const perTier = tiers.map((tier) => ({ tier, rec: recFor(tier, "lora") }));
  // Chip color rolls up EVERY (tier, flavor) cell — so the LoRA chip is green if ANY
  // flavor (Plain or a variant) is verified on ANY image, not just plain LoRA.
  const allRecs = tiers.flatMap((tier) => flavors.map((f) => recFor(tier, f)));
  const rollup = rollupStatus(allRecs);
  const color = chipColor(rollup);
  // Chip glyph mirrors the serverless badge: ✓ proven, ⚠ incompatible, … running.
  const glyph =
    rollup === "verified" ? " ✓" : rollup === "incompatible" ? " ⚠" : rollup === "pending" ? " …" : "";

  return (
    <Popover
      header={`${label} fine-tuning`}
      triggerType="custom"
      dismissButton={false}
      position="top"
      size="medium"
      content={
        <SpaceBetween size="xs">
          <Box variant="small">{METHOD_BLURB[method]}</Box>
          {isAdapterMethod && (
            <Box variant="small" color="text-status-inactive">
              {method === "lora"
                ? "5 adapter flavors, each proven independently per image: Plain, DoRA, rsLoRA, PiSSA, LoRA+."
                : "3 adapter flavors on a 4-bit base, each proven independently per image: Plain, rsLoRA, LoRA+ (DoRA & PiSSA need full-precision weights)."}
            </Box>
          )}
          {blockedByGate ? (
            <Box variant="small" color="text-status-warning">
              {model.displayName} is gated — accept its license on Hugging Face and save an HF
              token under Settings to enable verification.
            </Box>
          ) : isAdapterMethod ? (
            // Tier × flavor grid: one section per image, each flavor a row with its
            // own status + verify/re-test button. So LoRA shows 5 flavors × N images
            // (10 for stable+latest); QLoRA shows 3 × N.
            tiers.map((tier) => (
              <Box key={tier}>
                <Box variant="awsui-key-label">
                  {tier} <Badge color="grey">{tierImages[tier]}</Badge>
                </Box>
                {flavors.map((flavor) => {
                  const rec = recFor(tier, flavor);
                  const pending = rec.status === "pending";
                  const verb = rec.status === "untested" ? "verify" : "re-test";
                  const bKey = busyKeyFor(model.id, tier, method, flavor);
                  return (
                    <Box key={flavor} padding={{ left: "s" }}>
                      <SpaceBetween direction="horizontal" size="xs">
                        <Box variant="awsui-key-label" display="inline">
                          {flavorLabel(flavor)}
                        </Box>
                        {statusIndicator(rec.status, rec.seed)}
                        <Button
                          variant="inline-link"
                          loading={!!busy[bKey] || pending}
                          disabled={pending}
                          onClick={() => onVerify(tier, method, flavor)}
                        >
                          {pending ? "verifying…" : verb}
                        </Button>
                      </SpaceBetween>
                      {rec.seed && (
                        <Box variant="small" color="text-status-inactive" padding={{ left: "s" }}>
                          Baseline: proven on the shipped image — run it to confirm in your account.
                        </Box>
                      )}
                      {(rec.status === "verified" ||
                        rec.status === "incompatible" ||
                        rec.status === "access_denied") &&
                        fmtVerifiedDate(rec.ts) && (
                          <Box variant="small" color="text-status-inactive" padding={{ left: "s" }}>
                            {rec.status === "verified" ? "Verified" : "Tested"}: {fmtVerifiedDate(rec.ts)}
                            {rec.reason ? ` — ${rec.reason}` : ""}
                          </Box>
                        )}
                      {rec.jobName && (
                        <Box variant="small" color="text-status-inactive" padding={{ left: "s" }}>
                          <span style={{ wordBreak: "break-all" }}>Job: {rec.jobName}</span>
                        </Box>
                      )}
                    </Box>
                  );
                })}
              </Box>
            ))
          ) : (
            // full/freeze: no variants — one row per image tier.
            perTier.map(({ tier, rec }) => {
              const bKey = busyKeyFor(model.id, tier, method);
              const pending = rec.status === "pending";
              const verb = rec.status === "untested" ? "verify" : "re-test";
              return (
                <Box key={tier}>
                  <SpaceBetween direction="horizontal" size="xs">
                    <Box variant="awsui-key-label" display="inline">
                      {tier} <Badge color="grey">{tierImages[tier]}</Badge>
                    </Box>
                    {statusIndicator(rec.status, rec.seed)}
                  </SpaceBetween>
                  {rec.seed && (
                    <Box variant="small" color="text-status-inactive">
                      Baseline: proven on the shipped image — run it here to confirm in your
                      own account (your run overrides this).
                    </Box>
                  )}
                  {(rec.status === "verified" ||
                    rec.status === "incompatible" ||
                    rec.status === "access_denied") &&
                    fmtVerifiedDate(rec.ts) && (
                      <Box variant="small" color="text-status-inactive">
                        {rec.status === "verified" ? "Verified" : "Tested"}: {fmtVerifiedDate(rec.ts)}
                      </Box>
                    )}
                  {rec.jobName && (
                    <Box variant="small" color="text-status-inactive">
                      <span style={{ wordBreak: "break-all" }}>Job: {rec.jobName}</span>
                    </Box>
                  )}
                  {rec.reason && (
                    <Box variant="small" color="text-status-inactive">
                      Reason: {rec.reason}
                    </Box>
                  )}
                  <Button
                    variant="inline-link"
                    loading={!!busy[bKey] || pending}
                    disabled={pending}
                    onClick={() => onVerify(tier, method)}
                  >
                    {pending ? "verifying…" : `${verb} on ${tier}${isFullWeight(method) ? " (g6e)" : ""}`}
                  </Button>
                </Box>
              );
            })
          )}
          {!blockedByGate && model.gated && hfTokenSet && (
            <Box variant="small" color="text-status-inactive">
              Gated model — your saved HF token (with license accepted) is used to download weights.
            </Box>
          )}
        </SpaceBetween>
      }
    >
      <Badge color={color}>
        {label}
        {glyph}
      </Badge>
    </Popover>
  );
}

export function CatalogPage() {
  const { notify } = useNotify();
  const [models, setModels] = useState<ModelSpec[]>([]);
  const [tiers, setTiers] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [statusFilter, setStatusFilter] = useState("all");
  // model id currently being (re)verified → its in-flight tier, so the button spins.
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  // "Add models from Hugging Face" modal (manual probe + image-diff discovery).
  const [addOpen, setAddOpen] = useState(false);
  // "Find serverless-customizable models" modal (live SageMaker Public Hub diff).
  const [serverlessFinderOpen, setServerlessFinderOpen] = useState(false);
  // Custom-model removal: the model the user is about to delete (confirm modal)
  // + the in-flight delete id (button spinner). A custom model added via probe
  // MUST be removable straight from its catalog row — not only from the Add modal.
  const [removeTarget, setRemoveTarget] = useState<ModelSpec | null>(null);
  const [removing, setRemoving] = useState(false);

  // silent=true skips the loading spinner so background polling doesn't flicker
  // the table (the auto-refresh while a smoke-test is pending uses this). To keep
  // the table perfectly still between polls, only swap state when the fetched
  // payload actually differs — an identical poll result must not produce a new
  // array reference, or React re-renders the whole table for nothing.
  function refresh(silent = false) {
    if (!silent) setLoading(true);
    getModels()
      .then((c) => {
        const nextModels = c.models;
        const nextTiers = c.imageTiers ?? {};
        setModels((prev) =>
          JSON.stringify(prev) === JSON.stringify(nextModels) ? prev : nextModels
        );
        setTiers((prev) =>
          JSON.stringify(prev) === JSON.stringify(nextTiers) ? prev : nextTiers
        );
      })
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => {
        if (!silent) setLoading(false);
      });
  }

  useEffect(() => refresh(), []);
  // Whether an HF token is stored — gates the verify button for gated models.
  const [hfTokenSet, setHfTokenSet] = useState(false);
  useEffect(() => {
    getHfTokenStatus()
      .then((s) => setHfTokenSet(s.isSet || !!s.usingSharedFallback))
      .catch(() => setHfTokenSet(false));
  }, []);

  const tierNames = useMemo(() => Object.keys(tiers), [tiers]);

  // Any (model, tier) currently verifying? Drives the auto-refresh below so a
  // pending smoke-test flips to verified/incompatible without a manual reload —
  // the backend reconcile resolves it; we just re-fetch to show the result.
  const anyPending = useMemo(
    () =>
      models.some((m) =>
        Object.values(m.verifications ?? {}).some((v) => v.status === "pending")
      ),
    [models]
  );

  useEffect(() => {
    if (!anyPending) return;
    const t = setInterval(() => refresh(true), 30000);
    return () => clearInterval(t);
  }, [anyPending]);

  async function runBackfill() {
    try {
      const r = await backfillVerifications();
      notify({
        type: "success",
        content: `Backfill complete — ${r.promoted} model/image pair(s) marked verified from run history.`,
      });
      refresh();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    }
  }

  // Remove a custom (probe-added) model straight from its catalog row. Refreshes
  // the list on success AND on a 404 (already gone), so the row never lingers.
  async function removeCustom(model: ModelSpec) {
    setRemoving(true);
    try {
      await deleteCustomModel(model.id);
      notify({ type: "success", content: `Removed custom model ${model.displayName}.` });
      setRemoveTarget(null);
      refresh();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
      setRemoveTarget(null);
      refresh(); // reconcile even on error (e.g. already deleted) so the row clears
    } finally {
      setRemoving(false);
    }
  }

  // Self-healing: classify why a model failed and, if it needs a newer stack,
  // build the recommended image (if missing) then smoke-test the model on it.
  async function diagnoseAndHeal(model: ModelSpec) {
    try {
      const d = await diagnoseModel(model.id);
      const c = d.classification;
      if (!c.needsNewerStack) {
        notify({
          type: "info",
          content: `${model.displayName}: ${c.explanation} — a newer image won't help (${c.category}).`,
        });
        return;
      }
      if (!d.recommendedTier) {
        notify({ type: "info", content: `${model.displayName} is already on the newest image tier.` });
        return;
      }
      if (d.action === "build_then_smoke_test") {
        const b = await buildImage(d.recommendedTier);
        notify({
          type: "info",
          content:
            `${model.displayName} needs the '${d.recommendedTier}' stack (${c.explanation}). ` +
            `Started building that image (${b.buildId}). Once it finishes, click "verify" under the ${d.recommendedTier} column.`,
        });
        return;
      }
      notify({
        type: "info",
        content: `${model.displayName} needs the '${d.recommendedTier}' stack — verifying it there now…`,
      });
      await verify(model, d.recommendedTier);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    }
  }

  // Smoke-test a model on a specific image tier. The backend records the launch
  // as PENDING (persisted) and the reconcile loop resolves it to verified/
  // incompatible — so progress survives navigation and needs no live polling
  // here. We just launch and refresh; the page auto-refreshes while anything is
  // pending (see the anyPending effect) and the result appears on its own.
  async function verify(model: ModelSpec, tier: string, method: FinetuningType = "lora",
                        variant: LoraVariant = "lora") {
    // Busy/verification keys are namespaced by method AND variant so the LoRA,
    // QLoRA, full, freeze and per-variant (DoRA/…) controls in one tier cell spin +
    // resolve independently (mirrors the backend's (tier, method, variant) key).
    // Shared busyKeyFor so the button and this launcher can't drift.
    const key = busyKeyFor(model.id, tier, method, variant);
    setBusy((b) => ({ ...b, [key]: true }));
    try {
      const { jobName, imageTag } = await smokeTestModel(model.id, tier, method, "llama_factory", variant);
      const label = variant === "lora" ? METHOD_LABELS[method] : VARIANT_LABELS[variant];
      // full/freeze are full-weight runs on the bigger g6e card (~$2.80/hr) and
      // take longer than a LoRA smoke-test on g5 — call that out so a one-click
      // verify isn't a surprise on the bill.
      const fullWeight = method === "full" || method === "freeze";
      notify({
        type: "info",
        content:
          `Verifying ${model.displayName} (${label}) on image ${imageTag} (${jobName}). This runs a ` +
          (fullWeight
            ? `tiny real full-weight job on g6e.2xlarge (~$2.80/hr, ~15–25 min). `
            : `tiny real job (~10–15 min). `) +
          `You can leave this page — it keeps running and the result appears here automatically when done.`,
      });
      refresh(); // show the new "verifying…" state immediately
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setBusy((b) => ({ ...b, [key]: false }));
    }
  }

  // Smoke-test a model on the SERVERLESS engine (its own verification surface,
  // distinct from the image tiers). Same launch-and-let-reconcile-resolve flow as
  // verify(); the serverless badge reflects verifications["serverless"].
  async function verifyServerless(model: ModelSpec) {
    const key = `${model.id}:serverless`;
    setBusy((b) => ({ ...b, [key]: true }));
    try {
      const { jobName } = await smokeTestModel(model.id, undefined, "lora", "sagemaker_serverless");
      notify({
        type: "info",
        content:
          `Verifying ${model.displayName} on the SageMaker Serverless engine (${jobName}). This runs ` +
          `a tiny real managed job. You can leave this page — the result appears here when done.`,
      });
      refresh();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setBusy((b) => ({ ...b, [key]: false }));
    }
  }

  // Filter models (by text + verification state) FIRST, then group the survivors
  // by provider into expandable parent rows. Providers with no surviving model
  // drop out, so the tree always reflects the active filter.
  const [filterText, setFilterText] = useState("");
  const rows: Row[] = useMemo(() => {
    const t = filterText.trim().toLowerCase();
    const matches = models.filter((m) => {
      // A model with ANY verification in flight stays visible under every filter
      // except the explicit "untested" view — otherwise a model you just clicked
      // verify on (its own-image status flips to pending) would silently vanish
      // from a filtered view. The "pending" filter shows exactly those rows.
      const isPending = Object.values(m.verifications ?? {}).some((v) => v.status === "pending");
      if (statusFilter === "pending") {
        if (!isPending) return false;
      } else if (statusFilter !== "all" && ownStatus(m) !== statusFilter && !isPending) {
        return false;
      }
      if (!t) return true;
      const hay = `${m.displayName} ${m.id} ${m.family} ${m.provider} ${m.hfModelId} ${m.template} ${m.imageTag}`.toLowerCase();
      return hay.includes(t);
    });
    const byProvider = new Map<string, ModelSpec[]>();
    for (const m of matches) {
      const arr = byProvider.get(m.provider) ?? [];
      arr.push(m);
      byProvider.set(m.provider, arr);
    }
    const out: Row[] = [];
    for (const provider of [...byProvider.keys()].sort()) {
      const ms = byProvider.get(provider)!.sort((a, b) => a.paramsB - b.paramsB);
      out.push({ kind: "provider", key: `prov:${provider}`, provider, models: ms });
      for (const m of ms) out.push({ kind: "model", key: `model:${m.id}`, parentProvider: provider, ...m });
    }
    return out;
  }, [models, filterText, statusFilter]);

  const modelCount = rows.filter((r) => r.kind === "model").length;
  const providerCount = rows.filter((r) => r.kind === "provider").length;

  const { items, collectionProps, actions } = useCollection(rows, {
    expandableRows: {
      getId: (r) => r.key,
      getParentId: (r) => (r.kind === "model" ? `prov:${r.parentProvider}` : null),
    },
  });

  const providerRows = useMemo(() => rows.filter((r) => r.kind === "provider"), [rows]);
  const allExpanded =
    providerRows.length > 0 &&
    collectionProps.expandableRows!.expandedItems.length >= providerRows.length;

  // A single "Fine-tuning methods" column: one colored chip per parameterization
  // the model allows (LoRA always; QLoRA/Full/Freeze when allowed_methods include
  // them — Full/Freeze are gated to ≤2B by the backend). Chip color = rolled-up
  // status across image tiers (green verified / red incompatible / blue running /
  // grey untested); clicking a chip opens a popover with the per-tier breakdown,
  // provenance, cost note, and a verify/re-test button per tier. This replaces the
  // old per-tier columns (which stacked label+status+button × 4 methods × 2 tiers
  // in every cell — far too dense to scan).
  const methodsColumn = {
    id: "methods",
    header: "Fine-tuning methods",
    minWidth: 280,
    cell: (r: Row) => {
      if (r.kind !== "model") return "";
      // Gated models need an HF token (license-accepted) to download weights — the
      // smoke-test injects the stored token, so allow verify once a token is saved;
      // otherwise the chip popover explains why verification is disabled.
      const blockedByGate = r.gated && !hfTokenSet;
      const allowed = r.allowedMethods ?? ["lora", "qlora"];
      // LoRA is always offered (the default); the rest only when allowed.
      const methods = METHOD_ORDER.filter((m) => m === "lora" || allowed.includes(m));
      return (
        <SpaceBetween direction="horizontal" size="xs">
          {methods.map((m) => (
            <MethodChip
              key={m}
              model={r}
              method={m}
              tiers={tierNames}
              tierImages={tiers}
              busy={busy}
              blockedByGate={blockedByGate}
              hfTokenSet={hfTokenSet}
              onVerify={(tier, method, variant) => verify(r, tier, method, variant)}
            />
          ))}
        </SpaceBetween>
      );
    },
  };

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Models grouped by provider, and which Docker image each is proven to run on. Verification is tied to the image tag — a model proven on one image is not automatically proven on another, so an image rebuild resets that column to untested until re-proved. Manage the images themselves on the Images page."
          actions={
            <SpaceBetween direction="horizontal" size="xs">
              <Button iconName="refresh" onClick={() => refresh()} loading={loading}>
                Refresh
              </Button>
              <Button
                iconName={allExpanded ? "treeview-collapse" : "treeview-expand"}
                onClick={() => actions.setExpandedItems(allExpanded ? [] : providerRows)}
              >
                {allExpanded ? "Collapse all" : "Expand all"}
              </Button>
              <Button variant="primary" iconName="add-plus" onClick={() => setAddOpen(true)}>
                Add from Hugging Face
              </Button>
              <Button iconName="search" onClick={() => setServerlessFinderOpen(true)}>
                Find serverless-customizable models
              </Button>
              <Button onClick={runBackfill}>Backfill from run history</Button>
            </SpaceBetween>
          }
        >
          Model catalog
        </Header>
      }
    >
      <SpaceBetween size="m">
        <Table
          {...collectionProps}
          variant="container"
          loading={loading}
          items={items}
          trackBy="key"
          stickyHeader
          resizableColumns
          header={
            <Header
              counter={`(${modelCount} models · ${providerCount} providers)`}
              description={
                <SpaceBetween direction="horizontal" size="xs">
                  <Box variant="small" color="text-status-inactive">
                    Fine-tuning method chips — color = verification status:
                  </Box>
                  <Badge color="green">verified</Badge>
                  <Badge color="grey">untested</Badge>
                  <Badge color="red">incompatible</Badge>
                  <Badge color="blue">verifying…</Badge>
                </SpaceBetween>
              }
            >
              Models by provider
            </Header>
          }
          filter={
            <SpaceBetween direction="horizontal" size="xs">
              <TextFilter
                filteringText={filterText}
                filteringPlaceholder="Find by model, id, provider, family, or template"
                countText={`${modelCount} match${modelCount === 1 ? "" : "es"}`}
                onChange={({ detail }) => setFilterText(detail.filteringText)}
              />
              <Select
                selectedOption={
                  STATUS_FILTER_OPTIONS.find((o) => o.value === statusFilter) ?? STATUS_FILTER_OPTIONS[0]
                }
                onChange={({ detail }) => setStatusFilter(detail.selectedOption.value!)}
                options={STATUS_FILTER_OPTIONS}
              />
            </SpaceBetween>
          }
          columnDefinitions={[
            {
              id: "model",
              header: "Provider / model",
              width: 340,
              cell: (r) =>
                r.kind === "provider" ? (
                  <SpaceBetween direction="horizontal" size="xs">
                    <ProviderIcon provider={r.provider} />
                    <Box variant="strong">
                      {r.provider}{" "}
                      <Box variant="span" color="text-status-inactive">
                        ({r.models.length})
                      </Box>
                    </Box>
                  </SpaceBetween>
                ) : (
                  <SpaceBetween size="xxs">
                    <Box variant="strong">{r.displayName}</Box>
                    <Box fontSize="body-s" color="text-status-inactive">
                      {r.hfModelId}
                    </Box>
                    <SpaceBetween direction="horizontal" size="xxs">
                      {/* Family is a neutral metadata label. It used to be blue,
                          which collided with the blue "verifying…" status chip —
                          and since blue/green/grey/red are now reserved to MEAN a
                          verification status, family must sit outside that palette.
                          Grey (matching custom/gated) keeps color = status only. */}
                      <Badge color="grey">{r.family}</Badge>
                      {/* custom = user-added (was green, which read as "verified").
                          Neutral grey + a user icon disambiguates without color. */}
                      {r.custom && (
                        <Badge color="grey">
                          <Icon name="user-profile" size="small" /> custom
                        </Badge>
                      )}
                      {r.gated && (
                        <Popover
                          dismissButton={false}
                          position="top"
                          size="medium"
                          triggerType="custom"
                          content={
                            <SpaceBetween size="xs">
                              <span>
                                Requires a stored HF token AND accepting this model's license on
                                Hugging Face (per-model approval).{" "}
                                {hfTokenSet
                                  ? "A token is saved."
                                  : "No token saved yet — add one under Settings."}
                              </span>
                              <Link external href={`https://huggingface.co/${r.hfModelId}`}>
                                Request access to {r.hfModelId} ↗
                              </Link>
                            </SpaceBetween>
                          }
                        >
                          {/* gated = needs HF license/token. Lock icon carries the
                              meaning; red ONLY when no token is saved (actionable),
                              else neutral grey so it doesn't read as a failure. */}
                          <Badge color={hfTokenSet ? "grey" : "red"}>
                            <Icon name="lock-private" size="small" /> gated
                          </Badge>
                        </Popover>
                      )}
                      {/* Serverless support: backend includes "sagemaker_serverless"
                          in engines[] ONLY when the engine is enabled (Settings flag)
                          AND the model has a Public-Hub mapping. The badge color
                          reflects the serverless verification namespace. */}
                      {(r.engines ?? []).includes("sagemaker_serverless") && (
                        <Popover
                          header="SageMaker Serverless"
                          triggerType="custom"
                          dismissButton={false}
                          position="top"
                          content={
                            <SpaceBetween size="xs">
                              <span>
                                This model can be fine-tuned with the managed{" "}
                                <b>SageMaker Serverless</b> engine (SFT/DPO, LoRA) — pick it in
                                the Engine dropdown on the Fine-Tune page.{" "}
                                {r.verifications?.["serverless"]?.status === "verified"
                                  ? "A serverless run has completed for this model (verified)."
                                  : r.verifications?.["serverless"]?.status === "incompatible"
                                  ? "The last serverless run failed (see below)."
                                  : r.verifications?.["serverless"]?.status === "pending"
                                  ? ""
                                  : "Not yet verified on serverless — the first run proves it."}
                              </span>
                              {r.verifications?.["serverless"]?.status === "pending" && (
                                <span>
                                  A real managed serverless training job is running in the
                                  background. It keeps going if you leave the page — it resolves to
                                  verified/incompatible automatically (the page refreshes).
                                </span>
                              )}
                              {/* The serverless verification IS a real SageMaker training job
                                  (same as the image-tier smoke test) — surface its name + when +
                                  any failure reason, mirroring the LLaMA-Factory verify popover. */}
                              {(r.verifications?.["serverless"]?.status === "verified" ||
                                r.verifications?.["serverless"]?.status === "incompatible") &&
                                fmtVerifiedDate(r.verifications?.["serverless"]?.ts ?? null) && (
                                  <div>
                                    {r.verifications?.["serverless"]?.status === "verified"
                                      ? "Verified"
                                      : "Tested"}
                                    : {fmtVerifiedDate(r.verifications?.["serverless"]?.ts ?? null)}
                                  </div>
                                )}
                              {r.verifications?.["serverless"]?.jobName && (
                                <div style={{ wordBreak: "break-all" }}>
                                  <Box variant="awsui-key-label" display="inline">
                                    Serverless job
                                  </Box>
                                  : {r.verifications?.["serverless"]?.jobName}
                                </div>
                              )}
                              {r.verifications?.["serverless"]?.reason && (
                                <div>Reason: {r.verifications?.["serverless"]?.reason}</div>
                              )}
                              {r.verifications?.["serverless"]?.status !== "pending" && (
                                <Button
                                  variant="inline-link"
                                  loading={!!busy[`${r.id}:serverless`]}
                                  onClick={() => verifyServerless(r)}
                                >
                                  {r.verifications?.["serverless"]?.status === "verified"
                                    ? "Re-verify on serverless"
                                    : "Verify on serverless"}
                                </Button>
                              )}
                            </SpaceBetween>
                          }
                        >
                          {/* Serverless is another verifiable surface, so it follows
                              the SAME color=status contract as the method chips
                              (chipColor): grey untested · green verified · red
                              incompatible · blue running. The ⚡ icon marks it as the
                              serverless engine; color is never used for "available". */}
                          <Badge color={chipColor(r.verifications?.["serverless"]?.status ?? "untested")}>
                            ⚡ serverless
                            {r.verifications?.["serverless"]?.status === "verified"
                              ? " ✓"
                              : r.verifications?.["serverless"]?.status === "incompatible"
                              ? " ⚠"
                              : r.verifications?.["serverless"]?.status === "pending"
                              ? " …"
                              : ""}
                          </Badge>
                        </Popover>
                      )}
                    </SpaceBetween>
                    {/* Shown only when the model is BROKEN on the image it actually
                        runs on (its imageTag tier). NOT when some OTHER tier it
                        doesn't use is incompatible (e.g. a stale `latest` failure on
                        a model pinned to `stable`) — that's not a failure the user
                        needs to heal, and diagnose() inspects this same own-tier, so
                        triggering on a different tier produced a "no failure reason"
                        dead-end. Mirrors the method-chip rollup (verified-on-its-image
                        wins). diagnose() only ever recommends a NEWER LLaMA-Factory
                        image, so it's irrelevant to serverless verification anyway. */}
                    {r.verifications?.[r.imageTag]?.status === "incompatible" && (
                      <Popover
                        header="Why did this fail?"
                        triggerType="custom"
                        dismissButton={false}
                        position="top"
                        size="medium"
                        content={
                          <SpaceBetween size="xs">
                            <Box variant="small">
                              This model failed verification on an image. This checks <b>why</b>:
                            </Box>
                            <Box variant="small" color="text-status-inactive">
                              • If the failure is just that the (frozen) training image is too old
                              for this model, it moves the model to a newer image and re-tests it
                              there — building that image first if it doesn't exist yet (a real
                              build + a tiny smoke-test job).
                              <br />
                              • If the failure isn't an image problem (e.g. out-of-memory or a
                              transient capacity error), it tells you a newer image won't help and
                              does nothing billable.
                            </Box>
                          </SpaceBetween>
                        }
                      >
                        <Button variant="inline-link" iconName="gen-ai" onClick={() => diagnoseAndHeal(r)}>
                          Why did this fail?
                        </Button>
                      </Popover>
                    )}
                    {/* Custom models (probe-added) can be removed right here — the
                        round trip add->remove must be reachable from the catalog,
                        not buried in the Add-from-HF modal. */}
                    {r.custom && (
                      <Button
                        variant="inline-link"
                        iconName="remove"
                        onClick={() => setRemoveTarget(r)}
                      >
                        Remove
                      </Button>
                    )}
                  </SpaceBetween>
                ),
            },
            {
              id: "params",
              header: "Size",
              width: 90,
              cell: (r) => (r.kind === "model" ? `${r.paramsB}B` : ""),
            },
            {
              id: "template",
              header: "Template",
              width: 130,
              cell: (r) => (r.kind === "model" ? r.template : ""),
            },
            {
              id: "imageTag",
              header: "Runs on",
              width: 110,
              // Plain muted text, NOT a badge — the image tier a model's own runs
              // use is a neutral label, and the dark grey Cloudscape badge read like
              // a status chip (competing with the verification chips beside it).
              cell: (r) =>
                r.kind === "model" ? (
                  <Box color="text-status-inactive">{r.imageTag}</Box>
                ) : (
                  ""
                ),
            },
            methodsColumn,
          ]}
          empty={<Box textAlign="center">No models match.</Box>}
        />
      </SpaceBetween>

      <AddFromHuggingFace
        visible={addOpen}
        onDismiss={() => setAddOpen(false)}
        tiers={tiers}
        onChanged={() => {
          // An explicit add fired onChanged — make the result obvious: do a
          // non-silent refresh (spinner) and close the modal so the new row is
          // plainly present, instead of a silent background reload that looks
          // like nothing happened.
          refresh(false);
          setAddOpen(false);
        }}
      />

      <ServerlessFinder
        visible={serverlessFinderOpen}
        onDismiss={() => setServerlessFinderOpen(false)}
        // Applying a tag changes a model's engines → silent refresh so the ⚡
        // serverless badge appears without closing the finder (the user may apply
        // several in one session).
        onChanged={() => refresh(true)}
      />

      <Modal
        visible={removeTarget !== null}
        onDismiss={() => setRemoveTarget(null)}
        header="Remove custom model"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="link" onClick={() => setRemoveTarget(null)} disabled={removing}>
                Cancel
              </Button>
              <Button
                variant="primary"
                loading={removing}
                onClick={() => removeTarget && removeCustom(removeTarget)}
              >
                Remove
              </Button>
            </SpaceBetween>
          </Box>
        }
      >
        Remove <b>{removeTarget?.displayName}</b> ({removeTarget?.hfModelId}) from the catalog?
        This only removes it from this app; any SageMaker resources are unaffected. You can
        re-add it any time from “Add from Hugging Face”.
      </Modal>
    </ContentLayout>
  );
}
