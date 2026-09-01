// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Link from "@cloudscape-design/components/link";
import Box from "@cloudscape-design/components/box";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Grid from "@cloudscape-design/components/grid";
import Table from "@cloudscape-design/components/table";
import StatusIndicator from "@cloudscape-design/components/status-indicator";

import Input from "@cloudscape-design/components/input";
import FormField from "@cloudscape-design/components/form-field";
import Toggle from "@cloudscape-design/components/toggle";
import Modal from "@cloudscape-design/components/modal";
import Select from "@cloudscape-design/components/select";
import Badge from "@cloudscape-design/components/badge";
import { ProviderIcon } from "./providerIcon";

import {
  checkConfig,
  getAgentModels,
  getConfig,
  getHfTokenStatus,
  getLimits,
  getSamplesStatus,
  setSamplesEnabled,
  type SamplesStatus,
  putAgentModels,
  putConfig,
  resetJobHistory,
  setHfToken,
  type AgentModelsView,
  type AwsConfigView,
  type Limits,
  type PreflightResult,
} from "./api";
import { useNotify, errText } from "./notifications";

// Read-only environment view + preflight. In the hosted deployment the AWS
// config is fixed by CDK at deploy time (Lambda env vars) — editing it here
// could only break the running app, so the fields are display-only. The
// "Check environment" preflight is the useful diagnostic: it verifies the
// Lambda role can actually reach the bucket / role / training image.
const FIELDS: { key: keyof AwsConfigView; label: string }[] = [
  { key: "region", label: "AWS region" },
  { key: "account", label: "Account id" },
  { key: "bucket", label: "S3 bucket" },
  { key: "roleArn", label: "SageMaker execution role ARN" },
  { key: "imageUri", label: "Training image URI" },
];

export function SettingsPage() {
  const { notify } = useNotify();
  const [cfg, setCfg] = useState<AwsConfigView | null>(null);
  const [limits, setLimits] = useState<Limits | null>(null);
  const [checking, setChecking] = useState(false);
  const [preflight, setPreflight] = useState<PreflightResult | null>(null);

  // "Reset job history" — soft-hides all existing SageMaker jobs from listings.
  const [resetModalOpen, setResetModalOpen] = useState(false);

  // Sample ("golden") runs — opt-in showcase for new users (overlay, not a copy).
  const [samples, setSamples] = useState<SamplesStatus | null>(null);
  const [samplesSaving, setSamplesSaving] = useState(false);
  async function toggleSamples(on: boolean) {
    setSamplesSaving(true);
    try {
      setSamples(await setSamplesEnabled(on));
      notify({
        type: "success",
        content: on
          ? "Sample races enabled — see them on the Races, Datasets and Leaderboard pages (read-only; clone one to make it your own)."
          : "Sample races hidden.",
      });
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setSamplesSaving(false);
    }
  }
  const [resetting, setResetting] = useState(false);

  // HF token (write-only): we only ever learn whether one is set.
  const [hfSet, setHfSet] = useState<boolean | null>(null);
  // No own token, but the platform's shared fallback token is covering for them.
  const [hfFallback, setHfFallback] = useState(false);
  const [hfDraft, setHfDraft] = useState("");
  const [hfSaving, setHfSaving] = useState(false);

  // Per-agent model selection (Settings → AI agents). `agentSaving` holds the
  // role key currently being saved so only that row's Select shows a spinner.
  const [agentModels, setAgentModels] = useState<AgentModelsView | null>(null);
  const [agentSaving, setAgentSaving] = useState<string | null>(null);
  // Provider→model is a two-step picker (mirrors the Baseline picker). The
  // provider dropdown is local UI state: switching provider re-scopes the model
  // dropdown WITHOUT saving (nothing persists until a model is chosen). Keyed by
  // role; absent → derive the provider from the role's currently-selected model.
  const [agentProvider, setAgentProvider] = useState<Record<string, string>>({});

  async function saveAgentModel(roleKey: string, modelKey: string) {
    setAgentSaving(roleKey);
    try {
      const next = await putAgentModels({ [roleKey]: modelKey });
      setAgentModels(next);
      // Clear the transient provider browse-state so the row re-derives it from
      // the freshly-saved selection (keeps provider + model in sync after save).
      setAgentProvider((m) => {
        const { [roleKey]: _drop, ...rest } = m;
        return rest;
      });
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setAgentSaving(null);
    }
  }

  // Serverless engine toggle (dark-launch switch; flips without a redeploy).
  const [serverlessSaving, setServerlessSaving] = useState(false);
  async function toggleServerless(on: boolean) {
    setServerlessSaving(true);
    try {
      const next = await putConfig({ enableSagemakerServerless: on });
      setCfg(next);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setServerlessSaving(false);
    }
  }

  useEffect(() => {
    getConfig()
      .then(setCfg)
      .catch((e) => notify({ type: "error", content: errText(e) }));
    getLimits()
      .then(setLimits)
      .catch(() => {/* limits are informational; ignore fetch errors */});
    getHfTokenStatus()
      .then((s) => {
        setHfSet(s.isSet);
        setHfFallback(!!s.usingSharedFallback);
      })
      .catch(() => setHfSet(null));
    getAgentModels()
      .then(setAgentModels)
      .catch((e) => notify({ type: "error", content: errText(e) }));
    getSamplesStatus()
      .then(setSamples)
      .catch(() => {/* samples are optional onboarding; ignore fetch errors */});
  }, []);

  async function saveHfToken() {
    if (!hfDraft.trim()) return;
    setHfSaving(true);
    try {
      const s = await setHfToken(hfDraft);
      setHfSet(s.isSet);
      if (s.isSet) setHfFallback(false); // now on their own token
      setHfDraft("");
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setHfSaving(false);
    }
  }

  async function runCheck() {
    setChecking(true);
    setPreflight(null);
    try {
      setPreflight(await checkConfig());
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setChecking(false);
    }
  }

  async function doResetJobHistory() {
    setResetting(true);
    try {
      await resetJobHistory();
      setResetModalOpen(false);
      notify({
        type: "success",
        content:
          "Job history reset. Existing runs are now hidden from listings; new runs will appear as usual.",
      });
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setResetting(false);
    }
  }

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="The AWS environment this app runs in. Configured at deploy time and shown here for reference. Run the environment check to confirm the app can reach its AWS resources."
        >
          Settings
        </Header>
      }
    >
      <SpaceBetween size="l">
        <Container
          header={
            <Header
              variant="h2"
              actions={
                <Button onClick={runCheck} loading={checking} iconName="status-positive">
                  Check environment
                </Button>
              }
              description="Set by the deployment (CDK). Read-only."
            >
              AWS environment
            </Header>
          }
        >
          {cfg ? (
            <ColumnLayout columns={2} variant="text-grid">
              {FIELDS.map((f) => (
                <div key={f.key}>
                  <Box variant="awsui-key-label">{f.label}</Box>
                  <Box>{(cfg[f.key] as string) || "—"}</Box>
                </div>
              ))}
            </ColumnLayout>
          ) : (
            <Box>Loading…</Box>
          )}
        </Container>

        {limits && (
          <Container
            header={
              <Header variant="h2" description="Enforced when launching a fine-tune run to prevent runaway cost. Set by the deployment.">
                Cost guardrails
              </Header>
            }
          >
            <ColumnLayout columns={3} variant="text-grid">
              <div>
                <Box variant="awsui-key-label">Max models per run</Box>
                <Box>{limits.maxModelsPerRace}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Max concurrent runs (per user)</Box>
                <Box>{limits.maxConcurrentRaces}</Box>
              </div>
              {limits.maxGlobalConcurrentRaces ? (
                <div>
                  <Box variant="awsui-key-label">Max concurrent runs (all users)</Box>
                  <Box>{limits.maxGlobalConcurrentRaces}</Box>
                </div>
              ) : null}
              <div>
                <Box variant="awsui-key-label">Allowed instance types</Box>
                <Box>{limits.allowedInstanceTypes.join(", ")}</Box>
              </div>
            </ColumnLayout>
          </Container>
        )}

        <Container
          header={
            <Header
              variant="h2"
              description="Choose which fine-tuning engines are available in the run picker."
            >
              Training engines
            </Header>
          }
        >
          <SpaceBetween size="m">
            <Box>
              <b>LLaMA-Factory</b> (self-hosted, default) is always on — every model,
              every objective (SFT/DPO/KTO) and method (LoRA/QLoRA), on GPU instances
              you manage. It cannot be disabled.
            </Box>
            <Toggle
              checked={!!cfg?.enableSagemakerServerless}
              disabled={serverlessSaving || !cfg}
              onChange={({ detail }) => toggleServerless(detail.checked)}
            >
              Enable <b>SageMaker Serverless</b> engine
            </Toggle>
            <Box variant="small" color="text-body-secondary">
              Fully managed, no infrastructure — pay per token. Supports <b>SFT</b> and{" "}
              <b>DPO</b> with <b>LoRA</b> on the 11 models with a SageMaker Public-Hub
              mapping (Qwen3, Qwen2.5, DeepSeek-R1-Distill, Llama-3.2-3B, GPT-OSS-20B).
              When on, an <b>Engine</b> picker appears on the Fine-Tune page for those
              models. Turning it off hides serverless everywhere and rejects serverless
              launches. Takes effect immediately (no redeploy).
            </Box>
            {serverlessSaving && <StatusIndicator type="loading">Saving…</StatusIndicator>}
          </SpaceBetween>
        </Container>

        <Container
          header={
            <Header
              variant="h2"
              description="Which Bedrock model each of the app's AI agents uses. Changes are saved to config (no redeploy) — except the agents marked 'after redeploy', which run on the AgentCore runtime."
            >
              AI agents
            </Header>
          }
        >
          {agentModels ? (
            <SpaceBetween size="l">
              {agentModels.roles.map((role) => {
                // The model currently saved for this role + its provider.
                const selectedModel = agentModels.models.find((m) => m.key === role.selectedKey);
                const selectedProvider = selectedModel?.provider ?? null;
                // Provider→model is two-step: the provider dropdown is browse-state
                // (re-scopes the model list without saving); when absent, default to
                // the saved model's provider so the row opens on the real selection.
                const browseProvider = agentProvider[role.key] ?? selectedProvider;
                const providers = [...new Set(agentModels.models.map((m) => m.provider))];
                const isDefault = role.selectedKey === agentModels.default;
                const saving = agentSaving === role.key;
                return (
                  <FormField
                    key={role.key}
                    label={
                      <SpaceBetween direction="horizontal" size="xs">
                        <span>{role.label}</span>
                        {role.deployTime && (
                          <Badge color="grey">applies after agent redeploy</Badge>
                        )}
                        {!isDefault && <Badge color="blue">overridden</Badge>}
                      </SpaceBetween>
                    }
                    description={role.description}
                  >
                    {/* Provider → Model, mirroring the Baseline picker. Picking a
                        provider only re-scopes the model dropdown; choosing a model
                        is what saves the override. Provider names are short, so it
                        gets a narrow 1/3 column and the model picker the wider 2/3. */}
                    <Grid gridDefinition={[{ colspan: 4 }, { colspan: 8 }]}>
                      <Select
                        selectedOption={
                          browseProvider
                            ? {
                                value: browseProvider,
                                label: browseProvider,
                                iconSvg: <ProviderIcon provider={browseProvider} size={16} />,
                              }
                            : null
                        }
                        disabled={saving}
                        placeholder="Choose provider"
                        options={providers.map((p) => ({
                          value: p,
                          label: p,
                          iconSvg: <ProviderIcon provider={p} size={16} />,
                        }))}
                        onChange={({ detail }) =>
                          setAgentProvider((m) => ({ ...m, [role.key]: detail.selectedOption.value! }))
                        }
                      />
                      <Select
                        selectedOption={
                          // Show the saved model only while browsing its provider;
                          // after switching provider, force an explicit model pick.
                          selectedModel && browseProvider === selectedProvider
                            ? {
                                value: selectedModel.key,
                                label: selectedModel.label,
                                iconSvg: <ProviderIcon provider={selectedModel.provider} size={16} />,
                                description: selectedModel.modelId,
                              }
                            : null
                        }
                        disabled={!browseProvider || saving}
                        loadingText="Saving…"
                        statusType={saving ? "loading" : "finished"}
                        placeholder={browseProvider ? "Choose model" : "Pick a provider first"}
                        options={agentModels.models
                          .filter((m) => m.provider === browseProvider)
                          .map((m) => ({
                            value: m.key,
                            label: m.label,
                            iconSvg: <ProviderIcon provider={m.provider} size={16} />,
                            description: m.modelId,
                          }))}
                        onChange={({ detail }) => saveAgentModel(role.key, detail.selectedOption.value!)}
                      />
                    </Grid>
                  </FormField>
                );
              })}
              <Alert type="info" header="In-process vs. deployed agents">
                The advisor, self-heal classifier and evaluation judge call Bedrock
                directly — a change here applies on their next run. The dataset agents
                and reward-prompt author run on the AgentCore runtime, so their model
                only changes after that agent is redeployed (the selection is saved
                and threaded through now). The evaluation judge can still be overridden
                per eval run; this sets its default.
              </Alert>
            </SpaceBetween>
          ) : (
            <Box>Loading…</Box>
          )}
        </Container>

        <Container
          header={
            <Header
              variant="h2"
              description="Unlocks gated models (Llama, Mistral, Gemma). Stored encrypted in AWS Secrets Manager and injected into training/eval jobs — never displayed back."
            >
              HuggingFace token
            </Header>
          }
        >
          <SpaceBetween size="m">
            <StatusIndicator type={hfSet ? "success" : hfFallback ? "warning" : "info"}>
              {hfSet === null
                ? "Status unknown"
                : hfSet
                ? "Your token is set"
                : hfFallback
                ? "No token of your own — using the platform's shared token (set yours below)"
                : "No token set — gated models are disabled"}
            </StatusIndicator>
            <FormField
              label={hfSet ? "Replace token" : "Set token"}
              description="Paste your hf_… token."
              constraintText={
                <span>
                  Don't have one? Create a token at{" "}
                  <Link external href="https://huggingface.co/settings/tokens">
                    huggingface.co/settings/tokens
                  </Link>{" "}
                  → "New token" → type <b>Read</b> (a read token is enough to download
                  models &amp; datasets) → copy the <code>hf_…</code> value and paste it here.
                </span>
              }
            >
              <Input
                value={hfDraft}
                type="password"
                placeholder="hf_xxxxxxxx"
                onChange={({ detail }) => setHfDraft(detail.value)}
              />
            </FormField>
            <Button variant="primary" loading={hfSaving} disabled={!hfDraft.trim()} onClick={saveHfToken}>
              Save token
            </Button>
            <Alert type="info" header="A token alone isn't enough for gated models">
              Gated models (Llama, Gemma, Mistral) also require <b>per-model license
              approval</b> on Hugging Face. After saving your token, open each gated model you
              want on huggingface.co — signed in with the <b>same account that owns this
              token</b> — and click <b>“Request access” / accept the license</b>. Approval is
              usually granted in minutes. Without it, runs fail at the weight-download step
              (the model shows “access denied” on the catalog). Your token is private to your
              account — each user sets their own.
            </Alert>
          </SpaceBetween>
        </Container>

        <Container
          header={
            <Header
              variant="h2"
              description="Hide existing fine-tuning runs and SageMaker jobs from all listings — a soft 'start blank'."
            >
              Job history
            </Header>
          }
        >
          <SpaceBetween size="m">
            <Box variant="small" color="text-body-secondary">
              SageMaker job records can't be deleted, so this stamps a cutoff at
              the current time: jobs created before now are hidden from the Races
              and Leaderboard listings, while new races appear as usual. It does
              not stop any in-progress jobs or delete data — it only changes what
              the UI shows. The action can't be undone (older jobs stay hidden).
            </Box>
            <Button iconName="undo" onClick={() => setResetModalOpen(true)}>
              Reset job history…
            </Button>
          </SpaceBetween>
        </Container>

        <Container
          header={
            <Header
              variant="h2"
              description="Start with a curated example instead of a blank app — a real dataset, fine-tuned models, and a populated leaderboard to explore."
            >
              Sample races
            </Header>
          }
        >
          <SpaceBetween size="m">
            <Box variant="small" color="text-body-secondary">
              Importing sample races doesn't copy or train anything — it just shows a
              shared, read-only showcase on your <b>Races</b>, <b>Datasets</b> and{" "}
              <b>Leaderboard</b> pages (tagged <i>sample</i>). Clone one to turn it into
              your own editable race. Turn it off any time to hide them.
              {samples != null && samples.sampleCount > 0 && (
                <> {samples.sampleCount} sample run{samples.sampleCount === 1 ? "" : "s"} available.</>
              )}
            </Box>
            {samples?.enabled ? (
              <Button
                iconName="close"
                loading={samplesSaving}
                onClick={() => toggleSamples(false)}
              >
                Hide sample runs
              </Button>
            ) : (
              <Button
                variant="primary"
                iconName="add-plus"
                loading={samplesSaving}
                disabled={samples != null && samples.sampleCount === 0}
                onClick={() => toggleSamples(true)}
              >
                Import sample runs
              </Button>
            )}
          </SpaceBetween>
        </Container>

        {preflight && (
          <Container
            header={
              <Header
                variant="h2"
                description={
                  preflight.ok
                    ? "All checks passed — the app can launch jobs."
                    : "Some checks failed — the app may not be able to launch jobs."
                }
              >
                Environment check{" "}
                <StatusIndicator type={preflight.ok ? "success" : "error"}>
                  {preflight.ok ? "ready" : "not ready"}
                </StatusIndicator>
              </Header>
            }
          >
            <Table
              variant="embedded"
              columnDefinitions={[
                {
                  id: "status",
                  header: "",
                  cell: (c) => (
                    <StatusIndicator type={c.ok ? "success" : "error"}>
                      {c.ok ? "OK" : "Fail"}
                    </StatusIndicator>
                  ),
                  width: 90,
                },
                { id: "check", header: "Check", cell: (c) => c.check },
                { id: "detail", header: "Detail", cell: (c) => c.detail },
              ]}
              items={preflight.checks}
              empty="No checks"
            />
          </Container>
        )}
      </SpaceBetween>

      <Modal
        visible={resetModalOpen}
        onDismiss={() => setResetModalOpen(false)}
        header="Reset job history?"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="link" onClick={() => setResetModalOpen(false)} disabled={resetting}>
                Cancel
              </Button>
              <Button variant="primary" loading={resetting} onClick={doResetJobHistory}>
                Reset history
              </Button>
            </SpaceBetween>
          </Box>
        }
      >
        <SpaceBetween size="s">
          <Box>
            This hides every existing race and SageMaker job from the Races and
            Leaderboard listings. Their data and S3 artifacts are not deleted, and
            in-progress jobs keep running — they're just no longer shown.
          </Box>
          <Box>
            <b>This can't be undone.</b> Only runs created after now will appear.
          </Box>
        </SpaceBetween>
      </Modal>
    </ContentLayout>
  );
}
