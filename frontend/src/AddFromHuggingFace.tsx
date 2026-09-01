// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useRef, useState } from "react";
import Modal from "@cloudscape-design/components/modal";
import Tabs from "@cloudscape-design/components/tabs";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Input from "@cloudscape-design/components/input";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Box from "@cloudscape-design/components/box";
import Badge from "@cloudscape-design/components/badge";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Table from "@cloudscape-design/components/table";
import StatusIndicator from "@cloudscape-design/components/status-indicator";

import {
  addCustomModel,
  deleteCustomModel,
  getJobStatus,
  newModelsForImage,
  probeModel,
  smokeTestModel,
  type NewModelsResult,
  type ProbedModel,
} from "./api";
import { useNotify, errText } from "./notifications";

// Two front doors to one engine (probe → addCustomModel / smokeTestModel),
// surfaced as tabs in a single modal launched from the Catalog page:
//   • "By model ID"     — manual: type an exact HF id, probe, review, add.
//   • "Newly supported" — discovery: diff the newest image's architectures
//                          against an older one to suggest models the latest
//                          LLaMA-Factory release just unlocked, then probe+add.
// onChanged() lets the Catalog refresh after any add/remove.
export function AddFromHuggingFace({
  visible,
  onDismiss,
  tiers,
  onChanged,
}: {
  visible: boolean;
  onDismiss: () => void;
  tiers: Record<string, string>;
  onChanged: () => void;
}) {
  const [activeTab, setActiveTab] = useState("by-id");
  // Bumped whenever either tab onboards/removes a model so the ByModelId
  // "Onboarded models" table reloads even when the add happened on the
  // discovery tab (otherwise a just-added model is invisible there).
  const [customVersion, setCustomVersion] = useState(0);
  const bumpCustom = () => {
    setCustomVersion((v) => v + 1);
    onChanged();
  };

  return (
    <Modal
      visible={visible}
      onDismiss={onDismiss}
      header="Add models from Hugging Face"
      size="large"
    >
      <Tabs
        activeTabId={activeTab}
        onChange={({ detail }) => setActiveTab(detail.activeTabId)}
        tabs={[
          {
            id: "by-id",
            label: "By model ID",
            content: <ByModelId onChanged={bumpCustom} customVersion={customVersion} />,
          },
          {
            id: "discover",
            label: "Newly supported",
            content: <NewlySupported tiers={tiers} onChanged={bumpCustom} />,
          },
        ]}
      />
    </Modal>
  );
}

// --- Tab 1: manual probe → review → add (was ModelOnboard on Settings) -------
function ByModelId({
  onChanged,
  customVersion,
}: {
  onChanged: () => void;
  customVersion: number;
}) {
  const { notify } = useNotify();
  const [repo, setRepo] = useState("");
  const [probing, setProbing] = useState(false);
  const [probed, setProbed] = useState<ProbedModel | null>(null);
  const [custom, setCustom] = useState<
    { id: string; displayName: string; hfModelId: string; template: string }[]
  >([]);

  function loadCustom() {
    fetch("/api/models/custom")
      .then((r) => r.json())
      .then((d) => setCustom(d.models ?? []))
      .catch(() => {});
  }
  // Reload on mount and whenever a model is onboarded/removed from either tab
  // (customVersion is bumped by the parent), so the table never goes stale.
  useEffect(loadCustom, [customVersion]);

  async function doProbe() {
    setProbing(true);
    setProbed(null);
    try {
      setProbed(await probeModel(repo.trim()));
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setProbing(false);
    }
  }

  async function doAdd() {
    if (!probed) return;
    try {
      await addCustomModel(probed);
      notify({
        type: "success",
        content: `Added ${probed.displayName} — it's now selectable on the Fine-tune page.`,
      });
      setProbed(null);
      setRepo("");
      loadCustom();
      onChanged();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    }
  }

  const [smoking, setSmoking] = useState(false);
  // Track the smoke-test poll so it's cleared on unmount/modal-close (otherwise it
  // keeps hitting the API every 15s forever + setStates on an unmounted component).
  const smokePoll = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => () => {
    if (smokePoll.current) clearInterval(smokePoll.current);
  }, []);
  async function doSmokeTest() {
    if (!probed) return;
    setSmoking(true);
    try {
      // The smoke test needs the model in the catalog first; add it, then launch.
      await addCustomModel(probed);
      loadCustom();
      onChanged();
      const { jobName } = await smokeTestModel(probed.id);
      notify({ type: "info", content: `Smoke test launched (${jobName}). Polling…` });
      let attempts = 0;
      const MAX_ATTEMPTS = 240; // ~60 min @ 15s — then stop polling, leave it added
      if (smokePoll.current) clearInterval(smokePoll.current);
      smokePoll.current = setInterval(async () => {
        attempts += 1;
        try {
          const st = await getJobStatus(jobName);
          if (["Completed", "Failed", "Stopped"].includes(st.status)) {
            if (smokePoll.current) clearInterval(smokePoll.current);
            smokePoll.current = null;
            setSmoking(false);
            if (st.status === "Completed") {
              notify({
                type: "success",
                content: `Smoke test passed — ${probed.displayName} trains cleanly with template "${probed.template}".`,
              });
            } else {
              notify({
                type: "error",
                content: `Smoke test ${st.status}: ${st.failureReason ?? "see logs"}. The model stays added but isn't verified.`,
              });
            }
            onChanged();
          } else if (attempts >= MAX_ATTEMPTS) {
            if (smokePoll.current) clearInterval(smokePoll.current);
            smokePoll.current = null;
            setSmoking(false);
            notify({
              type: "info",
              content: "Smoke test is taking a while — check the model's status on the catalog later. It stays added.",
            });
          }
        } catch {
          /* transient — keep polling until the attempt cap */
        }
      }, 15000);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
      setSmoking(false);
    }
  }

  async function removeModel(id: string) {
    try {
      await deleteCustomModel(id);
      loadCustom();
      onChanged();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    }
  }

  return (
    <SpaceBetween size="m">
      <Box variant="p" color="text-status-inactive">
        Add a model by Hugging Face id. Facts (template, size, context, instance) are
        auto-derived; an optional smoke test confirms it trains.
      </Box>

      <FormField label="Hugging Face model id" description="e.g. Qwen/Qwen3-4B-Instruct-2507">
        <SpaceBetween direction="horizontal" size="xs">
          <Input
            value={repo}
            placeholder="org/model-name"
            onChange={({ detail }) => setRepo(detail.value)}
          />
          <Button loading={probing} disabled={!repo.trim()} onClick={doProbe}>
            Probe
          </Button>
        </SpaceBetween>
      </FormField>

      {probed && (
        <Container header={<Header variant="h3">{probed.displayName}</Header>}>
          <SpaceBetween size="s">
            <ColumnLayout columns={4} variant="text-grid">
              <div>
                <Box variant="awsui-key-label">Template</Box>
                <Box>
                  {probed.template ?? "—"}{" "}
                  {probed.templateKnown ? (
                    <Badge color="green">known</Badge>
                  ) : (
                    <Badge color="red">unknown</Badge>
                  )}
                </Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Params</Box>
                <Box>{probed.paramsB}B</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Cutoff</Box>
                <Box>{probed.defaultCutoffLen}</Box>
              </div>
              <div>
                <Box variant="awsui-key-label">Instance</Box>
                <Box>{probed.suggestedInstance}</Box>
              </div>
            </ColumnLayout>
            <Box variant="small" color="text-status-inactive">
              {probed.templateMatch} · arch {probed.architectures.join(", ") || "?"} ·{" "}
              {probed.gated ? "gated (needs HF token)" : "ungated"}
            </Box>
            {!probed.templateKnown && (
              <Alert type="warning">
                No known chat template matched this architecture, so it can't be added safely
                (training would fail at config parse). Supported templates:{" "}
                {probed.knownTemplates.slice(0, 12).join(", ")}…
              </Alert>
            )}
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="primary" disabled={!probed.templateKnown} onClick={doAdd}>
                Add to catalog
              </Button>
              <Button
                iconName="status-in-progress"
                loading={smoking}
                disabled={!probed.templateKnown}
                onClick={doSmokeTest}
              >
                Add &amp; smoke-test (tiny real job)
              </Button>
            </SpaceBetween>
          </SpaceBetween>
        </Container>
      )}

      <Table
        variant="embedded"
        trackBy="id"
        header={<Header variant="h3" counter={`(${custom.length})`}>Onboarded models</Header>}
        columnDefinitions={[
          { id: "name", header: "Model", cell: (m) => m.displayName },
          { id: "hf", header: "HF id", cell: (m) => m.hfModelId },
          { id: "template", header: "Template", cell: (m) => m.template },
          {
            id: "actions",
            header: "",
            cell: (m) => (
              <Button variant="inline-link" iconName="remove" onClick={() => removeModel(m.id)}>
                Remove
              </Button>
            ),
          },
        ]}
        items={custom}
        empty={
          <Box textAlign="center" color="text-status-inactive" padding="s">
            No onboarded models yet. Probe a Hugging Face id above to add one.
          </Box>
        }
      />

      <Box variant="small" color="text-status-inactive">
        <StatusIndicator type="info">
          Onboarded models appear on the Fine-tune page alongside the built-in catalog.
        </StatusIndicator>
      </Box>
    </SpaceBetween>
  );
}

// --- Tab 2: image-diff discovery (was findNewModels on the Catalog page) -----
function NewlySupported({
  tiers,
  onChanged,
}: {
  tiers: Record<string, string>;
  onChanged: () => void;
}) {
  const { notify } = useNotify();
  const [discovering, setDiscovering] = useState(false);
  const [discovery, setDiscovery] = useState<NewModelsResult | null>(null);
  const [onboarded, setOnboarded] = useState<Record<string, string>>({}); // repo → status

  // Diff the newest image's supported architectures against an older one and
  // suggest models the new LLaMA-Factory release added. Targets the newest tier.
  async function findNewModels() {
    setDiscovering(true);
    try {
      const tags = Object.values(tiers);
      if (tags.length === 0) throw new Error("no image tiers configured");
      const newest = tags.slice().sort((a, b) => {
        const va = a.split(".").map(Number);
        const vb = b.split(".").map(Number);
        for (let i = 0; i < Math.max(va.length, vb.length); i++)
          if ((va[i] || 0) !== (vb[i] || 0)) return (va[i] || 0) - (vb[i] || 0);
        return 0;
      })[tags.length - 1];
      setDiscovery(await newModelsForImage(newest));
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setDiscovering(false);
    }
  }

  // Probe a suggested HF repo and, if its template is recognized, add it to the
  // catalog pinned to the discovery's image tier (NOT verified — user smoke-tests).
  async function probeAndOnboard(repo: string, imageTier: string) {
    setOnboarded((o) => ({ ...o, [repo]: "probing" }));
    try {
      const probed = await probeModel(repo);
      if (!probed.template || !probed.templateKnown) {
        setOnboarded((o) => ({ ...o, [repo]: "no recognized template — onboard manually" }));
        return;
      }
      await addCustomModel({ ...probed, imageTag: imageTier });
      setOnboarded((o) => ({ ...o, [repo]: "added — smoke-test it on the catalog" }));
      onChanged();
      // Re-run discovery so the just-added model drops off the "Newly supported"
      // list (the backend now filters out repos already in the catalog).
      findNewModels();
    } catch (e) {
      let msg = e instanceof Error ? e.message : String(e);
      if (/404|not found|could not fetch HF/i.test(msg)) {
        msg = "not found on Hugging Face (the architecture may be ahead of public weights)";
      } else if (/NetworkError|Failed to fetch/i.test(msg)) {
        msg = "couldn't reach the server — retry";
      }
      setOnboarded((o) => ({ ...o, [repo]: `failed: ${msg}` }));
    }
  }

  return (
    <SpaceBetween size="m">
      <Box variant="p" color="text-status-inactive">
        Diff the newest Docker image's supported architectures against an older one to find
        models the latest LLaMA-Factory release just unlocked. Probe + add the ones you want,
        then smoke-test them on the catalog (they're added untested).
      </Box>
      <Box>
        <Button onClick={findNewModels} loading={discovering} iconName="search">
          Find new models
        </Button>
      </Box>

      {discovery && (
        <SpaceBetween size="m">
          <Box>
            Comparing image <b>{discovery.newTag}</b>
            {discovery.baseTag ? <> against <b>{discovery.baseTag}</b></> : null}
            {discovery.transformers ? ` (transformers ${discovery.transformers})` : ""}.
          </Box>
          {discovery.note && <Alert type="info">{discovery.note}</Alert>}
          {discovery.suggestions.length === 0 ? (
            <Box color="text-status-inactive">
              No suggested HF models for the new architectures
              {discovery.newArchitectures.length > 0
                ? ` (new arches: ${discovery.newArchitectures.slice(0, 20).join(", ")}${
                    discovery.newArchitectures.length > 20 ? "…" : ""
                  })`
                : ""}
              .
            </Box>
          ) : (
            <Table
              variant="embedded"
              items={discovery.suggestions}
              trackBy="architecture"
              columnDefinitions={[
                {
                  id: "arch",
                  header: "Architecture",
                  cell: (s) => <Badge color="blue">{s.architecture}</Badge>,
                },
                {
                  id: "repos",
                  header: "Suggested models (probe & add)",
                  cell: (s) => (
                    <SpaceBetween size="xxs">
                      {s.repos.map((repo) => (
                        <SpaceBetween key={repo} direction="horizontal" size="xs">
                          <Button
                            variant="inline-link"
                            loading={onboarded[repo] === "probing"}
                            onClick={() =>
                              probeAndOnboard(
                                repo,
                                Object.entries(tiers).find(([, t]) => t === discovery.newTag)?.[0] ??
                                  "latest"
                              )
                            }
                          >
                            {repo}
                          </Button>
                          {onboarded[repo] && onboarded[repo] !== "probing" && (
                            <Box fontSize="body-s" color="text-status-inactive">
                              {onboarded[repo]}
                            </Box>
                          )}
                        </SpaceBetween>
                      ))}
                    </SpaceBetween>
                  ),
                },
              ]}
            />
          )}
        </SpaceBetween>
      )}
    </SpaceBetween>
  );
}
