// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import Modal from "@cloudscape-design/components/modal";
import Box from "@cloudscape-design/components/box";
import Button from "@cloudscape-design/components/button";
import SpaceBetween from "@cloudscape-design/components/space-between";
import Alert from "@cloudscape-design/components/alert";
import Badge from "@cloudscape-design/components/badge";
import Table from "@cloudscape-design/components/table";
import TextFilter from "@cloudscape-design/components/text-filter";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import { useCollection } from "@cloudscape-design/collection-hooks";

import {
  addCustomModel,
  probeModel,
  serverlessCandidates,
  setServerlessTag,
  type ServerlessModelRow,
} from "./api";
import { useNotify, errText } from "./notifications";

// "Find serverless-customizable models": a browsable catalog of EVERY model AWS
// offers serverless customization (SFT/DPO/RLVR/RLAIF) for, on the live SageMaker
// Public Hub. Pick one or many and add/enable them in your catalog in one action.
//   • enabled     — already in your catalog with serverless on (nothing to do)
//   • addable      — in your catalog, just needs the serverless tag (one click)
//   • onboardable  — not in your catalog yet; we probe HF + add it WITH serverless on
//   • unavailable  — no public HF repo (e.g. Amazon Nova) or vision-language model
// Adding never trains — each model stays UNVERIFIED until you smoke-test it.
const STATE_LABEL: Record<string, string> = {
  enabled: "Enabled",
  addable: "In catalog",
  onboardable: "Available",
  unavailable: "Unavailable",
};

export function ServerlessFinder({
  visible,
  onDismiss,
  onChanged,
}: {
  visible: boolean;
  onDismiss: () => void;
  onChanged: () => void;
}) {
  const { notify } = useNotify();
  const [rows, setRows] = useState<ServerlessModelRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState<string>("");
  const [busy, setBusy] = useState<Record<string, string>>({}); // hubId → status text
  const [selected, setSelected] = useState<ServerlessModelRow[]>([]);
  const [bulkRunning, setBulkRunning] = useState(false);

  async function reload() {
    setLoading(true);
    try {
      const out = await serverlessCandidates();
      setRows(out.allModels);
      setNote(out.note);
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setLoading(false);
    }
  }

  // Auto-load when the modal opens (so it reads as a catalog, not a search box).
  useEffect(() => {
    if (visible && rows === null) reload();
    if (!visible) setSelected([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible]);

  // Enable serverless on an already-in-catalog row (writes the runtime tag).
  async function enableOne(row: ServerlessModelRow): Promise<boolean> {
    setBusy((b) => ({ ...b, [row.hubId]: "applying" }));
    try {
      await setServerlessTag(row.id, row.hubId);
      setBusy((b) => ({ ...b, [row.hubId]: "enabled — smoke-test it" }));
      return true;
    } catch (e) {
      setBusy((b) => ({ ...b, [row.hubId]: `failed: ${e instanceof Error ? e.message : String(e)}` }));
      return false;
    }
  }

  // Onboard a not-yet-in-catalog model: probe HF for its facts/template, then add
  // it WITH the serverless tag baked in (still unverified until smoke-tested).
  async function onboardOne(row: ServerlessModelRow): Promise<boolean> {
    setBusy((b) => ({ ...b, [row.hubId]: "probing" }));
    try {
      const probed = await probeModel(row.hfModelId);
      if (!probed.template || !probed.templateKnown) {
        setBusy((b) => ({
          ...b,
          [row.hubId]: "no recognized chat template — add manually via Add from Hugging Face",
        }));
        return false;
      }
      await addCustomModel({ ...probed, serverlessModelId: row.hubId });
      setBusy((b) => ({ ...b, [row.hubId]: "added with serverless enabled — smoke-test it" }));
      return true;
    } catch (e) {
      let msg = e instanceof Error ? e.message : String(e);
      if (/404|not found|could not fetch HF/i.test(msg)) {
        msg = "not found on Hugging Face (the repo may be gated or private)";
      }
      setBusy((b) => ({ ...b, [row.hubId]: `failed: ${msg}` }));
      return false;
    }
  }

  async function applyOne(row: ServerlessModelRow): Promise<boolean> {
    const ok = row.state === "addable" ? await enableOne(row) : await onboardOne(row);
    if (ok) onChanged();
    return ok;
  }

  // Only rows the user can act on are selectable.
  const actionable = (r: ServerlessModelRow) => r.state === "addable" || r.state === "onboardable";

  async function applySelected() {
    const todo = selected.filter(actionable);
    if (todo.length === 0) return;
    setBulkRunning(true);
    let changed = false;
    // Sequential so HF probes + tag writes don't stampede; each updates its own row.
    for (const row of todo) {
      const ok = row.state === "addable" ? await enableOne(row) : await onboardOne(row);
      changed = changed || ok;
    }
    if (changed) onChanged();
    await reload(); // refresh states (added rows flip to enabled)
    setSelected([]);
    setBulkRunning(false);
  }

  const { items, collectionProps, filterProps, filteredItemsCount } = useCollection(rows ?? [], {
    filtering: {
      filteringFunction: (item, text) => {
        const hay = `${item.displayName} ${item.hubId} ${item.hfModelId} ${item.recipes.join(" ")} ${item.state}`.toLowerCase();
        return hay.includes(text.toLowerCase());
      },
    },
    sorting: { defaultState: { sortingColumn: { sortingField: "displayName" } } },
  });

  const selectableTodo = selected.filter(actionable).length;

  function rowAction(r: ServerlessModelRow) {
    const status = busy[r.hubId];
    if (status && status !== "applying" && status !== "probing") {
      return (
        <Box fontSize="body-s" color="text-status-inactive">
          {status}
        </Box>
      );
    }
    if (r.state === "enabled") {
      return (
        <StatusIndicator type={r.verified ? "success" : "info"}>
          {r.verified ? "enabled ✓" : "enabled"}
        </StatusIndicator>
      );
    }
    if (r.state === "unavailable") {
      return (
        <Box fontSize="body-s" color="text-status-inactive">
          {r.reason}
        </Box>
      );
    }
    return (
      <Button
        variant="inline-link"
        loading={status === "applying" || status === "probing"}
        onClick={() => applyOne(r)}
      >
        {r.state === "addable" ? "Enable serverless" : "Add + enable serverless"}
      </Button>
    );
  }

  return (
    <Modal
      visible={visible}
      onDismiss={onDismiss}
      header="Serverless-customizable models"
      size="max"
      footer={
        <Box float="right">
          <Button variant="link" onClick={onDismiss}>
            Close
          </Button>
        </Box>
      }
    >
      <SpaceBetween size="m">
        <Box variant="p" color="text-status-inactive">
          Every model AWS offers SageMaker Serverless customization (SFT / DPO / RLVR / RLAIF) for,
          live from the Public Hub. Select models and add or enable them in your catalog — adding
          never trains, so each stays untested until you smoke-test it. Vision-language models and
          ones without a public Hugging Face repo (e.g. Amazon Nova) are listed but can't be added
          here.
        </Box>
        {note && <Alert type="info">{note}</Alert>}

        <Table<ServerlessModelRow>
          {...collectionProps}
          variant="embedded"
          loading={loading}
          loadingText="Querying the SageMaker Public Hub…"
          items={items}
          trackBy="hubId"
          selectionType="multi"
          selectedItems={selected}
          onSelectionChange={({ detail }) =>
            setSelected(detail.selectedItems.filter(actionable))
          }
          isItemDisabled={(r) => !actionable(r)}
          resizableColumns
          filter={
            <TextFilter
              {...filterProps}
              filteringPlaceholder="Find a model"
              countText={`${filteredItemsCount ?? 0} match${filteredItemsCount === 1 ? "" : "es"}`}
            />
          }
          header={
            <SpaceBetween direction="horizontal" size="xs">
              <Button iconName="refresh" onClick={reload} loading={loading}>
                Refresh
              </Button>
              <Button
                variant="primary"
                disabled={selectableTodo === 0}
                loading={bulkRunning}
                onClick={applySelected}
              >
                Add / enable selected{selectableTodo > 0 ? ` (${selectableTodo})` : ""}
              </Button>
            </SpaceBetween>
          }
          empty={
            <Box textAlign="center" color="inherit" padding={{ vertical: "l" }}>
              No serverless-customizable models found (or the hub is unreachable).
            </Box>
          }
          columnDefinitions={[
            {
              id: "model",
              header: "Model",
              sortingField: "displayName",
              cell: (r) => (
                <SpaceBetween direction="horizontal" size="xs">
                  <b>{r.displayName}</b>
                  {r.gated && <Badge color="grey">gated</Badge>}
                </SpaceBetween>
              ),
            },
            {
              id: "hubId",
              header: "Public Hub id",
              sortingField: "hubId",
              cell: (r) => (
                <Box variant="code" fontSize="body-s">
                  {r.hubId}
                </Box>
              ),
            },
            {
              id: "recipes",
              header: "Recipes",
              cell: (r) => (
                <SpaceBetween direction="horizontal" size="xxs">
                  {r.recipes.map((x) => (
                    <Badge key={x} color="blue">
                      {x.replace("_lora", "")}
                    </Badge>
                  ))}
                </SpaceBetween>
              ),
            },
            {
              id: "state",
              header: "State",
              sortingField: "state",
              cell: (r) => STATE_LABEL[r.state] ?? r.state,
              width: 120,
            },
            {
              id: "action",
              header: "",
              cell: rowAction,
              width: 240,
            },
          ]}
        />
      </SpaceBetween>
    </Modal>
  );
}
