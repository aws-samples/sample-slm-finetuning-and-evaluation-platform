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
import Toggle from "@cloudscape-design/components/toggle";
import TextFilter from "@cloudscape-design/components/text-filter";
import { useCollection } from "@cloudscape-design/collection-hooks";

import Modal from "@cloudscape-design/components/modal";

import { archiveDataset, getDatasets, type CurrentSplit, type Dataset } from "./api";
import { InvestigateDataset } from "./InvestigateDataset";
import { DatasetPicker, shapeLabel } from "./DatasetPicker";
import { HfBadge } from "./hfBits";
import { useNotify, errText } from "./notifications";

export function DatasetsPage({
  onUseDataset,
  focusSplitId,
}: {
  onUseDataset: (d: Dataset) => void;
  focusSplitId?: string | null;
}) {
  const { notify } = useNotify();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [investigate, setInvestigate] = useState<Dataset | null>(null);
  // True when the modal was opened via a deep-link (Runs/Leaderboard) → show the
  // dataset DETAILS only, not the full agentic Investigate wizard.
  const [investigateDetailsOnly, setInvestigateDetailsOnly] = useState(false);
  // "Create dataset" modal — reuses the full DatasetPicker (upload / HF / etc.).
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<CurrentSplit | null>(null);

  function refresh() {
    setLoading(true);
    // The master page always fetches the FULL set (incl. archived) so it can
    // manage availability; the Show-archived toggle filters client-side.
    getDatasets(true)
      .then(setDatasets)
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoading(false));
  }

  useEffect(refresh, []);

  // Deep-link from a run's Dataset column: once datasets load, auto-open the
  // target dataset's Investigate panel (its stats) so the link lands somewhere
  // meaningful rather than just the page.
  useEffect(() => {
    if (!focusSplitId || datasets.length === 0) return;
    const d = datasets.find((x) => x.splitId === focusSplitId);
    if (d) {
      setInvestigateDetailsOnly(true); // deep-link → details view, not the wizard
      setInvestigate(d);
    }
  }, [focusSplitId, datasets]);

  async function toggleArchive(splitId: string, archived: boolean) {
    try {
      await archiveDataset(splitId, archived);
      notify({ type: "success", content: archived ? "Dataset archived." : "Dataset restored." });
      refresh();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    }
  }

  const archivedCount = datasets.filter((d) => d.archived).length;
  // The Show-archived toggle filters BEFORE the collection (so its header counter
  // reflects the archive scope, not the text filter); the collection then handles
  // free-text search + sorting on top.
  const visible = useMemo(
    () => (showArchived ? datasets : datasets.filter((d) => !d.archived)),
    [datasets, showArchived]
  );
  const { items, collectionProps, filterProps, filteredItemsCount } = useCollection(visible, {
    filtering: {
      // Match against name, split id, type/shape and mode so a quick query finds a
      // dataset by any of the columns the eye scans.
      filteringFunction: (d, text) => {
        const hay = `${d.name ?? ""} ${d.splitId} ${shapeLabel(d.shape)} ${d.mode ?? ""}`.toLowerCase();
        return hay.includes(text.toLowerCase());
      },
    },
    // Newest first by default — the most-recently-created dataset is usually the
    // one you just made and want to act on.
    sorting: { defaultState: { sortingColumn: { sortingField: "mtime" }, isDescending: true } },
  });

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Every dataset you've created. A dataset is a held-out split (train + test) — or a test-only set — with a stable id. Pick one to fine-tune models on it."
        >
          Datasets
        </Header>
      }
    >
      <SpaceBetween size="l">
        <Table
          {...collectionProps}
          variant="container"
          resizableColumns
          trackBy="splitId"
          loading={loading}
          loadingText="Loading datasets…"
          filter={
            <TextFilter
              {...filterProps}
              filteringPlaceholder="Find by name, split id, type, or mode"
              countText={`${filteredItemsCount ?? 0} match${filteredItemsCount === 1 ? "" : "es"}`}
            />
          }
          header={
            <Header
              counter={
                showArchived ? `(${visible.length})` : `(${visible.length} of ${datasets.length})`
              }
              actions={
                <SpaceBetween direction="horizontal" size="m">
                  <Toggle
                    checked={showArchived}
                    onChange={({ detail }) => setShowArchived(detail.checked)}
                  >
                    Show archived{archivedCount ? ` (${archivedCount})` : ""}
                  </Toggle>
                  <Button iconName="refresh" onClick={refresh} loading={loading}>
                    Refresh
                  </Button>
                  <Button
                    variant="primary"
                    iconName="add-plus"
                    onClick={() => {
                      setCreated(null);
                      setCreating(true);
                    }}
                  >
                    Create dataset
                  </Button>
                </SpaceBetween>
              }
            >
              Dataset library
            </Header>
          }
          columnDefinitions={[
            {
              id: "name",
              header: "Name",
              sortingField: "name",
              cell: (d) => (
                <SpaceBetween direction="horizontal" size="xs">
                  {d.name ? <b>{d.name}</b> : <Box color="text-status-inactive">(unnamed)</Box>}
                  {d.source === "huggingface" && <HfBadge dataset={d.hfDataset} />}
                  {d.evalOnly && <Badge color="blue">test-only</Badge>}
                  {d.hasVal && <Badge color="green">val</Badge>}
                  {d.archived && <Badge color="grey">archived</Badge>}
                </SpaceBetween>
              ),
            },
            {
              id: "type",
              header: "Type",
              // Fine-tuning objective the dataset's shape feeds (SFT/DPO/KTO).
              cell: (d) => {
                const lbl = shapeLabel(d.shape);
                const color =
                  lbl === "DPO" ? "green"
                  : lbl === "KTO" ? "blue"
                  : lbl === "RLVR" ? "severity-high"
                  : "grey";
                return <Badge color={color}>{lbl}</Badge>;
              },
            },
            { id: "id", header: "Split id", cell: (d) => d.splitId },
            { id: "train", header: "Train rows", cell: (d) => d.trainRows ?? "—" },
            { id: "eval", header: "Test rows", cell: (d) => d.evalRows ?? "—" },
            { id: "val", header: "Val rows", cell: (d) => (d.hasVal ? d.valRows ?? "—" : "—") },
            { id: "mode", header: "Mode", cell: (d) => d.mode ?? "—" },
            {
              id: "baseline",
              header: "Sonnet baseline",
              cell: (d) =>
                d.hasBaseline ? <Badge color="green">yes</Badge> : <Badge color="grey">no</Badge>,
            },
            {
              id: "created",
              header: "Created",
              sortingField: "mtime",
              // mtime is a Unix timestamp in SECONDS (→ ×1000 for JS Date). 0/absent
              // (e.g. a just-created stub) shows an em dash rather than the epoch.
              cell: (d) =>
                d.mtime ? new Date(d.mtime * 1000).toLocaleString() : "—",
            },
            {
              id: "actions",
              header: "Actions",
              // Explicit width so the 3 inline-link buttons stay on one line — without
              // it the column gets squeezed and each label wraps to many rows.
              minWidth: 320,
              cell: (d) => (
                <SpaceBetween direction="horizontal" size="xs">
                  <Button
                    variant="inline-link"
                    iconName="search"
                    onClick={() => {
                      setInvestigateDetailsOnly(false); // explicit Investigate → full wizard
                      setInvestigate(d);
                    }}
                  >
                    Investigate
                  </Button>
                  {!d.evalOnly && (
                    <Button
                      variant="inline-link"
                      iconName="arrow-right"
                      onClick={() => onUseDataset(d)}
                    >
                      Fine-tune on this
                    </Button>
                  )}
                  {d.archived ? (
                    <Button
                      variant="inline-link"
                      iconName="undo"
                      onClick={() => toggleArchive(d.splitId, false)}
                    >
                      Restore
                    </Button>
                  ) : (
                    <Button
                      variant="inline-link"
                      iconName="close"
                      onClick={() => toggleArchive(d.splitId, true)}
                    >
                      Archive
                    </Button>
                  )}
                </SpaceBetween>
              ),
            },
          ]}
          items={items}
          empty={
            <Box textAlign="center" padding="l">
              <SpaceBetween size="s">
                <span>{datasets.length === 0 ? "No datasets yet." : "No datasets match."}</span>
                <Box variant="small">
                  {datasets.length === 0
                    ? "Create one on the Fine-tune page (upload + split)."
                    : "Clear the filter or toggle Show archived to see more datasets."}
                </Box>
              </SpaceBetween>
            </Box>
          }
        />
      </SpaceBetween>
      {investigate && (
        <InvestigateDataset
          splitId={investigate.splitId}
          name={investigate.name}
          detailsOnly={investigateDetailsOnly}
          onClose={() => setInvestigate(null)}
          onFineTune={() => {
            const d = investigate;
            setInvestigate(null);
            onUseDataset(d); // routes to the Fine-tune page with this dataset pre-selected
          }}
        />
      )}

      <Modal
        visible={creating}
        size="large"
        onDismiss={() => setCreating(false)}
        header="Create a dataset"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              <Button onClick={() => setCreating(false)}>Done</Button>
              {created && (
                <Button
                  variant="primary"
                  iconName="arrow-right"
                  onClick={() => {
                    setCreating(false);
                    onUseDataset({
                      splitId: created.splitId,
                      name: created.name ?? null,
                      trainRows: created.trainRows,
                      evalRows: created.evalRows,
                      mode: null,
                      // Forward the shape so a just-created DPO/KTO/RLVR dataset
                      // opens the Fine-Tune page with the correct objective.
                      shape: created.shape,
                      hasVal: created.hasVal,
                      valRows: created.valRows,
                      hasBaseline: false,
                      mtime: 0,
                    });
                  }}
                >
                  Fine-tune on this
                </Button>
              )}
            </SpaceBetween>
          </Box>
        }
      >
        <DatasetPicker
          selected={created}
          onSelect={(cs) => {
            setCreated(cs);
            if (cs) refresh(); // a newly created dataset appears in the library
          }}
        />
      </Modal>
    </ContentLayout>
  );
}
