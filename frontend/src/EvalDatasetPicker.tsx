// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Select from "@cloudscape-design/components/select";
import SegmentedControl from "@cloudscape-design/components/segmented-control";
import FileUpload from "@cloudscape-design/components/file-upload";
import Input from "@cloudscape-design/components/input";
import Button from "@cloudscape-design/components/button";
import Alert from "@cloudscape-design/components/alert";
import Table from "@cloudscape-design/components/table";

import {
  getDatasets,
  uploadEvalDataset,
  type CurrentSplit,
  type Dataset,
} from "./api";
import { fileUploadI18n } from "./i18n";
import { useNotify, errText } from "./notifications";

// Dataset chooser for STANDALONE EVALUATE. Evaluation only needs held-out rows
// (no training split), so this is deliberately simpler than the fine-tuning
// DatasetPicker: reuse an existing dataset's test set, or upload a JSONL of
// test rows directly. Emits the chosen dataset as a CurrentSplit.
export function EvalDatasetPicker({
  selected,
  onSelect,
}: {
  selected: CurrentSplit | null;
  onSelect: (split: CurrentSplit | null) => void;
}) {
  const { notify } = useNotify();
  const [mode, setMode] = useState<"existing" | "upload">("existing");
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  // upload state
  const [name, setName] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [rowErrors, setRowErrors] = useState<{ line: number; message: string }[] | null>(null);

  function loadDatasets() {
    getDatasets()
      .then(setDatasets)
      .catch((e) => notify({ type: "error", content: errText(e) }));
  }
  useEffect(loadDatasets, []);

  function datasetToCurrent(d: Dataset): CurrentSplit {
    return {
      splitId: d.splitId,
      name: d.name || undefined,
      trainRows: d.trainRows ?? 0,
      evalRows: d.evalRows ?? 0,
      origin: "existing dataset",
    };
  }

  async function doUpload() {
    setUploading(true);
    setRowErrors(null);
    try {
      const r = await uploadEvalDataset(files[0], name);
      if (!r.ok || !r.splitId) {
        setRowErrors(r.errors ?? [{ line: 0, message: "no valid rows" }]);
        return;
      }
      onSelect({
        splitId: r.splitId,
        name: r.name || undefined,
        trainRows: 0,
        evalRows: r.evalRows,
        origin: "uploaded test set",
      });
      loadDatasets(); // now selectable under "Use existing" too
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setUploading(false);
    }
  }

  return (
    <Container
      header={
        <Header
          variant="h2"
          description="Pick an existing dataset's held-out set, or upload a JSONL of test rows to score on. No training split needed."
        >
          Step 1 — Eval dataset
        </Header>
      }
    >
      <SpaceBetween size="m">
        <SegmentedControl
          selectedId={mode}
          onChange={({ detail }) => setMode(detail.selectedId as "existing" | "upload")}
          options={[
            { id: "existing", text: "Use existing dataset" },
            { id: "upload", text: "Upload test JSONL" },
          ]}
        />

        {mode === "existing" ? (
          <FormField label="Dataset" description="Its test rows are scored. Pick from your library.">
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
                label: d.name || d.splitId,
                description: `${d.evalRows ?? "?"} test rows`,
              }))}
              placeholder="Choose a dataset"
              empty="No datasets yet — switch to 'Upload test JSONL'"
              filteringType="auto"
            />
          </FormField>
        ) : (
          <SpaceBetween size="m">
            <FormField label="Dataset name" description="Optional label, e.g. holdout-v1.">
              <Input value={name} placeholder="holdout-v1" onChange={({ detail }) => setName(detail.value)} />
            </FormField>
            <FormField
              label="Test file (JSONL)"
              description="One chat-template object per line (messages array with ≥1 assistant turn)."
            >
              <FileUpload
                onChange={({ detail }) => {
                  setFiles(detail.value);
                  // Clear any prior validation errors so a freshly chosen file
                  // doesn't show a stale invalid-rows table from the last upload.
                  setRowErrors(null);
                }}
                value={files}
                accept=".jsonl,.json,.txt"
                i18nStrings={fileUploadI18n}
                showFileSize
              />
            </FormField>
            <Button
              variant="primary"
              loading={uploading}
              disabled={files.length === 0}
              onClick={doUpload}
            >
              Use this test set
            </Button>
            {rowErrors && (
              <Alert type="error" header="Upload blocked — fix invalid rows">
                <Table
                  variant="embedded"
                  columnDefinitions={[
                    { id: "line", header: "Line", cell: (r) => r.line, width: 90 },
                    { id: "msg", header: "Problem", cell: (r) => r.message },
                  ]}
                  items={rowErrors}
                  empty="No rows"
                />
              </Alert>
            )}
          </SpaceBetween>
        )}

        {selected && (
          <Alert type="success">
            Eval set: <b>{selected.name || selected.splitId}</b> · {selected.evalRows} rows
          </Alert>
        )}
      </SpaceBetween>
    </Container>
  );
}
