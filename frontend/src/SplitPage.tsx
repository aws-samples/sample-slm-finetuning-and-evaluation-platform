// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import Container from "@cloudscape-design/components/container";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import FileUpload from "@cloudscape-design/components/file-upload";
import Button from "@cloudscape-design/components/button";
import Tabs from "@cloudscape-design/components/tabs";
import Input from "@cloudscape-design/components/input";

import {
  assertSplit,
  autoSplit,
  type CurrentSplit,
  type SplitReport,
} from "./api";
import { SplitResult } from "./SplitResult";
import { fileUploadI18n } from "./i18n";
import { useNotify, errText } from "./notifications";

interface SplitPageProps {
  // Called whenever a split succeeds (split persisted, has a splitId).
  onSplitReady: (split: CurrentSplit) => void;
  // Called when the user clicks "Use for render config" — also navigates.
  onUseForRender: (split: CurrentSplit) => void;
}

export function SplitPage({ onSplitReady, onUseForRender }: SplitPageProps) {
  const { notify } = useNotify();
  const [report, setReport] = useState<SplitReport | null>(null);
  const [loading, setLoading] = useState(false);

  // assert mode
  const [trainFiles, setTrainFiles] = useState<File[]>([]);
  const [evalFiles, setEvalFiles] = useState<File[]>([]);

  // auto mode
  const [autoFiles, setAutoFiles] = useState<File[]>([]);
  const [ratio, setRatio] = useState("0.2");
  const [seed, setSeed] = useState("42");

  // Human-friendly dataset/project name (shared across both modes).
  const [name, setName] = useState("");

  function reset() {
    setReport(null);
  }

  function toCurrentSplit(r: SplitReport): CurrentSplit | null {
    if (!r.ok || !r.splitId) return null;
    const origin =
      r.mode === "auto"
        ? `auto-split (ratio ${r.evalRatio}, seed ${r.seed})`
        : "two-file assert";
    return {
      splitId: r.splitId,
      name: r.name || undefined,
      trainRows: r.trainRows,
      evalRows: r.evalRows,
      origin,
    };
  }

  async function run(fn: () => Promise<SplitReport>) {
    setLoading(true);
    setReport(null);
    try {
      const r = await fn();
      setReport(r);
      const cs = toCurrentSplit(r);
      if (cs) onSplitReady(cs); // make it available to the Render page
    } catch (e) {
      notify({ type: "error", header: "Split failed", content: errText(e) });
    } finally {
      setLoading(false);
    }
  }

  const ratioNum = Number(ratio);
  const seedNum = Number(seed);
  const ratioValid = Number.isFinite(ratioNum) && ratioNum > 0 && ratioNum < 1;
  const seedValid = Number.isInteger(seedNum);

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Build a held-out test set that is disjoint from training (test ∩ train = ∅). Evaluating on training rows inflates scores via memorization."
        >
          Train / eval split
        </Header>
      }
    >
      <SpaceBetween size="l">
        <Container header={<Header variant="h2">Dataset</Header>}>
          <FormField
            label="Dataset name"
            description="A human-friendly name for this dataset/project (e.g. 'support-tickets-v1'). Shown across the app instead of the raw split id."
          >
            <Input
              value={name}
              placeholder="support-tickets-v1"
              onChange={({ detail }) => setName(detail.value)}
            />
          </FormField>
        </Container>

        <Container>
          <Tabs
            onChange={reset}
            tabs={[
              {
                id: "assert",
                label: "Two files (assert disjoint)",
                content: (
                  <SpaceBetween size="m">
                    <FormField
                      label="Training dataset (JSONL)"
                      description="Rows used to fine-tune."
                    >
                      <FileUpload
                        onChange={({ detail }) => {
                          setTrainFiles(detail.value);
                          reset();
                        }}
                        value={trainFiles}
                        accept=".jsonl,.json,.txt"
                        i18nStrings={fileUploadI18n}
                        showFileSize
                      />
                    </FormField>
                    <FormField
                      label="Eval dataset (JSONL)"
                      description="Held-out rows with ground truth. Asserted disjoint from training."
                    >
                      <FileUpload
                        onChange={({ detail }) => {
                          setEvalFiles(detail.value);
                          reset();
                        }}
                        value={evalFiles}
                        accept=".jsonl,.json,.txt"
                        i18nStrings={fileUploadI18n}
                        showFileSize
                      />
                    </FormField>
                    <Button
                      variant="primary"
                      loading={loading}
                      disabled={trainFiles.length === 0 || evalFiles.length === 0}
                      onClick={() => run(() => assertSplit(trainFiles[0], evalFiles[0], name))}
                    >
                      Check disjointness
                    </Button>
                  </SpaceBetween>
                ),
              },
              {
                id: "auto",
                label: "One file (auto-split)",
                content: (
                  <SpaceBetween size="m">
                    <FormField
                      label="Dataset (JSONL)"
                      description="Split deterministically into train/eval — disjoint by construction."
                    >
                      <FileUpload
                        onChange={({ detail }) => {
                          setAutoFiles(detail.value);
                          reset();
                        }}
                        value={autoFiles}
                        accept=".jsonl,.json,.txt"
                        i18nStrings={fileUploadI18n}
                        showFileSize
                      />
                    </FormField>
                    <FormField
                      label="Eval ratio"
                      description="Fraction held out for eval (0–1)."
                      errorText={ratio !== "" && !ratioValid ? "Must be between 0 and 1." : undefined}
                    >
                      <Input
                        value={ratio}
                        type="number"
                        onChange={({ detail }) => {
                          setRatio(detail.value);
                          reset();
                        }}
                      />
                    </FormField>
                    <FormField
                      label="Seed"
                      description="Deterministic shuffle seed; same seed → same split."
                      errorText={seed !== "" && !seedValid ? "Must be an integer." : undefined}
                    >
                      <Input
                        value={seed}
                        type="number"
                        onChange={({ detail }) => {
                          setSeed(detail.value);
                          reset();
                        }}
                      />
                    </FormField>
                    <Button
                      variant="primary"
                      loading={loading}
                      disabled={autoFiles.length === 0 || !ratioValid || !seedValid}
                      onClick={() => run(() => autoSplit(autoFiles[0], ratioNum, seedNum, name))}
                    >
                      Split
                    </Button>
                  </SpaceBetween>
                ),
              },
            ]}
          />
        </Container>

        {report && (
          <SplitResult
            report={report}
            onUseForRender={() => {
              const cs = toCurrentSplit(report);
              if (cs) onUseForRender(cs);
            }}
          />
        )}
      </SpaceBetween>
    </ContentLayout>
  );
}
