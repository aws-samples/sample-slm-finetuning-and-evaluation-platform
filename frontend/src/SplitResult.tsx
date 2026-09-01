// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import Container from "@cloudscape-design/components/container";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import ColumnLayout from "@cloudscape-design/components/column-layout";
import Box from "@cloudscape-design/components/box";
import Alert from "@cloudscape-design/components/alert";
import Button from "@cloudscape-design/components/button";
import StatusIndicator from "@cloudscape-design/components/status-indicator";
import ExpandableSection from "@cloudscape-design/components/expandable-section";
import Table from "@cloudscape-design/components/table";

import type { RowError, SplitReport } from "./api";
import { MessagesTable } from "./MessagesCell";

// Per-row validation errors (folded in from the old Dataset Validation page).
function ErrorsCard({ title, errors }: { title: string; errors: RowError[] }) {
  if (errors.length === 0) return null;
  return (
    <Container header={<Header variant="h2">{title}</Header>}>
      <Table
        variant="embedded"
        columnDefinitions={[
          { id: "line", header: "Line", cell: (e) => e.line, width: 100 },
          { id: "message", header: "Problem", cell: (e) => e.message },
        ]}
        items={errors}
        empty="No errors"
      />
    </Container>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <Box variant="awsui-key-label">{label}</Box>
      <Box variant="h2">{value}</Box>
    </div>
  );
}

export function SplitResult({
  report,
  onUseForRender,
}: {
  report: SplitReport;
  onUseForRender?: () => void;
}) {
  const disjoint = report.overlapCount === 0;
  const hasInvalid = report.trainInvalidRows > 0 || report.evalInvalidRows > 0;
  const hasLeakageWarning = disjoint && !hasInvalid && report.promptCollisionCount > 0;

  return (
    <SpaceBetween size="l">
      <Container
        header={
          <Header
            variant="h2"
            description={
              report.mode === "auto"
                ? `Auto-split · ratio ${report.evalRatio} · seed ${report.seed}`
                : "Two-file disjointness assertion"
            }
            actions={
              // Only show when a handler is wired (the standalone SplitPage).
              // In the DatasetPicker upload flow no handler is passed — the
              // dataset is auto-selected on success — so the button is omitted
              // there rather than rendering a dead no-op.
              report.ok && report.splitId && onUseForRender ? (
                <Button variant="primary" iconName="arrow-right" onClick={onUseForRender}>
                  Use for fine-tune
                </Button>
              ) : undefined
            }
          >
            Split summary
          </Header>
        }
      >
        <SpaceBetween size="m">
          {hasInvalid ? (
            <Alert type="error" header="Dataset has invalid rows — split blocked">
              {report.trainInvalidRows > 0 && (
                <div>{report.trainInvalidRows} invalid row(s) in the training file.</div>
              )}
              {report.evalInvalidRows > 0 && (
                <div>{report.evalInvalidRows} invalid row(s) in the test file.</div>
              )}
              Fix the rows listed below and try again. Nothing was persisted.
            </Alert>
          ) : !report.ok ? (
            <Alert type="error" header="Test set is NOT disjoint from training">
              {report.overlapCount} test row(s) are identical to training rows.
              Evaluating on these inflates scores via memorization — remove them
              from the test set (or from training) before fine-tuning.
            </Alert>
          ) : hasLeakageWarning ? (
            <Alert type="warning" header="Possible prompt leakage">
              {report.promptCollisionCount} test prompt(s) also appear in
              training with a different answer. The full rows differ (so this
              passes the disjointness check), but the shared inputs are worth a
              look.
            </Alert>
          ) : report.hasVal ? (
            <Alert type="success" header="train / validation / test all disjoint ✓">
              Validation set ({report.valRows} rows
              {report.valMode === "carve" ? `, carved ${report.valRatio} of train` : ", uploaded"}) is
              used for in-training validation &amp; early stopping. The {report.evalRows}-row test set is held
              out for the final leaderboard score and never used for stopping.
            </Alert>
          ) : (
            <Alert type="success" header="test ∩ train = ∅ (disjoint)">
              The test set shares no rows with training. No validation set — add
              one to enable early stopping.
            </Alert>
          )}

          <ColumnLayout columns={4} variant="text-grid">
            <Stat
              label="Disjoint"
              value={
                <StatusIndicator type={disjoint ? "success" : "error"}>
                  {disjoint ? "Yes" : "No"}
                </StatusIndicator>
              }
            />
            <Stat label="Train rows" value={report.trainRows} />
            <Stat label="Test rows" value={report.evalRows} />
            <Stat
              label="Full-row overlap"
              value={
                <StatusIndicator type={report.overlapCount === 0 ? "success" : "error"}>
                  {report.overlapCount}
                </StatusIndicator>
              }
            />
          </ColumnLayout>

          <ColumnLayout columns={4} variant="text-grid">
            <Stat
              label="Prompt collisions"
              value={
                <StatusIndicator
                  type={report.promptCollisionCount === 0 ? "success" : "warning"}
                >
                  {report.promptCollisionCount}
                </StatusIndicator>
              }
            />
            <Stat
              label="Train invalid"
              value={
                <StatusIndicator type={report.trainInvalidRows === 0 ? "success" : "error"}>
                  {report.trainInvalidRows}
                </StatusIndicator>
              }
            />
            <Stat
              label="Test invalid"
              value={
                <StatusIndicator type={report.evalInvalidRows === 0 ? "success" : "error"}>
                  {report.evalInvalidRows}
                </StatusIndicator>
              }
            />
            <Stat
              label="Test fraction"
              value={
                report.trainRows + report.evalRows > 0
                  ? `${Math.round(
                      (report.evalRows / (report.trainRows + report.evalRows)) * 100
                    )}%`
                  : "—"
              }
            />
          </ColumnLayout>

          {report.hasVal && (
            <ColumnLayout columns={4} variant="text-grid">
              <Stat label="Validation rows" value={report.valRows} />
              <Stat
                label="Validation source"
                value={report.valMode === "carve" ? `carved ${report.valRatio} of train` : "uploaded file"}
              />
              <Stat
                label="Val invalid"
                value={
                  <StatusIndicator type={report.valInvalidRows === 0 ? "success" : "error"}>
                    {report.valInvalidRows}
                  </StatusIndicator>
                }
              />
              <Stat label="Early stopping" value={<StatusIndicator type="success">available</StatusIndicator>} />
            </ColumnLayout>
          )}

          {report.splitId && (
            <Box>
              <Box variant="awsui-key-label">Split id</Box>
              <Box>{report.splitId}</Box>
            </Box>
          )}

          {report.messages.length > 0 && (
            <Box>
              <Box variant="awsui-key-label">Notes</Box>
              <SpaceBetween size="xxs">
                {report.messages.map((m, i) => (
                  <Box key={i}>{m}</Box>
                ))}
              </SpaceBetween>
            </Box>
          )}
        </SpaceBetween>
      </Container>

      <ErrorsCard title="Training file — invalid rows" errors={report.trainErrors ?? []} />
      <ErrorsCard title="Test file — invalid rows" errors={report.evalErrors ?? []} />

      {report.overlapExamples.length > 0 && (
        <Container header={<Header variant="h2">Overlapping rows</Header>}>
          <ExpandableSection
            defaultExpanded
            headerText={`${report.overlapCount} identical row(s) in both splits${
              report.overlapExamples.length < report.overlapCount
                ? ` (showing first ${report.overlapExamples.length})`
                : ""
            }`}
          >
            <MessagesTable rows={report.overlapExamples} empty="None" />
          </ExpandableSection>
        </Container>
      )}

      <ColumnLayout columns={2}>
        <Container
          header={
            <Header variant="h2" counter={`(${report.trainRows})`}>
              Train preview
            </Header>
          }
        >
          <MessagesTable rows={report.trainPreview} empty="No train rows" />
        </Container>
        <Container
          header={
            <Header variant="h2" counter={`(${report.evalRows})`}>
              Test preview
            </Header>
          }
        >
          <MessagesTable rows={report.evalPreview} empty="No test rows" />
        </Container>
      </ColumnLayout>

      {report.hasVal && (
        <Container
          header={
            <Header variant="h2" counter={`(${report.valRows})`} description="In-training validation / early stopping set">
              Validation preview
            </Header>
          }
        >
          <MessagesTable rows={report.valPreview} empty="No val rows" />
        </Container>
      )}
    </SpaceBetween>
  );
}
