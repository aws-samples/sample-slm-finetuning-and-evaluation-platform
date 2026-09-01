// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { useEffect, useState } from "react";
import ContentLayout from "@cloudscape-design/components/content-layout";
import Header from "@cloudscape-design/components/header";
import SpaceBetween from "@cloudscape-design/components/space-between";
import FormField from "@cloudscape-design/components/form-field";
import Input from "@cloudscape-design/components/input";
import Textarea from "@cloudscape-design/components/textarea";
import Select from "@cloudscape-design/components/select";
import Button from "@cloudscape-design/components/button";
import Box from "@cloudscape-design/components/box";
import Badge from "@cloudscape-design/components/badge";
import Table from "@cloudscape-design/components/table";
import TextFilter from "@cloudscape-design/components/text-filter";
import FileUpload from "@cloudscape-design/components/file-upload";
import Modal from "@cloudscape-design/components/modal";
import { useCollection } from "@cloudscape-design/collection-hooks";

import {
  createFeedback,
  deleteFeedback,
  listFeedback,
  setFeedbackStatus,
  type FeedbackBoard,
  type FeedbackEntry,
} from "./api";
import { useNotify, errText } from "./notifications";

const TYPE_LABEL: Record<string, string> = {
  issue: "Issue",
  idea: "Idea / feature",
  praise: "Praise",
};
const TYPE_COLOR: Record<string, "red" | "blue" | "green"> = {
  issue: "red",
  idea: "blue",
  praise: "green",
};
const STATUS_LABEL: Record<string, string> = {
  open: "Open",
  planned: "Planned",
  done: "Done",
  wont_do: "Won't do",
};

const MAX_MB = 10;

// "2026-06-18T14:07:44+00:00" → "Jun 18, 2026, 14:07". The createdStamp is a
// "%Y%m%d-%H%M%S-%f" string; parse it leniently for display, fall back to raw.
function fmtStamp(stamp: string): string {
  const m = stamp.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/);
  if (!m) return stamp;
  const [, y, mo, d, h, mi] = m;
  const date = new Date(Number(y), Number(mo) - 1, Number(d), Number(h), Number(mi));
  if (Number.isNaN(date.getTime())) return stamp;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function FeedbackPage() {
  const { notify } = useNotify();
  const [board, setBoard] = useState<FeedbackBoard | null>(null);
  const [loading, setLoading] = useState(true);

  // "Add feedback" dialog + its form state.
  const [addOpen, setAddOpen] = useState(false);
  const [type, setType] = useState<"issue" | "idea" | "praise">("issue");
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [fileError, setFileError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Delete confirmation.
  const [removeTarget, setRemoveTarget] = useState<FeedbackEntry | null>(null);
  const [removing, setRemoving] = useState(false);

  function load() {
    setLoading(true);
    listFeedback()
      .then(setBoard)
      .catch((e) => notify({ type: "error", content: errText(e) }))
      .finally(() => setLoading(false));
  }
  useEffect(load, []);

  const maxAttachments = board?.maxAttachments ?? 5;
  const me = board?.me ?? "";
  const statuses = board?.statuses ?? ["open", "planned", "done", "wont_do"];

  // Sortable + filterable table over the feedback entries.
  const { items, collectionProps, filterProps, filteredItemsCount } = useCollection(
    board?.feedback ?? [],
    {
      filtering: {
        filteringFunction: (item, text) => {
          const hay = `${item.title} ${item.body} ${item.author} ${item.type} ${item.status}`.toLowerCase();
          return hay.includes(text.toLowerCase());
        },
      },
      sorting: { defaultState: { sortingColumn: { sortingField: "createdStamp" }, isDescending: true } },
    }
  );

  function resetForm() {
    setType("issue");
    setTitle("");
    setBody("");
    setFiles([]);
    setFileError(null);
  }

  function onFilesChange(next: File[]) {
    setFileError(null);
    if (next.length > maxAttachments) {
      setFileError(`At most ${maxAttachments} screenshots.`);
      next = next.slice(0, maxAttachments);
    }
    const tooBig = next.find((f) => f.size > MAX_MB * 1024 * 1024);
    if (tooBig) setFileError(`"${tooBig.name}" is over ${MAX_MB} MB.`);
    const notImage = next.find((f) => !f.type.startsWith("image/"));
    if (notImage) setFileError(`"${notImage.name}" is not an image.`);
    setFiles(next);
  }

  const canSubmit =
    !!title.trim() &&
    !fileError &&
    files.every((f) => f.type.startsWith("image/") && f.size <= MAX_MB * 1024 * 1024);

  async function submit() {
    if (!canSubmit) return;
    setSubmitting(true);
    try {
      await createFeedback({ type, title: title.trim(), body, files });
      setAddOpen(false);
      resetForm();
      notify({ type: "success", content: "Thanks — your feedback was posted." });
      load();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setSubmitting(false);
    }
  }

  async function changeStatus(entry: FeedbackEntry, status: string) {
    try {
      await setFeedbackStatus(entry.id, status);
      load();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    }
  }

  async function remove(entry: FeedbackEntry) {
    setRemoving(true);
    try {
      await deleteFeedback(entry.id);
      setRemoveTarget(null);
      load();
    } catch (e) {
      notify({ type: "error", content: errText(e) });
    } finally {
      setRemoving(false);
    }
  }

  return (
    <ContentLayout
      header={
        <Header
          variant="h1"
          description="Report an issue, request a feature, or share what worked. Everyone sees the board — attach screenshots as proof."
        >
          Feedback
        </Header>
      }
    >
      <SpaceBetween size="l">
        <Table<FeedbackEntry>
          {...collectionProps}
          variant="container"
          loading={loading}
          items={items}
          trackBy={(e) => e.id}
          resizableColumns
          wrapLines
          filter={
            <TextFilter
              {...filterProps}
              filteringPlaceholder="Find feedback"
              countText={`${filteredItemsCount ?? 0} match${filteredItemsCount === 1 ? "" : "es"}`}
            />
          }
          header={
            <Header
              variant="h2"
              counter={`(${board?.feedback.length ?? 0})`}
              actions={
                <SpaceBetween direction="horizontal" size="xs">
                  <Button iconName="refresh" onClick={load} loading={loading}>
                    Refresh
                  </Button>
                  <Button variant="primary" iconName="add-plus" onClick={() => setAddOpen(true)}>
                    Add feedback
                  </Button>
                </SpaceBetween>
              }
            >
              All feedback
            </Header>
          }
          empty={
            <Box textAlign="center" color="inherit" padding={{ vertical: "l" }}>
              <SpaceBetween size="s">
                <span>No feedback yet.</span>
                <Button onClick={() => setAddOpen(true)}>Add the first one</Button>
              </SpaceBetween>
            </Box>
          }
          columnDefinitions={[
            {
              id: "type",
              header: "Type",
              sortingField: "type",
              cell: (e) => <Badge color={TYPE_COLOR[e.type] ?? "grey"}>{TYPE_LABEL[e.type] ?? e.type}</Badge>,
              width: 130,
            },
            {
              id: "title",
              header: "Title",
              sortingField: "title",
              cell: (e) => (
                <SpaceBetween size="xxs">
                  <b>{e.title}</b>
                  {e.body && (
                    <Box variant="small" color="text-body-secondary">
                      {e.body.length > 160 ? e.body.slice(0, 160) + "…" : e.body}
                    </Box>
                  )}
                </SpaceBetween>
              ),
            },
            {
              id: "attachments",
              header: "Screenshots",
              sortingComparator: (a, b) => a.attachments.length - b.attachments.length,
              cell: (e) =>
                e.attachments.length === 0 ? (
                  <Box color="text-status-inactive">—</Box>
                ) : (
                  <SpaceBetween direction="horizontal" size="xxs">
                    {e.attachments.map((a) =>
                      a.url ? (
                        <a key={a.key} href={a.url} target="_blank" rel="noreferrer" title={a.name}>
                          <img
                            src={a.url}
                            alt={a.name}
                            style={{
                              height: 40,
                              width: 56,
                              objectFit: "cover",
                              borderRadius: 4,
                              // Cloudscape divider token → adapts to dark mode (falls
                              // back to the light grey if the var isn't present).
                              border: "1px solid var(--color-border-divider-default, #d5dbdb)",
                            }}
                          />
                        </a>
                      ) : (
                        <Badge key={a.key} color="grey">
                          {a.name}
                        </Badge>
                      )
                    )}
                  </SpaceBetween>
                ),
              width: 160,
            },
            {
              id: "status",
              header: "Status",
              sortingField: "status",
              cell: (e) => (
                <Select
                  selectedOption={{ value: e.status, label: STATUS_LABEL[e.status] ?? e.status }}
                  onChange={({ detail }) => changeStatus(e, detail.selectedOption.value!)}
                  options={statuses.map((s) => ({ value: s, label: STATUS_LABEL[s] ?? s }))}
                  expandToViewport
                />
              ),
              width: 150,
            },
            {
              id: "author",
              header: "Author",
              sortingField: "author",
              cell: (e) => e.author,
              width: 150,
            },
            {
              id: "createdStamp",
              header: "Submitted",
              sortingField: "createdStamp",
              cell: (e) => <Box variant="small">{fmtStamp(e.createdStamp)}</Box>,
              width: 170,
            },
            {
              id: "actions",
              header: "",
              cell: (e) =>
                e.author === me ? (
                  <Button variant="inline-link" onClick={() => setRemoveTarget(e)}>
                    Delete
                  </Button>
                ) : (
                  ""
                ),
              width: 90,
            },
          ]}
        />
      </SpaceBetween>

      {/* Add-feedback dialog. */}
      <Modal
        visible={addOpen}
        onDismiss={() => !submitting && setAddOpen(false)}
        header="Add feedback"
        size="medium"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="link" onClick={() => setAddOpen(false)} disabled={submitting}>
                Cancel
              </Button>
              <Button variant="primary" loading={submitting} disabled={!canSubmit} onClick={submit}>
                Post feedback
              </Button>
            </SpaceBetween>
          </Box>
        }
      >
        <SpaceBetween size="m">
          <FormField label="Type">
            <Select
              selectedOption={{ value: type, label: TYPE_LABEL[type] }}
              onChange={({ detail }) =>
                setType(detail.selectedOption.value as "issue" | "idea" | "praise")
              }
              options={(board?.types ?? ["issue", "idea", "praise"]).map((t) => ({
                value: t,
                label: TYPE_LABEL[t] ?? t,
              }))}
            />
          </FormField>
          <FormField label={<span>Title <i>- required</i></span>}>
            <Input
              value={title}
              placeholder="Short summary, e.g. 'Verify on serverless spins forever'"
              onChange={({ detail }) => setTitle(detail.value)}
            />
          </FormField>
          <FormField
            label="Details"
            description="What happened, what you expected, steps to reproduce — or just what you liked."
          >
            <Textarea
              value={body}
              onChange={({ detail }) => setBody(detail.value)}
              rows={5}
              placeholder="The more specific, the more actionable."
            />
          </FormField>
          <FormField
            label="Screenshots (optional)"
            description={`Images only (png/jpg/gif/webp), up to ${maxAttachments}, ${MAX_MB} MB each.`}
            errorText={fileError ?? undefined}
          >
            <FileUpload
              multiple
              value={files}
              onChange={({ detail }) => onFilesChange(detail.value)}
              accept="image/*"
              showFileLastModified
              showFileSize
              constraintText={`Up to ${maxAttachments} images, ${MAX_MB} MB each.`}
              i18nStrings={{
                uploadButtonText: (multiple) => (multiple ? "Choose images" : "Choose image"),
                dropzoneText: (multiple) =>
                  multiple ? "Drop screenshots here" : "Drop a screenshot here",
                removeFileAriaLabel: (i) => `Remove image ${i + 1}`,
                limitShowFewer: "Show fewer",
                limitShowMore: "Show more",
                errorIconAriaLabel: "Error",
              }}
            />
          </FormField>
        </SpaceBetween>
      </Modal>

      {/* Delete confirmation. */}
      <Modal
        visible={removeTarget !== null}
        onDismiss={() => !removing && setRemoveTarget(null)}
        header="Delete feedback"
        footer={
          <Box float="right">
            <SpaceBetween direction="horizontal" size="xs">
              <Button variant="link" onClick={() => setRemoveTarget(null)} disabled={removing}>
                Cancel
              </Button>
              <Button
                variant="primary"
                loading={removing}
                onClick={() => removeTarget && remove(removeTarget)}
              >
                Delete
              </Button>
            </SpaceBetween>
          </Box>
        }
      >
        Delete your feedback <b>{removeTarget?.title}</b>? Its screenshots will be removed too. This
        can't be undone.
      </Modal>
    </ContentLayout>
  );
}
