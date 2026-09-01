// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Small shared UI bits for Hugging Face-sourced datasets:
//  - HfBadge: a "🤗 HF" badge so HF-imported datasets are visually distinct.
//  - MessagesPreview: renders chat rows as readable, colored JSON instead of a
//    raw escaped one-liner (no \n\n soup, keys highlighted).

import Badge from "@cloudscape-design/components/badge";
import Box from "@cloudscape-design/components/box";
import Popover from "@cloudscape-design/components/popover";

// The Hugging Face mark (🤗) + label. Used wherever a dataset is HF-sourced.
export function HfBadge({ dataset }: { dataset?: string | null }) {
  const badge = <Badge color="blue">🤗 HF</Badge>;
  if (!dataset) return badge;
  return (
    <Popover header="Hugging Face dataset" position="top" triggerType="custom" dismissButton={false} content={dataset}>
      {badge}
    </Popover>
  );
}

// Prefix a display name with "hf:" when the dataset came from Hugging Face, so
// it's identifiable at a glance even without the badge.
export function datasetDisplayName(name: string | null | undefined, source?: string | null): string {
  const base = name || "(unnamed)";
  return source === "huggingface" && !base.startsWith("hf:") ? `hf: ${base}` : base;
}

// Role label colors as Cloudscape semantic text tokens so they track light/dark
// mode (the old fixed hex — esp. tool #8d6e00 + the #414d5c fallback — went
// near-invisible on the now-tokenized dark preview surface).
const ROLE_COLORS: Record<string, string> = {
  system: "var(--color-text-status-info, #0972d3)",
  user: "var(--color-text-link-default, #0972d3)",
  assistant: "var(--color-text-status-success, #037f51)",
  tool: "var(--color-text-status-warning, #8d6e00)",
};
const ROLE_COLOR_FALLBACK = "var(--color-text-body-secondary, #5f6b7a)";

type Turn = { role: string; content: string };
type ChatRow = { messages?: Turn[] } & Record<string, unknown>;

// Render one chat row's turns as a clean, color-coded block. Each turn shows its
// role (colored) and content on its own lines — newlines are rendered, not
// escaped — so the structure is readable at a glance.
function RowBlock({ row }: { row: ChatRow }) {
  const msgs = Array.isArray(row.messages) ? row.messages : null;
  if (!msgs) {
    // Not a chat row — fall back to pretty JSON.
    return (
      <Box variant="code" fontSize="body-s">
        <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(row, null, 2)}</pre>
      </Box>
    );
  }
  return (
    <div
      style={{
        // Cloudscape tokens so the preview block adapts to dark mode (the old
        // hardcoded light grey + near-white bg + near-black text were invisible
        // on a dark surface).
        border: "1px solid var(--color-border-divider-default, #e9ebed)",
        borderRadius: 8,
        padding: "8px 10px",
        background: "var(--color-background-container-content, #fbfbfb)",
      }}
    >
      {msgs.map((t, i) => (
        <div key={i} style={{ marginBottom: i < msgs.length - 1 ? 6 : 0 }}>
          <span
            style={{
              color: ROLE_COLORS[t.role] ?? ROLE_COLOR_FALLBACK,
              fontWeight: 700,
              fontFamily: "monospace",
              fontSize: 12,
            }}
          >
            {t.role}
          </span>
          <div
            style={{
              whiteSpace: "pre-wrap",
              fontSize: 13,
              marginTop: 2,
              color: "var(--color-text-body-default, #16191f)",
              wordBreak: "break-word",
            }}
          >
            {t.content}
          </div>
        </div>
      ))}
    </div>
  );
}

export function MessagesPreview({ rows, limit = 2 }: { rows: ChatRow[]; limit?: number }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {rows.slice(0, limit).map((r, i) => (
        <RowBlock key={i} row={r} />
      ))}
    </div>
  );
}
