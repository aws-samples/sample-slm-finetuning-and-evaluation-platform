// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import SpaceBetween from "@cloudscape-design/components/space-between";
import Box from "@cloudscape-design/components/box";
import Badge from "@cloudscape-design/components/badge";
import Table from "@cloudscape-design/components/table";

import type { PreviewRow } from "./api";

function badgeColor(role: string): "blue" | "green" | "grey" | "red" {
  switch (role) {
    case "assistant":
      return "green";
    case "user":
      return "blue";
    case "system":
      return "grey";
    default:
      return "red";
  }
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

// Render a chat row's turns as role-coloured badges + content snippets.
export function MessagesCell({ row }: { row: PreviewRow }) {
  return (
    <SpaceBetween size="xs">
      {row.messages.map((turn, i) => (
        <Box key={i}>
          <Badge color={badgeColor(turn.role)}>{turn.role}</Badge>{" "}
          <span>{truncate(turn.content, 140)}</span>
        </Box>
      ))}
    </SpaceBetween>
  );
}

// A small embedded table of chat rows (numbered), shared by validation +
// split previews.
export function MessagesTable({
  rows,
  empty,
}: {
  rows: PreviewRow[];
  empty: string;
}) {
  return (
    <Table
      variant="embedded"
      columnDefinitions={[
        { id: "row", header: "#", cell: (r) => r.index + 1, width: 60 },
        { id: "messages", header: "Messages", cell: (r) => <MessagesCell row={r} /> },
      ]}
      items={rows.map((row, index) => ({ ...row, index }))}
      empty={empty}
    />
  );
}
