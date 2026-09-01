// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import Icon from "@cloudscape-design/components/icon";
import Box from "@cloudscape-design/components/box";

// Shared "this came from an AI agent" marker, used everywhere one of the platform's
// Strands-on-AgentCore agents surfaces (dataset investigate, failure triage, results
// interpreter, run-config advisor, …). Keeps the agent provenance visually consistent:
// the same gen-ai glyph + an "AI" label so a user always knows the output is
// agent-generated (and advisory), not a deterministic computation.

/** Inline pill: a gen-ai glyph + "AI" label. Drop next to an agent panel/header title. */
export function AgentBadge({ label = "AI agent" }: { label?: string }) {
  return (
    <Box
      display="inline-block"
      color="text-status-info"
      fontSize="body-s"
      fontWeight="bold"
    >
      <Icon name="gen-ai" size="small" /> {label}
    </Box>
  );
}

/** A one-line caption explaining the output is from an AI agent (advisory). Render
 *  directly under an agent panel's header. */
export function AgentCaption({ children }: { children: React.ReactNode }) {
  return (
    <Box variant="small" color="text-body-secondary">
      <Icon name="gen-ai" size="small" /> {children}
    </Box>
  );
}

/** The left-nav marker: a small standalone gen-ai glyph used in a SideNavigation
 *  item's `info` slot to flag pages that host an AI agent. */
export function AgentNavMark() {
  return (
    <Box color="text-status-info" display="inline-block" padding={{ left: "xxs" }}>
      <Icon name="gen-ai" size="small" />
    </Box>
  );
}
