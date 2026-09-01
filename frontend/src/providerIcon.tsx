// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

/** Provider icon: real vendor logos from @lobehub/icons (purpose-built for AI
 * model providers), with a colored-monogram FALLBACK for any provider lobehub
 * doesn't cover.
 *
 * We deep-import each logo's raw SVG component (`.../<Provider>/components/Color`
 * or `Mono`) rather than the package's top-level entry — the raw SVGs depend
 * only on react (no antd / @lobehub/ui), so we avoid that heavy peer-dep tree.
 * Each component takes a `size` prop and renders a 24x24 viewBox SVG. */

import type { ComponentType } from "react";

// Color logos (preferred). Keyed by a substring of catalog.provider_for's
// friendly name (matched lowercase, first hit wins).
import Qwen from "@lobehub/icons/es/Qwen/components/Color";
import DeepSeek from "@lobehub/icons/es/DeepSeek/components/Color";
import Mistral from "@lobehub/icons/es/Mistral/components/Color";
import Meta from "@lobehub/icons/es/Meta/components/Color";
import Gemma from "@lobehub/icons/es/Gemma/components/Color";
import Microsoft from "@lobehub/icons/es/Microsoft/components/Color";
import Nvidia from "@lobehub/icons/es/Nvidia/components/Color";
import InternLM from "@lobehub/icons/es/InternLM/components/Color";
import ChatGLM from "@lobehub/icons/es/ChatGLM/components/Color";
import Nova from "@lobehub/icons/es/Nova/components/Color";
import Aws from "@lobehub/icons/es/Aws/components/Color";
import TII from "@lobehub/icons/es/TII/components/Color";
import Claude from "@lobehub/icons/es/Claude/components/Color";
import Cohere from "@lobehub/icons/es/Cohere/components/Color";
// Mono-only logos (no color variant shipped).
import OpenAI from "@lobehub/icons/es/OpenAI/components/Mono";
import IBM from "@lobehub/icons/es/IBM/components/Mono";
import Liquid from "@lobehub/icons/es/Liquid/components/Mono";

type LogoComponent = ComponentType<{ size?: number | string }>;

// match (substring of the friendly provider name, lowercase) → logo component.
// Order matters: first match wins. Covers every provider in catalog.provider_for
// plus the Nova/Anthropic baselines; unmatched providers fall back to a monogram.
const PROVIDER_LOGOS: { match: string; Logo: LogoComponent }[] = [
  { match: "alibaba", Logo: Qwen },
  { match: "qwen", Logo: Qwen },
  { match: "anthropic", Logo: Claude }, // Claude baselines (Anthropic ships mono only)
  { match: "claude", Logo: Claude },
  { match: "cohere", Logo: Cohere }, // Command R/R+ baselines
  { match: "command", Logo: Cohere },
  { match: "deepseek", Logo: DeepSeek },
  { match: "mistral", Logo: Mistral },
  { match: "meta", Logo: Meta },
  { match: "llama", Logo: Meta },
  { match: "google", Logo: Gemma },
  { match: "gemma", Logo: Gemma },
  { match: "microsoft", Logo: Microsoft },
  { match: "phi", Logo: Microsoft },
  { match: "nvidia", Logo: Nvidia },
  { match: "internlm", Logo: InternLM },
  { match: "zhipu", Logo: ChatGLM }, // GLM
  { match: "thudm", Logo: ChatGLM },
  { match: "glm", Logo: ChatGLM },
  { match: "amazon", Logo: Nova }, // Nova baselines
  { match: "nova", Logo: Nova },
  { match: "aws", Logo: Aws },
  { match: "falcon", Logo: TII },
  { match: "tii", Logo: TII },
  { match: "openai", Logo: OpenAI }, // gpt-oss
  { match: "gpt", Logo: OpenAI },
  { match: "ibm", Logo: IBM }, // Granite
  { match: "granite", Logo: IBM },
  { match: "liquid", Logo: Liquid }, // LFM2
];

// --- monogram fallback (for providers lobehub doesn't cover) ----------------

const FALLBACK_COLORS: { match: string; color: string; short: string }[] = [
  { match: "openbmb", color: "#00a3a3", short: "Cp" }, // MiniCPM
  { match: "minicpm", color: "#00a3a3", short: "Cp" },
  { match: "allen", color: "#f0529c", short: "Ol" }, // OLMo
];

function hashColor(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return `hsl(${h}, 45%, 45%)`;
}

function Monogram({ provider, size }: { provider: string; size: number }) {
  const lower = (provider || "").toLowerCase();
  const hit = FALLBACK_COLORS.find((p) => lower.includes(p.match));
  const color = hit?.color ?? hashColor(lower);
  const label = hit?.short ?? (provider || "?").slice(0, 2);
  return (
    <span
      aria-hidden="true"
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: size,
        height: size,
        minWidth: size,
        borderRadius: "50%",
        background: color,
        color: "#fff",
        fontSize: size * 0.42,
        fontWeight: 700,
        lineHeight: 1,
        letterSpacing: "-0.02em",
      }}
      title={provider}
    >
      {label}
    </span>
  );
}

export function ProviderIcon({ provider, size = 22 }: { provider: string; size?: number }) {
  const lower = (provider || "").toLowerCase();
  const hit = PROVIDER_LOGOS.find((p) => lower.includes(p.match));
  if (hit) {
    const { Logo } = hit;
    // Wrap so the logo sits in a consistent square box like the monogram did.
    return (
      <span
        title={provider}
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: size,
          height: size,
          minWidth: size,
        }}
      >
        <Logo size={size} />
      </span>
    );
  }
  return <Monogram provider={provider} size={size} />;
}
