// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Light/dark theme via Cloudscape's global visual mode. The mode is applied to
// <body> by applyMode (Cloudscape reads design tokens off that), persisted in
// localStorage, and applied as early as possible (main.tsx, before render) so
// there's no light-flash on a dark-mode reload.
import { applyMode, Mode } from "@cloudscape-design/global-styles";

const STORAGE_KEY = "slm-color-mode";

export function storedMode(): Mode {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "dark") return Mode.Dark;
    if (v === "light") return Mode.Light;
  } catch {
    // localStorage may be unavailable (private mode / SSR) — fall through.
  }
  // First visit: respect the OS preference.
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return Mode.Dark;
  }
  return Mode.Light;
}

export function applyStoredMode(): Mode {
  const mode = storedMode();
  applyMode(mode);
  return mode;
}

export function isDark(mode: Mode): boolean {
  return mode === Mode.Dark;
}

// Toggle, persist, apply, and return the new mode (so the caller can update UI state).
export function toggleMode(current: Mode): Mode {
  const next = current === Mode.Dark ? Mode.Light : Mode.Dark;
  applyMode(next);
  try {
    localStorage.setItem(STORAGE_KEY, next === Mode.Dark ? "dark" : "light");
  } catch {
    // ignore persistence failure — the mode still applies for this session.
  }
  return next;
}
