// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import Flashbar, { type FlashbarProps } from "@cloudscape-design/components/flashbar";

// App-level notifications. One Flashbar lives in AppLayout's `notifications` slot;
// any component calls useNotify() to push transient success/error/info/warning
// toasts instead of hand-rolling its own inline <Alert> + useState (which ~10
// pages were doing). Each flash is dismissible, stacks, and collapses ("N more")
// when several pile up; success/info auto-dismiss, errors persist until dismissed.

type NotifyType = "success" | "error" | "info" | "warning" | "in-progress";

interface NotifyInput {
  type: NotifyType;
  content: ReactNode;
  header?: string;
  // Auto-dismiss after this many ms. Defaults: success/info 5s, warning 8s,
  // error/in-progress never (must be dismissed or replaced).
  autoDismissMs?: number;
}

interface NotifyApi {
  notify: (n: NotifyInput) => string; // returns the flash id
  dismiss: (id: string) => void;
  clear: () => void;
}

const Ctx = createContext<NotifyApi | null>(null);

const DEFAULT_DISMISS: Record<NotifyType, number | null> = {
  success: 5000,
  info: 5000,
  warning: 8000,
  error: null,
  "in-progress": null,
};

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<FlashbarProps.MessageDefinition[]>([]);
  const seq = useRef(0);
  const timers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const dismiss = useCallback((id: string) => {
    setItems((prev) => prev.filter((i) => i.id !== id));
    const t = timers.current[id];
    if (t) {
      clearTimeout(t);
      delete timers.current[id];
    }
  }, []);

  const notify = useCallback(
    (n: NotifyInput): string => {
      // Dedupe identical messages (same type + content + header) so a retry loop
      // can't stack ten copies. If a dupe is already showing, reuse its id and
      // schedule NO new timer (the original keeps its lifecycle).
      let dupeId = "";
      const id = `flash-${++seq.current}`;
      const item: FlashbarProps.MessageDefinition = {
        id,
        type: n.type,
        header: n.header,
        content: n.content,
        dismissible: true,
        onDismiss: () => dismiss(id),
        // in-progress shows the spinner; the rest are static.
        loading: n.type === "in-progress",
      };
      setItems((prev) => {
        const dupe = prev.find(
          (p) => p.type === item.type && p.content === item.content && p.header === item.header
        );
        if (dupe) {
          dupeId = dupe.id ?? "";
          return prev;
        }
        return [item, ...prev];
      });
      if (dupeId) return dupeId; // no timer for the unused id
      const ttl = n.autoDismissMs ?? DEFAULT_DISMISS[n.type];
      if (ttl) {
        timers.current[id] = setTimeout(() => dismiss(id), ttl);
      }
      return id;
    },
    [dismiss]
  );

  // Clear any pending auto-dismiss timers if the provider ever unmounts, so a
  // late callback can't setState on an unmounted tree.
  useEffect(() => {
    const t = timers.current;
    return () => Object.values(t).forEach(clearTimeout);
  }, []);

  const clear = useCallback(() => {
    Object.values(timers.current).forEach(clearTimeout);
    timers.current = {};
    setItems([]);
  }, []);

  const api = useMemo<NotifyApi>(() => ({ notify, dismiss, clear }), [notify, dismiss, clear]);

  return (
    <Ctx.Provider value={api}>
      {/* The Flashbar itself is rendered by the app shell via <AppNotifications/>;
          here we only provide the API + state. */}
      <FlashContext.Provider value={items}>{children}</FlashContext.Provider>
    </Ctx.Provider>
  );
}

// Separate read-only context so AppLayout can render the Flashbar while the rest
// of the tree uses the imperative notify() api.
const FlashContext = createContext<FlashbarProps.MessageDefinition[]>([]);

export function AppNotifications() {
  const items = useContext(FlashContext);
  if (items.length === 0) return null;
  return <Flashbar items={items} stackItems />;
}

export function useNotify(): NotifyApi {
  const ctx = useContext(Ctx);
  if (!ctx) {
    // A no-op fallback so a component used outside the provider never crashes.
    return {
      notify: () => "",
      dismiss: () => undefined,
      clear: () => undefined,
    };
  }
  return ctx;
}

// Convenience: turn a thrown error into a notification message string.
export function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
