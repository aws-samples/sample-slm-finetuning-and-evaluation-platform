// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import ReactDOM from "react-dom/client";
import "@cloudscape-design/global-styles/index.css";
import App from "./App";
import { ensureAuth } from "./auth";
import { applyStoredMode } from "./theme";

// Apply the persisted (or OS-preferred) light/dark mode BEFORE the first render,
// so a dark-mode user never sees a light flash on reload.
applyStoredMode();

// Gate the app behind Cognito login when auth is configured (runtime
// /config.json). In open mode (local dev / no Cognito) this resolves
// immediately and the app renders as before.
ensureAuth().then(() => {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
});
