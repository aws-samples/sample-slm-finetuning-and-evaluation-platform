// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Cognito Hosted-UI login for the SPA.
//
// Flow (implicit grant — no backend token exchange needed for this SPA):
//   1. On load, fetch runtime /config.json (written by CDK) for the Cognito
//      domain / client id / region. If auth isn't configured, the app runs
//      open (local dev / pre-Cognito deploys).
//   2. If we just came back from Hosted UI, the URL hash carries the tokens —
//      capture the id_token into sessionStorage and clean the URL.
//   3. If we have no token, redirect to the Hosted UI login.
//   4. Install a global fetch wrapper so every /api request carries
//      `Authorization: Bearer <id_token>`. A 401 clears the token and re-logs-in.

export interface RuntimeConfig {
  // e.g. slm-platform-<stack-guid>.auth.us-east-1.amazoncognito.com. The
  // discriminator is the CloudFormation stack id, never the AWS account number —
  // this file is served unauthenticated (see stack.py's cognito_domain_prefix).
  cognitoDomain?: string;
  cognitoClientId?: string;
  region?: string;
}

const TOKEN_KEY = "slm_id_token";

let config: RuntimeConfig = {};

async function loadConfig(): Promise<RuntimeConfig> {
  try {
    const res = await fetch("/config.json", { cache: "no-store" });
    if (res.ok) return (await res.json()) as RuntimeConfig;
  } catch {
    /* no runtime config → auth disabled */
  }
  return {};
}

function authConfigured(): boolean {
  return Boolean(config.cognitoDomain && config.cognitoClientId);
}

function redirectUri(): string {
  // Back to the app root; Hosted UI requires an exact-match callback URL.
  return `${window.location.origin}/`;
}

function loginUrl(): string {
  const u = new URL(`https://${config.cognitoDomain}/login`);
  u.searchParams.set("client_id", config.cognitoClientId!);
  u.searchParams.set("response_type", "token"); // implicit grant
  u.searchParams.set("scope", "openid email profile");
  u.searchParams.set("redirect_uri", redirectUri());
  return u.toString();
}

export function logout(): void {
  sessionStorage.removeItem(TOKEN_KEY);
  if (authConfigured()) {
    const u = new URL(`https://${config.cognitoDomain}/logout`);
    u.searchParams.set("client_id", config.cognitoClientId!);
    u.searchParams.set("logout_uri", redirectUri());
    window.location.assign(u.toString());
  }
}

function captureTokenFromHash(): boolean {
  if (!window.location.hash.includes("id_token=")) return false;
  const params = new URLSearchParams(window.location.hash.slice(1));
  const idToken = params.get("id_token");
  if (idToken) {
    sessionStorage.setItem(TOKEN_KEY, idToken);
    // Strip the token fragment from the URL.
    window.history.replaceState({}, document.title, window.location.pathname);
    return true;
  }
  return false;
}

function getToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export interface CurrentUser {
  email?: string;
  name?: string;
  username?: string; // Cognito username / sub fallback
  firstName?: string; // given_name, else first token of name, else shortUsername
  // The username with any identity-provider prefix stripped: usernames created by
  // an external IdP arrive as "<IdP>_<username>", so this is the bare username.
  shortUsername?: string;
}

// Decode the stored id_token's claims for DISPLAY only (name/email in the top
// nav). No signature check — the token was already verified by Cognito at login
// and by API Gateway on every request; this is purely cosmetic. Returns null in
// open/local mode (no token), so the UI can show a "local" state instead.
export function getCurrentUser(): CurrentUser | null {
  const token = getToken();
  if (!token) return null;
  try {
    const payload = token.split(".")[1];
    // base64url → base64, then decode UTF-8 safely.
    const json = decodeURIComponent(
      atob(payload.replace(/-/g, "+").replace(/_/g, "/"))
        .split("")
        .map((c) => "%" + c.charCodeAt(0).toString(16).padStart(2, "0"))
        .join("")
    );
    const c = JSON.parse(json) as Record<string, unknown>;
    const username =
      (typeof c["cognito:username"] === "string" && c["cognito:username"]) ||
      (typeof c.sub === "string" && c.sub) ||
      undefined;
    const givenName = typeof c.given_name === "string" ? c.given_name : undefined;
    const name = typeof c.name === "string" ? c.name : undefined;
    // Usernames minted by an external IdP look like "<IdP>_<username>", so the
    // bare username is the part after the prefix. Native users have no prefix.
    const shortUsername =
      username && username.includes("_") ? username.split("_").pop() : username;
    return {
      email: typeof c.email === "string" ? c.email : undefined,
      name,
      username,
      shortUsername,
      firstName: givenName || (name ? name.split(" ")[0] : undefined) || shortUsername,
    };
  } catch {
    return null;
  }
}

function installFetchInterceptor(): void {
  const original = window.fetch.bind(window);
  window.fetch = async (input, init = {}) => {
    const url = typeof input === "string" ? input : (input as Request).url;
    const isApi = url.startsWith("/api") || url.includes("/api/");
    const token = getToken();
    if (isApi && token) {
      const headers = new Headers(init.headers || (input as Request).headers);
      headers.set("Authorization", `Bearer ${token}`);
      init = { ...init, headers };
    }
    const res = await original(input as RequestInfo, init);
    if (isApi && res.status === 401 && authConfigured()) {
      sessionStorage.removeItem(TOKEN_KEY);
      window.location.assign(loginUrl());
    }
    return res;
  };
}

// Resolve auth before the app renders. Returns when the app may proceed
// (either auth is disabled, or we hold a valid id token). Otherwise redirects
// to the Hosted UI and never resolves.
export async function ensureAuth(): Promise<void> {
  config = await loadConfig();
  if (!authConfigured()) return; // open mode (local dev / no Cognito)

  installFetchInterceptor();
  captureTokenFromHash();

  if (!getToken()) {
    window.location.assign(loginUrl());
    // Halt rendering while the browser navigates away.
    await new Promise(() => {});
  }
}
