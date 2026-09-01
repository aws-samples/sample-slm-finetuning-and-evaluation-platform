// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

// Cognito Hosted-UI login for the SPA.
//
// Flow (authorization CODE grant with PKCE — no client secret, no backend needed):
//   1. On load, fetch runtime /config.json (written by CDK) for the Cognito
//      domain / client id / region. If auth isn't configured, the app runs
//      open (local dev / pre-Cognito deploys).
//   2. To sign in, mint a random PKCE verifier and a `state` nonce, keep both in
//      sessionStorage, and send the SHA-256 of the verifier as code_challenge.
//   3. Hosted UI redirects back with `?code=&state=`. Reject the response unless
//      `state` matches what we stored — that binding is what makes a crafted
//      callback link useless — then exchange the code at /oauth2/token using the
//      verifier, and keep the id_token in sessionStorage.
//   4. Install a global fetch wrapper so every /api request carries
//      `Authorization: Bearer <id_token>`. A 401 clears the token and re-logs-in.
//
// Why not the implicit grant: it returns the id_token in the URL fragment, which
// leaks through history and referrers, and the SPA cannot tell a token it asked
// for from one an attacker pasted in — so a single link could put a victim into
// the attacker's tenant. A code is single-use and worthless without the verifier
// that never leaves the browser which started the flow.

export interface RuntimeConfig {
  // e.g. slm-platform-<stack-guid>.auth.us-east-1.amazoncognito.com. The
  // discriminator is the CloudFormation stack id, never the AWS account number —
  // this file is served unauthenticated (see stack.py's cognito_domain_prefix).
  cognitoDomain?: string;
  cognitoClientId?: string;
  region?: string;
}

const TOKEN_KEY = "slm_id_token";
const VERIFIER_KEY = "slm_pkce_verifier";
const STATE_KEY = "slm_oauth_state";

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

// base64url of arbitrary bytes (no padding) — the encoding PKCE and OAuth use.
function b64url(bytes: Uint8Array): string {
  let out = "";
  for (const b of bytes) out += String.fromCharCode(b);
  return btoa(out).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomUrlSafe(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return b64url(bytes);
}

async function codeChallenge(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return b64url(new Uint8Array(digest));
}

// Start the code+PKCE flow. Stores the verifier and state, then navigates away.
async function beginLogin(): Promise<void> {
  // 32 bytes → 43 base64url chars, the minimum RFC 7636 allows.
  const verifier = randomUrlSafe(32);
  const state = randomUrlSafe(16);
  sessionStorage.setItem(VERIFIER_KEY, verifier);
  sessionStorage.setItem(STATE_KEY, state);
  const u = new URL(`https://${config.cognitoDomain}/oauth2/authorize`);
  u.searchParams.set("client_id", config.cognitoClientId!);
  u.searchParams.set("response_type", "code");
  u.searchParams.set("scope", "openid email profile");
  u.searchParams.set("redirect_uri", redirectUri());
  u.searchParams.set("state", state);
  u.searchParams.set("code_challenge", await codeChallenge(verifier));
  u.searchParams.set("code_challenge_method", "S256");
  window.location.assign(u.toString());
}

// Exchange `?code=` for tokens, if this load is a callback. Returns true when a
// token was obtained. The state check is the security-critical part: without it
// the app would accept a code (or in the old implicit flow, a token) that some
// other page initiated, which is exactly the session-fixation this replaces.
async function completeLoginFromQuery(): Promise<boolean> {
  const q = new URLSearchParams(window.location.search);
  const code = q.get("code");
  const returnedState = q.get("state");
  const stored = sessionStorage.getItem(STATE_KEY);
  const verifier = sessionStorage.getItem(VERIFIER_KEY);
  // Always clear the one-shot values, whether or not this succeeds.
  sessionStorage.removeItem(STATE_KEY);
  sessionStorage.removeItem(VERIFIER_KEY);
  const clean = () =>
    window.history.replaceState({}, document.title, window.location.pathname);

  if (q.get("error")) {
    clean();
    return false;
  }
  if (!code) return false;
  if (!stored || returnedState !== stored || !verifier) {
    // Unsolicited or replayed callback — drop it and start a fresh flow.
    clean();
    return false;
  }
  try {
    const res = await fetch(`https://${config.cognitoDomain}/oauth2/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        client_id: config.cognitoClientId!,
        code,
        redirect_uri: redirectUri(),
        code_verifier: verifier,
      }).toString(),
    });
    if (!res.ok) {
      clean();
      return false;
    }
    const tokens = (await res.json()) as { id_token?: string };
    if (!tokens.id_token) {
      clean();
      return false;
    }
    sessionStorage.setItem(TOKEN_KEY, tokens.id_token);
    clean();
    return true;
  } catch {
    clean();
    return false;
  }
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
      await beginLogin();
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
  await completeLoginFromQuery();

  if (!getToken()) {
    await beginLogin();
    // Halt rendering while the browser navigates away.
    await new Promise(() => {});
  }
}
