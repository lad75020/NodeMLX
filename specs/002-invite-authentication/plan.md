# Implementation Plan: Invite-Only Authentication

## Scope and compatibility

- Keep user creation and password recovery local CLI operations only; no registration, invitation-code, or web administration route is added.
- Preserve the existing HTTP-only `nodemlx_session` cookie and `/ws?token=` WebSocket credential contract for Angular and NativeMac.
- Change the configured session inactivity lifetime from 30 days to seven days. Every valid authenticated HTTP lookup or WebSocket operation renews both the SQLite expiry and, for HTTP, the cookie lifetime.
- Make password reset update the password hash and delete all sessions for that user in one SQLite transaction. Existing WebSocket activity must revalidate its session so a reset takes effect on already-open connections.

## Affected modules

| Area | Files | Change |
|---|---|---|
| Auth persistence | `src/auth/sqlite-store.js` | Rolling expiry, session revalidation, transactional password reset helpers. |
| Runtime configuration | `src/config.js` | Seven-day session default. |
| Local administration | `scripts/onboard-user.js`, `package.json` | Add a reset subcommand/script with hidden prompt support and no secret output. |
| HTTP/WebSocket protocol | `src/bootstrap.js` | Renew the cookie on authenticated HTTP activity and revalidate/renew each WebSocket operation. |
| Native client | `NativeMac/.../ChatStore.swift` | Always restore cookie session before using cached WebSocket credentials; clear stale persisted token. |
| Documentation | `docs/API.md`, `api-docs.js` | Document rolling seven-day policy, CLI reset, and generic login failures. |
| Tests | `test/auth-store.test.js`, `test/config.test.js` | Cover expiry renewal, reset invalidation, JWT/session validation, and configuration. |

## Storage and failure behavior

`users.password_hash` is updated in place. `sessions` is already keyed by `user_id`; all rows for the target user are deleted in the same transaction as the hash update. No backfill is needed: prior session rows retain their existing `expires_at` values and become rolling once used. A missing user reset fails without disclosing or printing password data. Invalid/expired/revoked sessions remain unauthenticated and their HTTP cookie is cleared.

## Verification

1. Run focused auth and configuration tests, then `npm test`.
2. Run `npm run build` because Angular’s existing auth contract is exercised by the feature.
3. Run `(cd NativeMac && swift build)` after the native stale-token restoration fix.
4. Smoke path: provision an isolated user, login, call `/api/auth/me`, validate a WebSocket token, logout/reset, and confirm old session credentials fail.
