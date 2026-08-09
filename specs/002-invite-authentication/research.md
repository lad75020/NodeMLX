# Research: Invite-Only Authentication

## Existing implementation findings

- The SQLite store already persists `users`, `sessions.expires_at`, and `sessions.last_seen_at`; it uses scrypt password hashes and HMAC-signed WebSocket JWTs.
- The current session reader updates only `last_seen_at`, while the cookie and database expiry remain at a 30-day absolute expiry. This does not meet the clarified rolling seven-day inactivity policy.
- Login has a generic `401 Invalid username or password` response once both values are supplied. The malformed/missing-credential branch is retained as a request-shape validation error, not an account-existence signal.
- `/api/auth/register` is already a disabled placeholder and CLI provisioning is implemented by `scripts/onboard-user.js`.
- WebSocket authentication validates the JWT and current SQLite session at connection time, but the captured session is otherwise not rechecked. A reset must therefore revalidate before processing later messages.
- Angular uses the HTTP cookie for restoration and retains only the narrow WebSocket JWT in `sessionStorage`; its current contract needs no schema change.
- NativeMac uses the shared URL-session cookie store, but previously attempted a durable cached WebSocket token before `/api/auth/me`. The feature changes this order so revoked/expired sessions are discovered and local token state is cleared.

## Design decisions

- Use the existing `sessions.expires_at` field as the rolling inactivity deadline; do not introduce another table or dependency.
- Renew `expires_at` and `last_seen_at` together on a valid session read. HTTP callers reissue the cookie with the same seven-day lifetime.
- Retain a stateless JWT signature check and current-session lookup. A live WebSocket rechecks the current session for every message, closing with the existing unauthorized semantics after reset, logout, or expiry.
- Provide `npm run user:reset -- <username>` using the established CLI, a hidden interactive prompt by default, and an optional existing `--password` automation input. The CLI reports only success/failure and never outputs a password or hash.
