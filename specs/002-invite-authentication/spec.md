# Feature Specification: Invite-Only Authentication

**Feature Branch**: `feature/time-machine-invite-authentication`
**Created**: 2026-08-09
**Status**: Clarified
**Input**: Deliver secure invited-user sign-in and authenticated local sessions.

## User Scenarios & Testing

### User Story 1 - Provision and sign in locally (Priority: P1)

A local administrator provisions an account using a CLI-only workflow; an invited user signs in through either the Angular browser client or native macOS client and reaches authenticated chat.

**Affected clients**: both
**Why this priority**: Authentication is the boundary protecting every per-user chat operation.
**Independent Test**: Create a test user in an isolated SQLite database, sign in with each client contract, and verify the authenticated identity response.

**Acceptance Scenarios**:

1. **Given** no self-registration endpoint, **When** an administrator creates a valid user with the local CLI, **Then** the user can sign in with that username and password.
2. **Given** valid credentials, **When** either client calls the login endpoint, **Then** it receives the authenticated user and a session usable by HTTP and WebSocket chat.
3. **Given** invalid credentials, **When** a sign-in attempt occurs, **Then** no session is created and the response does not disclose whether the username exists.

### User Story 2 - Maintain and end an authenticated session (Priority: P2)

An authenticated user can restore a session across normal application use, and activity renews the session lifetime without silently extending it after expiry or logout.

**Affected clients**: both
**Why this priority**: Local users should not be repeatedly interrupted while expired or revoked credentials must remain ineffective.
**Independent Test**: Exercise login, session restoration, authenticated access, renewal on activity, logout, and use of an expired session.

**Acceptance Scenarios**:

1. **Given** an authenticated active session, **When** the user accesses an authenticated HTTP or WebSocket operation, **Then** the session lifetime is renewed according to the configured policy.
2. **Given** logout or an expired session, **When** either client restores or uses it, **Then** it is unauthenticated and cached WebSocket credentials are removed.

### User Story 3 - Administrator password reset (Priority: P3)

A local administrator can reset an invited user's password using a local CLI; no email recovery or browser self-service reset is exposed.

**Affected clients**: none
**Why this priority**: The product remains local-first while retaining a safe recovery path.
**Independent Test**: Reset a password in an isolated database, verify the old password fails and the new password succeeds, and verify existing sessions are invalidated.

## Edge Cases

- Duplicate, malformed, and overlong usernames are rejected consistently by provisioning and authentication.
- Concurrent or stale sessions cannot regain access after a password reset, expiry, or logout.
- Missing/invalid session cookies and malformed WebSocket tokens fail closed without exposing identity information.
- Angular and NativeMac clients surface generic, actionable errors without storing passwords or durable access tokens.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST keep account provisioning local and CLI-only; it MUST NOT add self-registration, invitation-code, or administrative web UI flows.
- **FR-002**: The system MUST authenticate invited users with a password hash and issue an HTTP session plus the existing narrowly scoped WebSocket credential.
- **FR-003**: Sessions MUST use rolling renewal on valid authenticated activity, expire deterministically, and be invalidated on logout.
- **FR-004**: A local CLI MUST let an administrator reset a user's password without printing the password or hash; reset MUST invalidate existing sessions for that user.
- **FR-005**: Login failures MUST use a generic response that does not reveal account existence.
- **FR-006**: Angular and NativeMac clients MUST restore, clear, and use authentication state consistently with the backend contract.

### API and Streaming Contract

- **Endpoints/events affected**: `/api/auth/login`, `/api/auth/logout`, `/api/auth/me`, authenticated `/ws` handshake.
- **Authentication behavior**: Cookie-backed rolling session remains the HTTP mechanism; existing short-lived WebSocket credential remains compatible.
- **Request/response or event payload changes**: Any session-expiry or error detail must preserve backward compatibility for both clients.
- **Documentation impact**: Update `docs/API.md` and `api-docs.js` for public behavior changes.

### Persistence and Runtime Impact

- **Stores affected**: SQLite `users` and `sessions` only.
- **Migration/backfill/compatibility**: Existing users and sessions must remain readable; schema changes require an additive, documented migration. Password reset must remove existing sessions atomically.
- **Inference runtime affected**: none.
- **Configuration/environment variables affected**: session lifetime configuration only if required for rolling policy.

## Non-Goals

- Email delivery, OAuth, social login, passwordless authentication, invitation codes, and browser-based administration.
- Mongo chat schema changes, model-runtime changes, and unrelated client redesign.

## Success Criteria

- **SC-001**: A CLI-provisioned user can sign in and restore an authenticated session from both clients.
- **SC-002**: Expired, logged-out, and password-reset sessions are rejected; valid activity renews a rolling session.
- **SC-003**: Focused backend tests, `npm test`, applicable Angular build, and NativeMac build pass, with documented client smoke paths.

## Assumptions and Open Questions

- **Decision**: Provisioning remains CLI-only; no self-registration or invitation codes.
- **Decision**: Session lifetime is rolling/renewable with a maximum 7-day inactivity duration.
- **Decision**: Password recovery is an administrator-only local CLI reset; no user self-service or email recovery.
- **Resolved**: The rolling lifetime is seven days; valid authenticated activity renews the expiry to seven days from that activity.

## Verification Plan

- **Automated**: Focused auth/session tests plus `npm test`.
- **Build**: `npm run build` and `(cd NativeMac && swift build)` because both client authentication surfaces are in scope.
- **Smoke test**: Provision user, sign in, restore session, open authenticated WebSocket, log out, reset password, and verify old sessions fail.
