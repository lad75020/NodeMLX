# Quickstart: Runtime Configuration and Bootstrap Validation

## Prerequisites

- macOS 14+ on Apple Silicon with NodeMLX's normal dependencies installed.
- A local NodeMLX checkout with dependencies installed.
- Do not bind the service to an untrusted network for this validation.

## Automated validation

```bash
npm test
```

Expected: focused configuration tests and the complete Node test suite pass.

## Manual smoke matrix

### 1. Safe defaults

```bash
node server.js
```

Expected: the service starts with its documented localhost binding, default Hugging Face cache location, bounded generation settings, and clear readiness output. Stop the process after observing startup.

### 2. Valid custom values

Start with supported custom model/cache/runtime locations and resource values within documented bounds.

Expected: startup identifies the effective configuration and uses the configured locations without source changes.

### 3. Malformed required values

Start with representative malformed required values, including an invalid port and malformed numeric resource setting.

Expected: startup fails before accepting traffic and identifies the setting plus the corrective format or range.

### 4. Out-of-range numeric values

Start with representative finite values below and above the documented resource bounds.

Expected: startup continues, uses the nearest safe bound, and emits an operator warning.

### 5. Optional runtime unavailable

Temporarily point a supported optional image-generation runtime setting at an unavailable interpreter/location, without changing core text-chat prerequisites.

Expected: core service starts; only the affected optional capability is reported unavailable with remediation guidance; no silent fallback is used.

### 6. Required platform prerequisite unavailable

Run on an unsupported platform only in a controlled test environment.

Expected: startup fails with the existing actionable Apple Silicon/macOS prerequisite message.

## Verification evidence

Verified on 2026-08-09:

- `npm test`: passed — 20 tests, 0 failures.
- `npm run build`: passed — Angular production output at `client/dist/chat-client`.
- Safe runtime smoke: with `PORT=19001` and `Z_IMAGE_PYTHON=/definitely/missing/z-image-python`, the server started at `http://127.0.0.1:19001`; `GET /api/auth/me` returned `200` with `{"authenticated":false}`. The unavailable z-image capability logged remediation while core chat remained available.
- Malformed configuration smoke: `PORT=not-a-port node server.js` failed before startup with a setting-specific `ConfigurationError` describing the valid range.
- Clamp smoke: with `PORT=19002`, `MLX_MAX_TOKENS_LIMIT=999999`, and `MLX_MAX_TOKENS=0`, the server started, logged both clamping warnings, and `GET /api/auth/me` returned `200` with `{"authenticated":false}`.

`hermes verify --json` completed bootstrap, build, and test successfully, but its readiness sub-check was not clean because it hard-codes `127.0.0.1:8000` while NodeMLX defaults to `127.0.0.1:18956`; its attempted server also encountered the pre-existing listener on port 18956. The scoped fresh-port smoke tests above provide the runtime/readiness evidence without interrupting that existing process.
