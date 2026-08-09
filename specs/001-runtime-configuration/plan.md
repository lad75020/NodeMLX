# Implementation Plan: Runtime Configuration and Bootstrap

**Branch**: `feature/time-machine-runtime-configuration` | **Date**: 2026-08-09 | **Spec**: [spec.md](spec.md)

## Summary

Make local NodeMLX startup configuration explicit, validated, and documented without changing its local-first deployment model. Consolidate validation in `src/config.js`, preserve existing safe defaults and bounded generation settings, reject malformed startup values before Fastify accepts traffic, and surface optional-runtime unavailability as a disabled capability with an actionable diagnostic. Update the operator documentation and focused Node tests.

## Technical Context

| Area | Current NodeMLX Context | Feature Decision |
|---|---|---|
| Backend | Node.js ESM, Fastify 5, `node-mlx`; entry point `server.js`, assembly `src/bootstrap.js` | Update `src/config.js`; consume validated configuration in `src/bootstrap.js` if startup capability reporting requires it. Keep `server.js` minimal. |
| Web client | Angular 22 in `client/` | No change. Startup/configuration behavior is operator-facing only. |
| Native client | Swift 6 package in `NativeMac/` | No change. No client protocol is introduced. |
| Storage | SQLite auth/session store and Mongo chat store | No schema or persistence change. Existing generated app-secret behavior remains unchanged. |
| Inference | node-mlx, Ollama, llama.cpp, MLX-VLM, DiffusionKit, and z-image adapters | Preserve runtime isolation. Validate capability availability without auto-installing or merging Python environments. |
| Testing | Node built-in test runner via `npm test` | Extend `test/config.test.js` for defaults, malformed values, bounds/clamping, and optional-runtime status decisions. |

**Target platform**: macOS 14+ on Apple Silicon for local inference.  
**Performance/resource constraints**: Configuration parsing is synchronous and negligible at startup. Numeric generation limits remain bounded; invalid numeric settings fail before user traffic, while out-of-range numeric values clamp and warn.  
**Compatibility constraints**: Preserve existing supported environment-variable names and safe local binding default. No Angular, NativeMac, HTTP, WebSocket, SQLite, or Mongo contract change.

## Constitution Check

- [x] Module boundaries are respected: configuration belongs in `src/config.js`; startup capability reporting remains in bootstrap if required.
- [x] Persistence impact is none; no SQLite or Mongo schema, migration, or backfill change is proposed.
- [x] Affected clients and API/WebSocket contracts are identified: neither client nor public protocol changes.
- [x] Runtime-specific dependencies remain isolated: the feature only reports optional-runtime availability and never combines DiffusionKit/z-image environments.
- [x] Applicable verification commands and operator smoke tests are listed.

## Project Structure and Ownership

```text
src/config.js                         # configuration parsing, defaults, validation, bounds
src/bootstrap.js                      # startup assembly and optional-capability diagnostics
server.js                             # minimal entry; remains unchanged unless wiring is required
test/config.test.js                   # focused configuration behavior tests
README.md                             # operator environment-variable/default documentation
src/inference/diffusionkit-runtime.js # existing optional-runtime resolution, consumed but not redesigned
src/inference/z-image-runtime.js      # existing optional-runtime resolution, consumed but not redesigned
```

**Structure decision**: Keep all setting parsing and validation in `src/config.js`. Bootstrap consumes one validated configuration object and reports capability state. Existing runtime adapters remain the authority on interpreter selection; this feature may call their existing discovery helpers but does not duplicate their environment-resolution logic.

## Implementation Phases

1. **Configuration contract** — inventory the currently supported startup variables and defaults in `src/config.js` and README; define a structured validation result/error convention that names the setting, accepted form/range, and corrective action. Preserve `HF_HOME` initialization before app import.
2. **Validated configuration** — normalize ports, binding host, model/runtime paths, temporary-image directory, and resource-sensitive numeric values. Retain bounded clamping for numeric values outside safe limits; fail malformed or unusable required startup values before application assembly.
3. **Optional capability readiness** — add an explicit startup readiness result for optional inference runtimes. A missing optional runtime disables only that capability, logs an actionable diagnostic, and leaves the core chat service available. A missing required Apple Silicon/node-mlx prerequisite remains a startup failure.
4. **Focused tests** — revise `test/config.test.js` and add only any narrowly needed readiness tests. Cover defaults, invalid ports/numeric values, numeric clamping, valid custom locations, unavailable required locations, and unavailable optional capability outcomes.
5. **Operator documentation and verification** — correct README defaults to match implementation (including the currently divergent port value), document every supported startup setting and optional-runtime behavior, run `npm test`, and execute the local startup smoke matrix in `quickstart.md`.

## Verification Commands

```bash
npm test
# Manual smoke checks from quickstart.md; do not expose the server beyond localhost.
```

`npm run build` and `(cd NativeMac && swift build)` are not required because this plan does not modify `client/` or `NativeMac/`.

## Complexity Tracking

| Added complexity | Why needed | Simpler alternative considered |
|---|---|---|
| Central validation result/error convention | Ensures malformed startup values fail with a precise correction before traffic is accepted. | Leave coercion to Fastify; rejected because it does not cover all configuration settings consistently. |
| Optional capability readiness state | Allows text chat to remain available while preventing misleading image-generation controls/requests. | Fail the full service for any missing optional runtime; rejected by clarification Q1. |
| README configuration matrix | Gives operators a reliable source for defaults and failure behavior. | Keep prose-only notes; rejected because the current documented port differs from the implementation. |
