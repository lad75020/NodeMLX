# NodeMLX Agent Boundaries

## Scope

NodeMLX is a Node.js/Fastify local-AI backend with an Angular web client and an optional SwiftUI macOS client. Preserve the existing local-first Apple Silicon inference architecture.

## Ownership Boundaries

| Area | Paths | Responsibilities | Coordinate before changing |
|---|---|---|---|
| Backend and protocol | `server.js`, `src/bootstrap.js`, `src/config.js`, `src/auth/`, `src/chat/`, `src/websocket/` | Fastify routes, authentication, WebSocket behavior, configuration, SQLite/Mongo interactions | Public HTTP/WebSocket contracts and persisted data |
| Inference runtime | `model-worker.js`, `src/inference/`, `scripts/` | Model processes, queueing/cancellation, Ollama/MLX/llama/VLM/image runtime adapters | Client-visible stream and image payloads; runtime configuration |
| Angular client | `client/src/` | Browser UI, authentication calls, chat stream display, model controls | Backend/API contract changes |
| Native macOS client | `NativeMac/` | SwiftUI user interface, auth client, WebSocket client, local model/chat state | Backend/API contract changes and behavior parity decisions |
| Documentation and tests | `docs/`, `api-docs.js`, `test/` | API reference and Node behavior verification | Any public contract or behavior change |

## Coordination Rules

1. Identify all affected clients before changing backend HTTP/WebSocket schemas or authentication behavior.
2. Keep SQLite auth/session and Mongo chat changes compatible with existing persisted data; document migration/backfill requirements in the feature plan.
3. Do not combine DiffusionKit and MLX z-image Python environments; their dependencies are intentionally isolated.
4. Do not put secrets, JWT material, local model paths, or machine-specific runtime configuration in source.
5. Keep changes inside the smallest relevant module; a cross-module change needs an explicit interface/compatibility note in the plan.

## Required Verification

- Run `npm test` for Node/backend/runtime changes.
- Run `npm run build` for Angular client changes.
- Run `swift build` from `NativeMac/` for native-client changes.
- For protocol or UX work, perform the feature-specific smoke path and update `docs/API.md`/`api-docs.js` when applicable.

## Working Tree Safety

The repository currently has user-owned uncommitted `.gitignore` changes. Do not overwrite or include unrelated working-tree changes in feature work.
