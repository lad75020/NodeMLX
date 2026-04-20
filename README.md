# NodeMLX Chat

A single-page chat app: Fastify + `node-mlx` backend, Angular 18 + Bootstrap frontend,
communicating over WebSockets. Runs on macOS 14+ / Apple Silicon.

## Layout

- `server.js` — Fastify server with `/ws` WebSocket and static Angular hosting.
- `client/` — Angular SPA (standalone components, Bootstrap).

## Setup

```bash
# one-time: install backend + build frontend
npm install
npm run build
```

## Run (production mode: backend serves built Angular)

```bash
npm start
# open http://127.0.0.1:3000
```

## Run (dev mode: Angular dev server with live reload)

Terminal 1:

```bash
npm run dev         # Fastify on :3000
```

Terminal 2:

```bash
npm run client:dev  # ng serve on :4200 (proxies /ws + /api to :3000)
# open http://127.0.0.1:4200
```

## Configuration

- `PORT` (default `3000`), `HOST` (default `127.0.0.1`).
- `MLX_MODEL` — any HuggingFace ID or key from `RECOMMENDED_MODELS`
  (default: `qwen-3-1.7b`). First run downloads the weights.

## Invite-Only User Onboarding

Create users from the command line (writes into `mlx-chat.db`):

```bash
npm run user:add -- <username>
```

Or provide password non-interactively:

```bash
npm run user:add -- <username> --password '<strong-password>'
```

Optional database path:

```bash
npm run user:add -- <username> --db /path/to/mlx-chat.db
```

## Wire protocol

Client → server: `{ "type": "prompt", "id": "...", "prompt": "..." }`
Server → client: `{ "type": "start" | "response" | "error", ... }`

## Backend API Documentation

- OpenAPI JSON: `GET /api/openapi.json`
- Browser docs: `GET /api/docs`
- Human-readable protocol guide: [`docs/API.md`](docs/API.md)

The current Angular frontend still uses the existing `/ws` WebSocket protocol.
Other frontends should authenticate with the HTTP auth endpoints, preserve the
`nodemlx_session` cookie for HTTP requests, then connect to `/ws?token=<jwt>`
using the JWT returned by login or session restore.
