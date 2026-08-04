# NodeMLX Chat

NodeMLX Chat is a macOS Apple Silicon chat application with a Fastify `node-mlx` backend, invite-only user onboarding, and an Angular 18 + Bootstrap frontend.

## Highlights

- Fastify backend serving REST APIs, `/ws` WebSocket chat streaming, and the built Angular client.
- `node-mlx` model worker with Hugging Face model discovery, recommended model defaults, saved/failed model registry, and image input support.
- SQLite database `mlx-chat.db` for users, sessions, saved models, failed models, and app secrets.
- Invite-only local user creation through `scripts/onboard-user.js`.
- Angular 18 standalone frontend in `client/`, with dev-server proxying to the backend.
- OpenAPI JSON, browser docs, and a human-readable API guide in `docs/API.md`.
- Optional native macOS SwiftUI shell under `NativeMac/`.

## Repository layout

```text
.
├── server.js                 # Fastify app, auth, WebSocket, model registry, APIs
├── model-worker.js           # MLX generation worker
├── package.json              # Backend and build scripts
├── mlx-chat.db               # Local SQLite runtime database
├── scripts/onboard-user.js   # Invite-only user creation
├── client/                   # Angular frontend
│   ├── package.json
│   └── proxy.conf.json
├── docs/API.md               # HTTP/WebSocket protocol guide
├── NativeMac/                # Optional SwiftUI native client shell
├── utils/                    # Local utilities such as GPU usage helper
└── README.md
```

## Prerequisites

- macOS 14 or newer on Apple Silicon; `node-mlx` exits on unsupported platforms.
- Node.js and npm.
- Internet access for first-time Hugging Face model downloads unless models are already cached.
- Enough RAM and disk space for the selected MLX model.
- Optional Angular CLI through the client dependencies for frontend development.

## Installation and setup

Install backend dependencies and build the frontend:

```bash
npm run setup
```

Equivalent manual steps:

```bash
npm install
npm run build
```

Create an invited user:

```bash
npm run user:add -- <username>
```

Provide a password non-interactively when needed:

```bash
npm run user:add -- <username> --password '<strong-password>'
```

## Running

Production-style backend serving the built Angular app:

```bash
npm start
# open http://127.0.0.1:3000
```

Development mode with backend watch and Angular live reload:

```bash
npm run dev
```

In a second terminal:

```bash
npm run client:dev
# open http://127.0.0.1:4200
```

## Testing and checks

No automated test script is declared in the root or client `package.json`. Use `npm run build` as the main compile check, then smoke-test login, model selection, and WebSocket chat from the browser.

## Configuration and security notes

- `PORT` defaults to `3000`; `HOST` defaults to `127.0.0.1`.
- `MLX_MODEL` can be set to any supported Hugging Face model ID or a key from `node-mlx` `RECOMMENDED_MODELS`; default is `qwen-3-1.7b`.
- `HF_HOME` defaults to `~/.cache/huggingface` if unset.
- `MLX_MAX_TOKENS` and `MLX_MAX_TOKENS_LIMIT` control generation limits.
- `JWT_SECRET` can be supplied; otherwise a secret is generated and stored in SQLite.
- Protect `mlx-chat.db`, logs, uploaded image temp files, and session cookies. Do not expose the server beyond trusted networks without TLS and access controls.
