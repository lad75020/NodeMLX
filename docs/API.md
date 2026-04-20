# NodeMLX Chat Backend API

The backend exposes a small HTTP API for authentication and a JSON WebSocket protocol for model operations. The existing Angular frontend uses the same protocol documented here; no WebSocket message shape is changed by the OpenAPI documentation work.

## OpenAPI

- JSON document: `GET /api/openapi.json`
- Browser docs: `GET /api/docs`

The OpenAPI document is OpenAPI `3.1.0`. WebSocket details are represented as a documented `GET /ws` upgrade operation plus schema components and an `x-websocket-protocol` extension because OpenAPI does not natively model bidirectional WebSocket frames as first-class operations.

## Authentication

The backend uses an HTTP-only cookie named `nodemlx_session` for HTTP auth and a JSON Web Token (JWT) for WebSocket auth. JWTs are signed with an HMAC secret stored in SQLite (`app_secrets`) unless `JWT_SECRET` is provided, and every WebSocket JWT is validated against the SQLite `sessions` table.

### `GET /api/auth/me`

Returns whether the current request is authenticated.

Unauthenticated response:

```json
{ "authenticated": false }
```

Authenticated response:

```json
{
  "authenticated": true,
  "user": { "id": 1, "username": "alice" }
}
```

### `POST /api/auth/login`

Request:

```json
{
  "username": "alice",
  "password": "correct horse battery staple"
}
```

On success, the server sets `nodemlx_session` and returns a JWT for WebSocket connections:

```json
{
  "user": { "id": 1, "username": "alice" },
  "token": "eyJhbGciOiJIUzI1NiIs..."
}
```

### `POST /api/auth/logout`

Clears the current session cookie.

Response:

```json
{ "ok": true }
```

### `POST /api/auth/register`

Registration is disabled. Users are created with:

```bash
npm run user:add -- <username>
```

## WebSocket Connection

Connect to `/ws?token=<jwt>` after login. The JWT is returned by `/api/auth/login` and `/api/auth/me`. The server verifies the JWT signature and checks that its `sid` still exists in the SQLite `sessions` table before accepting the WebSocket upgrade.

The server immediately sends the current MLX model state after connection, usually one of:

```json
{ "type": "modelLoading", "modelId": "mlx-community/..." }
```

```json
{ "type": "modelReady", "modelId": "mlx-community/...", "isVLM": false, "canGenerateImages": false }
```

```json
{ "type": "modelError", "error": "No model loaded." }
```

Example:

```text
wss://example.test/ws?token=eyJhbGciOiJIUzI1NiIs...
```

## WebSocket RPC

RPC messages are JSON objects with a `type` and client-generated `requestId`. The response is always an `rpcResult` with the same `requestId`.

Example request:

```json
{ "type": "listChats", "requestId": "req-1" }
```

Success response:

```json
{ "type": "rpcResult", "requestId": "req-1", "data": { "chats": [] } }
```

Error response:

```json
{ "type": "rpcResult", "requestId": "req-1", "error": "Chat not found." }
```

Supported RPC types:

- `listModels`: optional `refresh: boolean`; returns MLX model list.
- `listOllamaModels`: returns locally available Ollama models and configured Ollama URL.
- `showOllamaModel`: requires `modelId`; returns Ollama capabilities/details.
- `pickLlamaModelFile`: opens a native backend-side file picker and validates a llama.cpp disk model file.
- `listFailedModels`: returns failed MLX model records.
- `listChats`: returns chat summaries for the authenticated user.
- `openChat`: requires `chatId`; returns messages for a chat.
- `deleteChat`: requires `chatId`.
- `deleteMessage`: requires `chatId` and `messageId`.

## Generation Queue

Only one generation runs at a time. When a request is queued, the server sends:

```json
{ "type": "queued", "id": "message-id", "position": 1 }
```

When it starts, the server sends:

```json
{ "type": "start", "id": "message-id", "chatId": "..." }
```

If a new chat is created, the server also sends:

```json
{
  "type": "chatCreated",
  "id": "message-id",
  "chat": { "id": "...", "startedAt": "2026-04-20T12:00:00.000Z", "title": null }
}
```

## MLX Generation

Client request:

```json
{
  "type": "prompt",
  "id": "message-id",
  "chatId": null,
  "prompt": "Write a haiku",
  "maxTokens": 4096
}
```

Optional image input uses `image`:

```json
{
  "dataUrl": "data:image/png;base64,...",
  "name": "image.png",
  "type": "image/png",
  "size": 12345
}
```

Final response:

```json
{
  "type": "response",
  "id": "message-id",
  "chatId": "...",
  "modelId": "mlx-community/...",
  "text": "...",
  "images": [],
  "tokenCount": 120,
  "tokensPerSecond": 18.4
}
```

## Ollama Generation

Client request:

```json
{
  "type": "ollamaPrompt",
  "id": "message-id",
  "chatId": null,
  "modelId": "gpt-oss:20b",
  "prompt": "Explain recursion",
  "enableThinking": true
}
```

Streaming chunks:

```json
{ "type": "ollamaChunk", "id": "message-id", "thinking": "..." }
```

```json
{ "type": "ollamaChunk", "id": "message-id", "text": "..." }
```

Done event:

```json
{
  "type": "ollamaDone",
  "id": "message-id",
  "chatId": "...",
  "modelId": "gpt-oss:20b",
  "text": "final answer",
  "thinking": "reasoning text",
  "images": [],
  "totalDuration": 123456789,
  "evalCount": 120
}
```

After completion, the server unloads the Ollama model by calling Ollama `/api/generate` with `keep_alive: 0`.

## Llama.cpp Generation

Disk model request:

```json
{
  "type": "llamaPrompt",
  "id": "message-id",
  "chatId": null,
  "modelSource": "disk",
  "modelPath": "/absolute/path/model.gguf",
  "prompt": "Hello",
  "maxTokens": 4096
}
```

Hugging Face model request:

```json
{
  "type": "llamaPrompt",
  "id": "message-id",
  "chatId": null,
  "modelSource": "huggingface",
  "hfModel": "owner/model",
  "prompt": "Hello",
  "maxTokens": 4096
}
```

The server uses `llama-cli -m <path>` for disk models and `llama-cli -hf <model>` for Hugging Face models. Disk model files larger than 16 GB are rejected.

Streaming chunks:

```json
{ "type": "llamaChunk", "id": "message-id", "thinking": "..." }
```

```json
{ "type": "llamaChunk", "id": "message-id", "text": "..." }
```

Done event:

```json
{
  "type": "llamaDone",
  "id": "message-id",
  "chatId": "...",
  "modelName": "model.gguf or owner/model",
  "text": "final answer",
  "thinking": "reasoning text"
}
```

For llama.cpp only, output before `[Start thinking]` is ignored, output through `[End thinking]` is streamed as `thinking`, and subsequent output is streamed as `text`.

## Cancel Inference

Client request:

```json
{ "type": "cancelInference" }
```

The server cancels queued work, aborts active Ollama and Llama.cpp requests, and terminates the active MLX worker process when applicable.

## GPU Usage Events

While inference is running, the server polls `utils/GPUUsage` every second and broadcasts:

```json
{ "type": "gpuUsage", "running": true, "gpu": 6, "memory": 61.8 }
```

When inference stops:

```json
{ "type": "gpuUsage", "running": false, "gpu": null, "memory": null }
```

If reported memory reaches 98% or higher, the server cancels the active backend inference process.

## Errors

Non-RPC errors are sent as:

```json
{ "type": "error", "id": "message-id", "error": "message" }
```

`id` is omitted for connection-level errors.
