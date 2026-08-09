/**
 * API documentation builders kept separate from server bootstrap and routes.
 * The generated document remains request-independent and is safe to reuse.
 */
export function createOpenApiDocument({
  sessionCookieName,
  minUsernameLength,
  maxUsernameLength,
  minPasswordLength,
}) {
  return {
  openapi: "3.1.0",
  info: {
    title: "NodeMLX Chat Backend API",
    version: "1.0.0",
    description: [
      "HTTP endpoints use rolling seven-day cookie-based sessions.",
      "Interactive model operations use the existing `/ws` WebSocket JSON protocol.",
      "The WebSocket protocol is documented here with message schemas; existing frontend messages are unchanged.",
    ].join("\n"),
  },
  servers: [{ url: "/" }],
  tags: [
    { name: "Auth", description: "Cookie session authentication endpoints." },
    { name: "WebSocket", description: "Bidirectional JSON protocol at `/ws`." },
    { name: "Docs", description: "OpenAPI document and browser documentation." },
  ],
  paths: {
    "/api/openapi.json": {
      get: {
        tags: ["Docs"],
        summary: "OpenAPI document",
        operationId: "getOpenApiDocument",
        responses: {
          "200": {
            description: "OpenAPI 3.1 document for this backend.",
            content: {
              "application/json": {
                schema: { type: "object" },
              },
            },
          },
        },
      },
    },
    "/api/docs": {
      get: {
        tags: ["Docs"],
        summary: "Interactive API documentation",
        operationId: "getApiDocs",
        responses: {
          "200": {
            description: "HTML documentation page loading `/api/openapi.json`.",
            content: { "text/html": { schema: { type: "string" } } },
          },
        },
      },
    },
    "/api/auth/me": {
      get: {
        tags: ["Auth"],
        summary: "Get current authenticated user",
        operationId: "getCurrentUser",
        responses: {
          "200": {
            description: "Authentication status. A valid cookie-backed session is renewed for seven days and the response refreshes the HTTP-only cookie.",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/AuthMeResponse" },
              },
            },
          },
        },
      },
    },
    "/api/auth/register": {
      post: {
        tags: ["Auth"],
        summary: "Registration placeholder",
        operationId: "registerUser",
        description: "Registration is intentionally disabled; users are created with `npm run user:add` and local administrators reset passwords with `npm run user:reset`.",
        responses: {
          "403": {
            description: "Registration disabled.",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/ErrorResponse" },
              },
            },
          },
        },
      },
    },
    "/api/auth/login": {
      post: {
        tags: ["Auth"],
        summary: "Create a cookie session",
        operationId: "login",
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/LoginRequest" },
            },
          },
        },
        responses: {
          "200": {
            description: "Login succeeded. A rolling seven-day `nodemlx_session` HTTP-only cookie and WebSocket JWT are issued.",
            headers: {
              "Set-Cookie": {
                schema: { type: "string" },
                description: "HTTP-only session cookie.",
              },
            },
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/LoginResponse" },
              },
            },
          },
          "400": {
            description: "Missing username or password.",
            content: { "application/json": { schema: { $ref: "#/components/schemas/ErrorResponse" } } },
          },
          "401": {
            description: "Generic invalid-credentials response; it does not disclose whether the username exists.",
            content: { "application/json": { schema: { $ref: "#/components/schemas/ErrorResponse" } } },
          },
        },
      },
    },
    "/api/auth/logout": {
      post: {
        tags: ["Auth"],
        summary: "Clear the current session",
        operationId: "logout",
        responses: {
          "200": {
            description: "Logout completed.",
            content: {
              "application/json": {
                schema: { $ref: "#/components/schemas/OkResponse" },
              },
            },
          },
        },
      },
    },
    "/ws": {
      get: {
        tags: ["WebSocket"],
        summary: "WebSocket JSON protocol",
        operationId: "connectWebSocket",
        description: [
          "Upgrade this endpoint to a WebSocket after authenticating with `/api/auth/login`.",
          "Pass the returned JWT as `/ws?token=<jwt>`; each operation rechecks the rolling SQLite session.",
          "Messages are JSON objects with a `type` discriminator.",
          "RPC-style messages include a client-generated `requestId` and receive an `rpcResult` event.",
          "Generation messages stream server events such as `start`, `queued`, `response`, `ollamaChunk`, `ollamaDone`, `llamaChunk`, `llamaDone`, and `error`.",
        ].join("\n"),
        security: [{ websocketJwt: [] }],
        responses: {
          "101": { description: "WebSocket upgrade accepted." },
          "401": { description: "Authentication required." },
        },
        "x-websocket-protocol": {
          clientMessages: [
            "WsRpcRequest",
            "SelectModelMessage",
            "PromptMessage",
            "OllamaPromptMessage",
            "LlamaPromptMessage",
            "CancelInferenceMessage",
          ],
          serverEvents: [
            "RpcResultEvent",
            "StartEvent",
            "QueuedEvent",
            "ChatCreatedEvent",
            "ResponseEvent",
            "OllamaChunkEvent",
            "OllamaDoneEvent",
            "LlamaChunkEvent",
            "LlamaDoneEvent",
            "GpuUsageEvent",
            "ModelLoadingEvent",
            "ModelReadyEvent",
            "ModelErrorEvent",
            "ErrorEvent",
          ],
        },
      },
    },
  },
  components: {
    securitySchemes: {
      cookieSession: {
        type: "apiKey",
        in: "cookie",
        name: sessionCookieName,
      },
      websocketJwt: {
        type: "apiKey",
        in: "query",
        name: "token",
        description: "JWT returned by `/api/auth/login` or `/api/auth/me`; the JWT subject is validated against the rolling SQLite session table.",
      },
    },
    schemas: {
      ErrorResponse: {
        type: "object",
        required: ["error"],
        properties: { error: { type: "string" } },
      },
      OkResponse: {
        type: "object",
        required: ["ok"],
        properties: { ok: { type: "boolean", const: true } },
      },
      User: {
        type: "object",
        required: ["id", "username"],
        properties: {
          id: { type: "integer" },
          username: { type: "string" },
        },
      },
      AuthMeResponse: {
        oneOf: [
          {
            type: "object",
            required: ["authenticated", "user", "token"],
            properties: {
              authenticated: { type: "boolean", const: true },
              user: { $ref: "#/components/schemas/User" },
              token: { type: "string", description: "JWT for `/ws?token=<jwt>`." },
            },
          },
          {
            type: "object",
            required: ["authenticated"],
            properties: { authenticated: { type: "boolean", const: false } },
          },
        ],
      },
      LoginRequest: {
        type: "object",
        required: ["username", "password"],
        properties: {
          username: { type: "string", minLength: minUsernameLength, maxLength: maxUsernameLength },
          password: { type: "string", minLength: minPasswordLength },
        },
      },
      LoginResponse: {
        type: "object",
        required: ["user", "token"],
        properties: {
          user: { $ref: "#/components/schemas/User" },
          token: { type: "string", description: "JWT for `/ws?token=<jwt>`." },
        },
      },
      ChatSummary: {
        type: "object",
        required: ["id", "startedAt", "title"],
        properties: {
          id: { type: "string" },
          startedAt: { type: "string", format: "date-time" },
          title: { type: ["string", "null"] },
        },
      },
      ChatImageAttachment: {
        type: "object",
        required: ["dataUrl", "name", "type", "size"],
        properties: {
          dataUrl: { type: "string", description: "Data URL or displayable image URL." },
          name: { type: "string" },
          type: { type: "string" },
          size: { type: "integer", minimum: 0 },
        },
      },
      ChatMessage: {
        type: "object",
        required: ["id", "role", "text"],
        properties: {
          id: { type: "string" },
          role: { type: "string", enum: ["user", "assistant"] },
          text: { type: "string" },
          thinking: { type: "string" },
          provider: { type: "string", enum: ["mlx", "ollama", "llamacpp"] },
          image: { $ref: "#/components/schemas/ChatImageAttachment" },
          images: { type: "array", items: { $ref: "#/components/schemas/ChatImageAttachment" } },
          modelId: { type: "string" },
          tokenCount: { type: "number" },
          tokensPerSecond: { type: "number" },
        },
      },
      WsRpcRequest: {
        type: "object",
        required: ["type", "requestId"],
        properties: {
          type: {
            type: "string",
            enum: [
              "listModels",
              "listOllamaModels",
              "showOllamaModel",
              "pickLlamaModelFile",
              "listFailedModels",
              "listChats",
              "openChat",
              "deleteChat",
              "deleteMessage",
            ],
          },
          requestId: { type: "string", description: "Client-generated id echoed by `rpcResult`." },
          refresh: { type: "boolean", description: "Only used by `listModels`." },
          modelId: { type: "string", description: "Only used by `showOllamaModel`." },
          chatId: { type: "string", description: "Used by chat RPCs." },
          messageId: { type: "string", description: "Only used by `deleteMessage`." },
        },
      },
      SelectModelMessage: {
        type: "object",
        required: ["type", "modelId"],
        properties: {
          type: { type: "string", const: "selectModel" },
          modelId: { type: "string" },
        },
      },
      PromptMessage: {
        type: "object",
        required: ["type", "id", "prompt"],
        properties: {
          type: { type: "string", const: "prompt" },
          id: { type: "string" },
          chatId: { type: ["string", "null"] },
          prompt: { type: "string" },
          image: { $ref: "#/components/schemas/ChatImageAttachment" },
          maxTokens: { type: "integer", minimum: 1 },
          imageWidth: { type: "integer" },
          imageHeight: { type: "integer" },
          steps: { type: "integer" },
          seed: { type: "integer" },
        },
      },
      OllamaPromptMessage: {
        type: "object",
        required: ["type", "id", "modelId", "prompt"],
        properties: {
          type: { type: "string", const: "ollamaPrompt" },
          id: { type: "string" },
          chatId: { type: ["string", "null"] },
          modelId: { type: "string" },
          prompt: { type: "string" },
          image: { $ref: "#/components/schemas/ChatImageAttachment" },
          seed: { type: "integer" },
          enableThinking: { type: "boolean" },
        },
      },
      LlamaPromptMessage: {
        type: "object",
        required: ["type", "id", "prompt", "modelSource"],
        properties: {
          type: { type: "string", const: "llamaPrompt" },
          id: { type: "string" },
          chatId: { type: ["string", "null"] },
          prompt: { type: "string" },
          modelSource: { type: "string", enum: ["disk", "huggingface"] },
          modelPath: { type: "string", description: "Required when `modelSource` is `disk`." },
          hfModel: { type: "string", description: "Required when `modelSource` is `huggingface`." },
          maxTokens: { type: "integer", minimum: 1 },
        },
      },
      CancelInferenceMessage: {
        type: "object",
        required: ["type"],
        properties: { type: { type: "string", const: "cancelInference" } },
      },
      RpcResultEvent: {
        type: "object",
        required: ["type", "requestId"],
        properties: {
          type: { type: "string", const: "rpcResult" },
          requestId: { type: "string" },
          data: { type: "object" },
          error: { type: "string" },
        },
      },
      StartEvent: {
        type: "object",
        required: ["type", "id"],
        properties: {
          type: { type: "string", const: "start" },
          id: { type: "string" },
          chatId: { type: ["string", "null"] },
        },
      },
      QueuedEvent: {
        type: "object",
        required: ["type", "id", "position"],
        properties: {
          type: { type: "string", const: "queued" },
          id: { type: "string" },
          position: { type: "integer", minimum: 1 },
        },
      },
      ChatCreatedEvent: {
        type: "object",
        required: ["type", "id", "chat"],
        properties: {
          type: { type: "string", const: "chatCreated" },
          id: { type: "string" },
          chat: { $ref: "#/components/schemas/ChatSummary" },
        },
      },
      ResponseEvent: {
        type: "object",
        required: ["type", "id", "text", "tokenCount", "tokensPerSecond"],
        properties: {
          type: { type: "string", const: "response" },
          id: { type: "string" },
          chatId: { type: ["string", "null"] },
          modelId: { type: "string" },
          text: { type: "string" },
          images: { type: "array", items: { $ref: "#/components/schemas/ChatImageAttachment" } },
          tokenCount: { type: "number" },
          tokensPerSecond: { type: "number" },
        },
      },
      OllamaChunkEvent: {
        type: "object",
        required: ["type", "id"],
        properties: {
          type: { type: "string", const: "ollamaChunk" },
          id: { type: "string" },
          text: { type: "string" },
          thinking: { type: "string" },
          images: { type: "array", items: { $ref: "#/components/schemas/ChatImageAttachment" } },
        },
      },
      OllamaDoneEvent: {
        type: "object",
        required: ["type", "id", "modelId", "text"],
        properties: {
          type: { type: "string", const: "ollamaDone" },
          id: { type: "string" },
          chatId: { type: ["string", "null"] },
          modelId: { type: "string" },
          text: { type: "string" },
          thinking: { type: "string" },
          images: { type: "array", items: { $ref: "#/components/schemas/ChatImageAttachment" } },
          totalDuration: { type: ["number", "null"] },
          evalCount: { type: ["number", "null"] },
        },
      },
      LlamaChunkEvent: {
        type: "object",
        required: ["type", "id"],
        properties: {
          type: { type: "string", const: "llamaChunk" },
          id: { type: "string" },
          text: { type: "string" },
          thinking: { type: "string" },
        },
      },
      LlamaDoneEvent: {
        type: "object",
        required: ["type", "id", "modelName", "text"],
        properties: {
          type: { type: "string", const: "llamaDone" },
          id: { type: "string" },
          chatId: { type: ["string", "null"] },
          modelName: { type: "string" },
          text: { type: "string" },
          thinking: { type: "string" },
        },
      },
      GpuUsageEvent: {
        type: "object",
        required: ["type", "running"],
        properties: {
          type: { type: "string", const: "gpuUsage" },
          running: { type: "boolean" },
          gpu: { type: ["number", "null"] },
          memory: { type: ["number", "null"] },
        },
      },
      ModelLoadingEvent: {
        type: "object",
        required: ["type"],
        properties: {
          type: { type: "string", const: "modelLoading" },
          modelId: { type: ["string", "null"] },
        },
      },
      ModelReadyEvent: {
        type: "object",
        required: ["type", "modelId"],
        properties: {
          type: { type: "string", const: "modelReady" },
          modelId: { type: "string" },
          isVLM: { type: "boolean" },
          canGenerateImages: { type: "boolean" },
        },
      },
      ModelErrorEvent: {
        type: "object",
        required: ["type", "error"],
        properties: {
          type: { type: "string", const: "modelError" },
          modelId: { type: "string" },
          error: { type: "string" },
          failed: { type: "array", items: { type: "object" } },
        },
      },
      ErrorEvent: {
        type: "object",
        required: ["type", "error"],
        properties: {
          type: { type: "string", const: "error" },
          id: { type: "string" },
          error: { type: "string" },
        },
      },
    },
  },
};
}

export function createApiDocsHtml(sessionCookieName) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NodeMLX Chat API</title>
    <style>
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      header { padding: 1rem 1.25rem; border-bottom: 1px solid #ddd; }
      main { max-width: 980px; margin: 0 auto; padding: 1.25rem; }
      code, pre { background: #f3f4f6; border-radius: 0.35rem; }
      code { padding: 0.08rem 0.25rem; }
      pre { padding: 0.8rem; overflow-x: auto; }
    </style>
  </head>
  <body>
    <header>
      <h1>NodeMLX Chat API</h1>
      <p>OpenAPI JSON: <a href="/api/openapi.json">/api/openapi.json</a></p>
    </header>
    <main>
      <h2>Connection Flow</h2>
      <ol>
        <li><code>POST /api/auth/login</code> with JSON credentials.</li>
        <li>Keep the returned <code>${sessionCookieName}</code> cookie for HTTP auth.</li>
        <li>Open <code>ws(s)://host/ws?token=&lt;jwt&gt;</code> with the returned JWT.</li>
        <li>Send JSON messages documented in <code>/api/openapi.json</code> under <code>x-websocket-protocol</code> and <code>components.schemas</code>.</li>
      </ol>
      <h2>WebSocket RPC Example</h2>
      <pre>{"type":"listChats","requestId":"client-generated-id"}</pre>
      <h2>WebSocket Generation Example</h2>
      <pre>{"type":"prompt","id":"message-id","chatId":null,"prompt":"Hello","maxTokens":4096}</pre>
      <p>See <code>docs/API.md</code> in the repository for a human-readable protocol guide.</p>
    </main>
  </body>
</html>`;
}
