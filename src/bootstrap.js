// Application bootstrap. Runtime behavior is composed from the modules below.
import { fileURLToPath } from "node:url";
import { basename, dirname, join } from "node:path";
import { execFile, spawn } from "node:child_process";
import { mkdir, rm, stat, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { promisify } from "node:util";

import Fastify from "fastify";
import fastifyWebsocket from "@fastify/websocket";
import fastifyStatic from "@fastify/static";
import { listModels } from "@huggingface/hub";
import { isSupported, RECOMMENDED_MODELS } from "node-mlx";
import { ObjectId } from "mongodb";
import { createChatStore } from "./chat/mongo-store.js";
import { createOpenApiDocument, createApiDocsHtml } from "../api-docs.js";
import {
  createConfig,
  clampInteger,
  optionalClampedInteger,
  imageExtensions,
} from "./config.js";
import { createAuthStore } from "./auth/sqlite-store.js";
import { createGenerationManager } from "./inference/generation-manager.js";
import { createLlamaInference } from "./inference/llama.js";
import { ModelProcess } from "./inference/model-process.js";
import { createOllamaInference } from "./inference/ollama.js";
import { parseGpuUsageOutput } from "./websocket/gpu-usage.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = dirname(__dirname);
const WORKER_PATH = join(PROJECT_ROOT, "model-worker.js");
const GPU_USAGE_PATH = join(PROJECT_ROOT, "utils", "GPUUsage");

const config = createConfig(RECOMMENDED_MODELS);
for (const warning of config.warnings) {
  console.warn(`Configuration warning: ${warning}`);
}
for (const readiness of Object.values(config.optionalCapabilities)) {
  if (readiness.state === "unavailable") {
    console.warn(`Optional capability unavailable: ${readiness.diagnostic}`);
  }
}
const {
  defaultModel: DEFAULT_MODEL,
  port: PORT,
  host: HOST,
  modelsOwner: MODELS_OWNER,
  modelsTtlMs: MODELS_TTL_MS,
  ollamaUrl: OLLAMA_URL,
  imageTmpDir: IMAGE_TMP_DIR,
  maxImageBytes: MAX_IMAGE_BYTES,
  maxLlamaModelBytes: MAX_LLAMA_MODEL_BYTES,
  maxLlamaOutputChars: MAX_LLAMA_OUTPUT_CHARS,
  sessionCookieName: SESSION_COOKIE_NAME,
  sessionTtlMs: SESSION_TTL_MS,
  minUsernameLength: MIN_USERNAME_LENGTH,
  maxUsernameLength: MAX_USERNAME_LENGTH,
  minPasswordLength: MIN_PASSWORD_LENGTH,
  maxGenerationTokens: MAX_GENERATION_TOKENS,
  defaultMaxTokens: DEFAULT_MAX_TOKENS,
} = config;
const IMAGE_EXTENSIONS = imageExtensions;

if (!isSupported()) {
  console.error("node-mlx requires macOS 14+ on Apple Silicon.");
  process.exit(1);
}

const execFileAsync = promisify(execFile);
// ─── SQLite — model registry and authentication ───────────────────────
const {
  stmt,
  normalizeUsername,
  verifyPassword,
  createSessionForUser,
  readSessionUser,
  createSessionJwt,
  verifySessionJwt,
  websocketTokenFromRequest,
  sessionCookieValue,
  setSessionCookie,
  clearSessionCookie,
} = createAuthStore({
  databasePath: join(PROJECT_ROOT, "mlx-chat.db"),
  sessionCookieName: SESSION_COOKIE_NAME,
  sessionTtlMs: SESSION_TTL_MS,
});

// ─── MongoDB — chat persistence ──────────────────────────────────────
const MONGO_URL = process.env.MONGO_URL ?? "mongodb://192.168.1.80:27017";
const MONGO_DB = process.env.MONGO_DB ?? "NodeMLX";
const {
  client: mongoClient,
  collection: chatsCol,
  createChat,
  appendChatMessages,
  ensureUserChat,
} = await createChatStore({
  url: MONGO_URL,
  databaseName: MONGO_DB,
});

function toChatSummary(doc) {
  return {
    id: doc._id.toString(),
    startedAt:
      doc.startedAt instanceof Date
        ? doc.startedAt.toISOString()
        : doc.startedAt,
    title: doc.title ?? null,
    messageCount: Array.isArray(doc.messages) ? doc.messages.length : 0,
  };
}

// ─── WebSocket broadcast helpers ─────────────────────────────────────
const sockets = new Set();
function broadcast(msg) {
  const payload = JSON.stringify(msg);
  for (const s of sockets) {
    if (s.readyState === 1) s.send(payload);
  }
}

async function persistPromptImage(image) {
  if (!image) return null;
  if (typeof image !== "object" || typeof image.dataUrl !== "string") {
    throw new Error("Invalid image payload.");
  }

  const match =
    /^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/.exec(
      image.dataUrl,
    );
  if (!match) {
    throw new Error("Image must be a JPEG, PNG, WebP, or GIF data URL.");
  }

  const mimeType = match[1];
  const extension = IMAGE_EXTENSIONS[mimeType];
  const buffer = Buffer.from(match[2], "base64");
  if (buffer.byteLength === 0) {
    throw new Error("Image is empty.");
  }
  if (buffer.byteLength > MAX_IMAGE_BYTES) {
    throw new Error("Image is too large. Maximum size is 10 MB.");
  }

  await mkdir(IMAGE_TMP_DIR, { recursive: true });
  const path = join(IMAGE_TMP_DIR, `${randomUUID()}.${extension}`);
  await writeFile(path, buffer);
  return {
    path,
    name: typeof image.name === "string" ? image.name.slice(0, 200) : "image",
    mimeType,
    size: buffer.byteLength,
  };
}

function cleanupPromptImage(imagePath) {
  if (!imagePath) return;
  void rm(imagePath, { force: true }).catch(() => {});
}

const modelProcess = new ModelProcess({
  workerPath: WORKER_PATH,
  broadcast,
  stmt,
  cleanupPromptImage,
  appendChatMessages,
});

// Load the default model (queued; fires once the worker is ready).
console.log(`Queuing initial model load: ${DEFAULT_MODEL}`);
modelProcess.load(DEFAULT_MODEL);

const generationManager = createGenerationManager({
  broadcast,
  parseGpuUsageOutput,
  gpuUsagePath: GPU_USAGE_PATH,
  execFileAsync,
  memoryCancelThreshold: config.memoryCancelThreshold,
  modelProcess,
});

const { listOllamaModels, showOllamaModel, streamOllamaPrompt } =
  createOllamaInference({
    ollamaUrl: OLLAMA_URL,
    maxImageBytes: MAX_IMAGE_BYTES,
    optionalClampedInteger,
    ensureUserChat,
    appendChatMessages,
    registerInferenceCancel: generationManager.registerInferenceCancel,
  });

const { pickLlamaModelFile, streamLlamaPrompt } = createLlamaInference({
  maxLlamaModelBytes: MAX_LLAMA_MODEL_BYTES,
  maxLlamaOutputChars: MAX_LLAMA_OUTPUT_CHARS,
  maxGenerationTokens: MAX_GENERATION_TOKENS,
  defaultMaxTokens: DEFAULT_MAX_TOKENS,
  clampInteger,
  ensureUserChat,
  appendChatMessages,
  registerInferenceCancel: generationManager.registerInferenceCancel,
  stat,
  basename,
  execFileAsync,
  spawn,
  environment: process.env,
});

// ─── HuggingFace model catalog cache ─────────────────────────────────
let modelsCache = null;
let modelsCacheAt = 0;

async function listMlxCommunityModels(force = false) {
  if (!force && modelsCache && Date.now() - modelsCacheAt < MODELS_TTL_MS) {
    return modelsCache;
  }
  const out = [];
  for await (const m of listModels({ search: { owner: MODELS_OWNER } })) {
    out.push({
      id: m.name ?? m.id,
      downloads: typeof m.downloads === "number" ? m.downloads : 0,
      likes: typeof m.likes === "number" ? m.likes : 0,
      updatedAt: m.updatedAt ? new Date(m.updatedAt).toISOString() : null,
    });
  }
  out.sort((a, b) => b.downloads - a.downloads);
  modelsCache = out;
  modelsCacheAt = Date.now();
  return out;
}

// ─── Fastify ─────────────────────────────────────────────────────────
const fastify = Fastify({ logger: true });
await fastify.register(fastifyWebsocket);
await fastify.register(fastifyStatic, {
  root: join(PROJECT_ROOT, "client", "dist", "chat-client", "browser"),
  prefix: "/",
  decorateReply: false,
});

const openApiDocument = createOpenApiDocument({
  sessionCookieName: SESSION_COOKIE_NAME,
  minUsernameLength: MIN_USERNAME_LENGTH,
  maxUsernameLength: MAX_USERNAME_LENGTH,
  minPasswordLength: MIN_PASSWORD_LENGTH,
});

fastify.get("/api/openapi.json", async (_request, reply) => {
  return reply.type("application/json").send(openApiDocument);
});

fastify.get("/api/docs", async (_request, reply) =>
  reply.type("text/html").send(createApiDocsHtml(SESSION_COOKIE_NAME)),
);

// ─── Auth endpoints (HTTP + cookie sessions) ──────────────────────────
fastify.get("/api/auth/me", async (request, reply) => {
  const session = readSessionUser(request);
  if (!session) {
    clearSessionCookie(reply, request);
    return { authenticated: false };
  }
  return {
    authenticated: true,
    user: {
      id: session.userId,
      username: session.username,
    },
    token: createSessionJwt(session),
  };
});

fastify.post("/api/auth/register", async (_request, reply) => {
  reply.code(403);
  return {
    error: "Registration is disabled. Ask an administrator for an invite.",
  };
});

fastify.post("/api/auth/login", async (request, reply) => {
  const body = request.body ?? {};
  const username = normalizeUsername(body.username);
  const password = body.password;
  if (!username || typeof password !== "string") {
    reply.code(400);
    return { error: "Username and password are required." };
  }

  const user = stmt.getUserForLogin.get(username);
  const valid = user
    ? await verifyPassword(password, user.passwordHash)
    : false;
  if (!valid) {
    reply.code(401);
    return { error: "Invalid username or password." };
  }

  stmt.updateUserLastLogin.run(new Date().toISOString(), user.id);
  const session = createSessionForUser(user.id);
  setSessionCookie(reply, request, session.sessionId);
  const sessionUser = {
    sessionId: session.sessionId,
    userId: user.id,
    username: user.username,
    expiresAt: session.expiresAt,
  };
  return {
    user: { id: user.id, username: user.username },
    token: createSessionJwt(sessionUser),
  };
});

fastify.post("/api/auth/logout", async (request, reply) => {
  const sessionId = sessionCookieValue(request);
  if (sessionId) stmt.deleteSession.run(sessionId);
  clearSessionCookie(reply, request);
  return { ok: true };
});

// ─── RPC handlers (invoked over WebSocket) ───────────────────────────
async function rpcListModels({ refresh }) {
  const hfModels = await listMlxCommunityModels(refresh === true);
  const savedRows = stmt.allSaved.all();
  if (savedRows.length === 0) return { models: hfModels };

  const hfIds = new Set(hfModels.map((m) => m.id));
  const savedIds = new Set(savedRows.map((r) => r.id));
  const merged = hfModels.map((m) => ({ ...m, saved: savedIds.has(m.id) }));
  for (const row of savedRows) {
    if (!hfIds.has(row.id)) {
      merged.push({
        id: row.id,
        downloads: 0,
        likes: 0,
        updatedAt: null,
        saved: true,
      });
    }
  }
  const savedOrder = new Map(savedRows.map((r, i) => [r.id, i]));
  merged.sort((a, b) => {
    const aSaved = a.saved ? (savedOrder.get(a.id) ?? Infinity) : Infinity;
    const bSaved = b.saved ? (savedOrder.get(b.id) ?? Infinity) : Infinity;
    if (aSaved !== bSaved) return aSaved - bSaved;
    return b.downloads - a.downloads;
  });
  return { models: merged };
}

function rpcListFailedModels() {
  return { failed: stmt.allFailed.all() };
}

async function rpcListOllamaModels() {
  return { models: await listOllamaModels(), url: OLLAMA_URL };
}

async function rpcShowOllamaModel({ modelId }) {
  return { model: await showOllamaModel(modelId) };
}

async function rpcPickLlamaModelFile() {
  return { model: await pickLlamaModelFile() };
}

async function rpcListChats(_payload, { userId }) {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  if (typeof userId !== "number") throw new Error("Unauthorized.");
  const docs = await chatsCol
    .find({ userId }, { projection: { messages: 0 } })
    .sort({ startedAt: -1 })
    .toArray();
  return {
    chats: docs.map((d) => ({
      id: d._id.toString(),
      startedAt:
        d.startedAt instanceof Date ? d.startedAt.toISOString() : d.startedAt,
      title: d.title ?? null,
    })),
  };
}

async function rpcOpenChat({ chatId }, { userId }) {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  if (typeof userId !== "number") throw new Error("Unauthorized.");
  let id;
  try {
    id = new ObjectId(chatId);
  } catch {
    throw new Error("Invalid chat id.");
  }
  const doc = await chatsCol.findOne({ _id: id, userId });
  if (!doc) throw new Error("Chat not found.");
  return {
    id: doc._id.toString(),
    startedAt:
      doc.startedAt instanceof Date
        ? doc.startedAt.toISOString()
        : doc.startedAt,
    title: doc.title ?? null,
    messages: (doc.messages ?? []).map((m) => ({
      id: m.id,
      role: m.role,
      text: m.text ?? "",
      thinking: m.thinking ?? undefined,
      provider: m.provider ?? undefined,
      image: m.image ?? undefined,
      images: m.images ?? undefined,
      modelId: m.modelId ?? undefined,
      tokenCount: m.tokenCount ?? undefined,
      tokensPerSecond: m.tokensPerSecond ?? undefined,
    })),
  };
}

async function rpcDeleteChat({ chatId }, { userId }) {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  if (typeof userId !== "number") throw new Error("Unauthorized.");
  let id;
  try {
    id = new ObjectId(chatId);
  } catch {
    throw new Error("Invalid chat id.");
  }
  await chatsCol.deleteOne({ _id: id, userId });
  return { ok: true };
}

async function rpcDeleteMessage({ chatId, messageId }, { userId }) {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  if (typeof userId !== "number") throw new Error("Unauthorized.");
  if (typeof messageId !== "string" || !messageId)
    throw new Error("Invalid message id.");
  let id;
  try {
    id = new ObjectId(chatId);
  } catch {
    throw new Error("Invalid chat id.");
  }
  await chatsCol.updateOne(
    { _id: id, userId },
    { $pull: { messages: { id: messageId } } },
  );
  return { ok: true };
}

const rpcHandlers = {
  listModels: rpcListModels,
  listOllamaModels: rpcListOllamaModels,
  showOllamaModel: rpcShowOllamaModel,
  pickLlamaModelFile: rpcPickLlamaModelFile,
  listFailedModels: rpcListFailedModels,
  listChats: rpcListChats,
  openChat: rpcOpenChat,
  deleteChat: rpcDeleteChat,
  deleteMessage: rpcDeleteMessage,
};

// ─── WebSocket endpoint ───────────────────────────────────────────────
fastify.register(async (instance) => {
  instance.get("/ws", { websocket: true }, (socket, request) => {
    const session = verifySessionJwt(websocketTokenFromRequest(request));
    if (!session || typeof session.userId !== "number") {
      socket.send(
        JSON.stringify({ type: "error", error: "Authentication required." }),
      );
      socket.close(4401, "Unauthorized");
      return;
    }

    sockets.add(socket);
    socket.send(JSON.stringify(modelProcess.greetingMessage()));

    socket.on("close", () => sockets.delete(socket));

    socket.on("message", async (raw) => {
      let payload;
      try {
        payload = JSON.parse(raw.toString());
      } catch {
        socket.send(JSON.stringify({ type: "error", error: "Invalid JSON." }));
        return;
      }

      // ── RPC dispatch ─────────────────────────────────────────────
      if (
        typeof payload?.type === "string" &&
        payload.type in rpcHandlers &&
        typeof payload.requestId === "string"
      ) {
        const requestId = payload.requestId;
        try {
          const data = await rpcHandlers[payload.type](payload, {
            userId: session.userId,
          });
          socket.send(JSON.stringify({ type: "rpcResult", requestId, data }));
        } catch (err) {
          socket.send(
            JSON.stringify({
              type: "rpcResult",
              requestId,
              error: err instanceof Error ? err.message : String(err),
            }),
          );
        }
        return;
      }

      // ── cancelInference ─────────────────────────────────────────
      if (payload?.type === "cancelInference") {
        generationManager.cancelAllInference("Inference cancelled.");
        return;
      }

      // ── selectModel ──────────────────────────────────────────────
      if (payload?.type === "selectModel") {
        try {
          modelProcess.load(payload.modelId);
        } catch (err) {
          socket.send(
            JSON.stringify({
              type: "modelError",
              error: err instanceof Error ? err.message : String(err),
            }),
          );
        }
        return;
      }

      // ── ollamaPrompt ──────────────────────────────────────────────
      if (payload?.type === "ollamaPrompt") {
        const id = payload.id ?? String(Date.now());
        generationManager.enqueue({
          id,
          socket,
          run: () =>
            streamOllamaPrompt(socket, { ...payload, id }, session.userId),
        });
        return;
      }

      // ── llamaPrompt ───────────────────────────────────────────────
      if (payload?.type === "llamaPrompt") {
        const id = payload.id ?? String(Date.now());
        generationManager.enqueue({
          id,
          socket,
          run: () =>
            streamLlamaPrompt(socket, { ...payload, id }, session.userId),
        });
        return;
      }

      // ── prompt ───────────────────────────────────────────────────
      if (payload?.type === "prompt" && typeof payload.prompt === "string") {
        const id = payload.id ?? String(Date.now());
        generationManager.enqueue({
          id,
          socket,
          run: async () => {
            let persistedImage = null;
            try {
              persistedImage = await persistPromptImage(payload.image);

              let chatId =
                typeof payload.chatId === "string" ? payload.chatId : null;
              if (chatsCol) {
                if (chatId) {
                  try {
                    const exists = await chatsCol.findOne(
                      { _id: new ObjectId(chatId), userId: session.userId },
                      { projection: { _id: 1 } },
                    );
                    if (!exists) chatId = null;
                  } catch {
                    chatId = null;
                  }
                }
                if (!chatId) {
                  const created = await createChat(session.userId);
                  chatId = created.id;
                  socket.send(
                    JSON.stringify({ type: "chatCreated", id, chat: created }),
                  );
                }
              }

              socket.send(JSON.stringify({ type: "start", id, chatId }));
              await modelProcess.generate(
                socket,
                id,
                payload.prompt,
                {
                  maxTokens: clampInteger(
                    payload.maxTokens,
                    1,
                    MAX_GENERATION_TOKENS,
                    DEFAULT_MAX_TOKENS,
                  ),
                  temperature: payload.temperature ?? 0.7,
                  topP: payload.topP ?? 0.95,
                  repetitionPenalty: payload.repetitionPenalty ?? 1.1,
                  imageWidth: optionalClampedInteger(
                    payload.imageWidth,
                    64,
                    2048,
                    8,
                  ),
                  imageHeight: optionalClampedInteger(
                    payload.imageHeight,
                    64,
                    2048,
                    8,
                  ),
                  steps: optionalClampedInteger(payload.steps, 1, 150),
                  seed: optionalClampedInteger(payload.seed, 0, 2 ** 31 - 1),
                },
                persistedImage?.path ?? null,
                {
                  chatId,
                  userId: session.userId,
                  userText: payload.prompt,
                  userImage: payload.image ?? null,
                  userAt: new Date(),
                },
              );
            } catch (err) {
              cleanupPromptImage(persistedImage?.path);
              socket.send(
                JSON.stringify({
                  type: "error",
                  id,
                  error: err instanceof Error ? err.message : String(err),
                }),
              );
            }
          },
        });
        return;
      }

      socket.send(
        JSON.stringify({ type: "error", error: "Unknown message type." }),
      );
    });
  });
});

// ─── Shutdown ─────────────────────────────────────────────────────────
const shutdown = async (signal) => {
  fastify.log.info(`${signal} received, shutting down.`);
  await fastify.close().catch(() => {});
  await mongoClient.close().catch(() => {});
  process.exit(0);
};
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

// Warm the catalog in the background.
listMlxCommunityModels().catch((err) =>
  fastify.log.warn({ err }, "Failed to preload mlx-community model list"),
);

await fastify.listen({ port: PORT, host: HOST });
console.log(`Chat server → http://${HOST}:${PORT}`);
