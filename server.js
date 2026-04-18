import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { fork } from "node:child_process";
import { createRequire } from "node:module";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";

// Force the standard HuggingFace cache location (~/.cache/huggingface/hub)
// before anything spawns so the worker inherits it.  Without this, the Swift
// hub library can fall back to ~/Documents (which OneDrive syncs on macOS).
if (!process.env.HF_HOME) {
  process.env.HF_HOME = join(homedir(), ".cache", "huggingface");
}
console.log(`HF_HOME → ${process.env.HF_HOME}`);
import Fastify from "fastify";
import fastifyWebsocket from "@fastify/websocket";
import fastifyStatic from "@fastify/static";
import { listModels } from "@huggingface/hub";
import { isSupported, RECOMMENDED_MODELS } from "node-mlx";
import { MongoClient, ObjectId } from "mongodb";

// better-sqlite3 is CommonJS; import it safely from an ESM module.
const require = createRequire(import.meta.url);
const Database = require("better-sqlite3");

const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKER_PATH = join(__dirname, "model-worker.js");

const DEFAULT_MODEL = process.env.MLX_MODEL ?? RECOMMENDED_MODELS["qwen-3-1.7b"];
const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "127.0.0.1";
const MODELS_OWNER = "mlx-community";
const MODELS_TTL_MS = 10 * 60 * 1000;
const IMAGE_TMP_DIR = join(tmpdir(), "nodemlx-chat-images");
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_GENERATION_TOKENS = clampInteger(
  Number(process.env.MLX_MAX_TOKENS_LIMIT ?? 32768),
  1,
  131072,
  32768
);
const DEFAULT_MAX_TOKENS = clampInteger(
  Number(process.env.MLX_MAX_TOKENS ?? 4096),
  1,
  MAX_GENERATION_TOKENS,
  Math.min(4096, MAX_GENERATION_TOKENS)
);
const IMAGE_EXTENSIONS = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
  "image/gif": "gif",
};

if (!isSupported()) {
  console.error("node-mlx requires macOS 14+ on Apple Silicon.");
  process.exit(1);
}

function clampInteger(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

function optionalClampedInteger(value, min, max, multiple = 1) {
  if (!Number.isFinite(value)) return undefined;
  const clamped = Math.min(max, Math.max(min, Math.trunc(value)));
  if (multiple <= 1) return clamped;
  return Math.min(max, Math.max(min, Math.round(clamped / multiple) * multiple));
}

// ─── SQLite — model registry ──────────────────────────────────────────
const db = new Database(join(__dirname, "mlx-chat.db"));
db.exec(`
  CREATE TABLE IF NOT EXISTS failed_models (
    model_id  TEXT PRIMARY KEY,
    error     TEXT NOT NULL,
    failed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
  );

  CREATE TABLE IF NOT EXISTS saved_models (
    model_id  TEXT PRIMARY KEY,
    is_vlm    INTEGER NOT NULL DEFAULT 0,
    last_used TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
  );
`);
const stmt = {
  upsertFailed: db.prepare(`
    INSERT INTO failed_models (model_id, error)
    VALUES (?, ?)
    ON CONFLICT(model_id) DO UPDATE
      SET error = excluded.error,
          failed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
  `),
  deleteFailed: db.prepare("DELETE FROM failed_models WHERE model_id = ?"),
  allFailed: db.prepare(
    "SELECT model_id AS id, error, failed_at AS failedAt FROM failed_models"
  ),
  isFailed: db.prepare(
    "SELECT 1 FROM failed_models WHERE model_id = ?"
  ),
  upsertSaved: db.prepare(`
    INSERT INTO saved_models (model_id, is_vlm)
    VALUES (?, ?)
    ON CONFLICT(model_id) DO UPDATE
      SET is_vlm    = excluded.is_vlm,
          last_used = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
  `),
  allSaved: db.prepare(
    "SELECT model_id AS id, is_vlm AS isVlm, last_used AS lastUsed FROM saved_models ORDER BY last_used DESC"
  ),
};

// ─── MongoDB — chat persistence ──────────────────────────────────────
const MONGO_URL = process.env.MONGO_URL ?? "mongodb://192.168.1.80:27017";
const MONGO_DB = process.env.MONGO_DB ?? "NodeMLX";
const mongoClient = new MongoClient(MONGO_URL);
let chatsCol = null;
try {
  await mongoClient.connect();
  chatsCol = mongoClient.db(MONGO_DB).collection("Chats");
  await chatsCol.createIndex({ startedAt: -1 });
  console.log(`Mongo → ${MONGO_URL}/${MONGO_DB}`);
} catch (err) {
  console.error("MongoDB connection failed:", err.message);
}

function toChatSummary(doc) {
  return {
    id: doc._id.toString(),
    startedAt: doc.startedAt instanceof Date ? doc.startedAt.toISOString() : doc.startedAt,
    title: doc.title ?? null,
    messageCount: Array.isArray(doc.messages) ? doc.messages.length : 0,
  };
}

async function createChat() {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  const now = new Date();
  const res = await chatsCol.insertOne({ startedAt: now, title: null, messages: [] });
  return { id: res.insertedId.toString(), startedAt: now.toISOString(), title: null, messageCount: 0 };
}

async function appendChatMessages(chatId, entries) {
  if (!chatsCol || !chatId) return;
  let id;
  try { id = new ObjectId(chatId); } catch { return; }
  const update = { $push: { messages: { $each: entries } } };
  const firstUserEntry = entries.find((e) => e.role === "user" && e.text);
  if (firstUserEntry) {
    const doc = await chatsCol.findOne({ _id: id }, { projection: { title: 1 } });
    if (doc && !doc.title) {
      update.$set = { title: firstUserEntry.text.slice(0, 80) };
    }
  }
  await chatsCol.updateOne({ _id: id }, update);
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

  const match = /^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/.exec(image.dataUrl);
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

// ─── ModelProcess — owns the child process lifecycle ─────────────────
class ModelProcess {
  #worker = null;
  #workerReady = false;
  #currentModelId = null;
  #pendingModelId = null;
  #loading = false;
  #lastError = null;
  #isVLM = false;
  #canGenerateImages = false;
  // Map<requestId, { socket: WebSocket, imagePath?: string }> for in-flight generate calls
  #pending = new Map();

  constructor() {
    this.#spawnWorker();
  }

  // ── Public accessors ──────────────────────────────────────────────
  get currentModelId() { return this.#currentModelId; }
  get loading()        { return this.#loading; }

  /** Returns the WS message a newly-connected client should receive. */
  greetingMessage() {
    if (this.#loading)         return { type: "modelLoading", modelId: this.#pendingModelId };
    if (this.#currentModelId)  return {
      type: "modelReady",
      modelId: this.#currentModelId,
      isVLM: this.#isVLM,
      canGenerateImages: this.#canGenerateImages,
    };
    return { type: "modelError", error: this.#lastError ?? "No model loaded." };
  }

  // ── Commands ──────────────────────────────────────────────────────
  load(modelId) {
    if (!modelId || typeof modelId !== "string") throw new Error("modelId required.");
    if (this.#loading) throw new Error("Another model is already loading.");
    if (this.#currentModelId === modelId) {
      broadcast({
        type: "modelReady",
        modelId,
        isVLM: this.#isVLM,
        canGenerateImages: this.#canGenerateImages,
      });
      return;
    }
    this.#loading = true;
    this.#currentModelId = null;
    this.#pendingModelId = modelId;
    this.#lastError = null;
    this.#isVLM = false;
    this.#canGenerateImages = false;
    broadcast({ type: "modelLoading", modelId });
    if (this.#workerReady) {
      this.#worker.send({ type: "loadModel", modelId });
    }
    // If the worker isn't ready yet, the workerReady handler flushes this.
  }

  generate(socket, id, prompt, options, imagePath = null, context = {}) {
    if (!this.#workerReady || !this.#worker?.connected) {
      throw new Error("Model worker not ready.");
    }
    if (this.#loading) throw new Error("A model is still loading.");
    if (!this.#currentModelId) throw new Error("No model loaded.");
    if (imagePath && !this.#isVLM) {
      throw new Error("The active node-mlx backend does not support image input for this model.");
    }
    this.#pending.set(id, { socket, imagePath, ...context });
    this.#worker.send({ type: "generate", id, prompt, options, imagePath });
  }

  // ── Worker lifecycle ──────────────────────────────────────────────
  #spawnWorker() {
    const w = fork(WORKER_PATH);
    this.#worker = w;
    this.#workerReady = false;

    w.on("message", (msg) => {
      if (msg.type === "workerReady") {
        this.#workerReady = true;
        // Flush any load that arrived before the worker was up.
        if (this.#loading && this.#pendingModelId) {
          w.send({ type: "loadModel", modelId: this.#pendingModelId });
        }
        return;
      }
      this.#onWorkerMessage(msg);
    });

    w.on("exit", (code, signal) => this.#onWorkerExit(code, signal));
    w.on("error", (err) => console.error("[worker] process error:", err));
  }

  #onWorkerMessage(msg) {
    switch (msg.type) {
      case "modelReady": {
        this.#currentModelId = msg.modelId;
        this.#pendingModelId = null;
        this.#loading = false;
        this.#lastError = null;
        this.#isVLM = msg.isVLM === true;
        this.#canGenerateImages = msg.canGenerateImages === true;
        // If it previously failed, clear the record now that it works.
        stmt.deleteFailed.run(msg.modelId);
        // Remember this model so it appears in the selector on future visits.
        stmt.upsertSaved.run(msg.modelId, this.#isVLM ? 1 : 0);
        broadcast({
          type: "modelReady",
          modelId: msg.modelId,
          isVLM: this.#isVLM,
          canGenerateImages: this.#canGenerateImages,
        });
        break;
      }

      case "modelError": {
        this.#pendingModelId = null;
        this.#loading = false;
        this.#lastError = msg.error;
        this.#isVLM = false;
        this.#canGenerateImages = false;
        stmt.upsertFailed.run(msg.modelId, msg.error);
        broadcast({
          type: "modelError",
          modelId: msg.modelId,
          error: msg.error,
          failed: stmt.allFailed.all(),
        });
        break;
      }

      case "generateResult": {
        const pending = this.#pending.get(msg.id);
        this.#pending.delete(msg.id);
        cleanupPromptImage(pending?.imagePath);
        const socket = pending?.socket;
        if (socket?.readyState === 1) {
          socket.send(JSON.stringify({
            type: "response",
            id: msg.id,
            chatId: pending?.chatId ?? null,
            modelId: msg.modelId,
            text: msg.text,
            images: msg.images ?? [],
            tokenCount: msg.tokenCount,
            tokensPerSecond: msg.tokensPerSecond,
          }));
        }
        if (pending?.chatId) {
          const now = new Date();
          const entries = [{
            id: msg.id,
            role: "user",
            text: pending.userText ?? "",
            image: pending.userImage ?? null,
            createdAt: pending.userAt ?? now,
          }, {
            id: `${msg.id}:reply`,
            role: "assistant",
            text: msg.text ?? "",
            images: msg.images ?? [],
            modelId: msg.modelId ?? null,
            tokenCount: msg.tokenCount ?? null,
            tokensPerSecond: msg.tokensPerSecond ?? null,
            createdAt: now,
          }];
          appendChatMessages(pending.chatId, entries).catch((err) =>
            console.error("Mongo append failed:", err.message)
          );
        }
        break;
      }

      case "generateError": {
        const pending = this.#pending.get(msg.id);
        this.#pending.delete(msg.id);
        cleanupPromptImage(pending?.imagePath);
        const socket = pending?.socket;
        if (socket?.readyState === 1) {
          socket.send(JSON.stringify({ type: "error", id: msg.id, error: msg.error }));
        }
        break;
      }
    }
  }

  #onWorkerExit(code, signal) {
    this.#workerReady = false;
    this.#worker = null;

    // If the crash happened while loading, record the model as failed.
    if (this.#loading && this.#pendingModelId) {
      const modelId = this.#pendingModelId;
      const error =
        `Crashed while loading (${signal ?? `exit ${code}`}) — ` +
        "this model is incompatible with the installed version of node-mlx.";
      this.#pendingModelId = null;
      this.#loading = false;
      this.#lastError = error;
      stmt.upsertFailed.run(modelId, error);
      broadcast({
        type: "modelError",
        modelId,
        error,
        failed: stmt.allFailed.all(),
      });
      this.#canGenerateImages = false;
    }

    // Fail any in-flight generate requests.
    for (const [id, pending] of this.#pending) {
      cleanupPromptImage(pending.imagePath);
      const socket = pending.socket;
      if (socket?.readyState === 1) {
        socket.send(
          JSON.stringify({ type: "error", id, error: "Model process crashed." })
        );
      }
    }
    this.#pending.clear();

    // Respawn so the server stays usable.
    setTimeout(() => this.#spawnWorker(), 500);
  }
}

const modelProcess = new ModelProcess();

// Load the default model (queued; fires once the worker is ready).
console.log(`Queuing initial model load: ${DEFAULT_MODEL}`);
modelProcess.load(DEFAULT_MODEL);

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
      likes:     typeof m.likes     === "number" ? m.likes     : 0,
      updatedAt: m.updatedAt ? new Date(m.updatedAt).toISOString() : null,
    });
  }
  out.sort((a, b) => b.downloads - a.downloads);
  modelsCache  = out;
  modelsCacheAt = Date.now();
  return out;
}

// ─── Fastify ─────────────────────────────────────────────────────────
const fastify = Fastify({ logger: true });
await fastify.register(fastifyWebsocket);
await fastify.register(fastifyStatic, {
  root: join(__dirname, "client", "dist", "chat-client", "browser"),
  prefix: "/",
  decorateReply: false,
});

fastify.get("/api/health", async () => ({
  ok: true,
  model: modelProcess.currentModelId,
  loading: modelProcess.loading,
}));

fastify.get("/api/models", async (req, reply) => {
  try {
    const hfModels = await listMlxCommunityModels(req.query?.refresh === "1");

    // Merge locally-saved models (those the user has successfully loaded before)
    // into the HuggingFace catalog.  Saved models that already appear in the HF
    // list get a `saved: true` flag so the UI can highlight them; saved models
    // that are NOT in the HF list (e.g. private or non-mlx-community repos) are
    // appended at the end with zeroed download/likes counters.
    const savedRows = stmt.allSaved.all();  // [{id, isVlm, lastUsed}]
    if (savedRows.length > 0) {
      const hfIds = new Set(hfModels.map((m) => m.id));
      const savedIds = new Set(savedRows.map((r) => r.id));

      // Mark HF models that are also in the saved set.
      const merged = hfModels.map((m) => ({ ...m, saved: savedIds.has(m.id) }));

      // Append saved models not present in the HF list.
      for (const row of savedRows) {
        if (!hfIds.has(row.id)) {
          merged.push({ id: row.id, downloads: 0, likes: 0, updatedAt: null, saved: true });
        }
      }

      // Sort: saved models first (by last_used desc), then HF models by downloads desc.
      const savedOrder = new Map(savedRows.map((r, i) => [r.id, i]));
      merged.sort((a, b) => {
        const aSaved = a.saved ? savedOrder.get(a.id) ?? Infinity : Infinity;
        const bSaved = b.saved ? savedOrder.get(b.id) ?? Infinity : Infinity;
        if (aSaved !== bSaved) return aSaved - bSaved;
        return b.downloads - a.downloads;
      });

      return { owner: MODELS_OWNER, models: merged };
    }

    return { owner: MODELS_OWNER, models: hfModels };
  } catch (err) {
    reply.code(502);
    return { error: err instanceof Error ? err.message : String(err), models: [] };
  }
});

fastify.get("/api/models/failed", async () => ({
  failed: stmt.allFailed.all(),
}));

fastify.delete("/api/models/failed/:modelId", async (req, reply) => {
  const { modelId } = req.params;
  stmt.deleteFailed.run(modelId);
  reply.code(204).send();
});

// ─── Chats REST ──────────────────────────────────────────────────────
fastify.get("/api/chats", async (_req, reply) => {
  if (!chatsCol) { reply.code(503); return { error: "Chat storage unavailable.", chats: [] }; }
  const docs = await chatsCol
    .find({}, { projection: { messages: 0 } })
    .sort({ startedAt: -1 })
    .toArray();
  return { chats: docs.map((d) => ({
    id: d._id.toString(),
    startedAt: d.startedAt instanceof Date ? d.startedAt.toISOString() : d.startedAt,
    title: d.title ?? null,
  })) };
});

fastify.post("/api/chats", async (_req, reply) => {
  try {
    const chat = await createChat();
    return chat;
  } catch (err) {
    reply.code(503);
    return { error: err instanceof Error ? err.message : String(err) };
  }
});

fastify.get("/api/chats/:id", async (req, reply) => {
  if (!chatsCol) { reply.code(503); return { error: "Chat storage unavailable." }; }
  let id;
  try { id = new ObjectId(req.params.id); } catch { reply.code(400); return { error: "Invalid chat id." }; }
  const doc = await chatsCol.findOne({ _id: id });
  if (!doc) { reply.code(404); return { error: "Chat not found." }; }
  return {
    id: doc._id.toString(),
    startedAt: doc.startedAt instanceof Date ? doc.startedAt.toISOString() : doc.startedAt,
    title: doc.title ?? null,
    messages: (doc.messages ?? []).map((m) => ({
      id: m.id,
      role: m.role,
      text: m.text ?? "",
      image: m.image ?? undefined,
      images: m.images ?? undefined,
      modelId: m.modelId ?? undefined,
      tokenCount: m.tokenCount ?? undefined,
      tokensPerSecond: m.tokensPerSecond ?? undefined,
    })),
  };
});

fastify.delete("/api/chats/:id", async (req, reply) => {
  if (!chatsCol) { reply.code(503); return { error: "Chat storage unavailable." }; }
  let id;
  try { id = new ObjectId(req.params.id); } catch { reply.code(400); return { error: "Invalid chat id." }; }
  await chatsCol.deleteOne({ _id: id });
  reply.code(204).send();
});

fastify.delete("/api/chats/:id/messages/:messageId", async (req, reply) => {
  if (!chatsCol) { reply.code(503); return { error: "Chat storage unavailable." }; }
  let id;
  try { id = new ObjectId(req.params.id); } catch { reply.code(400); return { error: "Invalid chat id." }; }
  const messageId = req.params.messageId;
  if (typeof messageId !== "string" || !messageId) {
    reply.code(400);
    return { error: "Invalid message id." };
  }
  await chatsCol.updateOne({ _id: id }, { $pull: { messages: { id: messageId } } });
  reply.code(204).send();
});

// ─── WebSocket endpoint ───────────────────────────────────────────────
fastify.register(async (instance) => {
  instance.get("/ws", { websocket: true }, (socket) => {
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

      // ── selectModel ──────────────────────────────────────────────
      if (payload?.type === "selectModel") {
        try {
          modelProcess.load(payload.modelId);
        } catch (err) {
          socket.send(JSON.stringify({
            type: "modelError",
            error: err instanceof Error ? err.message : String(err),
          }));
        }
        return;
      }

      // ── prompt ───────────────────────────────────────────────────
      if (payload?.type === "prompt" && typeof payload.prompt === "string") {
        const id = payload.id ?? String(Date.now());
        let persistedImage = null;
        try {
          persistedImage = await persistPromptImage(payload.image);

          let chatId = typeof payload.chatId === "string" ? payload.chatId : null;
          if (chatsCol) {
            if (chatId) {
              try {
                const exists = await chatsCol.findOne({ _id: new ObjectId(chatId) }, { projection: { _id: 1 } });
                if (!exists) chatId = null;
              } catch { chatId = null; }
            }
            if (!chatId) {
              const created = await createChat();
              chatId = created.id;
              socket.send(JSON.stringify({ type: "chatCreated", id, chat: created }));
            }
          }

          socket.send(JSON.stringify({ type: "start", id, chatId }));
          modelProcess.generate(socket, id, payload.prompt, {
            maxTokens:          clampInteger(payload.maxTokens, 1, MAX_GENERATION_TOKENS, DEFAULT_MAX_TOKENS),
            temperature:        payload.temperature        ?? 0.7,
            topP:               payload.topP               ?? 0.95,
            repetitionPenalty:  payload.repetitionPenalty  ?? 1.1,
            imageWidth:         optionalClampedInteger(payload.imageWidth, 64, 2048, 8),
            imageHeight:        optionalClampedInteger(payload.imageHeight, 64, 2048, 8),
            steps:              optionalClampedInteger(payload.steps, 1, 150),
            seed:               optionalClampedInteger(payload.seed, 0, 2 ** 31 - 1),
          }, persistedImage?.path ?? null, {
            chatId,
            userText: payload.prompt,
            userImage: payload.image ?? null,
            userAt: new Date(),
          });
        } catch (err) {
          cleanupPromptImage(persistedImage?.path);
          socket.send(JSON.stringify({
            type: "error",
            id,
            error: err instanceof Error ? err.message : String(err),
          }));
        }
        return;
      }

      socket.send(JSON.stringify({ type: "error", error: "Unknown message type." }));
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
process.on("SIGINT",  () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

// Warm the catalog in the background.
listMlxCommunityModels().catch((err) =>
  fastify.log.warn({ err }, "Failed to preload mlx-community model list")
);

await fastify.listen({ port: PORT, host: HOST });
console.log(`Chat server → http://${HOST}:${PORT}`);
