import { fileURLToPath } from "node:url";
import { basename, dirname, join } from "node:path";
import { homedir } from "node:os";
import { execFile, fork, spawn } from "node:child_process";
import { createRequire } from "node:module";
import { mkdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { randomBytes, randomUUID, scrypt, timingSafeEqual } from "node:crypto";
import { promisify } from "node:util";

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
const GPU_USAGE_PATH = join(__dirname, "utils", "GPUUsage");

const DEFAULT_MODEL = process.env.MLX_MODEL ?? RECOMMENDED_MODELS["qwen-3-1.7b"];
const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "127.0.0.1";
const MODELS_OWNER = "mlx-community";
const MODELS_TTL_MS = 10 * 60 * 1000;
const OLLAMA_URL = process.env.OLLAMA_URL ?? "http://127.0.0.1:11434";
const IMAGE_TMP_DIR = join(tmpdir(), "nodemlx-chat-images");
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_LLAMA_MODEL_BYTES = 16_000_000_000;
const MAX_LLAMA_OUTPUT_CHARS = 2_000_000;
const SESSION_COOKIE_NAME = "nodemlx_session";
const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;
const MIN_USERNAME_LENGTH = 3;
const MAX_USERNAME_LENGTH = 40;
const MIN_PASSWORD_LENGTH = 8;
const MAX_GENERATION_TOKENS = clampInteger(
  Number(process.env.MLX_MAX_TOKENS_LIMIT ?? 32768),
  1,
  131072,
  32768
);
const MEMORY_CANCEL_THRESHOLD = 98;
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
db.pragma("foreign_keys = ON");
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

  CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    last_login_at TEXT
  );

  CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    user_id      INTEGER NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    expires_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
  );

  CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
  CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
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
  createUser: db.prepare(`
    INSERT INTO users (username, password_hash)
    VALUES (?, ?)
  `),
  getUserForLogin: db.prepare(`
    SELECT id, username, password_hash AS passwordHash
    FROM users
    WHERE username = ?
  `),
  createSession: db.prepare(`
    INSERT INTO sessions (id, user_id, expires_at)
    VALUES (?, ?, ?)
  `),
  getSessionUser: db.prepare(`
    SELECT
      s.id AS sessionId,
      s.user_id AS userId,
      u.username AS username
    FROM sessions s
    JOIN users u ON u.id = s.user_id
    WHERE s.id = ?
      AND datetime(s.expires_at) > datetime('now')
  `),
  touchSession: db.prepare(`
    UPDATE sessions
    SET last_seen_at = ?
    WHERE id = ?
  `),
  deleteSession: db.prepare("DELETE FROM sessions WHERE id = ?"),
  deleteExpiredSessions: db.prepare(`
    DELETE FROM sessions
    WHERE datetime(expires_at) <= datetime('now')
  `),
  updateUserLastLogin: db.prepare(`
    UPDATE users
    SET last_login_at = ?
    WHERE id = ?
  `),
};
stmt.deleteExpiredSessions.run();

const scryptAsync = promisify(scrypt);
const execFileAsync = promisify(execFile);

function normalizeUsername(value) {
  if (typeof value !== "string") return "";
  return value.trim();
}

function validateUsername(username) {
  if (username.length < MIN_USERNAME_LENGTH || username.length > MAX_USERNAME_LENGTH) {
    throw new Error(`Username must be ${MIN_USERNAME_LENGTH}-${MAX_USERNAME_LENGTH} characters.`);
  }
  if (!/^[A-Za-z0-9._-]+$/.test(username)) {
    throw new Error("Username may contain only letters, numbers, dot, underscore, and hyphen.");
  }
}

function validatePassword(password) {
  if (typeof password !== "string" || password.length < MIN_PASSWORD_LENGTH) {
    throw new Error(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
  }
}

async function hashPassword(password) {
  const salt = randomBytes(16).toString("hex");
  const derived = await scryptAsync(password, salt, 64);
  return `${salt}:${Buffer.from(derived).toString("hex")}`;
}

async function verifyPassword(password, storedHash) {
  if (typeof storedHash !== "string") return false;
  const [salt, expectedHex] = storedHash.split(":");
  if (!salt || !expectedHex) return false;
  const expected = Buffer.from(expectedHex, "hex");
  if (expected.length === 0) return false;
  const actual = Buffer.from(await scryptAsync(password, salt, expected.length));
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

function parseCookies(cookieHeader) {
  const out = {};
  if (typeof cookieHeader !== "string" || cookieHeader.length === 0) return out;
  for (const part of cookieHeader.split(";")) {
    const eq = part.indexOf("=");
    if (eq <= 0) continue;
    const key = part.slice(0, eq).trim();
    const value = part.slice(eq + 1).trim();
    if (!key) continue;
    try {
      out[key] = decodeURIComponent(value);
    } catch {
      out[key] = value;
    }
  }
  return out;
}

function serializeCookie(name, value, options = {}) {
  const pairs = [`${name}=${encodeURIComponent(value)}`];
  pairs.push(`Path=${options.path ?? "/"}`);
  pairs.push(`SameSite=${options.sameSite ?? "Lax"}`);
  if (typeof options.maxAge === "number") pairs.push(`Max-Age=${Math.max(0, Math.trunc(options.maxAge))}`);
  if (options.httpOnly !== false) pairs.push("HttpOnly");
  if (options.secure === true) pairs.push("Secure");
  return pairs.join("; ");
}

function isSecureRequest(request) {
  if (request.protocol === "https") return true;
  const forwardedProto = request.headers["x-forwarded-proto"];
  return typeof forwardedProto === "string" && forwardedProto.split(",")[0].trim() === "https";
}

function sessionCookieValue(request) {
  const cookies = parseCookies(request.headers.cookie);
  const raw = cookies[SESSION_COOKIE_NAME];
  if (typeof raw !== "string" || raw.length === 0) return null;
  return raw;
}

function setSessionCookie(reply, request, sessionId) {
  reply.header("Set-Cookie", serializeCookie(SESSION_COOKIE_NAME, sessionId, {
    path: "/",
    sameSite: "Lax",
    httpOnly: true,
    secure: isSecureRequest(request),
    maxAge: Math.floor(SESSION_TTL_MS / 1000),
  }));
}

function clearSessionCookie(reply, request) {
  reply.header("Set-Cookie", serializeCookie(SESSION_COOKIE_NAME, "", {
    path: "/",
    sameSite: "Lax",
    httpOnly: true,
    secure: isSecureRequest(request),
    maxAge: 0,
  }));
}

function createSessionForUser(userId) {
  const sessionId = randomBytes(32).toString("hex");
  const expiresAt = new Date(Date.now() + SESSION_TTL_MS).toISOString();
  stmt.createSession.run(sessionId, userId, expiresAt);
  return sessionId;
}

function readSessionUser(request) {
  stmt.deleteExpiredSessions.run();
  const sessionId = sessionCookieValue(request);
  if (!sessionId) return null;
  const session = stmt.getSessionUser.get(sessionId);
  if (!session) return null;
  stmt.touchSession.run(new Date().toISOString(), sessionId);
  return session;
}

// ─── MongoDB — chat persistence ──────────────────────────────────────
const MONGO_URL = process.env.MONGO_URL ?? "mongodb://192.168.1.80:27017";
const MONGO_DB = process.env.MONGO_DB ?? "NodeMLX";
const mongoClient = new MongoClient(MONGO_URL);
let chatsCol = null;
try {
  await mongoClient.connect();
  chatsCol = mongoClient.db(MONGO_DB).collection("Chats");
  await chatsCol.createIndex({ startedAt: -1 });
  await chatsCol.createIndex({ userId: 1, startedAt: -1 });
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

async function createChat(userId) {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  if (typeof userId !== "number") throw new Error("Missing user id.");
  const now = new Date();
  const res = await chatsCol.insertOne({ userId, startedAt: now, title: null, messages: [] });
  return { id: res.insertedId.toString(), startedAt: now.toISOString(), title: null, messageCount: 0 };
}

async function appendChatMessages(userId, chatId, entries) {
  if (!chatsCol || !chatId || typeof userId !== "number") return;
  let id;
  try { id = new ObjectId(chatId); } catch { return; }
  const update = { $push: { messages: { $each: entries } } };
  const firstUserEntry = entries.find((e) => e.role === "user" && e.text);
  if (firstUserEntry) {
    const doc = await chatsCol.findOne({ _id: id, userId }, { projection: { title: 1 } });
    if (doc && !doc.title) {
      update.$set = { title: firstUserEntry.text.slice(0, 80) };
    }
  }
  await chatsCol.updateOne({ _id: id, userId }, update);
}

async function ensureUserChat(userId, chatId) {
  if (!chatsCol) return { chatId: null, created: null };
  if (chatId) {
    try {
      const id = new ObjectId(chatId);
      const exists = await chatsCol.findOne(
        { _id: id, userId },
        { projection: { _id: 1 } }
      );
      if (exists) return { chatId, created: null };
    } catch {}
  }
  const created = await createChat(userId);
  return { chatId: created.id, created };
}

// ─── WebSocket broadcast helpers ─────────────────────────────────────
const sockets = new Set();
function broadcast(msg) {
  const payload = JSON.stringify(msg);
  for (const s of sockets) {
    if (s.readyState === 1) s.send(payload);
  }
}

function parseGpuUsageOutput(output) {
  const text = typeof output === "string" ? output : "";
  const gpuMatch = /GPU Usage:\s*([0-9]+(?:\.[0-9]+)?)%/i.exec(text);
  const memoryMatch = /Memory Usage:\s*([0-9]+(?:\.[0-9]+)?)%/i.exec(text);
  if (!gpuMatch || !memoryMatch) return null;
  return {
    gpu: Number(gpuMatch[1]),
    memory: Number(memoryMatch[1]),
  };
}

let gpuUsagePollTimer = null;
let gpuUsagePolling = false;
let activeInferenceCount = 0;
const activeInferenceCancels = new Set();

function registerInferenceCancel(cancel) {
  activeInferenceCancels.add(cancel);
  return () => activeInferenceCancels.delete(cancel);
}

async function pollGpuUsageOnce() {
  if (gpuUsagePolling || activeInferenceCount <= 0) return;
  gpuUsagePolling = true;
  try {
    const { stdout } = await execFileAsync(GPU_USAGE_PATH);
    const usage = parseGpuUsageOutput(stdout);
    if (usage && activeInferenceCount > 0) {
      broadcast({ type: "gpuUsage", running: true, ...usage });
      if (usage.memory >= MEMORY_CANCEL_THRESHOLD) {
        cancelAllInference(`Memory usage reached ${usage.memory}%. Inference stopped.`);
      }
    }
  } catch (err) {
    console.warn(`GPU usage polling failed: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    gpuUsagePolling = false;
  }
}

function startGpuUsagePolling() {
  activeInferenceCount += 1;
  if (activeInferenceCount !== 1) return;
  broadcast({ type: "gpuUsage", running: true, gpu: null, memory: null });
  void pollGpuUsageOnce();
  gpuUsagePollTimer = setInterval(() => {
    void pollGpuUsageOnce();
  }, 1000);
}

function stopGpuUsagePolling() {
  activeInferenceCount = Math.max(0, activeInferenceCount - 1);
  if (activeInferenceCount > 0) return;
  if (gpuUsagePollTimer) {
    clearInterval(gpuUsagePollTimer);
    gpuUsagePollTimer = null;
  }
  broadcast({ type: "gpuUsage", running: false, gpu: null, memory: null });
}

async function validateLlamaModelFile(modelPath) {
  const filePath = typeof modelPath === "string" ? modelPath.trim() : "";
  if (!filePath) throw new Error("Choose a Llama.cpp model file first.");

  const info = await stat(filePath).catch((err) => {
    throw new Error(`Llama.cpp model file is not available: ${err instanceof Error ? err.message : String(err)}`);
  });
  if (!info.isFile()) throw new Error("Llama.cpp model path must point to a file.");
  if (info.size > MAX_LLAMA_MODEL_BYTES) {
    throw new Error("Llama.cpp model file must be 16 GB or smaller.");
  }
  return {
    path: filePath,
    name: basename(filePath),
    size: info.size,
  };
}

async function pickLlamaModelFile() {
  const { stdout } = await execFileAsync("osascript", [
    "-e",
    'POSIX path of (choose file with prompt "Choose a Llama.cpp model file")',
  ]);
  return validateLlamaModelFile(stdout.trim());
}

function validateHuggingFaceModelName(value) {
  const model = typeof value === "string" ? value.trim() : "";
  if (!model) throw new Error("Enter a Hugging Face model name first.");
  if (model.length > 300 || /\s/.test(model)) {
    throw new Error("Hugging Face model name must be a single model id without whitespace.");
  }
  return model;
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

function promptImageForOllama(image) {
  if (!image) return null;
  if (typeof image !== "object" || typeof image.dataUrl !== "string") {
    throw new Error("Invalid image payload.");
  }

  const match = /^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/.exec(image.dataUrl);
  if (!match) {
    throw new Error("Image must be a JPEG, PNG, WebP, or GIF data URL.");
  }

  const buffer = Buffer.from(match[2], "base64");
  if (buffer.byteLength === 0) {
    throw new Error("Image is empty.");
  }
  if (buffer.byteLength > MAX_IMAGE_BYTES) {
    throw new Error("Image is too large. Maximum size is 10 MB.");
  }

  return {
    base64: match[2],
    attachment: {
      dataUrl: image.dataUrl,
      name: typeof image.name === "string" ? image.name.slice(0, 200) : "image",
      type: match[1],
      size: buffer.byteLength,
    },
  };
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
    return new Promise((resolve) => {
      this.#pending.set(id, { socket, imagePath, resolve, ...context });
      this.#worker.send({ type: "generate", id, prompt, options, imagePath });
    });
  }

  cancelGeneration(reason = "Inference cancelled.") {
    if (this.#pending.size === 0) return false;

    for (const [id, pending] of this.#pending) {
      cleanupPromptImage(pending.imagePath);
      const socket = pending.socket;
      if (socket?.readyState === 1) {
        socket.send(JSON.stringify({ type: "error", id, error: reason }));
      }
      pending.resolve?.();
    }
    this.#pending.clear();

    if (this.#worker) {
      const worker = this.#worker;
      let exited = false;
      worker.removeAllListeners("exit");
      worker.removeAllListeners("message");
      worker.once("exit", () => { exited = true; });
      worker.kill("SIGTERM");
      setTimeout(() => {
        if (!exited) {
          try { worker.kill("SIGKILL"); } catch {}
        }
      }, 2000);
    }
    this.#worker = null;
    this.#workerReady = false;
    this.#currentModelId = null;
    this.#pendingModelId = null;
    this.#loading = false;
    this.#lastError = reason;
    this.#isVLM = false;
    this.#canGenerateImages = false;
    broadcast({ type: "modelError", error: reason, failed: stmt.allFailed.all() });

    setTimeout(() => this.#spawnWorker(), 500);
    return true;
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
        if (pending?.chatId && typeof pending.userId === "number") {
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
          appendChatMessages(pending.userId, pending.chatId, entries).catch((err) =>
            console.error("Mongo append failed:", err.message)
          );
        }
        pending?.resolve?.();
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
        pending?.resolve?.();
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
      pending.resolve?.();
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

// ─── GenerationQueue — serializes user generation requests ───────────
class GenerationQueue {
  #active = false;
  #items = [];

  enqueue(item) {
    const willWait = this.#active || this.#items.length > 0;
    this.#items.push(item);
    if (willWait) this.#notifyQueuedPositions();
    void this.#drain();
  }

  cancelQueued(reason = "Inference cancelled.") {
    const queued = this.#items;
    this.#items = [];
    for (const item of queued) {
      if (item.socket?.readyState === 1) {
        item.socket.send(JSON.stringify({ type: "error", id: item.id, error: reason }));
      }
    }
  }

  async #drain() {
    if (this.#active) return;

    while (this.#items.length > 0) {
      const item = this.#items.shift();
      this.#notifyQueuedPositions();

      if (item.socket?.readyState !== 1) continue;

      this.#active = true;
      startGpuUsagePolling();
      try {
        await item.run();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (item.socket?.readyState === 1) {
          item.socket.send(JSON.stringify({ type: "error", id: item.id, error: message }));
        }
      } finally {
        stopGpuUsagePolling();
        this.#active = false;
      }
    }
  }

  #notifyQueuedPositions() {
    this.#items = this.#items.filter((item) => item.socket?.readyState === 1);
    this.#items.forEach((item, index) => {
      item.socket.send(JSON.stringify({
        type: "queued",
        id: item.id,
        position: index + 1,
      }));
    });
  }
}

const generationQueue = new GenerationQueue();

function cancelAllInference(reason = "Inference cancelled.") {
  generationQueue.cancelQueued(reason);
  modelProcess.cancelGeneration(reason);
  for (const cancel of [...activeInferenceCancels]) {
    try {
      cancel(reason);
    } catch (err) {
      console.warn(`Inference cancel hook failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

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

// ─── Ollama helpers ─────────────────────────────────────────────────
async function listOllamaModels() {
  const res = await fetch(`${OLLAMA_URL}/api/tags`);
  if (!res.ok) {
    throw new Error(`Ollama model list failed (${res.status}).`);
  }
  const body = await res.json();
  const models = Array.isArray(body?.models) ? body.models : [];
  return models
    .map((m) => ({
      id: typeof m.name === "string" ? m.name : "",
      name: typeof m.name === "string" ? m.name : "",
      modifiedAt: typeof m.modified_at === "string" ? m.modified_at : null,
      size: typeof m.size === "number" ? m.size : 0,
      digest: typeof m.digest === "string" ? m.digest : null,
      details: m.details && typeof m.details === "object" ? m.details : null,
    }))
    .filter((m) => m.id.length > 0)
    .sort((a, b) => a.id.localeCompare(b.id));
}

async function showOllamaModel(model) {
  const id = typeof model === "string" ? model.trim() : "";
  if (!id) throw new Error("Ollama model required.");

  const res = await fetch(`${OLLAMA_URL}/api/show`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: id }),
  });
  if (!res.ok) {
    const message = await res.text().catch(() => "");
    throw new Error(`Ollama model details failed (${res.status})${message ? `: ${message}` : "."}`);
  }

  const body = await res.json();
  return {
    id,
    capabilities: Array.isArray(body?.capabilities)
      ? body.capabilities.filter((capability) => typeof capability === "string")
      : [],
    details: body?.details && typeof body.details === "object" ? body.details : null,
    modifiedAt: typeof body?.modified_at === "string" ? body.modified_at : null,
  };
}

function collectOllamaImages(parsed) {
  const images = [];
  addOllamaImageCandidate(images, parsed?.images);
  addOllamaImageCandidate(images, parsed?.image);
  addOllamaImageCandidate(images, parsed?.message?.images);
  addOllamaImageCandidate(images, parsed?.message?.image);
  addOllamaImageCandidate(images, parsed?.response?.images);
  addOllamaImageCandidate(images, parsed?.response?.image);
  addOllamaImageCandidate(images, parsed?.artifacts);
  addOllamaImageCandidate(images, parsed?.data);
  return images;
}

function addOllamaImageCandidate(images, candidate) {
  if (!candidate) return;
  if (Array.isArray(candidate)) {
    for (const item of candidate) addOllamaImageCandidate(images, item);
    return;
  }

  const normalized = normalizeOllamaImage(candidate, images.length + 1);
  if (normalized) {
    if (!images.some((image) => image.dataUrl === normalized.dataUrl)) images.push(normalized);
    return;
  }

  if (typeof candidate !== "object") return;
  for (const key of ["images", "image", "artifacts", "data"]) {
    if (Object.hasOwn(candidate, key)) addOllamaImageCandidate(images, candidate[key]);
  }
}

function normalizeOllamaImage(candidate, index) {
  if (typeof candidate === "string") {
    return normalizeOllamaImageSource(candidate, undefined, undefined, index);
  }
  if (!candidate || typeof candidate !== "object") return null;

  const mimeType =
    firstString(candidate.mimeType, candidate.mime_type, candidate.mediaType, candidate.media_type, candidate.type) ??
    undefined;
  const name = firstString(candidate.name, candidate.filename, candidate.fileName, candidate.alt) ?? undefined;

  for (const key of ["dataUrl", "data_url", "url", "src", "b64_json", "base64", "image", "data"]) {
    const value = candidate[key];
    if (typeof value !== "string") continue;
    const normalized = normalizeOllamaImageSource(value, mimeType, name, index);
    if (normalized) return normalized;
  }

  return null;
}

function normalizeOllamaImageSource(rawSource, hintMimeType, hintName, index) {
  const source = rawSource.trim();
  if (!source) return null;

  const dataUrl = normalizeImageDataUrl(source, hintMimeType);
  if (dataUrl) {
    return {
      dataUrl: dataUrl.url,
      name: hintName ?? `Ollama image ${index}`,
      type: dataUrl.type,
      size: dataUrl.size,
    };
  }

  if (/^https?:\/\//i.test(source)) {
    return {
      dataUrl: source,
      name: hintName ?? imageNameFromUrl(source, index),
      type: imageTypeFromUrl(source),
      size: 0,
    };
  }

  const base64 = source.replace(/\s+/g, "");
  const type = inferBase64ImageMime(base64, hintMimeType);
  if (!type) return null;
  return {
    dataUrl: `data:${type};base64,${base64}`,
    name: hintName ?? `Ollama image ${index}`,
    type,
    size: Math.floor((base64.length * 3) / 4),
  };
}

function normalizeImageDataUrl(source, hintMimeType) {
  const dataMatch = /^data:(image\/[^;,]+);base64,([\s\S]+)$/i.exec(source);
  if (dataMatch) {
    const type = dataMatch[1].toLowerCase();
    const base64 = dataMatch[2].replace(/\s+/g, "");
    if (!isLikelyBase64(base64)) return null;
    return {
      url: `data:${type};base64,${base64}`,
      type,
      size: Math.floor((base64.length * 3) / 4),
    };
  }

  const mimeMatch = /^(image\/[^;,]+);base64,([\s\S]+)$/i.exec(source);
  if (mimeMatch) {
    const type = mimeMatch[1].toLowerCase();
    const base64 = mimeMatch[2].replace(/\s+/g, "");
    if (!isLikelyBase64(base64)) return null;
    return {
      url: `data:${type};base64,${base64}`,
      type,
      size: Math.floor((base64.length * 3) / 4),
    };
  }

  if (hintMimeType && /^image\//i.test(hintMimeType)) {
    const base64 = source.replace(/\s+/g, "");
    if (!isLikelyBase64(base64)) return null;
    const type = hintMimeType.toLowerCase();
    return {
      url: `data:${type};base64,${base64}`,
      type,
      size: Math.floor((base64.length * 3) / 4),
    };
  }

  return null;
}

function inferBase64ImageMime(base64, hintMimeType) {
  if (!isLikelyBase64(base64)) return null;
  if (hintMimeType && /^image\//i.test(hintMimeType)) return hintMimeType.toLowerCase();

  let bytes;
  try {
    bytes = Buffer.from(base64.slice(0, 96), "base64");
  } catch {
    return null;
  }
  if (bytes.length < 8) return null;
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return "image/png";
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "image/jpeg";
  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) return "image/gif";
  if (
    bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
    bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50
  ) return "image/webp";
  if (bytes.toString("utf8", 0, Math.min(bytes.length, 16)).trimStart().startsWith("<svg")) return "image/svg+xml";
  return null;
}

function isLikelyBase64(value) {
  return value.length >= 64 && /^[A-Za-z0-9+/]+={0,2}$/.test(value);
}

function firstString(...values) {
  return values.find((value) => typeof value === "string" && value.trim().length > 0)?.trim();
}

function imageNameFromUrl(source, index) {
  try {
    const url = new URL(source);
    return decodeURIComponent(url.pathname.split("/").filter(Boolean).pop() || `Ollama image ${index}`);
  } catch {
    return `Ollama image ${index}`;
  }
}

function imageTypeFromUrl(source) {
  const path = source.split("?")[0].toLowerCase();
  if (path.endsWith(".jpg") || path.endsWith(".jpeg")) return "image/jpeg";
  if (path.endsWith(".png")) return "image/png";
  if (path.endsWith(".gif")) return "image/gif";
  if (path.endsWith(".webp")) return "image/webp";
  if (path.endsWith(".svg")) return "image/svg+xml";
  return "image";
}

function mergeOllamaImages(existing, next) {
  const bySource = new Map();
  for (const image of [...existing, ...next]) {
    if (!bySource.has(image.dataUrl)) bySource.set(image.dataUrl, image);
  }
  return [...bySource.values()];
}

function ollamaResponseText(parsed) {
  if (typeof parsed?.response === "string") return parsed.response;
  if (typeof parsed?.message?.content === "string") return parsed.message.content;
  if (typeof parsed?.content === "string") return parsed.content;
  if (typeof parsed?.text === "string") return parsed.text;
  return "";
}

function ollamaResponseThinking(parsed) {
  if (typeof parsed?.thinking === "string") return parsed.thinking;
  if (typeof parsed?.message?.thinking === "string") return parsed.message.thinking;
  return "";
}

function ollamaThinkValue(model) {
  return /\bgpt-oss\b/i.test(model) ? "medium" : true;
}

async function unloadOllamaModel(model) {
  const id = typeof model === "string" ? model.trim() : "";
  if (!id) return;

  try {
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: id, keep_alive: 0 }),
    });
    if (!res.ok) {
      const message = await res.text().catch(() => "");
      console.warn(`Ollama unload failed (${res.status})${message ? `: ${message}` : "."}`);
    }
  } catch (err) {
    console.warn(`Ollama unload failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function extractOllamaImagesFromText(text) {
  const trimmed = text.trim();
  if (!trimmed) return { text, images: [] };

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      const images = collectOllamaImages(parsed);
      if (images.length > 0) {
        const nestedText = ollamaResponseText(parsed);
        return { text: nestedText.trim(), images };
      }
    } catch {
      // Keep the original text when it is not a complete JSON response.
    }
  }

  const image = normalizeOllamaImageSource(trimmed, undefined, undefined, 1);
  if (image) return { text: "", images: [image] };
  return { text, images: [] };
}

async function streamOllamaPrompt(socket, payload, userId) {
  const id = payload.id ?? String(Date.now());
  const model = typeof payload.modelId === "string" ? payload.modelId.trim() : "";
  const prompt = typeof payload.prompt === "string" ? payload.prompt.trim() : "";
  const image = promptImageForOllama(payload.image);
  if (!model) throw new Error("Ollama model required.");
  if (!prompt && !image) throw new Error("Prompt required.");

  const { chatId, created } = await ensureUserChat(
    userId,
    typeof payload.chatId === "string" ? payload.chatId : null
  );
  if (created) {
    socket.send(JSON.stringify({ type: "chatCreated", id, chat: created }));
  }

  const controller = new AbortController();
  const onClose = () => controller.abort(new Error("Client disconnected."));
  const unregisterCancel = registerInferenceCancel((reason) => {
    if (!controller.signal.aborted) {
      controller.abort(new Error(typeof reason === "string" ? reason : "Inference cancelled."));
    }
  });
  socket.once?.("close", onClose);

  const userAt = new Date();
  let fullText = "";
  let fullThinking = "";
  let generatedImages = [];
  let finalStats = {};
  let responseCompleted = false;
  socket.send(JSON.stringify({ type: "start", id, chatId }));

  try {
    const requestBody = {
      model,
      prompt,
      stream: true,
    };
    if (payload.enableThinking === true) {
      requestBody.think = ollamaThinkValue(model);
    }
    if (image) {
      requestBody.images = [image.base64];
    }
    const seed = optionalClampedInteger(payload.seed, 0, 2 ** 31 - 1);
    if (typeof seed === "number") {
      requestBody.options = { seed };
    }

    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify(requestBody),
    });
    if (!res.ok || !res.body) {
      const message = await res.text().catch(() => "");
      throw new Error(`Ollama request failed (${res.status})${message ? `: ${message}` : "."}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newline = buffer.indexOf("\n");
      while (newline !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) {
          const parsed = JSON.parse(line);
          if (typeof parsed.error === "string") throw new Error(parsed.error);
          const responseText = ollamaResponseText(parsed);
          const thinkingText = ollamaResponseThinking(parsed);
          const images = collectOllamaImages(parsed);
          if (responseText.length > 0 || thinkingText.length > 0 || images.length > 0) {
            if (responseText.length > 0) fullText += responseText;
            if (thinkingText.length > 0) fullThinking += thinkingText;
            if (images.length > 0) generatedImages = mergeOllamaImages(generatedImages, images);
            socket.send(JSON.stringify({ type: "ollamaChunk", id, text: responseText, thinking: thinkingText, images }));
          }
          if (parsed.done === true) finalStats = parsed;
        }
        newline = buffer.indexOf("\n");
      }
    }

    const trailing = buffer.trim();
    if (trailing) {
      const parsed = JSON.parse(trailing);
      if (typeof parsed.error === "string") throw new Error(parsed.error);
      const responseText = ollamaResponseText(parsed);
      const thinkingText = ollamaResponseThinking(parsed);
      const images = collectOllamaImages(parsed);
      if (responseText.length > 0 || thinkingText.length > 0 || images.length > 0) {
        if (responseText.length > 0) fullText += responseText;
        if (thinkingText.length > 0) fullThinking += thinkingText;
        if (images.length > 0) generatedImages = mergeOllamaImages(generatedImages, images);
        socket.send(JSON.stringify({ type: "ollamaChunk", id, text: responseText, thinking: thinkingText, images }));
      }
      if (parsed.done === true) finalStats = parsed;
    }

    const extractedFromText = extractOllamaImagesFromText(fullText);
    if (extractedFromText.images.length > 0) {
      fullText = extractedFromText.text;
      generatedImages = mergeOllamaImages(generatedImages, extractedFromText.images);
    }

    socket.send(JSON.stringify({
      type: "ollamaDone",
      id,
      chatId,
      modelId: model,
      text: fullText,
      thinking: fullThinking || undefined,
      images: generatedImages,
      totalDuration: finalStats.total_duration ?? null,
      evalCount: finalStats.eval_count ?? null,
    }));
    responseCompleted = true;

    if (chatId) {
      await appendChatMessages(userId, chatId, [{
        id,
        role: "user",
        text: prompt,
        image: image?.attachment ?? null,
        provider: "ollama",
        createdAt: userAt,
      }, {
        id: `${id}:reply`,
        role: "assistant",
        text: fullText,
        images: generatedImages,
        provider: "ollama",
        modelId: model,
        tokenCount: finalStats.eval_count ?? null,
        createdAt: new Date(),
      }]);
    }
  } catch (err) {
    if (controller.signal.aborted) {
      const reason = controller.signal.reason;
      throw new Error(reason instanceof Error ? reason.message : "Inference cancelled.");
    }
    throw err;
  } finally {
    unregisterCancel();
    if (responseCompleted || controller.signal.aborted) {
      await unloadOllamaModel(model);
    }
    socket.off?.("close", onClose);
  }
}

async function streamLlamaPrompt(socket, payload, userId) {
  const id = payload.id ?? String(Date.now());
  const prompt = typeof payload.prompt === "string" ? payload.prompt.trim() : "";
  const modelSource = payload.modelSource === "huggingface" ? "huggingface" : "disk";
  const model = modelSource === "huggingface"
    ? { name: validateHuggingFaceModelName(payload.hfModel), source: "huggingface" }
    : { ...(await validateLlamaModelFile(payload.modelPath)), source: "disk" };
  if (!prompt) throw new Error("Prompt required.");

  const { chatId, created } = await ensureUserChat(
    userId,
    typeof payload.chatId === "string" ? payload.chatId : null
  );
  if (created) {
    socket.send(JSON.stringify({ type: "chatCreated", id, chat: created }));
  }

  const userAt = new Date();
  let fullText = "";
  let fullThinking = "";
  let stderr = "";
  let settled = false;
  let cancelReason = null;
  let outputLimitError = null;
  let parseBuffer = "";
  let llamaPhase = "before-thinking";
  socket.send(JSON.stringify({ type: "start", id, chatId }));
  const maxTokens = clampInteger(payload.maxTokens, 1, MAX_GENERATION_TOKENS, DEFAULT_MAX_TOKENS);

  const sendLlamaChunk = (chunk) => {
    if (!chunk) return;
    if (socket.readyState === 1) {
      socket.send(JSON.stringify({ type: "llamaChunk", id, ...chunk }));
    }
  };
  const appendLlamaText = (text) => {
    if (!text) return;
    const remaining = MAX_LLAMA_OUTPUT_CHARS - fullText.length;
    if (remaining <= 0) {
      outputLimitError = new Error("Llama.cpp output exceeded the 2,000,000 character safety limit.");
      try { child.kill("SIGTERM"); } catch {}
      return;
    }
    const safeText = text.length > remaining ? text.slice(0, remaining) : text;
    fullText = `${fullText}${safeText}`;
    sendLlamaChunk({ text: safeText });
    if (safeText.length < text.length) {
      outputLimitError = new Error("Llama.cpp output exceeded the 2,000,000 character safety limit.");
      try { child.kill("SIGTERM"); } catch {}
    }
  };
  const appendLlamaThinking = (thinking) => {
    if (!thinking) return;
    fullThinking = `${fullThinking}${thinking}`;
    sendLlamaChunk({ thinking });
  };
  const consumeLlamaOutput = (raw) => {
    parseBuffer += raw;
    while (parseBuffer) {
      if (llamaPhase === "before-thinking") {
        const start = parseBuffer.indexOf("[Start thinking]");
        if (start === -1) {
          parseBuffer = parseBuffer.slice(Math.max(0, parseBuffer.length - "[Start thinking]".length + 1));
          return;
        }
        parseBuffer = parseBuffer.slice(start + "[Start thinking]".length);
        llamaPhase = "thinking";
      }

      if (llamaPhase === "thinking") {
        const end = parseBuffer.indexOf("[End thinking]");
        if (end === -1) {
          const keep = "[End thinking]".length - 1;
          const emitLength = Math.max(0, parseBuffer.length - keep);
          if (emitLength > 0) {
            appendLlamaThinking(parseBuffer.slice(0, emitLength));
            parseBuffer = parseBuffer.slice(emitLength);
          }
          return;
        }
        appendLlamaThinking(parseBuffer.slice(0, end));
        parseBuffer = parseBuffer.slice(end + "[End thinking]".length);
        llamaPhase = "answer";
      }

      if (llamaPhase === "answer") {
        appendLlamaText(parseBuffer);
        parseBuffer = "";
      }
    }
  };
  const flushLlamaOutput = () => {
    if (llamaPhase === "thinking" && parseBuffer) {
      appendLlamaThinking(parseBuffer);
    } else if (llamaPhase === "answer" && parseBuffer) {
      appendLlamaText(parseBuffer);
    }
    parseBuffer = "";
  };

  const modelArgs = model.source === "huggingface"
    ? ["-hf", model.name]
    : ["-m", model.path];
  const child = spawn("llama-cli", [
    "--simple-io",
    "--single-turn",
    "--no-display-prompt",
    "--log-disable",
    "-n", String(maxTokens),
    ...modelArgs,
    "-p", prompt,
  ], {
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      LLAMA_LOG_COLORS: "off",
      NO_COLOR: "1",
    },
  });

  const unregisterCancel = registerInferenceCancel((reason) => {
    if (settled) return;
    cancelReason = typeof reason === "string" ? reason : "Inference cancelled.";
    try { child.kill("SIGTERM"); } catch {}
    setTimeout(() => {
      if (!settled) {
        try { child.kill("SIGKILL"); } catch {}
      }
    }, 2000);
  });
  const onClose = () => {
    try { child.kill("SIGTERM"); } catch {}
  };
  socket.once?.("close", onClose);

  try {
    await new Promise((resolve, reject) => {
      child.stdout.on("data", (chunk) => {
        if (outputLimitError) return;
        const text = chunk.toString();
        if (!text) return;
        consumeLlamaOutput(text);
      });
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.on("error", (err) => reject(err));
      child.on("close", (code, signal) => {
        settled = true;
        if (code === 0) resolve();
        else if (outputLimitError) reject(outputLimitError);
        else if (cancelReason) reject(new Error(cancelReason));
        else reject(new Error(
          signal
            ? `llama-cli stopped by ${signal}.`
            : `llama-cli exited with code ${code}${stderr ? `: ${stderr.trim()}` : "."}`
        ));
      });
    });
    flushLlamaOutput();
    if (outputLimitError) throw outputLimitError;

    socket.send(JSON.stringify({
      type: "llamaDone",
      id,
      chatId,
      modelName: model.name,
      text: fullText,
      thinking: fullThinking || undefined,
    }));

    if (chatId) {
      await appendChatMessages(userId, chatId, [{
        id,
        role: "user",
        text: prompt,
        provider: "llamacpp",
        createdAt: userAt,
      }, {
        id: `${id}:reply`,
        role: "assistant",
        text: fullText,
        thinking: fullThinking || undefined,
        provider: "llamacpp",
        modelId: model.name,
        createdAt: new Date(),
      }]);
    }
  } finally {
    settled = true;
    unregisterCancel();
    socket.off?.("close", onClose);
  }
}

// ─── Fastify ─────────────────────────────────────────────────────────
const fastify = Fastify({ logger: true });
await fastify.register(fastifyWebsocket);
await fastify.register(fastifyStatic, {
  root: join(__dirname, "client", "dist", "chat-client", "browser"),
  prefix: "/",
  decorateReply: false,
});

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
  };
});

fastify.post("/api/auth/register", async (_request, reply) => {
  reply.code(403);
  return { error: "Registration is disabled. Ask an administrator for an invite." };
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
  const valid = user ? await verifyPassword(password, user.passwordHash) : false;
  if (!valid) {
    reply.code(401);
    return { error: "Invalid username or password." };
  }

  stmt.updateUserLastLogin.run(new Date().toISOString(), user.id);
  const sessionId = createSessionForUser(user.id);
  setSessionCookie(reply, request, sessionId);
  return { user: { id: user.id, username: user.username } };
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
      merged.push({ id: row.id, downloads: 0, likes: 0, updatedAt: null, saved: true });
    }
  }
  const savedOrder = new Map(savedRows.map((r, i) => [r.id, i]));
  merged.sort((a, b) => {
    const aSaved = a.saved ? savedOrder.get(a.id) ?? Infinity : Infinity;
    const bSaved = b.saved ? savedOrder.get(b.id) ?? Infinity : Infinity;
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
      startedAt: d.startedAt instanceof Date ? d.startedAt.toISOString() : d.startedAt,
      title: d.title ?? null,
    })),
  };
}

async function rpcOpenChat({ chatId }, { userId }) {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  if (typeof userId !== "number") throw new Error("Unauthorized.");
  let id;
  try { id = new ObjectId(chatId); } catch { throw new Error("Invalid chat id."); }
  const doc = await chatsCol.findOne({ _id: id, userId });
  if (!doc) throw new Error("Chat not found.");
  return {
    id: doc._id.toString(),
    startedAt: doc.startedAt instanceof Date ? doc.startedAt.toISOString() : doc.startedAt,
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
  try { id = new ObjectId(chatId); } catch { throw new Error("Invalid chat id."); }
  await chatsCol.deleteOne({ _id: id, userId });
  return { ok: true };
}

async function rpcDeleteMessage({ chatId, messageId }, { userId }) {
  if (!chatsCol) throw new Error("Chat storage unavailable.");
  if (typeof userId !== "number") throw new Error("Unauthorized.");
  if (typeof messageId !== "string" || !messageId) throw new Error("Invalid message id.");
  let id;
  try { id = new ObjectId(chatId); } catch { throw new Error("Invalid chat id."); }
  await chatsCol.updateOne({ _id: id, userId }, { $pull: { messages: { id: messageId } } });
  return { ok: true };
}

const rpcHandlers = {
  listModels:       rpcListModels,
  listOllamaModels: rpcListOllamaModels,
  showOllamaModel:  rpcShowOllamaModel,
  pickLlamaModelFile: rpcPickLlamaModelFile,
  listFailedModels: rpcListFailedModels,
  listChats:        rpcListChats,
  openChat:         rpcOpenChat,
  deleteChat:       rpcDeleteChat,
  deleteMessage:    rpcDeleteMessage,
};

// ─── WebSocket endpoint ───────────────────────────────────────────────
fastify.register(async (instance) => {
  instance.get("/ws", { websocket: true }, (socket, request) => {
    const session = readSessionUser(request);
    if (!session || typeof session.userId !== "number") {
      socket.send(JSON.stringify({ type: "error", error: "Authentication required." }));
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
      if (typeof payload?.type === "string" && payload.type in rpcHandlers && typeof payload.requestId === "string") {
        const requestId = payload.requestId;
        try {
          const data = await rpcHandlers[payload.type](payload, { userId: session.userId });
          socket.send(JSON.stringify({ type: "rpcResult", requestId, data }));
        } catch (err) {
          socket.send(JSON.stringify({
            type: "rpcResult",
            requestId,
            error: err instanceof Error ? err.message : String(err),
          }));
        }
        return;
      }

      // ── cancelInference ─────────────────────────────────────────
      if (payload?.type === "cancelInference") {
        cancelAllInference("Inference cancelled.");
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

      // ── ollamaPrompt ──────────────────────────────────────────────
      if (payload?.type === "ollamaPrompt") {
        const id = payload.id ?? String(Date.now());
        generationQueue.enqueue({
          id,
          socket,
          run: () => streamOllamaPrompt(socket, { ...payload, id }, session.userId),
        });
        return;
      }

      // ── llamaPrompt ───────────────────────────────────────────────
      if (payload?.type === "llamaPrompt") {
        const id = payload.id ?? String(Date.now());
        generationQueue.enqueue({
          id,
          socket,
          run: () => streamLlamaPrompt(socket, { ...payload, id }, session.userId),
        });
        return;
      }

      // ── prompt ───────────────────────────────────────────────────
      if (payload?.type === "prompt" && typeof payload.prompt === "string") {
        const id = payload.id ?? String(Date.now());
        generationQueue.enqueue({
          id,
          socket,
          run: async () => {
            let persistedImage = null;
            try {
              persistedImage = await persistPromptImage(payload.image);

              let chatId = typeof payload.chatId === "string" ? payload.chatId : null;
              if (chatsCol) {
                if (chatId) {
                  try {
                    const exists = await chatsCol.findOne(
                      { _id: new ObjectId(chatId), userId: session.userId },
                      { projection: { _id: 1 } }
                    );
                    if (!exists) chatId = null;
                  } catch { chatId = null; }
                }
                if (!chatId) {
                  const created = await createChat(session.userId);
                  chatId = created.id;
                  socket.send(JSON.stringify({ type: "chatCreated", id, chat: created }));
                }
              }

              socket.send(JSON.stringify({ type: "start", id, chatId }));
              await modelProcess.generate(socket, id, payload.prompt, {
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
                userId: session.userId,
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
          },
        });
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
