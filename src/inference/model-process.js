import { fork } from "node:child_process";

// Owns the model worker lifecycle and its model/generation state.
export class ModelProcess {
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

  constructor({
    workerPath,
    broadcast,
    stmt,
    cleanupPromptImage,
    appendChatMessages,
  }) {
    this.workerPath = workerPath;
    this.broadcast = broadcast;
    this.stmt = stmt;
    this.cleanupPromptImage = cleanupPromptImage;
    this.appendChatMessages = appendChatMessages;
    this.#spawnWorker();
  }

  // ── Public accessors ──────────────────────────────────────────────
  get currentModelId() {
    return this.#currentModelId;
  }
  get loading() {
    return this.#loading;
  }

  /** Returns the WS message a newly-connected client should receive. */
  greetingMessage() {
    if (this.#loading)
      return { type: "modelLoading", modelId: this.#pendingModelId };
    if (this.#currentModelId)
      return {
        type: "modelReady",
        modelId: this.#currentModelId,
        isVLM: this.#isVLM,
        canGenerateImages: this.#canGenerateImages,
      };
    return { type: "modelError", error: this.#lastError ?? "No model loaded." };
  }

  // ── Commands ──────────────────────────────────────────────────────
  load(modelId) {
    if (!modelId || typeof modelId !== "string")
      throw new Error("modelId required.");
    if (this.#loading) throw new Error("Another model is already loading.");
    if (this.#currentModelId === modelId) {
      this.broadcast({
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
    this.broadcast({ type: "modelLoading", modelId });
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
      throw new Error(
        "The active node-mlx backend does not support image input for this model.",
      );
    }
    return new Promise((resolve) => {
      this.#pending.set(id, { socket, imagePath, resolve, ...context });
      this.#worker.send({ type: "generate", id, prompt, options, imagePath });
    });
  }

  cancelGeneration(reason = "Inference cancelled.") {
    if (this.#pending.size === 0) return false;

    for (const [id, pending] of this.#pending) {
      this.cleanupPromptImage(pending.imagePath);
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
      worker.once("exit", () => {
        exited = true;
      });
      worker.kill("SIGTERM");
      setTimeout(() => {
        if (!exited) {
          try {
            worker.kill("SIGKILL");
          } catch {}
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
    this.broadcast({
      type: "modelError",
      error: reason,
      failed: this.stmt.allFailed.all(),
    });

    setTimeout(() => this.#spawnWorker(), 500);
    return true;
  }

  // ── Worker lifecycle ──────────────────────────────────────────────
  #spawnWorker() {
    const w = fork(this.workerPath);
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
        this.stmt.deleteFailed.run(msg.modelId);
        // Remember this model so it appears in the selector on future visits.
        this.stmt.upsertSaved.run(msg.modelId, this.#isVLM ? 1 : 0);
        this.broadcast({
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
        this.stmt.upsertFailed.run(msg.modelId, msg.error);
        this.broadcast({
          type: "modelError",
          modelId: msg.modelId,
          error: msg.error,
          failed: this.stmt.allFailed.all(),
        });
        break;
      }

      case "generateResult": {
        const pending = this.#pending.get(msg.id);
        this.#pending.delete(msg.id);
        this.cleanupPromptImage(pending?.imagePath);
        const socket = pending?.socket;
        if (socket?.readyState === 1) {
          socket.send(
            JSON.stringify({
              type: "response",
              id: msg.id,
              chatId: pending?.chatId ?? null,
              modelId: msg.modelId,
              text: msg.text,
              images: msg.images ?? [],
              tokenCount: msg.tokenCount,
              tokensPerSecond: msg.tokensPerSecond,
            }),
          );
        }
        if (pending?.chatId && typeof pending.userId === "number") {
          const now = new Date();
          const entries = [
            {
              id: msg.id,
              role: "user",
              text: pending.userText ?? "",
              image: pending.userImage ?? null,
              createdAt: pending.userAt ?? now,
            },
            {
              id: `${msg.id}:reply`,
              role: "assistant",
              text: msg.text ?? "",
              images: msg.images ?? [],
              modelId: msg.modelId ?? null,
              tokenCount: msg.tokenCount ?? null,
              tokensPerSecond: msg.tokensPerSecond ?? null,
              createdAt: now,
            },
          ];
          this.appendChatMessages(
            pending.userId,
            pending.chatId,
            entries,
          ).catch((err) => console.error("Mongo append failed:", err.message));
        }
        pending?.resolve?.();
        break;
      }

      case "generateError": {
        const pending = this.#pending.get(msg.id);
        this.#pending.delete(msg.id);
        this.cleanupPromptImage(pending?.imagePath);
        const socket = pending?.socket;
        if (socket?.readyState === 1) {
          socket.send(
            JSON.stringify({ type: "error", id: msg.id, error: msg.error }),
          );
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
      this.stmt.upsertFailed.run(modelId, error);
      this.broadcast({
        type: "modelError",
        modelId,
        error,
        failed: this.stmt.allFailed.all(),
      });
      this.#canGenerateImages = false;
    }

    // Fail any in-flight generate requests.
    for (const [id, pending] of this.#pending) {
      this.cleanupPromptImage(pending.imagePath);
      const socket = pending.socket;
      if (socket?.readyState === 1) {
        socket.send(
          JSON.stringify({
            type: "error",
            id,
            error: "Model process crashed.",
          }),
        );
      }
      pending.resolve?.();
    }
    this.#pending.clear();

    // Respawn so the server stays usable.
    setTimeout(() => this.#spawnWorker(), 500);
  }
}
