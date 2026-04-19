import { Injectable, signal, computed } from "@angular/core";

export type Role = "user" | "assistant";
export type InferenceMode = "mlx" | "ollama";

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  thinking?: string;
  provider?: InferenceMode;
  image?: ChatImageAttachment;
  images?: ChatImageAttachment[];
  pending?: boolean;
  queued?: boolean;
  queuePosition?: number;
  tokensPerSecond?: number;
  tokenCount?: number;
  modelId?: string;
}

export interface ChatImageAttachment {
  dataUrl: string;
  name: string;
  type: string;
  size: number;
}

export interface ChatGenerationOptions {
  maxTokens?: number;
  imageWidth?: number;
  imageHeight?: number;
  steps?: number;
  seed?: number;
}

export interface HFModel {
  id: string;
  downloads: number;
  likes: number;
  updatedAt: string | null;
  /** True when this model has been successfully loaded at least once on this machine. */
  saved?: boolean;
}

export interface FailedModel {
  id: string;
  error: string;
  failedAt: string;
}

export interface OllamaModel {
  id: string;
  name: string;
  modifiedAt: string | null;
  size: number;
  digest: string | null;
  details: Record<string, unknown> | null;
}

export interface OllamaModelDetails {
  id: string;
  capabilities: string[];
  details: Record<string, unknown> | null;
  modifiedAt: string | null;
}

export interface GpuUsage {
  gpu: number;
  memory: number;
}

export interface ChatSummary {
  id: string;
  startedAt: string;
  title: string | null;
}

type ServerEvent =
  | { type: "start"; id: string; chatId?: string | null }
  | { type: "queued"; id: string; position: number }
  | { type: "chatCreated"; id: string; chat: ChatSummary }
  | { type: "response"; id: string; chatId?: string | null; modelId?: string; text: string; images?: ChatImageAttachment[]; tokenCount: number; tokensPerSecond: number }
  | { type: "ollamaChunk"; id: string; text?: string; thinking?: string; images?: ChatImageAttachment[] }
  | { type: "ollamaDone"; id: string; chatId?: string | null; modelId: string; text: string; thinking?: string; images?: ChatImageAttachment[]; totalDuration?: number | null; evalCount?: number | null }
  | { type: "error"; id?: string; error: string }
  | { type: "modelLoading"; modelId: string | null }
  | { type: "modelReady";   modelId: string; isVLM?: boolean; canGenerateImages?: boolean }
  | { type: "modelError";   modelId?: string; error: string; failed?: FailedModel[] }
  | { type: "gpuUsage"; running: boolean; gpu: number | null; memory: number | null }
  | { type: "rpcResult"; requestId: string; data?: any; error?: string };

@Injectable({ providedIn: "root" })
export class ChatService {
  readonly messages      = signal<ChatMessage[]>([]);
  readonly connected     = signal(false);
  readonly busy          = signal(false);
  readonly inferenceMode = signal<InferenceMode>("mlx");
  readonly currentModel  = signal<string | null>(null);
  readonly pendingModel  = signal<string | null>(null);
  readonly modelLoading  = signal(false);
  readonly modelError    = signal<string | null>(null);
  readonly supportsVision = signal(false);
  readonly supportsImageGeneration = signal(false);

  readonly availableModels = signal<HFModel[]>([]);
  readonly modelsLoading   = signal(false);
  readonly modelsError     = signal<string | null>(null);

  readonly failedModels = signal<Map<string, string>>(new Map());

  readonly ollamaModels        = signal<OllamaModel[]>([]);
  readonly ollamaModelsLoading = signal(false);
  readonly ollamaModelsError   = signal<string | null>(null);
  readonly currentOllamaModel  = signal<string | null>(null);
  readonly ollamaCapabilities  = signal<string[]>([]);
  readonly ollamaCapabilitiesLoading = signal(false);
  readonly ollamaCapabilitiesError   = signal<string | null>(null);
  readonly ollamaUrl           = signal<string | null>(null);

  readonly chats         = signal<ChatSummary[]>([]);
  readonly currentChatId = signal<string | null>(null);
  readonly chatsLoading  = signal(false);
  readonly gpuUsage      = signal<GpuUsage | null>(null);
  readonly inferenceRunning = signal(false);

  readonly isFailedModel = computed(() => {
    const map = this.failedModels();
    return (id: string) => map.has(id);
  });

  readonly failedModelError = computed(() => {
    const map = this.failedModels();
    return (id: string) => map.get(id) ?? null;
  });

  private socket?: WebSocket;
  private reconnectTimer?: ReturnType<typeof setTimeout>;
  private shouldReconnect = true;
  private readonly outbox: string[] = [];
  private readonly pendingRpc = new Map<string, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  private ollamaDetailsRequestSeq = 0;

  connect(): void {
    if (
      this.socket &&
      (this.socket.readyState === WebSocket.OPEN ||
        this.socket.readyState === WebSocket.CONNECTING)
    ) return;
    this.shouldReconnect = true;

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.socket = ws;

    ws.addEventListener("open", () => {
      this.connected.set(true);
      while (this.outbox.length > 0) {
        const msg = this.outbox.shift();
        if (msg) ws.send(msg);
      }
    });
    ws.addEventListener("close", (event) => {
      if (this.socket === ws) this.socket = undefined;
      this.connected.set(false);
      this.busy.set(false);
      this.gpuUsage.set(null);
      this.inferenceRunning.set(false);
      // Fail any in-flight RPCs so callers can retry.
      for (const [, handlers] of this.pendingRpc) {
        handlers.reject(new Error("Connection closed."));
      }
      this.pendingRpc.clear();
      if (!this.shouldReconnect || event.code === 1000 || event.code === 4401) return;
      this.scheduleReconnect();
    });
    ws.addEventListener("error", () => this.connected.set(false));
    ws.addEventListener("message", (ev) => {
      try { this.handleEvent(JSON.parse(ev.data) as ServerEvent); } catch {}
    });
  }

  disconnect(clearState = false): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
    const ws = this.socket;
    this.socket = undefined;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close(1000, "logout");
    }
    this.connected.set(false);
    this.busy.set(false);
    this.gpuUsage.set(null);
    this.inferenceRunning.set(false);

    for (const [, handlers] of this.pendingRpc) {
      handlers.reject(new Error("Connection closed."));
    }
    this.pendingRpc.clear();
    this.outbox.length = 0;

    if (clearState) this.resetState();
  }

  // ── RPC over WebSocket ────────────────────────────────────────────
  private sendWs(obj: unknown): void {
    const json = JSON.stringify(obj);
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(json);
    } else {
      this.outbox.push(json);
    }
  }

  private rpc<T = any>(type: string, payload: Record<string, unknown> = {}): Promise<T> {
    const requestId = crypto.randomUUID();
    return new Promise<T>((resolve, reject) => {
      this.pendingRpc.set(requestId, { resolve, reject });
      this.sendWs({ type, requestId, ...payload });
    });
  }

  cancelInference(): void {
    this.sendWs({ type: "cancelInference" });
  }

  // ── WS-backed data loaders ────────────────────────────────────────
  async loadAvailableModels(refresh = false): Promise<void> {
    this.modelsLoading.set(true);
    this.modelsError.set(null);
    try {
      const body = await this.rpc<{ models: HFModel[] }>("listModels", { refresh });
      this.availableModels.set(body.models ?? []);
    } catch (err) {
      this.modelsError.set(err instanceof Error ? err.message : String(err));
    } finally {
      this.modelsLoading.set(false);
    }
  }

  async loadChats(): Promise<void> {
    this.chatsLoading.set(true);
    try {
      const body = await this.rpc<{ chats: ChatSummary[] }>("listChats");
      this.chats.set(body.chats ?? []);
    } catch {
      this.chats.set([]);
    } finally {
      this.chatsLoading.set(false);
    }
  }

  async openChat(id: string): Promise<void> {
    try {
      const body = await this.rpc<{ id: string; messages: ChatMessage[] }>("openChat", { chatId: id });
      this.currentChatId.set(body.id);
      this.messages.set((body.messages ?? []).map((m) => {
        if (m.role === "assistant" && m.provider === "ollama") {
          const extracted = this.extractOllamaImages(m.text ?? "");
          return {
            ...m,
            text: extracted.text,
            images: [...(m.images ?? []), ...extracted.images],
            pending: false,
          };
        }
        return { ...m, pending: false };
      }));
      this.busy.set(false);
    } catch {}
  }

  newChat(): void {
    this.currentChatId.set(null);
    this.messages.set([]);
    this.busy.set(false);
  }

  async deleteChat(id: string): Promise<void> {
    try { await this.rpc("deleteChat", { chatId: id }); } catch {}
    this.chats.update((list) => list.filter((c) => c.id !== id));
    if (this.currentChatId() === id) this.newChat();
  }

  async deleteMessage(messageId: string): Promise<void> {
    const chatId = this.currentChatId();
    this.messages.update((list) => list.filter((m) => m.id !== messageId));
    if (!chatId) return;
    try { await this.rpc("deleteMessage", { chatId, messageId }); } catch {}
  }

  async loadFailedModels(): Promise<void> {
    try {
      const body = await this.rpc<{ failed: FailedModel[] }>("listFailedModels");
      this.applyFailedList(body.failed ?? []);
    } catch {}
  }

  async loadOllamaModels(): Promise<void> {
    this.ollamaModelsLoading.set(true);
    this.ollamaModelsError.set(null);
    try {
      const body = await this.rpc<{ models: OllamaModel[]; url?: string }>("listOllamaModels");
      const models = body.models ?? [];
      this.ollamaModels.set(models);
      this.ollamaUrl.set(body.url ?? null);
      if (!this.currentOllamaModel() && models.length > 0) {
        this.selectOllamaModel(models[0].id);
      } else if (this.currentOllamaModel()) {
        void this.loadOllamaModelDetails(this.currentOllamaModel()!);
      }
    } catch (err) {
      this.ollamaModels.set([]);
      this.ollamaModelsError.set(err instanceof Error ? err.message : String(err));
    } finally {
      this.ollamaModelsLoading.set(false);
    }
  }

  // ── WS commands (fire-and-forget, existing events) ────────────────
  setInferenceMode(mode: InferenceMode): void {
    if (this.inferenceMode() === mode) return;
    this.inferenceMode.set(mode);
    if (mode === "ollama" && this.ollamaModels().length === 0) {
      void this.loadOllamaModels();
    }
  }

  selectModel(modelId: string): void {
    const id = modelId.trim();
    if (!id || this.modelLoading()) return;
    if (id === this.currentModel()) return;
    this.pendingModel.set(id);
    this.modelLoading.set(true);
    this.modelError.set(null);
    this.sendWs({ type: "selectModel", modelId: id });
  }

  selectOllamaModel(modelId: string): void {
    const id = modelId.trim();
    if (!id) return;
    this.currentOllamaModel.set(id);
    void this.loadOllamaModelDetails(id);
    this.ollamaModels.update((models) => {
      if (models.some((m) => m.id === id)) return models;
      return [{ id, name: id, modifiedAt: null, size: 0, digest: null, details: null }, ...models];
    });
  }

  private async loadOllamaModelDetails(modelId: string): Promise<void> {
    const id = modelId.trim();
    if (!id) return;
    const requestSeq = ++this.ollamaDetailsRequestSeq;
    this.ollamaCapabilitiesError.set(null);

    this.ollamaCapabilities.set([]);
    this.ollamaCapabilitiesLoading.set(true);
    try {
      const body = await this.rpc<{ model: OllamaModelDetails }>("showOllamaModel", { modelId: id });
      if (requestSeq !== this.ollamaDetailsRequestSeq || this.currentOllamaModel() !== id) return;
      const details: OllamaModelDetails = {
        id,
        capabilities: body.model?.capabilities ?? [],
        details: body.model?.details ?? null,
        modifiedAt: body.model?.modifiedAt ?? null,
      };
      this.ollamaCapabilities.set(details.capabilities);
    } catch (err) {
      if (requestSeq !== this.ollamaDetailsRequestSeq || this.currentOllamaModel() !== id) return;
      this.ollamaCapabilities.set([]);
      this.ollamaCapabilitiesError.set(err instanceof Error ? err.message : String(err));
    } finally {
      if (requestSeq === this.ollamaDetailsRequestSeq) {
        this.ollamaCapabilitiesLoading.set(false);
      }
    }
  }

  send(prompt: string, image?: ChatImageAttachment, options: ChatGenerationOptions = {}): void {
    if (this.inferenceMode() === "ollama") {
      this.sendOllama(prompt, image, options);
      return;
    }

    const text = prompt.trim() || (image ? "Describe this image." : "");
    if ((!text && !image) || this.busy() || this.modelLoading() || !this.currentModel()) return;

    const id = crypto.randomUUID();
    this.messages.update((list) => [
      ...list,
      { id, role: "user", text, image },
      { id: `${id}:reply`, role: "assistant", text: "", pending: true },
    ]);
    this.busy.set(true);
    this.sendWs({
      type: "prompt",
      id,
      chatId: this.currentChatId(),
      prompt: text,
      image,
      maxTokens: options.maxTokens,
      imageWidth: options.imageWidth,
      imageHeight: options.imageHeight,
      steps: options.steps,
      seed: options.seed,
    });
  }

  private sendOllama(
    prompt: string,
    image?: ChatImageAttachment,
    options: ChatGenerationOptions = {}
  ): void {
    const text = prompt.trim() || (image ? "Describe this image." : "");
    const modelId = this.currentOllamaModel();
    const supportsVision = this.ollamaCapabilities().some(
      (capability) => capability.replace(/[-_\s]+/g, "").toLowerCase() === "vision"
    );
    const supportsThinking = this.ollamaCapabilities().some(
      (capability) => capability.replace(/[-_\s]+/g, "").toLowerCase() === "thinking"
    );
    if ((!text && !image) || this.busy() || !modelId || (image && !supportsVision)) return;

    const id = crypto.randomUUID();
    this.messages.update((list) => [
      ...list,
      { id, role: "user", text, image, provider: "ollama" },
      { id: `${id}:reply`, role: "assistant", text: "", pending: true, modelId, provider: "ollama" },
    ]);
    this.busy.set(true);
    this.sendWs({
      type: "ollamaPrompt",
      id,
      chatId: this.currentChatId(),
      modelId,
      prompt: text,
      image,
      imageWidth: options.imageWidth,
      imageHeight: options.imageHeight,
      steps: options.steps,
      seed: options.seed,
      enableThinking: supportsThinking,
    });
  }

  // ── Event handling ────────────────────────────────────────────────
  private handleEvent(event: ServerEvent): void {
    switch (event.type) {
      case "rpcResult": {
        const handlers = this.pendingRpc.get(event.requestId);
        if (!handlers) return;
        this.pendingRpc.delete(event.requestId);
        if (event.error) handlers.reject(new Error(event.error));
        else handlers.resolve(event.data ?? {});
        return;
      }

      case "gpuUsage":
        this.inferenceRunning.set(event.running);
        this.gpuUsage.set(
          event.running && typeof event.gpu === "number" && typeof event.memory === "number"
            ? { gpu: event.gpu, memory: event.memory }
            : null
        );
        if (!event.running) this.busy.set(false);
        return;

      case "start":
        if (event.chatId && !this.currentChatId()) this.currentChatId.set(event.chatId);
        this.messages.update((list) =>
          list.map((msg) =>
            msg.id === `${event.id}:reply`
              ? { ...msg, queued: false, queuePosition: undefined }
              : msg
          )
        );
        return;

      case "queued":
        this.messages.update((list) =>
          list.map((msg) =>
            msg.id === `${event.id}:reply`
              ? { ...msg, pending: true, queued: true, queuePosition: event.position }
              : msg
          )
        );
        return;

      case "chatCreated":
        this.currentChatId.set(event.chat.id);
        this.chats.update((list) => [event.chat, ...list.filter((c) => c.id !== event.chat.id)]);
        return;

      case "response":
        this.messages.update((list) =>
          list.map((msg) =>
            msg.id === `${event.id}:reply`
              ? { ...msg, text: event.text, images: event.images ?? [], pending: false,
                  queued: false, queuePosition: undefined,
                  tokenCount: event.tokenCount, tokensPerSecond: event.tokensPerSecond,
                  modelId: event.modelId }
              : msg
          )
        );
        this.busy.set(false);
        {
          const cid = this.currentChatId();
          if (cid) {
            const current = this.chats().find((c) => c.id === cid);
            if (!current || current.title === null) void this.loadChats();
          }
        }
        return;

      case "ollamaChunk":
        this.messages.update((list) =>
          list.map((msg) =>
            msg.id === `${event.id}:reply`
              ? this.withExtractedOllamaImages({
                  ...msg,
                  text: `${msg.text ?? ""}${event.text ?? ""}`,
                  thinking: `${msg.thinking ?? ""}${event.thinking ?? ""}` || undefined,
                  images: this.mergeImages(msg.images ?? [], event.images ?? []),
                  pending: true,
                  queued: false,
                  queuePosition: undefined,
                  provider: "ollama",
                })
              : msg
          )
        );
        return;

      case "ollamaDone":
        this.messages.update((list) =>
          list.map((msg) =>
            msg.id === `${event.id}:reply`
              ? this.withExtractedOllamaImages({
                  ...msg,
                  text: event.text,
                  thinking: event.thinking ?? msg.thinking,
                  images: this.mergeImages(msg.images ?? [], event.images ?? []),
                  pending: false,
                  queued: false,
                  queuePosition: undefined,
                  provider: "ollama",
                  modelId: event.modelId,
                  tokenCount: event.evalCount ?? undefined,
                })
              : msg
          )
        );
        this.busy.set(false);
        {
          const cid = this.currentChatId();
          if (cid) {
            const current = this.chats().find((c) => c.id === cid);
            if (!current || current.title === null) void this.loadChats();
          }
        }
        return;

      case "error": {
        const replyId = event.id ? `${event.id}:reply` : undefined;
        this.messages.update((list) => {
          if (replyId && list.some((m) => m.id === replyId)) {
            return list.map((msg) =>
              msg.id === replyId
                ? { ...msg, text: `⚠ ${event.error}`, pending: false, queued: false, queuePosition: undefined }
                : msg
            );
          }
          return [...list, { id: crypto.randomUUID(), role: "assistant", text: `⚠ ${event.error}` }];
        });
        this.busy.set(false);
        return;
      }

      case "modelLoading":
        this.modelLoading.set(true);
        this.pendingModel.set(event.modelId);
        this.modelError.set(null);
        this.supportsVision.set(false);
        this.supportsImageGeneration.set(false);
        return;

      case "modelReady":
        this.currentModel.set(event.modelId);
        this.pendingModel.set(null);
        this.modelLoading.set(false);
        this.modelError.set(null);
        this.supportsVision.set(event.isVLM === true);
        this.supportsImageGeneration.set(event.canGenerateImages === true);
        this.failedModels.update((m) => { const n = new Map(m); n.delete(event.modelId); return n; });
        return;

      case "modelError":
        this.modelLoading.set(false);
        this.pendingModel.set(null);
        this.modelError.set(event.error);
        this.supportsVision.set(false);
        this.supportsImageGeneration.set(false);
        if (event.failed) this.applyFailedList(event.failed);
        return;
    }
  }

  private applyFailedList(list: FailedModel[]): void {
    const map = new Map<string, string>();
    for (const f of list) map.set(f.id, f.error);
    this.failedModels.set(map);
  }

  private withExtractedOllamaImages(message: ChatMessage): ChatMessage {
    const extracted = this.extractOllamaImages(message.text ?? "");
    const images = this.mergeImages(message.images ?? [], extracted.images);
    return {
      ...message,
      text: extracted.text,
      images: images.length > 0 ? images : undefined,
    };
  }

  private extractOllamaImages(text: string): { text: string; images: ChatImageAttachment[] } {
    const images: ChatImageAttachment[] = [];
    let cleaned = text;

    cleaned = cleaned.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)/g, (match, alt, src) => {
      this.addDetectedImage(images, src, alt || undefined);
      return "";
    });

    cleaned = cleaned.replace(/<img\b[^>]*>/gi, (match) => {
      const src = /\bsrc=["']([^"']+)["']/i.exec(match)?.[1];
      const alt = /\balt=["']([^"']*)["']/i.exec(match)?.[1];
      if (!src) return match;
      this.addDetectedImage(images, src, alt || undefined);
      return "";
    });

    cleaned = cleaned.replace(/data:image\/(?:png|jpeg|jpg|gif|webp|svg\+xml);base64,[A-Za-z0-9+/=]+/gi, (src) => {
      this.addDetectedImage(images, src);
      return "";
    });

    cleaned = cleaned.replace(/https?:\/\/[^\s<>"')]+\.(?:png|jpe?g|gif|webp|svg)(?:\?[^\s<>"')]*)?/gi, (src) => {
      this.addDetectedImage(images, src);
      return "";
    });

    return {
      text: cleaned.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim(),
      images,
    };
  }

  private addDetectedImage(images: ChatImageAttachment[], rawSrc: string, alt?: string): void {
    const src = rawSrc.trim().replace(/^<|>$/g, "");
    if (!this.isDisplayableImageSource(src)) return;
    if (images.some((image) => image.dataUrl === src)) return;

    images.push({
      dataUrl: src,
      name: alt?.trim() || this.imageNameFromSource(src),
      type: this.imageTypeFromSource(src),
      size: 0,
    });
  }

  private mergeImages(
    existing: ChatImageAttachment[],
    detected: ChatImageAttachment[]
  ): ChatImageAttachment[] {
    const bySource = new Map<string, ChatImageAttachment>();
    for (const image of [...existing, ...detected]) {
      if (!bySource.has(image.dataUrl)) bySource.set(image.dataUrl, image);
    }
    return [...bySource.values()];
  }

  private isDisplayableImageSource(src: string): boolean {
    return /^data:image\//i.test(src) || /^https?:\/\//i.test(src);
  }

  private imageNameFromSource(src: string): string {
    if (/^data:image\//i.test(src)) return "Ollama image";
    try {
      const url = new URL(src);
      const filename = url.pathname.split("/").filter(Boolean).pop();
      return filename ? decodeURIComponent(filename) : "Ollama image";
    } catch {
      return "Ollama image";
    }
  }

  private imageTypeFromSource(src: string): string {
    const dataMatch = /^data:(image\/[^;]+);/i.exec(src);
    if (dataMatch) return dataMatch[1];

    const clean = src.split("?")[0].toLowerCase();
    if (clean.endsWith(".jpg") || clean.endsWith(".jpeg")) return "image/jpeg";
    if (clean.endsWith(".png")) return "image/png";
    if (clean.endsWith(".gif")) return "image/gif";
    if (clean.endsWith(".webp")) return "image/webp";
    if (clean.endsWith(".svg")) return "image/svg+xml";
    return "image";
  }

  private resetState(): void {
    this.messages.set([]);
    this.currentChatId.set(null);
    this.chats.set([]);
    this.chatsLoading.set(false);
    this.currentModel.set(null);
    this.modelLoading.set(false);
    this.pendingModel.set(null);
    this.modelError.set(null);
    this.supportsVision.set(false);
    this.supportsImageGeneration.set(false);
    this.availableModels.set([]);
    this.modelsLoading.set(false);
    this.modelsError.set(null);
    this.failedModels.set(new Map());
    this.inferenceMode.set("mlx");
    this.ollamaModels.set([]);
    this.ollamaModelsLoading.set(false);
    this.ollamaModelsError.set(null);
    this.currentOllamaModel.set(null);
    this.ollamaCapabilities.set([]);
    this.ollamaCapabilitiesLoading.set(false);
    this.ollamaCapabilitiesError.set(null);
    this.ollamaDetailsRequestSeq += 1;
    this.ollamaUrl.set(null);
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => { this.reconnectTimer = undefined; this.connect(); }, 1500);
  }
}
