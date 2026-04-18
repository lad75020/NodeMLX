import { Injectable, signal, computed } from "@angular/core";

export type Role = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  image?: ChatImageAttachment;
  images?: ChatImageAttachment[];
  pending?: boolean;
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

export interface ChatSummary {
  id: string;
  startedAt: string;
  title: string | null;
}

type ServerEvent =
  | { type: "start"; id: string; chatId?: string | null }
  | { type: "chatCreated"; id: string; chat: ChatSummary }
  | { type: "response"; id: string; chatId?: string | null; modelId?: string; text: string; images?: ChatImageAttachment[]; tokenCount: number; tokensPerSecond: number }
  | { type: "error"; id?: string; error: string }
  | { type: "modelLoading"; modelId: string | null }
  | { type: "modelReady";   modelId: string; isVLM?: boolean; canGenerateImages?: boolean }
  | { type: "modelError";   modelId?: string; error: string; failed?: FailedModel[] };

@Injectable({ providedIn: "root" })
export class ChatService {
  readonly messages      = signal<ChatMessage[]>([]);
  readonly connected     = signal(false);
  readonly busy          = signal(false);
  readonly currentModel  = signal<string | null>(null);
  readonly pendingModel  = signal<string | null>(null);
  readonly modelLoading  = signal(false);
  readonly modelError    = signal<string | null>(null);
  /** True when the active backend can run image-conditioned inference for this model. */
  readonly supportsVision = signal(false);
  /** True when the active backend returns generated images from text prompts. */
  readonly supportsImageGeneration = signal(false);

  readonly availableModels = signal<HFModel[]>([]);
  readonly modelsLoading   = signal(false);
  readonly modelsError     = signal<string | null>(null);

  /** Map of model-id → error string for all known-incompatible models. */
  readonly failedModels = signal<Map<string, string>>(new Map());

  readonly chats         = signal<ChatSummary[]>([]);
  readonly currentChatId = signal<string | null>(null);
  readonly chatsLoading  = signal(false);

  /** Quick O(1) check used in the model selector. */
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

  connect(): void {
    if (
      this.socket &&
      (this.socket.readyState === WebSocket.OPEN ||
        this.socket.readyState === WebSocket.CONNECTING)
    ) return;

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    this.socket = ws;

    ws.addEventListener("open",    () => this.connected.set(true));
    ws.addEventListener("close",   () => { this.connected.set(false); this.busy.set(false); this.scheduleReconnect(); });
    ws.addEventListener("error",   () => this.connected.set(false));
    ws.addEventListener("message", (ev) => {
      try { this.handleEvent(JSON.parse(ev.data) as ServerEvent); } catch {}
    });
  }

  // ── REST calls ────────────────────────────────────────────────────
  async loadAvailableModels(refresh = false): Promise<void> {
    this.modelsLoading.set(true);
    this.modelsError.set(null);
    try {
      const res = await fetch(`/api/models${refresh ? "?refresh=1" : ""}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json() as { models: HFModel[] };
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
      const res = await fetch("/api/chats");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json() as { chats: ChatSummary[] };
      this.chats.set(body.chats ?? []);
    } catch {
      this.chats.set([]);
    } finally {
      this.chatsLoading.set(false);
    }
  }

  async openChat(id: string): Promise<void> {
    try {
      const res = await fetch(`/api/chats/${id}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json() as { id: string; messages: ChatMessage[] };
      this.currentChatId.set(body.id);
      this.messages.set((body.messages ?? []).map((m) => ({ ...m, pending: false })));
      this.busy.set(false);
    } catch {}
  }

  newChat(): void {
    this.currentChatId.set(null);
    this.messages.set([]);
    this.busy.set(false);
  }

  async deleteMessage(messageId: string): Promise<void> {
    const chatId = this.currentChatId();
    this.messages.update((list) => list.filter((m) => m.id !== messageId));
    if (!chatId) return;
    try {
      await fetch(`/api/chats/${chatId}/messages/${encodeURIComponent(messageId)}`, {
        method: "DELETE",
      });
    } catch {}
  }

  async deleteChat(id: string): Promise<void> {
    try {
      await fetch(`/api/chats/${id}`, { method: "DELETE" });
    } catch {}
    this.chats.update((list) => list.filter((c) => c.id !== id));
    if (this.currentChatId() === id) this.newChat();
  }

  async loadFailedModels(): Promise<void> {
    try {
      const res  = await fetch("/api/models/failed");
      if (!res.ok) return;
      const body = await res.json() as { failed: FailedModel[] };
      this.applyFailedList(body.failed);
    } catch {}
  }

  // ── WS commands ───────────────────────────────────────────────────
  selectModel(modelId: string): void {
    const id = modelId.trim();
    if (!id || this.modelLoading() || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    if (id === this.currentModel()) return;
    this.pendingModel.set(id);
    this.modelLoading.set(true);
    this.modelError.set(null);
    this.socket.send(JSON.stringify({ type: "selectModel", modelId: id }));
  }

  send(prompt: string, image?: ChatImageAttachment, options: ChatGenerationOptions = {}): void {
    const text = prompt.trim() || (image ? "Describe this image." : "");
    if ((!text && !image) || this.busy() || this.modelLoading() || !this.currentModel() ||
        !this.socket || this.socket.readyState !== WebSocket.OPEN) return;

    const id = crypto.randomUUID();
    this.messages.update((list) => [
      ...list,
      { id, role: "user", text, image },
      { id: `${id}:reply`, role: "assistant", text: "", pending: true },
    ]);
    this.busy.set(true);
    this.socket.send(JSON.stringify({
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
    }));
  }

  // ── Event handling ────────────────────────────────────────────────
  private handleEvent(event: ServerEvent): void {
    switch (event.type) {
      case "start":
        if (event.chatId && !this.currentChatId()) this.currentChatId.set(event.chatId);
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

      case "error": {
        const replyId = event.id ? `${event.id}:reply` : undefined;
        this.messages.update((list) => {
          if (replyId && list.some((m) => m.id === replyId)) {
            return list.map((msg) =>
              msg.id === replyId ? { ...msg, text: `⚠ ${event.error}`, pending: false } : msg
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
        // Remove from failed set if it loaded OK this time.
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

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => { this.reconnectTimer = undefined; this.connect(); }, 1500);
  }
}
