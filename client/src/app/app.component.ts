import {
  AfterViewChecked,
  Component,
  ElementRef,
  effect,
  OnInit,
  ViewChild,
  inject,
} from "@angular/core";
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { ChatImageAttachment, ChatMessage, ChatService } from "./chat.service";
import { ModelSelectorComponent } from "./model-selector.component";
import { OllamaModelSelectorComponent } from "./ollama-model-selector.component";
import { InferenceModeToggleComponent } from "./inference-mode-toggle.component";
import { AuthService } from "./auth.service";

const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const ALLOWED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);
const DEFAULT_MAX_TOKENS = 4096;
const MAX_GENERATION_TOKENS = 32768;
const MAX_IMAGE_DIMENSION = 2048;
const MAX_IMAGE_STEPS = 150;
const MAX_IMAGE_SEED = 2 ** 31 - 1;

interface ImageGenerationPreset {
  width: number;
  height: number;
  steps: number;
}

interface FormattedTextSegment {
  text: string;
  bold: boolean;
}

type FormattedBlock =
  | { kind: "chapter"; segments: FormattedTextSegment[] }
  | { kind: "paragraph"; segments: FormattedTextSegment[] }
  | { kind: "list"; items: FormattedTextSegment[][] }
  | { kind: "table"; rows: FormattedTextSegment[][][] }
  | { kind: "code"; code: string; language: string | null };

@Component({
  selector: "app-root",
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ModelSelectorComponent,
    OllamaModelSelectorComponent,
    InferenceModeToggleComponent,
  ],
  templateUrl: "./app.component.html",
  styleUrl: "./app.component.scss",
})
export class AppComponent implements OnInit, AfterViewChecked {
  protected readonly auth = inject(AuthService);
  protected readonly chat = inject(ChatService);
  protected authUsername = "";
  protected authPassword = "";
  protected prompt = "";
  protected selectedImage: ChatImageAttachment | null = null;
  protected imageError: string | null = null;
  protected maxTokens = DEFAULT_MAX_TOKENS;
  protected imageWidth = 768;
  protected imageHeight = 768;
  protected imageSteps = 30;
  protected imageSeed: number | null = null;

  private lastImagePresetModel: string | null = null;
  private formattedTextCache = new Map<string, FormattedBlock[]>();
  private readonly modeEffect = effect(() => {
    const mode = this.chat.inferenceMode();
    const canUseImage = mode === "mlx"
      ? this.chat.supportsVision()
      : mode === "ollama" && (this.chat.ollamaCapabilitiesLoading() || this.hasOllamaCapability("vision"));
    if (this.selectedImage && !canUseImage) {
      this.clearImage();
    }
  });
  private readonly imagePresetEffect = effect(() => {
    const modelId = this.chat.inferenceMode() === "mlx"
      ? this.chat.currentModel()
      : this.chat.currentOllamaModel();
    if (!modelId || !this.usesImageGenerationControls()) {
      this.lastImagePresetModel = null;
      return;
    }
    if (modelId === this.lastImagePresetModel) return;
    this.applyImagePreset(modelId);
    this.lastImagePresetModel = modelId;
  });

  @ViewChild("scrollAnchor") private scrollAnchor?: ElementRef<HTMLDivElement>;
  private lastLength = 0;

  ngOnInit(): void {
    void this.initialize();
  }

  protected async submitAuth(): Promise<void> {
    if (this.auth.loading()) return;
    const username = this.authUsername.trim();
    const password = this.authPassword;
    if (!username || !password) {
      this.auth.error.set("Username and password are required.");
      return;
    }

    const ok = await this.auth.login(username, password);
    if (!ok) return;
    this.authPassword = "";
    await this.connectChat();
  }

  protected async logout(): Promise<void> {
    if (this.auth.loading()) return;
    this.chat.disconnect(true);
    await this.auth.logout();
    this.authPassword = "";
  }

  private async initialize(): Promise<void> {
    await this.auth.restoreSession();
    if (this.auth.isAuthenticated()) {
      await this.connectChat();
    }
  }

  private async connectChat(): Promise<void> {
    this.chat.connect();
    await this.chat.loadChats();
  }

  protected formatChatLabel(startedAt: string): string {
    const d = new Date(startedAt);
    if (Number.isNaN(d.getTime())) return startedAt;
    return d.toLocaleString(undefined, {
      year: "numeric", month: "short", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
    });
  }

  protected onNewChat(): void {
    this.chat.newChat();
  }

  protected onSelectChat(id: string): void {
    if (this.chat.currentChatId() === id) return;
    void this.chat.openChat(id);
  }

  protected async onCopyMessage(msg: { text?: string }): Promise<void> {
    await this.copyText(msg.text ?? "");
  }

  protected async onCopyCode(code: string): Promise<void> {
    await this.copyText(code);
  }

  private async copyText(text: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); } finally { document.body.removeChild(ta); }
    }
  }

  protected onDeleteMessage(id: string): void {
    if (!confirm("Delete this message?")) return;
    void this.chat.deleteMessage(id);
  }

  protected onDeleteChat(event: Event, id: string): void {
    event.stopPropagation();
    if (!confirm("Delete this chat?")) return;
    void this.chat.deleteChat(id);
  }

  ngAfterViewChecked(): void {
    const list = this.chat.messages();
    if (list.length !== this.lastLength) {
      this.lastLength = list.length;
      queueMicrotask(() =>
        this.scrollAnchor?.nativeElement.scrollIntoView({ behavior: "smooth" })
      );
    }
  }

  submit(): void {
    if ((!this.prompt.trim() && !this.selectedImage) || this.chat.busy()) return;
    const options = this.usesImageGenerationControls()
      ? {
          imageWidth: this.normalizedImageDimension(this.imageWidth, "width"),
          imageHeight: this.normalizedImageDimension(this.imageHeight, "height"),
          steps: this.normalizedImageSteps(),
          seed: this.normalizedImageSeed(),
        }
      : { maxTokens: this.normalizedMaxTokens() };
    this.chat.send(this.prompt, this.selectedImage ?? undefined, options);
    this.prompt = "";
    this.clearImage();
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      this.submit();
    }
  }

  async onImageSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;

    this.imageError = null;
    if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
      this.imageError = "Use a JPEG, PNG, WebP, or GIF image.";
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      this.imageError = "Image must be 10 MB or smaller.";
      return;
    }

    try {
      const dataUrl = await this.readFileAsDataUrl(file);
      this.selectedImage = {
        dataUrl,
        name: file.name,
        type: file.type,
        size: file.size,
      };
    } catch (err) {
      this.imageError = err instanceof Error ? err.message : String(err);
    }
  }

  clearImage(): void {
    this.selectedImage = null;
    this.imageError = null;
  }

  protected pickLlamaModelFile(): void {
    void this.chat.pickLlamaModelFile();
  }

  protected canSend(): boolean {
    if (this.chat.inferenceMode() === "ollama") {
      return (
        this.chat.connected() &&
        !this.chat.busy() &&
        !!this.chat.currentOllamaModel() &&
        (!this.selectedImage || this.hasOllamaCapability("vision")) &&
        (!!this.prompt.trim() || !!this.selectedImage)
      );
    }
    if (this.chat.inferenceMode() === "llamacpp") {
      return (
        this.chat.connected() &&
        !this.chat.busy() &&
        !!this.chat.currentLlamaModel() &&
        !!this.prompt.trim()
      );
    }

    return (
      this.chat.connected() &&
      !this.chat.busy() &&
      !this.chat.modelLoading() &&
      !!this.chat.currentModel() &&
      (!this.selectedImage || this.chat.supportsVision()) &&
      (!!this.prompt.trim() || !!this.selectedImage)
    );
  }

  protected canAttachImage(): boolean {
    if (!this.chat.connected() || this.chat.busy()) return false;
    if (this.chat.inferenceMode() === "ollama") {
      return !!this.chat.currentOllamaModel() && this.hasOllamaCapability("vision");
    }
    if (this.chat.inferenceMode() === "llamacpp") return false;

    return !this.chat.modelLoading() && !!this.chat.currentModel() && this.chat.supportsVision();
  }

  protected imageButtonTooltip(): string {
    if (this.chat.inferenceMode() === "llamacpp") {
      return "Image input is not available for Llama.cpp";
    }
    if (this.chat.inferenceMode() === "ollama") {
      if (!this.chat.currentOllamaModel()) return "Select an Ollama model first";
      if (this.chat.ollamaCapabilitiesLoading()) return "Model capabilities are loading…";
      if (!this.hasOllamaCapability("vision")) return "Image input is not available for this Ollama model";
      return "Attach an image";
    }
    if (!this.chat.currentModel()) return "Select a model first";
    if (this.chat.modelLoading())  return "Model is loading…";
    if (!this.chat.supportsVision()) {
      return "Image input is not available for this model in the current backend";
    }
    return "Attach an image";
  }

  protected usesImageGenerationControls(): boolean {
    if (this.chat.inferenceMode() === "mlx") return this.chat.supportsImageGeneration();
    if (this.chat.inferenceMode() === "ollama") {
      return this.hasOllamaCapability("image") || this.hasOllamaCapability("imagegeneration");
    }
    return false;
  }

  protected canEditImageSettings(): boolean {
    if (!this.chat.connected() || this.chat.busy()) return false;
    if (this.chat.inferenceMode() === "ollama") {
      return !!this.chat.currentOllamaModel() && this.usesImageGenerationControls();
    }
    return !this.chat.modelLoading() && !!this.chat.currentModel() && this.chat.supportsImageGeneration();
  }

  protected hasOllamaCapability(capability: string): boolean {
    const target = this.normalizedCapability(capability);
    return this.chat.ollamaCapabilities().some((value) => this.normalizedCapability(value) === target);
  }

  protected promptPlaceholder(): string {
    if (this.chat.inferenceMode() === "llamacpp") {
      return this.chat.currentLlamaModel()
        ? "Type a prompt for Llama.cpp"
        : "Pick a Llama.cpp model file first";
    }
    if (this.chat.inferenceMode() === "ollama") {
      if (this.usesImageGenerationControls()) return "Describe the image you want to generate with Ollama";
      return this.hasOllamaCapability("vision")
        ? "Type a message... attach an image for vision models"
        : "Type a message for Ollama";
    }
    return this.chat.supportsImageGeneration()
      ? "Describe an image to generate..."
      : "Type a message... attach an image for multimodal models";
  }

  protected pendingLabel(message: ChatMessage): string {
    if (message.queued) {
      return message.queuePosition && message.queuePosition > 1
        ? `queue · position ${message.queuePosition}`
        : "queue";
    }
    return this.usesImageGenerationControls() && message.role === "assistant"
      ? "generating…"
      : "thinking…";
  }

  protected capabilityLabel(capability: string): string {
    const normalized = capability.replace(/[-_]+/g, " ").trim();
    if (!normalized) return capability;
    return normalized[0].toUpperCase() + normalized.slice(1);
  }

  private normalizedCapability(capability: string): string {
    return capability.replace(/[-_\s]+/g, "").trim().toLowerCase();
  }

  protected normalizedMaxTokens(): number {
    if (!Number.isFinite(this.maxTokens)) return DEFAULT_MAX_TOKENS;
    return Math.min(
      MAX_GENERATION_TOKENS,
      Math.max(1, Math.trunc(this.maxTokens))
    );
  }

  protected resetImagePreset(): void {
    const modelId = this.chat.inferenceMode() === "mlx"
      ? this.chat.currentModel()
      : this.chat.currentOllamaModel();
    if (!modelId) return;
    this.applyImagePreset(modelId);
  }

  protected formattedBlocks(message: ChatMessage): FormattedBlock[] {
    if (message.pending) return this.parseFormattedBlocks(message.text ?? "");
    const key = `${message.id}::${message.text ?? ""}`;
    const cached = this.formattedTextCache.get(key);
    if (cached) return cached;
    const parsed = this.parseFormattedBlocks(message.text ?? "");
    this.formattedTextCache.set(key, parsed);
    return parsed;
  }

  private applyImagePreset(modelId: string): void {
    const preset = this.imagePresetForModel(modelId);
    this.imageWidth = preset.width;
    this.imageHeight = preset.height;
    this.imageSteps = preset.steps;
    this.imageSeed = null;
  }

  private imagePresetForModel(modelId: string): ImageGenerationPreset {
    if (/(^|\/)(mlx-)?z-image($|[-_])/i.test(modelId)) {
      return { width: 1024, height: 1024, steps: 9 };
    }
    return { width: 768, height: 768, steps: 30 };
  }

  private normalizedImageDimension(value: number, key: "width" | "height"): number {
    const preset = this.imagePresetForModel(this.chat.currentModel() ?? "");
    if (!Number.isFinite(value)) return preset[key];
    const clamped = Math.min(
      MAX_IMAGE_DIMENSION,
      Math.max(64, Math.trunc(value))
    );
    return Math.max(64, Math.round(clamped / 8) * 8);
  }

  private normalizedImageSteps(): number {
    const preset = this.imagePresetForModel(this.chat.currentModel() ?? "");
    if (!Number.isFinite(this.imageSteps)) return preset.steps;
    return Math.min(MAX_IMAGE_STEPS, Math.max(1, Math.trunc(this.imageSteps)));
  }

  private normalizedImageSeed(): number | undefined {
    const seed = this.imageSeed;
    if (typeof seed !== "number" || !Number.isFinite(seed)) return undefined;
    return Math.min(MAX_IMAGE_SEED, Math.max(0, Math.trunc(seed)));
  }

  private parseFormattedBlocks(text: string): FormattedBlock[] {
    const normalized = text.replace(/\r\n?/g, "\n");
    const lines = normalized.split("\n");
    const blocks: FormattedBlock[] = [];
    let i = 0;

    while (i < lines.length) {
      const rawLine = lines[i];
      const line = rawLine.trim();
      if (!line) {
        i += 1;
        continue;
      }

      const codeMatch = /^\s*```([A-Za-z0-9_+.#-]*)\s*$/.exec(rawLine);
      if (codeMatch) {
        const codeLines: string[] = [];
        i += 1;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
          codeLines.push(lines[i]);
          i += 1;
        }
        if (i < lines.length) i += 1;
        blocks.push({
          kind: "code",
          code: codeLines.join("\n"),
          language: codeMatch[1] ? codeMatch[1] : null,
        });
        continue;
      }

      const chapterMatch = /^\s*###\s+(.+?)\s*$/.exec(rawLine);
      if (chapterMatch) {
        blocks.push({ kind: "chapter", segments: this.parseInlineSegments(chapterMatch[1]) });
        i += 1;
        continue;
      }

      const bulletText = this.parseBulletLine(rawLine);
      if (bulletText !== null) {
        const items: FormattedTextSegment[][] = [];
        while (i < lines.length) {
          const item = this.parseBulletLine(lines[i]);
          if (item === null) break;
          items.push(this.parseInlineSegments(item));
          i += 1;
        }
        blocks.push({ kind: "list", items });
        continue;
      }

      const tableRow = this.parseTableRow(rawLine);
      if (tableRow) {
        const rows: string[][] = [];
        while (i < lines.length) {
          const row = this.parseTableRow(lines[i]);
          if (!row) break;
          rows.push(row);
          i += 1;
        }

        const contentRows = rows.filter((row) => !this.isTableSeparatorRow(row));
        if (contentRows.length > 0) {
          blocks.push({
            kind: "table",
            rows: contentRows.map((row) =>
              row.map((cell) => this.parseInlineSegments(cell))
            ),
          });
          continue;
        }
      }

      const paragraphLines: string[] = [line];
      i += 1;
      while (i < lines.length) {
        const next = lines[i];
        const trimmedNext = next.trim();
        if (!trimmedNext) {
          i += 1;
          break;
        }
        if (/^\s*```/.test(next) || /^\s*###\s+/.test(next) || this.parseBulletLine(next) !== null || this.parseTableRow(next)) {
          break;
        }
        paragraphLines.push(trimmedNext);
        i += 1;
      }
      blocks.push({
        kind: "paragraph",
        segments: this.parseInlineSegments(paragraphLines.join(" ")),
      });
    }

    if (blocks.length === 0 && text.trim()) {
      blocks.push({ kind: "paragraph", segments: this.parseInlineSegments(text.trim()) });
    }

    return blocks;
  }

  private parseInlineSegments(text: string): FormattedTextSegment[] {
    const segments: FormattedTextSegment[] = [];
    const pattern = /\*\*(.+?)\*\*/g;
    let cursor = 0;
    let match: RegExpExecArray | null;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > cursor) {
        segments.push({ text: text.slice(cursor, match.index), bold: false });
      }
      segments.push({ text: match[1], bold: true });
      cursor = match.index + match[0].length;
    }

    if (cursor < text.length) {
      segments.push({ text: text.slice(cursor), bold: false });
    }

    if (segments.length === 0) {
      return [{ text, bold: false }];
    }
    return segments.filter((segment) => segment.text.length > 0);
  }

  private parseBulletLine(line: string): string | null {
    const match = /^\s*\*\s+(.+?)\s*$/.exec(line);
    return match ? match[1] : null;
  }

  private parseTableRow(line: string): string[] | null {
    const trimmed = line.trim();
    if (!trimmed.includes("|")) return null;

    let cells = trimmed.split("|").map((part) => part.trim());
    if (cells[0] === "") cells = cells.slice(1);
    if (cells[cells.length - 1] === "") cells = cells.slice(0, -1);
    if (cells.length < 2) return null;
    return cells;
  }

  private isTableSeparatorRow(row: string[]): boolean {
    return row.every((cell) => /^:?-{3,}:?$/.test(cell));
  }

  private readFileAsDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read image."));
      reader.onload = () => {
        if (typeof reader.result === "string") resolve(reader.result);
        else reject(new Error("Could not read image."));
      };
      reader.readAsDataURL(file);
    });
  }
}
