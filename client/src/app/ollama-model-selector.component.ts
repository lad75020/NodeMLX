import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  OnInit,
  ViewChild,
  computed,
  inject,
  signal,
} from "@angular/core";
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { ChatService, OllamaModel } from "./chat.service";

@Component({
  selector: "app-ollama-model-selector",
  standalone: true,
  imports: [CommonModule, FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: "./ollama-model-selector.component.html",
  styleUrl: "./ollama-model-selector.component.scss",
})
export class OllamaModelSelectorComponent implements OnInit {
  protected readonly chat = inject(ChatService);
  private readonly host = inject(ElementRef<HTMLElement>);

  protected readonly query = signal("");
  protected readonly open = signal(false);
  protected readonly highlighted = signal(0);
  protected readonly typedModelId = computed(() => this.query().trim());

  protected readonly filtered = computed<OllamaModel[]>(() => {
    const q = this.query().trim().toLowerCase();
    const all = this.chat.ollamaModels();
    if (!q) return all.slice(0, 50);
    return all.filter((m) => m.id.toLowerCase().includes(q)).slice(0, 50);
  });

  protected readonly statusLabel = computed(() => {
    const err = this.chat.ollamaModelsError();
    if (err) return `Ollama unavailable`;
    const cur = this.chat.currentOllamaModel();
    return cur ? this.shorten(cur) : "No Ollama model";
  });

  @ViewChild("input") private inputRef?: ElementRef<HTMLInputElement>;

  ngOnInit(): void {
    void this.chat.loadOllamaModels();
  }

  protected shorten(id: string): string {
    return id.replace(/:latest$/, "");
  }

  protected openPanel(): void {
    this.open.set(true);
    this.highlighted.set(0);
  }

  protected onInput(value: string): void {
    this.query.set(value);
    this.open.set(true);
    this.highlighted.set(0);
  }

  protected select(model: OllamaModel): void {
    this.selectModelId(model.id);
  }

  protected selectTypedModel(): void {
    const id = this.typedModelId();
    if (!id) return;
    this.selectModelId(id);
  }

  protected onKey(event: KeyboardEvent): void {
    const list = this.filtered();
    if (event.key === "ArrowDown") {
      event.preventDefault();
      this.open.set(true);
      this.highlighted.update((i) => Math.min(i + 1, Math.max(list.length - 1, 0)));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      this.highlighted.update((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const picked = list[this.highlighted()];
      if (picked) this.select(picked);
      else this.selectTypedModel();
    } else if (event.key === "Escape") {
      this.open.set(false);
    }
  }

  protected refresh(): void {
    void this.chat.loadOllamaModels();
  }

  protected sizeLabel(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) return "";
    const gb = bytes / 1024 / 1024 / 1024;
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    return `${(bytes / 1024 / 1024).toFixed(0)} MB`;
  }

  private selectModelId(modelId: string): void {
    this.query.set("");
    this.open.set(false);
    this.chat.selectOllamaModel(modelId);
    this.inputRef?.nativeElement.blur();
  }

  protected trackById = (_: number, m: OllamaModel): string => m.id;

  @HostListener("document:click", ["$event"])
  protected onDocClick(event: MouseEvent): void {
    if (!this.host.nativeElement.contains(event.target as Node)) {
      this.open.set(false);
    }
  }
}
