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
import { ChatService, HFModel } from "./chat.service";

@Component({
  selector: "app-model-selector",
  standalone: true,
  imports: [CommonModule, FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: "./model-selector.component.html",
  styleUrl: "./model-selector.component.scss",
})
export class ModelSelectorComponent implements OnInit {
  protected readonly chat = inject(ChatService);
  private readonly host = inject(ElementRef<HTMLElement>);

  protected readonly query = signal("");
  protected readonly open = signal(false);
  protected readonly highlighted = signal(0);
  protected readonly typedModelId = computed(() => this.query().trim());

  protected readonly filtered = computed<HFModel[]>(() => {
    const q = this.query().trim().toLowerCase();
    const all = this.chat.availableModels();
    if (!q) return all.slice(0, 50);
    // When searching: saved models surface to the top of matching results.
    const matches = all.filter((m) => m.id.toLowerCase().includes(q));
    matches.sort((a, b) => {
      if (a.saved && !b.saved) return -1;
      if (!a.saved && b.saved) return 1;
      return 0;
    });
    return matches.slice(0, 50);
  });

  protected readonly statusLabel = computed(() => {
    if (this.chat.modelLoading()) {
      const target = this.chat.pendingModel();
      return target ? `Loading ${this.shorten(target)}…` : "Loading model…";
    }
    const err = this.chat.modelError();
    if (err) return `Error: ${err}`;
    const cur = this.chat.currentModel();
    return cur ? this.shorten(cur) : "No model loaded";
  });

  @ViewChild("input") private inputRef?: ElementRef<HTMLInputElement>;

  ngOnInit(): void {
    this.chat.loadAvailableModels();
    this.chat.loadFailedModels();
  }

  protected shorten(id: string): string {
    return id.replace(/^mlx-community\//, "");
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

  protected select(model: HFModel): void {
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
    this.chat.loadAvailableModels(true);
  }

  private selectModelId(modelId: string): void {
    this.query.set("");
    this.open.set(false);
    this.chat.selectModel(modelId);
    this.inputRef?.nativeElement.blur();
  }

  protected trackById = (_: number, m: HFModel): string => m.id;

  @HostListener("document:click", ["$event"])
  protected onDocClick(event: MouseEvent): void {
    if (!this.host.nativeElement.contains(event.target as Node)) {
      this.open.set(false);
    }
  }
}
