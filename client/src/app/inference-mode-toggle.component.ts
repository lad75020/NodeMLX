import { ChangeDetectionStrategy, Component, inject } from "@angular/core";
import { CommonModule } from "@angular/common";
import { ChatService, InferenceMode } from "./chat.service";

@Component({
  selector: "app-inference-mode-toggle",
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: "./inference-mode-toggle.component.html",
  styleUrl: "./inference-mode-toggle.component.scss",
})
export class InferenceModeToggleComponent {
  protected readonly chat = inject(ChatService);

  protected setMode(mode: InferenceMode): void {
    this.chat.setInferenceMode(mode);
  }
}
