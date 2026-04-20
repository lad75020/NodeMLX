import SwiftUI
import UniformTypeIdentifiers

struct ComposerView: View {
    @EnvironmentObject private var store: ChatStore
    @Binding var prompt: String
    @Binding var selectedImage: ChatImageAttachment?
    @Binding var imageError: String?
    @Binding var maxTokens: Int
    @Binding var imageWidth: Int
    @Binding var imageHeight: Int
    @Binding var imageSteps: Int
    @Binding var imageSeed: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let selectedImage {
                HStack(spacing: 10) {
                    AttachmentImageView(image: selectedImage)
                        .frame(width: 72, height: 52)
                    VStack(alignment: .leading) {
                        Text(selectedImage.name)
                            .lineLimit(1)
                        Text(DisplayFormatters.bytes(selectedImage.size))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Remove") {
                        self.selectedImage = nil
                    }
                }
                .padding(8)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
            }

            if let imageError {
                Text(imageError)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            if usesImageGenerationControls {
                imageGenerationControls
            } else if store.inferenceMode == .mlx {
                HStack {
                    Text("Max tokens")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Stepper(value: $maxTokens, in: 1...32768, step: 256) {
                        TextField("Max tokens", value: $maxTokens, format: .number)
                            .frame(width: 90)
                    }
                    .disabled(!canEditTextGeneration)
                }
            }

            HStack(alignment: .bottom, spacing: 8) {
                Button {
                    pickImage()
                } label: {
                    Image(systemName: "paperclip")
                }
                .disabled(!canAttachImage)
                .help(imageButtonTooltip)

                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(minHeight: 54, maxHeight: 120)
                    .scrollContentBackground(.hidden)
                    .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.tertiary))
                    .disabled(!canEditPrompt)

                Button {
                    submit()
                } label: {
                    if store.busy {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Image(systemName: "paperplane.fill")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSend)
                .keyboardShortcut(.return, modifiers: [.command])
                .help("Send")
            }
        }
        .frame(maxWidth: 860)
        .frame(maxWidth: .infinity)
    }

    private var imageGenerationControls: some View {
        HStack(spacing: 10) {
            numericField("Width", value: $imageWidth, width: 78)
            numericField("Height", value: $imageHeight, width: 78)
            numericField("Steps", value: $imageSteps, width: 64)
            HStack(spacing: 4) {
                Text("Seed")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("Random", value: $imageSeed, format: .number)
                    .frame(width: 92)
                    .textFieldStyle(.roundedBorder)
            }
            Button("Preset") {
                applyImagePreset()
            }
            .disabled(!canEditImageSettings)
        }
        .disabled(!canEditImageSettings)
    }

    private func numericField(_ label: String, value: Binding<Int>, width: CGFloat) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextField(label, value: value, format: .number)
                .frame(width: width)
                .textFieldStyle(.roundedBorder)
        }
    }

    private var canEditPrompt: Bool {
        guard store.connected, !store.busy else { return false }
        switch store.inferenceMode {
        case .mlx:
            return !store.modelLoading && store.currentModel != nil
        case .ollama:
            return store.currentOllamaModel != nil
        case .llamacpp:
            return hasSelectedLlamaModel
        }
    }

    private var canSend: Bool {
        guard canEditPrompt else { return false }
        switch store.inferenceMode {
        case .mlx:
            return (selectedImage == nil || store.supportsVision) && (!prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || selectedImage != nil)
        case .ollama:
            return (selectedImage == nil || store.hasOllamaCapability("vision")) && (!prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || selectedImage != nil)
        case .llamacpp:
            return !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    private var canAttachImage: Bool {
        guard store.connected, !store.busy else { return false }
        switch store.inferenceMode {
        case .mlx:
            return !store.modelLoading && store.currentModel != nil && store.supportsVision
        case .ollama:
            return store.currentOllamaModel != nil && store.hasOllamaCapability("vision")
        case .llamacpp:
            return false
        }
    }

    private var usesImageGenerationControls: Bool {
        switch store.inferenceMode {
        case .mlx:
            return store.supportsImageGeneration
        case .ollama:
            return store.hasOllamaCapability("image") || store.hasOllamaCapability("imagegeneration")
        case .llamacpp:
            return false
        }
    }

    private var canEditImageSettings: Bool {
        guard store.connected, !store.busy else { return false }
        switch store.inferenceMode {
        case .mlx:
            return !store.modelLoading && store.currentModel != nil && store.supportsImageGeneration
        case .ollama:
            return store.currentOllamaModel != nil && usesImageGenerationControls
        case .llamacpp:
            return false
        }
    }

    private var canEditTextGeneration: Bool {
        store.connected && !store.busy && !store.modelLoading && store.currentModel != nil
    }

    private var hasSelectedLlamaModel: Bool {
        store.llamaModelSource == .disk
            ? store.currentLlamaModel != nil
            : !store.llamaHuggingFaceModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var imageButtonTooltip: String {
        switch store.inferenceMode {
        case .llamacpp:
            return "Image input is not available for Llama.cpp"
        case .ollama:
            if store.currentOllamaModel == nil { return "Select an Ollama model first" }
            if store.ollamaCapabilitiesLoading { return "Model capabilities are loading" }
            if !store.hasOllamaCapability("vision") { return "Image input is not available for this Ollama model" }
            return "Attach image"
        case .mlx:
            if store.currentModel == nil { return "Select a model first" }
            if store.modelLoading { return "Model is loading" }
            if !store.supportsVision { return "Image input is not available for this model" }
            return "Attach image"
        }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .gif]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let attachment = try url.imageAttachment()
                guard attachment.size <= 10 * 1024 * 1024 else {
                    imageError = "Image must be 10 MB or smaller."
                    return
                }
                selectedImage = attachment
                imageError = nil
            } catch {
                imageError = "Could not read image."
            }
        }
    }

    private func submit() {
        let options = GenerationOptions(
            maxTokens: max(1, min(32768, maxTokens)),
            imageWidth: normalizedImageDimension(imageWidth),
            imageHeight: normalizedImageDimension(imageHeight),
            steps: max(1, min(150, imageSteps)),
            seed: imageSeed.map { max(0, min(Int(Int32.max), $0)) }
        )
        store.send(prompt: prompt, image: selectedImage, options: options)
        prompt = ""
        selectedImage = nil
        imageError = nil
    }

    private func applyImagePreset() {
        let model = store.inferenceMode == .mlx ? store.currentModel : store.currentOllamaModel
        if let model, model.range(of: #"(^|/)(mlx-)?z-image($|[-_])"#, options: [.regularExpression, .caseInsensitive]) != nil {
            imageWidth = 1024
            imageHeight = 1024
            imageSteps = 9
        } else {
            imageWidth = 768
            imageHeight = 768
            imageSteps = 30
        }
        imageSeed = nil
    }

    private func normalizedImageDimension(_ value: Int) -> Int {
        max(64, min(2048, Int(round(Double(value) / 8.0) * 8)))
    }
}
