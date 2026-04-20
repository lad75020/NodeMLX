import SwiftUI

struct ChatDetailView: View {
    @EnvironmentObject private var store: ChatStore
    @Binding var columnVisibility: NavigationSplitViewVisibility
    @State private var prompt = ""
    @State private var selectedImage: ChatImageAttachment?
    @State private var imageError: String?
    @State private var maxTokens = 4096
    @State private var imageWidth = 768
    @State private var imageHeight = 768
    @State private var imageSteps = 30
    @State private var imageSeed: Int?

    var body: some View {
        VStack(spacing: 0) {
            ChatHeaderView()
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.bar)

            Divider()

            MessageListView()

            Divider()

            ComposerView(
                prompt: $prompt,
                selectedImage: $selectedImage,
                imageError: $imageError,
                maxTokens: $maxTokens,
                imageWidth: $imageWidth,
                imageHeight: $imageHeight,
                imageSteps: $imageSteps,
                imageSeed: $imageSeed
            )
            .padding(14)
        }
    }
}

struct ChatHeaderView: View {
    @EnvironmentObject private var store: ChatStore

    var body: some View {
        HStack(spacing: 12) {
            Picker("Mode", selection: Binding(
                get: { store.inferenceMode },
                set: { store.setInferenceMode($0) }
            )) {
                ForEach(InferenceMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 260)
            .disabled(store.busy)

            if let usage = store.gpuUsage {
                Label("GPU \(usage.gpu, specifier: "%.0f")%  Memory \(usage.memory, specifier: "%.0f")%", systemImage: "gauge.with.dots.needle.bottom.50percent")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

            if store.inferenceRunning {
                Button("Cancel", role: .destructive) {
                    store.cancelInference()
                }
            }

            Spacer()

            switch store.inferenceMode {
            case .mlx:
                MLXModelPickerView()
                    .frame(width: 360)
            case .ollama:
                OllamaModelPickerView()
                    .frame(width: 360)
            case .llamacpp:
                LlamaModelPickerView()
                    .frame(width: 420)
            }
        }
    }
}
