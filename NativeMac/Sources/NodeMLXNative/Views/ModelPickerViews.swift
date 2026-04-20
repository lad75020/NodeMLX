import SwiftUI

struct MLXModelPickerView: View {
    @EnvironmentObject private var store: ChatStore
    @State private var query = ""

    private var filtered: [HFModel] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let models = q.isEmpty ? store.availableModels : store.availableModels.filter { $0.id.lowercased().contains(q) }
        return Array(models.prefix(50))
    }

    var body: some View {
        Menu {
            TextField("Search or paste model id", text: $query)
            Button("Refresh Models") {
                Task { try? await store.loadAvailableModels(refresh: true) }
            }
            Divider()
            if !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Button("Load \(query)") {
                    store.selectModel(query)
                    query = ""
                }
            }
            ForEach(filtered) { model in
                Button {
                    store.selectModel(model.id)
                } label: {
                    HStack {
                        Text(DisplayFormatters.shortModelId(model.id))
                        if model.saved == true {
                            Image(systemName: "checkmark.circle.fill")
                        }
                    }
                }
            }
        } label: {
            Label(status, systemImage: store.modelLoading ? "hourglass" : "cpu")
                .lineLimit(1)
        }
        .help(store.modelError ?? "MLX model")
    }

    private var status: String {
        if store.modelLoading {
            return store.pendingModel.map { "Loading \(DisplayFormatters.shortModelId($0))" } ?? "Loading model"
        }
        if let error = store.modelError {
            return "Error: \(error)"
        }
        return store.currentModel.map(DisplayFormatters.shortModelId) ?? "No model loaded"
    }
}

struct OllamaModelPickerView: View {
    @EnvironmentObject private var store: ChatStore
    @State private var query = ""

    private var filtered: [OllamaModel] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let models = q.isEmpty ? store.ollamaModels : store.ollamaModels.filter { $0.id.lowercased().contains(q) }
        return Array(models.prefix(50))
    }

    var body: some View {
        Menu {
            TextField("Search or paste model id", text: $query)
            Button("Refresh Ollama Models") {
                Task { await store.loadOllamaModels() }
            }
            Divider()
            if !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Button("Use \(query)") {
                    Task { await store.selectOllamaModel(query) }
                    query = ""
                }
            }
            ForEach(filtered) { model in
                Button {
                    Task { await store.selectOllamaModel(model.id) }
                } label: {
                    Text("\(DisplayFormatters.shortModelId(model.id)) \(DisplayFormatters.bytes(model.size))")
                }
            }
        } label: {
            Label(store.currentOllamaModel.map(DisplayFormatters.shortModelId) ?? "No Ollama model", systemImage: "server.rack")
                .lineLimit(1)
        }
        .task {
            if store.ollamaModels.isEmpty {
                await store.loadOllamaModels()
            }
        }
        .help(store.ollamaModelsError ?? store.ollamaURL ?? "Ollama model")
    }
}

struct LlamaModelPickerView: View {
    @EnvironmentObject private var store: ChatStore

    var body: some View {
        HStack(spacing: 8) {
            Picker("Source", selection: $store.llamaModelSource) {
                Text("Disk").tag(LlamaModelSource.disk)
                Text("Hugging Face").tag(LlamaModelSource.huggingface)
            }
            .pickerStyle(.segmented)
            .frame(width: 170)
            .disabled(store.busy)

            if store.llamaModelSource == .disk {
                Button {
                    store.setLlamaModelFromPanel()
                } label: {
                    Label(store.currentLlamaModel?.name ?? "Pick GGUF", systemImage: "folder")
                        .lineLimit(1)
                }
            } else {
                TextField("owner/model", text: $store.llamaHuggingFaceModel)
                    .textFieldStyle(.roundedBorder)
            }
        }
    }
}
