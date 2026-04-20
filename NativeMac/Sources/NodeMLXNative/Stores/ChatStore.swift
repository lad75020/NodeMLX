import AppKit
import Foundation

@MainActor
final class ChatStore: ObservableObject {
    @Published var serverURLString: String = UserDefaults.standard.string(forKey: "serverURL") ?? "http://127.0.0.1:3000"
    @Published var user: AuthUser?
    @Published var websocketToken: String? = UserDefaults.standard.string(forKey: "websocketToken")
    @Published var authLoading = false
    @Published var authError: String?

    @Published var messages: [ChatMessage] = []
    @Published var chats: [ChatSummary] = []
    @Published var currentChatId: String?
    @Published var connected = false
    @Published var connectionError: String?
    @Published var busy = false
    @Published var inferenceMode: InferenceMode = .mlx
    @Published var currentModel: String?
    @Published var pendingModel: String?
    @Published var modelLoading = false
    @Published var modelError: String?
    @Published var supportsVision = false
    @Published var supportsImageGeneration = false
    @Published var availableModels: [HFModel] = []
    @Published var modelsLoading = false
    @Published var modelsError: String?
    @Published var failedModels: [String: String] = [:]
    @Published var ollamaModels: [OllamaModel] = []
    @Published var ollamaModelsLoading = false
    @Published var ollamaModelsError: String?
    @Published var currentOllamaModel: String?
    @Published var ollamaCapabilities: [String] = []
    @Published var ollamaCapabilitiesLoading = false
    @Published var ollamaCapabilitiesError: String?
    @Published var ollamaURL: String?
    @Published var currentLlamaModel: LlamaModelFile?
    @Published var llamaModelSource: LlamaModelSource = .disk
    @Published var llamaHuggingFaceModel = ""
    @Published var llamaModelError: String?
    @Published var gpuUsage: GpuUsage?
    @Published var inferenceRunning = false

    private let socket = ChatSocketClient()
    private var pendingRPC: [String: CheckedContinuation<JSONValue?, Error>] = [:]
    private var ollamaDetailsRequest = 0

    init() {
        socket.delegate = self
    }

    var serverURL: URL? {
        URL(string: serverURLString.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    var isAuthenticated: Bool {
        user != nil || websocketToken != nil
    }

    func saveServerURL() {
        UserDefaults.standard.set(serverURLString, forKey: "serverURL")
    }

    func restoreSession() async {
        saveServerURL()
        guard let serverURL else { return }
        authLoading = true
        authError = nil
        defer { authLoading = false }

        if let token = websocketToken, !token.isEmpty {
            connectSocket(token: token)
            await loadInitialData()
            return
        }

        do {
            let response = try await AuthClient(serverURL: serverURL).restoreSession()
            guard response.authenticated == true, let token = response.token else {
                user = nil
                websocketToken = nil
                return
            }
            user = response.user
            setToken(token)
            connectSocket(token: token)
            await loadInitialData()
        } catch {
            user = nil
            websocketToken = nil
        }
    }

    func login(username: String, password: String) async {
        saveServerURL()
        guard let serverURL else {
            authError = "Enter a valid server URL."
            return
        }
        authLoading = true
        authError = nil
        defer { authLoading = false }

        do {
            let response = try await AuthClient(serverURL: serverURL).login(username: username, password: password)
            guard let user = response.user, let token = response.token else {
                authError = response.error ?? "Authentication failed."
                return
            }
            self.user = user
            setToken(token)
            connectSocket(token: token)
            await loadInitialData()
        } catch {
            authError = error.localizedDescription
        }
    }

    func logout() async {
        if let serverURL {
            await AuthClient(serverURL: serverURL).logout()
        }
        socket.disconnect()
        user = nil
        setToken(nil)
        resetState()
    }

    func connectSocket(token: String? = nil) {
        guard let serverURL, let token = token ?? websocketToken else { return }
        socket.connect(serverURL: serverURL, token: token)
    }

    func loadInitialData() async {
        do {
            async let chats: Void = loadChats()
            async let models: Void = loadAvailableModels(refresh: false)
            async let failed: Void = loadFailedModels()
            _ = try await (chats, models, failed)
        } catch {}
    }

    func newChat() {
        currentChatId = nil
        messages = []
        busy = false
    }

    func loadChats() async throws {
        let value = try await rpc("listChats")
        let response = try value?.decode(ListChatsResponse.self)
        chats = response?.chats ?? []
    }

    func openChat(_ id: String) async {
        do {
            let value = try await rpc("openChat", payload: ["chatId": .string(id)])
            let response = try value?.decode(OpenChatResponse.self)
            currentChatId = response?.id
            messages = (response?.messages ?? []).map { message in
                var copy = message
                copy.pending = false
                if copy.role == .assistant, copy.provider == .ollama {
                    copy = copy.withExtractedOllamaImages()
                }
                return copy
            }
            busy = false
        } catch {
            authError = error.localizedDescription
        }
    }

    func deleteChat(_ id: String) async {
        _ = try? await rpc("deleteChat", payload: ["chatId": .string(id)])
        chats.removeAll { $0.id == id }
        if currentChatId == id {
            newChat()
        }
    }

    func deleteMessage(_ id: String) async {
        messages.removeAll { $0.id == id }
        guard let currentChatId else { return }
        _ = try? await rpc("deleteMessage", payload: ["chatId": .string(currentChatId), "messageId": .string(id)])
    }

    func loadAvailableModels(refresh: Bool) async throws {
        modelsLoading = true
        modelsError = nil
        defer { modelsLoading = false }
        do {
            let value = try await rpc("listModels", payload: ["refresh": .bool(refresh)])
            let response = try value?.decode(ListModelsResponse.self)
            availableModels = response?.models ?? []
        } catch {
            modelsError = error.localizedDescription
        }
    }

    func loadFailedModels() async throws {
        let value = try await rpc("listFailedModels")
        let response = try value?.decode(ListFailedModelsResponse.self)
        failedModels = Dictionary(uniqueKeysWithValues: (response?.failed ?? []).map { ($0.id, $0.error) })
    }

    func loadOllamaModels() async {
        ollamaModelsLoading = true
        ollamaModelsError = nil
        defer { ollamaModelsLoading = false }
        do {
            let value = try await rpc("listOllamaModels")
            let response = try value?.decode(ListOllamaModelsResponse.self)
            let models = response?.models ?? []
            ollamaModels = models
            ollamaURL = response?.url
            if currentOllamaModel == nil, let first = models.first {
                await selectOllamaModel(first.id)
            } else if let currentOllamaModel {
                await loadOllamaModelDetails(currentOllamaModel)
            }
        } catch {
            ollamaModels = []
            ollamaModelsError = error.localizedDescription
        }
    }

    func setInferenceMode(_ mode: InferenceMode) {
        guard !busy, inferenceMode != mode else { return }
        inferenceMode = mode
        if mode == .ollama, ollamaModels.isEmpty {
            Task { await loadOllamaModels() }
        }
    }

    func selectModel(_ id: String) {
        let modelId = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelId.isEmpty, !modelLoading, modelId != currentModel else { return }
        pendingModel = modelId
        modelLoading = true
        modelError = nil
        socket.send(["type": .string("selectModel"), "modelId": .string(modelId)])
    }

    func selectOllamaModel(_ id: String) async {
        let modelId = id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelId.isEmpty else { return }
        currentOllamaModel = modelId
        if !ollamaModels.contains(where: { $0.id == modelId }) {
            ollamaModels.insert(OllamaModel(id: modelId, name: modelId, modifiedAt: nil, size: 0, digest: nil), at: 0)
        }
        await loadOllamaModelDetails(modelId)
    }

    func setLlamaModelFromPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = []
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.title = "Choose a Llama.cpp model"
        if panel.runModal() == .OK, let url = panel.url {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            currentLlamaModel = LlamaModelFile(path: url.path, name: url.lastPathComponent, size: size)
            llamaModelError = nil
        }
    }

    func send(prompt: String, image: ChatImageAttachment?, options: GenerationOptions) {
        switch inferenceMode {
        case .mlx:
            sendMLX(prompt: prompt, image: image, options: options)
        case .ollama:
            sendOllama(prompt: prompt, image: image, options: options)
        case .llamacpp:
            sendLlama(prompt: prompt, options: options)
        }
    }

    func cancelInference() {
        socket.send(["type": .string("cancelInference")])
    }

    func hasOllamaCapability(_ capability: String) -> Bool {
        let target = normalizedCapability(capability)
        return ollamaCapabilities.contains { normalizedCapability($0) == target }
    }

    private func loadOllamaModelDetails(_ id: String) async {
        ollamaDetailsRequest += 1
        let requestId = ollamaDetailsRequest
        ollamaCapabilities = []
        ollamaCapabilitiesLoading = true
        ollamaCapabilitiesError = nil
        defer {
            if requestId == ollamaDetailsRequest {
                ollamaCapabilitiesLoading = false
            }
        }
        do {
            let value = try await rpc("showOllamaModel", payload: ["modelId": .string(id)])
            guard requestId == ollamaDetailsRequest, currentOllamaModel == id else { return }
            let response = try value?.decode(ShowOllamaModelResponse.self)
            ollamaCapabilities = response?.model.capabilities ?? []
        } catch {
            guard requestId == ollamaDetailsRequest, currentOllamaModel == id else { return }
            ollamaCapabilitiesError = error.localizedDescription
        }
    }

    private func sendMLX(prompt: String, image: ChatImageAttachment?, options: GenerationOptions) {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && image != nil
            ? "Describe this image."
            : prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard (!text.isEmpty || image != nil), !busy, !modelLoading, currentModel != nil else { return }
        let id = UUID().uuidString
        messages.append(ChatMessage(id: id, role: .user, text: text, image: image))
        messages.append(ChatMessage(id: "\(id):reply", role: .assistant, text: "", pending: true))
        busy = true
        sendPrompt(type: "prompt", id: id, prompt: text, image: image, options: options, extra: ["chatId": currentChatId.map(SocketValue.string) ?? .null])
    }

    private func sendOllama(prompt: String, image: ChatImageAttachment?, options: GenerationOptions) {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && image != nil
            ? "Describe this image."
            : prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let modelId = currentOllamaModel, (!text.isEmpty || image != nil), !busy, image == nil || hasOllamaCapability("vision") else { return }
        let id = UUID().uuidString
        messages.append(ChatMessage(id: id, role: .user, text: text, provider: .ollama, image: image))
        messages.append(ChatMessage(id: "\(id):reply", role: .assistant, text: "", provider: .ollama, pending: true, modelId: modelId))
        busy = true
        let extra: [String: SocketValue] = [
            "chatId": currentChatId.map(SocketValue.string) ?? .null,
            "modelId": .string(modelId),
            "enableThinking": .bool(hasOllamaCapability("thinking"))
        ]
        sendPrompt(type: "ollamaPrompt", id: id, prompt: text, image: image, options: options, extra: extra)
    }

    private func sendLlama(prompt: String, options: GenerationOptions) {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let modelName = llamaModelSource == .disk ? currentLlamaModel?.name : llamaHuggingFaceModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !busy, let modelName, !modelName.isEmpty else { return }
        let id = UUID().uuidString
        messages.append(ChatMessage(id: id, role: .user, text: text, provider: .llamacpp))
        messages.append(ChatMessage(id: "\(id):reply", role: .assistant, text: "", provider: .llamacpp, pending: true, modelId: modelName))
        busy = true
        var payload: [String: SocketValue] = [
            "type": .string("llamaPrompt"),
            "id": .string(id),
            "chatId": currentChatId.map(SocketValue.string) ?? .null,
            "modelSource": .string(llamaModelSource.rawValue),
            "hfModel": .string(llamaHuggingFaceModel.trimmingCharacters(in: .whitespacesAndNewlines)),
            "prompt": .string(text)
        ]
        if let path = currentLlamaModel?.path { payload["modelPath"] = .string(path) }
        if let maxTokens = options.maxTokens { payload["maxTokens"] = .int(maxTokens) }
        socket.send(payload)
    }

    private func sendPrompt(type: String, id: String, prompt: String, image: ChatImageAttachment?, options: GenerationOptions, extra: [String: SocketValue]) {
        var payload: [String: SocketValue] = [
            "type": .string(type),
            "id": .string(id),
            "prompt": .string(prompt)
        ]
        for (key, value) in extra { payload[key] = value }
        if let image { payload["image"] = .image(image) }
        if let maxTokens = options.maxTokens { payload["maxTokens"] = .int(maxTokens) }
        if let width = options.imageWidth { payload["imageWidth"] = .int(width) }
        if let height = options.imageHeight { payload["imageHeight"] = .int(height) }
        if let steps = options.steps { payload["steps"] = .int(steps) }
        if let seed = options.seed { payload["seed"] = .int(seed) }
        socket.send(payload)
    }

    private func rpc(_ type: String, payload: [String: SocketValue] = [:]) async throws -> JSONValue? {
        try await withCheckedThrowingContinuation { continuation in
            let requestId = UUID().uuidString
            pendingRPC[requestId] = continuation
            var message = payload
            message["type"] = .string(type)
            message["requestId"] = .string(requestId)
            socket.send(message)
        }
    }

    private func setToken(_ token: String?) {
        websocketToken = token
        if let token {
            UserDefaults.standard.set(token, forKey: "websocketToken")
        } else {
            UserDefaults.standard.removeObject(forKey: "websocketToken")
        }
    }

    private func normalizedCapability(_ capability: String) -> String {
        capability
            .replacingOccurrences(of: "[-_\\s]+", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private func resetState() {
        messages = []
        chats = []
        currentChatId = nil
        connected = false
        connectionError = nil
        busy = false
        inferenceMode = .mlx
        currentModel = nil
        pendingModel = nil
        modelLoading = false
        modelError = nil
        supportsVision = false
        supportsImageGeneration = false
        availableModels = []
        modelsError = nil
        failedModels = [:]
        ollamaModels = []
        ollamaModelsError = nil
        currentOllamaModel = nil
        ollamaCapabilities = []
        ollamaCapabilitiesError = nil
        currentLlamaModel = nil
        llamaHuggingFaceModel = ""
        llamaModelError = nil
        gpuUsage = nil
        inferenceRunning = false
    }
}

extension ChatStore: ChatSocketClientDelegate {
    func socketClientDidConnect() {
        connected = true
        connectionError = nil
    }

    func socketClientDidDisconnect(error: String?) {
        connected = false
        connectionError = error
        busy = false
        gpuUsage = nil
        inferenceRunning = false
        for (_, continuation) in pendingRPC {
            continuation.resume(throwing: URLError(.networkConnectionLost))
        }
        pendingRPC.removeAll()
    }

    func socketClientDidReceive(_ event: SocketEvent) {
        switch event {
        case .rpcResult(let requestId, let data, let error):
            guard let continuation = pendingRPC.removeValue(forKey: requestId) else { return }
            if let error {
                continuation.resume(throwing: AuthError.message(error))
            } else {
                continuation.resume(returning: data)
            }

        case .gpuUsage(let running, let gpu, let memory):
            inferenceRunning = running
            gpuUsage = running && gpu != nil && memory != nil ? GpuUsage(gpu: gpu!, memory: memory!) : nil
            if !running { busy = false }

        case .start(let id, let chatId):
            if let chatId, currentChatId == nil { currentChatId = chatId }
            updateReply(id) { message in
                message.queued = false
                message.queuePosition = nil
            }

        case .queued(let id, let position):
            updateReply(id) { message in
                message.pending = true
                message.queued = true
                message.queuePosition = position
            }

        case .chatCreated(_, let chat):
            currentChatId = chat.id
            chats.removeAll { $0.id == chat.id }
            chats.insert(chat, at: 0)

        case .response(let id, _, let modelId, let text, let images, let tokenCount, let tokensPerSecond):
            updateReply(id) { message in
                message.text = text
                message.images = images
                message.pending = false
                message.queued = false
                message.queuePosition = nil
                message.tokenCount = tokenCount
                message.tokensPerSecond = tokensPerSecond
                message.modelId = modelId
            }
            busy = false
            reloadChatsIfNeeded()

        case .ollamaChunk(let id, let text, let thinking, let images):
            updateReply(id) { message in
                message.text += text ?? ""
                message.thinking = [message.thinking, thinking].compactMap { $0 }.joined()
                if message.thinking?.isEmpty == true { message.thinking = nil }
                message.images = (message.images ?? []).merged(with: images ?? [])
                message.pending = true
                message.queued = false
                message.queuePosition = nil
                message.provider = .ollama
                message = message.withExtractedOllamaImages()
            }

        case .ollamaDone(let id, _, let modelId, let text, let thinking, let images, let evalCount):
            updateReply(id) { message in
                message.text = text
                message.thinking = thinking ?? message.thinking
                message.images = (message.images ?? []).merged(with: images ?? [])
                message.pending = false
                message.queued = false
                message.queuePosition = nil
                message.provider = .ollama
                message.modelId = modelId
                message.tokenCount = evalCount
                message = message.withExtractedOllamaImages()
            }
            busy = false
            reloadChatsIfNeeded()

        case .llamaChunk(let id, let text, let thinking):
            updateReply(id) { message in
                message.text += text ?? ""
                message.thinking = [message.thinking, thinking].compactMap { $0 }.joined()
                if message.thinking?.isEmpty == true { message.thinking = nil }
                message.pending = true
                message.queued = false
                message.queuePosition = nil
                message.provider = .llamacpp
            }

        case .llamaDone(let id, _, let modelName, let text, let thinking):
            updateReply(id) { message in
                message.text = text
                message.thinking = thinking ?? message.thinking
                message.pending = false
                message.queued = false
                message.queuePosition = nil
                message.provider = .llamacpp
                message.modelId = modelName
            }
            busy = false
            reloadChatsIfNeeded()

        case .error(let id, let error):
            if let id {
                updateReply(id) { message in
                    message.text = "Warning: \(error)"
                    message.pending = false
                    message.queued = false
                    message.queuePosition = nil
                }
            } else {
                messages.append(ChatMessage(id: UUID().uuidString, role: .assistant, text: "Warning: \(error)"))
            }
            busy = false

        case .modelLoading(let modelId):
            modelLoading = true
            pendingModel = modelId
            modelError = nil
            supportsVision = false
            supportsImageGeneration = false

        case .modelReady(let modelId, let isVLM, let canGenerateImages):
            currentModel = modelId
            pendingModel = nil
            modelLoading = false
            modelError = nil
            supportsVision = isVLM == true
            supportsImageGeneration = canGenerateImages == true
            failedModels.removeValue(forKey: modelId)

        case .modelError(_, let error, let failed):
            modelLoading = false
            pendingModel = nil
            modelError = error
            supportsVision = false
            supportsImageGeneration = false
            if let failed {
                failedModels = Dictionary(uniqueKeysWithValues: failed.map { ($0.id, $0.error) })
            }

        case .unknown:
            break
        }
    }

    private func updateReply(_ id: String, body: (inout ChatMessage) -> Void) {
        let replyId = "\(id):reply"
        guard let index = messages.firstIndex(where: { $0.id == replyId }) else { return }
        body(&messages[index])
    }

    private func reloadChatsIfNeeded() {
        guard let currentChatId else { return }
        if chats.first(where: { $0.id == currentChatId })?.title == nil {
            Task { try? await loadChats() }
        }
    }
}

private struct ListChatsResponse: Decodable { let chats: [ChatSummary] }
private struct OpenChatResponse: Decodable { let id: String; let messages: [ChatMessage] }
private struct ListModelsResponse: Decodable { let models: [HFModel] }
private struct ListFailedModelsResponse: Decodable { let failed: [FailedModel] }
private struct ListOllamaModelsResponse: Decodable { let models: [OllamaModel]; let url: String? }
private struct ShowOllamaModelResponse: Decodable { let model: OllamaModelDetails }

private extension Array where Element == ChatImageAttachment {
    func merged(with other: [ChatImageAttachment]) -> [ChatImageAttachment] {
        var seen = Set<String>()
        var result: [ChatImageAttachment] = []
        for image in self + other where seen.insert(image.dataUrl).inserted {
            result.append(image)
        }
        return result
    }
}

private extension ChatMessage {
    func withExtractedOllamaImages() -> ChatMessage {
        let extracted = text.extractOllamaImages()
        var copy = self
        copy.text = extracted.text
        copy.images = (images ?? []).merged(with: extracted.images)
        if copy.images?.isEmpty == true { copy.images = nil }
        return copy
    }
}
