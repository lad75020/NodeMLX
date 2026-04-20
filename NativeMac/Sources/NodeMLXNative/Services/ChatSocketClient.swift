import Foundation

@MainActor
protocol ChatSocketClientDelegate: AnyObject {
    func socketClientDidConnect()
    func socketClientDidDisconnect(error: String?)
    func socketClientDidReceive(_ event: SocketEvent)
}

@MainActor
final class ChatSocketClient: NSObject {
    private static let maximumMessageSize = 64 * 1024 * 1024

    weak var delegate: ChatSocketClientDelegate?

    private var task: URLSessionWebSocketTask?
    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var outbox: [String] = []
    private var opened = false

    override init() {
        super.init()
    }

    var isConnected: Bool {
        opened && task?.state == .running
    }

    func connect(serverURL: URL, token: String) {
        disconnect()

        var components = URLComponents(url: serverURL, resolvingAgainstBaseURL: false)
        components?.scheme = serverURL.scheme == "https" ? "wss" : "ws"
        components?.path = "/ws"
        components?.queryItems = [URLQueryItem(name: "token", value: token)]

        guard let url = components?.url else { return }
        opened = false
        let task = session.webSocketTask(with: url)
        task.maximumMessageSize = Self.maximumMessageSize
        self.task = task
        task.resume()
        receiveLoop()
    }

    func disconnect() {
        let hadTask = task != nil
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        opened = false
        outbox.removeAll()
        if hadTask {
            delegate?.socketClientDidDisconnect(error: nil)
        }
    }

    func send(_ payload: [String: SocketValue]) {
        guard let data = try? encoder.encode(payload), let text = String(data: data, encoding: .utf8) else { return }
        guard let task, opened, task.state == .running else {
            outbox.append(text)
            return
        }
        task.send(.string(text)) { [weak self] error in
            guard let error else { return }
            Task { @MainActor in
                self?.handleDisconnect(error: error.localizedDescription)
            }
        }
    }

    private func flushOutbox() {
        let messages = outbox
        outbox.removeAll()
        for text in messages {
            task?.send(.string(text)) { [weak self] error in
                guard let error else { return }
                Task { @MainActor in
                    self?.handleDisconnect(error: error.localizedDescription)
                }
            }
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self else { return }
            Task { @MainActor in
                switch result {
                case .success(let message):
                    if let event = self.decode(message) {
                        self.delegate?.socketClientDidReceive(event)
                    }
                    self.receiveLoop()
                case .failure(let error):
                    self.handleDisconnect(error: error.localizedDescription)
                }
            }
        }
    }

    private func handleOpen() {
        opened = true
        delegate?.socketClientDidConnect()
        flushOutbox()
    }

    private func handleDisconnect(error: String?) {
        opened = false
        task = nil
        delegate?.socketClientDidDisconnect(error: error)
    }

    private func decode(_ message: URLSessionWebSocketTask.Message) -> SocketEvent? {
        switch message {
        case .data(let data):
            return try? decoder.decode(SocketEvent.self, from: data)
        case .string(let string):
            return try? decoder.decode(SocketEvent.self, from: Data(string.utf8))
        @unknown default:
            return nil
        }
    }
}

extension ChatSocketClient: URLSessionWebSocketDelegate {
    nonisolated func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        Task { @MainActor in
            self.handleOpen()
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        let text = reason.flatMap { String(data: $0, encoding: .utf8) }
        Task { @MainActor in
            self.handleDisconnect(error: text)
        }
    }
}

enum SocketValue: Encodable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case image(ChatImageAttachment)

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        case .image(let value): try container.encode(value)
        }
    }
}

enum SocketEvent: Decodable {
    case rpcResult(requestId: String, data: JSONValue?, error: String?)
    case start(id: String, chatId: String?)
    case queued(id: String, position: Int)
    case chatCreated(id: String, chat: ChatSummary)
    case response(id: String, chatId: String?, modelId: String?, text: String, images: [ChatImageAttachment]?, tokenCount: Int?, tokensPerSecond: Double?)
    case ollamaChunk(id: String, text: String?, thinking: String?, images: [ChatImageAttachment]?)
    case ollamaDone(id: String, chatId: String?, modelId: String, text: String, thinking: String?, images: [ChatImageAttachment]?, evalCount: Int?)
    case llamaChunk(id: String, text: String?, thinking: String?)
    case llamaDone(id: String, chatId: String?, modelName: String, text: String, thinking: String?)
    case error(id: String?, error: String)
    case modelLoading(modelId: String?)
    case modelReady(modelId: String, isVLM: Bool?, canGenerateImages: Bool?)
    case modelError(modelId: String?, error: String, failed: [FailedModel]?)
    case gpuUsage(running: Bool, gpu: Double?, memory: Double?)
    case unknown

    private enum CodingKeys: String, CodingKey {
        case type, requestId, data, error, id, chatId, chat, position, modelId, text, images
        case tokenCount, tokensPerSecond, thinking, evalCount, modelName, isVLM, canGenerateImages
        case failed, running, gpu, memory
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decodeIfPresent(String.self, forKey: .type)
        switch type {
        case "rpcResult":
            self = .rpcResult(
                requestId: try container.decode(String.self, forKey: .requestId),
                data: try container.decodeIfPresent(JSONValue.self, forKey: .data),
                error: try container.decodeIfPresent(String.self, forKey: .error)
            )
        case "start":
            self = .start(id: try container.decode(String.self, forKey: .id), chatId: try container.decodeIfPresent(String.self, forKey: .chatId))
        case "queued":
            self = .queued(id: try container.decode(String.self, forKey: .id), position: try container.decode(Int.self, forKey: .position))
        case "chatCreated":
            self = .chatCreated(id: try container.decode(String.self, forKey: .id), chat: try container.decode(ChatSummary.self, forKey: .chat))
        case "response":
            self = .response(
                id: try container.decode(String.self, forKey: .id),
                chatId: try container.decodeIfPresent(String.self, forKey: .chatId),
                modelId: try container.decodeIfPresent(String.self, forKey: .modelId),
                text: try container.decodeIfPresent(String.self, forKey: .text) ?? "",
                images: try container.decodeIfPresent([ChatImageAttachment].self, forKey: .images),
                tokenCount: try container.decodeIfPresent(Int.self, forKey: .tokenCount),
                tokensPerSecond: try container.decodeIfPresent(Double.self, forKey: .tokensPerSecond)
            )
        case "ollamaChunk":
            self = .ollamaChunk(
                id: try container.decode(String.self, forKey: .id),
                text: try container.decodeIfPresent(String.self, forKey: .text),
                thinking: try container.decodeIfPresent(String.self, forKey: .thinking),
                images: try container.decodeIfPresent([ChatImageAttachment].self, forKey: .images)
            )
        case "ollamaDone":
            self = .ollamaDone(
                id: try container.decode(String.self, forKey: .id),
                chatId: try container.decodeIfPresent(String.self, forKey: .chatId),
                modelId: try container.decode(String.self, forKey: .modelId),
                text: try container.decodeIfPresent(String.self, forKey: .text) ?? "",
                thinking: try container.decodeIfPresent(String.self, forKey: .thinking),
                images: try container.decodeIfPresent([ChatImageAttachment].self, forKey: .images),
                evalCount: try container.decodeIfPresent(Int.self, forKey: .evalCount)
            )
        case "llamaChunk":
            self = .llamaChunk(id: try container.decode(String.self, forKey: .id), text: try container.decodeIfPresent(String.self, forKey: .text), thinking: try container.decodeIfPresent(String.self, forKey: .thinking))
        case "llamaDone":
            self = .llamaDone(id: try container.decode(String.self, forKey: .id), chatId: try container.decodeIfPresent(String.self, forKey: .chatId), modelName: try container.decode(String.self, forKey: .modelName), text: try container.decodeIfPresent(String.self, forKey: .text) ?? "", thinking: try container.decodeIfPresent(String.self, forKey: .thinking))
        case "error":
            self = .error(id: try container.decodeIfPresent(String.self, forKey: .id), error: try container.decodeIfPresent(String.self, forKey: .error) ?? "Unknown error.")
        case "modelLoading":
            self = .modelLoading(modelId: try container.decodeIfPresent(String.self, forKey: .modelId))
        case "modelReady":
            self = .modelReady(modelId: try container.decode(String.self, forKey: .modelId), isVLM: try container.decodeIfPresent(Bool.self, forKey: .isVLM), canGenerateImages: try container.decodeIfPresent(Bool.self, forKey: .canGenerateImages))
        case "modelError":
            self = .modelError(modelId: try container.decodeIfPresent(String.self, forKey: .modelId), error: try container.decodeIfPresent(String.self, forKey: .error) ?? "Model error.", failed: try container.decodeIfPresent([FailedModel].self, forKey: .failed))
        case "gpuUsage":
            self = .gpuUsage(running: try container.decode(Bool.self, forKey: .running), gpu: try container.decodeIfPresent(Double.self, forKey: .gpu), memory: try container.decodeIfPresent(Double.self, forKey: .memory))
        default:
            self = .unknown
        }
    }
}
