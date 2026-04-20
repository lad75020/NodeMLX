import Foundation

enum Role: String, Codable {
    case user
    case assistant
}

enum InferenceMode: String, Codable, CaseIterable, Identifiable {
    case mlx
    case ollama
    case llamacpp

    var id: String { rawValue }

    var label: String {
        switch self {
        case .mlx: "MLX"
        case .ollama: "Ollama"
        case .llamacpp: "Llama.cpp"
        }
    }
}

enum LlamaModelSource: String, Codable, CaseIterable, Identifiable {
    case disk
    case huggingface

    var id: String { rawValue }
}

struct AuthUser: Codable, Equatable {
    let id: Int
    let username: String
}

struct ChatImageAttachment: Codable, Hashable, Identifiable {
    var id: String { dataUrl }
    let dataUrl: String
    let name: String
    let type: String
    let size: Int
}

struct ChatMessage: Codable, Identifiable, Equatable {
    let id: String
    var role: Role
    var text: String
    var thinking: String? = nil
    var provider: InferenceMode? = nil
    var image: ChatImageAttachment? = nil
    var images: [ChatImageAttachment]? = nil
    var pending: Bool? = nil
    var queued: Bool? = nil
    var queuePosition: Int? = nil
    var tokensPerSecond: Double? = nil
    var tokenCount: Int? = nil
    var modelId: String? = nil
}

struct ChatSummary: Codable, Identifiable, Equatable {
    let id: String
    let startedAt: String
    let title: String?
}

struct HFModel: Codable, Identifiable, Hashable {
    let id: String
    let downloads: Int
    let likes: Int
    let updatedAt: String?
    let saved: Bool?
}

struct FailedModel: Codable, Hashable {
    let id: String
    let error: String
    let failedAt: String?
}

struct OllamaModel: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let modifiedAt: String?
    let size: Int
    let digest: String?
}

struct OllamaModelDetails: Codable {
    let id: String?
    let capabilities: [String]
    let modifiedAt: String?
}

struct LlamaModelFile: Codable, Hashable {
    let path: String
    let name: String
    let size: Int
}

struct GpuUsage: Equatable {
    let gpu: Double
    let memory: Double
}

struct GenerationOptions {
    var maxTokens: Int?
    var imageWidth: Int?
    var imageHeight: Int?
    var steps: Int?
    var seed: Int?
}
