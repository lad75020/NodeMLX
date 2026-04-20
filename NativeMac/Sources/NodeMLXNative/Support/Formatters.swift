import Foundation

enum DisplayFormatters {
    static func chatDate(_ value: String) -> String {
        let iso = ISO8601DateFormatter()
        guard let date = iso.date(from: value) else { return value }
        return date.formatted(date: .abbreviated, time: .shortened)
    }

    static func shortModelId(_ value: String) -> String {
        value
            .replacingOccurrences(of: "mlx-community/", with: "")
            .replacingOccurrences(of: ":latest", with: "")
    }

    static func bytes(_ value: Int) -> String {
        guard value > 0 else { return "" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .file)
    }
}
