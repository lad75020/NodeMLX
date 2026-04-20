import AppKit
import Foundation

extension ChatImageAttachment {
    var nsImage: NSImage? {
        guard dataUrl.hasPrefix("data:image/"),
              let comma = dataUrl.firstIndex(of: ",")
        else { return nil }
        let base64 = String(dataUrl[dataUrl.index(after: comma)...])
        guard let data = Data(base64Encoded: base64) else { return nil }
        return NSImage(data: data)
    }
}

extension URL {
    func imageAttachment() throws -> ChatImageAttachment {
        let data = try Data(contentsOf: self)
        let type = mimeTypeForImageURL(self)
        let base64 = data.base64EncodedString()
        return ChatImageAttachment(
            dataUrl: "data:\(type);base64,\(base64)",
            name: lastPathComponent,
            type: type,
            size: data.count
        )
    }
}

func mimeTypeForImageURL(_ url: URL) -> String {
    switch url.pathExtension.lowercased() {
    case "jpg", "jpeg": "image/jpeg"
    case "png": "image/png"
    case "webp": "image/webp"
    case "gif": "image/gif"
    default: "image/png"
    }
}

extension String {
    func extractOllamaImages() -> (text: String, images: [ChatImageAttachment]) {
        var cleaned = self
        var images: [ChatImageAttachment] = []

        let patterns = [
            #"!\[[^\]]*\]\(([^)\s]+)(?:\s+["'][^"']*["'])?\)"#,
            #"<img\b[^>]*src=["']([^"']+)["'][^>]*>"#,
            #"data:image/(?:png|jpeg|jpg|gif|webp|svg\+xml);base64,[A-Za-z0-9+/=]+"#,
            #"https?://[^\s<>"')]+\.(?:png|jpe?g|gif|webp|svg)(?:\?[^\s<>"')]*)?"#
        ]

        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { continue }
            let source = cleaned as NSString
            let matches = regex.matches(in: cleaned, range: NSRange(location: 0, length: source.length))
            for match in matches.reversed() {
                let srcRange = match.numberOfRanges > 1 ? match.range(at: 1) : match.range(at: 0)
                guard srcRange.location != NSNotFound else { continue }
                let src = source.substring(with: srcRange)
                addDetectedImage(src, to: &images)
                cleaned = (cleaned as NSString).replacingCharacters(in: match.range, with: "")
            }
        }

        cleaned = cleaned
            .replacingOccurrences(of: #"[ \t]+\n"#, with: "\n", options: .regularExpression)
            .replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        return (cleaned, images)
    }

    private func addDetectedImage(_ source: String, to images: inout [ChatImageAttachment]) {
        let src = source.trimmingCharacters(in: CharacterSet(charactersIn: "<> \n\t"))
        guard src.hasPrefix("data:image/") || src.hasPrefix("http://") || src.hasPrefix("https://") else { return }
        guard !images.contains(where: { $0.dataUrl == src }) else { return }
        images.append(ChatImageAttachment(dataUrl: src, name: imageName(src), type: imageType(src), size: 0))
    }

    private func imageName(_ src: String) -> String {
        guard let url = URL(string: src), !src.hasPrefix("data:image/") else { return "Ollama image" }
        return url.lastPathComponent.isEmpty ? "Ollama image" : url.lastPathComponent
    }

    private func imageType(_ src: String) -> String {
        if let match = src.range(of: #"^data:(image/[^;]+);"#, options: .regularExpression) {
            return String(src[match]).replacingOccurrences(of: "data:", with: "").replacingOccurrences(of: ";", with: "")
        }
        guard let url = URL(string: src) else { return "image" }
        return mimeTypeForImageURL(url)
    }
}
