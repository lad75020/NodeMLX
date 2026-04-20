import SwiftUI

struct MessageListView: View {
    @EnvironmentObject private var store: ChatStore

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 14) {
                    if store.messages.isEmpty {
                        ContentUnavailableView("Start a chat", systemImage: "bubble.left", description: Text("Choose a backend and model, then send a prompt."))
                            .padding(.top, 80)
                    }

                    ForEach(store.messages) { message in
                        MessageRowView(message: message)
                            .id(message.id)
                    }
                }
                .padding(18)
                .frame(maxWidth: 980)
                .frame(maxWidth: .infinity)
            }
            .onChange(of: store.messages) { _, messages in
                guard let last = messages.last else { return }
                withAnimation {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }
}

struct MessageRowView: View {
    @EnvironmentObject private var store: ChatStore
    let message: ChatMessage
    @State private var thinkingExpanded = false

    var body: some View {
        HStack(alignment: .top) {
            if message.role == .user { Spacer(minLength: 80) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 8) {
                if let image = message.image {
                    AttachmentImageView(image: image)
                }

                if let images = message.images, !images.isEmpty {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 8)], spacing: 8) {
                        ForEach(images) { image in
                            AttachmentImageView(image: image)
                        }
                    }
                }

                if let thinking = message.thinking, !thinking.isEmpty {
                    DisclosureGroup("Thinking", isExpanded: $thinkingExpanded) {
                        Text(thinking)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .disabled(message.pending == true)
                    .padding(8)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }

                if !message.text.isEmpty || message.pending == true {
                    messageText
                        .padding(.horizontal, 12)
                        .padding(.vertical, 9)
                        .background(message.role == .user ? Color.accentColor.opacity(0.18) : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
                        .textSelection(.enabled)
                        .contextMenu {
                            Button("Copy") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(message.text, forType: .string)
                            }
                            if message.pending != true {
                                Button("Delete", role: .destructive) {
                                    Task { await store.deleteMessage(message.id) }
                                }
                            }
                        }
                }

                if let meta {
                    Text(meta)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: 720, alignment: message.role == .user ? .trailing : .leading)

            if message.role == .assistant { Spacer(minLength: 80) }
        }
    }

    @ViewBuilder
    private var messageText: some View {
        if message.pending == true, message.text.isEmpty {
            HStack(spacing: 8) {
                ProgressView()
                    .controlSize(.small)
                Text(message.queued == true ? "queue" : "thinking...")
            }
            .foregroundStyle(.secondary)
        } else if message.role == .assistant, let attributed = try? AttributedString(markdown: message.text) {
            Text(attributed)
        } else {
            Text(message.text)
        }
    }

    private var meta: String? {
        guard message.role == .assistant else { return nil }
        if let count = message.images?.count, count > 0 {
            return "\(count) image\(count == 1 ? "" : "s")" + (message.modelId.map { " · \($0)" } ?? "")
        }
        if let tps = message.tokensPerSecond {
            return "\(message.tokenCount ?? 0) tokens · \(String(format: "%.1f", tps)) tok/s" + (message.modelId.map { " · \($0)" } ?? "")
        }
        if let modelId = message.modelId, message.provider == .ollama || message.provider == .llamacpp {
            return modelId
        }
        return nil
    }
}

struct AttachmentImageView: View {
    let image: ChatImageAttachment

    var body: some View {
        Group {
            if let nsImage = image.nsImage {
                Image(nsImage: nsImage)
                    .resizable()
                    .scaledToFit()
            } else if let url = URL(string: image.dataUrl) {
                AsyncImage(url: url) { phase in
                    switch phase {
                    case .success(let image):
                        image.resizable().scaledToFit()
                    case .failure:
                        Image(systemName: "photo")
                    default:
                        ProgressView()
                    }
                }
            } else {
                Image(systemName: "photo")
            }
        }
        .frame(maxHeight: 280)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
