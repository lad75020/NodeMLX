import SwiftUI

struct SidebarView: View {
    @EnvironmentObject private var store: ChatStore
    @Binding var selection: String?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                StatusBadge()
                Spacer()
                Button {
                    Task { await store.logout() }
                } label: {
                    Image(systemName: "rectangle.portrait.and.arrow.right")
                }
                .buttonStyle(.borderless)
                .help("Sign out")
            }
            .padding([.horizontal, .top], 12)
            .padding(.bottom, 8)

            if let error = store.connectionError, !store.connected {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
            }

            Button {
                store.newChat()
                selection = nil
            } label: {
                Label("New Chat", systemImage: "plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 12)
            .padding(.bottom, 8)

            List(selection: $selection) {
                if store.chats.isEmpty {
                    Text("No previous chats")
                        .foregroundStyle(.secondary)
                }
                ForEach(store.chats) { chat in
                    VStack(alignment: .leading, spacing: 3) {
                        Text(DisplayFormatters.chatDate(chat.startedAt))
                            .lineLimit(1)
                        if let title = chat.title {
                            Text(title)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .tag(chat.id)
                    .contextMenu {
                        Button("Delete", role: .destructive) {
                            Task { await store.deleteChat(chat.id) }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
        }
    }
}

private struct StatusBadge: View {
    @EnvironmentObject private var store: ChatStore

    var body: some View {
        Label(store.connected ? "Connected" : "Disconnected", systemImage: store.connected ? "checkmark.circle.fill" : "xmark.circle")
            .font(.caption.weight(.semibold))
            .foregroundStyle(store.connected ? .green : .secondary)
    }
}
