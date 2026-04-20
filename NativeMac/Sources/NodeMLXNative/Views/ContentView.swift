import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: ChatStore
    @SceneStorage("selectedChatId") private var selectedChatId: String?
    @State private var columnVisibility: NavigationSplitViewVisibility = .all

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            SidebarView(selection: $selectedChatId)
                .navigationSplitViewColumnWidth(min: 220, ideal: 280, max: 360)
        } detail: {
            ChatDetailView(columnVisibility: $columnVisibility)
        }
        .onChange(of: selectedChatId) { _, newValue in
            guard let newValue, newValue != store.currentChatId else { return }
            Task { await store.openChat(newValue) }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button {
                    columnVisibility = columnVisibility == .all ? .detailOnly : .all
                } label: {
                    Image(systemName: "sidebar.left")
                }
                .help(columnVisibility == .all ? "Hide sidebar" : "Show sidebar")
            }
        }
    }
}
