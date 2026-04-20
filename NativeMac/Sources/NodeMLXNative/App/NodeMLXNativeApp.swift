import SwiftUI

@main
struct NodeMLXNativeApp: App {
    @StateObject private var store = ChatStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(store)
                .task {
                    await store.restoreSession()
                }
        }
        .commands {
            CommandGroup(after: .newItem) {
                Button("New Chat") {
                    store.newChat()
                }
                .keyboardShortcut("n", modifiers: [.command])

                Button("Cancel Inference") {
                    store.cancelInference()
                }
                .keyboardShortcut(".", modifiers: [.command])
                .disabled(!store.inferenceRunning)
            }
        }

        Settings {
            SettingsView()
                .environmentObject(store)
        }
    }
}
