import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var store: ChatStore

    var body: some View {
        Form {
            TextField("Server URL", text: $store.serverURLString)
            Button("Save and Reconnect") {
                store.saveServerURL()
                store.connectSocket()
            }
        }
        .padding(20)
        .frame(width: 420)
    }
}
