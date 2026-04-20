import SwiftUI

struct RootView: View {
    @EnvironmentObject private var store: ChatStore

    var body: some View {
        Group {
            if store.isAuthenticated {
                ContentView()
            } else {
                LoginView()
            }
        }
        .frame(minWidth: 980, minHeight: 680)
    }
}
