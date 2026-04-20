import SwiftUI

struct LoginView: View {
    @EnvironmentObject private var store: ChatStore
    @State private var username = ""
    @State private var password = ""

    var body: some View {
        VStack(spacing: 18) {
            VStack(spacing: 6) {
                Image(systemName: "bubble.left.and.bubble.right.fill")
                    .font(.system(size: 42))
                    .foregroundStyle(.tint)
                Text("Chat with AI")
                    .font(.title2.weight(.semibold))
                Text("Sign in with your invited account to continue.")
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Server")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("http://127.0.0.1:3000", text: $store.serverURLString)
                    .textFieldStyle(.roundedBorder)

                Text("Username")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("Username", text: $username)
                    .textFieldStyle(.roundedBorder)

                Text("Password")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                SecureField("Password", text: $password)
                    .textFieldStyle(.roundedBorder)

                if let error = store.authError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                Button {
                    Task { await store.login(username: username, password: password) }
                } label: {
                    HStack {
                        if store.authLoading {
                            ProgressView()
                                .controlSize(.small)
                        }
                        Text("Sign In")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(store.authLoading || username.isEmpty || password.isEmpty)
                .keyboardShortcut(.defaultAction)
            }
            .frame(width: 360)
        }
        .padding(32)
    }
}
