import Foundation

struct AuthClient {
    struct AuthResponse: Decodable {
        let authenticated: Bool?
        let user: AuthUser?
        let token: String?
        let error: String?
    }

    var serverURL: URL
    var session: URLSession = .shared

    func restoreSession() async throws -> AuthResponse {
        let (data, response) = try await session.data(for: request(path: "/api/auth/me"))
        try validate(response: response, data: data)
        return try JSONDecoder().decode(AuthResponse.self, from: data)
    }

    func login(username: String, password: String) async throws -> AuthResponse {
        var request = request(path: "/api/auth/login")
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["username": username, "password": password])

        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            let body = try? JSONDecoder().decode(AuthResponse.self, from: data)
            throw AuthError.message(body?.error ?? "Authentication failed.")
        }
        return try JSONDecoder().decode(AuthResponse.self, from: data)
    }

    func logout() async {
        var request = request(path: "/api/auth/logout")
        request.httpMethod = "POST"
        _ = try? await session.data(for: request)
    }

    private func request(path: String) -> URLRequest {
        var components = URLComponents(url: serverURL, resolvingAgainstBaseURL: false)
        components?.path = path
        return URLRequest(url: components?.url ?? serverURL)
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            let body = try? JSONDecoder().decode(AuthResponse.self, from: data)
            throw AuthError.message(body?.error ?? "Request failed with status \(http.statusCode).")
        }
    }
}

enum AuthError: LocalizedError {
    case message(String)

    var errorDescription: String? {
        switch self {
        case .message(let message): message
        }
    }
}
