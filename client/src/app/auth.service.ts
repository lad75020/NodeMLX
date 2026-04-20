import { Injectable, computed, signal } from "@angular/core";

interface AuthUser {
  id: number;
  username: string;
}

interface AuthPayload {
  authenticated?: boolean;
  user?: AuthUser;
  token?: string;
  error?: string;
}

const WS_TOKEN_KEY = "nodemlx_ws_token";

@Injectable({ providedIn: "root" })
export class AuthService {
  readonly user = signal<AuthUser | null>(null);
  readonly websocketToken = signal<string | null>(sessionStorage.getItem(WS_TOKEN_KEY));
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly isAuthenticated = computed(() => this.user() !== null);

  async restoreSession(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const res = await fetch("/api/auth/me", { credentials: "same-origin" });
      const body = (await res.json()) as AuthPayload;
      if (!res.ok || body.authenticated !== true || !body.user) {
        this.user.set(null);
        this.setWebsocketToken(null);
        return;
      }
      this.user.set(body.user);
      this.setWebsocketToken(body.token ?? null);
    } catch {
      this.user.set(null);
      this.setWebsocketToken(null);
    } finally {
      this.loading.set(false);
    }
  }

  async login(username: string, password: string): Promise<boolean> {
    return this.submitCredentials(username, password);
  }

  async logout(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        credentials: "same-origin",
      });
    } finally {
      this.user.set(null);
      this.setWebsocketToken(null);
      this.loading.set(false);
    }
  }

  private async submitCredentials(username: string, password: string): Promise<boolean> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ username, password }),
      });

      const body = (await res.json().catch(() => ({}))) as AuthPayload;
      if (!res.ok || !body.user) {
        this.user.set(null);
        this.error.set(body.error ?? "Authentication failed.");
        return false;
      }

      this.user.set(body.user);
      this.setWebsocketToken(body.token ?? null);
      this.error.set(null);
      return true;
    } catch {
      this.user.set(null);
      this.setWebsocketToken(null);
      this.error.set("Could not reach the server.");
      return false;
    } finally {
      this.loading.set(false);
    }
  }

  private setWebsocketToken(token: string | null): void {
    const value = token?.trim() || null;
    this.websocketToken.set(value);
    if (value) sessionStorage.setItem(WS_TOKEN_KEY, value);
    else sessionStorage.removeItem(WS_TOKEN_KEY);
  }
}
