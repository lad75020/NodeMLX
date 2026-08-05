import { createHmac, randomBytes, scrypt, timingSafeEqual } from "node:crypto";
import { createRequire } from "node:module";
import { promisify } from "node:util";

const require = createRequire(import.meta.url);
const Database = require("better-sqlite3");
const scryptAsync = promisify(scrypt);

/**
 * Owns the local model registry and all authentication persistence and tokens.
 * The HTTP layer consumes the returned statements and narrowly scoped helpers.
 */
export function createAuthStore({
  databasePath,
  sessionCookieName,
  sessionTtlMs,
  jwtSecret = process.env.JWT_SECRET,
}) {
  const db = new Database(databasePath);
  db.pragma("foreign_keys = ON");
  db.exec(`
    CREATE TABLE IF NOT EXISTS failed_models (
      model_id  TEXT PRIMARY KEY,
      error     TEXT NOT NULL,
      failed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );

    CREATE TABLE IF NOT EXISTS saved_models (
      model_id  TEXT PRIMARY KEY,
      is_vlm    INTEGER NOT NULL DEFAULT 0,
      last_used TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );

    CREATE TABLE IF NOT EXISTS users (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      username      TEXT NOT NULL COLLATE NOCASE UNIQUE,
      password_hash TEXT NOT NULL,
      created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      last_login_at TEXT
    );

    CREATE TABLE IF NOT EXISTS sessions (
      id           TEXT PRIMARY KEY,
      user_id      INTEGER NOT NULL,
      created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      expires_at   TEXT NOT NULL,
      last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

    CREATE TABLE IF NOT EXISTS app_secrets (
      name  TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  `);

  const stmt = {
    upsertFailed: db.prepare(`
      INSERT INTO failed_models (model_id, error)
      VALUES (?, ?)
      ON CONFLICT(model_id) DO UPDATE
        SET error = excluded.error,
            failed_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
    `),
    deleteFailed: db.prepare("DELETE FROM failed_models WHERE model_id = ?"),
    allFailed: db.prepare(
      "SELECT model_id AS id, error, failed_at AS failedAt FROM failed_models",
    ),
    isFailed: db.prepare("SELECT 1 FROM failed_models WHERE model_id = ?"),
    upsertSaved: db.prepare(`
      INSERT INTO saved_models (model_id, is_vlm)
      VALUES (?, ?)
      ON CONFLICT(model_id) DO UPDATE
        SET is_vlm    = excluded.is_vlm,
            last_used = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
    `),
    allSaved: db.prepare(
      "SELECT model_id AS id, is_vlm AS isVlm, last_used AS lastUsed FROM saved_models ORDER BY last_used DESC",
    ),
    createUser: db.prepare(`
      INSERT INTO users (username, password_hash)
      VALUES (?, ?)
    `),
    getUserForLogin: db.prepare(`
      SELECT id, username, password_hash AS passwordHash
      FROM users
      WHERE username = ?
    `),
    createSession: db.prepare(`
      INSERT INTO sessions (id, user_id, expires_at)
      VALUES (?, ?, ?)
    `),
    getSessionUser: db.prepare(`
      SELECT
        s.id AS sessionId,
        s.user_id AS userId,
        u.username AS username,
        s.expires_at AS expiresAt
      FROM sessions s
      JOIN users u ON u.id = s.user_id
      WHERE s.id = ?
        AND datetime(s.expires_at) > datetime('now')
    `),
    touchSession: db.prepare(`
      UPDATE sessions
      SET last_seen_at = ?
      WHERE id = ?
    `),
    deleteSession: db.prepare("DELETE FROM sessions WHERE id = ?"),
    deleteExpiredSessions: db.prepare(`
      DELETE FROM sessions
      WHERE datetime(expires_at) <= datetime('now')
    `),
    getAppSecret: db.prepare("SELECT value FROM app_secrets WHERE name = ?"),
    setAppSecret: db.prepare(`
      INSERT INTO app_secrets (name, value)
      VALUES (?, ?)
      ON CONFLICT(name) DO UPDATE SET value = excluded.value
    `),
    updateUserLastLogin: db.prepare(`
      UPDATE users
      SET last_login_at = ?
      WHERE id = ?
    `),
  };
  stmt.deleteExpiredSessions.run();

  let secret = jwtSecret;
  if (!secret) {
    secret = stmt.getAppSecret.get("jwt_secret")?.value;
    if (!secret) {
      secret = randomBytes(48).toString("base64url");
      stmt.setAppSecret.run("jwt_secret", secret);
    }
  }

  function signJwt(headerB64, payloadB64) {
    return createHmac("sha256", secret)
      .update(`${headerB64}.${payloadB64}`)
      .digest("base64url");
  }

  function readSessionUser(request) {
    stmt.deleteExpiredSessions.run();
    const sessionId = sessionCookieValue(request);
    if (!sessionId) return null;
    const session = stmt.getSessionUser.get(sessionId);
    if (!session) return null;
    stmt.touchSession.run(new Date().toISOString(), sessionId);
    return session;
  }

  function sessionCookieValue(request) {
    const cookies = parseCookies(request.headers.cookie);
    const raw = cookies[sessionCookieName];
    if (typeof raw !== "string" || raw.length === 0) return null;
    return raw;
  }

  return {
    stmt,
    normalizeUsername(value) {
      return typeof value === "string" ? value.trim() : "";
    },
    async verifyPassword(password, storedHash) {
      if (typeof storedHash !== "string") return false;
      const [salt, expectedHex] = storedHash.split(":");
      if (!salt || !expectedHex) return false;
      const expected = Buffer.from(expectedHex, "hex");
      if (expected.length === 0) return false;
      const actual = Buffer.from(
        await scryptAsync(password, salt, expected.length),
      );
      return (
        actual.length === expected.length && timingSafeEqual(actual, expected)
      );
    },
    createSessionForUser(userId) {
      const sessionId = randomBytes(32).toString("hex");
      const expiresAt = new Date(Date.now() + sessionTtlMs).toISOString();
      stmt.createSession.run(sessionId, userId, expiresAt);
      return { sessionId, expiresAt };
    },
    readSessionUser,
    createSessionJwt(session) {
      const now = Math.floor(Date.now() / 1000);
      const exp = Math.floor(new Date(session.expiresAt).getTime() / 1000);
      const header = Buffer.from(
        JSON.stringify({ alg: "HS256", typ: "JWT" }),
      ).toString("base64url");
      const payload = Buffer.from(
        JSON.stringify({
          sub: String(session.userId),
          sid: session.sessionId,
          username: session.username,
          iat: now,
          exp,
        }),
      ).toString("base64url");
      return `${header}.${payload}.${signJwt(header, payload)}`;
    },
    verifySessionJwt(token) {
      if (typeof token !== "string" || token.length > 4096) return null;
      const parts = token.split(".");
      if (parts.length !== 3) return null;
      const [headerB64, payloadB64, signature] = parts;
      let header;
      try {
        header = JSON.parse(
          Buffer.from(headerB64, "base64url").toString("utf8"),
        );
      } catch {
        return null;
      }
      if (!header || header.alg !== "HS256" || header.typ !== "JWT")
        return null;
      const actual = Buffer.from(signature);
      const expected = Buffer.from(signJwt(headerB64, payloadB64));
      if (
        actual.length !== expected.length ||
        !timingSafeEqual(actual, expected)
      )
        return null;
      let payload;
      try {
        payload = JSON.parse(
          Buffer.from(payloadB64, "base64url").toString("utf8"),
        );
      } catch {
        return null;
      }
      if (
        !payload ||
        typeof payload.sid !== "string" ||
        typeof payload.exp !== "number"
      )
        return null;
      if (payload.exp <= Math.floor(Date.now() / 1000)) return null;
      const session = getSessionUser(payload.sid);
      return session && String(session.userId) === String(payload.sub)
        ? session
        : null;
    },
    websocketTokenFromRequest(request) {
      for (const requestUrl of [request.url, request.raw?.url]) {
        try {
          const token = new URL(
            requestUrl,
            "http://localhost",
          ).searchParams.get("token");
          if (token && token.trim()) return token.trim();
        } catch {}
      }
      return null;
    },
    sessionCookieValue,
    setSessionCookie(reply, request, sessionId) {
      reply.header(
        "Set-Cookie",
        serializeCookie(sessionCookieName, sessionId, {
          path: "/",
          sameSite: "Lax",
          httpOnly: true,
          secure: isSecureRequest(request),
          maxAge: Math.floor(sessionTtlMs / 1000),
        }),
      );
    },
    clearSessionCookie(reply, request) {
      reply.header(
        "Set-Cookie",
        serializeCookie(sessionCookieName, "", {
          path: "/",
          sameSite: "Lax",
          httpOnly: true,
          secure: isSecureRequest(request),
          maxAge: 0,
        }),
      );
    },
  };

  function getSessionUser(sessionId) {
    stmt.deleteExpiredSessions.run();
    const session = stmt.getSessionUser.get(sessionId);
    if (!session) return null;
    stmt.touchSession.run(new Date().toISOString(), sessionId);
    return session;
  }
}

function parseCookies(cookieHeader) {
  const out = {};
  if (typeof cookieHeader !== "string" || cookieHeader.length === 0) return out;
  for (const part of cookieHeader.split(";")) {
    const eq = part.indexOf("=");
    if (eq <= 0) continue;
    const key = part.slice(0, eq).trim();
    const value = part.slice(eq + 1).trim();
    if (!key) continue;
    try {
      out[key] = decodeURIComponent(value);
    } catch {
      out[key] = value;
    }
  }
  return out;
}

function serializeCookie(name, value, options = {}) {
  const pairs = [`${name}=${encodeURIComponent(value)}`];
  pairs.push(`Path=${options.path ?? "/"}`);
  pairs.push(`SameSite=${options.sameSite ?? "Lax"}`);
  if (typeof options.maxAge === "number")
    pairs.push(`Max-Age=${Math.max(0, Math.trunc(options.maxAge))}`);
  if (options.httpOnly !== false) pairs.push("HttpOnly");
  if (options.secure === true) pairs.push("Secure");
  return pairs.join("; ");
}

function isSecureRequest(request) {
  if (request.protocol === "https") return true;
  const forwardedProto = request.headers["x-forwarded-proto"];
  return (
    typeof forwardedProto === "string" &&
    forwardedProto.split(",")[0].trim() === "https"
  );
}
