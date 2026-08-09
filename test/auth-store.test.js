import assert from "node:assert/strict";
import { randomBytes, scrypt } from "node:crypto";
import test from "node:test";
import { promisify } from "node:util";

import { createAuthStore } from "../src/auth/sqlite-store.js";

const scryptAsync = promisify(scrypt);

async function passwordHash(password) {
  const salt = randomBytes(16).toString("hex");
  const derived = await scryptAsync(password, salt, 64);
  return `${salt}:${Buffer.from(derived).toString("hex")}`;
}

async function createStore() {
  const store = createAuthStore({
    databasePath: ":memory:",
    sessionCookieName: "test_session",
    sessionTtlMs: 7 * 24 * 60 * 60 * 1000,
    jwtSecret: "test-secret",
  });
  const hash = await passwordHash("initial-password");
  const userId = Number(store.stmt.createUser.run("invited-user", hash).lastInsertRowid);
  return { store, userId };
}

test("valid session activity renews the seven-day inactivity deadline", async () => {
  const { store, userId } = await createStore();
  const session = store.createSessionForUser(userId);
  const before = store.stmt.getSessionUser.get(session.sessionId);

  const active = store.readSessionUser({
    headers: { cookie: `test_session=${session.sessionId}` },
  });

  assert.equal(active?.userId, userId);
  assert.ok(new Date(active.expiresAt) >= new Date(before.expiresAt));
  assert.equal(store.stmt.getSessionUser.get(session.sessionId)?.expiresAt, active.expiresAt);
});

test("expired sessions fail closed and cannot be renewed", async () => {
  const { store, userId } = await createStore();
  const session = store.createSessionForUser(userId);
  store.stmt.createSession.run(
    "expired-session",
    userId,
    "2000-01-01T00:00:00.000Z",
  );

  assert.equal(store.readSessionUserById("expired-session"), null);
  assert.equal(store.stmt.getSessionUser.get("expired-session"), undefined);
  assert.ok(store.readSessionUserById(session.sessionId));
});

test("password reset replaces the password and atomically invalidates all sessions", async () => {
  const { store, userId } = await createStore();
  const first = store.createSessionForUser(userId);
  const second = store.createSessionForUser(userId);
  const oldHash = store.stmt.getUserForLogin.get("invited-user").passwordHash;
  const nextHash = await passwordHash("replacement-password");

  assert.equal(store.resetPasswordAndInvalidateSessions(userId, nextHash), true);
  assert.equal(store.readSessionUserById(first.sessionId), null);
  assert.equal(store.readSessionUserById(second.sessionId), null);
  assert.equal(store.stmt.getUserForLogin.get("invited-user").passwordHash, nextHash);
  assert.equal(await store.verifyPassword("initial-password", oldHash), true);
  assert.equal(await store.verifyPassword("initial-password", nextHash), false);
  assert.equal(await store.verifyPassword("replacement-password", nextHash), true);
});

test("a revoked session also invalidates its previously issued WebSocket token", async () => {
  const { store, userId } = await createStore();
  const created = store.createSessionForUser(userId);
  const session = {
    sessionId: created.sessionId,
    userId,
    username: "invited-user",
    expiresAt: created.expiresAt,
  };
  const token = store.createSessionJwt(session);

  assert.equal(store.verifySessionJwt(token)?.userId, userId);
  assert.equal(store.resetPasswordAndInvalidateSessions(userId, await passwordHash("new-password")), true);
  assert.equal(store.verifySessionJwt(token), null);
});
