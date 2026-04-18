#!/usr/bin/env node

import { randomBytes, scrypt } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createRequire } from "node:module";
import { promisify } from "node:util";
import { stdin as input, stdout as output, argv, exit } from "node:process";
import { createInterface } from "node:readline/promises";

const require = createRequire(import.meta.url);
const Database = require("better-sqlite3");

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEFAULT_DB_PATH = resolve(__dirname, "..", "mlx-chat.db");

const MIN_USERNAME_LENGTH = 3;
const MAX_USERNAME_LENGTH = 40;
const MIN_PASSWORD_LENGTH = 8;
const scryptAsync = promisify(scrypt);

function usage() {
  console.log(
    [
      "Usage:",
      "  node scripts/onboard-user.js <username> [--password <password>] [--db <path>]",
      "",
      "Options:",
      "  -p, --password   Password for the new user. If omitted, you'll be prompted securely.",
      "  --db             Path to sqlite database (default: ./mlx-chat.db)",
      "  -h, --help       Show this help.",
    ].join("\n")
  );
}

function parseArgs(rawArgs) {
  const options = {
    username: null,
    password: null,
    dbPath: DEFAULT_DB_PATH,
  };

  for (let i = 0; i < rawArgs.length; i += 1) {
    const token = rawArgs[i];
    if (token === "-h" || token === "--help") {
      usage();
      exit(0);
    }
    if (token === "-p" || token === "--password") {
      const next = rawArgs[i + 1];
      if (!next) {
        throw new Error("Missing value for --password.");
      }
      options.password = next;
      i += 1;
      continue;
    }
    if (token === "--db") {
      const next = rawArgs[i + 1];
      if (!next) {
        throw new Error("Missing value for --db.");
      }
      options.dbPath = resolve(next);
      i += 1;
      continue;
    }
    if (token.startsWith("-")) {
      throw new Error(`Unknown option: ${token}`);
    }
    if (options.username) {
      throw new Error("Only one username can be provided.");
    }
    options.username = token;
  }

  return options;
}

function normalizeUsername(value) {
  if (typeof value !== "string") return "";
  return value.trim();
}

function validateUsername(username) {
  if (username.length < MIN_USERNAME_LENGTH || username.length > MAX_USERNAME_LENGTH) {
    throw new Error(`Username must be ${MIN_USERNAME_LENGTH}-${MAX_USERNAME_LENGTH} characters.`);
  }
  if (!/^[A-Za-z0-9._-]+$/.test(username)) {
    throw new Error("Username may contain only letters, numbers, dot, underscore, and hyphen.");
  }
}

function validatePassword(password) {
  if (typeof password !== "string" || password.length < MIN_PASSWORD_LENGTH) {
    throw new Error(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
  }
}

async function hashPassword(password) {
  const salt = randomBytes(16).toString("hex");
  const derived = await scryptAsync(password, salt, 64);
  return `${salt}:${Buffer.from(derived).toString("hex")}`;
}

async function promptHidden(rl, label) {
  const original = rl._writeToOutput;
  rl._writeToOutput = (chunk) => {
    if (typeof chunk === "string" && chunk.startsWith(label)) {
      rl.output.write(chunk);
      return;
    }
    rl.output.write("*");
  };
  try {
    const value = await rl.question(label);
    rl.output.write("\n");
    return value;
  } finally {
    rl._writeToOutput = original;
  }
}

async function main() {
  const options = parseArgs(argv.slice(2));
  const rl = createInterface({ input, output, terminal: true });

  try {
    const usernameInput =
      options.username ??
      (await rl.question("Username: "));
    const username = normalizeUsername(usernameInput);
    validateUsername(username);

    let password = options.password;
    if (!password) {
      if (!input.isTTY || !output.isTTY) {
        throw new Error("Password is required in non-interactive mode. Use --password.");
      }
      password = await promptHidden(rl, "Password: ");
      const confirmation = await promptHidden(rl, "Confirm password: ");
      if (password !== confirmation) {
        throw new Error("Passwords do not match.");
      }
    }
    validatePassword(password);

    const db = new Database(options.dbPath);
    db.pragma("foreign_keys = ON");
    db.exec(`
      CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT NOT NULL COLLATE NOCASE UNIQUE,
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        last_login_at TEXT
      );
    `);

    const findUser = db.prepare("SELECT id FROM users WHERE username = ?");
    const createUser = db.prepare(`
      INSERT INTO users (username, password_hash)
      VALUES (?, ?)
    `);

    const existing = findUser.get(username);
    if (existing) {
      throw new Error(`User '${username}' already exists.`);
    }

    const passwordHash = await hashPassword(password);
    const result = createUser.run(username, passwordHash);

    console.log(
      `Created user '${username}' (id=${result.lastInsertRowid}) in ${options.dbPath}`
    );
    console.log("Registration is disabled, so this account can now sign in directly.");
  } finally {
    rl.close();
  }
}

main().catch((err) => {
  console.error(`Error: ${err instanceof Error ? err.message : String(err)}`);
  console.error("Run with --help for usage.");
  exit(1);
});
