import { existsSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

const DEFAULT_PORT = 18956;
const DEFAULT_HOST = "127.0.0.1";
const MAX_GENERATION_TOKEN_LIMIT = 131072;

export class ConfigurationError extends Error {
  constructor(setting, message) {
    super(`${setting}: ${message}`);
    this.name = "ConfigurationError";
    this.setting = setting;
  }
}

export function ensureHuggingFaceHome() {
  if (!process.env.HF_HOME) {
    process.env.HF_HOME = join(homedir(), ".cache", "huggingface");
  }
  console.log(`HF_HOME → ${process.env.HF_HOME}`);
}

export function clampInteger(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

export function optionalClampedInteger(value, min, max, multiple = 1) {
  if (!Number.isFinite(value)) return undefined;
  const clamped = Math.min(max, Math.max(min, Math.trunc(value)));
  if (multiple <= 1) return clamped;
  return Math.min(max, Math.max(min, Math.round(clamped / multiple) * multiple));
}

function readRequiredInteger(env, setting, { defaultValue, min, max, warnings }) {
  const raw = env[setting];
  if (raw === undefined) return defaultValue;
  if (typeof raw !== "string" || !/^-?\d+$/.test(raw.trim())) {
    throw new ConfigurationError(
      setting,
      `must be a whole number from ${min} to ${max}; correct ${setting} and restart.`,
    );
  }

  const value = Number(raw);
  if (!Number.isSafeInteger(value)) {
    throw new ConfigurationError(
      setting,
      `must be a safe whole number from ${min} to ${max}; correct ${setting} and restart.`,
    );
  }
  const effective = Math.min(max, Math.max(min, value));
  if (effective !== value) {
    warnings.push(
      `${setting}=${raw} is outside the supported range ${min}-${max}; using ${effective}.`,
    );
  }
  return effective;
}

function readPort(env) {
  if (env.PORT === undefined) return DEFAULT_PORT;
  const raw = env.PORT;
  if (typeof raw !== "string" || !/^\d+$/.test(raw.trim())) {
    throw new ConfigurationError(
      "PORT",
      "must be a whole number from 1 to 65535; correct PORT and restart.",
    );
  }
  const port = Number(raw);
  if (!Number.isSafeInteger(port) || port < 1 || port > 65535) {
    throw new ConfigurationError(
      "PORT",
      "must be a whole number from 1 to 65535; correct PORT and restart.",
    );
  }
  return port;
}

function readHost(env) {
  if (env.HOST === undefined) return DEFAULT_HOST;
  if (typeof env.HOST !== "string" || env.HOST.trim() === "") {
    throw new ConfigurationError(
      "HOST",
      "must be a non-empty host or IP address; use 127.0.0.1 for local-only binding.",
    );
  }
  return env.HOST.trim();
}

function optionalRuntimeReadiness(env, setting, capability, remediation, pathExists) {
  const configuredPath = env[setting];
  if (configuredPath === undefined || configuredPath === "") {
    return { state: "not-configured", capability, path: null, diagnostic: null, remediation };
  }
  if (typeof configuredPath !== "string") {
    return {
      state: "unavailable",
      capability,
      path: null,
      diagnostic: `${setting} must be an interpreter path or environment directory. ${remediation}`,
      remediation,
    };
  }
  const path = configuredPath.trim();
  if (path && pathExists(path)) {
    return { state: "available", capability, path, diagnostic: null, remediation };
  }
  return {
    state: "unavailable",
    capability,
    path,
    diagnostic: `${capability} is disabled because ${setting} does not exist at the configured location. ${remediation}`,
    remediation,
  };
}

/**
 * Create the complete startup configuration before Fastify is assembled.
 * Invalid operator input throws ConfigurationError; finite resource values
 * outside supported limits are clamped and recorded as warnings.
 */
export function createConfig(recommendedModels, env = process.env, { pathExists = existsSync } = {}) {
  const warnings = [];
  const maxGenerationTokens = readRequiredInteger(env, "MLX_MAX_TOKENS_LIMIT", {
    defaultValue: 32768,
    min: 1,
    max: MAX_GENERATION_TOKEN_LIMIT,
    warnings,
  });
  const defaultMaxTokens = readRequiredInteger(env, "MLX_MAX_TOKENS", {
    defaultValue: Math.min(4096, maxGenerationTokens),
    min: 1,
    max: maxGenerationTokens,
    warnings,
  });

  return {
    defaultModel: env.MLX_MODEL ?? recommendedModels["qwen-3-1.7b"],
    port: readPort(env),
    host: readHost(env),
    warnings,
    optionalCapabilities: {
      diffusionKit: optionalRuntimeReadiness(
        env,
        "DIFFUSIONKIT_PYTHON",
        "DiffusionKit image generation",
        "Install DiffusionKit in its separate environment, then set DIFFUSIONKIT_PYTHON to that interpreter or environment directory.",
        pathExists,
      ),
      zImage: optionalRuntimeReadiness(
        env,
        "Z_IMAGE_PYTHON",
        "MLX z-image generation",
        "Create the separate z-image environment, then set Z_IMAGE_PYTHON to that interpreter or environment directory.",
        pathExists,
      ),
    },
    modelsOwner: "mlx-community",
    modelsTtlMs: 10 * 60 * 1000,
    ollamaUrl: env.OLLAMA_URL ?? "http://127.0.0.1:11434",
    imageTmpDir: join(tmpdir(), "nodemlx-chat-images"),
    maxImageBytes: 10 * 1024 * 1024,
    maxLlamaModelBytes: 16_000_000_000,
    maxLlamaOutputChars: 2_000_000,
    sessionCookieName: "nodemlx_session",
    sessionTtlMs: 30 * 24 * 60 * 60 * 1000,
    minUsernameLength: 3,
    maxUsernameLength: 40,
    minPasswordLength: 8,
    maxGenerationTokens,
    defaultMaxTokens,
    memoryCancelThreshold: 98,
  };
}

export const imageExtensions = {
  "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
};
