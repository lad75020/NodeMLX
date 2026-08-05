import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

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

export function createConfig(recommendedModels, env = process.env) {
  const maxGenerationTokens = clampInteger(Number(env.MLX_MAX_TOKENS_LIMIT ?? 32768), 1, 131072, 32768);
  return {
    defaultModel: env.MLX_MODEL ?? recommendedModels["qwen-3-1.7b"],
    // Deliberately retain the historical coercion behavior: malformed PORT
    // values are passed to Fastify, which remains the authority for validation.
    port: env.PORT ? Number(env.PORT) : 18956,
    host: env.HOST ?? "127.0.0.1",
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
    defaultMaxTokens: clampInteger(Number(env.MLX_MAX_TOKENS ?? 4096), 1, maxGenerationTokens, Math.min(4096, maxGenerationTokens)),
    memoryCancelThreshold: 98,
  };
}

export const imageExtensions = {
  "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
};
