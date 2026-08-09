import test from "node:test";
import assert from "node:assert/strict";
import {
  clampInteger,
  ConfigurationError,
  createConfig,
  optionalClampedInteger,
} from "../src/config.js";
import { parseGpuUsageOutput } from "../src/websocket/gpu-usage.js";

test("numeric configuration helpers preserve limits and rounding", () => {
  assert.equal(clampInteger(9.9, 1, 8, 4), 8);
  assert.equal(clampInteger(Number.NaN, 1, 8, 4), 4);
  assert.equal(optionalClampedInteger(65, 64, 2048, 8), 64);
  assert.equal(optionalClampedInteger(Number.NaN, 1, 2), undefined);
});

test("createConfig returns documented local-safe defaults", () => {
  const models = { "qwen-3-1.7b": "default-model" };
  const config = createConfig(models, {});
  assert.equal(config.port, 18956);
  assert.equal(config.host, "127.0.0.1");
  assert.equal(config.maxGenerationTokens, 32768);
  assert.equal(config.defaultMaxTokens, 4096);
  assert.equal(config.sessionTtlMs, 7 * 24 * 60 * 60 * 1000);
  assert.deepEqual(config.warnings, []);
});

test("createConfig rejects malformed startup values with setting-specific guidance", () => {
  const models = { "qwen-3-1.7b": "default-model" };

  for (const env of [
    { PORT: "not-a-port" },
    { PORT: "0" },
    { MLX_MAX_TOKENS_LIMIT: "many" },
    { MLX_MAX_TOKENS: "" },
    { HOST: "" },
  ]) {
    assert.throws(
      () => createConfig(models, env),
      (error) => error instanceof ConfigurationError && /PORT|MLX_MAX_TOKENS|HOST/.test(error.message),
    );
  }
});

test("createConfig clamps finite out-of-range generation limits and reports warnings", () => {
  const models = { "qwen-3-1.7b": "default-model" };
  const config = createConfig(models, {
    MLX_MAX_TOKENS_LIMIT: "999999",
    MLX_MAX_TOKENS: "0",
  });

  assert.equal(config.maxGenerationTokens, 131072);
  assert.equal(config.defaultMaxTokens, 1);
  assert.deepEqual(config.warnings, [
    "MLX_MAX_TOKENS_LIMIT=999999 is outside the supported range 1-131072; using 131072.",
    "MLX_MAX_TOKENS=0 is outside the supported range 1-131072; using 1.",
  ]);
});

test("createConfig retains a usable optional runtime location and diagnoses an unavailable one", () => {
  const models = { "qwen-3-1.7b": "default-model" };
  const config = createConfig(
    models,
    {
      DIFFUSIONKIT_PYTHON: "/opt/local/diffusionkit/bin/python",
      Z_IMAGE_PYTHON: "/missing/z-image/bin/python",
    },
    { pathExists: (path) => path.startsWith("/opt/local/") },
  );

  assert.equal(config.optionalCapabilities.diffusionKit.state, "available");
  assert.equal(config.optionalCapabilities.diffusionKit.path, "/opt/local/diffusionkit/bin/python");
  assert.equal(config.optionalCapabilities.zImage.state, "unavailable");
  assert.match(config.optionalCapabilities.zImage.diagnostic, /Z_IMAGE_PYTHON/);
  assert.match(config.optionalCapabilities.zImage.remediation, /separate z-image environment/);
});

test("GPU output parsing requires both metrics", () => {
  assert.deepEqual(parseGpuUsageOutput("GPU Usage: 14.5%\nMemory Usage: 98%"), {
    gpu: 14.5,
    memory: 98,
  });
  assert.equal(parseGpuUsageOutput("GPU Usage: 14.5%"), null);
});
