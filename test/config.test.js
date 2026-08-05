import test from "node:test";
import assert from "node:assert/strict";
import {
  clampInteger,
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

test("createConfig accepts an injected environment and retains PORT coercion", () => {
  const models = { "qwen-3-1.7b": "default-model" };
  assert.equal(createConfig(models, {}).port, 18956);
  assert.equal(createConfig(models, { PORT: "19001" }).port, 19001);
  assert.ok(Number.isNaN(createConfig(models, { PORT: "not-a-port" }).port));
  assert.equal(createConfig(models, { PORT: "0" }).port, 0);
});

test("GPU output parsing requires both metrics", () => {
  assert.deepEqual(parseGpuUsageOutput("GPU Usage: 14.5%\nMemory Usage: 98%"), {
    gpu: 14.5,
    memory: 98,
  });
  assert.equal(parseGpuUsageOutput("GPU Usage: 14.5%"), null);
});
