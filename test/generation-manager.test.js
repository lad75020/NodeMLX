import assert from "node:assert/strict";
import test from "node:test";

import { createGenerationManager } from "../src/inference/generation-manager.js";

function socket() {
  const messages = [];
  return {
    readyState: 1,
    messages,
    send(message) {
      messages.push(JSON.parse(message));
    },
  };
}

test("serializes generation requests and reports queued positions", async () => {
  const broadcasts = [];
  let releaseFirst;
  const manager = createGenerationManager({
    broadcast: (message) => broadcasts.push(message),
    parseGpuUsageOutput: () => null,
    gpuUsagePath: "GPUUsage",
    execFileAsync: async () => ({ stdout: "" }),
    memoryCancelThreshold: 98,
    cancelModelGeneration: () => {},
  });
  const firstSocket = socket();
  const secondSocket = socket();
  const runs = [];

  manager.enqueue({
    id: "first",
    socket: firstSocket,
    run: async () => {
      runs.push("first");
      await new Promise((resolve) => {
        releaseFirst = resolve;
      });
    },
  });
  manager.enqueue({
    id: "second",
    socket: secondSocket,
    run: async () => {
      runs.push("second");
    },
  });

  await new Promise(setImmediate);
  assert.deepEqual(runs, ["first"]);
  assert.deepEqual(secondSocket.messages, [
    { type: "queued", id: "second", position: 1 },
  ]);

  releaseFirst();
  await new Promise(setImmediate);
  assert.deepEqual(runs, ["first", "second"]);
  assert.deepEqual(
    broadcasts
      .filter(({ type }) => type === "gpuUsage")
      .map(({ running }) => running),
    [true, false, true, false],
  );
});

test("registerInferenceCancel remains safe when passed as an injected callback", () => {
  const manager = createGenerationManager({
    broadcast: () => {},
    parseGpuUsageOutput: () => null,
    gpuUsagePath: "GPUUsage",
    execFileAsync: async () => ({ stdout: "" }),
    memoryCancelThreshold: 98,
    cancelModelGeneration: () => {},
  });
  const registerInferenceCancel = manager.registerInferenceCancel;

  const unregister = registerInferenceCancel(() => {});

  assert.equal(typeof unregister, "function");
  unregister();
});

test("cancels queued and active inference when GPU memory reaches the threshold", async () => {
  const broadcasts = [];
  const modelCancellationReasons = [];
  const manager = createGenerationManager({
    broadcast: (message) => broadcasts.push(message),
    parseGpuUsageOutput: () => ({ gpu: 12, memory: 98 }),
    gpuUsagePath: "GPUUsage",
    execFileAsync: async () => ({ stdout: "usage" }),
    memoryCancelThreshold: 98,
    modelProcess: {
      cancelGeneration: (reason) => modelCancellationReasons.push(reason),
    },
  });
  const activeSocket = socket();
  const queuedSocket = socket();
  let activeReason;

  manager.enqueue({
    id: "active",
    socket: activeSocket,
    run: () =>
      new Promise((resolve) => {
        manager.registerInferenceCancel((reason) => {
          activeReason = reason;
          resolve();
        });
      }),
  });
  manager.enqueue({ id: "queued", socket: queuedSocket, run: async () => {} });

  await new Promise(setImmediate);
  const reason = "Memory usage reached 98%. Inference stopped.";
  assert.equal(activeReason, reason);
  assert.deepEqual(modelCancellationReasons, [reason]);
  assert.deepEqual(queuedSocket.messages.at(-1), {
    type: "error",
    id: "queued",
    error: reason,
  });
  assert.deepEqual(
    broadcasts
      .filter(({ type }) => type === "gpuUsage")
      .map(({ running }) => running),
    [true, true, false],
  );
});
