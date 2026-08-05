import assert from "node:assert/strict";
import { fork } from "node:child_process";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  buildMlxVlmGenerateArgs,
  isMlxVlmModel,
  resolveMlxVlmPython,
} from "../src/inference/mlx-vlm.js";

const PROJECT_ROOT = new URL("..", import.meta.url);
const WORKER_PATH = new URL("../model-worker.js", import.meta.url);

test("accepts a Conda environment directory for MLX_VLM_PYTHON", () => {
  assert.equal(
    resolveMlxVlmPython(process.cwd()),
    join(process.cwd(), "bin", "python"),
  );
  assert.equal(resolveMlxVlmPython("python3"), "python3");
});

test("routes Qwen3.5 checkpoints to the MLX-VLM runtime", () => {
  assert.equal(isMlxVlmModel("mlx-community/Qwen3.5-9B-8bit"), true);
  assert.equal(isMlxVlmModel("mlx-community/Qwen3-4B-Instruct-4bit"), false);
  assert.equal(isMlxVlmModel("my-org/qwen3_5-custom"), true);
});

test("builds a safe MLX-VLM CLI invocation from generation options", () => {
  assert.deepEqual(
    buildMlxVlmGenerateArgs({
      modelId: "mlx-community/Qwen3.5-9B-8bit",
      prompt: "Explain attention.",
      imagePath: "/tmp/reference.png",
      options: {
        maxTokens: 512,
        temperature: 0.2,
        repetitionPenalty: 1.05,
        repetitionContextSize: 64,
      },
    }),
    [
      "-m",
      "mlx_vlm",
      "generate",
      "--model",
      "mlx-community/Qwen3.5-9B-8bit",
      "--prompt",
      "Explain attention.",
      "--max-tokens",
      "512",
      "--temperature",
      "0.2",
      "--repetition-penalty",
      "1.05",
      "--repetition-context-size",
      "64",
      "--image",
      "/tmp/reference.png",
      "--no-verbose",
    ],
  );
});

test("worker routes Qwen3.5 through mlx-vlm and preserves IPC results", async () => {
  const directory = await mkdtemp(join(tmpdir(), "nodemlx-mlx-vlm-test-"));
  const fakePython = join(directory, "fake-python");
  const argumentLog = join(directory, "arguments.json");
  const prompt = 'A prompt with shell-looking text: $(touch /tmp/pwned) && "x"';

  await writeFile(
    fakePython,
    `#!/usr/bin/env node
const fs = require("node:fs");
const args = process.argv.slice(2);
if (args[0] === "-c") process.exit(0);
fs.writeFileSync(process.env.MLX_VLM_ARGUMENT_LOG, JSON.stringify(args));
process.stdout.write("mock MLX-VLM response\\n");
`,
  );
  await chmod(fakePython, 0o755);

  try {
    const result = await new Promise((resolve, reject) => {
      const worker = fork(WORKER_PATH, [], {
        cwd: PROJECT_ROOT,
        env: {
          ...process.env,
          MLX_VLM_PYTHON: fakePython,
          MLX_VLM_ARGUMENT_LOG: argumentLog,
        },
        silent: true,
      });
      const timeout = setTimeout(() => {
        worker.kill("SIGKILL");
        reject(new Error("MLX-VLM worker integration test timed out"));
      }, 8_000);
      let receivedResult = null;
      let settled = false;

      const finish = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        if (worker.connected) worker.send({ type: "exit" });
        if (error) reject(error);
      };

      worker.on("error", finish);
      worker.on("message", (message) => {
        if (message.type === "workerReady") {
          worker.send({
            type: "loadModel",
            modelId: "mlx-community/Qwen3.5-9B-8bit",
          });
        } else if (message.type === "modelReady") {
          assert.equal(message.isVLM, true);
          assert.equal(message.canGenerateImages, false);
          worker.send({
            type: "generate",
            id: "mlx-vlm-test",
            prompt,
            imagePath: "/tmp/test-image.png",
            options: { maxTokens: 24, temperature: 0 },
          });
        } else if (message.type === "generateResult") {
          receivedResult = message;
          finish();
        } else if (
          message.type === "modelError" ||
          message.type === "generateError"
        ) {
          finish(new Error(message.error));
        }
      });
      worker.on("exit", (code) => {
        if (settled && !receivedResult) return;
        if (code !== 0) {
          finish(new Error(`MLX-VLM worker exited with code ${code}`));
          return;
        }
        if (receivedResult) resolve(receivedResult);
      });
    });

    assert.deepEqual(result, {
      type: "generateResult",
      id: "mlx-vlm-test",
      modelId: "mlx-community/Qwen3.5-9B-8bit",
      text: "mock MLX-VLM response",
      images: [],
      tokenCount: 0,
      tokensPerSecond: 0,
    });
    assert.deepEqual(JSON.parse(await readFile(argumentLog, "utf8")), [
      "-m",
      "mlx_vlm",
      "generate",
      "--model",
      "mlx-community/Qwen3.5-9B-8bit",
      "--prompt",
      prompt,
      "--max-tokens",
      "24",
      "--temperature",
      "0",
      "--image",
      "/tmp/test-image.png",
      "--no-verbose",
    ]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("worker reports an actionable error when mlx-vlm is unavailable", async () => {
  const error = await new Promise((resolve, reject) => {
    const worker = fork(WORKER_PATH, [], {
      cwd: PROJECT_ROOT,
      env: { ...process.env, MLX_VLM_PYTHON: "/usr/bin/false" },
      silent: true,
    });
    const timeout = setTimeout(() => {
      worker.kill("SIGKILL");
      reject(new Error("MLX-VLM missing-runtime test timed out"));
    }, 8_000);

    const finish = (value) => {
      clearTimeout(timeout);
      if (worker.connected) worker.send({ type: "exit" });
      resolve(value);
    };

    worker.on("error", reject);
    worker.on("message", (message) => {
      if (message.type === "workerReady") {
        worker.send({
          type: "loadModel",
          modelId: "mlx-community/Qwen3.5-9B-8bit",
        });
      } else if (message.type === "modelError") {
        finish(message.error);
      } else if (message.type === "modelReady") {
        reject(new Error("Missing MLX-VLM runtime unexpectedly became ready"));
      }
    });
  });

  assert.match(error, /requires the MLX-VLM runtime/);
  assert.match(error, /pip install -U mlx-vlm/);
});
