/**
 * model-worker.js
 * Runs in a child process (fork).  All node-mlx calls live here so that a
 * native crash (SIGTRAP / Fatal error) kills only this process and not the
 * Fastify server.
 *
 * IPC protocol  (parent → worker)
 *   { type: "loadModel",  modelId }
 *   { type: "generate",   id, prompt, options, imagePath? }
 *   { type: "exit" }
 *
 * IPC protocol  (worker → parent)
 *   { type: "workerReady" }
 *   { type: "modelReady",    modelId, isVLM, canGenerateImages }
 *   { type: "modelError",    modelId, error }
 *   { type: "generateResult", id, modelId, text, images?, tokenCount, tokensPerSecond }
 *   { type: "generateError",  id, error }
 */

import { homedir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { loadModel } from "node-mlx";

// Inherit from parent (server.js sets this before fork), but guard here too
// so the worker is correct even if run standalone.
if (!process.env.HF_HOME) {
  process.env.HF_HOME = join(homedir(), ".cache", "huggingface");
}

let model = null;
let currentModelId = null;
let currentMode = "llm";

const IMAGE_OUTPUT_DIR = join(tmpdir(), "nodemlx-generated-images");
const DIFFUSIONKIT_COMPAT_CLI = join(process.cwd(), "scripts", "diffusionkit_cli_compat.py");
const Z_IMAGE_CLI = join(process.cwd(), "scripts", "z_image_cli.py");

function isImageGenerationModel(modelId) {
  return /(^|\/)(mlx-)?(stable-diffusion|sd3|sdxl|flux|z-image|diffusion|text-to-image)/i.test(modelId)
    || /diffusionkit/i.test(modelId);
}

function isZImageModel(modelId) {
  return /(^|\/)(mlx-)?z-image($|[-_])/i.test(modelId);
}

// Phi-4 was trained on a distinctive chat template that uses `<|im_sep|>` as
// role/content separator.  node-mlx passes prompts through without applying a
// chat template, which leaves phi-4 out of distribution and makes it emit
// gibberish.  Wrap the prompt manually here — *only* for phi-4 — so other
// models keep receiving raw prompts as before.
function isPhi4Model(modelId) {
  return /(^|\/)(mlx-)?phi-?4(?:-|$)/i.test(modelId)
      && !/phi-?4-?(?:mm|multimodal)/i.test(modelId);
}

function formatPromptForModel(prompt, modelId) {
  if (isPhi4Model(modelId)) {
    return (
      "<|im_start|>system<|im_sep|>You are a helpful assistant.<|im_end|>" +
      `<|im_start|>user<|im_sep|>${prompt}<|im_end|>` +
      "<|im_start|>assistant<|im_sep|>"
    );
  }
  return prompt;
}

function imageGenerationPreset(modelId) {
  if (isZImageModel(modelId)) {
    return { width: 1024, height: 1024, steps: 9 };
  }
  return { width: 768, height: 768, steps: 30 };
}

function clampInteger(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(value)));
}

function clampImageDimension(value, fallback) {
  const clamped = clampInteger(value, 64, 2048, fallback);
  return Math.max(64, Math.round(clamped / 8) * 8);
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${command} exited with code ${code}\n${stderr || stdout}`.trim()));
    });
  });
}

async function generateImage(prompt, modelId, options = {}) {
  await mkdir(IMAGE_OUTPUT_DIR, { recursive: true });
  const outputPath = join(IMAGE_OUTPUT_DIR, `${randomUUID()}.png`);
  const isZImage = isZImageModel(modelId);
  const preset = imageGenerationPreset(modelId);
  const width = clampImageDimension(options.imageWidth, preset.width);
  const height = clampImageDimension(options.imageHeight, preset.height);
  const steps = clampInteger(options.steps, 1, 150, preset.steps);
  const cfg = Number.isFinite(options.cfg) ? options.cfg : 7;
  const seed = clampInteger(options.seed, 0, 2 ** 31 - 1, Math.floor(Math.random() * 2 ** 31));

  try {
    if (isZImage) {
      await runCommand(process.env.Z_IMAGE_PYTHON ?? process.env.DIFFUSIONKIT_PYTHON ?? "python3", [
        Z_IMAGE_CLI,
        "--prompt", prompt,
        "--model-id", modelId,
        "--height", String(height),
        "--width", String(width),
        "--seed", String(seed),
        "--steps", String(steps),
        "--output-path", outputPath,
      ]);
    } else {
      await runCommand(process.env.DIFFUSIONKIT_PYTHON ?? "python3", [
        DIFFUSIONKIT_COMPAT_CLI,
        "--prompt", prompt,
        "--model-version", modelId,
        "--height", String(height),
        "--width", String(width),
        "--seed", String(seed),
        "--steps", String(steps),
        "--cfg", String(cfg),
        "--output-path", outputPath,
      ]);
    }
    const bytes = await readFile(outputPath);
    return {
      text: "",
      images: [{
        dataUrl: `data:image/png;base64,${bytes.toString("base64")}`,
        name: `generated-${seed}.png`,
        type: "image/png",
        size: bytes.byteLength,
      }],
      tokenCount: 0,
      tokensPerSecond: 0,
    };
  } catch (err) {
    if (err?.code === "ENOENT") {
      throw new Error("DiffusionKit was not found. Install it with `pip install diffusionkit` to run DiffusionKit MLX image-generation models.");
    }
    throw err;
  } finally {
    await rm(outputPath, { force: true }).catch(() => {});
  }
}

process.on("message", async (msg) => {
  switch (msg.type) {
    case "loadModel": {
      if (model) {
        try { model.unload(); } catch {}
        model = null;
        currentModelId = null;
      }
      currentMode = isImageGenerationModel(msg.modelId) ? "image-generation" : "llm";
      try {
        if (currentMode === "image-generation") {
          currentModelId = msg.modelId;
          process.send({
            type: "modelReady",
            modelId: msg.modelId,
            isVLM: false,
            canGenerateImages: true,
          });
          break;
        }

        model = loadModel(msg.modelId);
        currentModelId = msg.modelId;
        process.send({
          type: "modelReady",
          modelId: msg.modelId,
          isVLM: model.isVLM(),
          canGenerateImages: false,
        });
      } catch (err) {
        process.send({
          type: "modelError",
          modelId: msg.modelId,
          error: err instanceof Error ? err.message : String(err),
        });
      }
      break;
    }

    case "generate": {
      if (currentMode !== "image-generation" && !model) {
        process.send({ type: "generateError", id: msg.id, error: "No model loaded." });
        return;
      }
      if (currentMode === "image-generation" && !currentModelId) {
        process.send({ type: "generateError", id: msg.id, error: "No image-generation model selected." });
        return;
      }
      try {
        const formattedPrompt = formatPromptForModel(msg.prompt, currentModelId);
        const result = currentMode === "image-generation"
          ? await generateImage(msg.prompt, currentModelId, msg.options ?? {})
          : (msg.imagePath
              ? model.generateWithImage(formattedPrompt, msg.imagePath, msg.options ?? {})
              : model.generate(formattedPrompt, msg.options ?? {}));
        process.send({
          type: "generateResult",
          id: msg.id,
          modelId: currentModelId,
          text: result.text,
          images: result.images ?? [],
          tokenCount: result.tokenCount,
          tokensPerSecond: result.tokensPerSecond,
        });
      } catch (err) {
        process.send({
          type: "generateError",
          id: msg.id,
          error: err instanceof Error ? err.message : String(err),
        });
      }
      break;
    }

    case "exit": {
      if (model) { try { model.unload(); } catch {} }
      process.exit(0);
      break;
    }
  }
});

// Tell the parent we are up and ready to receive messages.
process.send({ type: "workerReady" });
