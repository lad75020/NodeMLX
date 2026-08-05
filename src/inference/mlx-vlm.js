import { statSync } from "node:fs";
import { join } from "node:path";

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * MLX_VLM_PYTHON normally names a Python executable. For convenience, accept
 * a Conda environment directory as well and use its standard interpreter.
 */
export function resolveMlxVlmPython(pythonPath) {
  if (typeof pythonPath !== "string" || pythonPath.length === 0) {
    return "python3";
  }

  try {
    return statSync(pythonPath).isDirectory()
      ? join(pythonPath, "bin", "python")
      : pythonPath;
  } catch {
    // Keep command names such as `python3` untouched; spawn reports a missing
    // configured path through the application's actionable runtime error.
    return pythonPath;
  }
}

/**
 * Qwen3.5 checkpoints use a hybrid linear/full-attention VLM architecture.
 * node-mlx 2.3.0 supports Qwen3, but not the Qwen3.5 architecture.
 */
export function isMlxVlmModel(modelId) {
  return (
    typeof modelId === "string" &&
    /(^|[\/_-])qwen3[._-]?5(?:[\/_-]|$)/i.test(modelId)
  );
}

/**
 * Build an argument vector (never a shell command) for mlx-vlm's supported
 * module entry point. Keeping this pure makes its model-routing contract easy
 * to test and prevents prompt or path interpolation into a shell.
 */
export function buildMlxVlmGenerateArgs({
  modelId,
  prompt,
  imagePath = null,
  options = {},
}) {
  const args = [
    "-m",
    "mlx_vlm",
    "generate",
    "--model",
    modelId,
    "--prompt",
    prompt,
  ];

  if (finiteNumber(options.maxTokens)) {
    args.push("--max-tokens", String(Math.trunc(options.maxTokens)));
  }
  if (finiteNumber(options.temperature)) {
    args.push("--temperature", String(options.temperature));
  }
  if (finiteNumber(options.repetitionPenalty)) {
    args.push("--repetition-penalty", String(options.repetitionPenalty));
  }
  if (finiteNumber(options.repetitionContextSize)) {
    args.push(
      "--repetition-context-size",
      String(Math.trunc(options.repetitionContextSize)),
    );
  }
  if (imagePath) {
    args.push("--image", imagePath);
  }

  // With --no-verbose, mlx-vlm writes only the generated text to stdout.
  args.push("--no-verbose");
  return args;
}
