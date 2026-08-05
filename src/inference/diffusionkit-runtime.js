import { statSync } from "node:fs";
import { join } from "node:path";

/**
 * Select the Python interpreter used by the optional DiffusionKit backends.
 *
 * An explicit `DIFFUSIONKIT_PYTHON` value wins. Otherwise NodeMLX uses the
 * project-local virtual environment, keeping MLX packages out of Homebrew's
 * mutable global Python installation. The legacy `python3` fallback remains
 * for deployments that manage the runtime externally.
 */
export function resolveDiffusionKitPython(pythonPath, projectRoot = process.cwd()) {
  if (typeof pythonPath === "string" && pythonPath.length > 0) {
    try {
      return statSync(pythonPath).isDirectory()
        ? join(pythonPath, "bin", "python")
        : pythonPath;
    } catch {
      return pythonPath;
    }
  }

  const projectVenvPython = join(projectRoot, ".venv", "bin", "python");
  try {
    return statSync(projectVenvPython).isFile()
      ? projectVenvPython
      : "python3";
  } catch {
    return "python3";
  }
}
