import { statSync } from "node:fs";
import { join } from "node:path";

/**
 * Select the Python interpreter for the MLX z-image backend.
 *
 * Z-image requires newer MLX, Diffusers, and Transformers dependencies than
 * DiffusionKit, so it has a separate project-local virtual environment. An
 * explicit Z_IMAGE_PYTHON setting wins; the existing DiffusionKit runtime is
 * retained only as a legacy fallback for externally managed deployments.
 */
export function resolveZImagePython(
  pythonPath,
  projectRoot = process.cwd(),
  diffusionKitPython = "python3",
) {
  if (typeof pythonPath === "string" && pythonPath.length > 0) {
    try {
      return statSync(pythonPath).isDirectory()
        ? join(pythonPath, "bin", "python")
        : pythonPath;
    } catch {
      return pythonPath;
    }
  }

  const projectVenvPython = join(projectRoot, ".z-image-venv", "bin", "python");
  try {
    return statSync(projectVenvPython).isFile()
      ? projectVenvPython
      : diffusionKitPython;
  } catch {
    return diffusionKitPython;
  }
}
