import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { resolveZImagePython } from "../src/inference/z-image-runtime.js";

test("prefers an explicitly configured z-image interpreter", async () => {
  const projectDir = await mkdtemp(join(tmpdir(), "nodemlx-z-image-runtime-test-"));
  const configuredPython = join(projectDir, "configured-python");

  try {
    await writeFile(configuredPython, "#!/usr/bin/env false\n");
    await chmod(configuredPython, 0o755);

    assert.equal(
      resolveZImagePython(configuredPython, projectDir),
      configuredPython,
    );
  } finally {
    await rm(projectDir, { recursive: true, force: true });
  }
});

test("uses the dedicated z-image virtual environment by default", async () => {
  const projectDir = await mkdtemp(join(tmpdir(), "nodemlx-z-image-runtime-test-"));
  const venvPython = join(projectDir, ".z-image-venv", "bin", "python");

  try {
    await mkdir(join(projectDir, ".z-image-venv", "bin"), { recursive: true });
    await writeFile(venvPython, "#!/usr/bin/env false\n");
    await chmod(venvPython, 0o755);

    assert.equal(resolveZImagePython(undefined, projectDir), venvPython);
  } finally {
    await rm(projectDir, { recursive: true, force: true });
  }
});

test("uses the general DiffusionKit interpreter only as a legacy z-image fallback", () => {
  assert.equal(
    resolveZImagePython(
      undefined,
      join(tmpdir(), "missing-nodemlx-project"),
      "/custom/diffusionkit-python",
    ),
    "/custom/diffusionkit-python",
  );
});
