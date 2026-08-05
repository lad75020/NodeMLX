import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { resolveDiffusionKitPython } from "../src/inference/diffusionkit-runtime.js";

test("prefers an explicitly configured DiffusionKit interpreter", async () => {
  const projectDir = await mkdtemp(join(tmpdir(), "nodemlx-diffusionkit-test-"));
  const configuredPython = join(projectDir, "configured-python");

  try {
    await writeFile(configuredPython, "#!/usr/bin/env false\n");
    await chmod(configuredPython, 0o755);

    assert.equal(
      resolveDiffusionKitPython(configuredPython, projectDir),
      configuredPython,
    );
  } finally {
    await rm(projectDir, { recursive: true, force: true });
  }
});

test("uses the project virtual environment for DiffusionKit by default", async () => {
  const projectDir = await mkdtemp(join(tmpdir(), "nodemlx-diffusionkit-test-"));
  const venvPython = join(projectDir, ".venv", "bin", "python");

  try {
    await mkdir(join(projectDir, ".venv", "bin"), { recursive: true });
    await writeFile(venvPython, "#!/usr/bin/env false\n");
    await chmod(venvPython, 0o755);

    assert.equal(resolveDiffusionKitPython(undefined, projectDir), venvPython);
  } finally {
    await rm(projectDir, { recursive: true, force: true });
  }
});

test("falls back to python3 when no DiffusionKit interpreter is configured", () => {
  assert.equal(
    resolveDiffusionKitPython(undefined, join(tmpdir(), "missing-nodemlx-project")),
    "python3",
  );
});
