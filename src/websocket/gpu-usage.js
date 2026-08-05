export function parseGpuUsageOutput(output) {
  const text = typeof output === "string" ? output : "";
  const gpuMatch = /GPU Usage:\s*([0-9]+(?:\.[0-9]+)?)%/i.exec(text);
  const memoryMatch = /Memory Usage:\s*([0-9]+(?:\.[0-9]+)?)%/i.exec(text);
  if (!gpuMatch || !memoryMatch) return null;
  return { gpu: Number(gpuMatch[1]), memory: Number(memoryMatch[1]) };
}
