// Coordinates serial generation work, GPU usage reporting, and cancellation.
export class GenerationManager {
  #broadcast;
  #parseGpuUsageOutput;
  #gpuUsagePath;
  #execFileAsync;
  #memoryCancelThreshold;
  #cancelModelGeneration;
  #active = false;
  #items = [];
  #gpuUsagePollTimer = null;
  #gpuUsagePolling = false;
  #activeInferenceCount = 0;
  #activeInferenceCancels = new Set();

  constructor({
    broadcast,
    parseGpuUsageOutput,
    gpuUsagePath,
    execFileAsync,
    memoryCancelThreshold,
    modelProcess,
    cancelModelGeneration,
  }) {
    this.#broadcast = broadcast;
    this.#parseGpuUsageOutput = parseGpuUsageOutput;
    this.#gpuUsagePath = gpuUsagePath;
    this.#execFileAsync = execFileAsync;
    this.#memoryCancelThreshold = memoryCancelThreshold;
    this.#cancelModelGeneration =
      cancelModelGeneration ??
      ((reason) => modelProcess?.cancelGeneration(reason));
  }

  enqueue(item) {
    const willWait = this.#active || this.#items.length > 0;
    this.#items.push(item);
    if (willWait) this.#notifyQueuedPositions();
    void this.#drain();
  }

  registerInferenceCancel = (cancel) => {
    this.#activeInferenceCancels.add(cancel);
    return () => this.#activeInferenceCancels.delete(cancel);
  };

  cancelAllInference(reason = "Inference cancelled.") {
    this.#cancelQueued(reason);
    this.#cancelModelGeneration(reason);
    for (const cancel of [...this.#activeInferenceCancels]) {
      try {
        cancel(reason);
      } catch (err) {
        console.warn(
          `Inference cancel hook failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }
  }

  #cancelQueued(reason) {
    const queued = this.#items;
    this.#items = [];
    for (const item of queued) {
      if (item.socket?.readyState === 1) {
        item.socket.send(
          JSON.stringify({ type: "error", id: item.id, error: reason }),
        );
      }
    }
  }

  async #drain() {
    if (this.#active) return;

    while (this.#items.length > 0) {
      const item = this.#items.shift();
      this.#notifyQueuedPositions();

      if (item.socket?.readyState !== 1) continue;

      this.#active = true;
      this.#startGpuUsagePolling();
      try {
        await item.run();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (item.socket?.readyState === 1) {
          item.socket.send(
            JSON.stringify({ type: "error", id: item.id, error: message }),
          );
        }
      } finally {
        this.#stopGpuUsagePolling();
        this.#active = false;
      }
    }
  }

  #notifyQueuedPositions() {
    this.#items = this.#items.filter((item) => item.socket?.readyState === 1);
    this.#items.forEach((item, index) => {
      item.socket.send(
        JSON.stringify({
          type: "queued",
          id: item.id,
          position: index + 1,
        }),
      );
    });
  }

  async #pollGpuUsageOnce() {
    if (this.#gpuUsagePolling || this.#activeInferenceCount <= 0) return;
    this.#gpuUsagePolling = true;
    try {
      const { stdout } = await this.#execFileAsync(this.#gpuUsagePath);
      const usage = this.#parseGpuUsageOutput(stdout);
      if (usage && this.#activeInferenceCount > 0) {
        this.#broadcast({ type: "gpuUsage", running: true, ...usage });
        if (usage.memory >= this.#memoryCancelThreshold) {
          this.cancelAllInference(
            `Memory usage reached ${usage.memory}%. Inference stopped.`,
          );
        }
      }
    } catch (err) {
      console.warn(
        `GPU usage polling failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      this.#gpuUsagePolling = false;
    }
  }

  #startGpuUsagePolling() {
    this.#activeInferenceCount += 1;
    if (this.#activeInferenceCount !== 1) return;
    this.#broadcast({
      type: "gpuUsage",
      running: true,
      gpu: null,
      memory: null,
    });
    void this.#pollGpuUsageOnce();
    this.#gpuUsagePollTimer = setInterval(() => {
      void this.#pollGpuUsageOnce();
    }, 1000);
  }

  #stopGpuUsagePolling() {
    this.#activeInferenceCount = Math.max(0, this.#activeInferenceCount - 1);
    if (this.#activeInferenceCount > 0) return;
    if (this.#gpuUsagePollTimer) {
      clearInterval(this.#gpuUsagePollTimer);
      this.#gpuUsagePollTimer = null;
    }
    this.#broadcast({
      type: "gpuUsage",
      running: false,
      gpu: null,
      memory: null,
    });
  }
}

export function createGenerationManager(options) {
  return new GenerationManager(options);
}
