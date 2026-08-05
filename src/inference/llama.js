// Llama.cpp inference and model selection. Runtime dependencies are injected.
export function createLlamaInference({
  maxLlamaModelBytes,
  maxLlamaOutputChars,
  maxGenerationTokens,
  defaultMaxTokens,
  clampInteger,
  ensureUserChat,
  appendChatMessages,
  registerInferenceCancel,
  stat,
  basename,
  execFileAsync,
  spawn,
  environment,
}) {
  async function validateLlamaModelFile(modelPath) {
    const filePath = typeof modelPath === "string" ? modelPath.trim() : "";
    if (!filePath) throw new Error("Choose a Llama.cpp model file first.");

    const info = await stat(filePath).catch((err) => {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`Llama.cpp model file is not available: ${message}`);
    });
    if (!info.isFile())
      throw new Error("Llama.cpp model path must point to a file.");
    if (info.size > maxLlamaModelBytes) {
      throw new Error("Llama.cpp model file must be 16 GB or smaller.");
    }
    return {
      path: filePath,
      name: basename(filePath),
      size: info.size,
    };
  }

  async function pickLlamaModelFile() {
    const { stdout } = await execFileAsync("osascript", [
      "-e",
      'POSIX path of (choose file with prompt "Choose a Llama.cpp model file")',
    ]);
    return validateLlamaModelFile(stdout.trim());
  }

  function validateHuggingFaceModelName(value) {
    const model = typeof value === "string" ? value.trim() : "";
    if (!model) throw new Error("Enter a Hugging Face model name first.");
    if (model.length > 300 || /\s/.test(model)) {
      throw new Error(
        "Hugging Face model name must be a single model id without whitespace.",
      );
    }
    return model;
  }

  async function streamLlamaPrompt(socket, payload, userId) {
    const id = payload.id ?? String(Date.now());
    const prompt =
      typeof payload.prompt === "string" ? payload.prompt.trim() : "";
    const modelSource =
      payload.modelSource === "huggingface" ? "huggingface" : "disk";
    const model =
      modelSource === "huggingface"
        ? {
            name: validateHuggingFaceModelName(payload.hfModel),
            source: "huggingface",
          }
        : {
            ...(await validateLlamaModelFile(payload.modelPath)),
            source: "disk",
          };
    if (!prompt) throw new Error("Prompt required.");

    const { chatId, created } = await ensureUserChat(
      userId,
      typeof payload.chatId === "string" ? payload.chatId : null,
    );
    if (created) {
      socket.send(JSON.stringify({ type: "chatCreated", id, chat: created }));
    }

    const userAt = new Date();
    let fullText = "";
    let fullThinking = "";
    let stderr = "";
    let settled = false;
    let cancelReason = null;
    let outputLimitError = null;
    let parseBuffer = "";
    let llamaPhase = "before-thinking";
    socket.send(JSON.stringify({ type: "start", id, chatId }));
    const maxTokens = clampInteger(
      payload.maxTokens,
      1,
      maxGenerationTokens,
      defaultMaxTokens,
    );

    const sendLlamaChunk = (chunk) => {
      if (!chunk || socket.readyState !== 1) return;
      socket.send(JSON.stringify({ type: "llamaChunk", id, ...chunk }));
    };
    let child;
    const stopForOutputLimit = () => {
      outputLimitError = new Error(
        "Llama.cpp output exceeded the 2,000,000 character safety limit.",
      );
      try {
        child.kill("SIGTERM");
      } catch {}
    };
    const appendLlamaText = (text) => {
      if (!text) return;
      const remaining = maxLlamaOutputChars - fullText.length;
      if (remaining <= 0) {
        stopForOutputLimit();
        return;
      }
      const safeText =
        text.length > remaining ? text.slice(0, remaining) : text;
      fullText = `${fullText}${safeText}`;
      sendLlamaChunk({ text: safeText });
      if (safeText.length < text.length) stopForOutputLimit();
    };
    const appendLlamaThinking = (thinking) => {
      if (!thinking) return;
      fullThinking = `${fullThinking}${thinking}`;
      sendLlamaChunk({ thinking });
    };
    const consumeLlamaOutput = (raw) => {
      parseBuffer += raw;
      while (parseBuffer) {
        if (llamaPhase === "before-thinking") {
          const start = parseBuffer.indexOf("[Start thinking]");
          if (start === -1) {
            parseBuffer = parseBuffer.slice(
              Math.max(0, parseBuffer.length - "[Start thinking]".length + 1),
            );
            return;
          }
          parseBuffer = parseBuffer.slice(start + "[Start thinking]".length);
          llamaPhase = "thinking";
        }

        if (llamaPhase === "thinking") {
          const end = parseBuffer.indexOf("[End thinking]");
          if (end === -1) {
            const keep = "[End thinking]".length - 1;
            const emitLength = Math.max(0, parseBuffer.length - keep);
            if (emitLength > 0) {
              appendLlamaThinking(parseBuffer.slice(0, emitLength));
              parseBuffer = parseBuffer.slice(emitLength);
            }
            return;
          }
          appendLlamaThinking(parseBuffer.slice(0, end));
          parseBuffer = parseBuffer.slice(end + "[End thinking]".length);
          llamaPhase = "answer";
        }

        if (llamaPhase === "answer") {
          appendLlamaText(parseBuffer);
          parseBuffer = "";
        }
      }
    };
    const flushLlamaOutput = () => {
      if (llamaPhase === "thinking" && parseBuffer) {
        appendLlamaThinking(parseBuffer);
      } else if (llamaPhase === "answer" && parseBuffer) {
        appendLlamaText(parseBuffer);
      }
      parseBuffer = "";
    };

    const modelArgs =
      model.source === "huggingface" ? ["-hf", model.name] : ["-m", model.path];
    child = spawn(
      "llama-cli",
      [
        "--simple-io",
        "--single-turn",
        "--no-display-prompt",
        "--log-disable",
        "-n",
        String(maxTokens),
        ...modelArgs,
        "-p",
        prompt,
      ],
      {
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...environment,
          LLAMA_LOG_COLORS: "off",
          NO_COLOR: "1",
        },
      },
    );

    const unregisterCancel = registerInferenceCancel((reason) => {
      if (settled) return;
      cancelReason =
        typeof reason === "string" ? reason : "Inference cancelled.";
      try {
        child.kill("SIGTERM");
      } catch {}
      setTimeout(() => {
        if (!settled) {
          try {
            child.kill("SIGKILL");
          } catch {}
        }
      }, 2000);
    });
    const onClose = () => {
      try {
        child.kill("SIGTERM");
      } catch {}
    };
    socket.once?.("close", onClose);

    try {
      await new Promise((resolve, reject) => {
        child.stdout.on("data", (chunk) => {
          if (outputLimitError) return;
          const text = chunk.toString();
          if (text) consumeLlamaOutput(text);
        });
        child.stderr.on("data", (chunk) => {
          stderr += chunk.toString();
        });
        child.on("error", reject);
        child.on("close", (code, signal) => {
          settled = true;
          if (code === 0) resolve();
          else if (outputLimitError) reject(outputLimitError);
          else if (cancelReason) reject(new Error(cancelReason));
          else
            reject(
              new Error(
                signal
                  ? `llama-cli stopped by ${signal}.`
                  : `llama-cli exited with code ${code}${stderr ? `: ${stderr.trim()}` : "."}`,
              ),
            );
        });
      });
      flushLlamaOutput();
      if (outputLimitError) throw outputLimitError;

      socket.send(
        JSON.stringify({
          type: "llamaDone",
          id,
          chatId,
          modelName: model.name,
          text: fullText,
          thinking: fullThinking || undefined,
        }),
      );

      if (chatId) {
        await appendChatMessages(userId, chatId, [
          {
            id,
            role: "user",
            text: prompt,
            provider: "llamacpp",
            createdAt: userAt,
          },
          {
            id: `${id}:reply`,
            role: "assistant",
            text: fullText,
            thinking: fullThinking || undefined,
            provider: "llamacpp",
            modelId: model.name,
            createdAt: new Date(),
          },
        ]);
      }
    } finally {
      settled = true;
      unregisterCancel();
      socket.off?.("close", onClose);
    }
  }

  return { pickLlamaModelFile, streamLlamaPrompt };
}
