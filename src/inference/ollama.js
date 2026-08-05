// Ollama inference and RPC helpers. All bootstrap dependencies are injected.
export function createOllamaInference({
  ollamaUrl,
  maxImageBytes,
  optionalClampedInteger,
  ensureUserChat,
  appendChatMessages,
  registerInferenceCancel,
}) {
  async function listOllamaModels() {
    const res = await fetch(`${ollamaUrl}/api/tags`);
    if (!res.ok) throw new Error(`Ollama model list failed (${res.status}).`);
    const body = await res.json();
    const models = Array.isArray(body?.models) ? body.models : [];
    return models
      .map((m) => ({
        id: typeof m.name === "string" ? m.name : "",
        name: typeof m.name === "string" ? m.name : "",
        modifiedAt: typeof m.modified_at === "string" ? m.modified_at : null,
        size: typeof m.size === "number" ? m.size : 0,
        digest: typeof m.digest === "string" ? m.digest : null,
        details: m.details && typeof m.details === "object" ? m.details : null,
      }))
      .filter((m) => m.id.length > 0)
      .sort((a, b) => a.id.localeCompare(b.id));
  }

  async function showOllamaModel(model) {
    const id = typeof model === "string" ? model.trim() : "";
    if (!id) throw new Error("Ollama model required.");
    const res = await fetch(`${ollamaUrl}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: id }),
    });
    if (!res.ok) {
      const message = await res.text().catch(() => "");
      throw new Error(
        `Ollama model details failed (${res.status})${message ? `: ${message}` : "."}`,
      );
    }
    const body = await res.json();
    return {
      id,
      capabilities: Array.isArray(body?.capabilities)
        ? body.capabilities.filter((value) => typeof value === "string")
        : [],
      details:
        body?.details && typeof body.details === "object" ? body.details : null,
      modifiedAt:
        typeof body?.modified_at === "string" ? body.modified_at : null,
    };
  }

  function promptImageForOllama(image) {
    if (!image) return null;
    if (typeof image !== "object" || typeof image.dataUrl !== "string")
      throw new Error("Invalid image payload.");
    const match =
      /^data:(image\/(?:jpeg|png|webp|gif));base64,([A-Za-z0-9+/=]+)$/.exec(
        image.dataUrl,
      );
    if (!match)
      throw new Error("Image must be a JPEG, PNG, WebP, or GIF data URL.");
    const buffer = Buffer.from(match[2], "base64");
    if (buffer.byteLength === 0) throw new Error("Image is empty.");
    if (buffer.byteLength > maxImageBytes)
      throw new Error("Image is too large. Maximum size is 10 MB.");
    return {
      base64: match[2],
      attachment: {
        dataUrl: image.dataUrl,
        name:
          typeof image.name === "string" ? image.name.slice(0, 200) : "image",
        type: match[1],
        size: buffer.byteLength,
      },
    };
  }

  function collectOllamaImages(parsed) {
    const images = [];
    for (const candidate of [
      parsed?.images,
      parsed?.image,
      parsed?.message?.images,
      parsed?.message?.image,
      parsed?.response?.images,
      parsed?.response?.image,
      parsed?.artifacts,
      parsed?.data,
    ])
      addOllamaImageCandidate(images, candidate);
    return images;
  }
  function addOllamaImageCandidate(images, candidate) {
    if (!candidate) return;
    if (Array.isArray(candidate)) {
      for (const item of candidate) addOllamaImageCandidate(images, item);
      return;
    }
    const normalized = normalizeOllamaImage(candidate, images.length + 1);
    if (normalized) {
      if (!images.some((image) => image.dataUrl === normalized.dataUrl))
        images.push(normalized);
      return;
    }
    if (typeof candidate !== "object") return;
    for (const key of ["images", "image", "artifacts", "data"])
      if (Object.hasOwn(candidate, key))
        addOllamaImageCandidate(images, candidate[key]);
  }
  function normalizeOllamaImage(candidate, index) {
    if (typeof candidate === "string")
      return normalizeOllamaImageSource(candidate, undefined, undefined, index);
    if (!candidate || typeof candidate !== "object") return null;
    const mimeType =
      firstString(
        candidate.mimeType,
        candidate.mime_type,
        candidate.mediaType,
        candidate.media_type,
        candidate.type,
      ) ?? undefined;
    const name =
      firstString(
        candidate.name,
        candidate.filename,
        candidate.fileName,
        candidate.alt,
      ) ?? undefined;
    for (const key of [
      "dataUrl",
      "data_url",
      "url",
      "src",
      "b64_json",
      "base64",
      "image",
      "data",
    ]) {
      if (typeof candidate[key] !== "string") continue;
      const normalized = normalizeOllamaImageSource(
        candidate[key],
        mimeType,
        name,
        index,
      );
      if (normalized) return normalized;
    }
    return null;
  }
  function normalizeOllamaImageSource(
    rawSource,
    hintMimeType,
    hintName,
    index,
  ) {
    const source = rawSource.trim();
    if (!source) return null;
    const dataUrl = normalizeImageDataUrl(source, hintMimeType);
    if (dataUrl)
      return {
        dataUrl: dataUrl.url,
        name: hintName ?? `Ollama image ${index}`,
        type: dataUrl.type,
        size: dataUrl.size,
      };
    if (/^https?:\/\//i.test(source))
      return {
        dataUrl: source,
        name: hintName ?? imageNameFromUrl(source, index),
        type: imageTypeFromUrl(source),
        size: 0,
      };
    const base64 = source.replace(/\s+/g, "");
    const type = inferBase64ImageMime(base64, hintMimeType);
    return type
      ? {
          dataUrl: `data:${type};base64,${base64}`,
          name: hintName ?? `Ollama image ${index}`,
          type,
          size: Math.floor((base64.length * 3) / 4),
        }
      : null;
  }
  function normalizeImageDataUrl(source, hintMimeType) {
    const match =
      /^data:(image\/[^;,]+);base64,([\s\S]+)$/i.exec(source) ??
      /^(image\/[^;,]+);base64,([\s\S]+)$/i.exec(source);
    if (match) {
      const type = match[1].toLowerCase();
      const base64 = match[2].replace(/\s+/g, "");
      return isLikelyBase64(base64)
        ? {
            url: `data:${type};base64,${base64}`,
            type,
            size: Math.floor((base64.length * 3) / 4),
          }
        : null;
    }
    if (hintMimeType && /^image\//i.test(hintMimeType)) {
      const base64 = source.replace(/\s+/g, "");
      const type = hintMimeType.toLowerCase();
      return isLikelyBase64(base64)
        ? {
            url: `data:${type};base64,${base64}`,
            type,
            size: Math.floor((base64.length * 3) / 4),
          }
        : null;
    }
    return null;
  }
  function inferBase64ImageMime(base64, hintMimeType) {
    if (!isLikelyBase64(base64)) return null;
    if (hintMimeType && /^image\//i.test(hintMimeType))
      return hintMimeType.toLowerCase();
    let bytes;
    try {
      bytes = Buffer.from(base64.slice(0, 96), "base64");
    } catch {
      return null;
    }
    if (bytes.length < 8) return null;
    if (
      bytes[0] === 0x89 &&
      bytes[1] === 0x50 &&
      bytes[2] === 0x4e &&
      bytes[3] === 0x47
    )
      return "image/png";
    if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff)
      return "image/jpeg";
    if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46)
      return "image/gif";
    if (
      bytes[0] === 0x52 &&
      bytes[1] === 0x49 &&
      bytes[2] === 0x46 &&
      bytes[3] === 0x46 &&
      bytes[8] === 0x57 &&
      bytes[9] === 0x45 &&
      bytes[10] === 0x42 &&
      bytes[11] === 0x50
    )
      return "image/webp";
    return bytes
      .toString("utf8", 0, Math.min(bytes.length, 16))
      .trimStart()
      .startsWith("<svg")
      ? "image/svg+xml"
      : null;
  }
  function isLikelyBase64(value) {
    return value.length >= 64 && /^[A-Za-z0-9+/]+={0,2}$/.test(value);
  }
  function firstString(...values) {
    return values
      .find((value) => typeof value === "string" && value.trim().length > 0)
      ?.trim();
  }
  function imageNameFromUrl(source, index) {
    try {
      const url = new URL(source);
      return decodeURIComponent(
        url.pathname.split("/").filter(Boolean).pop() ||
          `Ollama image ${index}`,
      );
    } catch {
      return `Ollama image ${index}`;
    }
  }
  function imageTypeFromUrl(source) {
    const path = source.split("?")[0].toLowerCase();
    if (path.endsWith(".jpg") || path.endsWith(".jpeg")) return "image/jpeg";
    if (path.endsWith(".png")) return "image/png";
    if (path.endsWith(".gif")) return "image/gif";
    if (path.endsWith(".webp")) return "image/webp";
    if (path.endsWith(".svg")) return "image/svg+xml";
    return "image";
  }
  function mergeOllamaImages(existing, next) {
    const bySource = new Map();
    for (const image of [...existing, ...next])
      if (!bySource.has(image.dataUrl)) bySource.set(image.dataUrl, image);
    return [...bySource.values()];
  }
  function ollamaResponseText(parsed) {
    if (typeof parsed?.response === "string") return parsed.response;
    if (typeof parsed?.message?.content === "string")
      return parsed.message.content;
    if (typeof parsed?.content === "string") return parsed.content;
    return typeof parsed?.text === "string" ? parsed.text : "";
  }
  function ollamaResponseThinking(parsed) {
    if (typeof parsed?.thinking === "string") return parsed.thinking;
    return typeof parsed?.message?.thinking === "string"
      ? parsed.message.thinking
      : "";
  }
  function ollamaThinkValue(model) {
    return /\bgpt-oss\b/i.test(model) ? "medium" : true;
  }
  async function unloadOllamaModel(model) {
    const id = typeof model === "string" ? model.trim() : "";
    if (!id) return;
    try {
      const res = await fetch(`${ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: id, keep_alive: 0 }),
      });
      if (!res.ok) {
        const message = await res.text().catch(() => "");
        console.warn(
          `Ollama unload failed (${res.status})${message ? `: ${message}` : "."}`,
        );
      }
    } catch (err) {
      console.warn(
        `Ollama unload failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
  function extractOllamaImagesFromText(text) {
    const trimmed = text.trim();
    if (!trimmed) return { text, images: [] };
    if (trimmed.startsWith("{") || trimmed.startsWith("["))
      try {
        const parsed = JSON.parse(trimmed);
        const images = collectOllamaImages(parsed);
        if (images.length > 0)
          return { text: ollamaResponseText(parsed).trim(), images };
      } catch {
        /* Preserve a non-JSON response. */
      }
    const image = normalizeOllamaImageSource(trimmed, undefined, undefined, 1);
    return image ? { text: "", images: [image] } : { text, images: [] };
  }

  async function streamOllamaPrompt(socket, payload, userId) {
    const id = payload.id ?? String(Date.now());
    const model =
      typeof payload.modelId === "string" ? payload.modelId.trim() : "";
    const prompt =
      typeof payload.prompt === "string" ? payload.prompt.trim() : "";
    const image = promptImageForOllama(payload.image);
    if (!model) throw new Error("Ollama model required.");
    if (!prompt && !image) throw new Error("Prompt required.");
    const { chatId, created } = await ensureUserChat(
      userId,
      typeof payload.chatId === "string" ? payload.chatId : null,
    );
    if (created)
      socket.send(JSON.stringify({ type: "chatCreated", id, chat: created }));
    const controller = new AbortController();
    const onClose = () => controller.abort(new Error("Client disconnected."));
    const unregisterCancel = registerInferenceCancel((reason) => {
      if (!controller.signal.aborted)
        controller.abort(
          new Error(
            typeof reason === "string" ? reason : "Inference cancelled.",
          ),
        );
    });
    socket.once?.("close", onClose);
    const userAt = new Date();
    let fullText = "";
    let fullThinking = "";
    let generatedImages = [];
    let finalStats = {};
    let responseCompleted = false;
    socket.send(JSON.stringify({ type: "start", id, chatId }));
    const consume = (parsed) => {
      if (typeof parsed.error === "string") throw new Error(parsed.error);
      const text = ollamaResponseText(parsed);
      const thinking = ollamaResponseThinking(parsed);
      const images = collectOllamaImages(parsed);
      if (text.length > 0 || thinking.length > 0 || images.length > 0) {
        fullText += text;
        fullThinking += thinking;
        if (images.length > 0)
          generatedImages = mergeOllamaImages(generatedImages, images);
        socket.send(
          JSON.stringify({ type: "ollamaChunk", id, text, thinking, images }),
        );
      }
      if (parsed.done === true) finalStats = parsed;
    };
    try {
      const requestBody = { model, prompt, stream: true };
      if (payload.enableThinking === true)
        requestBody.think = ollamaThinkValue(model);
      if (image) requestBody.images = [image.base64];
      const seed = optionalClampedInteger(payload.seed, 0, 2 ** 31 - 1);
      if (typeof seed === "number") requestBody.options = { seed };
      const res = await fetch(`${ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify(requestBody),
      });
      if (!res.ok || !res.body) {
        const message = await res.text().catch(() => "");
        throw new Error(
          `Ollama request failed (${res.status})${message ? `: ${message}` : "."}`,
        );
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let newline = buffer.indexOf("\n");
        while (newline !== -1) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (line) consume(JSON.parse(line));
          newline = buffer.indexOf("\n");
        }
      }
      const trailing = buffer.trim();
      if (trailing) consume(JSON.parse(trailing));
      const extracted = extractOllamaImagesFromText(fullText);
      if (extracted.images.length > 0) {
        fullText = extracted.text;
        generatedImages = mergeOllamaImages(generatedImages, extracted.images);
      }
      socket.send(
        JSON.stringify({
          type: "ollamaDone",
          id,
          chatId,
          modelId: model,
          text: fullText,
          thinking: fullThinking || undefined,
          images: generatedImages,
          totalDuration: finalStats.total_duration ?? null,
          evalCount: finalStats.eval_count ?? null,
        }),
      );
      responseCompleted = true;
      if (chatId)
        await appendChatMessages(userId, chatId, [
          {
            id,
            role: "user",
            text: prompt,
            image: image?.attachment ?? null,
            provider: "ollama",
            createdAt: userAt,
          },
          {
            id: `${id}:reply`,
            role: "assistant",
            text: fullText,
            images: generatedImages,
            provider: "ollama",
            modelId: model,
            tokenCount: finalStats.eval_count ?? null,
            createdAt: new Date(),
          },
        ]);
    } catch (err) {
      if (controller.signal.aborted) {
        const reason = controller.signal.reason;
        throw new Error(
          reason instanceof Error ? reason.message : "Inference cancelled.",
        );
      }
      throw err;
    } finally {
      unregisterCancel();
      if (responseCompleted || controller.signal.aborted)
        await unloadOllamaModel(model);
      socket.off?.("close", onClose);
    }
  }

  return { listOllamaModels, showOllamaModel, streamOllamaPrompt };
}
