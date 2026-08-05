import { ensureHuggingFaceHome } from "./src/config.js";

// Set this before loading the application graph so the model worker inherits it.
ensureHuggingFaceHome();
await import("./src/bootstrap.js");
