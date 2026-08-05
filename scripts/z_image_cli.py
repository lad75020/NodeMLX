#!/usr/bin/env python3
"""CLI adapter for the uqer1244/MLX_z-image custom MLX pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

MODEL_ID = "uqer1244/MLX-z-image"
MODEL_CACHE_NAME = "models--uqer1244--MLX-z-image"


def _default_project_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "MLX_z-image"


def _snapshot_from_cache(candidate: Path) -> Path | None:
    """Return a usable model snapshot from a Hugging Face cache or snapshot."""
    candidate = candidate.expanduser()
    if (candidate / "model_index.json").is_file():
        return candidate.resolve()

    if candidate.name != MODEL_CACHE_NAME:
        candidate = candidate / MODEL_CACHE_NAME
    snapshots = candidate / "snapshots"
    if not snapshots.is_dir():
        return None

    revision_file = candidate / "refs" / "main"
    if revision_file.is_file():
        revision = revision_file.read_text(encoding="utf-8").strip()
        snapshot = snapshots / revision
        if (snapshot / "model_index.json").is_file():
            return snapshot.resolve()

    for snapshot in sorted(snapshots.iterdir(), reverse=True):
        if (snapshot / "model_index.json").is_file():
            return snapshot.resolve()
    return None


def resolve_model_dir(
    configured_dir: str | None,
    project_dir: Path,
    model_id: str = MODEL_ID,
) -> Path:
    """Locate z-image weights without duplicating a local HF cache."""
    del model_id  # The custom upstream pipeline currently supports this model only.

    candidates: list[Path] = []
    if configured_dir:
        candidates.append(Path(configured_dir))

    # A source checkout that already has the model beside its pipeline remains
    # supported, matching the upstream project's default layout.
    candidates.append(project_dir / "Z-Image-Turbo-MLX")

    for variable in ("MLX_Z_IMAGE_CACHE_DIR", "HF_HUB_CACHE"):
        if value := os.environ.get(variable):
            candidates.append(Path(value))
    if hf_home := os.environ.get("HF_HOME"):
        candidates.extend((Path(hf_home) / "hub", Path(hf_home)))

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    # NodeMLX installations on this host keep large model caches on the
    # external volume. This optional candidate is ignored on other machines.
    candidates.append(Path("/Volumes/WDBlack4TB/HFModels"))

    for candidate in candidates:
        if model_dir := _snapshot_from_cache(candidate):
            return model_dir

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "MLX z-image model files were not found. Set MLX_Z_IMAGE_MODEL_DIR to "
        "the model snapshot or cache directory. Searched:\n  - " + searched
    )


def _load_pipeline(project_dir: Path):
    sys.path.insert(0, str(project_dir))
    try:
        from mlx_pipeline import ZImagePipeline
    except ModuleNotFoundError as error:
        if error.name != "qkv_fusion_debug":
            raise

        # Upstream master imports this optional debugging module even though
        # it is absent from the repository and unused by the pipeline.
        shim = types.ModuleType("qkv_fusion_debug")
        shim.inspect_attention = lambda *args, **kwargs: None
        shim.try_fuse_qkv = lambda *args, **kwargs: None
        sys.modules["qkv_fusion_debug"] = shim
        from mlx_pipeline import ZImagePipeline
    return ZImagePipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MLX z-image pipeline")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--project-dir", default=os.environ.get("MLX_Z_IMAGE_DIR"))
    parser.add_argument("--model-dir", default=os.environ.get("MLX_Z_IMAGE_MODEL_DIR"))
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser() if args.project_dir else _default_project_dir()
    project_dir = project_dir.resolve()
    if not (project_dir / "mlx_pipeline.py").is_file():
        raise FileNotFoundError(
            f"MLX_z-image source checkout not found at {project_dir}. "
            "Clone https://github.com/uqer1244/MLX_z-image or set MLX_Z_IMAGE_DIR."
        )

    model_dir = resolve_model_dir(args.model_dir, project_dir, args.model_id)
    print(f"Using MLX z-image model files at {model_dir}")
    ZImagePipeline = _load_pipeline(project_dir)
    pipeline = ZImagePipeline(
        model_path=str(model_dir),
        text_encoder_path=str(model_dir / "text_encoder"),
        repo_id=args.model_id,
    )
    image = pipeline.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
