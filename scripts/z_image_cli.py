#!/usr/bin/env python3
"""Small CLI adapter for uqer1244/MLX-z-image.

The upstream project is a custom MLX text-to-image pipeline rather than a
DiffusionKit model. This wrapper lets the NodeMLX worker call it with the same
prompt/output arguments used for other image-generation backends.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _default_project_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "MLX_z-image"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MLX z-image pipeline")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="uqer1244/MLX-z-image")
    parser.add_argument("--project-dir", default=os.environ.get("MLX_Z_IMAGE_DIR"))
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser() if args.project_dir else _default_project_dir()
    project_dir = project_dir.resolve()
    if not project_dir.exists():
        raise FileNotFoundError(
            f"MLX_z-image checkout not found at {project_dir}. "
            "Clone https://github.com/uqer1244/MLX_z-image or set MLX_Z_IMAGE_DIR."
        )

    sys.path.insert(0, str(project_dir))
    from mlx_pipeline import ZImagePipeline

    model_path = project_dir / "Z-Image-Turbo-MLX"
    pipeline = ZImagePipeline(
        model_path=str(model_path),
        text_encoder_path=str(model_path / "text_encoder"),
        repo_id=args.model_id,
    )
    image = pipeline.generate(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
    )
    image.save(args.output_path)
    print(f"Image saved to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
