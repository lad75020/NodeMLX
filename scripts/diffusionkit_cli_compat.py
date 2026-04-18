#!/usr/bin/env python3
"""Compatibility launcher for DiffusionKit with newer mlx.core.

Some DiffusionKit releases pass ``memory_efficient_threshold`` to
``mx.fast.scaled_dot_product_attention``. Current MLX versions removed that
keyword, so strip it while leaving the rest of DiffusionKit's low-memory mode
unchanged.
"""

import sys

import mlx.core as mx

_FLUX_4BIT_MODEL = "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized"


def _patch_scaled_dot_product_attention() -> None:
    original = mx.fast.scaled_dot_product_attention

    def compatible_scaled_dot_product_attention(*args, **kwargs):
        kwargs.pop("memory_efficient_threshold", None)
        return original(*args, **kwargs)

    mx.fast.scaled_dot_product_attention = compatible_scaled_dot_product_attention


def _patch_flux_quantized_loader() -> None:
    """Strip redundant FLUX key-projection bias tensors from old checkpoints.

    DiffusionKit's FLUX attention defines k_proj with bias=False because key
    bias is redundant after softmax. The argmaxinc 4-bit FLUX checkpoint still
    includes those bias tensors, which newer mlx.nn rejects as unknown params.
    """
    import mlx.nn as nn
    import diffusionkit.mlx as diffusion_mlx
    from diffusionkit.mlx import model_io
    from diffusionkit.mlx.mmdit import MMDiT
    from huggingface_hub import hf_hub_download
    from mlx.utils import tree_flatten, tree_unflatten

    def compatible_load_flux(
        key: str = "argmaxinc/mlx-FLUX.1-schnell",
        float16: bool = False,
        model_key: str = "argmaxinc/mlx-FLUX.1-schnell",
        low_memory_mode: bool = True,
        only_modulation_dict: bool = False,
    ):
        dtype = model_io._FLOAT16 if float16 else mx.float32
        config = model_io.FLUX_SCHNELL
        config.low_memory_mode = low_memory_mode
        model = MMDiT(config)

        flux_weights = model_io._MMDIT[key][model_key]
        flux_weights_ckpt = model_io.LOCAl_SD3_CKPT or hf_hub_download(
            key, flux_weights
        )
        hf_hub_download(key, "config.json")
        weights = mx.load(flux_weights_ckpt)

        if model_key in (
            "argmaxinc/mlx-FLUX.1-schnell",
            "argmaxinc/mlx-FLUX.1-dev",
        ):
            weights = model_io.flux_state_dict_adjustments(
                weights,
                prefix="",
                hidden_size=config.hidden_size,
                mlp_ratio=config.mlp_ratio,
            )
        elif model_key == _FLUX_4BIT_MODEL:
            nn.quantize(model)

        weights = {
            k: v
            for k, v in weights.items()
            if not k.endswith(".attn.k_proj.bias")
        }
        weights = {
            k: v.astype(dtype) if v.dtype != mx.uint32 else v
            for k, v in weights.items()
        }
        if only_modulation_dict:
            weights = {k: v for k, v in weights.items() if "adaLN" in k}
            return tree_flatten(weights)
        model.update(tree_unflatten(tree_flatten(weights)))

        return model

    if not hasattr(model_io, "_original_load_flux"):
        model_io._original_load_flux = model_io.load_flux

    model_io.load_flux = compatible_load_flux
    diffusion_mlx.load_flux = compatible_load_flux


def main() -> int:
    _patch_scaled_dot_product_attention()
    _patch_flux_quantized_loader()
    from diffusionkit.mlx.scripts.generate_images import cli

    return cli()


if __name__ == "__main__":
    sys.exit(main())
