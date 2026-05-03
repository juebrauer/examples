#!/usr/bin/env python3
"""
SmolVLM2 Video Analysis Example

Uses SmolVLM2 models to analyze a local video file.

Install (example):
    pip install -U torch transformers decord accelerate

Run (example):
    python smolvlm_video_analysis_example.py \
        --video /path/to/video.mp4 \
        --prompt "Explain what happens in this video in detail." \
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Dict

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None


#DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
                    




def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_required_packages() -> bool:
    missing = []
    if torch is None:
        missing.append("torch")
    if AutoProcessor is None or AutoModelForImageTextToText is None:
        missing.append("transformers")

    if missing:
        print(
            "Warning: missing important package(s): " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Install with: pip install -U torch transformers",
            file=sys.stderr,
        )
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a video using SmolVLM2-500M-Video-Instruct."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to local video file (e.g. .mp4).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this video in detail.",
        help="Question/instruction for the model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (only used with --do-sample).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype. 'auto' uses bfloat16 on CUDA, else float32.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        choices=["auto", "flash_attention_2", "eager"],
        default="auto",
        help="Attention implementation for CUDA. 'auto' tries flash_attention_2 first.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    return device_arg


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_arg]


def to_device_preserve_int_tensors(
    inputs: Dict[str, torch.Tensor], device: str, float_dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        tensor = value.to(device)
        if torch.is_floating_point(tensor):
            tensor = tensor.to(float_dtype)
        moved[key] = tensor
    return moved


def main() -> int:
    args = parse_args()

    if not check_required_packages():
        return 1

    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        return 1

    # Decord is required for video loading in transformers for this model family.
    try:
        import decord  # noqa: F401
    except ImportError:
        print(
            "Warning: missing important package 'decord'.",
            file=sys.stderr,
        )
        print(
            "Install with: pip install decord",
            file=sys.stderr,
        )
        return 1

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Loading processor and model: {args.model}")
    print(f"Device: {device} | DType: {dtype}")

    processor = AutoProcessor.from_pretrained(args.model)

    model_kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        use_flash = args.attn_impl in {"auto", "flash_attention_2"}
        if use_flash and not has_module("flash_attn"):
            print(
                "Warning: flash-attn package not found. Falling back to eager attention.",
                file=sys.stderr,
            )
        elif use_flash:
            model_kwargs["_attn_implementation"] = "flash_attention_2"
    elif args.attn_impl == "flash_attention_2":
        print(
            "Warning: --attn-impl flash_attention_2 requires CUDA; using CPU eager attention.",
            file=sys.stderr,
        )

    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except Exception as exc:
        # Fallback when flash-attn is not installed or unsupported.
        if model_kwargs.get("_attn_implementation") == "flash_attention_2":
            print(
                "Warning: flash_attention_2 not available, falling back to eager attention.",
                file=sys.stderr,
            )
            model_kwargs.pop("_attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
        else:
            raise exc

    model = model.to(device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": args.video},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = to_device_preserve_int_tensors(model_inputs, device, dtype)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Model Output ===")
    print(generated_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
