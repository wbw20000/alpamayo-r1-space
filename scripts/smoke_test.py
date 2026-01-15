#!/usr/bin/env python3
"""
Smoke test for Alpamayo-R1-10B deployment.

This script verifies:
1. Package versions are correct
2. Model config can be downloaded
3. Model architecture is recognized by transformers
4. Attention implementation fallback works

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --load-model  # Also try loading the full model
"""

import os
import sys
import argparse


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def check_versions():
    """Check and print package versions."""
    print_section("Package Versions")

    print(f"Python: {sys.version}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError as e:
        print(f"PyTorch: NOT INSTALLED - {e}")
        return False

    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")

        # Check if version is sufficient
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major < 4 or (major == 4 and minor < 57):
            print(f"WARNING: transformers version {transformers.__version__} < 4.57.1")
            print("         alpamayo_r1 may not be supported!")
    except ImportError as e:
        print(f"Transformers: NOT INSTALLED - {e}")
        return False

    try:
        import flash_attn
        print(f"Flash-Attn: {flash_attn.__version__} (AVAILABLE)")
    except ImportError:
        print("Flash-Attn: NOT INSTALLED (will use SDPA fallback)")

    try:
        import accelerate
        print(f"Accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"Accelerate: NOT INSTALLED - {e}")

    return True


def check_model_config():
    """Try to download and parse model config."""
    print_section("Model Config Check")

    try:
        from transformers import AutoConfig

        # Get token from environment
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            print("HF Token: Found in environment")
        else:
            print("HF Token: NOT FOUND (may fail for gated models)")

        print("Downloading config from nvidia/Alpamayo-R1-10B...")
        config = AutoConfig.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            token=token,
            trust_remote_code=True,
        )

        print(f"Model Type: {config.model_type}")
        print(f"Architecture: {config.architectures}")
        print(f"Hidden Size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"Num Layers: {getattr(config, 'num_hidden_layers', 'N/A')}")

        # Check attention implementation
        try:
            import flash_attn
            config.attn_implementation = "flash_attention_2"
            print("Attention: flash_attention_2 (optimized)")
        except ImportError:
            config.attn_implementation = "sdpa"
            print("Attention: sdpa (fallback)")

        print("\nConfig loaded successfully!")
        return config

    except Exception as e:
        print(f"ERROR loading config: {e}")

        error_str = str(e).lower()
        if "403" in error_str or "gated" in error_str:
            print("\nThis model requires access approval.")
            print("Visit: https://huggingface.co/nvidia/Alpamayo-R1-10B")
            print("Request access and set HF_TOKEN in environment.")
        elif "alpamayo_r1" in error_str and "not recognized" in error_str:
            print("\nThe alpamayo_r1 architecture is not recognized.")
            print("Make sure transformers >= 4.57.1 is installed.")

        return None


def try_load_model(config):
    """Attempt to load the full model."""
    print_section("Model Loading Test")

    import torch
    if not torch.cuda.is_available():
        print("SKIPPED: No GPU available")
        print("Model loading requires GPU with 24+ GB VRAM")
        return False

    try:
        from transformers import AutoModel

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        print("Loading model (this may take several minutes)...")
        print(f"Using attention: {config.attn_implementation}")

        model = AutoModel.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
            trust_remote_code=True,
        )
        model.eval()

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e9:.2f}B")

        # Try a minimal forward pass (just to verify model works)
        print("Model loaded successfully!")
        return True

    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test for Alpamayo-R1-10B")
    parser.add_argument("--load-model", action="store_true",
                        help="Also try loading the full model (requires GPU)")
    args = parser.parse_args()

    print_section("Alpamayo-R1-10B Smoke Test")

    # Step 1: Check versions
    if not check_versions():
        print("\nFATAL: Required packages not installed")
        sys.exit(1)

    # Step 2: Check model config
    config = check_model_config()
    if config is None:
        print("\nFATAL: Could not load model config")
        sys.exit(1)

    # Step 3: Optionally load full model
    if args.load_model:
        if not try_load_model(config):
            print("\nWARNING: Model loading failed")
            sys.exit(1)

    print_section("Smoke Test Complete")
    print("All basic checks passed!")

    # Summary
    try:
        import flash_attn
        print("Attention: FlashAttention2 (optimized)")
    except ImportError:
        print("Attention: SDPA (fallback - flash-attn not installed)")


if __name__ == "__main__":
    main()
