# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Alpamayo-R1-10B Inference Demo

This demo uses the NVlabs/alpamayo package (alpamayo_r1) for model loading.
The package is installed via postBuild script with --no-deps fallback.
"""

# --- hotfix: gradio_client bool schema crash on HF Spaces ---
# Must be applied BEFORE importing gradio
try:
    import gradio_client.utils as gc_utils
    _orig_get_type = gc_utils.get_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _orig_get_type(schema)

    gc_utils.get_type = _patched_get_type
except Exception:
    pass
# --- end hotfix ---

import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# Try to import spaces for ZeroGPU
try:
    import spaces
    ZERO_GPU = True
except (ImportError, Exception):
    ZERO_GPU = False

# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Global model variables
model = None
processor = None
_attn_implementation = None  # Track which attention implementation is used
_alpamayo_available = False  # Track if alpamayo_r1 package is available


# ========== Runtime Installation of alpamayo_r1 ==========
def _install_alpamayo_if_missing():
    """
    Install NVlabs/alpamayo package at runtime if not available.
    This is a fallback in case postBuild didn't run or failed.
    """
    try:
        import alpamayo_r1
        print(f"[INFO] alpamayo_r1 already installed at: {alpamayo_r1.__file__}")
        return True
    except ImportError:
        print("[WARNING] alpamayo_r1 not found, attempting runtime installation...")

    import subprocess

    # Try full install first
    try:
        print("[INFO] Attempting full pip install of NVlabs/alpamayo...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "git+https://github.com/NVlabs/alpamayo.git@main"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("[INFO] Full install succeeded!")
            return True
        else:
            print(f"[WARNING] Full install failed: {result.stderr[:500]}")
    except Exception as e:
        print(f"[WARNING] Full install exception: {e}")

    # Fallback to --no-deps
    try:
        print("[INFO] Attempting --no-deps install...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
             "git+https://github.com/NVlabs/alpamayo.git@main"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("[INFO] --no-deps install succeeded!")
            return True
        else:
            print(f"[ERROR] --no-deps install failed: {result.stderr[:500]}")
    except Exception as e:
        print(f"[ERROR] --no-deps install exception: {e}")

    return False


# Run installation check at startup
_install_alpamayo_if_missing()


# ========== alpamayo_r1 Registration & Integration ==========
def _register_alpamayo_with_transformers():
    """
    Import alpamayo_r1 package and ensure it's registered with transformers.
    This is called at module load time to trigger any HF integration.
    """
    global _alpamayo_available

    print("[INFO] Checking alpamayo_r1 package availability...")

    try:
        import alpamayo_r1
        print(f"[INFO] alpamayo_r1 found at: {alpamayo_r1.__file__}")
        _alpamayo_available = True

        # Try to import HF integration modules if they exist
        try:
            from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
            print("[INFO] AlpamayoR1 class available")
        except ImportError as e:
            print(f"[WARNING] AlpamayoR1 import failed: {e}")

        try:
            from alpamayo_r1 import helper
            print("[INFO] alpamayo_r1.helper module available")
        except ImportError as e:
            print(f"[WARNING] alpamayo_r1.helper import failed: {e}")

        # Check if alpamayo_r1 registered itself with transformers AutoConfig
        try:
            from transformers import AutoConfig
            # This will fail if alpamayo_r1 is not registered
            # We don't actually load the config here, just check registration
            print("[INFO] transformers AutoConfig available for alpamayo_r1 check")
        except Exception as e:
            print(f"[WARNING] AutoConfig check failed: {e}")

        return True

    except ImportError as e:
        print(f"[ERROR] alpamayo_r1 package not available: {e}")
        print("[ERROR] Please ensure postBuild script ran successfully")
        _alpamayo_available = False
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error importing alpamayo_r1: {e}")
        _alpamayo_available = False
        return False


# Run registration at module load time
_register_alpamayo_with_transformers()

# Print startup info
print("=" * 60)
print("Alpamayo-R1-10B Inference Demo - Startup Info")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: NOT INSTALLED")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"ZeroGPU Mode: {ZERO_GPU}")
print(f"alpamayo_r1 Available: {_alpamayo_available}")
print("=" * 60)


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return token


def get_system_info():
    """Get system and package version information."""
    import transformers

    info_lines = [
        f"Python: {sys.version.split()[0]}",
        f"PyTorch: {torch.__version__}",
        f"Transformers: {transformers.__version__}",
        f"CUDA Available: {torch.cuda.is_available()}",
    ]

    if torch.cuda.is_available():
        info_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        info_lines.append(f"CUDA Version: {torch.version.cuda}")

    return "\n".join(info_lines)


def check_flash_attn():
    """Check if flash-attn is available and return appropriate attention implementation."""
    try:
        import flash_attn
        print(f"[INFO] flash-attn version: {flash_attn.__version__}")
        return "flash_attention_2"
    except ImportError:
        print("[WARNING] flash-attn not available, falling back to SDPA")
        return "sdpa"


def load_model_impl():
    """Load the Alpamayo-R1-10B model with proper error handling."""
    global model, processor, _attn_implementation, _alpamayo_available

    if model is not None:
        return f"Model already loaded!\n\nAttention: {_attn_implementation}\n\n{get_system_info()}"

    try:
        # Print system info
        sys_info = get_system_info()
        print(f"[INFO] System Info:\n{sys_info}")

        # Get HF token and set it in environment for huggingface_hub
        token = get_hf_token()
        if token:
            os.environ["HF_TOKEN"] = token
            print("[INFO] HF_TOKEN found and set.")
        else:
            print("[WARNING] No HF_TOKEN found. If model is gated, authentication will fail.")

        # Use eager attention - AlpamayoR1 doesn't support SDPA yet
        _attn_implementation = "eager"
        print(f"[INFO] Using attention implementation: {_attn_implementation}")

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        if not torch.cuda.is_available():
            return (
                "ERROR: No GPU available.\n\n"
                "This model requires a GPU with at least 24GB VRAM.\n"
                "On HuggingFace Spaces, make sure ZeroGPU is properly configured.\n\n"
                f"{sys_info}"
            )

        # Check if alpamayo_r1 package is available
        if not _alpamayo_available:
            return (
                "ERROR: alpamayo_r1 package not available.\n\n"
                "The postBuild script may have failed to install NVlabs/alpamayo.\n"
                "Please check the Space build logs.\n\n"
                f"{sys_info}"
            )

        # Load using NVlabs AlpamayoR1 class (primary method)
        print("[INFO] Loading Alpamayo-R1-10B model using NVlabs AlpamayoR1...")

        # Correct import path as per NVlabs/alpamayo test_inference.py
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1 import helper

        # Load model with bfloat16 precision
        # Note: AlpamayoR1 doesn't support SDPA yet, use "eager" instead
        print("[INFO] Calling AlpamayoR1.from_pretrained...")
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            dtype=torch.bfloat16,
            attn_implementation="eager",  # AlpamayoR1 doesn't support SDPA
        )

        # Move to device
        print(f"[INFO] Moving model to {device}...")
        model = model.to(device)
        model.eval()
        print("[INFO] Model loaded successfully using AlpamayoR1!")

        # Load processor using helper
        print("[INFO] Loading processor...")
        processor = helper.get_processor(model.tokenizer)
        print("[INFO] Processor loaded successfully!")

        # Build success message
        return (
            f"Model loaded successfully on {device}!\n\n"
            f"Attention Implementation: eager\n"
            f"Model Type: AlpamayoR1 (NVlabs/alpamayo)\n\n"
            f"{sys_info}"
        )

    except ImportError as ie:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Import error during model loading:\n{error_trace}")

        return (
            f"ERROR: Missing dependencies.\n\n"
            f"Import Error: {ie}\n\n"
            f"Please check if alpamayo_r1 package is properly installed.\n"
            f"Build logs may have more details.\n\n"
            f"{error_trace}"
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Model loading failed:\n{error_trace}")

        # Check for common errors
        error_str = str(e).lower()
        if "403" in error_str or "gated" in error_str or "access" in error_str:
            return (
                "ACCESS DENIED: This model requires approval.\n\n"
                "Please visit https://huggingface.co/nvidia/Alpamayo-R1-10B "
                "and request access. After approval, make sure your HF_TOKEN "
                "is set in Space Secrets.\n\n"
                f"Error: {e}"
            )

        if "out of memory" in error_str or "oom" in error_str:
            return (
                "ERROR: GPU Out of Memory.\n\n"
                "This model requires at least 24GB VRAM.\n"
                "The ZeroGPU A10G has 24GB, but other processes may be using memory.\n\n"
                f"Error: {e}"
            )

        return f"Error loading model: {str(e)}\n\n{error_trace}"


# Apply ZeroGPU decorator if available
if ZERO_GPU:
    load_model = spaces.GPU(duration=300)(load_model_impl)
else:
    load_model = load_model_impl


def visualize_trajectory(trajectory, camera_image=None):
    """Visualize predicted trajectory."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]

    ax1.plot(y_coords, x_coords, 'b-', linewidth=2, label='Predicted Trajectory')
    ax1.scatter(y_coords[0], x_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(y_coords[-1], x_coords[-1], c='red', s=100, marker='*', label='End (6.4s)', zorder=5)
    ax1.scatter([0], [0], c='orange', s=200, marker='s', label='Ego Vehicle', zorder=10)

    ax1.set_xlabel('Lateral (m)', fontsize=12)
    ax1.set_ylabel('Longitudinal (m)', fontsize=12)
    ax1.set_title("Bird's Eye View - Trajectory Prediction", fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    ax2 = axes[1]
    if camera_image is not None:
        ax2.imshow(camera_image)
        ax2.set_title('Front Camera View', fontsize=14)
        ax2.axis('off')
    else:
        time_steps = np.arange(len(x_coords)) * 0.1
        ax2.plot(time_steps, x_coords, 'r-', label='X (forward)', linewidth=2)
        ax2.plot(time_steps, y_coords, 'g-', label='Y (lateral)', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Position (m)', fontsize=12)
        ax2.set_title('Trajectory Components Over Time', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_image = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    return result_image


def run_inference_impl(front_wide, front_tele, left, right, user_command, num_samples, temperature, top_p):
    """Run inference with the Alpamayo-R1-10B model.

    Note: This demo shows model loading capability. Full trajectory inference
    requires properly formatted driving data (multi-frame camera sequences,
    ego-motion history) from datasets like Physical-AI-AV.
    """
    global model, processor

    # Auto-load model if not loaded (handles ZeroGPU session resets)
    if model is None:
        print("[INFO] Model not loaded, auto-loading...")
        load_result = load_model_impl()
        print(f"[INFO] Auto-load result: {load_result[:100]}...")
        if model is None:
            return (f"Failed to auto-load model:\n\n{load_result}", None, "Error: Model failed to load")

    if all(img is None for img in [front_wide, front_tele, left, right]):
        return ("Please upload at least one camera image.", None, "Error: No images provided")

    try:
        # Preprocess images for visualization
        target_size = (576, 320)
        camera_images = {}
        camera_names = ["front_wide", "front_tele", "left", "right"]
        images = [front_wide, front_tele, left, right]

        for name, img in zip(camera_names, images):
            if img is not None:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                camera_images[name] = img
            else:
                camera_images[name] = Image.new("RGB", target_size, (0, 0, 0))

        if not user_command or not user_command.strip():
            user_command = "Drive safely and follow traffic rules."

        # For this demo, we show that the model is loaded and ready
        # Full inference requires properly formatted Physical-AI-AV dataset
        device = next(model.parameters()).device

        # Generate demo trajectory visualization
        t = np.linspace(0, 6.4, 64)
        trajectory = np.zeros((64, 12))
        trajectory[:, 0] = t * 5  # Forward motion
        trajectory[:, 1] = 0.5 * np.sin(t * 0.5)  # Slight lateral motion
        trajectory[:, 3] = 1.0  # Quaternion w
        trajectory[:, 7] = 1.0
        trajectory[:, 11] = 1.0

        vis_image = visualize_trajectory(trajectory, front_wide)

        reasoning_output = f"""## Model Status: Ready for Inference

**Model**: NVIDIA Alpamayo-R1-10B (VLA for Autonomous Driving)
**Device**: {device}
**Attention**: eager implementation
**Command**: {user_command}

---
## Demo Mode Information

This Space demonstrates successful model loading of Alpamayo-R1-10B.

**For full trajectory inference**, the model requires:
- Multi-frame camera sequences (4 cameras × 4 frames = 16 images)
- Ego-motion history (position and rotation)
- Properly formatted data from Physical-AI-AV dataset

**Visualization below shows a sample trajectory format.**

---
## Sample Trajectory Statistics
- **Prediction Horizon**: 6.4 seconds (64 waypoints @ 10Hz)
- **Forward Distance**: {trajectory[-1, 0]:.2f} m
- **Lateral Offset**: {trajectory[-1, 1]:.2f} m

For research use, please refer to:
- [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)
- [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
"""

        return (reasoning_output, vis_image, "Demo completed - Model loaded successfully!")

    except Exception as e:
        import traceback
        error_msg = f"Error during inference: {str(e)}\n\n{traceback.format_exc()}"
        return (error_msg, None, f"Error: {str(e)}")


# Apply ZeroGPU decorator if available
if ZERO_GPU:
    run_inference = spaces.GPU(duration=120)(run_inference_impl)
else:
    run_inference = run_inference_impl


# Create the interface
with gr.Blocks(title="Alpamayo-R1-10B Inference Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # NVIDIA Alpamayo-R1-10B Inference Demo

    A Vision-Language-Action (VLA) model for autonomous driving that provides:
    - **Chain-of-Causation Reasoning**: Interpretable decision-making process
    - **Trajectory Prediction**: 6.4-second future path prediction at 10Hz

    **Note**: Click "Load Model" to start (requires GPU and model access approval).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Control")
            load_btn = gr.Button("Load Model", variant="primary", size="lg")
            model_status = gr.Textbox(
                label="Model Status",
                value="Model not loaded\n\nClick 'Load Model' to initialize.",
                interactive=False,
                lines=8
            )

            gr.Markdown("### Camera Inputs")
            with gr.Row():
                front_wide = gr.Image(label="Front Wide (Required)", type="pil", height=180)
                front_tele = gr.Image(label="Front Telephoto", type="pil", height=180)

            with gr.Row():
                left_cam = gr.Image(label="Left Camera", type="pil", height=180)
                right_cam = gr.Image(label="Right Camera", type="pil", height=180)

            gr.Markdown("### Driving Command")
            user_command = gr.Textbox(
                label="Command (optional)",
                placeholder="E.g., 'Turn left at the next intersection'",
                value="Drive safely and follow traffic rules."
            )

            gr.Markdown("### Inference Parameters")
            with gr.Row():
                num_samples = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Samples")

            with gr.Row():
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")

            run_btn = gr.Button("Run Inference", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            status_output = gr.Textbox(label="Status", value="Ready", interactive=False)
            reasoning_output = gr.Markdown(label="Reasoning", value="Results will appear here...")
            trajectory_output = gr.Image(label="Trajectory Visualization", type="pil", height=400)

    load_btn.click(fn=load_model, outputs=[model_status])

    run_btn.click(
        fn=run_inference,
        inputs=[front_wide, front_tele, left_cam, right_cam, user_command, num_samples, temperature, top_p],
        outputs=[reasoning_output, trajectory_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
