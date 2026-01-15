---
title: Alpamayo-R1-10B Inference Demo
emoji: 🚗
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.27.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
---

# NVIDIA Alpamayo-R1-10B Inference Demo

A Gradio-based demo for **NVIDIA Alpamayo-R1-10B**, a Vision-Language-Action (VLA) model for autonomous driving.

## Features

- **Multi-Camera Input**: Support for 4 camera views (front wide, front tele, left, right)
- **Chain-of-Causation Reasoning**: Interpretable decision-making traces
- **Trajectory Prediction**: 6.4-second future path prediction at 10Hz (64 waypoints)
- **Interactive Visualization**: Bird's eye view trajectory plots

## Model Overview

Alpamayo 1 (formerly Alpamayo-R1) is NVIDIA's open 10B-parameter reasoning VLA model that:

- Takes multi-camera video input (4 cameras × 4 frames @ 10Hz)
- Outputs driving trajectories and reasoning traces
- Uses Chain-of-Causation (CoC) reasoning for interpretable decisions
- Built on Cosmos-Reason VLM backbone (8.2B params) + Action Expert (2.3B params)

## Hardware Requirements

- **GPU**: Uses ZeroGPU (NVIDIA H200 with 70GB VRAM)
- **Model Size**: ~22GB (preloaded during build)

## Usage

1. Click **"Load Model"** to initialize
2. Upload camera images (at least front wide camera)
3. Optionally enter a driving command
4. Click **"Run Inference"**
5. View reasoning traces and trajectory visualization

## License

- **Model Weights**: Non-commercial use only
- **Inference Code**: Apache 2.0

## References

- [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [GitHub: NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)
