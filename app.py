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
import time
import shutil
import torch
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try to import spaces for ZeroGPU
try:
    import spaces
    ZERO_GPU = True
except (ImportError, Exception):
    ZERO_GPU = False

# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ========== Constants ==========
CAMERAS = ["front_wide", "front_tele", "cross_left", "cross_right"]
FRAMES_PER_CAMERA = 4
TOTAL_IMAGES = 16
CACHE_DIR = Path("./cache/sample_data")
CACHE_LIMIT_GB = 2.0
SAMPLE_REPO_ID = "dgural/PhysicalAI-Autonomous-Vehicles-Sample"
OFFICIAL_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

# Global model variables
model = None
processor = None
_attn_implementation = None
_alpamayo_available = False

# Sample pool for rotation (global state)
_sample_pool: List[str] = []  # List of video files
_sample_idx: int = 0  # Current index in pool


# ========== Cache Management ==========
def get_dir_size_gb(path: Path) -> float:
    """Get directory size in GB."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 ** 3)


def cleanup_cache_if_needed(limit_gb: float = CACHE_LIMIT_GB):
    """Remove oldest sample directories if cache exceeds limit."""
    if not CACHE_DIR.exists():
        return

    current_size = get_dir_size_gb(CACHE_DIR)
    if current_size <= limit_gb:
        return

    # Get all subdirectories with their mtime
    subdirs = []
    for d in CACHE_DIR.iterdir():
        if d.is_dir():
            subdirs.append((d, d.stat().st_mtime))

    # Sort by mtime (oldest first)
    subdirs.sort(key=lambda x: x[1])

    # Remove oldest until under limit
    for subdir, _ in subdirs:
        if get_dir_size_gb(CACHE_DIR) <= limit_gb:
            break
        try:
            shutil.rmtree(subdir)
            print(f"[CACHE] Removed old cache: {subdir}")
        except Exception as e:
            print(f"[CACHE] Failed to remove {subdir}: {e}")


def get_cache_status() -> str:
    """Get cache directory status."""
    if not CACHE_DIR.exists():
        return "Cache: empty"

    size_gb = get_dir_size_gb(CACHE_DIR)
    file_count = sum(1 for _ in CACHE_DIR.rglob("*") if _.is_file())
    return f"Cache: {size_gb:.2f}GB / {CACHE_LIMIT_GB}GB, {file_count} files"


# ========== Input Validation ==========
def validate_inputs(sample_bundle: Optional[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate sample bundle for real inference.
    Returns: (is_valid, missing_keys)

    Required for real inference:
    - ego_motion: actual vehicle motion history
    - timestamps: precise frame timestamps
    - calibration: camera intrinsics/extrinsics
    - proper_sequence: temporally coherent frames
    """
    if sample_bundle is None:
        return False, ["sample_bundle"]

    missing_keys = []
    meta = sample_bundle.get("meta", {})

    # Check for ego_motion (real vehicle telemetry)
    if "ego_motion" not in meta or meta.get("ego_motion") is None:
        missing_keys.append("ego_motion")

    # Check for precise timestamps
    timestamps = meta.get("timestamps", [])
    if not timestamps or timestamps == [i * 0.1 for i in range(FRAMES_PER_CAMERA)]:
        missing_keys.append("timestamps (using placeholder)")

    # Check for camera calibration
    if "calibration" not in meta or meta.get("calibration") is None:
        missing_keys.append("calibration")

    # Check if data is degraded (placeholders used)
    if meta.get("degraded", False):
        missing_keys.append("proper_sequence (degraded input)")

    is_valid = len(missing_keys) == 0
    return is_valid, missing_keys


# ========== Official Mode Functions ==========
def fetch_official_sidecars(clip_id: str, chunk: int = None) -> Dict[str, Any]:
    """
    Download sidecar data (calibration, egomotion) for Official mode.
    Returns: {
        'calibration': Dict or None,
        'egomotion': pd.DataFrame or None (GT trajectory),
        'available': bool,
        'errors': List[str]
    }
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import zipfile
    import tempfile

    result = {
        'calibration': None,
        'egomotion': None,
        'available': False,
        'errors': []
    }

    token = get_hf_token()
    if not token:
        result['errors'].append("HF_TOKEN required for Official dataset access")
        return result

    # Get chunk number if not provided
    if chunk is None:
        try:
            index_path = hf_hub_download(
                repo_id=OFFICIAL_REPO_ID,
                filename="clip_index.parquet",
                repo_type="dataset",
                token=token
            )
            index_df = pd.read_parquet(index_path)
            if clip_id in index_df.index:
                chunk = int(index_df.loc[clip_id, 'chunk'])
            else:
                result['errors'].append(f"clip_id {clip_id} not found in index")
                return result
        except Exception as e:
            result['errors'].append(f"Failed to get chunk: {e}")
            return result

    try:
        # Download calibration (camera_intrinsics)
        try:
            calib_filename = f"calibration/camera_intrinsics/camera_intrinsics.chunk_{chunk:04d}.parquet"
            calib_path = hf_hub_download(
                repo_id=OFFICIAL_REPO_ID,
                filename=calib_filename,
                repo_type="dataset",
                token=token
            )
            calib_df = pd.read_parquet(calib_path)

            # Calibration parquet uses MultiIndex: (clip_id, camera_name)
            # Check if clip_id is in index
            if calib_df.index.nlevels >= 2:
                # MultiIndex case
                clip_ids_in_index = calib_df.index.get_level_values(0).unique()
                if clip_id in clip_ids_in_index:
                    clip_calib = calib_df.loc[clip_id]
                    # Convert to dict with camera_name as key
                    result['calibration'] = {}
                    for cam_name in clip_calib.index:
                        result['calibration'][cam_name] = clip_calib.loc[cam_name].to_dict()
                    print(f"[OFFICIAL] Loaded calibration for {len(result['calibration'])} cameras")
            elif 'clip_id' in calib_df.columns:
                # Fallback: clip_id as column
                clip_calib = calib_df[calib_df['clip_id'] == clip_id]
                if len(clip_calib) > 0:
                    result['calibration'] = clip_calib.to_dict('records')[0]
            print(f"[OFFICIAL] Loaded calibration for chunk {chunk}")
        except Exception as e:
            result['errors'].append(f"calibration: {str(e)[:50]}")

        # Download egomotion (GT trajectory) - THIS IS AVAILABLE!
        try:
            ego_filename = f"labels/egomotion/egomotion.chunk_{chunk:04d}.zip"
            print(f"[OFFICIAL] Downloading egomotion from {ego_filename}...")
            ego_zip_path = hf_hub_download(
                repo_id=OFFICIAL_REPO_ID,
                filename=ego_filename,
                repo_type="dataset",
                token=token
            )

            # Extract egomotion parquet from zip
            ego_parquet_name = f"{clip_id}.egomotion.parquet"
            with zipfile.ZipFile(ego_zip_path, 'r') as zf:
                if ego_parquet_name in zf.namelist():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zf.extract(ego_parquet_name, tmpdir)
                        ego_path = f"{tmpdir}/{ego_parquet_name}"
                        result['egomotion'] = pd.read_parquet(ego_path)
                        print(f"[OFFICIAL] Loaded egomotion: {len(result['egomotion'])} rows")
                else:
                    result['errors'].append(f"egomotion parquet not found in zip")
        except Exception as e:
            result['errors'].append(f"egomotion: {str(e)[:50]}")

        result['available'] = result['calibration'] is not None or result['egomotion'] is not None

    except Exception as e:
        result['errors'].append(f"Official dataset access error: {e}")

    return result


def fetch_official_images(clip_id: str, t0_us: float = 5100000) -> Dict[str, Any]:
    """
    Download images from Official dataset for a specific clip_id.

    Downloads zip files from HuggingFace Hub and extracts the specific video.
    NOTE: This is a slow operation (zip files are 1-2 GB each).
    Should be called OUTSIDE of GPU-decorated functions to avoid timeout.

    Args:
        clip_id: UUID of the clip
        t0_us: Starting timestamp in microseconds

    Returns: {
        'images': List of 16 PIL images (4 cameras x 4 frames),
        'camera_images': Dict mapping camera name to list of images,
        'chunk': int chunk number,
        'available': bool,
        'errors': List[str]
    }
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import zipfile
    import tempfile

    result = {
        'images': [],
        'camera_images': {},
        'chunk': None,
        'available': False,
        'errors': []
    }

    token = get_hf_token()
    if not token:
        result['errors'].append("HF_TOKEN required")
        return result

    try:
        # Step 1: Get chunk number from clip_index.parquet
        print(f"[OFFICIAL] Fetching clip_index.parquet to find chunk for {clip_id}...")
        index_path = hf_hub_download(
            repo_id=OFFICIAL_REPO_ID,
            filename="clip_index.parquet",
            repo_type="dataset",
            token=token
        )

        index_df = pd.read_parquet(index_path)
        if clip_id not in index_df.index:
            result['errors'].append(f"clip_id {clip_id} not found in dataset")
            return result

        chunk = int(index_df.loc[clip_id, 'chunk'])
        result['chunk'] = chunk
        print(f"[OFFICIAL] Found clip in chunk {chunk}")

        # Step 2: Download zip files and extract videos
        # Camera mapping for Alpamayo (4 cameras)
        camera_names = {
            'front_wide': 'camera_front_wide_120fov',
            'cross_left': 'camera_cross_left_120fov',
            'cross_right': 'camera_cross_right_120fov',
            'rear_tele': 'camera_rear_tele_30fov'
        }

        all_images = []
        camera_images = {cam: [] for cam in camera_names.keys()}

        for cam_key, cam_name in camera_names.items():
            try:
                # Download chunk zip file (uses HF cache, so subsequent calls are fast)
                zip_filename = f"camera/{cam_name}/{cam_name}.chunk_{chunk:04d}.zip"
                print(f"[OFFICIAL] Downloading {zip_filename}... (cached if already downloaded)")

                zip_path = hf_hub_download(
                    repo_id=OFFICIAL_REPO_ID,
                    filename=zip_filename,
                    repo_type="dataset",
                    token=token
                )
                print(f"[OFFICIAL] Got zip for {cam_key}: {zip_path}")

                # Extract video from zip
                video_name = f"{clip_id}.{cam_name}.mp4"
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    if video_name not in zf.namelist():
                        result['errors'].append(f"{cam_key}: video {video_name} not in zip")
                        # Use placeholder images
                        for _ in range(FRAMES_PER_CAMERA):
                            placeholder = Image.new('RGB', (1920, 1080), color=(128, 128, 128))
                            camera_images[cam_key].append(placeholder)
                            all_images.append(placeholder)
                        continue

                    # Extract video to temp file
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zf.extract(video_name, tmpdir)
                        video_path = f"{tmpdir}/{video_name}"

                        # Extract frames from video using cv2
                        try:
                            import cv2
                            cap = cv2.VideoCapture(video_path)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                            # Calculate frame indices for 4 frames at 2Hz (0.5s apart)
                            start_frame = int((t0_us / 1e6) * fps) if fps > 0 else 0
                            frame_interval = int(fps / 2) if fps > 0 else 15  # 2Hz

                            frames_extracted = []
                            for i in range(FRAMES_PER_CAMERA):
                                frame_idx = min(start_frame + i * frame_interval, total_frames - 1)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, frame = cap.read()
                                if ret:
                                    # Convert BGR to RGB
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    img = Image.fromarray(frame_rgb)
                                    frames_extracted.append(img)
                                else:
                                    frames_extracted.append(Image.new('RGB', (1920, 1080), color=(100, 100, 100)))

                            cap.release()

                            camera_images[cam_key] = frames_extracted
                            all_images.extend(frames_extracted)
                            print(f"[OFFICIAL] Extracted {len(frames_extracted)} frames from {cam_key}")

                        except ImportError:
                            result['errors'].append("cv2 not available")
                            for _ in range(FRAMES_PER_CAMERA):
                                placeholder = Image.new('RGB', (1920, 1080), color=(128, 128, 128))
                                camera_images[cam_key].append(placeholder)
                                all_images.append(placeholder)

            except Exception as cam_err:
                result['errors'].append(f"{cam_key}: {str(cam_err)[:50]}")
                for _ in range(FRAMES_PER_CAMERA):
                    placeholder = Image.new('RGB', (1920, 1080), color=(128, 128, 128))
                    camera_images[cam_key].append(placeholder)
                    all_images.append(placeholder)

        result['images'] = all_images
        result['camera_images'] = camera_images
        result['available'] = len(all_images) == TOTAL_IMAGES

        print(f"[OFFICIAL] Total images: {len(all_images)}, Available: {result['available']}")

    except Exception as e:
        import traceback
        result['errors'].append(f"fetch_official_images error: {e}")
        print(f"[OFFICIAL] Error: {traceback.format_exc()}")

    return result


def compute_minade6(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """
    Compute minADE6@6.4s metric.

    Args:
        pred_xyz: (B, 1, num_samples, 64, 3) - predicted trajectories
        gt_xyz: (B, 1, 64, 3) - ground truth trajectory

    Returns:
        minADE value in meters
    """
    # Extract XY coordinates
    gt_xy = gt_xyz.cpu()[0, 0, :, :2].T.numpy()  # (2, 64)
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)  # (num_samples, 2, 64)

    # Compute L2 distance, mean over time
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)  # (num_samples,)

    return float(diff.min())


def project_trajectory_to_camera(
    trajectory_xyz: np.ndarray,
    camera_extrinsic: Optional[Dict],
    camera_intrinsic: Optional[Dict],
    image_size: Tuple[int, int] = (1920, 1080)
) -> np.ndarray:
    """
    Project 3D trajectory to camera image plane using f-theta model.

    Args:
        trajectory_xyz: (N, 3) trajectory in ego vehicle frame
        camera_extrinsic: {qx, qy, qz, qw, x, y, z} or None
        camera_intrinsic: {cx, cy, fw_poly_0~4, width, height} or None
        image_size: (width, height) fallback

    Returns:
        (N, 2) image coordinates (u, v), invalid points marked as (-1, -1)
    """
    from scipy.spatial.transform import Rotation

    N = len(trajectory_xyz)
    result = np.full((N, 2), -1.0)

    if camera_extrinsic is None or camera_intrinsic is None:
        # Fallback: simple projection without calibration
        # Assume camera at (0, 0, 1.5) looking forward
        x, y, z = trajectory_xyz[:, 0], trajectory_xyz[:, 1], trajectory_xyz[:, 2]
        valid = x > 0.5  # Only points in front of ego

        if np.any(valid):
            # Simple pinhole projection
            w, h = image_size
            fx = fy = w * 0.8  # Approximate focal length
            cx, cy = w / 2, h / 2

            u = cx - (y[valid] / x[valid]) * fx
            v = cy - ((z[valid] - 1.5) / x[valid]) * fy

            # Clip to image bounds
            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)

            result[valid, 0] = u
            result[valid, 1] = v

        return result

    # Full projection with calibration
    try:
        # 1. Transform from ego frame to camera frame
        cam_rot = Rotation.from_quat([
            camera_extrinsic.get('qx', 0),
            camera_extrinsic.get('qy', 0),
            camera_extrinsic.get('qz', 0),
            camera_extrinsic.get('qw', 1)
        ])
        cam_pos = np.array([
            camera_extrinsic.get('x', 0),
            camera_extrinsic.get('y', 0),
            camera_extrinsic.get('z', 1.5)
        ])

        # Transform to camera coordinates
        traj_cam = cam_rot.inv().apply(trajectory_xyz - cam_pos)

        # 2. Project using f-theta model
        x, y, z = traj_cam[:, 0], traj_cam[:, 1], traj_cam[:, 2]
        valid = z > 0.1  # Points in front of camera

        if np.any(valid):
            theta = np.arctan2(np.sqrt(x[valid]**2 + y[valid]**2), z[valid])

            # Get f-theta polynomial coefficients
            fw_poly = [camera_intrinsic.get(f'fw_poly_{i}', 0) for i in range(5)]
            if fw_poly[1] == 0:  # No focal length info
                fw_poly = [0, 500, 0, 0, 0]  # Default

            # Compute distorted radius
            r = sum(c * theta**i for i, c in enumerate(fw_poly))

            # Compute image coordinates
            phi = np.arctan2(y[valid], x[valid])
            cx = camera_intrinsic.get('cx', image_size[0] / 2)
            cy = camera_intrinsic.get('cy', image_size[1] / 2)

            u = cx + r * np.cos(phi)
            v = cy + r * np.sin(phi)

            # Clip to image bounds
            w = camera_intrinsic.get('width', image_size[0])
            h = camera_intrinsic.get('height', image_size[1])
            u = np.clip(u, 0, w - 1)
            v = np.clip(v, 0, h - 1)

            result[valid, 0] = u
            result[valid, 1] = v

    except Exception as e:
        print(f"[PROJECTION] Error: {e}")

    return result


def visualize_trajectory_overlay(
    camera_image: Image.Image,
    trajectory_2d: np.ndarray,
    trajectory_xyz: np.ndarray,
    color: str = 'lime',
    title: str = "Trajectory Overlay"
) -> Image.Image:
    """
    Visualize trajectory overlaid on camera image with BEV side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Camera view with trajectory overlay
    ax1 = axes[0]
    ax1.imshow(camera_image)

    # Filter valid points
    valid = (trajectory_2d[:, 0] >= 0) & (trajectory_2d[:, 1] >= 0)
    valid_traj = trajectory_2d[valid]

    if len(valid_traj) > 1:
        ax1.plot(valid_traj[:, 0], valid_traj[:, 1],
                 color=color, linewidth=3, marker='o', markersize=4, alpha=0.8)
        ax1.scatter(valid_traj[0, 0], valid_traj[0, 1],
                    c='green', s=120, marker='o', zorder=10, label='Start')
        ax1.scatter(valid_traj[-1, 0], valid_traj[-1, 1],
                    c='red', s=120, marker='*', zorder=10, label='End (6.4s)')
        ax1.legend(loc='upper right', fontsize=10)

    ax1.set_title("Camera View with Trajectory", fontsize=12)
    ax1.axis('off')

    # Right: Bird's Eye View
    ax2 = axes[1]
    x_coords = trajectory_xyz[:, 0]
    y_coords = trajectory_xyz[:, 1]

    ax2.plot(y_coords, x_coords, 'b-', linewidth=2, label='Predicted Trajectory')
    ax2.scatter(y_coords[0], x_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(y_coords[-1], x_coords[-1], c='red', s=100, marker='*', label='End (6.4s)', zorder=5)
    ax2.scatter([0], [0], c='orange', s=200, marker='s', label='Ego Vehicle', zorder=10)

    ax2.set_xlabel('Lateral (m)', fontsize=11)
    ax2.set_ylabel('Longitudinal (m)', fontsize=11)
    ax2.set_title("Bird's Eye View", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_image = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    return result_image


def build_sample_pool(repo_id: str = SAMPLE_REPO_ID) -> List[str]:
    """
    Build a shuffled pool of video files for sample rotation.
    Returns list of unique sample UUIDs from the dataset.
    File format: data/UUID.camera_type.mp4
    """
    from huggingface_hub import HfApi
    import random

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[POOL] Failed to list repo: {e}")
        return []

    # Get all mp4 files
    mp4_files = [f for f in files if f.lower().endswith('.mp4')]

    # Extract unique sample UUIDs from filenames
    # Format: data/UUID.camera_type.mp4
    sample_groups: Dict[str, List[str]] = {}
    for mp4 in mp4_files:
        filename = Path(mp4).name  # e.g., "01d3588e-bca7-4a18-8e74-c6cfe9e996db.camera_front_wide_120fov.mp4"
        # Extract UUID (everything before the first ".camera")
        if '.camera' in filename:
            sample_key = filename.split('.camera')[0]
        else:
            # Fallback: use first part before dot
            sample_key = filename.split('.')[0]

        if sample_key not in sample_groups:
            sample_groups[sample_key] = []
        sample_groups[sample_key].append(mp4)

    # Get unique sample keys that have videos for required cameras
    valid_samples = []
    for key, videos in sample_groups.items():
        # Check if this sample has at least one video for our required cameras
        # Map camera patterns to check
        camera_patterns = {
            "front_wide": ["front_wide"],
            "front_tele": ["front_tele", "tele_30fov"],
            "cross_left": ["cross_left", "left_120fov"],
            "cross_right": ["cross_right", "right_120fov"],
        }
        has_all_cameras = True
        for cam, patterns in camera_patterns.items():
            if not any(any(p in v.lower() for p in patterns) for v in videos):
                has_all_cameras = False
                break
        if has_all_cameras:
            valid_samples.append(key)

    # If no valid multi-camera samples, use all sample keys
    if not valid_samples:
        valid_samples = list(sample_groups.keys())

    # Shuffle the samples
    random.shuffle(valid_samples)

    print(f"[POOL] Built sample pool with {len(valid_samples)} unique samples (UUIDs)")
    return valid_samples


def get_next_sample_id() -> Tuple[str, int]:
    """
    Get the next sample ID from the pool, rotating through available samples.
    Returns: (sample_id, index_in_pool)
    """
    global _sample_pool, _sample_idx

    # Build pool if empty
    if not _sample_pool:
        _sample_pool = build_sample_pool()

    if not _sample_pool:
        return "unknown", -1

    # Get current sample and advance index
    sample_id = _sample_pool[_sample_idx]
    current_idx = _sample_idx

    # Rotate to next sample
    _sample_idx = (_sample_idx + 1) % len(_sample_pool)

    return sample_id, current_idx


# ========== Sample Data Download ==========
def find_image_files_in_repo(repo_id: str) -> Dict[str, List[str]]:
    """
    Search for image files in the dataset repo, grouped by camera.
    Returns: {camera_name: [file_paths]}
    """
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[SAMPLE] Failed to list repo files: {e}")
        return {}

    # Filter image files
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in files if f.lower().endswith(image_exts)]

    # Try to group by camera name
    camera_files: Dict[str, List[str]] = {cam: [] for cam in CAMERAS}

    # Camera name patterns to search for
    camera_patterns = {
        "front_wide": ["front_wide", "frontwide", "front-wide", "cam_front", "front"],
        "front_tele": ["front_tele", "fronttele", "front-tele", "tele", "telephoto"],
        "cross_left": ["cross_left", "crossleft", "cross-left", "left", "cam_left"],
        "cross_right": ["cross_right", "crossright", "cross-right", "right", "cam_right"],
    }

    for img_file in image_files:
        img_lower = img_file.lower()
        matched = False
        for cam, patterns in camera_patterns.items():
            if any(p in img_lower for p in patterns):
                camera_files[cam].append(img_file)
                matched = True
                break

        # If no camera matched, try to distribute evenly
        if not matched:
            # Find camera with fewest files
            min_cam = min(camera_files, key=lambda c: len(camera_files[c]))
            camera_files[min_cam].append(img_file)

    # Sort files for each camera
    for cam in camera_files:
        camera_files[cam].sort()

    return camera_files


def download_sample_frames(repo_id: str = SAMPLE_REPO_ID) -> Tuple[Optional[Dict[str, List[Image.Image]]], Dict[str, Any]]:
    """
    Download sample frames from dataset repo.
    Returns: (camera_images_dict, meta)
    """
    from huggingface_hub import hf_hub_download

    cleanup_cache_if_needed()

    sample_id = f"frames_{int(time.time())}"
    sample_dir = CACHE_DIR / "frames" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "source": "dataset_frames",
        "camera_names": CAMERAS,
        "timestamps": [i * 0.1 for i in range(FRAMES_PER_CAMERA)],  # placeholder
        "degraded": False,
        "notes": [],
    }

    # Find image files
    camera_files = find_image_files_in_repo(repo_id)

    # Check if we have enough images
    total_available = sum(len(files) for files in camera_files.values())
    if total_available < TOTAL_IMAGES:
        meta["notes"].append(f"Only {total_available} images found in repo")
        return None, meta

    # Download images
    camera_images: Dict[str, List[Image.Image]] = {cam: [] for cam in CAMERAS}

    for cam in CAMERAS:
        files = camera_files.get(cam, [])
        if len(files) < FRAMES_PER_CAMERA:
            # Need to pad with repetition
            meta["degraded"] = True
            meta["notes"].append(f"{cam}: only {len(files)} frames, repeating to fill {FRAMES_PER_CAMERA}")

        for i in range(FRAMES_PER_CAMERA):
            if files:
                file_idx = min(i, len(files) - 1)  # Repeat last frame if needed
                file_path = files[file_idx]

                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        local_dir=str(sample_dir),
                    )
                    img = Image.open(local_path).convert("RGB")
                    camera_images[cam].append(img)
                except Exception as e:
                    print(f"[SAMPLE] Failed to download {file_path}: {e}")
                    # Use placeholder
                    camera_images[cam].append(Image.new("RGB", (576, 320), (128, 128, 128)))
                    meta["degraded"] = True
                    meta["notes"].append(f"Failed to download {file_path}")
            else:
                # No files for this camera - use placeholder
                camera_images[cam].append(Image.new("RGB", (576, 320), (128, 128, 128)))
                meta["degraded"] = True
                meta["notes"].append(f"{cam}: no images, using placeholder")

    return camera_images, meta


def extract_frames_from_video(video_path: str, num_frames: int = 4) -> List[Image.Image]:
    """Extract evenly spaced frames from video using PyAV."""
    import av

    frames = []
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames or 1000  # fallback estimate

        # Calculate target frame indices
        if total_frames > 1:
            indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
        else:
            indices = [0] * num_frames

        frame_idx = 0
        target_idx = 0

        for frame in container.decode(video=0):
            if target_idx >= len(indices):
                break
            if frame_idx >= indices[target_idx]:
                img = frame.to_image().convert("RGB")
                frames.append(img)
                target_idx += 1
            frame_idx += 1

        container.close()

        # Pad if not enough frames
        while len(frames) < num_frames:
            frames.append(frames[-1].copy() if frames else Image.new("RGB", (576, 320), (128, 128, 128)))

    except Exception as e:
        print(f"[VIDEO] Frame extraction error: {e}")

    return frames[:num_frames]


def download_sample_from_video(repo_id: str = SAMPLE_REPO_ID, target_sample_id: Optional[str] = None) -> Tuple[Optional[Dict[str, List[Image.Image]]], Dict[str, Any]]:
    """
    Download mp4 videos for each camera and extract frames.
    Uses sample rotation to return different samples on each call.
    Returns: (camera_images_dict, meta)
    """
    from huggingface_hub import HfApi, hf_hub_download
    import random

    cleanup_cache_if_needed()

    # Get next sample ID from rotation pool if not specified
    if target_sample_id is None:
        target_sample_id, pool_idx = get_next_sample_id()
    else:
        pool_idx = -1

    cache_sample_id = f"video_{target_sample_id}_{int(time.time())}"
    sample_dir = CACHE_DIR / "video" / cache_sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "source": "video_multi_camera",
        "sample_id": target_sample_id,
        "pool_index": pool_idx,
        "camera_names": CAMERAS,
        "timestamps": [i * 0.1 for i in range(FRAMES_PER_CAMERA)],
        "degraded": False,
        "notes": [],
        "ego_motion": None,  # Explicitly mark as missing
        "calibration": None,  # Explicitly mark as missing
    }

    # Find mp4 files grouped by camera
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        meta["notes"].append(f"Failed to list repo: {e}")
        return None, meta

    mp4_files = sorted([f for f in files if f.lower().endswith('.mp4')])
    if not mp4_files:
        meta["notes"].append("No mp4 files found in repo")
        return None, meta

    # Camera patterns matching the repo structure
    camera_patterns = {
        "front_wide": ["front_wide", "frontwide"],
        "front_tele": ["front_tele", "fronttele", "tele_30fov"],
        "cross_left": ["cross_left", "crossleft", "left_120fov"],
        "cross_right": ["cross_right", "crossright", "right_120fov"],
    }

    # Group videos by camera, filtering by target_sample_id if specified
    camera_videos: Dict[str, List[str]] = {cam: [] for cam in CAMERAS}
    for mp4 in mp4_files:
        mp4_lower = mp4.lower()
        # If we have a target sample, only include matching videos
        if target_sample_id and target_sample_id != "unknown":
            if target_sample_id.lower() not in mp4_lower:
                continue

        for cam, patterns in camera_patterns.items():
            if any(p in mp4_lower for p in patterns):
                camera_videos[cam].append(mp4)
                break

    # If no matching videos for target_sample_id, fall back to random selection
    total_matched = sum(len(v) for v in camera_videos.values())
    if total_matched == 0:
        meta["notes"].append(f"No videos matched sample_id '{target_sample_id}', using random selection")
        # Re-populate without sample_id filter, then randomly select
        for mp4 in mp4_files:
            mp4_lower = mp4.lower()
            for cam, patterns in camera_patterns.items():
                if any(p in mp4_lower for p in patterns):
                    camera_videos[cam].append(mp4)
                    break

    # Download one video per camera and extract frames
    # Use random selection within each camera's video list for variety
    camera_images: Dict[str, List[Image.Image]] = {cam: [] for cam in CAMERAS}

    for cam in CAMERAS:
        videos = camera_videos.get(cam, [])
        if not videos:
            # Fallback: use any available video
            meta["degraded"] = True
            meta["notes"].append(f"{cam}: no dedicated video, using fallback")
            videos = mp4_files[:1]

        if videos:
            # Random selection from available videos for this camera
            video_file = random.choice(videos) if len(videos) > 1 else videos[0]
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=video_file,
                    repo_type="dataset",
                    local_dir=str(sample_dir),
                )
                frames = extract_frames_from_video(local_path, FRAMES_PER_CAMERA)
                camera_images[cam] = frames
                meta["notes"].append(f"{cam}: extracted {len(frames)} frames from {Path(video_file).name}")
            except Exception as e:
                print(f"[VIDEO] Failed to process {cam}: {e}")
                camera_images[cam] = [Image.new("RGB", (576, 320), (128, 128, 128)) for _ in range(FRAMES_PER_CAMERA)]
                meta["degraded"] = True
                meta["notes"].append(f"{cam}: failed to extract frames - {e}")
        else:
            camera_images[cam] = [Image.new("RGB", (576, 320), (128, 128, 128)) for _ in range(FRAMES_PER_CAMERA)]
            meta["degraded"] = True
            meta["notes"].append(f"{cam}: no video available")

    return camera_images, meta


def prepare_sample_bundle_with_id(source: str, target_sample_id: Optional[str], pool_idx: int) -> Tuple[Optional[Dict], str]:
    """
    Main function to prepare sample bundle with a specific sample ID.
    Returns: (sample_bundle, status_markdown)
    """
    print(f"[SAMPLE] Preparing sample with source: {source}, target_id: {target_sample_id}, pool_idx: {pool_idx}")

    camera_images = None
    meta = None

    if "Dataset" in source:
        # Try frames first
        camera_images, meta = download_sample_frames()

        # Fallback to video if frames failed
        if camera_images is None:
            print("[SAMPLE] Frames not available, trying video fallback...")
            camera_images, meta = download_sample_from_video(target_sample_id=target_sample_id)
            if meta:
                meta["pool_index"] = pool_idx
    else:
        # Video fallback directly
        camera_images, meta = download_sample_from_video(target_sample_id=target_sample_id)
        if meta:
            meta["pool_index"] = pool_idx

    if camera_images is None:
        status = f"""## Sample Preparation Failed

**Source**: {source}
**Notes**: {'; '.join(meta.get('notes', ['Unknown error']) if meta else ['Unknown error'])}

{get_cache_status()}
"""
        return None, status

    # Flatten to list of 16 images (4 cameras × 4 frames)
    images_flat: List[Image.Image] = []
    for cam in CAMERAS:
        images_flat.extend(camera_images[cam])

    # Build bundle
    bundle = {
        "images": images_flat,
        "camera_images": camera_images,
        "meta": meta,
    }

    # Build status
    degraded_flag = "YES" if meta.get("degraded") else "NO"
    image_sizes = [f"{img.size[0]}x{img.size[1]}" for img in images_flat[:4]]
    sample_id = meta.get('sample_id', 'unknown')
    current_pool_idx = meta.get('pool_index', pool_idx)

    status = f"""## Sample Prepared Successfully

**Sample ID**: `{sample_id}` (pool index: {current_pool_idx})
**Source**: {meta.get('source', 'unknown')}
**Total Images**: {len(images_flat)}
**Cameras**: {', '.join(CAMERAS)}
**Frames per Camera**: {FRAMES_PER_CAMERA}
**Degraded**: {degraded_flag}
**Image Sizes (sample)**: {', '.join(image_sizes)}

### Data Availability
- **Ego Motion**: {'Available' if meta.get('ego_motion') else '❌ Missing (placeholder)'}
- **Calibration**: {'Available' if meta.get('calibration') else '❌ Missing'}
- **Timestamps**: {'Real' if meta.get('timestamps') != [i * 0.1 for i in range(FRAMES_PER_CAMERA)] else '❌ Placeholder (0.1s intervals)'}

### Notes
{chr(10).join('- ' + n for n in meta.get('notes', ['None'])) if meta.get('notes') else '- None'}

{get_cache_status()}
"""

    print(f"[SAMPLE] Bundle prepared: {len(images_flat)} images, degraded={meta.get('degraded')}, sample_id={sample_id}")
    return bundle, status


def prepare_sample_bundle(source: str = "Dataset frames (preferred)") -> Tuple[Optional[Dict], str]:
    """
    Main function to prepare sample bundle (legacy, uses global rotation).
    Returns: (sample_bundle, status_markdown)
    """
    return prepare_sample_bundle_with_id(source, None, -1)


# ========== Runtime Installation of alpamayo_r1 ==========
def _install_alpamayo_if_missing():
    """Install NVlabs/alpamayo package at runtime if not available."""
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
    """Import alpamayo_r1 package and ensure it's registered with transformers."""
    global _alpamayo_available

    print("[INFO] Checking alpamayo_r1 package availability...")

    try:
        import alpamayo_r1
        print(f"[INFO] alpamayo_r1 found at: {alpamayo_r1.__file__}")
        _alpamayo_available = True

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

        return True

    except ImportError as e:
        print(f"[ERROR] alpamayo_r1 package not available: {e}")
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


def load_model_impl():
    """Load the Alpamayo-R1-10B model with proper error handling."""
    global model, processor, _attn_implementation, _alpamayo_available

    if model is not None:
        return f"Model already loaded!\n\nAttention: {_attn_implementation}\n\n{get_system_info()}"

    try:
        sys_info = get_system_info()
        print(f"[INFO] System Info:\n{sys_info}")

        token = get_hf_token()
        if token:
            os.environ["HF_TOKEN"] = token
            print("[INFO] HF_TOKEN found and set.")
        else:
            print("[WARNING] No HF_TOKEN found. If model is gated, authentication will fail.")

        _attn_implementation = "eager"
        print(f"[INFO] Using attention implementation: {_attn_implementation}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        if not torch.cuda.is_available():
            return (
                "ERROR: No GPU available.\n\n"
                "This model requires a GPU with at least 24GB VRAM.\n"
                "On HuggingFace Spaces, make sure ZeroGPU is properly configured.\n\n"
                f"{sys_info}"
            )

        if not _alpamayo_available:
            return (
                "ERROR: alpamayo_r1 package not available.\n\n"
                "The postBuild script may have failed to install NVlabs/alpamayo.\n"
                "Please check the Space build logs.\n\n"
                f"{sys_info}"
            )

        print("[INFO] Loading Alpamayo-R1-10B model using NVlabs AlpamayoR1...")

        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        from alpamayo_r1 import helper

        print("[INFO] Calling AlpamayoR1.from_pretrained...")
        model = AlpamayoR1.from_pretrained(
            "nvidia/Alpamayo-R1-10B",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        print(f"[INFO] Moving model to {device}...")
        model = model.to(device)
        model.eval()
        print("[INFO] Model loaded successfully using AlpamayoR1!")

        print("[INFO] Loading processor...")
        processor = helper.get_processor(model.tokenizer)
        print("[INFO] Processor loaded successfully!")

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
            f"{error_trace}"
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Model loading failed:\n{error_trace}")

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
                "This model requires at least 24GB VRAM.\n\n"
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


def visualize_camera_grid(camera_images: Dict[str, List[Image.Image]]) -> Image.Image:
    """Create a 4x4 grid visualization of all camera frames."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    for row, cam in enumerate(CAMERAS):
        frames = camera_images.get(cam, [])
        for col in range(FRAMES_PER_CAMERA):
            ax = axes[row, col]
            if col < len(frames):
                ax.imshow(frames[col])
            else:
                ax.imshow(np.zeros((320, 576, 3), dtype=np.uint8))

            if col == 0:
                ax.set_ylabel(cam, fontsize=10)
            if row == 0:
                ax.set_title(f"Frame {col}", fontsize=10)
            ax.axis('off')

    plt.suptitle("Alpamayo Sample: 4 Cameras × 4 Frames", fontsize=14)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result_image = Image.open(buf).copy()
    plt.close(fig)
    buf.close()

    return result_image


def run_inference_impl(sample_bundle: Optional[Dict], user_command: str, num_samples: int, temperature: float, top_p: float,
                       demo_mode: bool = True, inference_mode: str = "Demo (Sample Data)",
                       clip_id: str = "", t0_us: float = 5100000):
    """Run inference with the Alpamayo-R1-10B model using 16 images."""
    global model, processor

    # Check if Official mode
    is_official_mode = inference_mode == "Official (Paper-Comparable)"

    # For Official mode, we need clip_id
    if is_official_mode:
        if not clip_id or not clip_id.strip():
            return ("Please enter a valid clip_id for Official (Paper-Comparable) mode.",
                    None, "Error: No clip_id", None, "clip_id required")

        # Check HF_TOKEN
        token = get_hf_token()
        if not token:
            return ("HF_TOKEN is required for Official mode.\n\n"
                    "Please set HF_TOKEN in your Space secrets to access nvidia/PhysicalAI-Autonomous-Vehicles dataset.",
                    None, "Error: No HF_TOKEN", None, "NOT COMPARABLE - HF_TOKEN required")

    # Check sample bundle (required for Demo mode)
    if not is_official_mode and sample_bundle is None:
        return ("Please prepare sample data first using 'Prepare Alpamayo Sample' button.", None, "Error: No sample data", None, "")

    # Initialize variables
    images = []
    camera_images = {}
    meta = {}
    sidecars = None
    minade_value = None
    metric_status_text = ""

    # For Official mode, use pre-downloaded data OR fetch new data
    if is_official_mode:
        # Check if sample_bundle contains pre-downloaded Official data
        if sample_bundle is not None and sample_bundle.get('clip_id') and sample_bundle.get('images'):
            print(f"[OFFICIAL] Using PRE-DOWNLOADED data for clip_id: {sample_bundle.get('clip_id')}")
            images = sample_bundle['images']
            camera_images = sample_bundle.get('camera_images', {})
            meta = sample_bundle.get('meta', {})
            sidecars = sample_bundle.get('sidecars', {})
            clip_id = sample_bundle.get('clip_id', clip_id)
            print(f"[OFFICIAL] Pre-downloaded data: {len(images)} images")
        else:
            # No pre-downloaded data, need to fetch (will likely timeout on ZeroGPU)
            print(f"[OFFICIAL] No pre-downloaded data, fetching for clip_id: {clip_id}")
            print(f"[OFFICIAL] ⚠️ WARNING: This may timeout. Use 'Prepare Official Data' button first!")

            # Fetch sidecars (calibration, etc.)
            sidecars = fetch_official_sidecars(clip_id.strip())
            if sidecars['errors']:
                print(f"[OFFICIAL] Sidecar errors: {sidecars['errors']}")

            # Fetch images from Official dataset
            print(f"[OFFICIAL] Fetching images from Official dataset...")
            official_images = fetch_official_images(clip_id.strip(), t0_us)

            if official_images['available']:
                # Use images from Official dataset
                images = official_images['images']
                camera_images = official_images['camera_images']
                meta = {
                    'source': 'official_dataset',
                    'sample_id': clip_id,
                    'chunk': official_images['chunk'],
                    'notes': [f'Official dataset - chunk {official_images["chunk"]}'],
                    'degraded': False
                }
                if official_images['errors']:
                    meta['notes'].extend(official_images['errors'])
                print(f"[OFFICIAL] Successfully loaded {len(images)} images from Official dataset")
            else:
                # Official dataset images not available - report errors
                error_msgs = official_images.get('errors', ['Unknown error'])
                meta = {
                    'source': 'official_mode_failed',
                    'sample_id': clip_id,
                    'notes': ['Failed to load Official dataset images'] + error_msgs,
                    'degraded': True
                }
                print(f"[OFFICIAL] Failed to load images: {error_msgs}")
    else:
        # Demo mode - use sample bundle
        images = sample_bundle.get("images", [])
        camera_images = sample_bundle.get("camera_images", {})
        meta = sample_bundle.get("meta", {})

        if len(images) != TOTAL_IMAGES:
            return (f"Invalid sample: expected {TOTAL_IMAGES} images, got {len(images)}", None, "Error: Invalid sample", None, "")

    # Validate inputs and detect missing keys
    is_valid_for_real, missing_keys = validate_inputs(sample_bundle)

    # Determine if we should run in demo mode (trajectory generation)
    # Demo mode is forced if: explicitly enabled OR missing required data (for non-official mode)
    force_demo = not is_valid_for_real and not is_official_mode
    actual_demo_mode = demo_mode or force_demo

    # Auto-load model if not loaded
    if model is None:
        print("[INFO] Model not loaded, auto-loading...")
        load_result = load_model_impl()
        print(f"[INFO] Auto-load result: {load_result[:100]}...")
        if model is None:
            return (f"Failed to auto-load model:\n\n{load_result}", None, "Error: Model failed to load", None, "")

    try:
        device = next(model.parameters()).device

        if not user_command or not user_command.strip():
            user_command = "Drive safely and follow traffic rules."

        degraded_flag = "YES" if meta.get("degraded") else "NO"
        notes = meta.get("notes", [])
        sample_id = meta.get('sample_id', clip_id if is_official_mode else 'unknown')

        # Generate trajectory
        t = np.linspace(0, 6.4, 64)
        trajectory = np.zeros((64, 3))  # (64, 3) for x, y, z
        reasoning_text = ""  # Chain-of-Causation reasoning from model

        # ========== REAL MODEL INFERENCE ==========
        # Check if we have images for real inference
        has_images = images is not None and len(images) == TOTAL_IMAGES
        use_real_inference = has_images and not actual_demo_mode

        if use_real_inference:
            print(f"[INFERENCE] Running REAL model inference with {len(images)} images...")
            try:
                from alpamayo_r1 import helper

                # Prepare images tensor: need (16,) list of PIL images -> tensor
                # Images should be in order: 4 cameras x 4 frames
                image_list = images  # Already a list of PIL images

                # Create message with images
                messages = helper.create_message(image_list)

                # Apply chat template
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                # Prepare model inputs
                model_inputs = {
                    "tokenized_data": {k: v.to(device) for k, v in inputs.items()},
                    "ego_history_xyz": torch.zeros(1, 4, 3, device=device, dtype=torch.float32),
                    "ego_history_rot": torch.zeros(1, 4, 4, device=device, dtype=torch.float32),
                }

                # Number of trajectory samples
                n_samples = num_samples if is_official_mode else 1
                print(f"[INFERENCE] Generating {n_samples} trajectory samples...")

                # Run model inference
                with torch.no_grad():
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=top_p,
                        temperature=temperature,
                        num_traj_samples=n_samples,
                        max_generation_length=256,
                        return_extra=True,
                    )

                # Extract trajectory: pred_xyz is (B, 1, num_samples, 64, 3)
                pred_np = pred_xyz.cpu().numpy()  # (1, 1, n_samples, 64, 3)
                print(f"[INFERENCE] Model output shape: {pred_np.shape}")

                # Get reasoning text from extra
                if extra and "reasoning" in extra:
                    reasoning_text = extra["reasoning"]
                elif extra and "chain_of_causation" in extra:
                    reasoning_text = extra["chain_of_causation"]

                if is_official_mode:
                    # Official mode: use all samples for minADE calculation
                    pred_trajectories = pred_np[0, 0]  # (n_samples, 64, 3)

                    # For now, use synthetic GT since egomotion is COMING SOON
                    gt_trajectory = np.zeros((64, 3))
                    gt_trajectory[:, 0] = t * 5.0
                    gt_trajectory[:, 1] = 0.3 * np.sin(t * 0.5)

                    # Compute minADE6
                    gt_xy = gt_trajectory[:, :2].T  # (2, 64)
                    pred_xy = pred_trajectories[:, :, :2].transpose(0, 2, 1)  # (n_samples, 2, 64)
                    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)  # (n_samples,)
                    minade_value = float(diff.min())
                    best_idx = int(diff.argmin())
                    trajectory = pred_trajectories[best_idx]

                    metric_status_text = f"REAL INFERENCE (minADE6={minade_value:.4f}m, synthetic GT)"
                    mode_status = "## Official Mode - REAL Inference"
                    mode_reason = f"Using clip_id: {clip_id} with {n_samples} trajectory samples"
                    trajectory_note = f"**REAL MODEL OUTPUT** - minADE6@6.4s: {minade_value:.4f} m (best of {n_samples} samples, synthetic GT)"
                    notes.append("Model inference: REAL (not synthetic)")
                    notes.append(f"Trajectory samples: {n_samples}")
                    notes.append("Ground truth: Synthetic (egomotion COMING SOON)")
                else:
                    # Non-official mode: use first sample
                    trajectory = pred_np[0, 0, 0]  # (64, 3)

                    mode_status = "## Real Inference Mode"
                    mode_reason = "All required data available"
                    trajectory_note = "**REAL MODEL OUTPUT** - Actual prediction from Alpamayo-R1"
                    metric_status_text = "REAL INFERENCE"
                    notes.append("Model inference: REAL (not synthetic)")

                # Add calibration status
                if sidecars and sidecars.get('calibration'):
                    notes.append("Calibration data: Available")
                else:
                    notes.append("Calibration data: Not available (using fallback projection)")

                print(f"[INFERENCE] Real inference complete! Trajectory shape: {trajectory.shape}")

            except Exception as infer_err:
                import traceback
                print(f"[INFERENCE] Real inference failed: {infer_err}")
                print(traceback.format_exc())
                # Fall back to synthetic
                use_real_inference = False
                notes.append(f"Real inference failed: {str(infer_err)[:100]}")
                notes.append("Falling back to synthetic trajectory")

        # ========== SYNTHETIC FALLBACK ==========
        if not use_real_inference:
            if is_official_mode and not actual_demo_mode:
                # Official mode fallback: generate synthetic but varied trajectories
                print("[INFERENCE] Using SYNTHETIC trajectories (fallback)")
                pred_trajectories = []
                for i in range(num_samples):
                    traj = np.zeros((64, 3))
                    traj[:, 0] = t * (4.5 + 0.5 * np.random.randn())  # Forward with variance
                    traj[:, 1] = 0.3 * np.sin(t * 0.5 + 0.3 * i)  # Lateral variance
                    traj[:, 2] = 0.0  # Height
                    pred_trajectories.append(traj)

                pred_trajectories = np.stack(pred_trajectories)  # (n_samples, 64, 3)

                # Ground truth trajectory (synthetic)
                gt_trajectory = np.zeros((64, 3))
                gt_trajectory[:, 0] = t * 5.0
                gt_trajectory[:, 1] = 0.3 * np.sin(t * 0.5)

                # Compute minADE
                gt_xy = gt_trajectory[:, :2].T  # (2, 64)
                pred_xy = pred_trajectories[:, :, :2].transpose(0, 2, 1)
                diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
                minade_value = float(diff.min())
                best_idx = int(diff.argmin())
                trajectory = pred_trajectories[best_idx]

                metric_status_text = f"SYNTHETIC (minADE6={minade_value:.4f}m)"
                mode_status = "## Official Mode - SYNTHETIC Fallback"
                mode_reason = f"Using clip_id: {clip_id} (no images for real inference)"
                trajectory_note = f"**SYNTHETIC trajectory** - minADE6@6.4s: {minade_value:.4f} m"
                notes.append("WARNING: Using synthetic trajectory (prepare sample data for real inference)")

                if sidecars and sidecars.get('calibration'):
                    notes.append("Calibration data: Available")
                else:
                    notes.append("Calibration data: Not available")

            elif actual_demo_mode:
                # Demo trajectory - synthetic smooth curve
                trajectory[:, 0] = t * 5  # Forward motion
                trajectory[:, 1] = 0.5 * np.sin(t * 0.5)  # Lateral sway
                trajectory[:, 2] = 0.0  # Height

                mode_status = "## Demo Mode"
                mode_reason = "Demo mode enabled" if demo_mode else "Forced demo mode (missing required data)"
                missing_keys_text = chr(10).join(f'- {k}' for k in missing_keys) if missing_keys else '- All keys present'
                trajectory_note = "**This is a DEMO trajectory (synthetic data).**"
                metric_status_text = "DEMO - no metric"
            else:
                # Non-official, non-demo fallback
                trajectory[:, 0] = t * 5
                trajectory[:, 1] = 0.5 * np.sin(t * 0.5)
                trajectory[:, 2] = 0.0

                mode_status = "## Fallback Mode"
                mode_reason = "No images available for real inference"
                trajectory_note = "**Synthetic trajectory (no images loaded).**"
                metric_status_text = "FALLBACK - no inference"

        # Get front camera image for visualization
        front_camera_image = None
        if "front_wide" in camera_images and camera_images["front_wide"]:
            front_camera_image = camera_images["front_wide"][0]

        # Create visualization with trajectory overlay
        if front_camera_image is not None:
            # Project trajectory to camera
            cam_extrinsic = None
            cam_intrinsic = None
            if sidecars and sidecars.get('calibration'):
                calib_data = sidecars['calibration']
                # New format: calibration is dict with camera_name as key
                # Try to get front_wide camera calibration
                calib = None
                if isinstance(calib_data, dict):
                    # Look for front_wide camera
                    for cam_key in ['camera_front_wide_120fov', 'front_wide', 'camera_front_wide']:
                        if cam_key in calib_data:
                            calib = calib_data[cam_key]
                            break
                    # If not found, use first available camera
                    if calib is None and len(calib_data) > 0:
                        first_key = list(calib_data.keys())[0]
                        if isinstance(calib_data[first_key], dict):
                            calib = calib_data[first_key]
                        else:
                            calib = calib_data  # Old format: flat dict

                if calib:
                    # Extract camera parameters (intrinsics only for f-theta model)
                    cam_intrinsic = {
                        'cx': calib.get('cx', front_camera_image.size[0] / 2),
                        'cy': calib.get('cy', front_camera_image.size[1] / 2),
                        'width': calib.get('width', front_camera_image.size[0]),
                        'height': calib.get('height', front_camera_image.size[1]),
                        'fw_poly_0': calib.get('fw_poly_0', 0),
                        'fw_poly_1': calib.get('fw_poly_1', 500),
                        'fw_poly_2': calib.get('fw_poly_2', 0),
                        'fw_poly_3': calib.get('fw_poly_3', 0),
                        'fw_poly_4': calib.get('fw_poly_4', 0),
                    }
                    print(f"[VIS] Using calibration: cx={cam_intrinsic['cx']:.1f}, cy={cam_intrinsic['cy']:.1f}")

            # Project trajectory to 2D
            traj_2d = project_trajectory_to_camera(
                trajectory,
                cam_extrinsic,
                cam_intrinsic,
                image_size=(front_camera_image.size[0], front_camera_image.size[1])
            )

            # Create overlay visualization
            title = f"Alpamayo-R1 - {mode_status.replace('## ', '')}"
            if minade_value is not None:
                title += f" | minADE6: {minade_value:.4f}m"
            vis_image = visualize_trajectory_overlay(front_camera_image, traj_2d, trajectory, title=title)
        else:
            # Fallback to BEV only visualization
            vis_image = visualize_trajectory(trajectory, None)

        # Build reasoning output
        reasoning_output = f"""{mode_status}

**Reason**: {mode_reason}
**Sample ID**: `{sample_id}`
**Clip ID**: `{clip_id if is_official_mode else 'N/A'}`

---
## Model Information

**Model**: NVIDIA Alpamayo-R1-10B (VLA for Autonomous Driving)
**Device**: {device}
**Attention**: eager implementation
**Command**: {user_command}

---
## Input Data Summary

**Source**: {meta.get('source', 'official' if is_official_mode else 'unknown')}
**Total Images**: {len(images) if images else 'N/A'} (4 cameras × 4 frames)
**Cameras**: {', '.join(CAMERAS)}
**Degraded Input**: {degraded_flag}

### Data Availability
- **Calibration**: {'Available' if (sidecars and sidecars.get('calibration')) else 'Not available'}
- **Ego Motion**: {'COMING SOON' if is_official_mode else ('Available' if meta.get('ego_motion') else 'Missing')}
- **Timestamps**: {'Available' if (sidecars and sidecars.get('timestamps')) else 'Placeholder'}

### Notes
{chr(10).join('- ' + n for n in notes) if notes else '- Standard input'}

---
## Trajectory Output

{trajectory_note}

**Prediction Horizon**: 6.4 seconds (64 waypoints @ 10Hz)
**Forward Distance**: {trajectory[-1, 0]:.2f} m
**Lateral Offset**: {trajectory[-1, 1]:.2f} m
{f'**minADE6@6.4s**: {minade_value:.4f} m' if minade_value is not None else ''}

---
## References
- [NVlabs/alpamayo](https://github.com/NVlabs/alpamayo)
- [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
"""

        status_msg = f"{metric_status_text} - Processing complete!"
        return (reasoning_output, vis_image, status_msg, minade_value, metric_status_text)

    except Exception as e:
        import traceback
        error_msg = f"Error during inference: {str(e)}\n\n{traceback.format_exc()}"
        return (error_msg, None, f"Error: {str(e)}", None, "ERROR")


# Apply ZeroGPU decorator if available
if ZERO_GPU:
    run_inference = spaces.GPU(duration=120)(run_inference_impl)
else:
    run_inference = run_inference_impl


# ========== Gradio UI ==========
with gr.Blocks(title="Alpamayo-R1-10B Inference Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # NVIDIA Alpamayo-R1-10B Inference Demo

    A Vision-Language-Action (VLA) model for autonomous driving that provides:
    - **Chain-of-Causation Reasoning**: Interpretable decision-making process
    - **Trajectory Prediction**: 6.4-second future path prediction at 10Hz

    **Input Format**: 4 cameras × 4 frames = 16 images (0.4s @ 10Hz)

    **Quick Start**:
    1. Click "Prepare Alpamayo Sample" to download sample data
    2. Click "Load Model" to initialize the model
    3. Click "Run Inference" to process the sample
    """)

    # State for sample bundle and sample rotation
    state_sample = gr.State(value=None)
    state_sample_pool = gr.State(value=[])
    state_sample_idx = gr.State(value=0)
    # State for Official mode pre-downloaded data
    state_official_data = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Control")
            load_btn = gr.Button("Load Model", variant="primary", size="lg")
            model_status = gr.Textbox(
                label="Model Status",
                value="Model not loaded\n\nClick 'Load Model' to initialize.",
                interactive=False,
                lines=6
            )

            gr.Markdown("### Sample Data Preparation")
            sample_source = gr.Dropdown(
                choices=["Dataset frames (preferred)", "Video fallback (mp4 -> frames)"],
                value="Dataset frames (preferred)",
                label="Sample Source"
            )
            prepare_btn = gr.Button("Prepare Alpamayo Sample (16 images)", variant="secondary", size="lg")
            sample_status = gr.Markdown(
                value=f"No sample prepared.\n\n{get_cache_status()}"
            )

            gr.Markdown("### Inference Mode")
            inference_mode = gr.Radio(
                choices=["Demo (Sample Data)", "Official (Paper-Comparable)"],
                value="Demo (Sample Data)",
                label="Select Mode",
                info="Demo: use sample data with synthetic trajectory. Official: use PhysicalAI-AV dataset with minADE6 metric."
            )

            # Official mode specific inputs
            with gr.Group(visible=False) as official_group:
                gr.Markdown("#### Official Mode Settings")
                gr.Markdown("⚠️ **Step 1**: Enter clip_id → **Step 2**: Prepare Data → **Step 3**: Run Inference")
                clip_id_input = gr.Textbox(
                    label="Clip ID (UUID)",
                    placeholder="e.g., 030c760c-ae38-49aa-9ad8-f5650a545d26",
                    value="",
                    info="Enter a valid clip_id from PhysicalAI-AV dataset"
                )
                t0_us_input = gr.Number(
                    label="t0 (microseconds)",
                    value=5100000,
                    precision=0,
                    info="Timestamp for trajectory prediction start"
                )
                prepare_official_btn = gr.Button("📥 Prepare Official Data (downloads ~4GB)", variant="secondary", size="lg")
                official_data_status = gr.Markdown(value="No Official data prepared.")

                # Image preview for Official mode
                gr.Markdown("#### Loaded Camera Images")
                official_image_gallery = gr.Gallery(
                    label="Camera Preview (4 cameras × 1 frame each)",
                    columns=4,
                    rows=1,
                    height=200,
                    object_fit="contain",
                    show_label=True
                )
                official_data_summary = gr.Markdown(value="")

            gr.Markdown("### Driving Command")
            user_command = gr.Textbox(
                label="Command (optional)",
                placeholder="E.g., 'Turn left at the next intersection'",
                value="Drive safely and follow traffic rules."
            )

            gr.Markdown("### Inference Parameters")
            demo_mode_cb = gr.Checkbox(
                value=True,
                label="Demo Mode (synthetic trajectory)",
                info="Enable to use synthetic trajectory. Disable for real model inference.",
                visible=True
            )

            with gr.Row():
                num_samples = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Samples")

            with gr.Row():
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.6, step=0.1, label="Temperature")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p")

            run_btn = gr.Button("Run Inference", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            status_output = gr.Textbox(label="Status", value="Ready", interactive=False)

            # Official mode metrics
            with gr.Row():
                minade_output = gr.Number(
                    label="minADE6@6.4s (m)",
                    value=None,
                    precision=4,
                    interactive=False,
                    visible=False
                )
                metric_status = gr.Textbox(
                    label="Metric Status",
                    value="",
                    interactive=False,
                    visible=False
                )

            reasoning_output = gr.Markdown(label="Reasoning", value="Results will appear here...")
            trajectory_output = gr.Image(label="Camera Grid / Trajectory Visualization", type="pil", height=400)

    # Event handlers
    load_btn.click(fn=load_model, outputs=[model_status])

    # Mode switch handler
    def on_mode_change(mode):
        """Toggle visibility of Official mode settings."""
        is_official = mode == "Official (Paper-Comparable)"
        return (
            gr.update(visible=is_official),  # official_group
            gr.update(visible=not is_official),  # demo_mode_cb
            gr.update(visible=is_official),  # minade_output
            gr.update(visible=is_official),  # metric_status
        )

    inference_mode.change(
        fn=on_mode_change,
        inputs=[inference_mode],
        outputs=[official_group, demo_mode_cb, minade_output, metric_status]
    )

    def on_prepare_sample(source, sample_pool, sample_idx):
        """Prepare sample with rotation through sample pool."""
        import random

        # Build pool if empty
        if not sample_pool:
            sample_pool = build_sample_pool()
            random.shuffle(sample_pool)
            sample_idx = 0

        # Get current sample ID
        if sample_pool:
            current_sample_id = sample_pool[sample_idx]
            # Rotate to next index for next call
            next_idx = (sample_idx + 1) % len(sample_pool)
        else:
            current_sample_id = None
            next_idx = 0

        # Prepare bundle with specific sample ID
        bundle, status = prepare_sample_bundle_with_id(source, current_sample_id, sample_idx)

        return bundle, status, sample_pool, next_idx

    prepare_btn.click(
        fn=on_prepare_sample,
        inputs=[sample_source, state_sample_pool, state_sample_idx],
        outputs=[state_sample, sample_status, state_sample_pool, state_sample_idx]
    )

    def on_prepare_official_data(clip_id, t0_us):
        """
        Prepare Official dataset data BEFORE GPU allocation.
        This avoids the 120-second timeout for large zip downloads.
        Returns: bundle, status_msg, gallery_images, data_summary
        """
        empty_gallery = []
        empty_summary = ""

        if not clip_id or not clip_id.strip():
            return None, "❌ Please enter a valid clip_id first.", empty_gallery, empty_summary

        token = get_hf_token()
        if not token:
            return None, "❌ HF_TOKEN is required. Please set it in Space secrets.", empty_gallery, empty_summary

        try:
            # Fetch images (this is the slow part - downloading zip files)
            print(f"[OFFICIAL PREPARE] Starting download for {clip_id}")
            official_images = fetch_official_images(clip_id.strip(), t0_us)

            if official_images['available']:
                # Also fetch sidecars
                sidecars = fetch_official_sidecars(clip_id.strip(), official_images['chunk'])

                # Build bundle
                bundle = {
                    'images': official_images['images'],
                    'camera_images': official_images['camera_images'],
                    'chunk': official_images['chunk'],
                    'sidecars': sidecars,
                    'clip_id': clip_id.strip(),
                    't0_us': t0_us,
                    'meta': {
                        'source': 'official_dataset',
                        'sample_id': clip_id,
                        'chunk': official_images['chunk'],
                        'degraded': False,
                        'notes': []
                    }
                }

                # Prepare gallery images (first frame from each camera)
                camera_images = official_images['camera_images']
                gallery_images = []
                cam_labels = ['Front Wide', 'Cross Left', 'Cross Right', 'Rear Tele']
                cam_keys = ['front_wide', 'cross_left', 'cross_right', 'rear_tele']
                for i, cam_key in enumerate(cam_keys):
                    if cam_key in camera_images and camera_images[cam_key]:
                        img = camera_images[cam_key][0]  # First frame
                        gallery_images.append((img, cam_labels[i]))

                # Build data summary
                calib_info = "Not available"
                if sidecars.get('calibration'):
                    calib_cams = list(sidecars['calibration'].keys()) if isinstance(sidecars['calibration'], dict) else []
                    calib_info = f"Available ({len(calib_cams)} cameras)"

                ego_info = "COMING SOON"
                if sidecars.get('egomotion') is not None:
                    ego_len = len(sidecars['egomotion'])
                    ego_info = f"Available ({ego_len} points)"

                data_summary = f"""### Data Summary
| Item | Value |
|------|-------|
| **Clip ID** | `{clip_id[:20]}...` |
| **Chunk** | {official_images['chunk']} |
| **Total Images** | {len(official_images['images'])} (4 cameras × 4 frames) |
| **t0** | {t0_us / 1e6:.2f}s |
| **Calibration** | {calib_info} |
| **Egomotion (GT)** | {ego_info} |
"""
                status_msg = "✅ **Data ready!** Click 'Run Inference' to process."

                if official_images['errors']:
                    status_msg += f" ⚠️ {len(official_images['errors'])} warnings"

                return bundle, status_msg, gallery_images, data_summary
            else:
                errors = official_images.get('errors', ['Unknown error'])
                error_msg = "❌ Failed:\n" + "\n".join(f"- {e}" for e in errors[:3])
                return None, error_msg, empty_gallery, empty_summary

        except Exception as e:
            import traceback
            print(f"[OFFICIAL PREPARE] Error: {traceback.format_exc()}")
            return None, f"❌ Error: {str(e)[:100]}", empty_gallery, empty_summary

    prepare_official_btn.click(
        fn=on_prepare_official_data,
        inputs=[clip_id_input, t0_us_input],
        outputs=[state_official_data, official_data_status, official_image_gallery, official_data_summary]
    )

    def run_inference_wrapper(sample_bundle, user_command, num_samples, temperature, top_p,
                               demo_mode, inference_mode, clip_id, t0_us, official_data):
        """Wrapper to handle different modes and return appropriate outputs."""
        # For Official mode, use pre-downloaded data if available
        if inference_mode == "Official (Paper-Comparable)" and official_data is not None:
            # Pass the pre-downloaded official data as sample_bundle
            sample_bundle = official_data

        result = run_inference(
            sample_bundle, user_command, num_samples, temperature, top_p,
            demo_mode, inference_mode, clip_id, t0_us
        )

        # Unpack result
        if len(result) == 5:
            reasoning, vis, status, minade, metric_status_text = result
            return reasoning, vis, status, minade, metric_status_text
        else:
            # Legacy format
            reasoning, vis, status = result
            return reasoning, vis, status, None, ""

    run_btn.click(
        fn=run_inference_wrapper,
        inputs=[state_sample, user_command, num_samples, temperature, top_p,
                demo_mode_cb, inference_mode, clip_id_input, t0_us_input, state_official_data],
        outputs=[reasoning_output, trajectory_output, status_output, minade_output, metric_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
