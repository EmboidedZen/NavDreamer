# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NavDreamer reproduces the paper "NavDreamer: Video Models as Zero-Shot 3D Navigators" (arXiv:2602.09765). It uses generative video models as zero-shot 3D navigators for UAVs: given a single RGB image and a language instruction, it generates a navigation video, decodes it into metric waypoints, and passes them to a trajectory planner.

## Commands

```bash
# Activate environment (Python 3.10, managed by uv)
source .venv/bin/activate

# Generate waypoints (supports video file or image directory)
python waypoint_generator.py --video <video_or_image_dir> --interval 10
# Results saved to outputs/<name>/ by default, override with --output_dir

# Pi3 submodule examples (from Pi3/)
python Pi3/example_vo.py      # VO pipeline inference
python Pi3/example_mm.py      # Pi3X multimodal inference

# MoGe submodule CLI (from MoGe/)
moge infer -i IMAGE_PATH -o OUTPUT --maps --glb --ply
moge app                      # Gradio demo
```

No test suite, linter, or build system is configured at the top level.

## Architecture

The full pipeline (partially implemented):

```
Image + Language Instruction
  → Video Generation (Wan 2.6 API, NOT integrated)
  → Sampling-based Optimization via Qwen3-VL (NOT integrated)
  → Action Decoder (waypoint_generator.py, IMPLEMENTED)
  → Ego-Planner (NOT integrated)
```

### Action Decoder pipeline (waypoint_generator.py)

This is the core implemented component. The `WaypointGenerator` class runs:

1. **Frame sampling** — extract N frames from video (fixed interval) or image directory (all images)
2. **Pi3X VO** (`run_pi3`) — chunked visual odometry producing camera poses (N,4,4) and local depth
3. **MoGe-2** (`run_moge`) — per-frame metric depth estimation in meters
4. **Scale recovery** (`compute_scale_factor`) — median ratio of MoGe metric depth to Pi3 predicted depth, filtered by validity mask (tau_min < D_ref < tau_max, Z_pred > 0, conf threshold)
5. **Waypoint extraction** — scale poses by factor S, compute yaw from forward vector

Output: list of `Waypoint(x, y, z, yaw)` in meters/radians.

### Output structure (`outputs/<name>/`)

```
outputs/<name>/
├── frames/              # 抽取的帧 (PNG)
├── pi3/
│   ├── camera_poses.npy # (N,4,4) cam-to-world SE(3) 位姿，归一化尺度，相对第0帧
│   ├── local_depth.npy  # (N,H,W) 世界点反投影回相机坐标系的 Z 分量 (Z_pred)
│   ├── conf.npy         # (N,H,W) 每像素置信度 (已 sigmoid)
│   └── points.npy       # (N,H,W,3) Sim(3) 对齐后的世界坐标稠密点云
├── moge/
│   └── depth_XXXX.npy   # (H,W) 每帧独立估计的度量深度 D_ref (米)
└── waypoints.json       # scale_factor S + 米制航点
```

Pi3 输出归一化尺度，MoGe 输出真实米制深度。通过 `S = median(D_ref / Z_pred)` 恢复真实尺度。

### Submodules

- **Pi3/** — Pi3X inverse dynamics model (DINOv2 ViT-L encoder). `Pi3XVO` splits long sequences into overlapping chunks and aligns via Sim(3) Umeyama. HuggingFace: `yyfz233/Pi3X`.
- **MoGe/** — MoGe-2 metric depth estimator (DINOv2 ViT-L encoder). HuggingFace: `Ruicheng/moge-2-vitl-normal`.

Model weights are centralized in `ckpts/` (`pi3x.safetensors`, `moge2.pt`), falling back to `from_pretrained()` which caches in `~/.cache/huggingface/hub/`.

Submodules are imported at runtime via `sys.path` manipulation (lines 16-19 of waypoint_generator.py), not installed as packages.

## Critical Implementation Details

- Pi3 images must be resized to H,W multiples of 14 (DINOv2 patch size) with total pixels ≤ 255,000
- MoGe depth output contains `inf` for invalid pixels — always filter by `mask` before computing ratios
- Pi3 `conf` from the VO pipeline is already sigmoided; raw model output is logits
- Scale factor S uses **median** (not mean) for outlier robustness
- Coordinate convention: OpenCV (X-right, Y-down, Z-forward), cam-to-world SE(3), all poses relative to frame 0
- bfloat16 on Ampere+ GPUs (compute capability ≥ 8), float16 on older GPUs
- GPU with ≥16GB VRAM recommended

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Pi3X + VO pipeline | Ready | `Pi3/` submodule |
| MoGe-2 metric depth | Ready | `MoGe/` submodule |
| Waypoint generator | Implemented | `waypoint_generator.py` |
| Video generation (Wan) | Not integrated | Requires API access |
| Qwen3-VL scoring | Not integrated | — |
| Ego-Planner | Not integrated | — |
| Full pipeline (main.py) | Not started | — |
