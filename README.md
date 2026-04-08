# NavDreamer

复现论文 *NavDreamer: Video Models as Zero-Shot 3D Navigators* (arXiv:2602.09765, Huang et al., 2026)。

NavDreamer 使用生成式视频模型作为零样本 3D 导航器：给定一张 RGB 图像和自然语言指令，生成导航视频，再将视频解码为可执行的度量航点，用于无人机实际飞行。

## 整体架构

```
图像 + 语言指令
      │
      ▼
┌──────────────────┐
│  视频生成 (Wan)    │  生成 K 个候选导航视频
└───────┬──────────┘
        ▼
┌──────────────────┐
│  采样优化          │  Qwen3-VL 对每个视频评分
│  (Section 3.1)    │  动作安全(0.8) + 场景一致性(0.8) + 任务表现(1.4)
└───────┬──────────┘
        ▼ 最优视频 V*
┌──────────────────┐
│  动作解码器        │  waypoint_generator.py (已实现)
│  (Section 3.2)    │  Pi3X VO + MoGe-2 → 度量航点
└───────┬──────────┘
        ▼
┌──────────────────┐
│  Ego-Planner      │  无碰撞轨迹规划
│  (Section 3.3)    │
└──────────────────┘
```

## 环境配置

**依赖**：Python 3.10、PyTorch 2.5+、CUDA GPU (≥16GB VRAM)

```bash
# 克隆仓库及子模块
git clone --recursive <repo_url>
cd NavDreamer

# 创建虚拟环境 (使用 uv)
uv venv .venv --python 3.10
source .venv/bin/activate

# 安装依赖
# ⚡️ 国内加速源（可选）
export HF_ENDPOINT=https://hf-mirror.com                      # HuggingFace 镜像

uv pip install torch torchvision numpy opencv-python huggingface_hub safetensors
cd Pi3 && uv pip install -r requirements.txt && cd ..
cd MoGe && uv pip install -r requirements.txt && cd ..
```

**模型权重**：

所有模型权重统一存放在项目根目录的 `ckpts/` 下：

```bash
ckpts/
├── pi3x.safetensors   # Pi3X 权重 (yyfz233/Pi3X)
└── moge2.pt           # MoGe-2 权重 (Ruicheng/moge-2-vitl-normal)
```

从 HuggingFace 下载后重命名放入即可。如本地不存在，代码会自动通过 `from_pretrained()` 下载并缓存到 `~/.cache/huggingface/hub/`。

## 使用方法

```bash
# 输入视频文件
python waypoint_generator.py --video best_video.mp4 --interval 10

# 输入图片目录（读取所有图片，忽略 --interval）
python waypoint_generator.py --video path/to/images/

# 指定输出目录
python waypoint_generator.py --video best_video.mp4 --output_dir results/exp1
```

### 全部参数


| 参数             | 默认值              | 说明               |
| -------------- | ---------------- | ---------------- |
| `--video`      | (必填)             | 视频文件路径或图片目录      |
| `--interval`   | 10               | 视频抽帧间隔（图片目录模式无效） |
| `--tau_min`    | 0.5              | 有效深度下限（米）        |
| `--tau_max`    | 30.0             | 有效深度上限（米）        |
| `--output_dir` | `outputs/<视频名>/` | 输出目录             |
| `--device`     | `cuda`           | 推理设备             |


## 输出结构

```
outputs/<name>/
├── frames/                # 抽取的帧 (PNG)
├── pi3/
│   ├── camera_poses.npy   # (N,4,4) cam-to-world SE(3) 位姿，归一化尺度
│   ├── local_depth.npy    # (N,H,W) 局部深度 Z_pred（相机坐标系 Z 分量）
│   ├── conf.npy           # (N,H,W) 每像素置信度（已 sigmoid）
│   └── points.npy         # (N,H,W,3) 世界坐标稠密点云
├── moge/
│   └── depth_XXXX.npy     # (H,W) 度量深度 D_ref（米）
└── waypoints.json         # 尺度因子 S + 度量航点
```

### waypoints.json 格式

```json
{
  "scale_factor": 1.1691,
  "num_frames": 24,
  "waypoints": [
    {"frame": 0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.12},
    {"frame": 1, "x": 0.35, "y": -0.02, "z": 1.21, "yaw": 0.15}
  ]
}
```

各字段含义：`x, y, z` 为世界坐标系下的米制位置，`yaw` 为航向角（弧度）。

## 动作解码器原理

### 尺度恢复（论文 Section 3.2, Eq.4-5）

Pi3X 输出的位姿和点云是**归一化尺度**的，MoGe-2 输出的深度是**真实米制**的。两者的桥梁是全局尺度因子 S：

```
有效掩码: M_t = { (u,v) | tau_min < D_ref(u,v) < tau_max  且  Z_pred(u,v) > 0 }
尺度因子: S = median( D_ref[M] / Z_pred[M] )     （跨所有帧）
度量航点: W_t = S × w_t
```

没有尺度校正时，户外场景约有 ~54% 的尺度误差；校正后降至 ~6%。

## 项目结构

```
NavDreamer/
├── waypoint_generator.py   # 核心: 视频/图片 → 度量航点
├── ckpts/                  # 模型权重 (统一管理，gitignored)
│   ├── pi3x.safetensors
│   └── moge2.pt
├── Pi3/                    # Pi3X 逆动力学模型 (git submodule)
│   ├── pi3/models/pi3x.py  # Pi3X 模型 (DINOv2 ViT-L)
│   └── pi3/pipe/pi3x_vo.py # VO 管线 (分块 Sim(3) 对齐)
├── MoGe/                   # MoGe-2 度量深度估计 (git submodule)
│   └── moge/model/v2.py    # MoGeV2 模型 (度量尺度头)
├── outputs/                # 推理输出
└── .venv/                  # Python 虚拟环境
```

## 实现进度


| 组件           | 状态  | 位置                      |
| ------------ | --- | ----------------------- |
| Pi3X + VO 管线 | 已就绪 | `Pi3/` 子模块              |
| MoGe-2 度量深度  | 已就绪 | `MoGe/` 子模块             |
| 航点生成器        | 已实现 | `waypoint_generator.py` |
| 视频生成 (Wan)   | 未集成 | 需要 API                  |
| Qwen3-VL 评分  | 未集成 | —                       |
| Ego-Planner  | 未集成 | —                       |


## 引用

```bibtex
@article{huang2026navdreamer,
  title={NavDreamer: Video Models as Zero-Shot 3D Navigators},
  author={Huang, Xijie and Gai, Weiqi and Wu, Tianyue and Wang, Congyu and Liu, Zhiyang and Zhou, Xin and Wu, Yuze and Gao, Fei},
  journal={arXiv preprint arXiv:2602.09765},
  year={2026}
}
```

