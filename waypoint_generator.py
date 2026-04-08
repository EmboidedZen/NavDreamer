"""
NavDreamer Waypoint Generator
Paper: NavDreamer: Video Models as Zero-Shot 3D Navigators (Section 3.2)

Pipeline:
    Selected Video V* → Frame Sampling → Pi3X (poses + pointmaps)
                                        → MoGe-2 (metric depth)
                                        → Scale Recovery (median ratio)
                                        → Metric Waypoints (x, y, z, yaw)
"""

import sys
import os

# Add submodule paths
_project_root = os.path.dirname(os.path.abspath(__file__))
for _sub in ("Pi3", "MoGe"):
    _p = os.path.join(_project_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float  # radians


class WaypointGenerator:
    """从评分后的视频生成度量空间航点。"""

    def __init__(
        self,
        device: str = "cuda",
        tau_min: float = 0.5,
        tau_max: float = 30.0,
        pi3_chunk_size: int = 16,
        pi3_overlap: int = 6,
        pi3_conf_thre: float = 0.05,
        moge_resolution_level: int = 9,
    ):
        self.device = torch.device(device)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.pi3_chunk_size = pi3_chunk_size
        self.pi3_overlap = pi3_overlap
        self.pi3_conf_thre = pi3_conf_thre
        self.moge_resolution_level = moge_resolution_level
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        self.pi3_model = None
        self.pi3_pipe = None
        self.moge_model = None

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load_pi3(self, repo_id: str = "yyfz233/Pi3X"):
        """加载 Pi3X 模型和 VO 管线。

        加载顺序: ckpts/pi3x.safetensors → from_pretrained (HuggingFace 缓存)
        """
        from pi3.models.pi3x import Pi3X
        from pi3.pipe.pi3x_vo import Pi3XVO

        local_path = os.path.join(_project_root, "ckpts", "pi3x.safetensors")
        if os.path.exists(local_path):
            from safetensors.torch import load_file
            model = Pi3X().to(self.device).eval()
            model.load_state_dict(load_file(local_path), strict=False)
            source = local_path
        else:
            model = Pi3X.from_pretrained(repo_id).to(self.device).eval()
            source = repo_id

        self.pi3_model = model
        self.pi3_pipe = Pi3XVO(model)
        print(f"[WaypointGenerator] Pi3X loaded from {source}.")

    def load_moge(self, repo_id: str = "Ruicheng/moge-2-vitl-normal"):
        """加载 MoGe-2 度量深度模型。

        加载顺序: ckpts/moge2.pt → from_pretrained (HuggingFace 缓存)
        """
        from moge.model.v2 import MoGeModel

        local_path = os.path.join(_project_root, "ckpts", "moge2.pt")
        if os.path.exists(local_path):
            self.moge_model = MoGeModel.from_pretrained(local_path).to(self.device).eval()
            source = local_path
        else:
            self.moge_model = MoGeModel.from_pretrained(repo_id).to(self.device).eval()
            source = repo_id
        print(f"[WaypointGenerator] MoGe-2 loaded from {source}.")

    # ------------------------------------------------------------------
    # Video → Frames
    # ------------------------------------------------------------------

    @staticmethod
    def extract_frames(video_path: str, interval: int = 10) -> list[np.ndarray]:
        """从视频或图片目录中按固定间隔抽帧，返回 RGB numpy 图像列表。"""
        if os.path.isdir(video_path):
            exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            files = sorted(
                [f for f in os.listdir(video_path) if f.lower().endswith(exts)],
                key=lambda x: (len(x), x),
            )
            frames = []
            for fname in files:
                img = cv2.imread(os.path.join(video_path, fname))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            source = os.path.basename(video_path.rstrip("/"))
        else:
            cap = cv2.VideoCapture(video_path)
            frames = []
            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                idx += 1
            cap.release()
            source = os.path.basename(video_path)
        print(f"[WaypointGenerator] Extracted {len(frames)} frames (interval={interval}) from {source}")
        return frames

    # ------------------------------------------------------------------
    # Pi3X: 解码相机位姿和深度
    # ------------------------------------------------------------------

    def run_pi3(self, frames: list[np.ndarray]) -> dict:
        """
        用 Pi3X VO 管线从帧序列中解码位姿和点云。

        Returns:
            dict with keys:
                camera_poses: (N, 4, 4) numpy, cam-to-world SE(3)
                local_depth:  (N, H', W') numpy, Pi3 预测的局部深度 (Z_pred)
                conf:         (N, H', W') numpy, 置信度
                points:       (N, H', W', 3) numpy, 世界坐标点云
        """
        assert self.pi3_pipe is not None, "Call load_pi3() first."
        from pi3.utils.basic import load_multimodal_data

        # load_multimodal_data 接受目录或视频路径，这里我们直接构造 tensor
        # 参考 Pi3 的 load_images_as_tensor 逻辑
        imgs = self._frames_to_pi3_input(frames)  # (1, N, 3, H, W)

        with torch.no_grad():
            res = self.pi3_pipe(
                imgs=imgs,
                dtype=self.dtype,
                chunk_size=self.pi3_chunk_size,
                overlap=self.pi3_overlap,
                conf_thre=self.pi3_conf_thre,
            )

        camera_poses = res["camera_poses"][0].cpu().numpy()   # (N, 4, 4)
        conf = res["conf"][0].cpu().numpy()                    # (N, H, W)
        points = res["points"][0].cpu().numpy()                # (N, H, W, 3)

        # 局部深度: 从世界点反推回相机坐标系的 Z 分量
        # local_depth = local_points[..., 2], 但 VO 管线只输出全局 points
        # 这里通过逆变换计算: local = inv(pose) @ world_point
        local_depth = self._compute_local_depth(camera_poses, points)  # (N, H, W)

        return {
            "camera_poses": camera_poses,
            "local_depth": local_depth,
            "conf": conf,
            "points": points,
        }

    def _frames_to_pi3_input(self, frames: list[np.ndarray]) -> torch.Tensor:
        """将 RGB numpy 帧列表转为 Pi3 所需的 (1, N, 3, H, W) tensor。"""
        PIXEL_LIMIT = 255000
        PATCH_SIZE = 14

        h0, w0 = frames[0].shape[:2]
        # 缩放使总像素不超过 PIXEL_LIMIT，且 H/W 为 14 的倍数
        scale = min(1.0, (PIXEL_LIMIT / (h0 * w0)) ** 0.5)
        h = int(h0 * scale) // PATCH_SIZE * PATCH_SIZE
        w = int(w0 * scale) // PATCH_SIZE * PATCH_SIZE

        imgs = []
        for frame in frames:
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).float() / 255.0  # (H, W, 3)
            imgs.append(tensor.permute(2, 0, 1))  # (3, H, W)

        imgs = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)  # (1, N, 3, H, W)
        return imgs

    @staticmethod
    def _compute_local_depth(
        camera_poses: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """
        将世界坐标点云反变换回各帧相机坐标系，提取 Z 分量。

        Args:
            camera_poses: (N, 4, 4) cam-to-world
            points: (N, H, W, 3) world coordinates

        Returns:
            local_depth: (N, H, W) depth in each camera's local frame
        """
        N, H, W, _ = points.shape
        local_depth = np.zeros((N, H, W), dtype=np.float32)
        for i in range(N):
            R = camera_poses[i, :3, :3]
            t = camera_poses[i, :3, 3]
            # world_to_cam: p_cam = R^T @ (p_world - t)
            pts = points[i].reshape(-1, 3)  # (H*W, 3)
            pts_cam = (pts - t) @ R  # (H*W, 3), equivalent to R^T @ (p - t) row-wise
            local_depth[i] = pts_cam[:, 2].reshape(H, W)
        return local_depth

    # ------------------------------------------------------------------
    # MoGe-2: 度量深度估计
    # ------------------------------------------------------------------

    def run_moge(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        对每帧运行 MoGe-2 获取度量深度图。

        Returns:
            list of (H_orig, W_orig) numpy depth maps in meters
        """
        assert self.moge_model is not None, "Call load_moge() first."

        depth_maps = []
        for i, frame in enumerate(frames):
            img_tensor = (
                torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            ).to(self.device)  # (3, H, W)

            output = self.moge_model.infer(
                img_tensor,
                resolution_level=self.moge_resolution_level,
                use_fp16=True,
            )
            depth = output["depth"].cpu().numpy()  # (H, W) metric meters
            depth_maps.append(depth)

        print(f"[WaypointGenerator] MoGe-2 processed {len(frames)} frames.")
        return depth_maps

    # ------------------------------------------------------------------
    # Scale Recovery (论文 Section 3.2, Eq.4-5)
    # ------------------------------------------------------------------

    def compute_scale_factor(
        self,
        pi3_local_depth: np.ndarray,
        moge_depth_maps: list[np.ndarray],
        pi3_conf: np.ndarray,
    ) -> float:
        """
        计算全局尺度因子 S = median(D_ref / Z_pred) 在有效区域内。

        论文公式:
            M_t = {(u,v) | tau_min < D_ref(u,v) < tau_max and Z_pred(u,v) > 0}
            S = median( union_t { D_ref(u,v) / Z_pred(u,v) for (u,v) in M_t } )

        Args:
            pi3_local_depth: (N, H_pi3, W_pi3) Pi3 预测的局部深度
            moge_depth_maps: list of (H_moge, W_moge) MoGe 度量深度
            pi3_conf: (N, H_pi3, W_pi3) Pi3 置信度

        Returns:
            scale factor S (float)
        """
        all_ratios = []
        N = pi3_local_depth.shape[0]

        for t in range(N):
            z_pred = pi3_local_depth[t]  # (H_pi3, W_pi3)
            d_ref = moge_depth_maps[t]    # (H_moge, W_moge)

            # 将 MoGe 深度 resize 到 Pi3 分辨率
            h_pi3, w_pi3 = z_pred.shape
            d_ref_resized = cv2.resize(
                d_ref, (w_pi3, h_pi3), interpolation=cv2.INTER_LINEAR
            )

            # 有效掩码 M_t (论文 Eq.4)
            mask = (
                (d_ref_resized > self.tau_min)
                & (d_ref_resized < self.tau_max)
                & (z_pred > 0)
                & (pi3_conf[t] > self.pi3_conf_thre)
            )

            if mask.sum() == 0:
                continue

            ratios = d_ref_resized[mask] / z_pred[mask]
            all_ratios.append(ratios)

        if len(all_ratios) == 0:
            print("[WaypointGenerator] Warning: no valid pixels for scale estimation, using S=1.0")
            return 1.0

        all_ratios = np.concatenate(all_ratios)
        S = float(np.median(all_ratios))
        print(f"[WaypointGenerator] Scale factor S = {S:.4f}")
        return S

    # ------------------------------------------------------------------
    # Waypoint Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_waypoints(
        camera_poses: np.ndarray, scale_factor: float
    ) -> list[Waypoint]:
        """
        从相机位姿提取度量航点 (x, y, z, yaw)。

        论文: W_t = S * w_t

        Args:
            camera_poses: (N, 4, 4) cam-to-world SE(3)，Pi3 输出（归一化尺度）
            scale_factor: 全局尺度因子 S

        Returns:
            list of Waypoint
        """
        waypoints = []
        for i in range(camera_poses.shape[0]):
            pose = camera_poses[i]
            # 位置: 平移向量 * 尺度因子
            position = pose[:3, 3] * scale_factor

            # 朝向: 相机 Z 轴在世界坐标系中的方向 (OpenCV: Z = forward)
            forward = pose[:3, 2]
            yaw = float(np.arctan2(forward[0], forward[2]))

            waypoints.append(Waypoint(
                x=float(position[0]),
                y=float(position[1]),
                z=float(position[2]),
                yaw=yaw,
            ))
        return waypoints

    # ------------------------------------------------------------------
    # End-to-End Pipeline
    # ------------------------------------------------------------------

    def generate(
        self,
        video_path: str,
        frame_interval: int = 10,
        output_dir: Optional[str] = None,
    ) -> list[Waypoint]:
        """
        完整流水线: 视频 → 度量航点。

        Args:
            video_path: 评分后选出的最优视频 V* 路径
            frame_interval: 抽帧间隔
            output_dir: 输出目录，默认以视频名称命名

        Returns:
            list of Waypoint (度量空间)
        """
        # 确定输出目录 (以视频/目录名命名)
        if output_dir is None:
            name = os.path.splitext(os.path.basename(video_path.rstrip("/")))[0]
            output_dir = os.path.join("outputs", name)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: 抽帧
        frames = self.extract_frames(video_path, interval=frame_interval)
        assert len(frames) >= 2, f"Need at least 2 frames, got {len(frames)}"

        # Step 2: Pi3X 解码位姿和深度
        pi3_result = self.run_pi3(frames)

        # Step 3: MoGe-2 度量深度
        moge_depths = self.run_moge(frames)

        # Step 4: 尺度恢复 (论文 Eq.4-5)
        scale_factor = self.compute_scale_factor(
            pi3_local_depth=pi3_result["local_depth"],
            moge_depth_maps=moge_depths,
            pi3_conf=pi3_result["conf"],
        )

        # Step 5: 提取度量航点
        waypoints = self.extract_waypoints(
            camera_poses=pi3_result["camera_poses"],
            scale_factor=scale_factor,
        )

        # 保存所有结果
        self._save_results(output_dir, frames, pi3_result, moge_depths,
                           scale_factor, waypoints)

        print(f"[WaypointGenerator] Generated {len(waypoints)} waypoints.")
        print(f"[WaypointGenerator] All results saved to {output_dir}/")
        return waypoints

    @staticmethod
    def _save_results(
        output_dir: str,
        frames: list[np.ndarray],
        pi3_result: dict,
        moge_depths: list[np.ndarray],
        scale_factor: float,
        waypoints: list["Waypoint"],
    ):
        """保存抽帧、Pi3/MoGe 中间结果和最终航点到输出目录。"""
        import json

        # 抽取的帧
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(
                os.path.join(frames_dir, f"{i:04d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

        # Pi3 结果
        pi3_dir = os.path.join(output_dir, "pi3")
        os.makedirs(pi3_dir, exist_ok=True)
        np.save(os.path.join(pi3_dir, "camera_poses.npy"), pi3_result["camera_poses"])
        np.save(os.path.join(pi3_dir, "local_depth.npy"), pi3_result["local_depth"])
        np.save(os.path.join(pi3_dir, "conf.npy"), pi3_result["conf"])
        np.save(os.path.join(pi3_dir, "points.npy"), pi3_result["points"])

        # MoGe 结果
        moge_dir = os.path.join(output_dir, "moge")
        os.makedirs(moge_dir, exist_ok=True)
        for i, depth in enumerate(moge_depths):
            np.save(os.path.join(moge_dir, f"depth_{i:04d}.npy"), depth)

        # 尺度因子 + 航点
        wp_list = [
            {"frame": i, "x": w.x, "y": w.y, "z": w.z, "yaw": w.yaw}
            for i, w in enumerate(waypoints)
        ]
        result = {
            "scale_factor": scale_factor,
            "num_frames": len(frames),
            "waypoints": wp_list,
        }
        with open(os.path.join(output_dir, "waypoints.json"), "w") as f:
            json.dump(result, f, indent=2)


# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NavDreamer Waypoint Generator")
    parser.add_argument("--video", type=str, required=True, help="Path to selected video V* or image directory")
    parser.add_argument("--interval", type=int, default=10, help="Frame sampling interval (video only)")
    parser.add_argument("--tau_min", type=float, default=0.5, help="Min valid depth (meters)")
    parser.add_argument("--tau_max", type=float, default=30.0, help="Max valid depth (meters)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: outputs/<video_name>)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gen = WaypointGenerator(
        device=args.device,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    )
    gen.load_pi3()
    gen.load_moge()

    gen.generate(args.video, frame_interval=args.interval, output_dir=args.output_dir)
