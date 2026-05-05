from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MetacellDataset(Dataset):
    """
    期望的 npz 数据格式:
    - geometries: [num_samples, N, H, W]，浮点数，建议范围 [0, 1]
    - responses:  [num_samples, M]，浮点数（目标透射响应）
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.geometries = data["geometries"].astype(np.float32)
        self.responses = data["responses"].astype(np.float32)

        if len(self.geometries) != len(self.responses):
            raise ValueError("geometries 与 responses 的样本数量不一致。")

    def __len__(self) -> int:
        return len(self.geometries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        geometry = torch.from_numpy(self.geometries[idx])
        response = torch.from_numpy(self.responses[idx])
        return geometry, response


@dataclass
class SyntheticConfig:
    num_samples: int = 2048
    condition_dim: int = 64  # M
    geometry_channels: int = 4  # N
    image_size: int = 32
    freq_min: float = 8.0
    freq_max: float = 12.0
    seed: int = 42


def create_synthetic_data(cfg: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成可复现的合成数据，用于快速验证训练流程是否可跑通。
    - 条件向量: 随机响应向量
    - 几何矩阵: 由条件向量驱动的模式图
    """
    rng = np.random.default_rng(cfg.seed)
    responses = rng.uniform(0.0, 1.0, size=(cfg.num_samples, cfg.condition_dim)).astype(np.float32)

    geometries = np.zeros(
        (cfg.num_samples, cfg.geometry_channels, cfg.image_size, cfg.image_size),
        dtype=np.float32,
    )

    yy, xx = np.mgrid[0 : cfg.image_size, 0 : cfg.image_size]
    yy = yy.astype(np.float32) / max(cfg.image_size - 1, 1)
    xx = xx.astype(np.float32) / max(cfg.image_size - 1, 1)

    for i in range(cfg.num_samples):
        c = responses[i]
        amp = 0.2 + 0.8 * c[0]
        fx = 1.0 + 7.0 * c[1 % cfg.condition_dim]
        fy = 1.0 + 7.0 * c[2 % cfg.condition_dim]
        phase = 2.0 * np.pi * c[3 % cfg.condition_dim]
        base = amp * (np.sin(2 * np.pi * fx * xx + phase) + np.cos(2 * np.pi * fy * yy + phase))
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)

        for ch in range(cfg.geometry_channels):
            shift = c[(4 + ch) % cfg.condition_dim]
            geo = np.clip(base + 0.25 * shift, 0.0, 1.0)
            geometries[i, ch] = geo

    return geometries, responses


def save_synthetic_npz(path: str, cfg: SyntheticConfig) -> None:
    geometries, responses = create_synthetic_data(cfg)
    np.savez_compressed(path, geometries=geometries, responses=responses)
