from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from metacell import (
    PAPER_FULL_SIZE,
    PAPER_QUARTER_SIZE,
    apply_outer_rim_to_quarter,
    enforce_diagonal_symmetry,
)


class MetacellDataset(Dataset):
    """
    Expected .npz fields:
    - geometries: [num_samples, N, H, W], usually H=W=15 for paper-ready data.
      Full 30x30 geometries are accepted and compressed to the upper-left quarter.
    - responses:  [num_samples, M], desired or simulated transmission response vectors.
    """

    def __init__(
        self,
        npz_path: str,
        image_size: int = PAPER_QUARTER_SIZE,
        enforce_paper_prior: bool = True,
    ):
        data = np.load(npz_path)
        self.geometries = self._prepare_geometries(data["geometries"], image_size, enforce_paper_prior)
        self.responses = data["responses"].astype(np.float32)

        if len(self.geometries) != len(self.responses):
            raise ValueError("geometries and responses must contain the same number of samples.")

    @staticmethod
    def _prepare_geometries(geometries: np.ndarray, image_size: int, enforce_paper_prior: bool) -> np.ndarray:
        arr = geometries.astype(np.float32)
        if arr.ndim != 4:
            raise ValueError(f"geometries should have shape [num_samples, N, H, W], got {arr.shape}.")

        if image_size == PAPER_QUARTER_SIZE and arr.shape[-2:] == (PAPER_FULL_SIZE, PAPER_FULL_SIZE):
            arr = arr[..., :PAPER_QUARTER_SIZE, :PAPER_QUARTER_SIZE]
        elif arr.shape[-2:] != (image_size, image_size):
            raise ValueError(f"Expected geometry size {(image_size, image_size)}, got {arr.shape[-2:]}.")

        arr = np.clip(arr, 0.0, 1.0)
        if enforce_paper_prior and image_size == PAPER_QUARTER_SIZE:
            arr = apply_outer_rim_to_quarter(enforce_diagonal_symmetry(arr))
        return arr.astype(np.float32)

    def __len__(self) -> int:
        return len(self.geometries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        geometry = torch.from_numpy(self.geometries[idx])
        response = torch.from_numpy(self.responses[idx])
        return geometry, response


@dataclass
class SyntheticConfig:
    num_samples: int = 6000
    condition_dim: int = 64
    geometry_channels: int = 2
    image_size: int = PAPER_QUARTER_SIZE
    freq_min: float = 8.0
    freq_max: float = 13.0
    seed: int = 42


def _place_block(quarter: np.ndarray, row: int, col: int, horizontal: bool) -> None:
    if horizontal:
        quarter[row, col : col + 5] = 1.0
    else:
        quarter[row : row + 5, col] = 1.0


def create_synthetic_data(cfg: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a lightweight paper-shaped dataset for pipeline checks.

    This is not a replacement for HFSS simulation data. It only encodes the paper priors:
    15x15 upper-left geometry, void outer rim, diagonal symmetry, and 1x5 block growth.
    """
    if cfg.image_size != PAPER_QUARTER_SIZE:
        raise ValueError("Synthetic paper-prior data generation expects image_size=15.")

    rng = np.random.default_rng(cfg.seed)
    responses = rng.uniform(0.0, 1.0, size=(cfg.num_samples, cfg.condition_dim)).astype(np.float32)
    geometries = np.zeros(
        (cfg.num_samples, cfg.geometry_channels, cfg.image_size, cfg.image_size),
        dtype=np.float32,
    )

    for i in range(cfg.num_samples):
        response = responses[i]
        for ch in range(cfg.geometry_channels):
            quarter = np.zeros((cfg.image_size, cfg.image_size), dtype=np.float32)
            density_control = response[(ch * 7) % cfg.condition_dim]
            block_count = 2 + int(10 * density_control)

            for block_idx in range(block_count):
                control = response[(block_idx + ch * 11) % cfg.condition_dim]
                horizontal = control >= 0.5
                if horizontal:
                    row = int(rng.integers(1, cfg.image_size))
                    col = int(rng.integers(1, cfg.image_size - 4))
                else:
                    row = int(rng.integers(1, cfg.image_size - 4))
                    col = int(rng.integers(1, cfg.image_size))
                _place_block(quarter, row, col, horizontal)

            geometries[i, ch] = apply_outer_rim_to_quarter(enforce_diagonal_symmetry(quarter))

    return geometries, responses


def save_synthetic_npz(path: str, cfg: SyntheticConfig) -> None:
    geometries, responses = create_synthetic_data(cfg)
    np.savez_compressed(
        path,
        geometries=geometries,
        responses=responses,
        freq_min=np.array(cfg.freq_min, dtype=np.float32),
        freq_max=np.array(cfg.freq_max, dtype=np.float32),
    )
