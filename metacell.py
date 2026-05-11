from __future__ import annotations

import numpy as np


PAPER_QUARTER_SIZE = 15
PAPER_FULL_SIZE = 30


def enforce_diagonal_symmetry(quarter: np.ndarray) -> np.ndarray:
    """把左上角矩阵处理成关于对角线对称。"""
    arr = np.asarray(quarter, dtype=np.float32)
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError(f"左上角图案应为方阵，当前形状为 {arr.shape}。")
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def apply_outer_rim_to_quarter(quarter: np.ndarray) -> np.ndarray:
    """论文中单元最外圈留空，这里先处理压缩表示里的上边和左边。"""
    arr = np.array(quarter, dtype=np.float32, copy=True)
    arr[..., 0, :] = 0.0
    arr[..., :, 0] = 0.0
    return arr


def binarize_geometry(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probability) >= threshold).astype(np.float32)


def expand_sandwich_channels(geometry: np.ndarray) -> np.ndarray:
    """
    把网络输出通道展开成实际三层结构。

    N=1 时三层共用一个图案；N=2 时上下层共用第 0 通道，中间层使用第 1 通道。
    """
    arr = np.asarray(geometry, dtype=np.float32)
    if arr.ndim < 3:
        raise ValueError(f"几何数据应至少包含 [通道, 高, 宽]，当前形状为 {arr.shape}。")

    channels = arr.shape[-3]
    if channels == 1:
        return np.repeat(arr, 3, axis=-3)
    if channels == 2:
        return np.concatenate([arr[..., :1, :, :], arr[..., 1:2, :, :], arr[..., :1, :, :]], axis=-3)
    if channels == 3:
        return arr
    raise ValueError("三层重构只支持 N=1、N=2，或已经展开好的 N=3。")


def reconstruct_full_metacell(quarter_geometry: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """
    从 15x15 的左上角图案恢复完整 30x30 三层单元。

    这里用的是论文中的对称先验：先做对角线对称，再沿水平和垂直方向镜像。
    """
    quarter = np.asarray(quarter_geometry, dtype=np.float32)
    thresholded = threshold is not None
    if thresholded:
        quarter = binarize_geometry(quarter, threshold)
    if quarter.shape[-2:] != (PAPER_QUARTER_SIZE, PAPER_QUARTER_SIZE):
        raise ValueError(
            f"左上角图案尺寸应为 {(PAPER_QUARTER_SIZE, PAPER_QUARTER_SIZE)}，"
            f"当前为 {quarter.shape[-2:]}。"
        )

    quarter = apply_outer_rim_to_quarter(enforce_diagonal_symmetry(quarter))
    if thresholded:
        quarter = binarize_geometry(quarter, 0.5)
    layers = expand_sandwich_channels(quarter)
    top_half = np.concatenate([layers, np.flip(layers, axis=-1)], axis=-1)
    full = np.concatenate([top_half, np.flip(top_half, axis=-2)], axis=-2)
    full[..., 0, :] = 0.0
    full[..., -1, :] = 0.0
    full[..., :, 0] = 0.0
    full[..., :, -1] = 0.0
    return full.astype(np.float32)
