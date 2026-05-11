from __future__ import annotations

import numpy as np


PAPER_QUARTER_SIZE = 15
PAPER_FULL_SIZE = 30


def enforce_diagonal_symmetry(quarter: np.ndarray) -> np.ndarray:
    """Average each upper-left quarter with its transpose to encode dual-polarization symmetry."""
    arr = np.asarray(quarter, dtype=np.float32)
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError(f"Expected square quarter matrices, got shape {arr.shape}.")
    return 0.5 * (arr + np.swapaxes(arr, -1, -2))


def apply_outer_rim_to_quarter(quarter: np.ndarray) -> np.ndarray:
    """Force the outer metacell rim to void in the compressed upper-left representation."""
    arr = np.array(quarter, dtype=np.float32, copy=True)
    arr[..., 0, :] = 0.0
    arr[..., :, 0] = 0.0
    return arr


def binarize_geometry(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probability) >= threshold).astype(np.float32)


def expand_sandwich_channels(geometry: np.ndarray) -> np.ndarray:
    """
    Convert paper cDCGAN output channels into physical triple-layer channels.

    N=1 means the same pattern is used on all three layers.
    N=2 means a sandwich metacell: top/bottom share channel 0 and middle uses channel 1.
    N=3 is already a full triple-layer representation.
    """
    arr = np.asarray(geometry, dtype=np.float32)
    if arr.ndim < 3:
        raise ValueError(f"Expected [..., channels, height, width], got shape {arr.shape}.")

    channels = arr.shape[-3]
    if channels == 1:
        return np.repeat(arr, 3, axis=-3)
    if channels == 2:
        return np.concatenate([arr[..., :1, :, :], arr[..., 1:2, :, :], arr[..., :1, :, :]], axis=-3)
    if channels == 3:
        return arr
    raise ValueError("Paper reconstruction supports N=1, N=2, or explicit 3-layer geometry.")


def reconstruct_full_metacell(quarter_geometry: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """
    Reconstruct full 30x30 triple-layer metacells from 15x15 upper-left quarters.

    The reconstruction applies the paper's prior knowledge: diagonal symmetry inside the
    quarter, horizontal/vertical mirror symmetry over the full cell, and a void outer rim.
    """
    quarter = np.asarray(quarter_geometry, dtype=np.float32)
    thresholded = threshold is not None
    if thresholded:
        quarter = binarize_geometry(quarter, threshold)
    if quarter.shape[-2:] != (PAPER_QUARTER_SIZE, PAPER_QUARTER_SIZE):
        raise ValueError(
            f"Expected quarter geometry size {(PAPER_QUARTER_SIZE, PAPER_QUARTER_SIZE)}, "
            f"got {quarter.shape[-2:]}."
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
