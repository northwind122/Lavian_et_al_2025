"""Alignment helpers for 2p split datasets.

This is a lightweight replacement for the missing fimpy.pipeline.alignment
module referenced in the notebooks. It performs simple rigid (x/y) alignment
using phase cross-correlation.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

from split_dataset import SplitDataset
from fimpy.pipeline.common import run_in_blocks


def _compute_reference(ds: SplitDataset, n_frames: int = 200) -> np.ndarray:
    """Compute a mean reference image from the first n_frames."""
    n = min(n_frames, ds.shape[0])
    ref = ds[:n]
    return ref.mean(axis=0)


def _align_block(
    block: np.ndarray,
    ref: np.ndarray,
    *,
    across_planes: bool = False,
    max_shift: Optional[float] = 15,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """Align a block to the reference using rigid x/y shifts."""
    block = np.asarray(block)
    aligned = np.empty_like(block)

    if block.ndim == 3:
        # (t, y, x)
        ref_2d = ref
        for t in range(block.shape[0]):
            shift, _, _ = phase_cross_correlation(ref_2d, block[t], upsample_factor=1)
            if max_shift is not None:
                shift = np.clip(shift, -max_shift, max_shift)
            aligned[t] = ndi_shift(block[t], shift, order=order, mode=mode, prefilter=False)
        return aligned

    if block.ndim != 4:
        raise ValueError(f"Expected 3D or 4D block, got shape {block.shape}")

    # (t, z, y, x)
    if across_planes:
        ref_2d = ref.mean(axis=0)
        for t in range(block.shape[0]):
            frame = block[t]
            shift, _, _ = phase_cross_correlation(ref_2d, frame.mean(axis=0), upsample_factor=1)
            if max_shift is not None:
                shift = np.clip(shift, -max_shift, max_shift)
            for z in range(frame.shape[0]):
                aligned[t, z] = ndi_shift(
                    frame[z], shift, order=order, mode=mode, prefilter=False
                )
    else:
        for t in range(block.shape[0]):
            for z in range(block.shape[1]):
                shift, _, _ = phase_cross_correlation(ref[z], block[t, z], upsample_factor=1)
                if max_shift is not None:
                    shift = np.clip(shift, -max_shift, max_shift)
                aligned[t, z] = ndi_shift(
                    block[t, z], shift, order=order, mode=mode, prefilter=False
                )

    return aligned


def align_2p_volume(
    dataset,
    output_dir: Optional[str] = None,
    *,
    across_planes: bool = False,
    reference_frames: int = 200,
    max_shift: Optional[float] = 15,
    n_jobs: int = 8,
    order: int = 1,
    mode: str = "nearest",
):
    """Align a SplitDataset (t,y,x) or (t,z,y,x) and save to <output_dir>/aligned.

    Parameters
    ----------
    dataset : SplitDataset or path
        Input dataset or path to a SplitDataset folder.
    output_dir : str
        Folder where the aligned dataset folder will be written.
    across_planes : bool
        If True, compute a single shift per frame and apply to all planes.
    reference_frames : int
        Number of initial frames to build the reference image.
    max_shift : float or None
        Max pixel shift in x/y to clip the estimate.
    n_jobs : int
        Parallel jobs for block processing.
    order : int
        Interpolation order for shifting.
    mode : str
        Padding mode for shifting.
    """
    if not isinstance(dataset, SplitDataset):
        dataset = SplitDataset(Path(dataset))

    ref = _compute_reference(dataset, n_frames=reference_frames)
    out_dir = Path(output_dir) if output_dir is not None else None

    return run_in_blocks(
        _align_block,
        dataset,
        ref,
        output_dir=out_dir,
        output_name="aligned",
        n_jobs=n_jobs,
        across_planes=across_planes,
        max_shift=max_shift,
        order=order,
        mode=mode,
    )
