"""Pure compute functions shared between the live tools and the exported script.

Rule R2 says every action emits code and re-running that code reproduces the
result. The obvious way to break that promise is to compute one way at runtime and
emit slightly different code — a different NaN policy, a different resampling
interpolation — and never notice, because nobody re-runs the script.

So this module is the single source of truth, and ``script.py`` inlines its
*source text* into every export via ``inspect.getsource``. The runtime path and the
exported path are then literally the same code.

Constraints that keep that possible:

* Import only ``numpy``, ``nibabel`` and ``nilearn``. Nothing from ``neurochat``.
* No global state, no I/O beyond reading the paths handed in.
* Round every returned float identically, so two runs compare byte for byte.
"""

from __future__ import annotations

import numpy as np

ROUND_DP = 6


def same_grid(a, b) -> bool:
    """True when two images share a voxel grid, so no resampling is needed."""
    return bool(
        a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-5, rtol=0)
    )


def region_mask(atlas_path: str, index: int, target_img=None) -> np.ndarray:
    """Boolean mask for one atlas label, on the target volume's grid.

    Label images are resampled with nearest-neighbour interpolation only. Any other
    interpolation invents fractional memberships between two structures and would
    make ``roi_stats`` quietly wrong at every region boundary.
    """
    import nibabel as nib

    atlas_img = nib.load(atlas_path)
    if target_img is not None and not same_grid(atlas_img, target_img):
        from nilearn import image as nli

        atlas_img = nli.resample_to_img(
            atlas_img,
            target_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    labels = np.asanyarray(atlas_img.dataobj)
    if labels.ndim == 4:
        labels = labels[..., 0]
    return np.rint(labels).astype(np.int32) == int(index)


def mask_from_file(mask_path: str, target_img=None) -> np.ndarray:
    """Boolean mask from a user-supplied mask image, on the target grid."""
    import nibabel as nib

    mask_img = nib.load(mask_path)
    if target_img is not None and not same_grid(mask_img, target_img):
        from nilearn import image as nli

        mask_img = nli.resample_to_img(
            mask_img,
            target_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    data = np.asanyarray(mask_img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    return data > 0


def volume_data(img) -> np.ndarray:
    """Float64 view of a 3D volume. 4D input collapses to its first frame."""
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    if data.ndim == 4:
        data = data[..., 0]
    return data


def thickness_mm(mask: np.ndarray, zooms) -> float:
    """How thick the region actually is, in millimetres.

    This, not voxel size, is what decides partial-volume contamination — a file's voxel
    size is not its resolution. PET is routinely written on a 1mm grid while its true
    effective resolution is 4-6mm, so any voxel-count measure reports that everything is
    fine when it is not.

    Measured as the diameter of the largest sphere that fits inside the region, via a
    distance transform. The obvious alternative — the smallest side of the bounding box
    — is wrong for curved structures: hippocampus is about 10mm thick but banana-shaped,
    so its bounding box is chunky in every direction and it looks comfortably large when
    it is nothing of the sort. The inscribed sphere does not care about curvature.
    """
    from scipy import ndimage

    if not mask.any():
        return 0.0
    # Crop to the bounding box first; the transform is over the whole array otherwise.
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    local = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    padded = np.pad(local, 1)  # so the region's own edge counts as outside
    distance = ndimage.distance_transform_edt(padded, sampling=np.abs(np.asarray(zooms[:3], float)))
    return round(float(distance.max()) * 2.0, 2)


def boundary_fraction(mask: np.ndarray) -> float:
    """What fraction of a mask's voxels touch its edge.

    This is the cheap, honest proxy for partial-volume contamination. A region that
    is large relative to the voxels is mostly interior, and its mean is mostly the
    thing you asked for. A region only a few voxels across is nearly all edge — and
    since every imaging modality blurs across boundaries, most of what you measure
    there is actually the tissue next door. PET is the bad case: at 4-6mm resolution
    a structure like the hippocampus is a handful of voxels wide.

    Returns 0.0 for a solid interior, approaching 1.0 for a region that is all edge.
    """
    from scipy import ndimage

    total = int(mask.sum())
    if total == 0:
        return 0.0
    interior = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))
    return round(1.0 - float(interior.sum()) / total, 4)


def summarize_roi(img, mask: np.ndarray, exclude_zeros: bool = False) -> dict:
    """Descriptive statistics inside a mask, with every exclusion counted.

    Silent NaN handling is how wrong numbers get into papers, so the return value
    always says how many voxels were dropped and why. Zeros are *included* by
    default and merely counted — an implicit "zeros are background" rule is a
    modelling assumption, not a detail, and the caller should have to ask for it.
    """
    data = volume_data(img)
    if data.shape != mask.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match volume shape {data.shape}; "
            "resample the mask to the volume grid first"
        )

    values = data[mask]
    n_in_mask = int(values.size)
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    finite = values[np.isfinite(values)]
    n_zero = int((finite == 0).sum())

    used = finite[finite != 0] if exclude_zeros else finite
    n_used = int(used.size)

    if n_used == 0:
        stats = dict(mean=None, sd=None, median=None, min=None, max=None)
    else:
        stats = dict(
            mean=round(float(np.mean(used)), ROUND_DP),
            sd=round(float(np.std(used, ddof=1)) if n_used > 1 else 0.0, ROUND_DP),
            median=round(float(np.median(used)), ROUND_DP),
            min=round(float(np.min(used)), ROUND_DP),
            max=round(float(np.max(used)), ROUND_DP),
        )

    return {
        "n_voxels_in_mask": n_in_mask,
        "n_voxels_used": n_used,
        **stats,
        # How much of this region is edge, and how narrow it physically is. Together
        # these say how much of the mean above is really the tissue next door.
        "boundary_fraction": boundary_fraction(mask),
        "thickness_mm": thickness_mm(mask, img.header.get_zooms()),
        "exclusions": {
            "nan": n_nan,
            "non_finite": n_inf,
            "zero": n_zero if exclude_zeros else 0,
            "zeros_included": not exclude_zeros,
            "n_zero_in_mask": n_zero,
        },
    }


def summarize_volume(img) -> dict:
    """Whole-volume descriptives, used when a volume is loaded."""
    data = volume_data(img)
    finite = data[np.isfinite(data)]
    return {
        "n_voxels": int(data.size),
        "n_nan": int(np.isnan(data).sum()),
        "n_non_finite": int(np.isinf(data).sum()),
        "n_zero": int((finite == 0).sum()),
        "min": round(float(finite.min()), ROUND_DP) if finite.size else None,
        "max": round(float(finite.max()), ROUND_DP) if finite.size else None,
        "mean": round(float(finite.mean()), ROUND_DP) if finite.size else None,
    }


def combine_volumes(img_a, img_b, method: str = "difference", epsilon: float = 1e-8):
    """Voxelwise difference or ratio of two volumes, b resampled onto a's grid.

    This is arithmetic, not inference. There is no test statistic here and none is
    implied: a difference map says two numbers differ, not that the difference means
    anything.
    """
    import nibabel as nib

    if method not in ("difference", "ratio"):
        raise ValueError(f"method must be 'difference' or 'ratio', got {method!r}")

    if not same_grid(img_a, img_b):
        from nilearn import image as nli

        img_b = nli.resample_to_img(
            img_b, img_a, interpolation="continuous", force_resample=True, copy_header=True
        )

    a = volume_data(img_a)
    b = volume_data(img_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "difference":
            result = a - b
        else:
            denominator = np.where(np.abs(b) < epsilon, np.nan, b)
            result = a / denominator

    return nib.Nifti1Image(result.astype(np.float32), img_a.affine, img_a.header)
