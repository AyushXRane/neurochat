#!/usr/bin/env python3
"""Generate the bundled sample data. Deterministic: same inputs, same bytes.

Everything written here is synthetic. No subject data, no derived atlas, nothing
with a redistribution licence attached. That is the point — `git clone` gives you
something that runs offline, and the real atlases are one `load_atlas()` away.

    python scripts/make_sample_data.py

Outputs, all on the FSL MNI152 4mm grid (45, 54, 45):

    demo16_atlas.nii.gz        16 geometric regions, integer-labelled
    demo16_labels.json         label list + indices for the R1 lookup table
    phantom_t1.nii.gz          structural-like volume (+ BIDS sidecar)
    phantom_pet_baseline.nii.gz    PET-like uptake, contains NaNs on purpose
    phantom_pet_followup.nii.gz    same phantom with regional change, for compare_volumes
    nospace_volume.nii.gz      identical grid with sform/qform codes zeroed
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "sample_data"

# FSL MNI152 4mm grid. Chosen so the whole sample set is a few hundred KB.
SHAPE = (45, 54, 45)
AFFINE = np.array(
    [
        [-4.0, 0.0, 0.0, 90.0],
        [0.0, 4.0, 0.0, -126.0],
        [0.0, 0.0, 4.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Half-axes of the ellipsoid standing in for a brain, in mm.
BRAIN_RADII = np.array([68.0, 86.0, 62.0])
BRAIN_CENTER = np.array([0.0, -18.0, 12.0])

# label index -> (name, centre in MNI mm, radii in mm)
REGION_SPECS = [
    ("Left Anterior Superior Block", (-30, 34, 30), (14, 14, 12)),
    ("Right Anterior Superior Block", (30, 34, 30), (14, 14, 12)),
    ("Left Anterior Inferior Block", (-32, 26, -18), (13, 13, 11)),
    ("Right Anterior Inferior Block", (32, 26, -18), (13, 13, 11)),
    ("Left Posterior Superior Block", (-28, -62, 40), (14, 14, 12)),
    ("Right Posterior Superior Block", (28, -62, 40), (14, 14, 12)),
    ("Left Posterior Inferior Block", (-30, -64, -14), (13, 13, 11)),
    ("Right Posterior Inferior Block", (30, -64, -14), (13, 13, 11)),
    ("Left Lateral Band", (-52, -14, 6), (10, 22, 12)),
    ("Right Lateral Band", (52, -14, 6), (10, 22, 12)),
    ("Left Deep Sphere", (-16, -8, 4), (11, 11, 11)),
    ("Right Deep Sphere", (16, -8, 4), (11, 11, 11)),
    ("Left Ventral Shell", (-24, -26, -22), (12, 14, 8)),
    ("Right Ventral Shell", (24, -26, -22), (12, 14, 8)),
    ("Midline Dorsal Cap", (0, -20, 62), (16, 20, 8)),
    ("Midline Ventral Core", (0, -26, -30), (12, 16, 9)),
]


def world_grid() -> np.ndarray:
    """(x, y, z) world coordinates of every voxel centre, shaped (*SHAPE, 3)."""
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in SHAPE], indexing="ij")
    voxels = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).astype(float)
    return (voxels @ AFFINE.T)[..., :3]


def ellipsoid(coords: np.ndarray, center, radii) -> np.ndarray:
    """Squared normalised radius; <= 1 is inside."""
    delta = (coords - np.asarray(center, dtype=float)) / np.asarray(radii, dtype=float)
    return (delta**2).sum(axis=-1)


def write_sidecar(stem: str, description: str, units: str = "arbitrary") -> None:
    """A BIDS sidecar naming the template space, so detection has something solid."""
    (OUT / f"{stem}.json").write_text(
        json.dumps(
            {
                "Space": "MNI152NLin6Asym",
                "Description": description,
                "Units": units,
            },
            indent=2,
        )
        + "\n"
    )


def save(img: nib.Nifti1Image, name: str, xform_code: int = 4) -> Path:
    img.header.set_xyzt_units("mm")
    img.header["descrip"] = b"neurochat synthetic sample data - not subject data"
    img.set_sform(AFFINE, code=xform_code)
    img.set_qform(AFFINE, code=xform_code)
    path = OUT / name
    # Fixed mtime in the gzip stream keeps regenerated files byte-comparable.
    nib.save(img, str(path))
    print(f"  {path.name:32s} {path.stat().st_size / 1024:7.1f} KB")
    return path


def build_atlas(coords: np.ndarray) -> None:
    labels_out = ["Background"]
    indices_out = [0]
    data = np.zeros(SHAPE, dtype=np.int16)
    # Painted in reverse so earlier entries win any overlap deterministically.
    for offset, (name, center, radii) in reversed(list(enumerate(REGION_SPECS))):
        index = offset + 1
        data[ellipsoid(coords, center, radii) <= 1.0] = index
    for offset, (name, _, _) in enumerate(REGION_SPECS):
        labels_out.append(name)
        indices_out.append(offset + 1)

    save(nib.Nifti1Image(data, AFFINE), "demo16_atlas.nii.gz")
    (OUT / "demo16_labels.json").write_text(
        json.dumps(
            {
                "atlas_id": "demo-16",
                "space": "MNI152NLin6Asym",
                "labels": labels_out,
                "indices": indices_out,
                "warning": (
                    "Synthetic geometric shapes. These are NOT anatomical structures. "
                    "Use only for offline smoke tests and demos."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  demo16_labels.json               {len(labels_out) - 1} regions")


def build_phantoms(coords: np.ndarray) -> None:
    rng = np.random.default_rng(20260728)
    brain = ellipsoid(coords, BRAIN_CENTER, BRAIN_RADII)
    inside = brain <= 1.0

    # --- structural-like phantom: three tissue-ish shells plus mild noise ---
    t1 = np.zeros(SHAPE, dtype=np.float32)
    t1[inside] = 320.0
    t1[brain <= 0.72] = 780.0
    t1[brain <= 0.30] = 540.0
    for _, center, radii in REGION_SPECS[:8]:
        t1[ellipsoid(coords, center, radii) <= 1.0] = 690.0
    t1 = t1 + rng.normal(0, 6.0, SHAPE).astype(np.float32)
    t1[~inside] = 0.0
    save(nib.Nifti1Image(np.round(t1, 3).astype(np.float32), AFFINE), "phantom_t1.nii.gz")
    write_sidecar("phantom_t1", "Synthetic structural-like phantom. Not subject data.")

    # --- PET-like baseline: smooth uptake, elevated in a few demo regions ---
    base = np.zeros(SHAPE, dtype=np.float32)
    base[inside] = 1.0
    base = base * (1.35 - 0.35 * np.clip(brain, 0, 1)).astype(np.float32)
    for name, center, radii in REGION_SPECS:
        if "Anterior" in name or "Deep" in name:
            base[ellipsoid(coords, center, radii) <= 1.0] *= 1.45
    base = (base + rng.normal(0, 0.03, SHAPE)).astype(np.float32)
    base[~inside] = 0.0

    # Deliberate NaNs: a slab of dropout inside the brain. roi_stats must report
    # how many voxels it excluded rather than quietly averaging around them.
    dropout = inside & (coords[..., 2] > 46) & (coords[..., 1] < -54)
    base[dropout] = np.nan
    print(f"  (NaN voxels injected into PET phantom: {int(dropout.sum())})")
    save(nib.Nifti1Image(np.round(base, 4).astype(np.float32), AFFINE), "phantom_pet_baseline.nii.gz")
    write_sidecar(
        "phantom_pet_baseline",
        "Synthetic PET-like uptake phantom with deliberate NaN dropout. Not subject data.",
        units="arbitrary uptake",
    )

    # --- follow-up: a real, localised change plus the same noise structure ---
    follow = base.copy()
    for name, center, radii in REGION_SPECS:
        if name.startswith("Left Posterior"):
            follow[ellipsoid(coords, center, radii) <= 1.0] *= 0.72
        if name == "Right Deep Sphere":
            follow[ellipsoid(coords, center, radii) <= 1.0] *= 1.18
    follow = (follow + rng.normal(0, 0.01, SHAPE)).astype(np.float32)
    follow[~inside] = 0.0
    save(
        nib.Nifti1Image(np.round(follow, 4).astype(np.float32), AFFINE),
        "phantom_pet_followup.nii.gz",
    )
    write_sidecar(
        "phantom_pet_followup",
        "Synthetic PET-like follow-up with localised change. Not subject data.",
        units="arbitrary uptake",
    )

    # --- the volume that refuses to be resolved (acceptance test 2) ---
    stripped = nib.Nifti1Image(np.round(t1, 3).astype(np.float32), AFFINE)
    save(stripped, "nospace_volume.nii.gz", xform_code=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coords = world_grid()
    print(f"Writing sample data to {OUT}")
    build_atlas(coords)
    build_phantoms(coords)
    print("Done.")


if __name__ == "__main__":
    main()
