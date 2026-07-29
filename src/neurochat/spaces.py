"""Deciding what space a volume is in, and refusing when we cannot tell.

Rule R1 says the model never invents coordinates. The first half of honouring that
is knowing which coordinate system a volume actually lives in. NIfTI headers are
often vague and frequently wrong, so this module reports *how* it knows, and
declines to resolve anatomical names when it does not know.

Detection order, most trustworthy first:

1. A BIDS JSON sidecar next to the file (``Space`` or ``SpatialReference``).
2. A ``space-<label>`` entity in the filename, per BIDS derivatives naming.
3. The NIfTI ``sform_code`` / ``qform_code``.

Grid geometry (shape + affine matching a known template grid) is computed as a
corroborating *hint* and reported, but it never upgrades an unknown space to a
resolvable one. Guessing from geometry is how a volume in scanner-native space
quietly gets labelled with MNI region names.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# NIfTI xform code meanings (nifti1.h).
XFORM_CODES = {
    0: "unknown",
    1: "scanner_anat",
    2: "aligned_anat",
    3: "talairach",
    4: "mni_152",
    5: "template_other",
}

#: Canonical template names we know how to reason about.
CANONICAL_SPACES = {
    "MNI152NLin6Asym",  # FSL's MNI152; the space FSL/Harvard-Oxford atlases live in
    "MNI152NLin2009cAsym",  # fMRIPrep's default output space
    "MNI152NLin2009aSym",
    "MNI152",  # MNI152, variant unspecified
    "MNI305",
    "Talairach",
    "fsaverage",
    "native",
    "unknown",
}

#: Free-text spellings we accept for the canonical names above.
_SPACE_ALIASES = {
    "mni": "MNI152",
    "mni152": "MNI152",
    "mni152nlin6asym": "MNI152NLin6Asym",
    "mni152nlin6sym": "MNI152NLin6Asym",
    "mni152lin": "MNI152",
    "fsl": "MNI152NLin6Asym",
    "fslmni": "MNI152NLin6Asym",
    "mni152nlin2009casym": "MNI152NLin2009cAsym",
    "mni152nlin2009aasym": "MNI152NLin2009aSym",
    "mni152nlin2009asym": "MNI152NLin2009aSym",
    "mni2009c": "MNI152NLin2009cAsym",
    "icbm152": "MNI152NLin2009cAsym",
    "mni305": "MNI305",
    "talairach": "Talairach",
    "tal": "Talairach",
    "native": "native",
    "scanner": "native",
    "subject": "native",
    "t1w": "native",
    "unknown": "unknown",
}

#: Spaces in which atlas region names can be resolved at all.
RESOLVABLE_SPACES = {
    "MNI152NLin6Asym",
    "MNI152NLin2009cAsym",
    "MNI152NLin2009aSym",
    "MNI152",
}

#: Rough worst-case disagreement, in millimetres, between MNI152 variants.
#: Nonlinear template variants differ by a few mm in some regions; a coordinate
#: looked up in one and applied in another is close but not identical.
_VARIANT_DISAGREEMENT_MM = {
    frozenset({"MNI152NLin6Asym", "MNI152NLin2009cAsym"}): 4.0,
    frozenset({"MNI152NLin6Asym", "MNI152NLin2009aSym"}): 4.0,
    frozenset({"MNI152NLin2009cAsym", "MNI152NLin2009aSym"}): 2.0,
    frozenset({"MNI152NLin6Asym", "MNI152"}): 4.0,
    frozenset({"MNI152NLin2009cAsym", "MNI152"}): 4.0,
    frozenset({"MNI152NLin2009aSym", "MNI152"}): 4.0,
}

#: Well-known template grids, used only to produce a hint string.
#: (shape, rounded voxel size) -> description
_KNOWN_GRIDS = {
    ((91, 109, 91), (2.0, 2.0, 2.0)): "FSL MNI152 2mm grid",
    ((182, 218, 182), (1.0, 1.0, 1.0)): "FSL MNI152 1mm grid",
    ((45, 54, 45), (4.0, 4.0, 4.0)): "FSL MNI152 4mm grid",
    ((193, 229, 193), (1.0, 1.0, 1.0)): "MNI152NLin2009cAsym 1mm grid",
    ((97, 115, 97), (2.0, 2.0, 2.0)): "MNI152NLin2009cAsym 2mm grid",
    ((99, 117, 95), (2.0, 2.0, 2.0)): "MNI152 2mm grid (nilearn template)",
}


def normalize_space_name(name: str | None) -> str | None:
    """Map a free-text space name onto a canonical one, or ``None`` if unrecognised."""
    if not name:
        return None
    raw = str(name).strip()
    if raw in CANONICAL_SPACES:
        return raw
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    if key in _SPACE_ALIASES:
        return _SPACE_ALIASES[key]
    # "space-MNI152NLin2009cAsym" or a URL ending in a template name.
    m = re.search(r"(mni152nlin\w+|mni305|mni152|talairach)", raw, flags=re.I)
    if m:
        return _SPACE_ALIASES.get(re.sub(r"[^a-z0-9]", "", m.group(1).lower()))
    return None


@dataclass(frozen=True)
class SpaceInfo:
    """What we believe about a volume's coordinate system, and why."""

    name: str
    source: str
    resolvable: bool
    detail: str
    grid_hint: str | None = None
    xform_codes: dict = field(default_factory=dict)

    @property
    def is_mni(self) -> bool:
        return self.name.startswith("MNI152")

    def to_dict(self) -> dict:
        return {
            "space": self.name,
            "source": self.source,
            "resolvable": self.resolvable,
            "detail": self.detail,
            "grid_hint": self.grid_hint,
            "xform_codes": self.xform_codes,
        }


def _grid_hint(img) -> str | None:
    shape = tuple(int(s) for s in img.shape[:3])
    zooms = tuple(round(float(z), 1) for z in img.header.get_zooms()[:3])
    return _KNOWN_GRIDS.get((shape, zooms))


def _sidecar_space(path: Path) -> tuple[str | None, str]:
    """Read a BIDS JSON sidecar sitting next to the image, if there is one."""
    for candidate in (
        path.with_suffix("").with_suffix(".json"),  # foo.nii.gz -> foo.json
        path.with_suffix(".json"),  # foo.nii -> foo.json
    ):
        if not candidate.exists():
            continue
        try:
            meta = json.loads(candidate.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(meta, dict):
            continue
        for key in ("Space", "SpatialReference", "TemplateFlowID", "space"):
            value = meta.get(key)
            if isinstance(value, str):
                canonical = normalize_space_name(value)
                if canonical:
                    return canonical, f"BIDS sidecar {candidate.name} ({key}={value!r})"
    return None, ""


def _filename_space(path: Path) -> tuple[str | None, str]:
    m = re.search(r"(?:^|[_/])space-([A-Za-z0-9]+)", path.name)
    if not m:
        return None, ""
    canonical = normalize_space_name(m.group(1))
    if canonical:
        return canonical, f"BIDS filename entity space-{m.group(1)}"
    return None, ""


def detect_space(img, path: str | Path | None = None, override: str | None = None) -> SpaceInfo:
    """Work out which space ``img`` is in.

    ``override`` is the user telling us directly; we take their word for it and
    say so in the provenance, because a human asserting a space is a different
    kind of claim from software inferring one.
    """
    path = Path(path) if path else None
    hint = _grid_hint(img)

    header = img.header
    try:
        sform_code = int(header["sform_code"])
        qform_code = int(header["qform_code"])
    except (KeyError, ValueError, TypeError):
        sform_code = qform_code = 0
    codes = {
        "sform_code": f"{sform_code} ({XFORM_CODES.get(sform_code, '?')})",
        "qform_code": f"{qform_code} ({XFORM_CODES.get(qform_code, '?')})",
    }

    if override:
        canonical = normalize_space_name(override)
        if canonical is None:
            known = ", ".join(sorted(CANONICAL_SPACES))
            raise ValueError(f"Unrecognised space {override!r}. Known spaces: {known}")
        return SpaceInfo(
            name=canonical,
            source="user_override",
            resolvable=canonical in RESOLVABLE_SPACES,
            detail=f"space asserted by the caller as {override!r}",
            grid_hint=hint,
            xform_codes=codes,
        )

    if path is not None:
        for finder in (_sidecar_space, _filename_space):
            canonical, why = finder(path)
            if canonical:
                return SpaceInfo(
                    name=canonical,
                    source=finder.__name__.strip("_"),
                    resolvable=canonical in RESOLVABLE_SPACES,
                    detail=why,
                    grid_hint=hint,
                    xform_codes=codes,
                )

    code = sform_code or qform_code
    which = "sform_code" if sform_code else "qform_code"
    if code == 4:
        return SpaceInfo(
            name="MNI152",
            source="nifti_header",
            resolvable=True,
            detail=(
                f"{which}=4 (mni_152). The header does not record which MNI152 variant, "
                "so cross-variant differences of a few mm are possible."
            ),
            grid_hint=hint,
            xform_codes=codes,
        )
    if code == 3:
        return SpaceInfo(
            name="Talairach",
            source="nifti_header",
            resolvable=False,
            detail=(
                f"{which}=3 (talairach). Talairach is not MNI; the bundled atlases are "
                "in MNI space, so region names are not resolved here."
            ),
            grid_hint=hint,
            xform_codes=codes,
        )
    if code in (1, 2):
        return SpaceInfo(
            name="native",
            source="nifti_header",
            resolvable=False,
            detail=(
                f"{which}={code} ({XFORM_CODES[code]}). This volume is in subject/scanner "
                "space, not a template space, so atlas region names do not apply."
            ),
            grid_hint=hint,
            xform_codes=codes,
        )
    if code == 5:
        return SpaceInfo(
            name="unknown",
            source="nifti_header",
            resolvable=False,
            detail=(
                f"{which}=5 (template_other): a template space the header does not name. "
                "Pass space= explicitly if you know which one."
            ),
            grid_hint=hint,
            xform_codes=codes,
        )

    return SpaceInfo(
        name="unknown",
        source="none",
        resolvable=False,
        detail=(
            "sform_code and qform_code are both 0 (unknown), there is no BIDS sidecar, "
            "and the filename carries no space- entity."
        ),
        grid_hint=hint,
        xform_codes=codes,
    )


def space_refusal_message(space: SpaceInfo, region_label: str, volume_name: str = "volume") -> str:
    """The message we return instead of a coordinate we cannot justify."""
    lines = [
        f"Cannot resolve {region_label!r}: {volume_name} has no recognized space in header. "
        f"Supply coords explicitly or specify space=.",
        f"Detected: space={space.name} (source: {space.source}). {space.detail}",
    ]
    if space.grid_hint:
        lines.append(
            f"Note: the voxel grid matches the {space.grid_hint}, which is suggestive but not "
            "proof. Geometry alone is not used to assign a space — pass "
            "space='MNI152NLin6Asym' (or the correct variant) to assert it yourself."
        )
    lines.append(f"Known spaces: {', '.join(sorted(RESOLVABLE_SPACES))}.")
    return " ".join(lines)


def variant_warning(volume_space: str, atlas_space: str) -> str | None:
    """Warn when both spaces are MNI152 but different variants."""
    if volume_space == atlas_space:
        return None
    pair = frozenset({volume_space, atlas_space})
    mm = _VARIANT_DISAGREEMENT_MM.get(pair)
    if mm is None:
        return None
    return (
        f"Volume is in {volume_space} but the atlas is in {atlas_space}. Both are MNI152 "
        f"variants, so coordinates are comparable but not identical — expect disagreement "
        f"up to roughly {mm:.0f}mm in some regions. No resampling between variants is performed."
    )


def affine_summary(affine: np.ndarray) -> dict:
    """A compact, human-checkable description of an affine.

    Returned instead of the 4x4 matrix in most places: R4 says small payloads, and
    a matrix of 16 floats is not something anyone reads anyway.
    """
    affine = np.asarray(affine, dtype=float)
    import nibabel as nib

    orientation = "".join(nib.aff2axcodes(affine))
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return {
        # Axis codes say which anatomical direction each voxel axis increases towards,
        # e.g. "LAS" = i grows leftward, j anterior, k superior. Unambiguous, unlike
        # the words "radiological" and "neurological".
        "orientation": orientation,
        "voxel_size_mm": [round(float(z), 4) for z in zooms],
        "origin_world_mm": [round(float(v), 3) for v in affine[:3, 3]],
        "determinant": round(float(np.linalg.det(affine[:3, :3])), 4),
    }
