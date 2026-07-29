"""Atlas grounding: the lookup table that stands between an LLM and a coordinate.

Rule R1: the model never emits raw coordinates. Ask a language model where left
entorhinal cortex is and it will give you a confident, wrong number. So every
anatomical name in this system is resolved here, against label indices actually
present in an actually-loaded atlas volume, with centroids actually computed from
the mask.

Two design choices worth stating outright:

* **Non-exact matches never resolve.** ``"left hippocampos"`` is a typo one edit
  away from a real label, and that is precisely why accepting it is dangerous.
  We normalise aggressively so that natural phrasings ("Hippocampus_L", "hippocampus
  (left)", "L hippocampus") hit *exactly*, and anything still unmatched comes back
  as a did-you-mean with the three closest real labels.

* **The centroid may not be inside the region.** Hippocampus is a curved structure;
  its centre of mass can land in the ventricle next door. So each region carries
  both a ``centroid`` and a ``representative`` point — the in-region voxel closest
  to the centroid — and records whether the two coincide.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .errors import RegionNotFoundError

# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

_LEFT_TOKENS = {"l", "lh", "left", "lft"}
_RIGHT_TOKENS = {"r", "rh", "right", "rgt"}

#: Synonyms applied to the side-stripped remainder of both labels and queries, so
#: that the two normalise into the same string. Deliberately small and anatomical;
#: this is not a place for clever guessing.
_SYNONYMS = {
    "caudate nucleus": "caudate",
    "nucleus caudatus": "caudate",
    "nucleus accumbens": "accumbens",
    "accumbens area": "accumbens",
    "globus pallidus": "pallidum",
    "pallidus": "pallidum",
    "brainstem": "brain stem",
    "amygdaloid complex": "amygdala",
    "putamen nucleus": "putamen",
    "lateral ventricles": "lateral ventricle",
    "white matter": "cerebral white matter",
    "cerebellum cortex": "cerebellar cortex",
}

_STOPWORDS = {"the", "of", "area", "region"}


def _strip_side(tokens: list[str]) -> tuple[str | None, list[str]]:
    side = None
    kept = []
    for tok in tokens:
        if tok in _LEFT_TOKENS and side is None:
            side = "left"
        elif tok in _RIGHT_TOKENS and side is None:
            side = "right"
        else:
            kept.append(tok)
    return side, kept


def normalize_label(text: str) -> str:
    """Fold a label or a user phrase into a comparable canonical string.

    ``"Hippocampus_L"``, ``"left hippocampus"``, ``"L. Hippocampus"`` and
    ``"hippocampus (left)"`` all become ``"left hippocampus"``.
    """
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"[_\-.,/()\[\]{}:;'\"]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split(" ") if t and t not in _STOPWORDS]
    side, tokens = _strip_side(tokens)
    rest = " ".join(tokens)
    rest = _SYNONYMS.get(rest, rest)
    return f"{side} {rest}".strip() if side else rest


def similarity(a: str, b: str) -> float:
    """Blend sequence similarity with token overlap, on normalised strings."""
    na, nb = normalize_label(a), normalize_label(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return 0.6 * seq + 0.4 * jaccard


# ---------------------------------------------------------------------------
# Regions and tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One labelled structure, with geometry measured from the atlas volume."""

    index: int
    label: str
    n_voxels: int
    volume_mm3: float
    centroid: tuple[float, float, float]
    representative: tuple[float, float, float]
    centroid_inside: bool
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]

    @property
    def normalized(self) -> str:
        return normalize_label(self.label)

    def to_dict(self, verbose: bool = False) -> dict:
        out = {
            "index": self.index,
            "label": self.label,
            "n_voxels": self.n_voxels,
            "volume_mm3": round(self.volume_mm3, 1),
            "centroid": [round(c, 2) for c in self.centroid],
        }
        if verbose:
            out.update(
                {
                    "representative": [round(c, 2) for c in self.representative],
                    "centroid_inside_region": self.centroid_inside,
                    "bbox_min": [round(c, 1) for c in self.bbox_min],
                    "bbox_max": [round(c, 1) for c in self.bbox_max],
                }
            )
        return out


@dataclass
class AtlasTable:
    """The R1 lookup table for one loaded atlas."""

    atlas_id: str
    description: str
    space: str
    resolution_mm: tuple[float, float, float]
    maps_path: str
    regions: list[Region]
    source: str = ""
    citation: str = ""

    def __post_init__(self):
        self._by_normalized: dict[str, list[Region]] = {}
        for region in self.regions:
            self._by_normalized.setdefault(region.normalized, []).append(region)

    # -- lookups ---------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        return [r.label for r in self.regions]

    def by_index(self, index: int) -> Region | None:
        for region in self.regions:
            if region.index == index:
                return region
        return None

    def suggest(self, query: str, n: int = 3) -> list[str]:
        """The n closest real labels. Used for did-you-mean, never for auto-select."""
        scored = sorted(
            ((similarity(query, r.label), r.label) for r in self.regions),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [label for score, label in scored[:n] if score > 0.2]

    def resolve(self, query: str) -> Region:
        """Resolve a name to a region, or raise with suggestions.

        Only exact matches (after normalisation) succeed. A near-miss is reported,
        not silently accepted — that is the difference between a typo caught and a
        wrong number in a paper.
        """
        if not query or not str(query).strip():
            raise RegionNotFoundError(
                "No region name given.", suggestions=self.labels[:3], atlas=self.atlas_id
            )
        key = normalize_label(query)
        hits = self._by_normalized.get(key, [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RegionNotFoundError(
                f"{query!r} matches {len(hits)} labels in atlas {self.atlas_id!r}: "
                f"{', '.join(h.label for h in hits)}. Use the exact label or its index.",
                suggestions=[h.label for h in hits],
                atlas=self.atlas_id,
            )
        suggestions = self.suggest(query)
        hint = (
            f" Did you mean: {', '.join(repr(s) for s in suggestions)}?"
            if suggestions
            else " No similar labels found."
        )
        raise RegionNotFoundError(
            f"{query!r} is not a label in atlas {self.atlas_id!r}.{hint} "
            f"Region names are resolved against the atlas label list, never guessed; "
            f"call list_regions() to see all {len(self.regions)} labels.",
            suggestions=suggestions,
            atlas=self.atlas_id,
            query=query,
        )

    def search(self, query: str | None = None, limit: int = 200) -> list[Region]:
        """Substring/token filter over labels, ordered by similarity when querying."""
        if not query:
            return self.regions[:limit]
        norm = normalize_label(query)
        tokens = set(norm.split())
        scored = []
        for region in self.regions:
            rn = region.normalized
            if norm and norm in rn:
                score = 1.0
            elif tokens and tokens <= set(rn.split()):
                score = 0.9
            else:
                score = similarity(query, region.label)
                if score < 0.45:
                    continue
            scored.append((score, region))
        scored.sort(key=lambda pair: (-pair[0], pair[1].label))
        return [region for _, region in scored[:limit]]

    # -- masks -----------------------------------------------------------

    def load_maps(self):
        import nibabel as nib

        return nib.load(self.maps_path)

    def mask_for(self, region: Region) -> np.ndarray:
        img = self.load_maps()
        return np.asanyarray(img.dataobj) == region.index

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "atlas_id": self.atlas_id,
            "description": self.description,
            "space": self.space,
            "resolution_mm": list(self.resolution_mm),
            "maps_path": self.maps_path,
            "source": self.source,
            "citation": self.citation,
            "regions": [asdict(r) for r in self.regions],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AtlasTable":
        regions = [
            Region(
                index=int(r["index"]),
                label=r["label"],
                n_voxels=int(r["n_voxels"]),
                volume_mm3=float(r["volume_mm3"]),
                centroid=tuple(r["centroid"]),
                representative=tuple(r["representative"]),
                centroid_inside=bool(r["centroid_inside"]),
                bbox_min=tuple(r["bbox_min"]),
                bbox_max=tuple(r["bbox_max"]),
            )
            for r in payload["regions"]
        ]
        return cls(
            atlas_id=payload["atlas_id"],
            description=payload["description"],
            space=payload["space"],
            resolution_mm=tuple(payload["resolution_mm"]),
            maps_path=payload["maps_path"],
            regions=regions,
            source=payload.get("source", ""),
            citation=payload.get("citation", ""),
        )


# ---------------------------------------------------------------------------
# Building a table from a labelled volume
# ---------------------------------------------------------------------------


def build_table(
    maps_path: str | Path,
    labels: list[str],
    atlas_id: str,
    space: str,
    description: str = "",
    source: str = "",
    citation: str = "",
    indices: list[int] | None = None,
    skip_labels: tuple[str, ...] = ("background", ""),
) -> AtlasTable:
    """Measure every region in a deterministic (integer-labelled) atlas volume."""
    import nibabel as nib
    from nibabel.affines import apply_affine

    img = nib.load(str(maps_path))
    if img.ndim > 3 and img.shape[3] > 1:
        raise ValueError(
            f"{maps_path} is 4D (shape {img.shape}): this looks like a probabilistic "
            "atlas. v1 supports deterministic (maxprob / integer-labelled) atlases only, "
            "because a probabilistic map has no single unambiguous region membership."
        )

    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.rint(data).astype(np.int32)
    affine = img.affine
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    voxel_volume = float(np.abs(np.linalg.det(affine[:3, :3])))

    if indices is None:
        indices = list(range(len(labels)))

    regions: list[Region] = []
    for index, label in zip(indices, labels):
        if str(label).strip().lower() in skip_labels:
            continue
        voxels = np.argwhere(data == index)
        if voxels.size == 0:
            # A label with no voxels at this threshold/resolution. Skipping it is
            # correct: resolving a name to an empty mask would be worse than failing.
            continue
        world = apply_affine(affine, voxels.astype(float))
        centroid = world.mean(axis=0)
        distances = np.linalg.norm(world - centroid, axis=1)
        representative = world[int(np.argmin(distances))]
        centroid_vox = np.rint(np.linalg.solve(affine, np.append(centroid, 1.0))[:3]).astype(int)
        inside = bool(
            np.all(centroid_vox >= 0)
            and np.all(centroid_vox < np.array(data.shape))
            and data[tuple(centroid_vox)] == index
        )
        regions.append(
            Region(
                index=int(index),
                label=str(label),
                n_voxels=int(voxels.shape[0]),
                volume_mm3=float(voxels.shape[0] * voxel_volume),
                centroid=tuple(float(c) for c in centroid),
                representative=tuple(float(c) for c in representative),
                centroid_inside=inside,
                bbox_min=tuple(float(c) for c in world.min(axis=0)),
                bbox_max=tuple(float(c) for c in world.max(axis=0)),
            )
        )

    return AtlasTable(
        atlas_id=atlas_id,
        description=description,
        space=space,
        resolution_mm=zooms,
        maps_path=str(Path(maps_path).resolve()),
        regions=regions,
        source=source,
        citation=citation,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasSpec:
    atlas_id: str
    description: str
    space: str
    bundled: bool
    citation: str
    loader: str


#: Atlases v1 knows how to load. ``bundled`` ones ship with the package and need
#: no network; the rest are fetched once through nilearn and cached under
#: ~/nilearn_data. Everything here is in an MNI152 variant.
ATLAS_REGISTRY: dict[str, AtlasSpec] = {
    "demo-16": AtlasSpec(
        atlas_id="demo-16",
        description=(
            "Bundled 16-region synthetic demo atlas in MNI152NLin6Asym. Geometric "
            "shapes with anatomically-inspired names, generated by "
            "scripts/make_sample_data.py. For smoke-testing offline only — the "
            "regions are NOT anatomy and must never be used for real analysis."
        ),
        space="MNI152NLin6Asym",
        bundled=True,
        citation="Synthetic. Generated by neurochat; not derived from any subject data.",
        loader="demo",
    ),
    "harvard-oxford-sub": AtlasSpec(
        atlas_id="harvard-oxford-sub",
        description="Harvard-Oxford subcortical structural atlas, maxprob thr25 2mm (21 regions).",
        space="MNI152NLin6Asym",
        bundled=False,
        citation=(
            "Harvard-Oxford atlas, Harvard Center for Morphometric Analysis, distributed "
            "with FSL. Makris et al. 2006; Frazier et al. 2005; Desikan et al. 2006; "
            "Goldstein et al. 2007."
        ),
        loader="fsl:sub-maxprob-thr25-2mm",
    ),
    "harvard-oxford-cort": AtlasSpec(
        atlas_id="harvard-oxford-cort",
        description=(
            "Harvard-Oxford cortical structural atlas, maxprob thr25 2mm, "
            "left/right split (96 regions)."
        ),
        space="MNI152NLin6Asym",
        bundled=False,
        citation=(
            "Harvard-Oxford atlas, Harvard Center for Morphometric Analysis, distributed "
            "with FSL. Makris et al. 2006; Frazier et al. 2005; Desikan et al. 2006; "
            "Goldstein et al. 2007."
        ),
        loader="fsl:cort-maxprob-thr25-2mm:split",
    ),
    "aal": AtlasSpec(
        atlas_id="aal",
        description="AAL (Automated Anatomical Labeling) SPM12 atlas, 116 regions.",
        space="MNI152",
        bundled=False,
        citation="Tzourio-Mazoyer et al. 2002, NeuroImage 15:273-289.",
        loader="aal",
    ),
    "schaefer-100": AtlasSpec(
        atlas_id="schaefer-100",
        description="Schaefer 2018 cortical parcellation, 100 parcels, 7 networks, 2mm.",
        space="MNI152NLin6Asym",
        bundled=False,
        citation="Schaefer et al. 2018, Cerebral Cortex 28:3095-3114.",
        loader="schaefer:100:7",
    ),
}


def atlas_fetch_code(atlas_id: str, maps_path: str) -> str:
    """The line an exported script uses to get this atlas back.

    Registry atlases emit their nilearn fetcher call rather than a hard-coded path,
    so the script survives being run on another machine. The bundled demo atlas has
    no fetcher, so it emits its literal path with a comment saying where it lives.
    """
    spec = ATLAS_REGISTRY.get(atlas_id)
    if spec is None:
        return f"ATLAS_PATH = {maps_path!r}"
    loader = spec.loader
    if loader == "demo":
        return (
            "# Bundled with neurochat; synthetic geometry, not anatomy.\n"
            f"ATLAS_PATH = {maps_path!r}"
        )
    if loader.startswith("fsl:"):
        parts = loader.split(":")
        split = len(parts) > 2 and parts[2] == "split"
        return (
            "ATLAS_PATH = datasets.fetch_atlas_harvard_oxford(\n"
            f"    {parts[1]!r}, symmetric_split={split}\n"
            ')["filename"]'
        )
    if loader == "aal":
        return 'ATLAS_PATH = datasets.fetch_atlas_aal()["maps"]'
    if loader.startswith("schaefer:"):
        _, n_rois, networks = loader.split(":")
        return (
            "ATLAS_PATH = datasets.fetch_atlas_schaefer_2018(\n"
            f"    n_rois={int(n_rois)}, yeo_networks={int(networks)}, resolution_mm=2\n"
            ')["maps"]'
        )
    return f"ATLAS_PATH = {maps_path!r}"


def cache_dir() -> Path:
    root = os.environ.get("NEUROCHAT_CACHE") or (Path.home() / ".cache" / "neurochat")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(atlas_id: str, maps_path: str) -> Path:
    stat = Path(maps_path).stat()
    key = hashlib.sha256(
        f"{atlas_id}|{maps_path}|{stat.st_size}|{int(stat.st_mtime)}".encode()
    ).hexdigest()[:16]
    return cache_dir() / f"atlas-{atlas_id}-{key}.json"


def bundled_atlas_path(name: str) -> Path:
    """Locate sample data whether running from a checkout or an installed wheel."""
    candidates = [
        Path(__file__).parent / "_sample_data" / name,
        Path(__file__).parent.parent.parent / "sample_data" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Bundled sample file {name!r} not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Run `python scripts/make_sample_data.py` to regenerate sample data."
    )


def _load_via_nilearn(spec: AtlasSpec) -> tuple[str, list[str], list[int]]:
    """Return (maps_path, labels, indices) for a registry atlas, fetching if needed."""
    from nilearn import datasets

    loader = spec.loader
    if loader == "demo":
        maps = bundled_atlas_path("demo16_atlas.nii.gz")
        labels_file = bundled_atlas_path("demo16_labels.json")
        payload = json.loads(Path(labels_file).read_text())
        return str(maps), payload["labels"], payload["indices"]

    if loader.startswith("fsl:"):
        parts = loader.split(":")
        name = parts[1]
        split = len(parts) > 2 and parts[2] == "split"
        bunch = datasets.fetch_atlas_harvard_oxford(name, symmetric_split=split)
        maps_path = bunch.get("filename")
        if not maps_path or not Path(str(maps_path)).exists():
            maps_path = _materialise(bunch["maps"], spec.atlas_id)
        return str(maps_path), list(bunch["labels"]), list(range(len(bunch["labels"])))

    if loader == "aal":
        bunch = datasets.fetch_atlas_aal()
        return (
            str(bunch["maps"]),
            list(bunch["labels"]),
            [int(i) for i in bunch["indices"]],
        )

    if loader.startswith("schaefer:"):
        _, n_rois, networks = loader.split(":")
        bunch = datasets.fetch_atlas_schaefer_2018(
            n_rois=int(n_rois), yeo_networks=int(networks), resolution_mm=2
        )
        labels = [
            lab.decode() if isinstance(lab, bytes) else str(lab) for lab in bunch["labels"]
        ]
        # Schaefer labels exclude background and are 1-indexed in the volume.
        return str(bunch["maps"]), labels, list(range(1, len(labels) + 1))

    raise ValueError(f"Unknown atlas loader {loader!r}")


def _materialise(img, atlas_id: str) -> str:
    """Write an in-memory image to the cache so the emitted script can point at a file."""
    path = cache_dir() / f"{atlas_id}-maps.nii.gz"
    if not path.exists():
        img.to_filename(str(path))
    return str(path)


def load_atlas_table(atlas_name: str, use_cache: bool = True) -> AtlasTable:
    """Load an atlas by registry name and return its grounded lookup table."""
    key = atlas_name.strip().lower()
    aliases = {
        "harvard-oxford": "harvard-oxford-sub",
        "harvardoxford": "harvard-oxford-sub",
        "ho-sub": "harvard-oxford-sub",
        "ho-cort": "harvard-oxford-cort",
        "demo": "demo-16",
        "schaefer": "schaefer-100",
    }
    key = aliases.get(key, key)
    if key not in ATLAS_REGISTRY:
        available = ", ".join(sorted(ATLAS_REGISTRY))
        close = difflib.get_close_matches(key, list(ATLAS_REGISTRY) + list(aliases), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"Unknown atlas {atlas_name!r}.{hint} Available: {available}")

    spec = ATLAS_REGISTRY[key]
    maps_path, labels, indices = _load_via_nilearn(spec)

    cache_file = _cache_path(spec.atlas_id, maps_path)
    if use_cache and cache_file.exists():
        try:
            return AtlasTable.from_dict(json.loads(cache_file.read_text()))
        except (json.JSONDecodeError, KeyError, OSError):
            cache_file.unlink(missing_ok=True)

    table = build_table(
        maps_path=maps_path,
        labels=labels,
        indices=indices,
        atlas_id=spec.atlas_id,
        space=spec.space,
        description=spec.description,
        source="bundled" if spec.bundled else "nilearn fetcher (cached in ~/nilearn_data)",
        citation=spec.citation,
    )
    if use_cache:
        try:
            cache_file.write_text(json.dumps(table.to_dict()))
        except OSError:
            pass
    return table
