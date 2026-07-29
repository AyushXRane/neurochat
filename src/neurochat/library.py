"""A library of scans on disk, and one region measured across all of them.

The engine always handled many volumes; the *interface* assumed you were looking at
one thing. Loading a cohort meant typing forty absolute paths and then unchecking
thirty-nine layers. This module fixes that without changing what the tools are.

Two ideas:

* **Discovery is cheap and read-only.** Scanning a folder reads NIfTI *headers* only —
  never the voxel data — so pointing this at a directory of hundreds of scans costs
  almost nothing and loads nothing into the session. Each entry carries the same space
  provenance a real load would report, so you can see before you commit which scans
  are going to refuse region names and why.
* **The cohort table is a client of the ten tools, not an eleventh tool.** v1 is exactly
  ten tools and that constraint is worth keeping. Measuring one region across a library
  is a deterministic action (R5) — it never touches the model — and it emits a single
  loop into the session script rather than N copies of the same three lines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import kernel
from .errors import NoAtlasLoadedError, SpaceUnknownError
from .session import Session
from .spaces import detect_space, normalize_space_name, variant_warning

NIFTI_SUFFIXES = (".nii", ".nii.gz")

#: Cap on how many files one scan will return. A directory with more than this is
#: almost certainly the wrong directory, and the response has a payload budget.
MAX_ENTRIES = 500

#: Cap on the cohort table, for the same reason.
MAX_TABLE_ROWS = 200


@dataclass(frozen=True)
class LibraryEntry:
    """One scan found on disk. Nothing is loaded; this is all header metadata."""

    path: str
    name: str
    shape: tuple[int, ...]
    voxel_size_mm: tuple[float, ...]
    size_bytes: int
    space: str
    space_resolvable: bool
    space_source: str
    suggested_space: str | None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "shape": list(self.shape),
            "voxel_size_mm": list(self.voxel_size_mm),
            "size_bytes": self.size_bytes,
            "space": self.space,
            "space_resolvable": self.space_resolvable,
            "space_source": self.space_source,
            "suggested_space": self.suggested_space,
        }


def _entry_name(path: Path, seen: set[str]) -> str:
    stem = path.name
    for suffix in NIFTI_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidate, counter = stem, 2
    while candidate in seen:
        candidate = f"{stem}_{counter}"
        counter += 1
    seen.add(candidate)
    return candidate


def _entries_from_paths(paths: list[Path]) -> tuple[list[LibraryEntry], list[str]]:
    """Read headers for a list of files. Returns (entries, unreadable descriptions).

    A file that will not open is counted and named rather than silently dropped — a
    scan that quietly finds 39 of your 40 files is worse than one that finds 39 and
    tells you which one it choked on.
    """
    import nibabel as nib

    entries: list[LibraryEntry] = []
    unreadable: list[str] = []
    seen: set[str] = set()
    for path in paths:
        try:
            img = nib.load(str(path))  # lazy: header only, no voxel data
            info = detect_space(img, path)
            entries.append(
                LibraryEntry(
                    path=str(path.resolve()),
                    name=_entry_name(path, seen),
                    shape=tuple(int(s) for s in img.shape),
                    voxel_size_mm=tuple(round(float(z), 3) for z in img.header.get_zooms()[:3]),
                    size_bytes=path.stat().st_size,
                    space=info.name,
                    space_resolvable=info.resolvable,
                    space_source=info.source,
                    suggested_space=info.suggested_space,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a bad file is data, not a crash
            unreadable.append(f"{path.name} ({type(exc).__name__})")
    return entries, unreadable


def sample_cohort(n_subjects: int = 12) -> dict:
    """A real multi-subject cohort, so the library has something honest to browse.

    OASIS-1 grey-matter density maps: real structural MRI from real people, already
    normalised, fetched through nilearn and cached. Structural only — there is no
    freely-fetchable PET cohort, which is a documented gap rather than an oversight.

    Worth knowing: these carry ``sform_code=2``, so every one of them lands on the
    "space is aligned to something unspecified" path. That is not a flaw in the data,
    it is what a large amount of real normalised data looks like, and it is exactly
    why the space assertion is one click rather than a wall.
    """
    from nilearn import datasets

    bunch = datasets.fetch_oasis_vbm(n_subjects=max(1, min(int(n_subjects), 100)))
    paths = [Path(p) for p in bunch["gray_matter_maps"]]
    entries, unreadable = _entries_from_paths(paths)

    notes = [
        f"OASIS-1: {len(entries)} real subjects, grey-matter density maps from structural "
        "MRI. Values are tissue density in arbitrary units, not a quantitative measure.",
        "Structural MRI only — no PET. There is no freely-fetchable PET cohort; see "
        "LIMITATIONS.md.",
    ]
    if unreadable:
        notes.append(f"{len(unreadable)} file(s) could not be read: {', '.join(unreadable[:3])}")
    blocked = [e for e in entries if not e.space_resolvable and e.suggested_space]
    if blocked:
        notes.append(
            f"All {len(blocked)} carry sform_code=2, so region names are off until you assert "
            f"the space — one click, recorded as your assertion."
        )
    return {
        "root": str(paths[0].parent.parent) if paths else "",
        "recursive": False,
        "n_found": len(entries),
        "entries": [e.to_dict() for e in entries],
        "notes": notes,
    }


def scan_directory(directory: str | Path, recursive: bool = True) -> dict:
    """Find NIfTI files under ``directory``, reading headers only.

    Files that cannot be opened are counted and named rather than silently dropped —
    a scan that quietly finds 39 of your 40 files is worse than one that finds 39 and
    tells you which one it choked on.
    """
    root = Path(directory).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"No such directory: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    paths: list[Path] = []
    walker = root.rglob("*") if recursive else root.glob("*")
    for candidate in walker:
        if candidate.is_file() and candidate.name.endswith(NIFTI_SUFFIXES):
            paths.append(candidate)
    paths.sort(key=lambda p: str(p).lower())

    truncated = len(paths) > MAX_ENTRIES
    if truncated:
        paths = paths[:MAX_ENTRIES]

    entries, unreadable = _entries_from_paths(paths)

    notes: list[str] = []
    if truncated:
        notes.append(
            f"More than {MAX_ENTRIES} NIfTI files here; showing the first {MAX_ENTRIES}. "
            "Point at a narrower directory."
        )
    if unreadable:
        shown = ", ".join(unreadable[:5])
        more = f" (+{len(unreadable) - 5} more)" if len(unreadable) > 5 else ""
        notes.append(f"{len(unreadable)} file(s) could not be read: {shown}{more}")

    blocked = [e for e in entries if not e.space_resolvable]
    if blocked:
        fixable = [e for e in blocked if e.suggested_space]
        note = (
            f"{len(blocked)} of {len(entries)} scans have no template space in their header, "
            "so region names are off for them."
        )
        if fixable:
            spaces = sorted({e.suggested_space for e in fixable})
            note += (
                f" {len(fixable)} sit on geometry that looks like {', '.join(spaces)} — if they "
                "really are normalised, you can assert that when you select or tabulate them."
            )
        notes.append(note)

    return {
        "root": str(root.resolve()),
        "recursive": recursive,
        "n_found": len(entries),
        "entries": [e.to_dict() for e in entries],
        "notes": notes,
    }


def region_across_library(
    session: Session,
    paths: list[str],
    region_label: str,
    exclude_zeros: bool = False,
    assume_space: str | None = None,
) -> dict:
    """Measure one atlas region in every scan in ``paths``. No model call involved.

    Each scan is measured on its own grid — the label map is resampled to each one
    individually — which is why voxel counts differ across resolutions and why that
    is correct rather than a bug. Scans whose space cannot be established are skipped
    and listed, never quietly averaged in.
    """
    import nibabel as nib

    if session.atlas is None:
        raise NoAtlasLoadedError(
            "Load an atlas before tabulating a region across the library — region names "
            "are resolved against the atlas, never guessed."
        )
    table_atlas = session.atlas
    region = table_atlas.resolve(region_label)

    if assume_space is not None:
        canonical = normalize_space_name(assume_space)
        if canonical is None:
            raise ValueError(f"Unrecognised space {assume_space!r}.")
        assume_space = canonical

    truncated = len(paths) > MAX_TABLE_ROWS
    paths = paths[:MAX_TABLE_ROWS]

    rows: list[dict] = []
    skipped: list[dict] = []
    used_paths: list[str] = []
    warnings: set[str] = set()

    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            skipped.append({"path": str(path), "reason": "file not found"})
            continue
        try:
            img = nib.load(str(path))
            info = detect_space(img, path, override=assume_space)
            if not info.resolvable:
                skipped.append(
                    {
                        "path": str(path),
                        "reason": f"space is {info.name} ({info.source}); region names do not apply",
                        "suggested_space": info.suggested_space,
                    }
                )
                continue
            note = variant_warning(info.name, table_atlas.space)
            if note:
                warnings.add(note)

            mask = kernel.region_mask(table_atlas.maps_path, region.index, target_img=img)
            if not mask.any():
                skipped.append({"path": str(path), "reason": "region does not overlap this volume"})
                continue
            stats = kernel.summarize_roi(img, mask, exclude_zeros=exclude_zeros)
            rows.append(
                {
                    "scan": path.name,
                    "path": str(path.resolve()),
                    "space": info.name,
                    "n_voxels": stats["n_voxels_in_mask"],
                    "n_used": stats["n_voxels_used"],
                    "mean": stats["mean"],
                    "sd": stats["sd"],
                    "median": stats["median"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "excluded_nan": stats["exclusions"]["nan"],
                }
            )
            used_paths.append(str(path.resolve()))
        except Exception as exc:  # noqa: BLE001 - one bad scan must not kill the table
            skipped.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    if not rows:
        # Same principle as a single volume: a refusal must carry its own fix. A whole
        # cohort written with sform_code=2 is the common case, not the exotic one.
        suggestions = {s.get("suggested_space") for s in skipped if s.get("suggested_space")}
        message = (
            f"No scan in the library could be measured for {region.label!r}. "
            + (skipped[0]["reason"] if skipped else "The library is empty.")
        )
        if len(suggestions) == 1:
            only = suggestions.pop()
            message += (
                f" All of them sit on geometry that looks like {only}. If these really are "
                f"normalised, re-run with assume_space='{only}' — recorded as your assertion."
            )
        elif suggestions:
            message += (
                f" They suggest more than one space ({', '.join(sorted(suggestions))}), so no "
                "single assertion covers the cohort — they may not belong together."
            )
        raise SpaceUnknownError(
            message, skipped=skipped[:5], suggested_space=sorted(suggestions) or None
        )

    step_key = f"step_{len(session.script.steps) + 1}_region_table"
    code = (
        f"COHORT = {used_paths!r}\n"
        f"RESULTS[{step_key!r}] = {{}}\n"
        f"for _path in COHORT:\n"
        f"    _img = nib.load(_path)\n"
        f"    _mask = region_mask(ATLAS_PATH, {region.index}, target_img=_img)  # {region.label}\n"
        f"    RESULTS[{step_key!r}][_path] = summarize_roi(\n"
        f"        _img, _mask, exclude_zeros={bool(exclude_zeros)}\n"
        f"    )"
    )
    session.script.add(
        "library.region_table",
        code,
        comment=(
            f"{region.label} measured across {len(rows)} scan(s); the label map is resampled "
            f"to each scan's own grid, so voxel counts differ with resolution"
        ),
    )

    columns = ["scan", "n_voxels", "mean", "sd", "median", "min", "max", "excluded_nan"]
    result_table = session.add_table("region_table", columns, rows)

    notes = sorted(warnings)
    if skipped:
        notes.append(
            f"{len(skipped)} scan(s) were skipped and are listed in 'skipped' — they are not "
            "included in any number above."
        )
    if truncated:
        notes.append(f"Table capped at {MAX_TABLE_ROWS} scans.")
    if any(row["excluded_nan"] for row in rows):
        total = sum(row["excluded_nan"] for row in rows)
        notes.append(
            f"{total} NaN voxels were excluded across the cohort; per-scan counts are in the "
            "excluded_nan column."
        )
    notes.append(
        "Descriptive only: this is a table of per-scan means, not a group comparison. There "
        "is no test here and none is implied."
    )

    payload = {
        "ok": True,
        "region": region.label,
        "region_index": region.index,
        "atlas_id": table_atlas.atlas_id,
        "atlas_space": table_atlas.space,
        "columns": columns,
        "rows": rows,
        "skipped": skipped,
        "n_measured": len(rows),
        "n_skipped": len(skipped),
        "table_id": result_table.table_id,
        "result_key": step_key,
        "code": code,
        "notes": notes,
    }
    payload["tool_trace"] = session.record_trace(
        "library.region_table",
        {"region_label": region_label, "n_paths": len(paths), "assume_space": assume_space},
        payload,
    )
    return payload


def rows_to_csv(rows: list[dict], columns: list[str]) -> str:
    """The table as CSV, because the next thing anyone does with it is paste it somewhere."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return buffer.getvalue()


def default_library_root() -> str:
    """A sensible starting directory for the folder box."""
    return os.environ.get("NEUROCHAT_LIBRARY", str(Path.home()))
