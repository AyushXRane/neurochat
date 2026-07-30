"""The v1 tool surface: exactly ten tools.

Framework-free on purpose. The MCP server, the web backend and the tests all call
these same functions, so there is one behaviour to reason about and one place where
the rules live:

* **R1** — anatomical names resolve through :mod:`neurochat.atlas`, never through
  model knowledge; a volume with no recognised space refuses region names outright.
* **R2** — every successful call appends runnable code to the session script.
* **R3** — no arbitrary execution: out-of-scope requests emit *commented* suggestions.
* **R4** — responses carry paths, scalars, small tables and downscaled PNGs. The
  50KB budget is enforced, not assumed.
* **R5** — none of this needs an LLM; the UI calls the same functions directly.

Every tool returns a plain dict. Failures come back as ``{"ok": False, ...}`` with a
message written for whoever has to act on it — including suggestions when a region
name misses — because an exception string swallowed by a transport helps nobody.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import kernel
from .atlas import atlas_fetch_code, load_atlas_table
from .errors import (
    NeurochatError,
    NoAtlasLoadedError,
    PayloadTooLargeError,
    SpaceMismatchError,
    SpaceUnknownError,
)
from .session import Session
from .spaces import RESOLVABLE_SPACES, normalize_space_name, space_refusal_message, variant_warning
from .viewer import LayerState, downscale_png, render_server_side

#: R4. Asserted in tests; a tool that would exceed it truncates or errors, never streams.
MAX_PAYLOAD_BYTES = 50_000

#: Colormaps Niivue and nilearn both understand, so the viewer and the exported
#: script agree about what the picture looked like.
KNOWN_COLORMAPS = (
    "gray", "hot", "cool", "warm", "winter", "bone", "copper", "inferno", "magma",
    "plasma", "viridis", "jet", "red", "green", "blue", "actc", "electric_blue",
)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _payload_size(payload: dict) -> int:
    return len(json.dumps(payload, default=str).encode("utf-8"))


def _finalize(session: Session, tool: str, args: dict, payload: dict, code: str = "") -> dict:
    """Attach the trace fragment, enforce the payload budget, hand back the result."""
    payload = {"ok": True, **payload}
    if code:
        payload["code"] = code
    payload["tool_trace"] = session.record_trace(tool, args, payload)
    size = _payload_size(payload)
    if size > MAX_PAYLOAD_BYTES:
        raise PayloadTooLargeError(
            f"{tool} produced a {size} byte response, over the {MAX_PAYLOAD_BYTES} byte "
            "budget. This is a bug in the tool, not in your request — please report it.",
            tool=tool,
            size=size,
        )
    payload["payload_bytes"] = size
    return payload


def _fail(session: Session, tool: str, args: dict, error: Exception) -> dict:
    if isinstance(error, NeurochatError):
        body = error.to_dict()
    else:
        body = {"error": type(error).__name__, "message": str(error)}
    body["ok"] = False
    body["tool_trace"] = session.record_trace(tool, args, body, error=body["error"])
    return body


def tool(name: str):
    """Wrap a tool so every failure comes back as a structured, readable dict."""

    def decorate(function):
        def wrapper(session: Session, **kwargs):
            try:
                return function(session, **kwargs)
            except NeurochatError as exc:
                return _fail(session, name, kwargs, exc)
            except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
                return _fail(session, name, kwargs, exc)

        wrapper.__name__ = function.__name__
        wrapper.__doc__ = function.__doc__
        wrapper.tool_name = name
        return wrapper

    return decorate


def _require_atlas(session: Session):
    if session.atlas is None:
        raise NoAtlasLoadedError(
            "No atlas loaded, so region names cannot be resolved — and they are never "
            "guessed. Call load_atlas('harvard-oxford-sub') first, or pass explicit coords."
        )
    return session.atlas


def _check_space_for_regions(session: Session, volume, region_label: str):
    """Guard R1: refuse anatomical names on a volume whose space we cannot establish."""
    if not volume.space.resolvable:
        raise SpaceUnknownError(
            space_refusal_message(volume.space, region_label, volume.name),
            volume=volume.name,
            detected_space=volume.space.name,
        )
    atlas = _require_atlas(session)
    if atlas.space not in RESOLVABLE_SPACES:
        raise SpaceMismatchError(
            f"Atlas {atlas.atlas_id!r} is in {atlas.space}, which is not an MNI152 variant."
        )
    return variant_warning(volume.space.name, atlas.space)


# ---------------------------------------------------------------------------
# 1. load_volume
# ---------------------------------------------------------------------------


@tool("load_volume")
def load_volume(session: Session, path: str, name: str | None = None, space: str | None = None) -> dict:
    """Load a NIfTI volume and describe it. Nothing is preprocessed, ever.

    ``space`` lets you assert the template space when the header does not record one.
    That assertion is yours, and it is recorded as yours in the provenance.
    """
    volume = session.add_volume(path, name=name, space=space)

    layer = LayerState(name=volume.name, path=volume.path, colormap="gray")
    session.viewer.add_layer(layer, on_top=len(session.viewer.state.layers) > 0)

    code = (
        f"VOLUMES[{volume.name!r}] = nib.load({volume.path!r})\n"
        f"RESULTS[{f'step_{len(session.script.steps) + 1}_load_volume'!r}] = "
        f"summarize_volume(VOLUMES[{volume.name!r}])"
    )
    session.script.add(
        "load_volume",
        code,
        comment=(
            f"{volume.name}: {'x'.join(str(s) for s in volume.shape)} @ "
            f"{volume.voxel_size_mm[0]}mm, space={volume.space.name} "
            f"(source: {volume.space.source})"
        ),
    )

    payload = {
        **volume.to_dict(),
        "space_resolvable": volume.space.resolvable,
        # Non-null only when we could not establish a space but geometry suggests one.
        # The UI turns this into a one-click "treat as …", which reloads with space=
        # set — so the assertion is still the human's, it just isn't a dead end.
        "suggested_space": volume.space.suggested_space if not volume.space.resolvable else None,
        "notes": [],
    }
    if not volume.space.resolvable:
        note = f"Region names cannot be resolved on this volume: {volume.space.detail}"
        if volume.space.suggested_space:
            note += (
                f" The geometry looks like {volume.space.grid_hint or 'a template'}; if this "
                f"volume really is normalised, reload it with "
                f"space='{volume.space.suggested_space}' and region names will work."
            )
        else:
            note += " Pass space= to assert one, or use explicit coordinates."
        payload["notes"].append(note)
    if volume.summary["n_nan"]:
        payload["notes"].append(
            f"{volume.summary['n_nan']} NaN voxels present. roi_stats will report how many "
            "it excludes from each region."
        )
    return _finalize(session, "load_volume", {"path": path, "name": name, "space": space}, payload, code)


# ---------------------------------------------------------------------------
# 2. load_atlas
# ---------------------------------------------------------------------------


@tool("load_atlas")
def load_atlas(session: Session, atlas_name: str) -> dict:
    """Load an atlas and publish its full label list.

    The label list is the point. It goes into the conversation so the model matches
    against vocabulary that actually exists rather than recalling names from training
    — NLI4VolVis's terminology failure mode, addressed with data instead of hope.
    """
    table = load_atlas_table(atlas_name)
    session.atlas = table

    labels = table.labels
    truncated = False
    if len(labels) > 250:
        labels, truncated = labels[:250], True

    code = "from nilearn import datasets\n" + atlas_fetch_code(table.atlas_id, table.maps_path)
    session.script.add(
        "load_atlas",
        code,
        comment=f"{table.atlas_id}: {len(table.regions)} regions in {table.space}",
    )

    payload = {
        "atlas_id": table.atlas_id,
        "description": table.description,
        "space": table.space,
        "resolution_mm": list(table.resolution_mm),
        "n_regions": len(table.regions),
        "labels": labels,
        "labels_truncated": truncated,
        "source": table.source,
        "citation": table.citation,
        "notes": [
            "Region names are resolved against this list only. A name that is not here "
            "returns the closest matches and asks — it is never mapped to a coordinate."
        ],
    }
    if truncated:
        payload["notes"].append(
            f"Showing the first 250 of {len(table.regions)} labels to stay inside the "
            "payload budget. Use list_regions(query=...) to search the rest."
        )
    if table.atlas_id == "demo-16":
        payload["notes"].append(
            "This is the synthetic demo atlas. Its regions are geometric shapes, not "
            "anatomy. Load 'harvard-oxford-sub' for real structures."
        )
    return _finalize(session, "load_atlas", {"atlas_name": atlas_name}, payload, code)


# ---------------------------------------------------------------------------
# 3. list_regions
# ---------------------------------------------------------------------------


@tool("list_regions")
def list_regions(session: Session, query: str | None = None, limit: int = 50) -> dict:
    """Filter the loaded atlas's labels. Rows carry coordinates, so they are clickable.

    A row in this table can be clicked to navigate the viewer with zero LLM calls (R5).
    """
    table = _require_atlas(session)
    limit = max(1, min(int(limit), 200))
    regions = table.search(query, limit=limit)

    rows = [
        {
            "index": region.index,
            "label": region.label,
            "n_voxels": region.n_voxels,
            "volume_mm3": round(region.volume_mm3, 1),
            "centroid": [round(c, 2) for c in region.centroid],
            "space": table.space,
        }
        for region in regions
    ]
    result_table = session.add_table("list_regions", ["index", "label", "n_voxels", "volume_mm3", "centroid"], rows)

    code = (
        f"# labels matching {query!r} in {table.atlas_id}\n"
        if query
        else f"# all labels in {table.atlas_id}\n"
    ) + "ATLAS_LABELS = " + repr([r["label"] for r in rows])
    session.script.add("list_regions", code, comment=f"{len(rows)} of {len(table.regions)} regions")

    payload = {
        "atlas_id": table.atlas_id,
        "space": table.space,
        "query": query,
        "n_matched": len(rows),
        "n_total": len(table.regions),
        "table_id": result_table.table_id,
        "regions": rows,
        "notes": [],
    }
    if query and not rows:
        payload["notes"].append(
            f"Nothing matched {query!r}. Closest labels: {', '.join(table.suggest(query))}."
        )
    if len(rows) == limit:
        payload["notes"].append(f"Truncated at limit={limit}; refine the query to see more.")
    return _finalize(session, "list_regions", {"query": query, "limit": limit}, payload, code)


# ---------------------------------------------------------------------------
# 4. navigate
# ---------------------------------------------------------------------------


@tool("navigate")
def navigate(
    session: Session,
    region_label: str | None = None,
    coords: list[float] | None = None,
    space: str | None = None,
    volume: str | None = None,
) -> dict:
    """Move the crosshair to a named region or to explicit coordinates.

    A ``region_label`` must exist in the loaded atlas or the call fails with
    suggestions. Coordinates are accepted only when you supply the numbers, and the
    response always echoes which space they were interpreted in.
    """
    if (region_label is None) == (coords is None):
        raise ValueError(
            "Give exactly one of region_label= or coords=. Region names are resolved "
            "through the atlas; coordinates are taken as given and echoed back with "
            "the space they were read in."
        )

    notes: list[str] = []

    if region_label is not None:
        table = _require_atlas(session)
        target_volume = session.volume(volume) if session.volumes else None
        if target_volume is not None:
            warning = _check_space_for_regions(session, target_volume, region_label)
            if warning:
                notes.append(warning)
        else:
            notes.append(
                "No volume loaded; coordinates are reported in the atlas's own space."
            )
        region = table.resolve(region_label)
        resolved_space = table.space
        label = region.label
        # Prefer the centre of mass, but only when it actually lands in the structure.
        # Hippocampus is curved enough that its centroid can sit in the ventricle next
        # door, and a crosshair outside the region you asked for is a wrong answer.
        resolved = region.centroid if region.centroid_inside else region.representative
        if not region.centroid_inside:
            notes.append(
                f"The centre of mass of {region.label} falls outside the structure "
                f"(it is a curved region), so the crosshair is at the in-region voxel "
                f"nearest to it. The centroid itself is "
                f"{[round(c, 1) for c in region.centroid]}."
            )
        detail = {
            "region_index": region.index,
            "atlas_id": table.atlas_id,
            "centroid": [round(c, 2) for c in region.centroid],
            "n_voxels": region.n_voxels,
            "volume_mm3": round(region.volume_mm3, 1),
        }
    else:
        if len(coords) != 3:
            raise ValueError(f"coords must be three numbers [x, y, z], got {coords!r}")
        resolved = tuple(float(c) for c in coords)
        label = None
        detail = {"source": "supplied by the caller"}
        if space:
            resolved_space = normalize_space_name(space)
            if resolved_space is None:
                raise ValueError(f"Unrecognised space {space!r}.")
        elif session.volumes:
            target_volume = session.volume(volume)
            resolved_space = target_volume.space.name
            notes.append(
                f"No space= given, so these coordinates are being read in {resolved_space}, "
                f"the space of volume {target_volume.name!r} ({target_volume.space.source})."
            )
        else:
            raise SpaceUnknownError(
                "Coordinates were supplied with no space= and no volume loaded, so there "
                "is nothing to interpret them against. Pass space='MNI152NLin6Asym' (or "
                "whichever applies)."
            )

    session.viewer.set_crosshair(resolved, resolved_space, label)

    coords_text = "[" + ", ".join(f"{c:.2f}" for c in resolved) + "]"
    code = (
        f"CROSSHAIR = {coords_text}  # {label or 'user-supplied'}, {resolved_space}\n"
        f"RESULTS[{f'step_{len(session.script.steps) + 1}_navigate'!r}] = "
        f"{{'label': {label!r}, 'coords': CROSSHAIR, 'space': {resolved_space!r}}}"
    )
    session.script.add(
        "navigate",
        code,
        comment=f"crosshair -> {label or coords_text} in {resolved_space}",
    )

    payload = {
        "label": label,
        "coords": [round(float(c), 2) for c in resolved],
        "space": resolved_space,
        "detail": detail,
        "notes": notes,
    }
    return _finalize(
        session,
        "navigate",
        {"region_label": region_label, "coords": coords, "space": space, "volume": volume},
        payload,
        code,
    )


# ---------------------------------------------------------------------------
# 5. set_display
# ---------------------------------------------------------------------------


@tool("set_display")
def set_display(
    session: Session,
    volume: str | None = None,
    colormap: str | None = None,
    min: float | None = None,  # noqa: A002 - the spec names these min/max
    max: float | None = None,  # noqa: A002
    opacity: float | None = None,
    visible: bool | None = None,
) -> dict:
    """Set colormap, intensity window and opacity for one layer.

    Windowing changes what you see, not what the data are. The applied settings are
    echoed back and written into the script so the exported figure matches.
    """
    target = session.volume(volume)
    layer = session.viewer.state.find(target.name)
    if layer is None:
        layer = LayerState(name=target.name, path=target.path)
        session.viewer.add_layer(layer)

    notes = []
    if colormap is not None:
        if colormap not in KNOWN_COLORMAPS:
            raise ValueError(
                f"Unknown colormap {colormap!r}. Known: {', '.join(KNOWN_COLORMAPS)}."
            )
        layer.colormap = colormap
    if min is not None:
        layer.cal_min = float(min)
    if max is not None:
        layer.cal_max = float(max)
    if opacity is not None:
        if not 0.0 <= float(opacity) <= 1.0:
            raise ValueError(f"opacity must be between 0 and 1, got {opacity}")
        layer.opacity = float(opacity)
    if visible is not None:
        layer.visible = bool(visible)

    if layer.cal_min is not None and layer.cal_max is not None and layer.cal_min >= layer.cal_max:
        raise ValueError(
            f"Display window is empty: min={layer.cal_min} is not below max={layer.cal_max}."
        )
    if layer.cal_max is not None and target.summary["max"] is not None:
        if layer.cal_max > target.summary["max"]:
            notes.append(
                f"max={layer.cal_max} is above the volume's actual maximum "
                f"({target.summary['max']}); the top of the colour range will be unused."
            )

    session.viewer.update_layer(layer)

    settings = {
        "colormap": layer.colormap,
        "vmin": layer.cal_min,
        "vmax": layer.cal_max,
        "opacity": layer.opacity,
        "visible": layer.visible,
    }
    code = (
        f"DISPLAY = globals().get('DISPLAY', {{}})\n"
        f"DISPLAY[{target.name!r}] = {settings!r}"
    )
    session.script.add("set_display", code, comment=f"display settings for {target.name}")

    payload = {"volume": target.name, "applied": settings, "notes": notes}
    return _finalize(
        session,
        "set_display",
        {"volume": volume, "colormap": colormap, "min": min, "max": max, "opacity": opacity},
        payload,
        code,
    )


# ---------------------------------------------------------------------------
# 6. overlay
# ---------------------------------------------------------------------------


@tool("overlay")
def overlay(session: Session, volume: str, on_top_of: str | None = None, opacity: float = 0.7) -> dict:
    """Stack one volume on top of another and report the resulting layer order."""
    top = session.volume(volume)
    base = session.volume(on_top_of) if on_top_of else None
    if base is not None and base.name == top.name:
        raise ValueError(f"Cannot overlay {top.name!r} on itself.")

    notes = []
    if base is not None:
        if base.space.name != top.space.name:
            notes.append(
                f"Layer spaces differ: {top.name} is {top.space.name}, {base.name} is "
                f"{base.space.name}. They are displayed on a shared world grid, but no "
                "registration is performed — alignment is your responsibility."
            )
        if tuple(base.shape[:3]) != tuple(top.shape[:3]):
            notes.append(
                f"Grids differ ({top.shape[:3]} vs {base.shape[:3]}). Niivue resamples for "
                "display only; roi_stats always resamples masks to each volume's own grid."
            )

    layer = session.viewer.state.find(top.name) or LayerState(name=top.name, path=top.path)
    layer.opacity = float(opacity)
    if layer.colormap == "gray":
        layer.colormap = "hot"
    if base is not None:
        base_layer = session.viewer.state.find(base.name)
        if base_layer is None:
            session.viewer.add_layer(LayerState(name=base.name, path=base.path), on_top=False)
    session.viewer.add_layer(layer, on_top=True)

    stack = [
        {"position": i, "name": l.name, "colormap": l.colormap, "opacity": l.opacity, "visible": l.visible}
        for i, l in enumerate(session.viewer.state.layers)
    ]
    code = (
        f"# overlay {top.name} on {base.name if base else 'the current base layer'} "
        f"at opacity {float(opacity)}\n"
        f"LAYER_STACK = {[l['name'] for l in stack]!r}"
    )
    session.script.add("overlay", code, comment="layer order, bottom first")

    payload = {"layers": stack, "top": top.name, "base": base.name if base else None, "notes": notes}
    return _finalize(
        session,
        "overlay",
        {"volume": volume, "on_top_of": on_top_of, "opacity": opacity},
        payload,
        code,
    )


# ---------------------------------------------------------------------------
# 7. roi_stats
# ---------------------------------------------------------------------------


@tool("roi_stats")
def roi_stats(
    session: Session,
    volume: str | None = None,
    region_label: str | None = None,
    mask_path: str | None = None,
    exclude_zeros: bool = False,
) -> dict:
    """Descriptive statistics inside a region or a mask, with exclusions counted.

    Descriptive only: a mean and a spread, no test statistic and no p-value. What it
    will not do is hide anything — every NaN, non-finite and zero voxel is counted in
    the response, because silent NaN handling is how wrong numbers enter papers.
    """
    if (region_label is None) == (mask_path is None):
        raise ValueError("Give exactly one of region_label= or mask_path=.")

    target = session.volume(volume)
    img = target.image()
    notes: list[str] = []

    if region_label is not None:
        warning = _check_space_for_regions(session, target, region_label)
        if warning:
            notes.append(warning)
        table = session.atlas
        region = table.resolve(region_label)
        mask = kernel.region_mask(table.maps_path, region.index, target_img=img)
        mask_code = (
            f"_mask = region_mask(ATLAS_PATH, {region.index}, "
            f"target_img=VOLUMES[{target.name!r}])  # {region.label}"
        )
        source = {
            "kind": "atlas_region",
            "atlas_id": table.atlas_id,
            "region_index": region.index,
            "label": region.label,
            "atlas_space": table.space,
            "atlas_resolution_mm": list(table.resolution_mm),
        }
        if tuple(round(z, 3) for z in table.resolution_mm) != tuple(
            round(z, 3) for z in target.voxel_size_mm
        ):
            notes.append(
                f"Atlas is {table.resolution_mm[0]}mm, volume is {target.voxel_size_mm[0]}mm. "
                "The label map was resampled to the volume grid with nearest-neighbour "
                "interpolation, which shifts region boundaries by up to one voxel."
            )
    else:
        mask_file = Path(mask_path).expanduser().resolve()
        if not mask_file.exists():
            raise FileNotFoundError(f"No such mask: {mask_file}")
        mask = kernel.mask_from_file(str(mask_file), target_img=img)
        mask_code = f"_mask = mask_from_file({str(mask_file)!r}, target_img=VOLUMES[{target.name!r}])"
        source = {"kind": "mask_file", "path": str(mask_file)}

    if not mask.any():
        raise ValueError(
            "The mask is empty on this volume's grid — no voxels to summarise. This "
            "usually means the volume and the mask do not overlap in world space."
        )

    stats = kernel.summarize_roi(img, mask, exclude_zeros=exclude_zeros)

    if stats["exclusions"]["nan"]:
        notes.append(
            f"{stats['exclusions']['nan']} of {stats['n_voxels_in_mask']} voxels in this "
            "region are NaN and were excluded from every statistic below."
        )

    # What the number means. A bare float looks equally authoritative whether it is a
    # quantitative PET SUV or an arbitrary MRI intensity that will not survive a change
    # of scanner, so say which — including saying that nobody recorded it.
    if target.units:
        notes.append(f"Values are in {target.units}.")
    else:
        notes.append(
            "No units are recorded for this volume, so these numbers carry no declared "
            "scale. MRI intensities in particular are arbitrary and are usually not "
            "comparable across scanners or sessions."
        )

    # How much to trust the number. Partial-volume contamination is invisible in the
    # output and worst exactly where people care most: small structures at PET
    # resolution.
    edge = stats.get("boundary_fraction") or 0.0
    narrow = stats.get("thickness_mm") or 0.0
    # 18mm is roughly three PET resolution elements. Below that, blur from neighbouring
    # tissue is a material part of the value. Judged on physical width rather than voxel
    # count, because PET is often stored on a finer grid than it was ever resolved at.
    if narrow and narrow < 18.0:
        notes.append(
            f"This region is only {narrow:.0f}mm thick, and {edge:.0%} of "
            "its voxels are on the boundary. If this is PET (4-6mm effective resolution, "
            "whatever the voxel size says) a material part of this value is signal blurred "
            "in from surrounding tissue. No partial-volume correction is applied."
        )
    elif edge >= 0.6:
        notes.append(
            f"{edge:.0%} of this region's voxels are on its boundary, so a substantial part "
            "of this value comes from the edge, where signal blurs across from neighbouring "
            "tissue. No partial-volume correction is applied."
        )
    if stats["exclusions"]["n_zero_in_mask"] and not exclude_zeros:
        notes.append(
            f"{stats['exclusions']['n_zero_in_mask']} voxels are exactly zero and were "
            "INCLUDED. Pass exclude_zeros=True if zero means 'no data' in this volume."
        )

    step_key = f"step_{len(session.script.steps) + 1}_roi_stats"
    code = (
        f"{mask_code}\n"
        f"RESULTS[{step_key!r}] = summarize_roi(VOLUMES[{target.name!r}], _mask, "
        f"exclude_zeros={bool(exclude_zeros)})"
    )
    session.script.add(
        "roi_stats",
        code,
        comment=f"{region_label or Path(str(mask_path)).name} in {target.name}",
    )

    rows = [{"metric": k, "value": v} for k, v in stats.items() if k != "exclusions"]
    result_table = session.add_table("roi_stats", ["metric", "value"], rows)

    payload = {
        "volume": target.name,
        "volume_space": target.space.name,
        "source": source,
        "stats": stats,
        "result_key": step_key,
        "table_id": result_table.table_id,
        "notes": notes,
    }
    return _finalize(
        session,
        "roi_stats",
        {
            "volume": volume,
            "region_label": region_label,
            "mask_path": mask_path,
            "exclude_zeros": exclude_zeros,
        },
        payload,
        code,
    )


# ---------------------------------------------------------------------------
# 8. compare_volumes
# ---------------------------------------------------------------------------


@tool("compare_volumes")
def compare_volumes(
    session: Session, a: str, b: str, method: str = "difference", name: str | None = None
) -> dict:
    """Voxelwise difference or ratio of two volumes. Arithmetic, not inference.

    The result is a new volume in the session. It carries no significance, no
    threshold and no claim — subtracting two images tells you they differ, and
    nothing at all about whether the difference means anything.
    """
    volume_a = session.volume(a)
    volume_b = session.volume(b)
    if method not in ("difference", "ratio"):
        raise ValueError(f"method must be 'difference' or 'ratio', got {method!r}")

    notes = []
    if volume_a.space.name != volume_b.space.name:
        notes.append(
            f"{volume_a.name} is in {volume_a.space.name} and {volume_b.name} is in "
            f"{volume_b.space.name}. No registration is performed; if these are not "
            "already aligned the result is meaningless."
        )
    if tuple(volume_a.shape[:3]) != tuple(volume_b.shape[:3]):
        notes.append(
            f"{volume_b.name} was resampled from {volume_b.shape[:3]} onto {volume_a.name}'s "
            f"grid {volume_a.shape[:3]} with continuous interpolation before the operation."
        )

    result_img = kernel.combine_volumes(volume_a.image(), volume_b.image(), method=method)
    result_name = name or f"{volume_a.name}_{'minus' if method == 'difference' else 'over'}_{volume_b.name}"
    out_path = session.temp_path(f"{result_name}.nii.gz")
    result_img.to_filename(str(out_path))

    volume = session.add_volume(out_path, name=result_name, space=volume_a.space.name)
    summary = volume.summary
    if summary["n_nan"]:
        notes.append(
            f"The result contains {summary['n_nan']} NaN voxels"
            + (
                " (ratio: division by near-zero was set to NaN rather than to a large number)."
                if method == "ratio"
                else " (inherited from the inputs)."
            )
        )

    step_key = f"step_{len(session.script.steps) + 1}_compare_volumes"
    code = (
        f"VOLUMES[{result_name!r}] = combine_volumes(\n"
        f"    VOLUMES[{volume_a.name!r}], VOLUMES[{volume_b.name!r}], method={method!r}\n"
        f")\n"
        f"nib.save(VOLUMES[{result_name!r}], _out({str(out_path)!r}))\n"
        f"RESULTS[{step_key!r}] = summarize_volume(VOLUMES[{result_name!r}])"
    )
    session.script.add("compare_volumes", code, comment=f"{method}: {volume_a.name} vs {volume_b.name}")

    payload = {
        "name": result_name,
        "path": str(out_path),
        "method": method,
        "inputs": [volume_a.name, volume_b.name],
        "space": volume.space.name,
        "summary": summary,
        "result_key": step_key,
        "notes": notes
        + [
            "Descriptive only. This is not a statistical map: there is no test, no "
            "threshold and no correction for anything."
        ],
    }
    return _finalize(session, "compare_volumes", {"a": a, "b": b, "method": method}, payload, code)


# ---------------------------------------------------------------------------
# 9. screenshot
# ---------------------------------------------------------------------------


@tool("screenshot")
def screenshot(session: Session, filename: str | None = None, timeout: float = 8.0) -> dict:
    """Capture the current view as a downscaled PNG and return its path.

    Prefers the live Niivue canvas when a browser is attached, because that is what
    the user is actually looking at. With no viewer attached it renders the same layer
    stack server-side with nilearn instead, so headless MCP sessions still produce
    pictures. The response always says which renderer ran — the two do not match
    pixel for pixel and pretending otherwise would be a lie of omission.
    """
    filename = filename or f"view-{len(session.script.steps) + 1:02d}.png"
    out_path = session.temp_path(filename)

    raw = session.viewer.request_snapshot(timeout=timeout)
    if raw:
        out_path.write_bytes(downscale_png(raw))
        renderer = "niivue-canvas"
        note = "Captured from the live Niivue canvas."
    else:
        if not session.viewer.state.layers:
            raise ValueError("Nothing to capture: no volume is loaded.")
        render_server_side(
            session.viewer.state.layers,
            session.viewer.state.crosshair_mm,
            out_path,
            title=session.viewer.state.crosshair_label,
        )
        renderer = "nilearn-fallback"
        note = (
            "No browser viewer attached, so this was rendered server-side with nilearn. "
            "It shows the same layers and crosshair, but it is not a pixel copy of Niivue."
        )

    layer = session.viewer.state.layers[0] if session.viewer.state.layers else None
    cut = ", ".join(f"{c:.1f}" for c in session.viewer.state.crosshair_mm)
    code = (
        f"# Live view is Niivue (WebGL, browser-side). The nilearn equivalent:\n"
        f"_display = plotting.plot_anat(\n"
        f"    VOLUMES[{(layer.name if layer else 'base')!r}], display_mode='ortho',\n"
        f"    cut_coords=({cut}), black_bg=True, annotate=True,\n"
        f")\n"
        + "".join(
            f"_display.add_overlay(VOLUMES[{l.name!r}], cmap={l.colormap!r}, alpha={l.opacity})\n"
            for l in session.viewer.state.layers[1:]
            if l.visible
        )
        + f"_display.savefig(_out({str(out_path)!r}), dpi=100)\n"
        f"_display.close()"
    )
    session.script.add("screenshot", code, comment=f"figure at [{cut}]")

    size_bytes = out_path.stat().st_size
    payload = {
        "path": str(out_path),
        "renderer": renderer,
        "size_bytes": size_bytes,
        "crosshair_mm": [round(float(c), 2) for c in session.viewer.state.crosshair_mm],
        "space": session.viewer.state.crosshair_space,
        "layers": [l.name for l in session.viewer.state.layers if l.visible],
        "notes": [note, "Downscaled to 768px on the long edge; the image itself is never "
                  "returned inline."],
    }
    return _finalize(session, "screenshot", {"filename": filename}, payload, code)


# ---------------------------------------------------------------------------
# 10. export_script
# ---------------------------------------------------------------------------


@tool("export_script")
def export_script(session: Session, path: str) -> dict:
    """Write the session as a standalone, runnable ``.py``.

    Needs only numpy, nibabel and nilearn — not neurochat. Running it reproduces the
    final state and prints the same numbers this session reported.
    """
    if not session.script.steps:
        raise ValueError("Nothing to export yet: no tool calls have been made.")

    written = session.script.export(path)
    text = written.read_text()

    payload = {
        "path": str(written),
        "n_steps": len(session.script.steps),
        "n_lines": text.count("\n") + 1,
        "size_bytes": written.stat().st_size,
        "tools_used": sorted({s.tool for s in session.script.steps}),
        "run_with": f"python {written}",
        "notes": [
            "Standalone: requires numpy, nibabel and nilearn, not neurochat.",
            "Prints a JSON object of every recorded result, so a re-run can be diffed "
            "against this session.",
        ],
    }
    return _finalize(session, "export_script", {"path": path}, payload, "")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS = {
    "load_volume": load_volume,
    "load_atlas": load_atlas,
    "list_regions": list_regions,
    "navigate": navigate,
    "set_display": set_display,
    "overlay": overlay,
    "roi_stats": roi_stats,
    "compare_volumes": compare_volumes,
    "screenshot": screenshot,
    "export_script": export_script,
}

assert len(TOOLS) == 10, "v1 is exactly ten tools; adding an eleventh is a spec change"


def call(session: Session, tool_name: str, /, **kwargs) -> dict:
    """Dispatch by name. Used by the web UI's deterministic (no-LLM) paths.

    ``session`` and ``tool_name`` are positional-only so that a tool argument called
    ``name`` — load_volume has one — cannot collide with this signature.
    """
    if tool_name not in TOOLS:
        raise KeyError(f"Unknown tool {tool_name!r}. Available: {', '.join(sorted(TOOLS))}")
    return TOOLS[tool_name](session, **kwargs)
