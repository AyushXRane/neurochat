"""MCP server — the tool surface, exposed to Claude Desktop, Claude Code, or any MCP client.

This is consumption mode 2 from the spec, and the one that is meant to get attention.
The web UI in :mod:`neurochat.server_web` is a client of the same ten functions in
:mod:`neurochat.tools`, not a parallel implementation.

Two ways to run:

    neurochat-mcp                      # stdio, headless; screenshots render via nilearn
    neurochat-mcp --transport streamable-http --port 8931

Set ``NEUROCHAT_BACKEND=http://127.0.0.1:8000`` to drive a running web viewer instead
of a headless session. Tool calls are then proxied to that backend, so the browser
canvas moves as the conversation goes and ``screenshot()`` captures what the user is
actually looking at.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import Image, MCPServer

from . import __version__, tools
from .atlas import ATLAS_REGISTRY
from .scope import check_scope, record_refusal
from .session import Session

INSTRUCTIONS = """\
neurochat explores already-preprocessed volumetric neuroimaging data (NIfTI: MRI, PET)
and emits reproducible nilearn code for everything it does.

Rules that are not negotiable, and that the tools enforce whether or not you follow them:

1. NEVER state an anatomical coordinate from your own knowledge. Every location comes
   from navigate() or list_regions(), resolved against the loaded atlas. If you catch
   yourself about to write "the left hippocampus is around -26, -24, -14", call the tool
   instead. Your recollection of that number is exactly the failure mode this tool exists
   to prevent.
2. ALWAYS state the space alongside any location: MNI152NLin6Asym, MNI152NLin2009cAsym,
   native, or voxel[i,j,k]. The tools return it; pass it on.
3. A region name that does not match returns suggestions. Present them and ASK. Do not
   pick one, even when the intended region is obvious.
4. Report exclusions. When roi_stats says it dropped NaN voxels, say so in your answer.
   A mean quoted without its exclusion count is a number that can end up in a paper wrong.
5. Refuse, don't attempt, when asked for: statistical inference (t-tests, p-values,
   thresholded significance maps), preprocessing (DICOM conversion, recon-all, fMRIPrep,
   motion correction), or running arbitrary code. Name the tool that does the job instead
   — nilearn.glm or FSL randomise for stats, fMRIPrep or dcm2niix for preprocessing.
6. Make no clinical or diagnostic claim, ever, in any wording.

Typical flow: load_volume -> load_atlas -> list_regions/navigate -> roi_stats ->
screenshot -> export_script. Every call appends to a session script the user can export
and re-run; that script, not the picture, is the deliverable.
"""


# ---------------------------------------------------------------------------
# Session wiring
# ---------------------------------------------------------------------------

_SESSION: Session | None = None
_BACKEND = os.environ.get("NEUROCHAT_BACKEND", "").rstrip("/")


def get_session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = Session(name="mcp")
    return _SESSION


def _proxy(tool_name: str, arguments: dict) -> dict:
    """Send the call to a running web backend so an attached browser viewer follows along."""
    import urllib.error
    import urllib.request

    payload = json.dumps({"tool": tool_name, "args": arguments}).encode()
    request = urllib.request.Request(
        f"{_BACKEND}/api/tool",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": "BackendUnreachable",
            "message": (
                f"NEUROCHAT_BACKEND is set to {_BACKEND} but the call failed ({exc}). "
                "Start the viewer with `neurochat serve`, or unset NEUROCHAT_BACKEND to "
                "run headless."
            ),
        }


def dispatch(tool_name: str, **arguments: Any) -> dict:
    """One entry point for every tool, local or proxied."""
    clean = {k: v for k, v in arguments.items() if v is not None}
    if _BACKEND:
        return _proxy(tool_name, clean)
    return tools.call(get_session(), tool_name, **clean)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = MCPServer(
    name="neurochat",
    title="neurochat — conversational neuroimaging viewer",
    version=__version__,
    instructions=INSTRUCTIONS,
)


@server.tool()
def load_volume(path: str, name: str | None = None, space: str | None = None) -> dict:
    """Load an already-preprocessed NIfTI volume (MRI, PET) into the session.

    Returns shape, voxel size, affine summary, detected space and how it was detected,
    value range, and NaN count. No preprocessing is performed on the file.

    Args:
        path: Path to a .nii or .nii.gz file.
        name: Short handle for later calls. Defaults to the filename stem.
        space: Assert the template space when the header does not record one, e.g.
            "MNI152NLin6Asym". Recorded as your assertion, not an inference.
    """
    return dispatch("load_volume", path=path, name=name, space=space)


@server.tool()
def load_atlas(atlas_name: str) -> dict:
    """Load an atlas and return its full label list.

    The label list is what grounds every later region name — match user phrasing against
    these strings, not against your own knowledge of neuroanatomy.

    Args:
        atlas_name: One of demo-16, harvard-oxford-sub, harvard-oxford-cort, aal,
            schaefer-100. Non-bundled atlases are fetched once via nilearn and cached.
    """
    return dispatch("load_atlas", atlas_name=atlas_name)


@server.tool()
def list_regions(query: str | None = None, limit: int = 50) -> dict:
    """List or search the loaded atlas's regions, with voxel counts and centroids.

    Args:
        query: Substring or fuzzy filter, e.g. "hippocampus". Omit for everything.
        limit: Maximum rows to return (capped at 200).
    """
    return dispatch("list_regions", query=query, limit=limit)


@server.tool()
def navigate(
    region_label: str | None = None,
    coords: list[float] | None = None,
    space: str | None = None,
    volume: str | None = None,
) -> dict:
    """Move the crosshair to a named atlas region or to explicit coordinates.

    Exactly one of region_label or coords. A region_label must exist in the loaded
    atlas; a near-miss returns suggestions and moves nothing. Coordinates are accepted
    only from you or the user, and the space they were read in is echoed back.

    Args:
        region_label: Label from the loaded atlas, e.g. "Left Hippocampus".
        coords: [x, y, z] in millimetres.
        space: Space for coords, e.g. "MNI152NLin6Asym". Defaults to the volume's space.
        volume: Which loaded volume to interpret against. Defaults to the most recent.
    """
    return dispatch(
        "navigate", region_label=region_label, coords=coords, space=space, volume=volume
    )


@server.tool()
def set_display(
    volume: str | None = None,
    colormap: str | None = None,
    min: float | None = None,
    max: float | None = None,
    opacity: float | None = None,
) -> dict:
    """Set colormap, intensity window and opacity for one layer.

    Args:
        volume: Layer to restyle. Defaults to the most recently loaded.
        colormap: gray, hot, cool, viridis, inferno, magma, plasma, jet, bone, ...
        min: Low end of the display window (does not alter the data).
        max: High end of the display window.
        opacity: 0.0 to 1.0.
    """
    return dispatch(
        "set_display", volume=volume, colormap=colormap, min=min, max=max, opacity=opacity
    )


@server.tool()
def overlay(volume: str, on_top_of: str | None = None, opacity: float = 0.7) -> dict:
    """Stack one loaded volume on top of another and return the layer order.

    Display only — no registration is performed. If the two volumes are not already
    aligned, say so rather than implying the overlay means anything.

    Args:
        volume: The layer to put on top.
        on_top_of: The base layer. Defaults to the current bottom layer.
        opacity: Opacity of the top layer, 0.0 to 1.0.
    """
    return dispatch("overlay", volume=volume, on_top_of=on_top_of, opacity=opacity)


@server.tool()
def roi_stats(
    volume: str | None = None,
    region_label: str | None = None,
    mask_path: str | None = None,
    exclude_zeros: bool = False,
) -> dict:
    """Descriptive statistics inside an atlas region or a mask file.

    Returns n_voxels, mean, sd, median, min, max, and an explicit count of every
    excluded voxel. Report those exclusion counts in your answer — they are not a
    footnote. There is no inference here: no test, no p-value, no threshold.

    Args:
        volume: Volume to measure. Defaults to the most recently loaded.
        region_label: Atlas region name. Mutually exclusive with mask_path.
        mask_path: Path to a binary mask NIfTI. Mutually exclusive with region_label.
        exclude_zeros: Drop exactly-zero voxels. Off by default; zeros are counted
            and included, because treating zero as "no data" is an assumption.
    """
    return dispatch(
        "roi_stats",
        volume=volume,
        region_label=region_label,
        mask_path=mask_path,
        exclude_zeros=exclude_zeros,
    )


@server.tool()
def compare_volumes(a: str, b: str, method: str = "difference", name: str | None = None) -> dict:
    """Voxelwise difference or ratio of two loaded volumes; adds the result to the session.

    Arithmetic, not inference. The output is not a statistical map and carries no
    significance of any kind — describe it as a difference image, nothing more.

    Args:
        a: First volume (the minuend / numerator). Defines the output grid.
        b: Second volume. Resampled onto a's grid if the grids differ.
        method: "difference" or "ratio".
        name: Handle for the result volume.
    """
    return dispatch("compare_volumes", a=a, b=b, method=method, name=name)


@server.tool()
def screenshot(filename: str | None = None) -> list:
    """Capture the current view as a downscaled PNG and return the image and its path.

    Uses the live Niivue canvas when a browser viewer is attached, otherwise renders
    the same layer stack server-side with nilearn. The response says which one ran.

    Args:
        filename: Optional name for the PNG inside the session's temp directory.
    """
    result = dispatch("screenshot", filename=filename)
    if not result.get("ok"):
        return [json.dumps(result, indent=2)]
    contents: list = [json.dumps(result, indent=2)]
    path = Path(result["path"])
    if path.exists():
        # Already downscaled to 768px on the long edge by the tool (R4).
        contents.insert(0, Image(path=str(path)))
    return contents


@server.tool()
def export_script(path: str) -> dict:
    """Write the whole session as a standalone runnable .py and return its path.

    This is the deliverable. It needs only numpy, nibabel and nilearn — not neurochat —
    and re-running it reproduces the numbers reported in this conversation.

    Args:
        path: Where to write the script, e.g. "~/analysis/session.py".
    """
    return dispatch("export_script", path=path)


# ---------------------------------------------------------------------------
# Resources: state, live script, tool traces
# ---------------------------------------------------------------------------


@server.resource("neurochat://session/state", mime_type="application/json")
def session_state() -> str:
    """Current session state: loaded volumes, atlas, viewer layers, crosshair."""
    if _BACKEND:
        return json.dumps(_proxy("__state__", {}), indent=2)
    return json.dumps(get_session().state(), indent=2, default=str)


@server.resource("neurochat://session/script", mime_type="text/x-python")
def session_script() -> str:
    """The live nilearn script accumulated so far, exactly as export_script would write it."""
    return get_session().script.render()


@server.resource("neurochat://debug/tool_trace", mime_type="application/json")
def tool_trace() -> str:
    """The last few full tool traces — arguments in, results out. Bounded, in memory."""
    return json.dumps(get_session().recent_traces(5), indent=2, default=str)


@server.resource("neurochat://atlases", mime_type="application/json")
def atlas_catalogue() -> str:
    """Which atlases can be loaded, their spaces, and whether they need a download."""
    return json.dumps(
        {
            spec.atlas_id: {
                "description": spec.description,
                "space": spec.space,
                "bundled": spec.bundled,
                "citation": spec.citation,
            }
            for spec in ATLAS_REGISTRY.values()
        },
        indent=2,
    )


@server.prompt()
def explore_volume(volume_path: str, atlas: str = "harvard-oxford-sub") -> str:
    """A starting prompt for exploring a volume with the rules already in mind."""
    return (
        f"Load the volume at {volume_path} and the {atlas} atlas. Tell me what the volume "
        "is — dimensions, voxel size, which space it is in and how you know, and whether "
        "it contains NaNs. Then wait for me. Do not guess any coordinates: resolve every "
        "region name through the atlas, and state the space with every location."
    )


def check_request_scope(text: str) -> dict:
    """Classify a user request against the Non-Goals before acting on it.

    Exposed for the web chat layer and for tests. Not an MCP tool: refusal is a
    property of the surface, not something a model should be able to call around.
    """
    refusal = check_scope(text)
    if refusal is None:
        return {"in_scope": True}
    if not _BACKEND:
        record_refusal(get_session(), refusal)
    return {"in_scope": False, **refusal.to_dict()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="neurochat-mcp", description="Run the neurochat MCP server."
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="stdio for Claude Desktop / Claude Code; streamable-http for network clients.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument(
        "--backend",
        default=None,
        help="URL of a running `neurochat serve` viewer to drive (sets NEUROCHAT_BACKEND).",
    )
    args = parser.parse_args(argv)

    if args.backend:
        global _BACKEND
        _BACKEND = args.backend.rstrip("/")

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
