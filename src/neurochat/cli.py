"""Command line entry point.

    neurochat demo       # load the bundled sample data and open the viewer
    neurochat serve      # empty session, viewer only
    neurochat mcp        # MCP server on stdio, for Claude Desktop / Claude Code
    neurochat check      # Phase 0 self-test: does atlas grounding actually work here?
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _real_template_path():
    """A real human brain image, with no download: the template nilearn ships.

    It is the asymmetric ICBM152 2009a template — an average of real brains, and a
    genuinely different MNI152 variant from the one the Harvard-Oxford atlas lives in.
    We label it accurately and let the variant warning fire, because that few-millimetre
    disagreement is real and is exactly the kind of thing this tool exists to surface.
    """
    from nilearn import datasets

    from .atlas import cache_dir

    path = cache_dir() / "mni152_icbm152_2009a_t1_1mm.nii.gz"
    if not path.exists():
        datasets.load_mni152_template().to_filename(str(path))
    return path


def _demo_session(offline: bool = False):
    from .atlas import bundled_atlas_path
    from .session import Session
    from . import tools

    session = Session(name="demo")
    sample = bundled_atlas_path("phantom_t1.nii.gz").parent

    if not offline:
        try:
            template = _real_template_path()
            # The header on this file says sform_code=2 ("aligned to something"), not 4,
            # so detection would decline it. We assert the space here because we know
            # what the file is — the same one-argument fix the UI offers to a user.
            tools.load_volume(
                session, path=str(template), name="mni152_t1", space="MNI152NLin2009aSym"
            )
            atlas = tools.load_atlas(session, atlas_name="harvard-oxford-sub")
            if not atlas.get("ok"):
                raise RuntimeError(atlas.get("message", "atlas load failed"))
            tools.set_display(session, volume="mni152_t1", colormap="gray")
            tools.navigate(session, region_label="Left Hippocampus")
            print("Loaded a real brain: nilearn's ICBM152 2009a template (an average of")
            print("152 real brains), with the Harvard-Oxford subcortical atlas — 21 real")
            print("anatomical structures. The crosshair is on the left hippocampus.")
            print()
            print("Note the tool warns that the volume and the atlas are different MNI152")
            print("variants. That disagreement is real, and it is the point.")
            print()
            print("For the offline synthetic phantoms instead: neurochat demo --offline")
            return session
        except Exception as exc:  # noqa: BLE001 - offline or fetch failure is not fatal
            print(f"Could not set up the real-data demo ({type(exc).__name__}: {exc}).")
            print("Falling back to the bundled synthetic phantoms.\n")

    tools.load_volume(session, path=str(sample / "phantom_t1.nii.gz"), name="t1")
    tools.load_volume(session, path=str(sample / "phantom_pet_baseline.nii.gz"), name="pet")
    tools.load_atlas(session, atlas_name="demo-16")
    tools.set_display(session, volume="pet", colormap="hot", min=0.6, max=2.2, opacity=0.7)
    tools.overlay(session, volume="pet", on_top_of="t1", opacity=0.7)
    tools.navigate(session, region_label="Left Deep Sphere")
    print("Loaded the bundled synthetic phantoms and the demo-16 atlas — no network needed.")
    print("Those regions are geometric shapes, NOT anatomy. For real structures on a real")
    print("brain, run `neurochat demo` without --offline (one-time ~26MB atlas download).")
    return session


def cmd_check(args) -> int:
    """Prove the R1 grounding path works on this machine before trusting anything else."""
    from .atlas import load_atlas_table
    from .errors import RegionNotFoundError

    name = args.atlas
    print(f"Loading atlas {name!r}…")
    table = load_atlas_table(name)
    print(f"  {table.atlas_id}: {len(table.regions)} regions in {table.space} "
          f"at {table.resolution_mm[0]}mm")

    probe = args.region or ("Left Deep Sphere" if table.atlas_id == "demo-16" else "Left Hippocampus")
    try:
        region = table.resolve(probe)
    except RegionNotFoundError as exc:
        print(f"  resolve({probe!r}) failed: {exc.message}")
        return 1
    coords = ", ".join(f"{c:.1f}" for c in region.centroid)
    print(f"  resolve({probe!r}) -> {region.label} at [{coords}] {table.space}")
    print(f"    {region.n_voxels} voxels, {region.volume_mm3:.0f} mm^3, "
          f"centroid inside region: {region.centroid_inside}")

    typo = probe[:-1] + "x"
    try:
        table.resolve(typo)
        print(f"  PROBLEM: resolve({typo!r}) returned a match instead of asking.")
        return 1
    except RegionNotFoundError as exc:
        print(f"  resolve({typo!r}) -> did-you-mean {exc.suggestions}")
    print("\nGrounding works: names resolve from the atlas, typos ask instead of guessing.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neurochat", description=__doc__)
    parser.add_argument("--version", action="version", version=f"neurochat {__version__}")
    sub = parser.add_subparsers(dest="command")

    for name, help_text in (
        ("demo", "load a real brain plus a real atlas and open the viewer"),
        ("serve", "start the viewer with an empty session"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        if name == "demo":
            p.add_argument(
                "--offline",
                action="store_true",
                help="use the bundled synthetic phantoms instead; no network at all",
            )

    p_mcp = sub.add_parser("mcp", help="run the MCP server (stdio by default)")
    p_mcp.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    p_mcp.add_argument("--host", default="127.0.0.1")
    p_mcp.add_argument("--port", type=int, default=8931)
    p_mcp.add_argument("--backend", default=None, help="URL of a running viewer to drive")

    p_check = sub.add_parser("check", help="self-test the atlas grounding harness")
    p_check.add_argument("--atlas", default="demo-16")
    p_check.add_argument("--region", default=None)

    args = parser.parse_args(argv)

    if args.command == "check":
        return cmd_check(args)
    if args.command == "mcp":
        from .server_mcp import main as mcp_main

        argv_mcp = ["--transport", args.transport, "--host", args.host, "--port", str(args.port)]
        if args.backend:
            argv_mcp += ["--backend", args.backend]
        mcp_main(argv_mcp)
        return 0
    if args.command in ("demo", "serve"):
        from .server_web import main as web_main

        session = _demo_session(offline=args.offline) if args.command == "demo" else None
        web_main(host=args.host, port=args.port, session=session)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
