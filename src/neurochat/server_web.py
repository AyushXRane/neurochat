"""FastAPI backend: viewer state over WebSocket, tools over HTTP, chat over SSE.

Local-first and single-session by design (Non-Goal 6: no accounts, no cloud storage,
no multi-tenancy). The web UI is a *client* of :mod:`neurochat.tools`, exactly like
the MCP server is — there is no second implementation of anything here.

Three routes carry the interesting behaviour:

* ``POST /api/tool`` — the deterministic path. The UI calls it directly when the user
  clicks an atlas region or a results row, so those actions cost zero LLM calls (R5).
  The MCP server also proxies through it when ``NEUROCHAT_BACKEND`` is set, which is
  how a Claude Desktop conversation drives an attached browser viewer.
* ``WS /ws`` — viewer commands out, canvas snapshots back in.
* ``GET /debug/tool_trace`` — the last N full traces, in memory and bounded, lifted
  straight from neuroglancer-chat.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, library, tools as tool_module
from .agent import ChatAgent
from .atlas import ATLAS_REGISTRY
from .errors import NeurochatError
from .session import Session


def web_root() -> Path:
    """Locate the UI whether running from a checkout or an installed wheel."""
    for candidate in (Path(__file__).parent / "_web", Path(__file__).parent.parent.parent / "web"):
        if (candidate / "index.html").exists():
            return candidate
    raise FileNotFoundError("web/index.html not found — is the package built correctly?")


def create_app(session: Session | None = None) -> FastAPI:
    session = session or Session(name="web")
    agent = ChatAgent(session)
    app = FastAPI(title="neurochat", version=__version__)
    app.state.session = session
    app.state.agent = agent

    root = web_root()
    app.mount("/static", StaticFiles(directory=str(root)), name="static")

    # -- pages -----------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (root / "index.html").read_text()

    # -- state and tools --------------------------------------------------

    @app.get("/api/state")
    def get_state() -> dict:
        return {
            **session.state(),
            "chat_available": agent.available,
            "atlases": {
                spec.atlas_id: {
                    "description": spec.description,
                    "space": spec.space,
                    "bundled": spec.bundled,
                }
                for spec in ATLAS_REGISTRY.values()
            },
        }

    @app.post("/api/tool")
    async def run_tool(request: Request) -> dict:
        """Run one tool. No LLM involved — this is the deterministic path (R5)."""
        body = await request.json()
        name = body.get("tool")
        arguments = {k: v for k, v in (body.get("args") or {}).items() if v is not None}
        if name == "__state__":
            return session.state()
        if name not in tool_module.TOOLS:
            raise HTTPException(404, f"Unknown tool {name!r}")
        return await asyncio.to_thread(tool_module.call, session, name, **arguments)

    @app.get("/api/script", response_class=PlainTextResponse)
    def get_script(full: bool = False) -> str:
        return session.script.render() if full else session.script.render_live()

    @app.get("/api/regions")
    def get_regions(query: str | None = None, limit: int = 200) -> dict:
        """The atlas panel's data source. Deterministic, no model call."""
        if session.atlas is None:
            return {"atlas_id": None, "regions": []}
        return {
            "atlas_id": session.atlas.atlas_id,
            "space": session.atlas.space,
            "regions": [r.to_dict() for r in session.atlas.search(query, limit=limit)],
        }

    @app.get("/debug/tool_trace")
    def tool_trace(n: int = Query(5, ge=1, le=50)) -> dict:
        """Recent full traces: arguments in, results out. Bounded, in memory."""
        return {"n": n, "traces": session.recent_traces(n)}

    # -- library: many scans on disk, one at a time in the viewer ----------

    app.state.library = {"root": None, "entries": []}

    @app.post("/api/library/scan")
    async def library_scan(request: Request) -> dict:
        """Find NIfTI files under a directory. Reads headers only; loads nothing."""
        body = await request.json()
        directory = (body.get("directory") or "").strip() or library.default_library_root()
        try:
            result = await asyncio.to_thread(
                library.scan_directory, directory, bool(body.get("recursive", True))
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise HTTPException(400, str(exc)) from exc
        app.state.library = {"root": result["root"], "entries": result["entries"]}
        return result

    @app.post("/api/library/sample_cohort")
    async def library_sample_cohort(request: Request) -> dict:
        """Fetch a real multi-subject cohort (OASIS-1) so the library has real data."""
        body = await request.json() if await request.body() else {}
        try:
            result = await asyncio.to_thread(
                library.sample_cohort, int(body.get("n_subjects", 12))
            )
        except Exception as exc:  # noqa: BLE001 - a failed download is a message, not a 500
            return {
                "ok": False,
                "error": type(exc).__name__,
                "message": f"Could not fetch the OASIS cohort ({exc}). It needs network access "
                "the first time; afterwards it is cached.",
            }
        app.state.library = {"root": result["root"], "entries": result["entries"]}
        return {"ok": True, **result}

    @app.post("/api/library/select")
    async def library_select(request: Request) -> dict:
        """Load one scan from the library and make it the only thing on screen.

        Deliberately routed through load_volume/set_display rather than around them,
        so selecting a scan is recorded in the script like any other action.
        """
        body = await request.json()
        path = body.get("path")
        if not path:
            raise HTTPException(400, "path is required")

        extra = {k: body[k] for k in ("name", "space") if body.get(k)}
        result = await asyncio.to_thread(
            tool_module.call, session, "load_volume", path=path, **extra
        )
        if not result.get("ok"):
            return result

        if body.get("solo", True):
            selected = result["name"]
            for layer in list(session.viewer.state.layers):
                if layer.name != selected and layer.visible:
                    await asyncio.to_thread(
                        tool_module.call, session, "set_display", volume=layer.name, visible=False
                    )
            await asyncio.to_thread(
                tool_module.call, session, "set_display", volume=selected, visible=True
            )
        result["state"] = session.state()
        return result

    @app.post("/api/library/region_table")
    async def library_region_table(request: Request) -> dict:
        """One atlas region measured across every scan in the library. No model call."""
        body = await request.json()
        region_label = body.get("region_label")
        if not region_label:
            raise HTTPException(400, "region_label is required")
        paths = body.get("paths") or [e["path"] for e in app.state.library["entries"]]
        if not paths:
            raise HTTPException(400, "The library is empty — scan a folder first.")
        try:
            return await asyncio.to_thread(
                library.region_across_library,
                session,
                paths,
                region_label,
                bool(body.get("exclude_zeros", False)),
                body.get("assume_space"),
            )
        except NeurochatError as exc:
            return {"ok": False, **exc.to_dict()}
        except (ValueError, KeyError) as exc:
            return {"ok": False, "error": type(exc).__name__, "message": str(exc)}

    @app.get("/api/library/table.csv", response_class=PlainTextResponse)
    def library_table_csv(table_id: str) -> PlainTextResponse:
        """The cohort table as CSV — the next thing anyone does with it is paste it."""
        table = session.tables.get(table_id)
        if table is None:
            raise HTTPException(404, "No such table in this session.")
        body = library.rows_to_csv(table.rows, table.columns)
        return PlainTextResponse(
            body,
            headers={"Content-Disposition": f'attachment; filename="{table.tool}-{table_id}.csv"'},
        )

    # -- files -------------------------------------------------------------

    def _authorised(path: Path) -> bool:
        """Only serve what this session actually loaded, plus its own artifacts."""
        resolved = path.resolve()
        if any(Path(v.path).resolve() == resolved for v in session.volumes.values()):
            return True
        if session.atlas and Path(session.atlas.maps_path).resolve() == resolved:
            return True
        return session.workdir.resolve() in resolved.parents

    @app.get("/api/file")
    def get_file(path: str) -> FileResponse:
        target = Path(path).expanduser()
        if not target.exists() or not _authorised(target):
            raise HTTPException(404, "Not a file this session has loaded.")
        return FileResponse(str(target))

    # -- viewer socket -----------------------------------------------------

    @app.websocket("/ws")
    async def viewer_socket(socket: WebSocket) -> None:
        await socket.accept()
        bridge = session.viewer
        bridge.client_connected()
        loop = asyncio.get_running_loop()
        changed = bridge.register_waiter(loop)
        last_revision = 0

        await socket.send_json({"type": "state", "state": bridge.state.to_dict()})

        async def pump() -> None:
            """Push viewer commands as tools emit them."""
            nonlocal last_revision
            while True:
                await changed.wait()
                changed.clear()
                for command in bridge.commands_since(last_revision):
                    last_revision = command["revision"]
                    await socket.send_json(command)

        pusher = asyncio.create_task(pump())
        try:
            while True:
                message = await socket.receive_json()
                if message.get("type") == "snapshot":
                    bridge.deliver_snapshot(message["request_id"], message.get("data_url", ""))
                elif message.get("type") == "crosshair":
                    # The user dragged the crosshair in the canvas. Record it so the
                    # next screenshot and the next emitted snippet agree with what
                    # they are looking at, but do not echo a command back.
                    coords = message.get("coords") or [0, 0, 0]
                    bridge.state.crosshair_mm = tuple(float(c) for c in coords)
                    bridge.state.crosshair_label = None
        except (WebSocketDisconnect, RuntimeError, KeyError, ValueError):
            pass
        finally:
            pusher.cancel()
            bridge.unregister_waiter(changed)
            bridge.client_disconnected()

    # -- chat --------------------------------------------------------------

    @app.post("/api/chat")
    async def chat(request: Request) -> StreamingResponse:
        body = await request.json()
        message = (body.get("message") or "").strip()
        if not message:
            raise HTTPException(400, "Empty message.")

        def events():
            # Starlette runs this sync generator in a worker thread, which is what
            # lets screenshot() block on a canvas snapshot without stalling the loop.
            for event in agent.run(message):
                yield f"data: {json.dumps(event, default=str)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = None


def main(host: str = "127.0.0.1", port: int = 8000, session: Session | None = None) -> None:
    import uvicorn

    global app
    app = create_app(session)
    print(f"neurochat {__version__} — http://{host}:{port}")
    print(f"  session artifacts: {app.state.session.workdir}")
    if not app.state.agent.available:
        print("  chat disabled (no ANTHROPIC_API_KEY); every deterministic control still works")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
