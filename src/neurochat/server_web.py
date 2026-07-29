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

from . import __version__, tools as tool_module
from .agent import ChatAgent
from .atlas import ATLAS_REGISTRY
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
