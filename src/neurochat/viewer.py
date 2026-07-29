"""Viewer state, and the snapshot loop that lets the model see its own output.

Niivue renders client-side in WebGL, so the backend never sees the pixels. The
bridge here holds the *desired* viewer state, streams commands to whatever browser
clients are attached, and can ask a client for a `canvas.toDataURL()` snapshot.

Three properties matter:

* **Tools stay synchronous.** They run in a worker thread and block on a
  ``threading.Event`` while the event loop delivers the snapshot. No async colouring
  spreads through the tool surface.
* **Headless still works.** With no browser attached, ``screenshot()`` falls back to
  rendering server-side with nilearn. Phase 1 (MCP, no viewer) is useful on its own,
  and the response always says which renderer produced the image.
* **Images are downscaled before they go anywhere** (R4): 768px on the long edge,
  never full-resolution canvas data.
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_SNAPSHOT_EDGE = 768
SNAPSHOT_TIMEOUT_S = 8.0


@dataclass
class LayerState:
    """One volume in the viewer's layer stack, bottom-first."""

    name: str
    path: str
    colormap: str = "gray"
    cal_min: float | None = None
    cal_max: float | None = None
    opacity: float = 1.0
    visible: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ViewerState:
    layers: list[LayerState] = field(default_factory=list)
    crosshair_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    crosshair_space: str = "unknown"
    crosshair_label: str | None = None

    def to_dict(self) -> dict:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "crosshair_mm": list(self.crosshair_mm),
            "crosshair_space": self.crosshair_space,
            "crosshair_label": self.crosshair_label,
        }

    def find(self, name: str) -> LayerState | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None


class ViewerBridge:
    """Fan-out of viewer commands to browser clients, plus snapshot round-trips."""

    def __init__(self) -> None:
        self.state = ViewerState()
        self._lock = threading.Lock()
        self._revision = 0
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._pending: dict[str, threading.Event] = {}
        self._snapshots: dict[str, bytes] = {}
        self._client_count = 0
        self.command_log: list[dict] = []

    # -- client bookkeeping ----------------------------------------------

    @property
    def attached(self) -> bool:
        with self._lock:
            return self._client_count > 0

    def client_connected(self) -> None:
        with self._lock:
            self._client_count += 1

    def client_disconnected(self) -> None:
        with self._lock:
            self._client_count = max(0, self._client_count - 1)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    # -- command fan-out --------------------------------------------------

    def push(self, command: dict) -> None:
        """Record a viewer command and wake every attached client.

        Safe to call from a worker thread; waiters are signalled through their own
        event loops.
        """
        with self._lock:
            self._revision += 1
            command = {**command, "revision": self._revision}
            self.command_log.append(command)
            if len(self.command_log) > 200:
                del self.command_log[:-200]
            waiters = list(self._waiters)
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass  # loop closed; the client's own cleanup will drop it

    def register_waiter(self, loop: asyncio.AbstractEventLoop) -> asyncio.Event:
        event = asyncio.Event()
        with self._lock:
            self._waiters.append((loop, event))
        return event

    def unregister_waiter(self, event: asyncio.Event) -> None:
        with self._lock:
            self._waiters = [pair for pair in self._waiters if pair[1] is not event]

    def commands_since(self, revision: int) -> list[dict]:
        with self._lock:
            return [c for c in self.command_log if c["revision"] > revision]

    # -- snapshots --------------------------------------------------------

    def request_snapshot(self, timeout: float = SNAPSHOT_TIMEOUT_S) -> bytes | None:
        """Ask an attached client for a PNG of its canvas. Blocks; returns None on timeout."""
        if not self.attached:
            return None
        request_id = uuid.uuid4().hex[:12]
        done = threading.Event()
        with self._lock:
            self._pending[request_id] = done
        self.push({"type": "snapshot_request", "request_id": request_id})
        got = done.wait(timeout)
        with self._lock:
            self._pending.pop(request_id, None)
            data = self._snapshots.pop(request_id, None)
        return data if got else None

    def deliver_snapshot(self, request_id: str, data_url: str) -> None:
        """Called by the WebSocket handler when a client returns canvas pixels."""
        payload = data_url.split(",", 1)[-1]
        try:
            raw = base64.b64decode(payload)
        except (ValueError, TypeError):
            raw = b""
        with self._lock:
            self._snapshots[request_id] = raw
            event = self._pending.get(request_id)
        if event is not None:
            event.set()

    # -- state mutation ---------------------------------------------------

    def set_crosshair(self, coords, space: str, label: str | None = None) -> None:
        self.state.crosshair_mm = tuple(float(c) for c in coords)
        self.state.crosshair_space = space
        self.state.crosshair_label = label
        self.push(
            {
                "type": "navigate",
                "coords": list(self.state.crosshair_mm),
                "space": space,
                "label": label,
            }
        )

    def add_layer(self, layer: LayerState, on_top: bool = True) -> None:
        existing = self.state.find(layer.name)
        if existing is not None:
            self.state.layers.remove(existing)
        if on_top:
            self.state.layers.append(layer)
        else:
            self.state.layers.insert(0, layer)
        self.push({"type": "layers", "layers": self.state.to_dict()["layers"]})

    def update_layer(self, layer: LayerState) -> None:
        self.push({"type": "layers", "layers": self.state.to_dict()["layers"]})


def downscale_png(raw: bytes, max_edge: int = MAX_SNAPSHOT_EDGE) -> bytes:
    """Shrink a PNG so its long edge is at most ``max_edge`` pixels."""
    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    if max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.LANCZOS)
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def render_server_side(
    layers: list[LayerState],
    crosshair_mm,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Fallback renderer: draw the current layer stack with nilearn, headless.

    Used when no browser canvas is attached. It is not pixel-identical to Niivue —
    different renderer, different conventions — and callers are told which one ran.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from nilearn import plotting

    visible = [layer for layer in layers if layer.visible]
    if not visible:
        raise ValueError("Nothing to render: no visible layers in the viewer.")

    base, *overlays = visible
    cut_coords = tuple(float(c) for c in crosshair_mm)
    figure = plt.figure(figsize=(9, 3.2), dpi=100, facecolor="black")
    display = plotting.plot_anat(
        base.path,
        figure=figure,
        display_mode="ortho",
        cut_coords=cut_coords,
        title=title,
        annotate=True,
        black_bg=True,
        vmin=base.cal_min,
        vmax=base.cal_max,
        cmap=base.colormap if base.colormap != "gray" else "gray",
    )
    import inspect

    # nilearn renamed the opacity kwarg from `alpha` to `transparency`; support both
    # so the fallback renderer works across the versions our dependency range allows.
    opacity_kwarg = (
        "transparency"
        if "transparency" in inspect.signature(display.add_overlay).parameters
        else "alpha"
    )
    for layer in overlays:
        display.add_overlay(
            layer.path,
            cmap=layer.colormap if layer.colormap != "gray" else "hot",
            vmin=layer.cal_min,
            vmax=layer.cal_max,
            **{opacity_kwarg: layer.opacity},
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    display.savefig(str(out_path), dpi=100)
    display.close()
    plt.close(figure)

    out_path.write_bytes(downscale_png(out_path.read_bytes()))
    return out_path
