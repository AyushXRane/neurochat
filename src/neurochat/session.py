"""Session state: what is loaded, what the viewer shows, what was done, and why.

Also holds the two things borrowed straight from neuroglancer-chat: a bounded ring
buffer of full tool traces (retrievable at ``/debug/tool_trace``), and cached result
tables whose rows carry coordinates, so clicking a row navigates the viewer with no
LLM call at all (R5).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .atlas import AtlasTable
from .errors import VolumeNotFoundError
from .script import ScriptAccumulator
from .spaces import SpaceInfo, affine_summary, detect_space, read_units
from .viewer import ViewerBridge

MAX_TRACES = 50
MAX_TABLES = 20


def default_workdir(session_id: str) -> Path:
    """Where derived volumes and screenshots live.

    Deliberately NOT a ``TemporaryDirectory``. An exported script references the
    difference maps and figures it produced; if those paths evaporate when the process
    exits, the script is broken the moment the session ends — which is the one thing
    R2 promises will not happen. Artifacts persist under ~/.neurochat/sessions and are
    the user's to delete.
    """
    import os

    root = os.environ.get("NEUROCHAT_WORKDIR")
    base = Path(root).expanduser() if root else Path.home() / ".neurochat" / "sessions"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"{stamp}-{session_id}"


@dataclass
class LoadedVolume:
    """A NIfTI in the session, described but not held in memory as an array."""

    name: str
    path: str
    space: SpaceInfo
    shape: tuple[int, ...]
    voxel_size_mm: tuple[float, ...]
    dtype: str
    affine: dict
    summary: dict
    #: What the numbers mean, from the BIDS sidecar. ``None`` when nothing declared
    #: them — which is the common case, and worth saying rather than glossing over.
    units: str | None = None
    modality: str | None = None
    loaded_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    )

    def image(self):
        import nibabel as nib

        return nib.load(self.path)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "shape": list(self.shape),
            "voxel_size_mm": list(self.voxel_size_mm),
            "dtype": self.dtype,
            "affine": self.affine,
            "space": self.space.to_dict(),
            "units": self.units,
            "modality": self.modality,
            "values": self.summary,
        }


@dataclass
class ResultTable:
    """A small table the UI can render and click. Rows carry coordinates."""

    table_id: str
    tool: str
    columns: list[str]
    rows: list[dict]
    created_at: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "tool": self.tool,
            "columns": self.columns,
            "rows": self.rows,
            "created_at": self.created_at,
        }


class Session:
    """Everything one conversation knows about."""

    def __init__(self, name: str = "session", workdir: str | Path | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        if workdir is None:
            workdir = default_workdir(self.id)
        self.workdir = Path(workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.volumes: dict[str, LoadedVolume] = {}
        self.atlas: AtlasTable | None = None
        self.script = ScriptAccumulator(session_id=self.id)
        self.viewer = ViewerBridge()
        self.traces: deque[dict] = deque(maxlen=MAX_TRACES)
        self.tables: dict[str, ResultTable] = {}
        self._table_order: deque[str] = deque(maxlen=MAX_TABLES)
        self.last_volume: str | None = None

        #: Counts calls that went through a language model. The no-LLM path (R5)
        #: asserts against this in tests: deterministic actions must not move it.
        self.llm_call_count = 0

    # -- volumes ---------------------------------------------------------

    def add_volume(self, path: str | Path, name: str | None = None, space: str | None = None) -> LoadedVolume:
        import nibabel as nib

        from . import kernel

        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"No such volume: {path}")
        img = nib.load(str(path))
        name = name or self._unique_name(path)
        volume = LoadedVolume(
            name=name,
            path=str(path),
            space=detect_space(img, path, override=space),
            shape=tuple(int(s) for s in img.shape),
            voxel_size_mm=tuple(round(float(z), 4) for z in img.header.get_zooms()[:3]),
            dtype=str(img.get_data_dtype()),
            affine=affine_summary(img.affine),
            summary=kernel.summarize_volume(img),
            **dict(zip(("units", "modality"), read_units(path))),
        )
        self.volumes[name] = volume
        self.last_volume = name
        return volume

    def _unique_name(self, path: Path) -> str:
        stem = path.name
        for suffix in (".nii.gz", ".nii", ".mgz", ".mgh", ".nrrd"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        candidate, counter = stem, 2
        while candidate in self.volumes:
            candidate = f"{stem}_{counter}"
            counter += 1
        return candidate

    def volume(self, name: str | None) -> LoadedVolume:
        if name is None:
            if self.last_volume is None:
                raise VolumeNotFoundError(
                    "No volume loaded. Call load_volume(path) first."
                )
            return self.volumes[self.last_volume]
        if name not in self.volumes:
            known = ", ".join(sorted(self.volumes)) or "(none loaded)"
            raise VolumeNotFoundError(
                f"No volume named {name!r} in this session. Loaded volumes: {known}.",
                loaded=sorted(self.volumes),
            )
        return self.volumes[name]

    # -- traces and tables ------------------------------------------------

    def record_trace(self, tool: str, args: dict, result: dict, error: str | None = None) -> dict:
        """Store the full trace; return the compact fragment that rides along in chat."""
        fragment = {
            "tool": tool,
            "args": sorted(k for k, v in args.items() if v is not None),
            "results": sorted(k for k in result if k not in ("tool_trace", "code")),
        }
        if error:
            fragment["error"] = error
        self.traces.append(
            {
                "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "tool": tool,
                "args": {k: _brief(v) for k, v in args.items()},
                "result": {k: _brief(v) for k, v in result.items()},
                "error": error,
            }
        )
        return fragment

    def recent_traces(self, n: int = 5) -> list[dict]:
        return list(self.traces)[-n:]

    def add_table(self, tool: str, columns: list[str], rows: list[dict]) -> ResultTable:
        table = ResultTable(
            table_id=f"{tool}-{uuid.uuid4().hex[:6]}", tool=tool, columns=columns, rows=rows
        )
        if len(self._table_order) == self._table_order.maxlen and self._table_order:
            self.tables.pop(self._table_order[0], None)
        self.tables[table.table_id] = table
        self._table_order.append(table.table_id)
        return table

    # -- housekeeping ------------------------------------------------------

    def temp_path(self, filename: str) -> Path:
        return self.workdir / filename

    def state(self) -> dict:
        return {
            "session_id": self.id,
            "created_at": self.created_at,
            "volumes": {name: vol.to_dict() for name, vol in self.volumes.items()},
            "atlas": (
                {
                    "atlas_id": self.atlas.atlas_id,
                    "space": self.atlas.space,
                    "n_regions": len(self.atlas.regions),
                }
                if self.atlas
                else None
            ),
            "viewer": self.viewer.state.to_dict(),
            "script_steps": len(self.script.steps),
            "llm_call_count": self.llm_call_count,
        }

    def close(self) -> None:
        """Sessions hold no OS resources; artifacts are kept on purpose."""
        return None


def _brief(value, limit: int = 400):
    """Keep stored traces small; they are a debugging aid, not a data store."""
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
        return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"
    if isinstance(value, dict):
        return {k: _brief(v, limit // 2) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        items = [_brief(v, limit // 2) for v in value[:10]]
        if len(value) > 10:
            items.append(f"… (+{len(value) - 10} more)")
        return items
    return value
