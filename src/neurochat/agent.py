"""The chat agent for the standalone web app (consumption mode 1).

Same ten tools as the MCP server, same rules, different transport. The agent loop
runs server-side and streams events to the browser so the interface can honour the
two failure modes NLI4VolVis reported:

* **Latency.** Tool calls execute the moment the model emits them, and each one
  pushes its viewer command over the WebSocket immediately — so the crosshair moves
  and the layers restyle *before* the explanatory prose finishes streaming. The
  deterministic paths (clicking a region, clicking a table row) never reach this
  module at all.
* **Terminology.** ``load_atlas`` and ``list_regions`` put the atlas's real label
  strings into the conversation, so the model matches against vocabulary that
  exists instead of recalling names.

Out-of-scope requests are classified by :mod:`neurochat.scope` *before* the model
is called, so a refusal is a property of the surface rather than a matter of
whether the system prompt happened to win that turn.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

from . import tools as tool_module
from .scope import check_scope, record_refusal
from .session import Session

MODEL = os.environ.get("NEUROCHAT_MODEL", "claude-opus-5")
MAX_TOKENS = 16000
MAX_TURNS = 12

SYSTEM_PROMPT = """\
You are neurochat, an assistant for exploring already-preprocessed volumetric \
neuroimaging data (NIfTI: MRI, PET). Every action you take also emits reproducible \
nilearn code into a script the user can export and re-run.

Rules you do not get to weigh against helpfulness:

1. NEVER state an anatomical coordinate from your own knowledge. Every location comes \
from navigate() or list_regions(), resolved against the loaded atlas. If you are about \
to write "the left hippocampus is around -26, -24, -14", call the tool instead — your \
recollection of that number is the exact failure this tool exists to prevent.
2. ALWAYS state the space with any location: MNI152NLin6Asym, MNI152NLin2009cAsym, \
native, or voxel[i,j,k]. The tools return it; pass it on.
3. A region name that does not match returns suggestions. Present them and ASK. Do not \
pick one, even when the intended region seems obvious.
4. Report exclusions. When roi_stats says it dropped NaN voxels, say so in your answer. \
A mean quoted without its exclusion count is how a wrong number reaches a paper.
5. Refuse, don't attempt: no statistical inference (t-tests, p-values, thresholded \
significance maps), no preprocessing (DICOM conversion, recon-all, fMRIPrep, motion \
correction), no running arbitrary code. Name the tool that does the job instead — \
nilearn.glm or FSL randomise for stats, fMRIPrep or dcm2niix for preprocessing.
6. Make no clinical or diagnostic claim, in any wording, ever.

Be brief. The user is looking at the viewer and the live script; you are the narration, \
not the output. Lead with what you found or did, then the caveats that change what they \
would do next. Do not re-describe what the script pane already shows."""


def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


_STR = {"type": "string"}
_NUM = {"type": "number"}
_BOOL = {"type": "boolean"}

ANTHROPIC_TOOLS: list[dict] = [
    _schema(
        "load_volume",
        "Load an already-preprocessed NIfTI volume (MRI, PET). Returns shape, voxel size, "
        "affine summary, detected space and how it was detected, value range and NaN count. "
        "No preprocessing is performed.",
        {
            "path": {**_STR, "description": "Path to a .nii or .nii.gz file."},
            "name": {**_STR, "description": "Short handle for later calls."},
            "space": {
                **_STR,
                "description": "Assert the template space when the header records none, "
                "e.g. 'MNI152NLin6Asym'. Recorded as the caller's assertion.",
            },
        },
        ["path"],
    ),
    _schema(
        "load_atlas",
        "Load an atlas and return its full label list. Match user phrasing against these "
        "strings, never against your own knowledge of neuroanatomy.",
        {
            "atlas_name": {
                **_STR,
                "description": "demo-16, harvard-oxford-sub, harvard-oxford-cort, aal, or "
                "schaefer-100. Non-bundled atlases are fetched once and cached.",
            }
        },
        ["atlas_name"],
    ),
    _schema(
        "list_regions",
        "List or search the loaded atlas's regions with voxel counts and centroids. Rows are "
        "clickable in the UI and navigate with no further model call.",
        {
            "query": {**_STR, "description": "Substring or fuzzy filter, e.g. 'hippocampus'."},
            "limit": {"type": "integer", "description": "Max rows (capped at 200)."},
        },
    ),
    _schema(
        "navigate",
        "Move the crosshair to a named atlas region or to explicit coordinates. Exactly one of "
        "region_label or coords. A near-miss on region_label returns suggestions and moves "
        "nothing.",
        {
            "region_label": {**_STR, "description": "Label from the loaded atlas."},
            "coords": {
                "type": "array",
                "items": _NUM,
                "description": "[x, y, z] in millimetres. Only when the user supplies numbers.",
            },
            "space": {**_STR, "description": "Space for coords, e.g. 'MNI152NLin6Asym'."},
            "volume": {**_STR, "description": "Which loaded volume to interpret against."},
        },
    ),
    _schema(
        "set_display",
        "Set colormap, intensity window and opacity for one layer. Changes what is shown, "
        "not the data.",
        {
            "volume": {**_STR, "description": "Layer to restyle."},
            "colormap": {**_STR, "description": "gray, hot, cool, viridis, inferno, magma, jet…"},
            "min": {**_NUM, "description": "Low end of the display window."},
            "max": {**_NUM, "description": "High end of the display window."},
            "opacity": {**_NUM, "description": "0.0 to 1.0."},
        },
    ),
    _schema(
        "overlay",
        "Stack one loaded volume on top of another and return the layer order. Display only — "
        "no registration is performed.",
        {
            "volume": {**_STR, "description": "The layer to put on top."},
            "on_top_of": {**_STR, "description": "The base layer."},
            "opacity": {**_NUM, "description": "Opacity of the top layer."},
        },
        ["volume"],
    ),
    _schema(
        "roi_stats",
        "Descriptive statistics inside an atlas region or a mask file: n_voxels, mean, sd, "
        "median, min, max, plus an explicit count of every excluded voxel. No test, no p-value, "
        "no threshold. Report the exclusion counts in your answer.",
        {
            "volume": {**_STR, "description": "Volume to measure."},
            "region_label": {**_STR, "description": "Atlas region name."},
            "mask_path": {**_STR, "description": "Path to a binary mask NIfTI."},
            "exclude_zeros": {
                **_BOOL,
                "description": "Drop exactly-zero voxels. Off by default; zeros are counted "
                "and included, because treating zero as 'no data' is an assumption.",
            },
        },
    ),
    _schema(
        "compare_volumes",
        "Voxelwise difference or ratio of two loaded volumes; adds the result to the session. "
        "Arithmetic, not inference — the output is not a statistical map.",
        {
            "a": {**_STR, "description": "First volume; defines the output grid."},
            "b": {**_STR, "description": "Second volume; resampled onto a's grid if needed."},
            "method": {"type": "string", "enum": ["difference", "ratio"]},
            "name": {**_STR, "description": "Handle for the result volume."},
        },
        ["a", "b"],
    ),
    _schema(
        "screenshot",
        "Capture the current view as a downscaled PNG and return its path. Uses the live "
        "Niivue canvas when a browser is attached, otherwise renders server-side with nilearn.",
        {"filename": {**_STR, "description": "Optional name for the PNG."}},
    ),
    _schema(
        "export_script",
        "Write the whole session as a standalone runnable .py and return its path. This is the "
        "deliverable: it needs only numpy, nibabel and nilearn, and re-running it reproduces "
        "the numbers reported in this conversation.",
        {"path": {**_STR, "description": "Where to write the script."}},
        ["path"],
    ),
]

assert {t["name"] for t in ANTHROPIC_TOOLS} == set(tool_module.TOOLS), (
    "the chat tool schemas and the tool surface have drifted apart"
)


class ChatAgent:
    """Runs the tool-use loop and yields events for the UI to render as they happen."""

    def __init__(self, session: Session, api_key: str | None = None):
        self.session = session
        self.history: list[dict] = []
        self._client = None
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # -- the loop ---------------------------------------------------------

    def run(self, user_message: str) -> Iterator[dict]:
        """Yield ``{"type": ...}`` events: text deltas, tool calls, results, errors."""
        refusal = check_scope(user_message)
        if refusal is not None:
            record_refusal(self.session, refusal)
            yield {"type": "refusal", **refusal.to_dict()}
            yield {"type": "script", "script": self.session.script.render_live()}
            yield {"type": "done"}
            return

        if not self.available:
            yield {
                "type": "error",
                "message": (
                    "No ANTHROPIC_API_KEY is set, so the chat pane is disabled. Everything "
                    "else still works: click a region in the atlas panel to navigate, use the "
                    "controls to restyle layers, and the script pane records it all. Set the "
                    "environment variable and restart to enable chat."
                ),
            }
            yield {"type": "done"}
            return

        self.history.append({"role": "user", "content": user_message})

        for _ in range(MAX_TURNS):
            try:
                message = yield from self._one_turn()
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
                yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                yield {"type": "done"}
                return

            if message.stop_reason == "refusal":
                yield {
                    "type": "error",
                    "message": (
                        "The model declined this request. If it was a legitimate "
                        "neuroimaging question, rephrasing usually helps."
                    ),
                }
                break

            self.history.append({"role": "assistant", "content": message.content})
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if not tool_uses:
                break

            results = []
            for block in tool_uses:
                yield {"type": "tool_use", "tool": block.name, "args": block.input}
                payload = self._execute(block.name, block.input)
                yield {
                    "type": "tool_result",
                    "tool": block.name,
                    "ok": payload.get("ok", False),
                    "result": payload,
                }
                yield {"type": "script", "script": self.session.script.render_live()}
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(payload, default=str),
                        "is_error": not payload.get("ok", False),
                    }
                )
            self.history.append({"role": "user", "content": results})
        else:
            yield {
                "type": "error",
                "message": f"Stopped after {MAX_TURNS} tool-use turns without settling.",
            }

        yield {"type": "state", "state": self.session.state()}
        yield {"type": "done"}

    def _one_turn(self):
        """Stream one assistant turn, yielding text deltas as they arrive."""
        self.session.llm_call_count += 1
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "tools": ANTHROPIC_TOOLS,
            "messages": self.history,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        client = self.client()

        # Opt into server-side refusal fallbacks. The safety classifiers on the
        # frontier models occasionally decline benign neuro/bio-adjacent wording, and
        # a declined request otherwise just stops. Older SDKs reject the parameter,
        # so fall back to the plain endpoint rather than failing the turn.
        try:
            stream = client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        except TypeError:
            stream = client.messages.stream(**kwargs)

        with stream as streaming:
            for event in streaming:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield {"type": "text", "text": event.delta.text}
            return streaming.get_final_message()

    def _execute(self, name: str, arguments: dict) -> dict:
        clean = {k: v for k, v in (arguments or {}).items() if v is not None}
        try:
            return tool_module.call(self.session, name, **clean)
        except KeyError as exc:
            return {"ok": False, "error": "UnknownTool", "message": str(exc)}
        except TypeError as exc:
            return {"ok": False, "error": "BadArguments", "message": str(exc)}
