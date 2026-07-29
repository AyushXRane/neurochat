"""Phase 1 success criterion: a conversation over MCP that ends in a correct runnable .py.

These drive the server over a real stdio transport, the same way Claude Desktop does,
rather than calling the Python functions directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.anyio

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "sample_data"


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _client():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(os.environ)
    env.pop("NEUROCHAT_BACKEND", None)
    env["PYTHONPATH"] = str(REPO / "src")
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "neurochat.server_mcp"], env=env
    )
    return stdio_client(params), ClientSession


def _payload(result) -> dict:
    """Pull the JSON body out of a tool result, whatever content shape it arrived in."""
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured:
        content = structured
        if set(content.keys()) == {"result"}:
            content = content["result"]
        if isinstance(content, dict) and content:
            return content
    for item in result.content:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    raise AssertionError(f"No JSON payload in tool result: {result}")


class TestMcpSurface:
    async def test_ten_tools_and_the_rules_are_advertised(self):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        transport, _ = await _client()
        async with transport as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                listing = await session.list_tools()

        names = {t.name for t in listing.tools}
        assert names == {
            "load_volume", "load_atlas", "list_regions", "navigate", "set_display",
            "overlay", "roi_stats", "compare_volumes", "screenshot", "export_script",
        }
        instructions = init.instructions or ""
        assert "NEVER state an anatomical coordinate" in instructions
        assert "Refuse, don't attempt" in instructions

    async def test_a_ten_turn_session_produces_a_runnable_script(self, tmp_path):
        from mcp import ClientSession

        transport, _ = await _client()
        script_path = tmp_path / "session.py"

        async with transport as (read, write):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()

                async def call(tool, **args):
                    return _payload(await mcp.call_tool(tool, args))

                assert call  # keep the helper obviously in use

                volume = await call(
                    "load_volume", path=str(SAMPLE / "phantom_pet_baseline.nii.gz"), name="pet"
                )
                assert volume["ok"] and volume["space"]["space"] == "MNI152NLin6Asym"
                assert volume["values"]["n_nan"] > 0

                await call("load_volume", path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
                atlas = await call("load_atlas", atlas_name="demo-16")
                assert atlas["n_regions"] == 16
                assert "Left Deep Sphere" in atlas["labels"]

                regions = await call("list_regions", query="sphere")
                assert regions["n_matched"] == 2

                where = await call("navigate", region_label="Left Deep Sphere")
                assert where["ok"] and where["space"] == "MNI152NLin6Asym"

                miss = await call("navigate", region_label="left deep sfere")
                assert miss["ok"] is False
                assert "Left Deep Sphere" in miss["suggestions"]

                await call("set_display", volume="pet", colormap="hot", min=0.5, max=2.2)
                await call("overlay", volume="pet", on_top_of="t1", opacity=0.6)

                stats = await call("roi_stats", volume="pet", region_label="Left Deep Sphere")
                assert stats["ok"] and stats["stats"]["n_voxels_in_mask"] > 0

                await call("load_volume", path=str(SAMPLE / "phantom_pet_followup.nii.gz"), name="pet2")
                diff = await call("compare_volumes", a="pet", b="pet2", method="difference")
                assert diff["ok"]

                exported = await call("export_script", path=str(script_path))
                assert exported["ok"]

        assert script_path.exists()
        text = script_path.read_text()
        assert "summarize_roi" in text and "nib.load" in text

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        results = json.loads(completed.stdout)
        assert any(key.endswith("roi_stats") for key in results)

    async def test_screenshot_returns_an_image_and_a_path(self, tmp_path):
        from mcp import ClientSession

        transport, _ = await _client()
        async with transport as (read, write):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()
                await mcp.call_tool(
                    "load_volume", {"path": str(SAMPLE / "phantom_t1.nii.gz"), "name": "t1"}
                )
                result = await mcp.call_tool("screenshot", {})

        kinds = [item.type for item in result.content]
        assert "image" in kinds, f"expected an image in the response, got {kinds}"
        body = _payload(result)
        assert body["renderer"] == "nilearn-fallback"
        assert Path(body["path"]).exists()

    async def test_resources_expose_script_and_traces(self):
        from mcp import ClientSession

        transport, _ = await _client()
        async with transport as (read, write):
            async with ClientSession(read, write) as mcp:
                await mcp.initialize()
                listing = await mcp.list_resources()
                uris = {str(r.uri) for r in listing.resources}
                assert "neurochat://session/script" in uris
                assert "neurochat://debug/tool_trace" in uris

                await mcp.call_tool(
                    "load_volume", {"path": str(SAMPLE / "phantom_t1.nii.gz"), "name": "t1"}
                )
                script = await mcp.read_resource("neurochat://session/script")
                traces = await mcp.read_resource("neurochat://debug/tool_trace")

        assert "nib.load" in script.contents[0].text
        assert "load_volume" in traces.contents[0].text
