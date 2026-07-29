"""The seven acceptance tests from Section 8 of the spec. The build is done when these pass.

Each test is named for the spec bullet it enforces. They are deliberately end-to-end:
they call the real tools against real files, and the reproducibility test really does
shell out to a clean interpreter.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurochat import tools
from neurochat.scope import check_scope
from neurochat.session import Session
from neurochat.tools import MAX_PAYLOAD_BYTES

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "sample_data"


@pytest.fixture
def s(tmp_path):
    return Session(name="acceptance", workdir=tmp_path / "work")


# ---------------------------------------------------------------------------
# 1. Grounding
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAcceptance1Grounding:
    """navigate(region_label="left hippocampus") lands within 5mm of the atlas centroid.
    A typo returns a did-you-mean, not a guess."""

    def test_named_region_resolves_within_5mm_of_centroid(self, s, ho_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="mni")
        tools.load_atlas(s, atlas_name="harvard-oxford-sub")

        result = tools.navigate(s, region_label="left hippocampus")
        assert result["ok"], result.get("message")

        centroid = np.array(result["detail"]["centroid"])
        returned = np.array(result["coords"])
        assert np.linalg.norm(returned - centroid) < 5.0
        assert result["space"] == "MNI152NLin6Asym"
        assert result["label"] == "Left Hippocampus"

    def test_the_returned_point_is_inside_the_structure(self, s, ho_atlas):
        """A coordinate 'near' a region but outside it is a wrong answer, not a close one."""
        import nibabel as nib

        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="mni")
        tools.load_atlas(s, atlas_name="harvard-oxford-sub")
        result = tools.navigate(s, region_label="left hippocampus")

        maps = nib.load(s.atlas.maps_path)
        i, j, k = np.rint(
            (np.linalg.inv(maps.affine) @ np.append(result["coords"], 1.0))[:3]
        ).astype(int)
        assert np.asanyarray(maps.dataobj)[i, j, k] == result["detail"]["region_index"]

    def test_typo_returns_did_you_mean_and_moves_nothing(self, s, ho_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="mni")
        tools.load_atlas(s, atlas_name="harvard-oxford-sub")
        before = s.viewer.state.crosshair_mm

        result = tools.navigate(s, region_label="left hippocampos")
        assert result["ok"] is False
        assert result["error"] == "RegionNotFoundError"
        assert "Left Hippocampus" in result["suggestions"]
        assert s.viewer.state.crosshair_mm == before, "a failed lookup moved the crosshair"

    def test_the_model_cannot_smuggle_in_a_coordinate_as_a_region(self, s, demo_atlas):
        """Coordinates are only accepted through coords=, and the space is echoed back."""
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="mni")
        tools.load_atlas(s, atlas_name="demo-16")

        assert tools.navigate(s, region_label="-26, -24, -14")["ok"] is False
        explicit = tools.navigate(s, coords=[-26, -24, -14], space="MNI152NLin6Asym")
        assert explicit["ok"] and explicit["space"] == "MNI152NLin6Asym"
        assert explicit["detail"]["source"] == "supplied by the caller"


# ---------------------------------------------------------------------------
# 2. Space refusal
# ---------------------------------------------------------------------------


class TestAcceptance2SpaceRefusal:
    """A volume with no space metadata refuses named regions and says which metadata is missing."""

    def test_named_region_on_a_spaceless_volume_errors_specifically(self, s, demo_atlas):
        loaded = tools.load_volume(s, path=str(SAMPLE / "nospace_volume.nii.gz"), name="mystery")
        assert loaded["space"]["space"] == "unknown"
        assert loaded["space_resolvable"] is False
        tools.load_atlas(s, atlas_name="demo-16")

        result = tools.navigate(s, region_label="Left Deep Sphere", volume="mystery")
        assert result["ok"] is False
        assert result["error"] == "SpaceUnknownError"
        message = result["message"]
        assert "sform_code" in message and "qform_code" in message
        assert "space=" in message

    def test_roi_stats_refuses_the_same_way(self, s, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "nospace_volume.nii.gz"), name="mystery")
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.roi_stats(s, volume="mystery", region_label="Left Deep Sphere")
        assert result["ok"] is False and result["error"] == "SpaceUnknownError"

    def test_matching_grid_geometry_is_reported_but_never_used_to_decide(self, s, demo_atlas):
        """The spaceless sample sits on a known MNI grid. That must not be enough."""
        loaded = tools.load_volume(s, path=str(SAMPLE / "nospace_volume.nii.gz"), name="mystery")
        assert loaded["space"]["grid_hint"] is not None
        assert loaded["space"]["resolvable"] is False

    def test_the_caller_can_assert_the_space_and_it_is_recorded_as_theirs(self, s, demo_atlas):
        tools.load_volume(
            s, path=str(SAMPLE / "nospace_volume.nii.gz"), name="asserted", space="MNI152NLin6Asym"
        )
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.navigate(s, region_label="Left Deep Sphere", volume="asserted")
        assert result["ok"] is True
        assert s.volumes["asserted"].space.source == "user_override"


# ---------------------------------------------------------------------------
# 3. Reproducibility — the headline test
# ---------------------------------------------------------------------------


class TestAcceptance3Reproducibility:
    """A 10-turn session, exported and re-run from a clean interpreter, produces
    byte-identical roi_stats output."""

    def test_ten_turn_session_reruns_identically(self, s, tmp_path, demo_atlas):
        live: dict[str, dict] = {}

        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")            # 1
        tools.load_volume(s, path=str(SAMPLE / "phantom_pet_baseline.nii.gz"), name="pet")  # 2
        tools.load_atlas(s, atlas_name="demo-16")                                           # 3
        tools.navigate(s, region_label="Left Deep Sphere")                                  # 4

        r = tools.roi_stats(s, volume="pet", region_label="Left Deep Sphere")               # 5
        live[r["result_key"]] = r["stats"]
        r = tools.roi_stats(s, volume="pet", region_label="Midline Dorsal Cap")             # 6
        live[r["result_key"]] = r["stats"]

        tools.load_volume(s, path=str(SAMPLE / "phantom_pet_followup.nii.gz"), name="pet2")  # 7
        r = tools.compare_volumes(s, a="pet", b="pet2", method="difference")                # 8
        live[r["result_key"]] = r["summary"]
        diff_name = r["name"]

        r = tools.roi_stats(s, volume=diff_name, region_label="Left Posterior Superior Block")  # 9
        live[r["result_key"]] = r["stats"]
        r = tools.roi_stats(s, volume="pet", region_label="Left Deep Sphere", exclude_zeros=True)  # 10
        live[r["result_key"]] = r["stats"]

        script = tmp_path / "session.py"
        exported = tools.export_script(s, path=str(script))
        assert exported["ok"], exported.get("message")
        assert len(s.script.steps) >= 10

        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        assert completed.returncode == 0, completed.stderr[-3000:]
        rerun = json.loads(completed.stdout)

        for key, value in live.items():
            assert key in rerun, f"{key} missing from the re-run's output"
            assert json.dumps(rerun[key], sort_keys=True) == json.dumps(value, sort_keys=True), (
                f"{key} differs between the session and the re-run"
            )

    def test_the_script_stands_alone(self, s, tmp_path, demo_atlas):
        """It must not import neurochat — that is the difference between a record and a receipt."""
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
        tools.load_atlas(s, atlas_name="demo-16")
        tools.roi_stats(s, volume="t1", region_label="Left Deep Sphere")
        script = tmp_path / "standalone.py"
        tools.export_script(s, path=str(script))

        # Prose may mention neurochat; executable lines may not import it.
        code_lines = [
            line
            for line in script.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        offenders = [line for line in code_lines if "neurochat" in line and "import" in line]
        assert not offenders, f"exported script imports neurochat: {offenders}"
        assert "def summarize_roi" in script.read_text(), (
            "the compute kernel must be inlined, not imported"
        )

    def test_the_header_records_the_environment(self, s, tmp_path, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
        script = tmp_path / "hdr.py"
        tools.export_script(s, path=str(script))
        text = script.read_text()
        for marker in ("Session:", "Exported:", "nibabel", "nilearn", "numpy"):
            assert marker in text


# ---------------------------------------------------------------------------
# 4. No-LLM path
# ---------------------------------------------------------------------------


class TestAcceptance4NoLlmPath:
    """Clicking a region in the atlas panel navigates the viewer with zero API calls."""

    def test_deterministic_navigation_does_not_touch_the_counter(self, s, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
        tools.load_atlas(s, atlas_name="demo-16")
        assert s.llm_call_count == 0

        # Exactly what the UI's click handler posts to /api/tool.
        for _ in range(5):
            result = tools.call(s, "navigate", region_label="Left Deep Sphere")
            assert result["ok"]
        tools.call(s, "set_display", volume="t1", colormap="viridis")
        tools.call(s, "list_regions", query="sphere")

        assert s.llm_call_count == 0, "a deterministic action went through the model"
        assert s.viewer.state.crosshair_label == "Left Deep Sphere"

    def test_the_viewer_actually_received_the_commands(self, s, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
        tools.load_atlas(s, atlas_name="demo-16")
        before = len(s.viewer.command_log)
        tools.call(s, "navigate", region_label="Right Deep Sphere")
        pushed = s.viewer.command_log[before:]
        assert any(c["type"] == "navigate" for c in pushed)
        assert s.llm_call_count == 0

    def test_result_rows_carry_coordinates_so_they_can_be_clicked(self, s, demo_atlas):
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.list_regions(s, query="sphere")
        table = s.tables[result["table_id"]]
        assert table.rows and all(len(row["centroid"]) == 3 for row in table.rows)
        assert all(row["space"] == "MNI152NLin6Asym" for row in table.rows)


# ---------------------------------------------------------------------------
# 5. Payload discipline
# ---------------------------------------------------------------------------


class TestAcceptance5PayloadDiscipline:
    """No tool response exceeds 50KB."""

    def _size(self, payload) -> int:
        return len(json.dumps(payload, default=str).encode("utf-8"))

    def test_every_tool_stays_under_the_budget(self, s, tmp_path, demo_atlas):
        calls = [
            ("load_volume", {"path": str(SAMPLE / "phantom_t1.nii.gz"), "name": "t1"}),
            ("load_volume", {"path": str(SAMPLE / "phantom_pet_baseline.nii.gz"), "name": "pet"}),
            ("load_atlas", {"atlas_name": "demo-16"}),
            ("list_regions", {}),
            ("list_regions", {"query": "block", "limit": 200}),
            ("navigate", {"region_label": "Left Deep Sphere"}),
            ("set_display", {"volume": "pet", "colormap": "hot", "min": 0.5, "max": 2.0}),
            ("overlay", {"volume": "pet", "on_top_of": "t1", "opacity": 0.6}),
            ("roi_stats", {"volume": "pet", "region_label": "Left Deep Sphere"}),
            ("compare_volumes", {"a": "pet", "b": "t1", "method": "difference"}),
            ("screenshot", {}),
            ("export_script", {"path": str(tmp_path / "out.py")}),
        ]
        for name, arguments in calls:
            payload = tools.call(s, name, **arguments)
            assert payload.get("ok"), f"{name}: {payload.get('message')}"
            size = self._size(payload)
            assert size <= MAX_PAYLOAD_BYTES, f"{name} returned {size} bytes"

    def test_a_large_atlas_label_list_is_truncated_not_streamed(self, s, monkeypatch):
        """Guard the one tool that could blow the budget: load_atlas on a big parcellation."""
        from neurochat import atlas as atlas_module

        table = atlas_module.load_atlas_table("demo-16")
        many = list(table.regions) * 40  # 640 fake regions
        object.__setattr__(table, "regions", many)
        monkeypatch.setattr(atlas_module, "load_atlas_table", lambda name, use_cache=True: table)
        monkeypatch.setattr(tools, "load_atlas_table", lambda name, use_cache=True: table)

        payload = tools.load_atlas(s, atlas_name="demo-16")
        assert payload["ok"] and payload["labels_truncated"] is True
        assert len(payload["labels"]) == 250
        assert self._size(payload) <= MAX_PAYLOAD_BYTES
        assert any("list_regions" in note for note in payload["notes"])

    def test_screenshots_are_paths_not_pixels(self, s, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
        payload = tools.screenshot(s)
        assert payload["ok"]
        blob = json.dumps(payload)
        assert "base64" not in blob and "data:image" not in blob
        assert Path(payload["path"]).exists()

    def test_the_rendered_image_is_downscaled(self, s, demo_atlas):
        from PIL import Image

        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
        payload = tools.screenshot(s)
        with Image.open(payload["path"]) as image:
            assert max(image.size) <= 768


# ---------------------------------------------------------------------------
# 6. Honest stats
# ---------------------------------------------------------------------------


class TestAcceptance6HonestStats:
    """roi_stats on a volume containing NaNs reports the exclusion count."""

    def test_nan_voxels_are_counted_and_surfaced(self, s, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_pet_baseline.nii.gz"), name="pet")
        tools.load_atlas(s, atlas_name="demo-16")

        # This region overlaps the deliberate NaN dropout slab in the phantom.
        result = tools.roi_stats(s, volume="pet", region_label="Left Posterior Superior Block")
        assert result["ok"]
        stats = result["stats"]
        assert stats["exclusions"]["nan"] > 0
        assert stats["n_voxels_used"] == stats["n_voxels_in_mask"] - stats["exclusions"]["nan"]
        assert any("NaN" in note for note in result["notes"])
        assert np.isfinite(stats["mean"])

    def test_zeros_are_included_by_default_and_counted(self, s, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_pet_baseline.nii.gz"), name="pet")
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.roi_stats(s, volume="pet", region_label="Midline Ventral Core")
        exclusions = result["stats"]["exclusions"]
        assert exclusions["zeros_included"] is True
        assert exclusions["zero"] == 0
        assert "n_zero_in_mask" in exclusions

    def test_excluding_zeros_is_opt_in_and_changes_the_reported_counts(self, s, demo_atlas):
        tools.load_volume(s, path=str(SAMPLE / "phantom_pet_baseline.nii.gz"), name="pet")
        tools.load_atlas(s, atlas_name="demo-16")
        kept = tools.roi_stats(s, volume="pet", region_label="Midline Ventral Core")
        dropped = tools.roi_stats(
            s, volume="pet", region_label="Midline Ventral Core", exclude_zeros=True
        )
        assert dropped["stats"]["exclusions"]["zeros_included"] is False
        assert dropped["stats"]["n_voxels_used"] <= kept["stats"]["n_voxels_used"]

    def test_an_all_nan_region_reports_none_rather_than_a_number(self, s, tmp_path, demo_atlas):
        import nibabel as nib

        base = nib.load(str(SAMPLE / "phantom_pet_baseline.nii.gz"))
        data = np.full(base.shape, np.nan, dtype=np.float32)
        path = tmp_path / "all_nan.nii.gz"
        nib.save(nib.Nifti1Image(data, base.affine, base.header), str(path))

        tools.load_volume(s, path=str(path), name="empty", space="MNI152NLin6Asym")
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.roi_stats(s, volume="empty", region_label="Left Deep Sphere")
        assert result["ok"]
        assert result["stats"]["mean"] is None
        assert result["stats"]["n_voxels_used"] == 0
        assert result["stats"]["exclusions"]["nan"] > 0

    def test_a_resampled_mask_says_so(self, s, ho_atlas):
        """The 2mm atlas onto a 4mm volume shifts boundaries. Say it, don't hide it."""
        tools.load_volume(s, path=str(SAMPLE / "phantom_t1.nii.gz"), name="t1")
        tools.load_atlas(s, atlas_name="harvard-oxford-sub")
        result = tools.roi_stats(s, volume="t1", region_label="Left Hippocampus")
        assert result["ok"]
        assert any("resampled" in note or "nearest-neighbour" in note for note in result["notes"])


# ---------------------------------------------------------------------------
# 7. Refusal
# ---------------------------------------------------------------------------


class TestAcceptance7Refusal:
    """Asking for a t-test returns a scope refusal, not an attempt."""

    def test_a_t_test_request_is_refused_with_a_pointer_to_real_tools(self):
        refusal = check_scope("can you run a t-test between the two groups?")
        assert refusal is not None
        assert refusal.category == "statistics"
        assert any("nilearn.glm" in item or "randomise" in item for item in refusal.use_instead)

    @pytest.mark.parametrize(
        "request_text,category",
        [
            ("compute a p-value for this contrast", "statistics"),
            ("is this difference statistically significant?", "statistics"),
            ("threshold the map at FDR q<0.05", "statistics"),
            ("run recon-all on this subject", "preprocessing"),
            ("convert these DICOMs to NIfTI", "preprocessing"),
            ("please smooth the image with a 6mm kernel", "preprocessing"),
            ("execute this python code for me", "code_execution"),
            ("does this patient have Alzheimer's?", "clinical"),
            ("is this scan normal?", "clinical"),
        ],
    )
    def test_out_of_scope_requests_are_classified(self, request_text, category):
        refusal = check_scope(request_text)
        assert refusal is not None, f"{request_text!r} was not refused"
        assert refusal.category == category

    @pytest.mark.parametrize(
        "request_text",
        [
            "what is the mean uptake in the left hippocampus?",
            "show me the amygdala",
            "overlay the PET on the T1",
            "how many voxels are in the left putamen?",
            "compare the baseline and follow-up scans",
            "export the script",
        ],
    )
    def test_in_scope_requests_are_not_refused(self, request_text):
        assert check_scope(request_text) is None, f"{request_text!r} was refused in error"

    def test_there_is_no_tool_to_do_it_with_anyway(self):
        """Defence in depth: even if the guard were bypassed, the surface has no stats tool."""
        assert len(tools.TOOLS) == 10
        for forbidden in ("ttest", "t_test", "glm", "threshold", "preprocess", "recon_all", "exec"):
            assert forbidden not in tools.TOOLS

    def test_a_refusal_is_recorded_in_the_script_as_a_comment(self, s):
        from neurochat.scope import record_refusal

        refusal = check_scope("run a t-test on these two groups")
        record_refusal(s, refusal)
        rendered = s.script.render_live()
        assert "NOT RUN" in rendered
        assert "SecondLevelModel" in rendered
        for line in rendered.splitlines():
            if "SecondLevelModel" in line:
                assert line.lstrip().startswith("#"), "suggested code must be commented out"
