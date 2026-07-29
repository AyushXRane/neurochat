"""The library: many scans on disk, one at a time on screen, one region across all.

The engine always handled multiple volumes; these tests cover the part that was
missing — discovering them without loading them, and measuring one region across a
cohort as a single scripted action rather than N repeated ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurochat import library
from neurochat.errors import NoAtlasLoadedError, RegionNotFoundError, SpaceUnknownError
from neurochat.session import Session


@pytest.fixture
def lib_session(tmp_path):
    return Session(name="lib", workdir=tmp_path / "w")


class TestDiscovery:
    def test_finds_the_nifti_files_and_nothing_else(self, sample_data):
        result = library.scan_directory(sample_data)
        names = {e["name"] for e in result["entries"]}
        assert "phantom_t1" in names and "nospace_volume" in names
        # The .json sidecars sitting alongside must not become entries.
        assert not any(e["path"].endswith(".json") for e in result["entries"])
        assert result["n_found"] == len(result["entries"])

    def test_discovery_reads_headers_only(self, sample_data, monkeypatch):
        """Pointing at a folder of hundreds of scans must not load any voxel data."""
        import nibabel as nib

        loaded_arrays = []
        original = nib.Nifti1Image.get_fdata

        def spy(self, *args, **kwargs):
            loaded_arrays.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nib.Nifti1Image, "get_fdata", spy)
        library.scan_directory(sample_data)
        assert not loaded_arrays, "discovery pulled voxel data into memory"

    def test_entries_carry_space_provenance_and_suggestions(self, sample_data):
        entries = {e["name"]: e for e in library.scan_directory(sample_data)["entries"]}
        assert entries["phantom_t1"]["space_resolvable"] is True
        blocked = entries["nospace_volume"]
        assert blocked["space_resolvable"] is False
        assert blocked["suggested_space"] == "MNI152NLin6Asym"

    def test_a_mixed_library_says_how_many_are_blocked(self, sample_data):
        notes = " ".join(library.scan_directory(sample_data)["notes"])
        assert "no template space" in notes

    def test_missing_directory_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            library.scan_directory(tmp_path / "nope")

    def test_a_file_is_not_a_directory(self, sample_data):
        with pytest.raises(NotADirectoryError):
            library.scan_directory(sample_data / "phantom_t1.nii.gz")

    def test_unreadable_files_are_named_not_silently_dropped(self, tmp_path):
        (tmp_path / "broken.nii.gz").write_bytes(b"not a nifti")
        result = library.scan_directory(tmp_path)
        assert result["n_found"] == 0
        assert any("broken.nii.gz" in note for note in result["notes"])

    def test_names_are_unique_even_when_filenames_collide(self, tmp_path, sample_data):
        import shutil

        for sub in ("a", "b"):
            (tmp_path / sub).mkdir()
            shutil.copy(sample_data / "phantom_t1.nii.gz", tmp_path / sub / "scan.nii.gz")
        names = [e["name"] for e in library.scan_directory(tmp_path)["entries"]]
        assert len(names) == len(set(names)) == 2


class TestRegionAcrossLibrary:
    def _paths(self, sample_data):
        return [
            str(sample_data / f)
            for f in ("phantom_t1.nii.gz", "phantom_pet_baseline.nii.gz", "phantom_pet_followup.nii.gz")
        ]

    def test_requires_an_atlas(self, lib_session, sample_data):
        with pytest.raises(NoAtlasLoadedError):
            library.region_across_library(lib_session, self._paths(sample_data), "Left Deep Sphere")

    def test_measures_every_scan_and_builds_a_table(self, lib_session, sample_data, demo_atlas):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        result = library.region_across_library(
            lib_session, self._paths(sample_data), "Left Deep Sphere"
        )
        assert result["ok"] and result["n_measured"] == 3
        assert {r["scan"] for r in result["rows"]} == {
            "phantom_t1.nii.gz", "phantom_pet_baseline.nii.gz", "phantom_pet_followup.nii.gz"
        }
        for row in result["rows"]:
            assert row["n_voxels"] > 0 and row["mean"] is not None
            assert row["path"].endswith(".nii.gz")  # rows are clickable

    def test_a_typo_asks_rather_than_measuring_the_wrong_region(
        self, lib_session, sample_data, demo_atlas
    ):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        with pytest.raises(RegionNotFoundError) as excinfo:
            library.region_across_library(lib_session, self._paths(sample_data), "left deep spher")
        assert "Left Deep Sphere" in excinfo.value.suggestions

    def test_it_emits_one_script_step_not_one_per_scan(self, lib_session, sample_data, demo_atlas):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        before = len(lib_session.script.steps)
        library.region_across_library(lib_session, self._paths(sample_data), "Left Deep Sphere")
        assert len(lib_session.script.steps) == before + 1

        code = lib_session.script.steps[-1].code
        assert "for _path in COHORT" in code, "the emitted code should loop, not repeat"
        assert code.count("summarize_roi") == 1

    def test_the_emitted_loop_actually_runs_and_reproduces(
        self, lib_session, sample_data, tmp_path, demo_atlas
    ):
        """The cohort table is only worth having if its script reproduces it."""
        import json
        import subprocess
        import sys

        from neurochat import tools

        tools.load_volume(lib_session, path=str(sample_data / "phantom_t1.nii.gz"), name="t1")
        tools.load_atlas(lib_session, atlas_name="demo-16")
        result = library.region_across_library(
            lib_session, self._paths(sample_data), "Left Deep Sphere"
        )
        script = tmp_path / "cohort.py"
        tools.export_script(lib_session, path=str(script))

        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        rerun = json.loads(completed.stdout)[result["result_key"]]

        for row in result["rows"]:
            assert rerun[row["path"]]["mean"] == row["mean"]
            assert rerun[row["path"]]["n_voxels_in_mask"] == row["n_voxels"]

    def test_unresolvable_scans_are_skipped_and_listed_never_averaged_in(
        self, lib_session, sample_data, demo_atlas
    ):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        paths = self._paths(sample_data) + [str(sample_data / "nospace_volume.nii.gz")]
        result = library.region_across_library(lib_session, paths, "Left Deep Sphere")
        assert result["n_measured"] == 3 and result["n_skipped"] == 1
        assert "nospace_volume" in result["skipped"][0]["path"]
        assert not any("nospace" in row["scan"] for row in result["rows"])

    def test_a_wholly_unresolvable_cohort_refuses_with_the_fix(
        self, lib_session, sample_data, demo_atlas
    ):
        """The OASIS case: every scan carries sform_code=2, so the whole table refuses."""
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        with pytest.raises(SpaceUnknownError) as excinfo:
            library.region_across_library(
                lib_session, [str(sample_data / "nospace_volume.nii.gz")], "Left Deep Sphere"
            )
        message = excinfo.value.message
        assert "assume_space='MNI152NLin6Asym'" in message
        assert "your assertion" in message

    def test_asserting_the_space_lets_the_cohort_through(
        self, lib_session, sample_data, demo_atlas
    ):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        result = library.region_across_library(
            lib_session,
            [str(sample_data / "nospace_volume.nii.gz")],
            "Left Deep Sphere",
            assume_space="MNI152NLin6Asym",
        )
        assert result["ok"] and result["n_measured"] == 1

    def test_nan_exclusions_are_reported_per_scan_and_in_total(
        self, lib_session, sample_data, demo_atlas
    ):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        result = library.region_across_library(
            lib_session,
            [str(sample_data / "phantom_pet_baseline.nii.gz")],
            "Left Posterior Superior Block",
        )
        assert result["rows"][0]["excluded_nan"] > 0
        assert any("NaN voxels were excluded" in note for note in result["notes"])

    def test_it_says_out_loud_that_this_is_not_a_group_comparison(
        self, lib_session, sample_data, demo_atlas
    ):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        result = library.region_across_library(
            lib_session, self._paths(sample_data), "Left Deep Sphere"
        )
        assert any("not a group comparison" in note for note in result["notes"])

    def test_a_missing_file_is_skipped_not_fatal(self, lib_session, sample_data, demo_atlas):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        paths = self._paths(sample_data) + ["/nonexistent/scan.nii.gz"]
        result = library.region_across_library(lib_session, paths, "Left Deep Sphere")
        assert result["n_measured"] == 3
        assert any("not found" in s["reason"] for s in result["skipped"])

    def test_the_whole_thing_is_deterministic_no_model_involved(
        self, lib_session, sample_data, demo_atlas
    ):
        from neurochat import tools

        tools.load_atlas(lib_session, atlas_name="demo-16")
        library.region_across_library(lib_session, self._paths(sample_data), "Left Deep Sphere")
        assert lib_session.llm_call_count == 0


class TestCsv:
    def test_rows_render_as_csv_with_the_declared_columns(self):
        rows = [{"scan": "a.nii.gz", "mean": 1.5, "sd": None}, {"scan": "b.nii.gz", "mean": 2.0, "sd": 0.1}]
        text = library.rows_to_csv(rows, ["scan", "mean", "sd"])
        lines = text.strip().splitlines()
        assert lines[0] == "scan,mean,sd"
        assert lines[1] == "a.nii.gz,1.5,"  # None becomes empty, not "None"
        assert lines[2] == "b.nii.gz,2.0,0.1"
