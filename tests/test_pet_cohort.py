"""The real PET cohort, and the affine repair it needs.

Until this existed, every claim about neurochat working on PET was untested — all
verification ran on structural MRI. These tests pin the one place the project
knowingly rewrites a header, which is exactly the kind of thing that should never
drift silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurochat import library

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def cohort():
    try:
        return library.pet_cohort(n_subjects=3)
    except Exception as exc:  # noqa: BLE001 - offline is a skip, not a failure
        pytest.skip(f"PET cohort unavailable ({type(exc).__name__}: {exc})")


class TestPetCohort:
    def test_it_downloads_real_pet_and_lands_in_a_resolvable_space(self, cohort):
        assert cohort["n_found"] >= 1
        entry = cohort["entries"][0]
        assert tuple(entry["shape"]) == (181, 217, 181)
        assert entry["voxel_size_mm"][0] == 1.0
        assert entry["space_resolvable"] is True, "the repaired affine should resolve"

    def test_it_says_it_repaired_the_affine(self, cohort):
        """A rewritten header must never be silent — including on cached runs.

        This caught a real bug: the disclosure was only emitted when a download
        actually happened, so the first run told you and every run afterwards did not.
        """
        notes = " ".join(cohort["notes"]).lower()
        assert "affine repaired" in notes
        assert "original.nii.gz" in notes, "must say where the untouched file is"
        assert "nan" in notes

    def test_the_disclosure_survives_a_cached_run(self):
        """Call it twice: the second call downloads nothing and must still disclose."""
        again = library.pet_cohort(n_subjects=1)
        assert any("affine repaired" in note.lower() for note in again["notes"])

    def test_the_original_download_is_kept(self, cohort):
        from pathlib import Path

        originals = list(Path(cohort["root"]).glob("*_original.nii.gz"))
        assert originals, "the untouched download should be kept alongside the repair"

    def test_regions_land_inside_the_brain_after_repair(self, cohort, ho_atlas):
        """The evidence the repair is correct, asserted rather than asserted-about.

        With the shipped affine, 0% of every atlas region falls inside the brain.
        With the repair, a substantial majority does — short of 100% only because
        PET's axial field of view clips the inferior brain.
        """
        import nibabel as nib

        from neurochat import kernel

        path = cohort["entries"][0]["path"]
        img = nib.load(path)
        data = np.asanyarray(img.dataobj)
        brain = np.isfinite(data)  # background is NaN in these maps

        region = ho_atlas.resolve("Left Hippocampus")
        mask = kernel.region_mask(ho_atlas.maps_path, region.index, target_img=img)
        inside = brain[mask].mean()
        assert inside > 0.4, f"only {inside:.0%} of hippocampus inside the brain — repair suspect"

    def test_the_shipped_affine_really_is_broken(self, cohort, ho_atlas):
        """Guards the repair's justification: without it, nothing lines up at all."""
        import nibabel as nib

        from neurochat import kernel

        from pathlib import Path

        originals = sorted(Path(cohort["root"]).glob("*_original.nii.gz"))
        if not originals:
            pytest.skip("no untouched download cached")
        img = nib.load(str(originals[0]))
        data = np.asanyarray(img.dataobj)
        brain = np.isfinite(data)

        region = ho_atlas.resolve("Left Hippocampus")
        mask = kernel.region_mask(ho_atlas.maps_path, region.index, target_img=img)
        inside = brain[mask].mean() if mask.sum() else 0.0
        assert inside < 0.05, (
            "the shipped affine now lines up — if OpenNeuro fixed the data upstream, "
            "the repair should be removed rather than left in"
        )

    def test_a_wrong_grid_is_refused_rather_than_rewritten(self, tmp_path, sample_data):
        """The repair is keyed to one identifiable grid and must not generalise."""
        import shutil

        src = sample_data / "phantom_t1.nii.gz"
        wrong = tmp_path / "wrong.nii.gz"
        shutil.copy(src, wrong)
        with pytest.raises(ValueError, match="refusing to rewrite"):
            library._repair_spm_origin(wrong, tmp_path / "out.nii.gz")

    def test_measuring_a_region_across_the_pet_cohort(self, cohort, ho_atlas, tmp_path):
        from neurochat import tools
        from neurochat.session import Session

        session = Session(name="pet", workdir=tmp_path / "w")
        tools.load_atlas(session, atlas_name="harvard-oxford-sub")
        result = library.region_across_library(
            session, [e["path"] for e in cohort["entries"]], "Left Hippocampus"
        )
        assert result["ok"] and result["n_measured"] == cohort["n_found"]
        # PET clips the inferior brain, so honest NaN accounting is mandatory here.
        assert all(row["excluded_nan"] > 0 for row in result["rows"])
        assert all(row["n_used"] < row["n_voxels"] for row in result["rows"])
        assert any("NaN voxels were excluded" in note for note in result["notes"])

    def test_subjects_actually_differ(self, cohort, ho_atlas, tmp_path):
        """A cohort where every value is identical would mean we measured the atlas."""
        from neurochat import tools
        from neurochat.session import Session

        if cohort["n_found"] < 2:
            pytest.skip("need at least two subjects")
        session = Session(name="pet", workdir=tmp_path / "w")
        tools.load_atlas(session, atlas_name="harvard-oxford-sub")
        result = library.region_across_library(
            session, [e["path"] for e in cohort["entries"]], "Left Hippocampus"
        )
        means = [row["mean"] for row in result["rows"]]
        assert len(set(means)) == len(means), "identical means across subjects is suspicious"
