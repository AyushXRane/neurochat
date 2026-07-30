"""Two warnings that stop a number being read more confidently than it deserves.

Both were documented in LIMITATIONS.md long before they were implemented, which meant
they were visible to people who read limitations files and invisible to everyone else.

* **Units** — is this a quantitative PET value or an arbitrary MRI intensity? Both
  arrive as a bare float and look equally authoritative.
* **Partial volume** — every modality blurs across boundaries, so a thin structure's
  mean is substantially the tissue next door. PET at 4-6mm is the bad case.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurochat import kernel, tools
from neurochat.session import Session
from neurochat.spaces import read_units


@pytest.fixture
def s(tmp_path):
    return Session(name="warn", workdir=tmp_path / "w")


class TestUnits:
    def test_units_are_read_from_the_sidecar(self, sample_data):
        units, _ = read_units(sample_data / "phantom_pet_baseline.nii.gz")
        assert units == "arbitrary uptake"

    def test_absent_sidecar_gives_none_rather_than_a_guess(self, tmp_path, sample_data):
        import shutil

        lonely = tmp_path / "no_sidecar.nii.gz"
        shutil.copy(sample_data / "phantom_t1.nii.gz", lonely)
        assert read_units(lonely) == (None, None)

    def test_a_loaded_volume_carries_its_units(self, s, sample_data):
        payload = tools.load_volume(
            s, path=str(sample_data / "phantom_pet_baseline.nii.gz"), name="pet"
        )
        assert payload["units"] == "arbitrary uptake"

    def test_roi_stats_states_the_units(self, s, sample_data, demo_atlas):
        tools.load_volume(s, path=str(sample_data / "phantom_pet_baseline.nii.gz"), name="pet")
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.roi_stats(s, volume="pet", region_label="Left Deep Sphere")
        assert any("arbitrary uptake" in note for note in result["notes"])

    def test_missing_units_are_called_out_not_glossed_over(self, s, tmp_path, sample_data):
        """The common case. Silence here reads as 'these numbers are fine'."""
        import shutil

        lonely = tmp_path / "unlabelled.nii.gz"
        shutil.copy(sample_data / "phantom_t1.nii.gz", lonely)
        tools.load_volume(s, path=str(lonely), name="mystery", space="MNI152NLin6Asym")
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.roi_stats(s, volume="mystery", region_label="Left Deep Sphere")
        assert any("No units are recorded" in note for note in result["notes"])
        assert any("not comparable across scanners" in note for note in result["notes"])


class TestThickness:
    def _ball(self, radius_vox, shape=(40, 40, 40)):
        grid = np.indices(shape).astype(float)
        centre = np.array(shape).reshape(3, 1, 1, 1) / 2
        return ((grid - centre) ** 2).sum(axis=0) ** 0.5 <= radius_vox

    def test_a_sphere_reports_its_diameter(self):
        mask = self._ball(8)
        assert kernel.thickness_mm(mask, (1.0, 1.0, 1.0)) == pytest.approx(16, abs=2)

    def test_thickness_scales_with_voxel_size(self):
        mask = self._ball(8)
        one = kernel.thickness_mm(mask, (1.0, 1.0, 1.0))
        two = kernel.thickness_mm(mask, (2.0, 2.0, 2.0))
        assert two == pytest.approx(one * 2, rel=0.1)

    def test_a_curved_structure_is_not_flattered_by_its_bounding_box(self):
        """The reason thickness is a distance transform and not a bounding box.

        A C-shape spans a big box in every direction while being thin throughout —
        exactly hippocampus's geometry, and exactly the structure that most needs the
        partial-volume warning.
        """
        mask = np.zeros((40, 40, 40), bool)
        mask[10:30, 10:14, 10:14] = True   # one arm, 4 voxels thick
        mask[26:30, 10:30, 10:14] = True   # the turn
        bbox_min_side = 4 * 1.0            # what a bounding-box measure would report... no:
        idx = np.argwhere(mask)
        bbox_min_side = float((idx.max(0) - idx.min(0) + 1).min())
        thickness = kernel.thickness_mm(mask, (1.0, 1.0, 1.0))
        assert thickness <= 6, f"thickness {thickness} should reflect the 4-voxel arm"
        assert bbox_min_side >= 4
        assert thickness < 20, "a bounding box would call this structure chunky"

    def test_empty_mask_is_zero_not_an_error(self):
        assert kernel.thickness_mm(np.zeros((5, 5, 5), bool), (1, 1, 1)) == 0.0

    def test_boundary_fraction_is_higher_for_smaller_regions(self):
        assert kernel.boundary_fraction(self._ball(3)) > kernel.boundary_fraction(self._ball(12))

    def test_both_land_in_the_stats(self, sample_data):
        import nibabel as nib

        img = nib.load(str(sample_data / "phantom_pet_baseline.nii.gz"))
        mask = kernel.region_mask(
            str(sample_data / "demo16_atlas.nii.gz"), 11, target_img=img
        )
        stats = kernel.summarize_roi(img, mask)
        assert stats["thickness_mm"] > 0
        assert 0.0 <= stats["boundary_fraction"] <= 1.0


class TestPartialVolumeWarning:
    @pytest.mark.network
    def test_thin_structures_warn_and_thick_ones_do_not(self, s, ho_atlas, sample_data):
        """Discrimination against real anatomy, not against a threshold in the abstract."""
        tools.load_volume(s, path=str(sample_data / "phantom_t1.nii.gz"), name="vol")
        tools.load_atlas(s, atlas_name="harvard-oxford-sub")

        def warns(region):
            result = tools.roi_stats(s, volume="vol", region_label=region)
            assert result["ok"], result.get("message")
            return any("thick" in note for note in result["notes"])

        # Thin: hippocampus is ~10mm through, amygdala and accumbens smaller still.
        for region in ("Left Hippocampus", "Left Amygdala", "Left Accumbens"):
            assert warns(region), f"{region} is thin and should warn"
        # Thick: no meaningful partial-volume concern at PET resolution.
        for region in ("Brain-Stem", "Left Cerebral White Matter"):
            assert not warns(region), f"{region} is thick and should not warn"

    def test_the_warning_names_pet_and_says_no_correction_is_applied(
        self, s, sample_data, demo_atlas
    ):
        tools.load_volume(s, path=str(sample_data / "phantom_pet_baseline.nii.gz"), name="pet")
        tools.load_atlas(s, atlas_name="demo-16")
        # A 4mm-voxel demo region, deliberately small.
        result = tools.roi_stats(s, volume="pet", region_label="Left Deep Sphere")
        note = next((n for n in result["notes"] if "thick" in n), "")
        assert note, "a 22mm sphere at 4mm voxels should trip the warning"
        assert "PET" in note
        assert "no partial-volume correction" in note.lower()

    def test_the_warning_does_not_blow_the_payload_budget(self, s, sample_data, demo_atlas):
        tools.load_volume(s, path=str(sample_data / "phantom_pet_baseline.nii.gz"), name="pet")
        tools.load_atlas(s, atlas_name="demo-16")
        result = tools.roi_stats(s, volume="pet", region_label="Left Deep Sphere")
        assert len(json.dumps(result, default=str).encode()) <= tools.MAX_PAYLOAD_BYTES
