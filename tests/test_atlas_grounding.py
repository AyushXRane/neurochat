"""Phase 0 tests: the atlas grounding harness.

If anything here fails, nothing downstream is trustworthy. These tests use no LLM,
no viewer and no UI — just an atlas volume and a lookup table.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurochat.atlas import (
    ATLAS_REGISTRY,
    load_atlas_table,
    normalize_label,
    similarity,
)
from neurochat.errors import RegionNotFoundError

#: Published-ish MNI centroids, used as an external check that our table is not
#: internally consistent but globally wrong. Tolerance is generous because atlas
#: version, threshold and resolution all move these by a couple of mm.
KNOWN_CENTROIDS_MNI = {
    "Left Hippocampus": (-26, -24, -14),
    "Right Hippocampus": (26, -24, -14),
    "Left Amygdala": (-23, -5, -18),
    "Right Amygdala": (23, -5, -18),
    "Left Thalamus": (-11, -19, 7),
    "Left Caudate": (-13, 10, 10),
    "Left Putamen": (-26, 0, 1),
    "Brain-Stem": (0, -28, -34),
}


class TestNormalization:
    @pytest.mark.parametrize(
        "text",
        [
            "Left Hippocampus",
            "left hippocampus",
            "LEFT HIPPOCAMPUS",
            "Hippocampus_L",
            "Hippocampus, left",
            "hippocampus (left)",
            "L. Hippocampus",
            "lh hippocampus",
            "  the   left   hippocampus  ",
        ],
    )
    def test_phrasings_collapse_to_one_key(self, text):
        assert normalize_label(text) == "left hippocampus"

    def test_side_is_not_invented(self):
        assert normalize_label("Hippocampus") == "hippocampus"

    def test_left_and_right_stay_distinct(self):
        assert normalize_label("Left Amygdala") != normalize_label("Right Amygdala")

    def test_synonyms_fold(self):
        assert normalize_label("Left Caudate Nucleus") == normalize_label("Left Caudate")
        assert normalize_label("globus pallidus") == normalize_label("pallidum")

    def test_similarity_is_bounded_and_ordered(self):
        assert similarity("left hippocampus", "left hippocampus") == pytest.approx(1.0)
        near = similarity("left hippocampos", "Left Hippocampus")
        far = similarity("left hippocampos", "Brain-Stem")
        assert 0.0 <= far < near < 1.0


class TestDemoAtlas:
    def test_registry_declares_it_bundled(self):
        assert ATLAS_REGISTRY["demo-16"].bundled is True

    def test_loads_offline_with_expected_shape(self, demo_atlas):
        assert demo_atlas.atlas_id == "demo-16"
        assert demo_atlas.space == "MNI152NLin6Asym"
        assert len(demo_atlas.regions) == 16
        assert "Background" not in demo_atlas.labels

    def test_every_region_has_measured_geometry(self, demo_atlas):
        for region in demo_atlas.regions:
            assert region.n_voxels > 0
            assert region.volume_mm3 > 0
            assert len(region.centroid) == 3
            lo, hi = np.array(region.bbox_min), np.array(region.bbox_max)
            assert np.all(np.array(region.centroid) >= lo - 1e-6)
            assert np.all(np.array(region.centroid) <= hi + 1e-6)

    def test_centroids_land_on_the_generator_specification(self, demo_atlas):
        # The generator placed these shapes at known coordinates; the table is built
        # by measuring the volume, so agreement means the whole pipeline is honest.
        expected = {
            "Left Deep Sphere": (-16, -8, 4),
            "Right Deep Sphere": (16, -8, 4),
            "Midline Dorsal Cap": (0, -20, 62),
        }
        for label, target in expected.items():
            region = demo_atlas.resolve(label)
            assert np.linalg.norm(np.array(region.centroid) - np.array(target)) < 5.0

    def test_representative_point_is_inside_the_mask(self, demo_atlas):
        mask_img = demo_atlas.load_maps()
        data = np.asanyarray(mask_img.dataobj)
        inv = np.linalg.inv(mask_img.affine)
        for region in demo_atlas.regions:
            world = np.append(np.array(region.representative), 1.0)
            i, j, k = np.rint((inv @ world)[:3]).astype(int)
            assert data[i, j, k] == region.index, f"{region.label} representative escaped its mask"

    def test_left_right_regions_are_mirrored(self, demo_atlas):
        left = demo_atlas.resolve("left deep sphere")
        right = demo_atlas.resolve("right deep sphere")
        assert left.centroid[0] < 0 < right.centroid[0]
        assert np.allclose(left.centroid[1:], right.centroid[1:], atol=1.0)


class TestResolution:
    def test_exact_and_paraphrased_names_resolve(self, demo_atlas):
        for query in ["Left Deep Sphere", "left deep sphere", "Deep_Sphere_L", "L deep sphere"]:
            assert demo_atlas.resolve(query).label == "Left Deep Sphere"

    def test_typo_asks_instead_of_guessing(self, demo_atlas):
        with pytest.raises(RegionNotFoundError) as excinfo:
            demo_atlas.resolve("left deep spher")
        error = excinfo.value
        assert "Left Deep Sphere" in error.suggestions
        assert len(error.suggestions) <= 3
        assert "did you mean" in error.message.lower()

    def test_nonsense_fails_without_suggesting_nonsense(self, demo_atlas):
        with pytest.raises(RegionNotFoundError):
            demo_atlas.resolve("zygomatic pizza lobe")

    def test_empty_query_fails(self, demo_atlas):
        with pytest.raises(RegionNotFoundError):
            demo_atlas.resolve("   ")

    def test_wrong_side_is_never_silently_swapped(self, demo_atlas):
        assert demo_atlas.resolve("right deep sphere").label == "Right Deep Sphere"
        assert demo_atlas.resolve("left deep sphere").label == "Left Deep Sphere"

    def test_search_filters_without_resolving(self, demo_atlas):
        hits = demo_atlas.search("deep sphere")
        assert {r.label for r in hits} == {"Left Deep Sphere", "Right Deep Sphere"}
        assert len(demo_atlas.search(None)) == 16

    def test_unknown_atlas_name_suggests_real_ones(self):
        with pytest.raises(KeyError) as excinfo:
            load_atlas_table("harvard-oxfrod")
        assert "harvard-oxford" in str(excinfo.value)


@pytest.mark.network
class TestHarvardOxfordGrounding:
    """The check that matters: do our coordinates match real published anatomy?"""

    def test_expected_label_set(self, ho_atlas):
        labels = set(ho_atlas.labels)
        for expected in ("Left Hippocampus", "Right Hippocampus", "Brain-Stem"):
            assert expected in labels

    @pytest.mark.parametrize("label,target", list(KNOWN_CENTROIDS_MNI.items()))
    def test_centroids_match_published_coordinates(self, ho_atlas, label, target):
        region = ho_atlas.resolve(label)
        distance = float(np.linalg.norm(np.array(region.centroid) - np.array(target)))
        assert distance < 8.0, f"{label} centroid {region.centroid} is {distance:.1f}mm from {target}"

    def test_hippocampus_typo_returns_did_you_mean(self, ho_atlas):
        with pytest.raises(RegionNotFoundError) as excinfo:
            ho_atlas.resolve("left hippocampos")
        assert excinfo.value.suggestions[0] == "Left Hippocampus"

    def test_representative_points_stay_inside_curved_structures(self, ho_atlas):
        maps = ho_atlas.load_maps()
        data = np.asanyarray(maps.dataobj)
        inv = np.linalg.inv(maps.affine)
        for label in ("Left Hippocampus", "Right Hippocampus", "Brain-Stem"):
            region = ho_atlas.resolve(label)
            i, j, k = np.rint((inv @ np.append(region.representative, 1.0))[:3]).astype(int)
            assert data[i, j, k] == region.index

    def test_table_is_cached_and_stable(self, ho_atlas):
        again = load_atlas_table("harvard-oxford-sub")
        assert [r.label for r in again.regions] == [r.label for r in ho_atlas.regions]
        assert again.resolve("left hippocampus").centroid == pytest.approx(
            ho_atlas.resolve("left hippocampus").centroid
        )
