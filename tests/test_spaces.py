"""Phase 0 tests: space detection, and refusing to resolve when we cannot tell."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from neurochat.spaces import (
    affine_summary,
    detect_space,
    normalize_space_name,
    space_refusal_message,
    variant_warning,
)

MNI_4MM_AFFINE = np.array(
    [
        [-4.0, 0.0, 0.0, 90.0],
        [0.0, 4.0, 0.0, -126.0],
        [0.0, 0.0, 4.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _write(tmp_path, name, xform_code, affine=MNI_4MM_AFFINE):
    img = nib.Nifti1Image(np.zeros((45, 54, 45), dtype=np.float32), affine)
    img.set_sform(affine, code=xform_code)
    img.set_qform(affine, code=xform_code)
    path = tmp_path / name
    nib.save(img, str(path))
    return path


class TestSpaceNames:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("MNI152NLin2009cAsym", "MNI152NLin2009cAsym"),
            ("mni152nlin2009casym", "MNI152NLin2009cAsym"),
            ("MNI", "MNI152"),
            ("fsl", "MNI152NLin6Asym"),
            ("space-MNI152NLin6Asym", "MNI152NLin6Asym"),
            ("native", "native"),
        ],
    )
    def test_aliases(self, text, expected):
        assert normalize_space_name(text) == expected

    def test_unknown_returns_none(self):
        assert normalize_space_name("brainspace9000") is None
        assert normalize_space_name(None) is None


class TestDetection:
    def test_sample_phantom_reads_its_bids_sidecar(self, sample_data):
        path = sample_data / "phantom_t1.nii.gz"
        info = detect_space(nib.load(str(path)), path)
        assert info.name == "MNI152NLin6Asym"
        assert info.source == "sidecar_space"
        assert info.resolvable is True

    def test_mni_header_code_is_trusted_but_variant_is_not_invented(self, tmp_path):
        path = _write(tmp_path, "vol.nii.gz", xform_code=4)
        info = detect_space(nib.load(str(path)), path)
        assert info.name == "MNI152"
        assert info.resolvable is True
        assert "variant" in info.detail

    def test_scanner_space_is_not_resolvable(self, tmp_path):
        path = _write(tmp_path, "vol.nii.gz", xform_code=1)
        info = detect_space(nib.load(str(path)), path)
        assert info.name == "native"
        assert info.resolvable is False

    def test_talairach_is_refused_rather_than_treated_as_mni(self, tmp_path):
        path = _write(tmp_path, "vol.nii.gz", xform_code=3)
        info = detect_space(nib.load(str(path)), path)
        assert info.name == "Talairach"
        assert info.resolvable is False

    def test_missing_metadata_refuses(self, sample_data):
        path = sample_data / "nospace_volume.nii.gz"
        info = detect_space(nib.load(str(path)), path)
        assert info.name == "unknown"
        assert info.resolvable is False
        assert "0" in info.xform_codes["sform_code"]

    def test_matching_grid_is_reported_as_a_hint_but_never_promoted(self, tmp_path):
        # Same grid as the FSL MNI152 4mm template, but the header says nothing.
        path = _write(tmp_path, "vol.nii.gz", xform_code=0)
        info = detect_space(nib.load(str(path)), path)
        assert info.grid_hint is not None
        assert info.resolvable is False, "geometry must not be enough to assign a space"

    def test_sidecar_beats_header(self, tmp_path):
        path = _write(tmp_path, "vol.nii.gz", xform_code=1)
        (tmp_path / "vol.json").write_text(json.dumps({"Space": "MNI152NLin2009cAsym"}))
        info = detect_space(nib.load(str(path)), path)
        assert info.name == "MNI152NLin2009cAsym"
        assert info.resolvable is True

    def test_bids_filename_entity(self, tmp_path):
        path = _write(tmp_path, "sub-01_space-MNI152NLin2009cAsym_T1w.nii.gz", xform_code=0)
        info = detect_space(nib.load(str(path)), path)
        assert info.name == "MNI152NLin2009cAsym"
        assert info.source == "filename_space"

    def test_user_override_is_taken_at_face_value_and_recorded(self, sample_data):
        path = sample_data / "nospace_volume.nii.gz"
        info = detect_space(nib.load(str(path)), path, override="MNI152NLin6Asym")
        assert info.name == "MNI152NLin6Asym"
        assert info.source == "user_override"
        assert info.resolvable is True

    def test_bad_override_is_rejected(self, sample_data):
        path = sample_data / "phantom_t1.nii.gz"
        with pytest.raises(ValueError, match="Unrecognised space"):
            detect_space(nib.load(str(path)), path, override="wherever")


class TestMessaging:
    def test_refusal_names_the_missing_metadata_and_the_way_out(self, sample_data):
        path = sample_data / "nospace_volume.nii.gz"
        info = detect_space(nib.load(str(path)), path)
        message = space_refusal_message(info, "left hippocampus", "nospace_volume")
        assert "left hippocampus" in message
        assert "sform_code" in message or "no recognized space" in message
        assert "space=" in message

    def test_cross_variant_use_warns_with_a_magnitude(self):
        warning = variant_warning("MNI152NLin2009cAsym", "MNI152NLin6Asym")
        assert warning is not None and "mm" in warning
        assert variant_warning("MNI152NLin6Asym", "MNI152NLin6Asym") is None

    def test_affine_summary_is_small_and_readable(self, sample_data):
        summary = affine_summary(nib.load(str(sample_data / "phantom_t1.nii.gz")).affine)
        assert summary["orientation"] == "LAS"
        assert summary["voxel_size_mm"] == [4.0, 4.0, 4.0]
        assert len(json.dumps(summary)) < 400
