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


class TestActionableRefusal:
    """A refusal must name the one argument that resolves it, without ever guessing.

    The motivating case: nilearn writes its own MNI152 template with sform_code=2, so
    the most canonically-MNI file in existence was being refused with no way forward.
    """

    def test_aligned_anat_is_distinguished_from_scanner_space(self, tmp_path):
        aligned = detect_space(nib.load(str(_write(tmp_path, "a.nii.gz", xform_code=2))))
        scanner = detect_space(nib.load(str(_write(tmp_path, "b.nii.gz", xform_code=1))))
        assert aligned.name == "aligned" and scanner.name == "native"
        assert aligned.resolvable is False and scanner.resolvable is False
        # "aligned to something" may well be a template; scanner space never is.
        assert "template or another subject" in aligned.detail

    def test_an_unresolvable_volume_on_a_known_grid_suggests_a_space(self, tmp_path):
        info = detect_space(nib.load(str(_write(tmp_path, "vol.nii.gz", xform_code=2))))
        assert info.resolvable is False, "a suggestion must not make it resolvable"
        assert info.suggested_space == "MNI152NLin6Asym"
        assert info.grid_hint is not None

    def test_the_refusal_leads_with_the_fix(self, tmp_path):
        info = detect_space(nib.load(str(_write(tmp_path, "vol.nii.gz", xform_code=0))))
        message = space_refusal_message(info, "left hippocampus", "vol")
        assert "space=" in message
        assert f"space='{info.suggested_space}'" in message
        # The suggestion must appear before the diagnosis, not buried after it.
        assert message.index("To proceed") < message.index("Detected:")
        assert "suggestive but not proof" in message

    def test_accepting_the_suggestion_is_recorded_as_the_users_assertion(self, tmp_path):
        path = _write(tmp_path, "vol.nii.gz", xform_code=2)
        info = detect_space(nib.load(str(path)), path)
        accepted = detect_space(nib.load(str(path)), path, override=info.suggested_space)
        assert accepted.resolvable is True
        assert accepted.source == "user_override"
        assert "asserted by the caller" in accepted.detail

    def test_load_volume_surfaces_the_suggestion_at_load_time(self, tmp_path):
        """The user can act on it while loading, not only after a failed lookup."""
        from neurochat import tools
        from neurochat.session import Session

        path = _write(tmp_path, "vol.nii.gz", xform_code=2)
        session = Session(name="t", workdir=tmp_path / "w")
        payload = tools.load_volume(session, path=str(path))
        assert payload["ok"] and payload["space_resolvable"] is False
        assert payload["suggested_space"] == "MNI152NLin6Asym"
        assert "reload it with" in payload["notes"][0]

    def test_a_resolvable_volume_offers_no_suggestion(self, sample_data, tmp_path):
        from neurochat import tools
        from neurochat.session import Session

        session = Session(name="t", workdir=tmp_path / "w")
        payload = tools.load_volume(session, path=str(sample_data / "phantom_t1.nii.gz"))
        assert payload["space_resolvable"] is True
        assert payload["suggested_space"] is None


class TestFieldOfViewHeuristic:
    """Geometry generalises better than an exact grid table — but is still only a hint."""

    def test_an_mni_field_of_view_is_recognised_at_any_resolution(self):
        from neurochat.spaces import looks_like_mni_fov

        for size, n in ((2.0, (91, 109, 91)), (4.0, (45, 54, 45)), (3.0, (60, 72, 60))):
            affine = np.diag([-size, size, size, 1.0])
            affine[:3, 3] = [90.0, -126.0, -72.0]
            assert looks_like_mni_fov(affine, n), f"{size}mm grid not recognised"

    def test_a_small_or_offset_field_of_view_is_not(self):
        from neurochat.spaces import looks_like_mni_fov

        cropped = np.diag([-2.0, 2.0, 2.0, 1.0])
        cropped[:3, 3] = [40.0, -40.0, -20.0]
        assert not looks_like_mni_fov(cropped, (40, 40, 40))

    def test_geometry_never_makes_a_volume_resolvable_on_its_own(self, tmp_path):
        """The whole guarantee in one assertion."""
        for code in (0, 1, 2):
            info = detect_space(nib.load(str(_write(tmp_path, f"v{code}.nii.gz", xform_code=code))))
            assert info.grid_hint is not None, "this fixture sits on a known MNI grid"
            assert info.resolvable is False, f"geometry promoted xform_code={code} to resolvable"


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
