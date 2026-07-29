# Limitations

Read this before using neurochat for anything that matters. It is written to be
specific about where the tool is wrong or absent, not to be reassuring.

## Not for clinical use

neurochat is not a medical device. It is not validated for diagnostic use, it makes no
clinical claim in any wording, and nothing it outputs should inform patient care.
Requests phrased as diagnosis, prognosis, or "is this scan normal?" are declined by a
deterministic classifier before the model sees them.

## No preprocessing

Input must be an already-preprocessed NIfTI. There is no DICOM conversion, no
`recon-all`, no fMRIPrep, no motion correction, no skull-stripping, no coregistration,
no smoothing, no bias-field correction. If your data is not already in the state you
want it in, use fMRIPrep, dcm2niix, FreeSurfer, or NeuroAgent — and note that
**neurochat cannot tell you whether your preprocessing was adequate.**

## No statistics, no inference

There are no p-values, no group comparisons, no thresholded "significant" maps, no
multiple-comparison correction, and no effect sizes. `roi_stats` is descriptive:
n, mean, sd, median, min, max. `compare_volumes` is arithmetic — a difference image
says two numbers differ and nothing about whether the difference means anything.

This is a scope decision, not a gap waiting to be filled quietly. Adding a
half-considered test would be worse than not having one. Use `nilearn.glm`, FSL
`randomise`, AFNI `3dttest++`, or SPM.

## No arbitrary code execution

The tool surface is fixed at ten tools. The model cannot run generated Python. When you
ask for something out of scope, the suggested code is written into your session script
**as a comment** and never executed.

## It checks coordinate compatibility, not provenance

`overlay` and `compare_volumes` warn when two volumes claim different spaces or sit on
different grids. Neither has any concept of **which brain a scan came from.**

Nothing connects `sub-01_PET.nii.gz` to `sub-01_T1w.nii.gz` rather than to
`sub-47_T1w.nii.gz`. If both are normalised to MNI they will align perfectly, the figure
will look entirely convincing, and it will be meaningless. No warning is issued, because
from the tool's point of view nothing is wrong: the coordinates are compatible.

An overlay is only interpretable when the two volumes are either (a) from the same person
and coregistered — an alignment step that must have happened during preprocessing, since
neurochat does not perform it — or (b) both normalised into the same template space.
Displaying a subject's PET on a *template* brain is a legitimate and standard figure, and
is not covered by this warning; displaying one subject's data on another subject's
anatomy is not, and neurochat cannot tell the difference.

Subject identity is your responsibility. The tool will not catch a mismatched pair.

## It is modality-agnostic mechanically and modality-blind interpretively

neurochat will load any 3D NIfTI — T1, T2, FLAIR, PET, SPECT, CT, a grey-matter density
map, someone else's statistical map — and measure the right voxels in all of them. The
mechanics genuinely do not depend on what the numbers represent.

**The interpretation does, and neurochat does not help you with it.**

- **No units, anywhere.** A returned `mean` is a bare number. An MRI intensity is in
  arbitrary units, is not comparable between scanners, and is frequently not comparable
  between two sessions on the same scanner. A PET SUV is genuinely quantitative. neurochat
  reports both identically and cannot tell you which one you are holding.
- **No reference-region normalisation.** Standard PET quantification divides by a
  reference region (usually cerebellum) to produce an SUVR, because raw uptake is
  confounded by injected dose, body weight and scan timing. `compare_volumes(method=
  "ratio")` can approximate this if you supply the reference volume yourself, but nothing
  prompts you to, and nothing checks that you did.
- **No partial-volume correction, and no warning about it.** PET resolution is roughly
  4–6mm against MRI's ~1mm, so a small structure measured in PET is substantially
  contaminated by neighbouring tissue — a hippocampal value that is partly ventricle and
  partly white matter. Correcting for this is a whole literature. neurochat does none of
  it and does not currently flag when your voxels are large relative to the region you
  asked about.

The practical consequence: neurochat gives you a correct mean of the correct voxels, and
says nothing about whether that mean means anything in your modality. Treat it as a
measurement tool, not an interpretation one.

## Atlas resolution and accuracy

This is where wrong numbers are most likely to come from, so read this section carefully.

- **Resampling shifts boundaries.** When an atlas and a volume are on different grids
  (e.g. a 2mm Harvard-Oxford atlas and a 4mm volume), the label map is resampled with
  nearest-neighbour interpolation. Region boundaries move by up to one voxel. `roi_stats`
  says so in its notes whenever this happens. For small structures — amygdala, accumbens,
  pallidum — one voxel at 4mm is a large fraction of the structure.
- **Maxprob atlases are hard assignments.** Harvard-Oxford maxprob thr25 assigns each
  voxel to a single label above a 25% probability threshold. A different threshold gives
  different masks and therefore different means. v1 does not support probabilistic
  (4D) atlases at all; loading one is an explicit error rather than a silent collapse.
- **MNI152 variants are not interchangeable.** `MNI152NLin6Asym` (FSL, where the
  Harvard-Oxford atlases live) and `MNI152NLin2009cAsym` (fMRIPrep's default) disagree by
  a few millimetres in places. neurochat warns when a volume and an atlas are different
  MNI152 variants and reports a rough magnitude — but it **does not resample between
  them**, and it cannot. If that few-millimetre disagreement matters for your structure,
  transform your data properly first.
- **Centroids are not landmarks.** A region's centre of mass is a summary statistic. For
  curved structures (hippocampus, cingulate) it can fall outside the region entirely; in
  that case neurochat returns the in-region voxel nearest to it and tells you it did.
  Neither point is a canonical anatomical landmark.
- **Region labels are the atlas's opinion.** Harvard-Oxford, AAL and Schaefer disagree
  with each other about boundaries and about what counts as a region. neurochat resolves
  names against whichever atlas you loaded and reports that atlas by name; it does not
  arbitrate.

## Space detection can be wrong in both directions

Space is detected from a BIDS sidecar, then a `space-` filename entity, then the NIfTI
`sform`/`qform` codes.

- **False confidence:** a header claiming `sform_code=4` (MNI152) is taken at face value.
  Plenty of files carry that code without having been normalised. neurochat cannot verify
  the claim, and it will happily resolve region names on such a volume.
- **False refusal:** a correctly-normalised volume can still be refused, and this is
  common rather than exotic. `sform_code=2` ("aligned to some other image") is written by
  a lot of tooling for data that really is in template space — including nilearn's own
  copy of the MNI152 template. neurochat declines it, because "aligned to something" does
  not say *which* something, and it could equally be another subject's scan.

  The refusal is deliberate: geometry is reported as a hint and never promoted to a
  decision. But it is not a dead end. When the geometry looks like a known template, the
  error and the `load_volume` response both name the exact argument that resolves it, and
  the web UI offers it as a single button. Accepting is recorded as `user_override` — your
  assertion, not an inference — and **if you assert the wrong space, nothing downstream
  will catch it.** That is the trade: the tool will not guess for you, which means when
  you do decide, you own it.

## Known failure cases

- **Non-MNI template spaces.** Talairach, MNI305, fsaverage and native/subject space are
  detected and refused for region-name resolution. There is no coordinate transformation
  between spaces anywhere in this tool.
- **4D volumes.** Only the first frame is used, silently, by every statistic. Timeseries,
  multi-echo, and multi-shell data are not meaningfully supported.
- **Surface data.** GIfTI and FreeSurfer surfaces are not supported. Volumes only.
- **Empty or non-overlapping masks.** If an atlas region does not overlap the volume in
  world space, `roi_stats` errors rather than returning statistics over zero voxels.
- **`compare_volumes` does not register.** If two volumes are not already aligned, the
  difference image is meaningless. neurochat warns when the spaces or grids differ; it
  cannot tell you whether alignment is actually correct.
- **Ratio maps near zero.** Division by values below 1e-8 produces NaN rather than a huge
  number. Those NaNs are then counted and reported by `roi_stats`, which is honest but can
  make a ratio map's statistics cover far fewer voxels than you expect.
- **The two renderers do not match.** `screenshot()` uses the live Niivue (WebGL) canvas
  when a browser is attached and nilearn's matplotlib renderer when one is not. They use
  different conventions and do not agree pixel for pixel. The response always says which
  one ran.
- **Exported scripts are not version-pinned.** The header records the versions that ran,
  but the script does not pin them. A different nilearn or numpy may produce different
  numbers, particularly through resampling.
- **Single session, single user.** No accounts, no cloud storage, no multi-tenancy, and no
  authentication on the local server. Do not expose it beyond `127.0.0.1`. Session
  artifacts are written to `~/.neurochat/sessions/` and are never cleaned up automatically
  — they persist on purpose, because an exported script references them.

## The bundled demo atlas is not anatomy

`sample_data/demo16_atlas.nii.gz` contains sixteen geometric shapes with names like
"Left Deep Sphere". They are generated by `scripts/make_sample_data.py` and correspond to
nothing anatomical. They exist so the whole tool — including region resolution,
did-you-mean, and honest NaN accounting — works offline on `git clone` with no downloads.

Real atlases are one `load_atlas("harvard-oxford-sub")` away, fetched once through nilearn
and cached. They are not bundled because their licences are not ours to relicense: the
Harvard-Oxford atlases are distributed with FSL under a non-commercial licence, and this
package is MIT.

## LLM-specific failure modes

- The model can still narrate a wrong interpretation of correct numbers. The tools
  constrain what it can *compute*, not what it can *say*.
- Refusals are enforced by a deterministic classifier for the four Non-Goal categories,
  but that classifier is a regex over the request text. Novel phrasings can slip past it.
  The defence in depth is that there is no tool to do the forbidden thing with.
- Region names are resolved by the tools, but the model chooses *which* region to ask
  about. If it picks the wrong structure, the coordinates will be precisely right for the
  wrong thing.
