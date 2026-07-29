# Glossary — neuroimaging terms, for people who don't have them yet

Written for someone approaching this project without a neuroimaging background. Each
section builds on the one before it, so it reads top to bottom rather than
alphabetically.

---

## 1. The scan itself

**Voxel** — a pixel, but three-dimensional. A tiny cube of brain. A pixel is a square on
a screen; a voxel is a cube in space. "2mm voxels" means each cube measures 2mm on a side.
Smaller voxels mean more detail and much bigger files.

**Volume** — the whole 3D grid of voxels; one scan. Picture a loaf of bread sliced into
images and stacked back into a block. Every voxel holds one number.

**MRI vs PET** — what that number *means* depends on the scan.

- **MRI** (magnetic resonance imaging) measures tissue structure. The number is roughly
  "how bright is the tissue here." Good for seeing anatomy — where grey matter, white
  matter and fluid are.
- **PET** (positron emission tomography) is different. A radioactive tracer is injected
  that binds to something of interest — glucose consumption, or amyloid plaques in
  Alzheimer's research — and the number is how much tracer accumulated there.

MRI shows what the brain *looks like*; PET shows what it's *doing* or what's
*accumulating* in it.

**NIfTI** (`.nii`, or `.nii.gz` when zipped) — the standard file format for brain volumes.
It holds the grid of numbers plus a header describing what they are. Named for the
Neuroimaging Informatics Technology Initiative.

**Header** — the metadata inside a NIfTI file: voxel size, orientation, and crucially
which coordinate system the scan lives in. A surprising amount of this project exists
because headers are frequently vague or wrong.

**DICOM** — the format scanners actually output, one file per slice, wrapped in hospital
metadata. It gets converted to NIfTI before analysis. neurochat does not do that
conversion; `dcm2niix` does.

---

## 2. The alignment problem

This is the central idea. Everything else depends on it.

Every brain is a different size and shape. If I say "look at voxel [40, 55, 30]" in your
scan and in mine, those are entirely different anatomical places. Raw voxel positions
mean nothing across people.

**Native space** (also *scanner space* or *subject space*) — the scan as it came off the
machine, in that individual's own geometry. Nothing about its coordinates transfers to
anybody else.

**Normalisation** — the fix. Mathematically warp and stretch an individual brain until it
lines up with a standard reference brain. Afterwards, everyone's scans sit on a shared
grid and a coordinate means the same anatomical location in all of them. This is
expensive, error-prone, and done *before* neurochat sees the data.

**Template** — the standard reference brain. Since no single person should get to define
"the" brain, a template is built by averaging many people's scans. It looks slightly
blurry, like a composite photograph, because that is exactly what it is.

**MNI / MNI152** — the most widely used template family, built at the Montreal
Neurological Institute from 152 people's scans. "The coordinates are (-26, -24, -14) in
MNI space" means: *on the standard brain map, 26mm left of centre, 24mm back, 14mm down.*
It is latitude and longitude for brains.

**MNI152 variants** — the annoying part. Several groups built that average from the same
152 people using different methods, and every result is called "MNI152." They sit a few
millimetres apart.

| Variant | Who uses it |
|---|---|
| `MNI152NLin6Asym` | FSL, and the Harvard-Oxford atlas |
| `MNI152NLin2009cAsym` | fMRIPrep's default output space |
| `MNI152NLin2009aSym` | nilearn's bundled template |
| `MNI152` | unspecified — the file says MNI but not which edition |

A coordinate from one variant is *nearly* the right spot in another. Usually that is fine;
occasionally it is not; it should never be silently ignored. neurochat warns when your
scan and your atlas come from different editions, and does **not** convert between them.

**Talairach** — an older coordinate system, from a single 60-year-old brain. Not
interchangeable with MNI. neurochat detects and refuses it.

**Affine** — the small matrix in the header that converts "voxel [40, 55, 30]" into
"x, y, z in millimetres." It is the file's map legend: scale, rotation and origin in one
object.

**sform / qform codes** — two integers in the header claiming which coordinate system the
affine points into. `0` unknown, `1` scanner space, `2` "aligned to something,
unspecified," `3` Talairach, `4` MNI152. These are how neurochat decides whether to trust
a file — and code `2` is genuinely ambiguous, which is why a scan carrying it gets a
question rather than either a refusal or an assumption.

**Orientation codes** (`RAS`, `LAS`) — which anatomical direction each axis increases
towards. `LAS` means the first axis grows leftward, the second anterior (forward), the
third superior (up). Unambiguous, unlike the words "radiological" and "neurological,"
which people use inconsistently.

---

## 3. The labels

A normalised scan still has no idea which part is the hippocampus. It is just numbers on
a shared grid.

**Atlas** — a second volume on the same grid, where each voxel holds a *region number*
rather than a brightness. A voxel says `9`, and a lookup table says `9 = Left
Hippocampus`. It is the country borders drawn over the map. Atlases are made by
anatomists hand-labelling real brains and combining the results.

**Parcellation** — the same idea, different word: carving the brain into labelled parcels.

**Region / ROI** — one named structure. ROI is "region of interest," the standard jargon
for "the part I want to measure."

**Mask** — a stencil. A grid of true/false, true wherever a region is. To measure the
hippocampus you build its mask and only look at the voxels underneath.

**Centroid** — a region's centre of mass: the average position of all its voxels. Useful
as "where is this thing," with one catch neurochat handles explicitly — for a curved
structure like the hippocampus, the centre of mass can land *outside* the structure, in
the ventricle next door.

**Maxprob / thr25** — seen in atlas names like `cort-maxprob-thr25-2mm`. Because anatomy
varies between people, atlas makers record probabilities ("this voxel is hippocampus in
40% of people"). *Maxprob thr25* means: give each voxel its most likely region, but only
if that probability reaches 25%. A different threshold yields different regions and
therefore different numbers.

**Deterministic vs probabilistic atlas** — deterministic gives each voxel exactly one
label (a 3D file). Probabilistic gives every voxel a probability for every region (a 4D
file). neurochat supports deterministic atlases only, and rejects 4D ones explicitly
rather than silently flattening them.

**The atlases neurochat can load**

| Atlas | What it covers |
|---|---|
| Harvard-Oxford subcortical | 21 deep structures — hippocampus, amygdala, thalamus, brainstem |
| Harvard-Oxford cortical | 96 cortical regions, split left/right |
| AAL | 116 regions, a long-standing standard |
| Schaefer 2018 | 100 cortical parcels grouped into functional networks |
| demo-16 | **synthetic geometric shapes, not anatomy** — for offline testing only |

---

## 4. Measuring things

**Resampling** — the atlas and your scan often have different voxel sizes (a 2mm atlas, a
1mm scan). One has to be rescaled onto the other's grid, like resizing a photo.

**Interpolation** — how the in-between values get filled in when resampling. For atlases
you must use **nearest-neighbour**: take the closest label, full stop. Anything smoother
invents a voxel that is "60% hippocampus, 40% ventricle," which is meaningless for a
label and quietly corrupts every region boundary. This is a classic silent bug; neurochat
always uses nearest-neighbour and tells you when resampling happened.

**NaN** — "not a number," the marker for a missing value. Real scans have holes: signal
dropout, motion artefacts, masked-out regions. Averaging a list containing NaN without
handling it gives you NaN or a wrong answer, so most code silently drops them — meaning
your "mean of 152 voxels" was secretly a mean of 134. neurochat counts them and reports
the count with every statistic.

**Mean / SD / median** — plain descriptive statistics of the values inside a region. Note
what these are *not*: there is no test here, no p-value, no claim that a difference
between two numbers means anything.

**Colormap and window** — display settings. The colormap turns numbers into colours (grey
for anatomy, hot orange for PET). The window is the range of values mapped across those
colours. Both change what you *see* and never the data.

**Overlay** — stacking one volume on another with transparency, so PET activity can be
seen on top of MRI anatomy.

---

## 5. The software you'll see named

**nibabel** — reads and writes NIfTI files. The plumbing.

**nilearn** — the Python library for analysing brain images: masking, resampling,
plotting, statistics. It is the field's default, which is why emitting *nilearn* code
specifically is what makes neurochat's output useful to somebody else.

**Niivue** — the browser-based WebGL viewer that draws the brain. neurochat drives it
rather than writing a viewer.

**FSL, FreeSurfer, SPM, AFNI** — the established neuroimaging toolkits, mostly
command-line, often multi-gigabyte. The Harvard-Oxford atlas ships with FSL.

**fMRIPrep** — the modern standard for preprocessing: the cleanup and normalisation that
happens *before* neurochat starts. Preprocessing is a Non-Goal here precisely because
fMRIPrep already does it well.

**dcm2niix** — converts scanner DICOM output into NIfTI.

**BIDS** — Brain Imaging Data Structure, a convention for naming and organising
neuroimaging files, including small `.json` "sidecar" files that record things the NIfTI
header cannot. neurochat reads those sidecars to work out a scan's space.

**MCP** — Model Context Protocol. A standard way to hand an AI assistant real tools. It
is how neurochat plugs into Claude Desktop and Claude Code: Claude can call the ten
functions rather than only talking about them.

---

## The one-paragraph version

A scan is a 3D grid of numbers. Normalisation warps everyone's brain onto a shared map
called MNI so that a coordinate means the same anatomical thing in every scan. An atlas is
a labelled overlay on that map saying which region is which. neurochat's whole job is to
let you ask questions in that vocabulary while guaranteeing the coordinates come from the
atlas rather than from a language model's memory — and to hand you the code as proof.
