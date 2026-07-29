# Third-party components and data

The MIT licence in [LICENSE](LICENSE) covers this project's own code. It does not
cover the components and datasets below.

## Bundled

**Niivue** — `web/vendor/niivue.umd.js`, version 0.69.0, BSD-2-Clause.
<https://github.com/niivue/niivue>. Vendored rather than loaded from a CDN so the
viewer works offline and needs no build step. BSD-2-Clause is compatible with MIT;
this file is redistributed under its own terms.

## Generated, not redistributed

**`sample_data/`** — every file is synthetic, produced by
`scripts/make_sample_data.py`, and contains no subject-derived or third-party
material. The `demo16` atlas is geometric shapes, not anatomy: see
[LIMITATIONS.md](LIMITATIONS.md).

## Fetched at runtime, never redistributed

The atlases neurochat can load are downloaded on demand through `nilearn` and
cached under `~/nilearn_data`. **They are not part of this repository or its
wheel, and they keep their own terms.** Check them before using neurochat's output
in any published or commercial work:

| Atlas | Terms | Citation |
|---|---|---|
| Harvard-Oxford (cortical, subcortical) | Distributed with FSL under the **FSL licence, which restricts commercial use**. Derived from data released by the Harvard Center for Morphometric Analysis. | Makris et al. 2006; Frazier et al. 2005; Desikan et al. 2006; Goldstein et al. 2007 |
| AAL (SPM12) | Free for academic use; see the AAL distribution for terms. | Tzourio-Mazoyer et al. 2002, *NeuroImage* 15:273–289 |
| Schaefer 2018 | MIT (CBIG). | Schaefer et al. 2018, *Cerebral Cortex* 28:3095–3114 |

If you publish work that used one of these, cite the atlas as well as this tool —
`load_atlas()` returns the citation string in its response for exactly that reason.
