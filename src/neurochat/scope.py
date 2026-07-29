"""Scope guard: what neurochat declines to do, and what to use instead.

Three of the Non-Goals — no preprocessing, no statistics, no clinical claims — are
things a user will absolutely ask for anyway, usually in the middle of an otherwise
reasonable conversation. Leaving that to the model's judgement makes the refusal a
matter of prompt luck, so it lives here instead: a deterministic classifier the chat
layer runs before dispatching, and a set of refusals that name the tools that *do*
do the job.

Refusing well means being useful about it. "I can't do that" is a dead end;
"I can't do that, fMRIPrep can, and here is the code you'd run" is a redirect. Per
R3 the suggested code may be written into the session script — as a comment, never
executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matched against the user's message with word boundaries, so "meaningful" does not
# trip "mean" and "registration form" does not trip "registration".
_PATTERNS = {
    "statistics": r"""
        \bt[-\s]?test|\banova\b|\bp[-\s]?values?\b|\bsignifican(t|ce)\b|\bglm\b
        |\bgeneral\s+linear\s+model\b|\bpermutation\s+test|\brandomise\b
        |\bcluster[-\s]?correct|\bmultiple\s+comparisons?\b|\bfdr\b|\bfwe\b
        |\bbonferroni\b|\bregression\b|\bcorrelat(e|ion)\b|\beffect\s+size\b
        |\bcohen'?s\s+d\b|\bconfidence\s+interval\b|\bgroup\s+(comparison|difference)
        |\bstatistical(ly)?\s+(significan|test|map)|\bz[-\s]?stat|\bcontrast\s+map
        |\bpower\s+analysis\b|\bbootstrap\b
    """,
    "preprocessing": r"""
        \bdicoms?\b|\bdcm2niix\b|\brecon[-\s]?all\b|\bfreesurfer\b|\bfmriprep\b
        |\bmotion\s+correct|\bslice[-\s]timing\b|\bskull[-\s]?strip|\bdefac(e|ing)\b
        |\bbrain\s+extract|\bbet\b|\bcoregist|\bnormali[sz]e\s+to\s+(mni|template)
        |\bspatial\s+normali[sz]|\bsmooth(ing)?\b|\bbias[-\s]field\b|\bn4\b
        |\bsegment(ation)?\b|\bpreprocess|\brealign|\bdistortion\s+correct
        |\btopup\b|\beddy\s+correct
    """,
    "code_execution": r"""
        \b(run|execute|eval(uate)?)\s+(this|the|my|following)?\s*(python|code|script|snippet)
        |\bexec\s*\(|\beval\s*\(|\bimport\s+os\b|\bsubprocess\b|\brun\s+arbitrary
    """,
    "clinical": r"""
        \bdiagnos(e|is|tic)\b|\bprognos(is|tic)\b|\bis\s+(this|the)\s+(patient|scan|brain)
        \s+(normal|abnormal|healthy)|\bdoes\s+(this|the)\s+patient\s+have
        |\bclinical(ly)?\s+(significan|relevan)|\btreatment\s+(plan|recommend)
        |\bshould\s+(i|we|they)\s+(treat|operate|refer)|\bis\s+it\s+(cancer|a\s+tumou?r)
        |\bpathologic(al)?\s+finding
    """,
}

_COMPILED = {
    name: re.compile(pattern, re.IGNORECASE | re.VERBOSE) for name, pattern in _PATTERNS.items()
}

_REFUSALS = {
    "statistics": {
        "message": (
            "That is statistical inference, which neurochat v1 does not do. There are no "
            "p-values, no group comparisons and no thresholded 'significant' maps here — "
            "the tool surface is descriptive exploration only, and adding a half-considered "
            "test would be worse than not having one."
        ),
        "use_instead": [
            "nilearn.glm (FirstLevelModel / SecondLevelModel) for mass-univariate models",
            "FSL randomise for permutation inference with cluster correction",
            "AFNI 3dttest++ or SPM for equivalent group analyses",
        ],
        "suggested_code": (
            "from nilearn.glm.second_level import SecondLevelModel\n"
            "# model = SecondLevelModel().fit(images, design_matrix=design)\n"
            "# z_map = model.compute_contrast(second_level_contrast='group')"
        ),
    },
    "preprocessing": {
        "message": (
            "That is preprocessing, which neurochat deliberately does not touch. Input is "
            "an already-preprocessed NIfTI. That territory is well covered by tools with "
            "hour-long jobs and 10GB installs behind them, and reimplementing it badly "
            "helps nobody."
        ),
        "use_instead": [
            "fMRIPrep for functional and anatomical preprocessing",
            "dcm2niix for DICOM to NIfTI conversion",
            "FreeSurfer recon-all for surface reconstruction and subcortical segmentation",
            "NeuroAgent (arXiv 2026) if you specifically want an LLM-driven preprocessing agent",
        ],
        "suggested_code": (
            "# In a shell, not here:\n"
            "# dcm2niix -o out/ -f sub-01_T1w dicom_dir/\n"
            "# fmriprep bids_dir/ out/ participant --participant-label 01"
        ),
    },
    "code_execution": {
        "message": (
            "neurochat does not execute generated code. The tool surface is fixed and "
            "auditable. Omega (napari) does run generated Python and that is defensible for "
            "a research demo, but this is a tool pointed at patient-derived scans, where "
            "'the model wrote some code and we ran it' is not an acceptable answer."
        ),
        "use_instead": [
            "Ask for one of the ten tools by name",
            "export_script() to get runnable code you can inspect and run yourself",
        ],
        "suggested_code": "",
    },
    "clinical": {
        "message": (
            "neurochat makes no clinical claims of any kind. It is not a medical device, it "
            "is not validated for diagnostic use, and nothing it outputs should inform "
            "patient care."
        ),
        "use_instead": ["A qualified clinician, using clinically validated tools"],
        "suggested_code": "",
    },
}


@dataclass(frozen=True)
class ScopeRefusal:
    category: str
    message: str
    use_instead: list[str]
    suggested_code: str
    matched: str

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "refused": True,
            "category": self.category,
            "message": self.message,
            "use_instead": self.use_instead,
            "suggested_code": self.suggested_code,
            "matched_phrase": self.matched,
        }

    def as_text(self) -> str:
        lines = [self.message]
        if self.use_instead:
            lines.append("Use instead:")
            lines.extend(f"  - {item}" for item in self.use_instead)
        return "\n".join(lines)


def check_scope(text: str) -> ScopeRefusal | None:
    """Classify a request. Returns ``None`` when it is in scope."""
    if not text:
        return None
    for category, pattern in _COMPILED.items():
        match = pattern.search(text)
        if match:
            refusal = _REFUSALS[category]
            return ScopeRefusal(
                category=category,
                message=refusal["message"],
                use_instead=list(refusal["use_instead"]),
                suggested_code=refusal["suggested_code"],
                matched=match.group(0).strip(),
            )
    return None


def record_refusal(session, refusal: ScopeRefusal) -> None:
    """Write the refusal into the session script as a comment (R3): suggested, not run."""
    body = refusal.suggested_code or "# (no equivalent code — out of scope by design)"
    session.script.add_suggestion(
        f"refused:{refusal.category}",
        body,
        comment=(
            f"Request declined as out of scope ({refusal.category}). "
            f"Matched phrase: {refusal.matched!r}. Nothing below was executed."
        ),
    )
