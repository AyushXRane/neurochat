"""neurochat — conversational exploration of volumetric neuroimaging data.

Ask questions of a brain volume in plain language; get back the figure **and** the
`nilearn` code that produced it.

Scope is deliberately narrow. See LIMITATIONS.md: no preprocessing, no statistics,
no clinical use, no arbitrary code execution.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
