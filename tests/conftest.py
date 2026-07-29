from __future__ import annotations

import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "sample_data"


@pytest.fixture(scope="session")
def sample_data() -> Path:
    if not SAMPLE_DATA.exists():
        pytest.fail("sample_data/ missing — run `python scripts/make_sample_data.py`")
    return SAMPLE_DATA


@pytest.fixture(scope="session")
def demo_atlas():
    from neurochat.atlas import load_atlas_table

    return load_atlas_table("demo-16")


@pytest.fixture(scope="session")
def ho_atlas():
    """Harvard-Oxford subcortical. Skipped when it is neither cached nor fetchable."""
    from neurochat.atlas import load_atlas_table

    try:
        return load_atlas_table("harvard-oxford-sub")
    except Exception as exc:  # noqa: BLE001 - network/fetch failures are a skip, not a bug
        pytest.skip(f"Harvard-Oxford atlas unavailable ({type(exc).__name__}: {exc})")


@pytest.fixture
def session(sample_data):
    from neurochat.session import Session

    return Session(name="test")
