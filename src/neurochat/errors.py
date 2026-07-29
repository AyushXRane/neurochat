"""Errors that are meant to be read by a human or an LLM, not caught silently.

Every error here carries enough text to tell the caller what to do next. Tools in
this package fail loudly on missing metadata; they never fall back to a default
that would make a wrong number look like a right one.
"""

from __future__ import annotations


class NeurochatError(Exception):
    """Base class. Carries a message plus an optional structured payload."""

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        return {"error": type(self).__name__, "message": self.message, **self.details}


class SpaceUnknownError(NeurochatError):
    """The volume's space could not be determined, so region names cannot resolve."""


class SpaceMismatchError(NeurochatError):
    """Volume and atlas live in spaces that cannot be combined."""


class RegionNotFoundError(NeurochatError):
    """A region name did not match any label in the loaded atlas.

    Always carries ``suggestions``: the closest real labels. We ask; we never pick.
    """

    def __init__(self, message: str, suggestions: list[str] | None = None, **details):
        super().__init__(message, suggestions=suggestions or [], **details)
        self.suggestions = suggestions or []


class NoAtlasLoadedError(NeurochatError):
    """A region name was used before any atlas was loaded."""


class VolumeNotFoundError(NeurochatError):
    """A volume name was referenced that is not in the session."""


class OutOfScopeError(NeurochatError):
    """The request is outside v1 scope (statistics, preprocessing, code execution)."""


class PayloadTooLargeError(NeurochatError):
    """A tool tried to return more than the payload budget allows."""
