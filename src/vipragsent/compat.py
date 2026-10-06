"""Small compatibility shims for the locked Python 3.10 runtime."""

from __future__ import annotations

from datetime import timezone
from enum import Enum


try:
    from datetime import UTC as UTC
except ImportError:  # Python 3.10: datetime.UTC was added in Python 3.11.
    UTC = timezone.utc


try:
    from enum import StrEnum as StrEnum
except ImportError:  # Python 3.10: StrEnum was added in Python 3.11.
    class StrEnum(str, Enum):
        """Python 3.11-compatible subset of enum.StrEnum."""

        def __new__(cls, value: str):
            if not isinstance(value, str):
                raise TypeError("StrEnum values must be strings")
            member = str.__new__(cls, value)
            member._value_ = value
            return member

        def __str__(self) -> str:
            return str.__str__(self)

        def __format__(self, format_spec: str) -> str:
            return str.__format__(self, format_spec)
