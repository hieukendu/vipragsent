"""Compatibility entry point for the current bounded Q1a extra handoff.

The compact handoff replaced the earlier local-inventory schema. Keep this
historical command usable by routing it to the canonical compact validator,
so callers cannot accidentally validate the obsolete schema and receive a
misleading ``KeyError``.
"""

from validate_q1a_extra_compact import main


if __name__ == "__main__":
    main()
