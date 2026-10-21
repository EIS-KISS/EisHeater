"""Access to the eisgeneratorpy bindings.

eisgeneratorpy is expected to be installed in the project venv (see pyproject.toml); this module
just re-exports it so the rest of the codebase has a single import point.
"""

from __future__ import annotations

import eisgenerator

__all__ = ["eisgenerator"]
