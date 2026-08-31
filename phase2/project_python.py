"""Single source of truth for the interpreter used to spawn child jobs.

Many controllers in `phase2/` shell out to a second Python process (sweep
workers, benchmark evaluators, watchdogs). Historically each of them hardcoded
one absolute conda path, which made the repository unrunnable on any other
machine. Use `project_python()` instead.

Resolution order:

1. ``$PROJECT_PYTHON`` -- explicit override, always wins.
2. ``$PHASE2_PYTHON`` -- legacy name still honoured by ``phase2/run.sh``.
3. ``sys.executable`` -- the interpreter already running this code.
4. ``"python3"`` -- last resort for exotic embedded interpreters.

The value is returned as-is and never validated, so callers can still point at
an interpreter that does not exist yet (for example when only writing a command
file for later execution).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["project_python", "project_python_path", "ENV_VARS"]

ENV_VARS = ("PROJECT_PYTHON", "PHASE2_PYTHON")


def project_python() -> str:
    """Return the interpreter path child processes should be launched with."""
    for name in ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if sys.executable:
        return sys.executable
    return "python3"


def project_python_path() -> Path:
    """`project_python()` as a `Path`, for callers that compare or `.exists()`."""
    return Path(project_python())
