"""Test-only shims for the heavy optional dependencies.

Several `phase1`/`phase2` modules import `torch`, `evo`, and `stripedhyena` at
module scope. Those are declared in `requirements.txt`, but the pure-logic tests
in this directory should stay runnable in a bare environment, so a few test
modules install fake stand-ins before importing the code under test.

Register those stand-ins with `register_stub()` rather than assigning into
`sys.modules` directly. pytest collects every test module into a single process,
so an unconditional stub stays in `sys.modules` for the rest of the session and
leaks into unrelated test modules -- which previously made
`tests/test_proteingym_esm2_qualification.py` fail with
`cannot import name 'LoRALinear' from 'phase2.lora_utils'` in a full-suite run
while passing on its own.

`register_stub()` installs the stub only when the real module cannot be
imported. In a provisioned environment nothing is stubbed and the tests exercise
the real modules; in a bare environment the stubs still make the module under
test importable.
"""
from __future__ import annotations

import importlib
import sys
import types

__all__ = ["module_available", "register_stub"]


def module_available(name: str) -> bool:
    """True when `name` can be imported (or is already imported) for real."""
    if name in sys.modules:
        return True
    try:
        importlib.import_module(name)
    except Exception:
        # Any failure -- missing package, missing transitive dep, import-time
        # error -- means the test needs the stub.
        return False
    return True


def register_stub(name: str, module: types.ModuleType) -> types.ModuleType:
    """Install `module` under `name` only if the real module is unavailable.

    Returns whichever module is registered under `name` afterwards, so callers
    can keep using the return value regardless of which branch was taken.
    """
    if module_available(name):
        return sys.modules[name]
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module
