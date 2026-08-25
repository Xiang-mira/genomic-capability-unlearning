"""Shim: re-export the canonical resolver from scripts/common/paths.py.

Loaded by explicit file path rather than `from paths import *` -- this file is itself named
paths.py and sits earlier on sys.path than scripts/common, so a plain import resolves to
itself and silently exports nothing.
"""
import importlib.util as _u, os as _os
_common = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "common", "paths.py")
_spec = _u.spec_from_file_location("_vb_paths_canonical", _common)
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
