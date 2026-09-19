"""
Windows DLL loader helper for PyTorch.
Ensures torch/lib is in Windows DLL search path to prevent WinError 1114 on Windows Python 3.11+.
Stores references to _AddedDllDirectory objects to prevent garbage collection.
"""
from __future__ import annotations

import os
import sys
from typing import Any, List

_DLL_HANDLES: List[Any] = []

if sys.platform == "win32":
    import site
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    if hasattr(site, "getusersitepackages"):
        try:
            candidates.append(site.getusersitepackages())
        except Exception:
            pass

    for base_dir in candidates:
        torch_lib = os.path.join(base_dir, "torch", "lib")
        if os.path.isdir(torch_lib):
            try:
                handle = os.add_dll_directory(torch_lib)
                _DLL_HANDLES.append(handle)
            except (AttributeError, OSError):
                pass
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
