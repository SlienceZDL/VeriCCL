"""Optional native core: C++ implementations of hot paths, loaded via ctypes.

The Python implementations stay the reference; every native entry point is
checked against them by differential tests. ``VERICCL_CORE`` selects the
behavior: ``auto`` (default) uses the library when it is built, ``off``
always uses Python, ``require`` fails when the library is missing. Build it
with ``python tools/build_native.py``.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading

ABI_VERSION = 1
_MODES = frozenset({"auto", "off", "require"})
_LOCK = threading.Lock()
_STATE = {"loaded": False, "library": None, "error": None}


class NativeUnavailable(RuntimeError):
    """VERICCL_CORE=require but the native core cannot be loaded."""


def mode() -> str:
    value = os.environ.get("VERICCL_CORE", "auto").strip().lower() or "auto"
    if value not in _MODES:
        raise NativeUnavailable("VERICCL_CORE must be one of auto, off, require")
    return value


def library_path() -> Path:
    override = os.environ.get("VERICCL_CORE_LIB")
    return Path(override) if override else Path(__file__).with_name("libvericcl_core.so")


def _load():
    path = library_path()
    if not path.is_file():
        return None, "native core not built: " + str(path)
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        return None, "native core failed to load: " + str(error)
    library.vc_abi_version.restype = ctypes.c_int
    if library.vc_abi_version() != ABI_VERSION:
        return None, "native core ABI version differs; rebuild with tools/build_native.py"
    for name in ("vc_predict",):
        function = getattr(library, name)
        function.argtypes = (ctypes.c_char_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t))
        function.restype = ctypes.c_int
    library.vc_free.argtypes = (ctypes.c_void_p,)
    library.vc_free.restype = None
    return library, None


def library():
    """The loaded core, or None when it is off or unavailable in auto mode."""
    selected = mode()
    if selected == "off":
        return None
    with _LOCK:
        if not _STATE["loaded"]:
            _STATE["library"], _STATE["error"] = _load()
            _STATE["loaded"] = True
    if _STATE["library"] is None and selected == "require":
        raise NativeUnavailable(_STATE["error"])
    return _STATE["library"]


def unavailable_reason():
    library()
    return _STATE["error"]


def call(name: str, payload: bytes) -> tuple[int, bytes]:
    native = library()
    if native is None:
        raise NativeUnavailable(unavailable_reason() or "native core is off")
    output = ctypes.POINTER(ctypes.c_uint8)()
    size = ctypes.c_size_t()
    status = getattr(native, name)(payload, len(payload), ctypes.byref(output), ctypes.byref(size))
    if status == 3 or not output:
        raise MemoryError("native core could not allocate its result")
    try:
        data = ctypes.string_at(output, size.value)
    finally:
        native.vc_free(output)
    return status, data
