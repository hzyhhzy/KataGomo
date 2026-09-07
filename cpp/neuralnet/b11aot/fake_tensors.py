# SPDX-License-Identifier: MIT
# Inert DLPack metadata pattern derived from doomoooo/KataGomo_fork's
# sm120_generate_cute_qkv_aot.py; see NOTICE.md and licenses/KATAGO-MIT.txt.
"""Describe CUDA tensor types without a CUDA allocation, context, or device."""
import ctypes


class _Device(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8),
                ("lanes", ctypes.c_uint16)]


class _Tensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _Device),
                ("ndim", ctypes.c_int), ("dtype", _DataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


_Deleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _ManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _Tensor), ("manager_ctx", ctypes.c_void_p),
                ("deleter", _Deleter)]


_capsule = ctypes.pythonapi.PyCapsule_New
_capsule.restype = ctypes.py_object
_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_keepalive = []


class StaticCudaTensorSpec:
    """An inert FP16 kDLCUDA producer; its data address is never dereferenced."""
    def __init__(self, shape, strides, address):
        if len(shape) != len(strides) or not shape:
            raise ValueError("nonempty matching shape/stride ranks are required")
        if any(n <= 0 for n in shape) or any(s < 0 for s in strides):
            raise ValueError("invalid tensor shape/stride")
        if address <= 0 or address % 16:
            raise ValueError("inert address must have 16-byte alignment")
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        self._strides = (ctypes.c_int64 * len(strides))(*strides)
        self._managed = _ManagedTensor(_Tensor(
            ctypes.c_void_p(address), _Device(2, 0), len(shape),
            _DataType(2, 16, 1), self._shape, self._strides, 0,
        ), None, _Deleter())

    def __dlpack__(self, stream=None):
        del stream
        return _capsule(ctypes.addressof(self._managed), b"dltensor", None)

    def __dlpack_device__(self):
        return (2, 0)


def dynamic_fp16_tensor(shape, strides, address, leading_dim):
    """Match FA4 to_cute_tensor(...).mark_layout_dynamic without torch CUDA IO."""
    from cutlass.cute.runtime import from_dlpack
    spec = StaticCudaTensorSpec(shape, strides, address)
    _keepalive.append(spec)
    return from_dlpack(spec, assumed_align=16, enable_tvm_ffi=False).mark_layout_dynamic(
        leading_dim=leading_dim)


def static_fp16_tensor(shape, strides, address, leading_dim):
    """Fully static layout; the exported tensor ABI contains only a data pointer."""
    from cutlass.cute.runtime import from_dlpack
    if leading_dim != len(shape) - 1:
        raise ValueError("static FP16 tensors require a contiguous final dimension")
    spec = StaticCudaTensorSpec(shape, strides, address)
    _keepalive.append(spec)
    return from_dlpack(spec, assumed_align=16, enable_tvm_ffi=False)
