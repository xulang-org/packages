#ifndef XU_METAL_API_H
#define XU_METAL_API_H
#include "xu_native.h"

enum {
    XM_UNAVAILABLE = -1, XM_HANDLE = -2, XM_BOUNDS = -3,
    XM_BUSY = -4, XM_DEVICE = -5, XM_ALIAS = -6,
    XM_ALLOC = -7, XM_PIPELINE = -8, XM_COMMAND = -9,
    XM_DTYPE = -10, XM_LIMIT = -11, XM_NUMERICS = -12, XM_SHAPE = -13
};
#define XM_API(name) XU_NATIVE_SYMBOL(xu_metal_native_##name)
#define XM_DECL(name, args) xu_int XM_API(name) args
XM_DECL(available, (void));
XM_DECL(abi_version, (void));
XM_DECL(live_handles, (void));
XM_DECL(device_new, (void));
XM_DECL(device_close, (xu_int device));
XM_DECL(device_release, (xu_int device));
XM_DECL(device_info, (xu_int device, xu_int key));
XM_DECL(device_name, (xu_int device));
XM_DECL(buffer_new, (xu_int device, xu_int words, xu_int dtype));
XM_DECL(buffer_put, (xu_int buffer, xu_int index, xu_int value));
XM_DECL(buffer_write, (xu_int buffer, xu_int offset, const xu_i64_span *source, xu_int first, xu_int count));
XM_DECL(buffer_snapshot, (xu_int buffer));
XM_DECL(buffer_close, (xu_int buffer));
XM_DECL(buffer_release, (xu_int buffer));
XM_DECL(snapshot_len, (xu_int snapshot));
XM_DECL(snapshot_byte, (xu_int snapshot, xu_int index));
XM_DECL(snapshot_low, (xu_int snapshot, xu_int index));
XM_DECL(snapshot_high, (xu_int snapshot, xu_int index));
XM_DECL(snapshot_copy, (xu_int snapshot, xu_int first, const xu_i64_span_mut *destination, xu_int offset, xu_int count));
XM_DECL(snapshot_close, (xu_int snapshot));
XM_DECL(madd_i64, (xu_int dst, xu_int a, xu_int b, xu_int factor, xu_int words));
XM_DECL(fence_wait, (xu_int fence));
XM_DECL(fence_close, (xu_int fence));
XM_DECL(fence_dispose, (xu_int fence));
XM_DECL(matrix_f32_new, (xu_int device, xu_int rows, xu_int columns));
XM_DECL(matrix_f32_info, (xu_int matrix, xu_int key));
XM_DECL(matrix_f32_write, (xu_int matrix, const xu_i64_span *bits));
XM_DECL(matrix_f32_snapshot, (xu_int matrix));
XM_DECL(matrix_f32_close, (xu_int matrix));
XM_DECL(matrix_f32_release, (xu_int matrix));
XM_DECL(snapshot_f32_copy, (xu_int snapshot, const xu_i64_span_mut *bits));
XM_DECL(gemm_f32, (xu_int dst, xu_int a, xu_int b));
#endif
