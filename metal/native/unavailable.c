#include "api.h"
/* Portable Metal-unavailable backend: no Apple SDK, objects or pointers. */
xu_int XM_API(available)(void) { return 0; }
xu_int XM_API(abi_version)(void) { return 1; }
xu_int XM_API(live_handles)(void) { return 0; }
xu_int XM_API(device_new)(void) { return XM_UNAVAILABLE; }
xu_int XM_API(device_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(device_release)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(device_info)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(device_name)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_new)(xu_int a0, xu_int a1, xu_int a2) { (void)a0; (void)a1; (void)a2; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_put)(xu_int a0, xu_int a1, xu_int a2) { (void)a0; (void)a1; (void)a2; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_snapshot)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_release)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_len)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_byte)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_low)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_high)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(madd_i64)(xu_int a0, xu_int a1, xu_int a2, xu_int a3, xu_int a4) { (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; return XM_UNAVAILABLE; }
xu_int XM_API(fence_wait)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(fence_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(fence_dispose)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
