#include "api.h"
/* Portable Metal-unavailable backend: no Apple SDK, objects or pointers. */
xu_int XM_API(available)(void) { return 0; }
xu_int XM_API(abi_version)(void) { return 2; }
xu_int XM_API(live_handles)(void) { return 0; }
xu_int XM_API(device_new)(void) { return XM_UNAVAILABLE; }
xu_int XM_API(device_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(device_release)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(device_info)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(device_name)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_new)(xu_int a0, xu_int a1, xu_int a2) { (void)a0; (void)a1; (void)a2; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_put)(xu_int a0, xu_int a1, xu_int a2) { (void)a0; (void)a1; (void)a2; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_write)(xu_int a0, xu_int a1, const xu_i64_span *a2, xu_int a3, xu_int a4) { (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_snapshot)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(buffer_release)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_len)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_byte)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_low)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_high)(xu_int a0, xu_int a1) { (void)a0; (void)a1; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_copy)(xu_int a0, xu_int a1, const xu_i64_span_mut *a2, xu_int a3, xu_int a4) { (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(madd_i64)(xu_int a0, xu_int a1, xu_int a2, xu_int a3, xu_int a4) { (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; return XM_UNAVAILABLE; }
xu_int XM_API(fence_wait)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(fence_close)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(fence_dispose)(xu_int a0) { (void)a0; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_write)(xu_int a, const xu_i64_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_snapshot)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_copy)(xu_int a, const xu_i64_span_mut *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(snapshot_f32_copy)(xu_int a, const xu_i64_span_mut *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(gemm_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(gemm_f32_nt)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(rmsnorm_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(softmax_f32)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(swiglu_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(add_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(layernorm_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(gelu_f32)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(mul_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(scale_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(rope_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(causal_attention_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(batch_new)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(batch_commit)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(batch_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(batch_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemm_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemm_f32_nt)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_rmsnorm_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(batch_softmax_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_swiglu_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_add_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_layernorm_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gelu_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_mul_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_scale_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_rope_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_causal_attention_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g, xu_int h) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f16_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f16_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f16_write)(xu_int a, const xu_i64_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f16_copy)(xu_int a, const xu_i64_span_mut *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f16_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f16_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(gemm_f16)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(gemm_f16_nt)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(f16_to_f32)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(f32_to_f16)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemm_f16)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemm_f16_nt)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_f16_to_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_f32_to_f16)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_0_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_0_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_0_write)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_0_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_0_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(gemv_q4_0_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemv_q4_0_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f32_write_bytes)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_f16_write_bytes)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q8_0_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q8_0_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q8_0_write)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q8_0_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q8_0_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(gemv_q8_0_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemv_q8_0_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_k_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_k_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_k_write)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_k_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q4_k_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(gemv_q4_k_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemv_q4_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(qk_norm_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(batch_qk_norm_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(kv_cache_append_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_kv_cache_append_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(attention_decode_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g, xu_int h) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; return XM_UNAVAILABLE; }
xu_int XM_API(batch_attention_decode_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g, xu_int h, xu_int i) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; return XM_UNAVAILABLE; }
xu_int XM_API(embedding_f32)(xu_int a, xu_int b, const xu_i64_span *c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_embedding_f32)(xu_int a, xu_int b, xu_int c, const xu_i64_span *d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(argmax_f32)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(batch_argmax_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_0_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_0_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_0_write)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_0_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_0_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(gemv_q5_0_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemv_q5_0_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q6_k_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q6_k_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q6_k_write)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q6_k_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q6_k_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(gemv_q6_k_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemv_q6_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(rope_interleaved_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(batch_rope_interleaved_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(kv_cache_append_f16)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_kv_cache_append_f16)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(attention_decode_f16)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g, xu_int h) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; return XM_UNAVAILABLE; }
xu_int XM_API(batch_attention_decode_f16)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g, xu_int h, xu_int i) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; return XM_UNAVAILABLE; }
xu_int XM_API(add_rmsnorm_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_add_rmsnorm_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(geglu_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_geglu_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(silu_f32)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(batch_silu_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(log_softmax_f32)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(batch_log_softmax_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(causal_attention_f16)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(batch_causal_attention_f16)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g, xu_int h) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; return XM_UNAVAILABLE; }
xu_int XM_API(moe_topk_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(batch_moe_topk_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g, xu_int h) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; return XM_UNAVAILABLE; }
xu_int XM_API(moe_weighted_add_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_moe_weighted_add_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(rope_yarn_f32)(xu_int a, xu_int b, const xu_i64_span *c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_rope_yarn_f32)(xu_int a, xu_int b, xu_int c, const xu_i64_span *d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(moe_gemv_q4_0_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_moe_gemv_q4_0_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(moe_gemv_q8_0_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_moe_gemv_q8_0_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(moe_gemv_q4_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_moe_gemv_q4_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(moe_gemv_q6_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_moe_gemv_q6_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(paged_attention_decode_f16)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, const xu_i64_span *f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_paged_attention_decode_f16)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, const xu_i64_span *g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(lora_add_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(batch_lora_add_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(adamw_step_f32)(xu_int a, xu_int b, xu_int c, xu_int d, const xu_i64_span *e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(batch_adamw_step_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, const xu_i64_span *f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(cross_entropy_grad_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_cross_entropy_grad_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_k_new)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_k_info)(xu_int a, xu_int b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_k_write)(xu_int a, const xu_u8_span *b) { (void)a; (void)b; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_k_close)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(matrix_q5_k_release)(xu_int a) { (void)a; return XM_UNAVAILABLE; }
xu_int XM_API(gemv_q5_k_f32)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemv_q5_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(moe_gemv_q5_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_moe_gemv_q5_k_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(paged_kv_cache_append_f16)(xu_int a, xu_int b, xu_int c, const xu_i64_span *d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(batch_paged_kv_cache_append_f16)(xu_int a, xu_int b, xu_int c, xu_int d, const xu_i64_span *e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(gemm_f32_tn)(xu_int a, xu_int b, xu_int c) { (void)a; (void)b; (void)c; return XM_UNAVAILABLE; }
xu_int XM_API(batch_gemm_f32_tn)(xu_int a, xu_int b, xu_int c, xu_int d) { (void)a; (void)b; (void)c; (void)d; return XM_UNAVAILABLE; }
xu_int XM_API(rmsnorm_backward_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
xu_int XM_API(batch_rmsnorm_backward_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f, xu_int g) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; return XM_UNAVAILABLE; }
xu_int XM_API(swiglu_backward_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e) { (void)a; (void)b; (void)c; (void)d; (void)e; return XM_UNAVAILABLE; }
xu_int XM_API(batch_swiglu_backward_f32)(xu_int a, xu_int b, xu_int c, xu_int d, xu_int e, xu_int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return XM_UNAVAILABLE; }
