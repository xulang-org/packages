// Batching, binary16 matrix (F16Matrix), and raw GGML Q4_0 matrix (Q4_0Matrix)
// native runtime implementation included after matrix_f32.inc.

@interface XMQ4Matrix : XMBuffer
@property(nonatomic) NSUInteger rows;
@property(nonatomic) NSUInteger columns;
@property(nonatomic) NSUInteger blocksPerRow;
@property(nonatomic) NSUInteger byteLen;
@property(nonatomic) int numberClass;
@end
@implementation XMQ4Matrix
@end

xu_int XM_API(batch_new)(xu_int h) { @autoreleasepool {
    ENTER;
    XMDevice *d = lookup(h, DEVICE);
    if (!d) RETURN(XM_HANDLE);
    XMBatch *b = [XMBatch new];
    if (!b) RETURN(XM_ALLOC);
    b.owner = d;
    b.readBuffers = [NSMutableArray new];
    b.writeBuffers = [NSMutableArray new];
    b.operations = [NSMutableArray new];
    if (!b.readBuffers || !b.writeBuffers || !b.operations) RETURN(XM_ALLOC);
    xu_int handle = insert(b, BATCH);
    if (handle < 0) RETURN(handle);
    d.children++;
    RETURN(handle);
} }

xu_int XM_API(batch_commit)(xu_int bh) { @autoreleasepool {
    ENTER;
    XMBatch *b = lookup(bh, BATCH);
    if (!b) RETURN(XM_HANDLE);
    if (b.submitted) RETURN(XM_BUSY);
    XMDevice *d = b.owner;
    xu_int fence = submit_command_rw(d, b.writeBuffers ?: @[], b.readBuffers ?: @[],
                                     b.command, b.operations, YES);
    if (fence < 0) RETURN(fence);
    if (!b.command) {
        for (XMBuffer *buf in b.readBuffers) buf.readBusy--;
        for (XMBuffer *buf in b.writeBuffers) buf.writeBusy--;
    }
    for (XMBuffer *buf in b.writeBuffers) buf.ownerBatch = nil;
    b.submitted = YES;
    b.command = nil;
    b.readBuffers = nil;
    b.writeBuffers = nil;
    b.operations = nil;
    RETURN(fence);
} }

static xu_int release_batch(xu_int bh) {
    ENTER;
    XMBatch *b = lookup(bh, BATCH);
    if (!b) RETURN(XM_HANDLE);
    if (!b.submitted) {
        for (XMBuffer *buf in b.readBuffers) buf.readBusy--;
        for (XMBuffer *buf in b.writeBuffers) {
            buf.writeBusy--;
            buf.ownerBatch = nil;
        }
        b.command = nil;
        b.readBuffers = nil;
        b.writeBuffers = nil;
        b.operations = nil;
    }
    b.owner.children--;
    remove_handle(bh);
    RETURN(0);
}
xu_int XM_API(batch_close)(xu_int bh) { @autoreleasepool { return release_batch(bh); } }
xu_int XM_API(batch_release)(xu_int bh) { @autoreleasepool { return release_batch(bh); } }

static BOOL ensure_quant_pipelines(XMDevice *d) {
    if (d.f16ToF32Pipeline && d.f32ToF16Pipeline && d.gemvQ40Pipeline &&
        d.gemvQ80Pipeline && d.gemvQ4KPipeline && d.gemvQ50Pipeline &&
        d.gemvQ6KPipeline && d.gemvQ5KPipeline && d.kvAppendF16Pipeline && d.attnDecodeF16Pipeline &&
        d.causalAttnF16Pipeline && d.moeGemvQ40Pipeline && d.moeGemvQ80Pipeline &&
        d.moeGemvQ4KPipeline && d.moeGemvQ6KPipeline && d.moeGemvQ5KPipeline &&
        d.pagedAttnDecodeF16Pipeline && d.pagedKvAppendF16Pipeline) return YES;
    NSString *src =
        @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void xu_f16_to_f32(device const half *src [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &ss [[buffer(4)]], constant uint &ds [[buffer(5)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) dst[gid.y * ds + gid.x] = float(src[gid.y * ss + gid.x]);\n"
        "}\n"
        "kernel void xu_f32_to_f16(device const float *src [[buffer(0)]], device half *dst [[buffer(1)]],\n"
        "    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &ss [[buffer(4)]], constant uint &ds [[buffer(5)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) dst[gid.y * ds + gid.x] = half(src[gid.y * ss + gid.x]);\n"
        "}\n"
        "kernel void xu_gemv_q4_0_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y;\n"
        "    uint tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= rows || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 5;\n"
        "    uint w_row_off = n * n_blocks * 18;\n"
        "    uint x_row_off = m * xs;\n"
        "    float acc = 0.0f;\n"
        "    for (uint b = tid; b < n_blocks; b += tg) {\n"
        "        device const uchar *blk = w + w_row_off + b * 18;\n"
        "        ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "        float d = float(as_type<half>(d_bits));\n"
        "        device const float *xp = x + x_row_off + (b << 5);\n"
        "        float bsum = 0.0f;\n"
        "        for (uint j = 0; j < 16; ++j) {\n"
        "            uchar bv = blk[2 + j];\n"
        "            float q0 = float(int(bv & 15u) - 8);\n"
        "            float q1 = float(int(bv >> 4) - 8);\n"
        "            bsum += q0 * xp[j] + q1 * xp[16 + j];\n"
        "        }\n"
        "        acc += bsum * d;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_gemv_q8_0_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= rows || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 5;\n"
        "    uint w_row_off = n * n_blocks * 34;\n"
        "    uint x_row_off = m * xs;\n"
        "    float acc = 0.0f;\n"
        "    for (uint b = tid; b < n_blocks; b += tg) {\n"
        "        device const uchar *blk = w + w_row_off + b * 34;\n"
        "        ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "        float d = float(as_type<half>(d_bits));\n"
        "        device const float *xp = x + x_row_off + (b << 5);\n"
        "        float bsum = 0.0f;\n"
        "        for (uint j = 0; j < 32; ++j) {\n"
        "            bsum += float(as_type<char>(blk[2 + j])) * xp[j];\n"
        "        }\n"
        "        acc += bsum * d;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_gemv_q4_k_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= rows || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 8;\n"
        "    uint w_row_off = n * n_blocks * 144;\n"
        "    uint x_row_off = m * xs;\n"
        "    float acc = 0.0f;\n"
        "    for (uint b = tid; b < n_blocks; b += tg) {\n"
        "        device const uchar *blk = w + w_row_off + b * 144;\n"
        "        ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "        ushort dm_bits = ushort(blk[2]) | (ushort(blk[3]) << 8);\n"
        "        float d = float(as_type<half>(d_bits));\n"
        "        float dmin = float(as_type<half>(dm_bits));\n"
        "        device const uchar *sc = blk + 4;\n"
        "        device const uchar *qs = blk + 16;\n"
        "        device const float *xp_base = x + x_row_off + (b << 8);\n"
        "        for (uint c = 0; c < 4; ++c) {\n"
        "            uint is0 = c << 1;\n"
        "            uint is1 = is0 + 1;\n"
        "            uchar sc1, m1, sc2, m2;\n"
        "            if (c < 2) {\n"
        "                sc1 = sc[is0] & 63u; m1 = sc[is0 + 4] & 63u;\n"
        "                sc2 = sc[is1] & 63u; m2 = sc[is1 + 4] & 63u;\n"
        "            } else {\n"
        "                sc1 = (sc[is0 + 4] & 15u) | ((sc[is0 - 4] >> 6) << 4);\n"
        "                m1  = (sc[is0 + 4] >> 4)  | ((sc[is0]     >> 6) << 4);\n"
        "                sc2 = (sc[is1 + 4] & 15u) | ((sc[is1 - 4] >> 6) << 4);\n"
        "                m2  = (sc[is1 + 4] >> 4)  | ((sc[is1]     >> 6) << 4);\n"
        "            }\n"
        "            float d1 = d * float(sc1), min1 = dmin * float(m1);\n"
        "            float d2 = d * float(sc2), min2 = dmin * float(m2);\n"
        "            device const uchar *qsp = qs + (c << 5);\n"
        "            device const float *xp = xp_base + (c << 6);\n"
        "            float sum1 = 0.0f, xsum1 = 0.0f, sum2 = 0.0f, xsum2 = 0.0f;\n"
        "            for (uint l = 0; l < 32; ++l) {\n"
        "                uchar qv = qsp[l];\n"
        "                float x0 = xp[l], x1 = xp[32 + l];\n"
        "                sum1 += x0 * float(qv & 15u); xsum1 += x0;\n"
        "                sum2 += x1 * float(qv >> 4);  xsum2 += x1;\n"
        "            }\n"
        "            acc += (d1 * sum1 - min1 * xsum1) + (d2 * sum2 - min2 * xsum2);\n"
        "        }\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_gemv_q5_0_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= rows || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 5;\n"
        "    uint w_row_off = n * n_blocks * 22;\n"
        "    uint x_row_off = m * xs;\n"
        "    float acc = 0.0f;\n"
        "    for (uint b = tid; b < n_blocks; b += tg) {\n"
        "        device const uchar *blk = w + w_row_off + b * 22;\n"
        "        ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "        float d = float(as_type<half>(d_bits));\n"
        "        uint qh = uint(blk[2]) | (uint(blk[3]) << 8) | (uint(blk[4]) << 16) | (uint(blk[5]) << 24);\n"
        "        device const float *xp = x + x_row_off + (b << 5);\n"
        "        float bsum = 0.0f;\n"
        "        for (uint j = 0; j < 16; ++j) {\n"
        "            uchar bv = blk[6 + j];\n"
        "            uint xh0 = ((qh >> j) & 1u) << 4;\n"
        "            uint xh1 = ((qh >> (j + 16)) & 1u) << 4;\n"
        "            float q0 = float(int((bv & 15u) | xh0) - 16);\n"
        "            float q1 = float(int((bv >> 4) | xh1) - 16);\n"
        "            bsum += q0 * xp[j] + q1 * xp[16 + j];\n"
        "        }\n"
        "        acc += bsum * d;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_gemv_q6_k_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= rows || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 8;\n"
        "    uint w_row_off = n * n_blocks * 210;\n"
        "    uint x_row_off = m * xs;\n"
        "    float acc = 0.0f;\n"
        "    for (uint b = tid; b < n_blocks; b += tg) {\n"
        "        device const uchar *blk = w + w_row_off + b * 210;\n"
        "        ushort d_bits = ushort(blk[208]) | (ushort(blk[209]) << 8);\n"
        "        float d = float(as_type<half>(d_bits));\n"
        "        device const uchar *ql = blk;\n"
        "        device const uchar *qh = blk + 128;\n"
        "        device const uchar *sc = blk + 192;\n"
        "        device const float *xp_base = x + x_row_off + (b << 8);\n"
        "        float bsum = 0.0f;\n"
        "        for (uint h = 0; h < 2; ++h) {\n"
        "            device const uchar *qlp = ql + (h << 6);\n"
        "            device const uchar *qhp = qh + (h << 5);\n"
        "            device const uchar *scp = sc + (h << 3);\n"
        "            device const float *yp = xp_base + (h << 7);\n"
        "            for (uint l = 0; l < 32; ++l) {\n"
        "                uint is_idx = l >> 4;\n"
        "                uchar ql0 = qlp[l], ql1 = qlp[32 + l], qhv = qhp[l];\n"
        "                int q1 = int((ql0 & 15u) | ((qhv & 3u) << 4)) - 32;\n"
        "                int q2 = int((ql1 & 15u) | (((qhv >> 2) & 3u) << 4)) - 32;\n"
        "                int q3 = int(((ql0 >> 4) & 15u) | (((qhv >> 4) & 3u) << 4)) - 32;\n"
        "                int q4 = int(((ql1 >> 4) & 15u) | (((qhv >> 6) & 3u) << 4)) - 32;\n"
        "                int s1 = int(as_type<char>(scp[is_idx])) * q1;\n"
        "                int s2 = int(as_type<char>(scp[is_idx + 2])) * q2;\n"
        "                int s3 = int(as_type<char>(scp[is_idx + 4])) * q3;\n"
        "                int s4 = int(as_type<char>(scp[is_idx + 6])) * q4;\n"
        "                bsum += float(s1) * yp[l] + float(s2) * yp[l + 32] +\n"
        "                        float(s3) * yp[l + 64] + float(s4) * yp[l + 96];\n"
        "            }\n"
        "        }\n"
        "        acc += d * bsum;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_kv_cache_append_f16(device const float *src [[buffer(0)]], device half *cache [[buffer(1)]],\n"
        "    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &ss [[buffer(4)]], constant uint &ds [[buffer(5)]],\n"
        "    constant uint &pos_off [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        cache[(pos_off + gid.y) * ds + gid.x] = half(src[gid.y * ss + gid.x]);\n"
        "    }\n"
        "}\n"
        "kernel void xu_attention_decode_f16(device const float *q [[buffer(0)]], device const half *k [[buffer(1)]],\n"
        "    device const half *v [[buffer(2)]], device float *dst [[buffer(3)]],\n"
        "    constant uint &kv_len [[buffer(4)]], constant uint &n_heads [[buffer(5)]],\n"
        "    constant uint &kv_mul [[buffer(6)]], constant uint &head_dim [[buffer(7)]],\n"
        "    constant uint &ks [[buffer(8)]], constant uint &vs [[buffer(9)]],\n"
        "    constant float &scale [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]],\n"
        "    uint tid [[thread_index_in_threadgroup]], uint tg [[threads_per_threadgroup]],\n"
        "    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],\n"
        "    uint sct [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint h = tgid;\n"
        "    if (h >= n_heads) return;\n"
        "    uint kv_h = h / kv_mul;\n"
        "    uint qbase = h * head_dim;\n"
        "    uint dbase = h * head_dim;\n"
        "    float o_reg[8];\n"
        "    for (uint i = 0; i < 8; ++i) { o_reg[i] = 0.0f; }\n"
        "    float rmax = -INFINITY;\n"
        "    float lsum = 0.0f;\n"
        "    for (uint t = 0; t < kv_len; ++t) {\n"
        "        uint kbase = t * ks + kv_h * head_dim;\n"
        "        uint vbase = t * vs + kv_h * head_dim;\n"
        "        float partial = 0.0f;\n"
        "        for (uint d = tid; d < head_dim; d += tg) { partial += q[qbase + d] * float(k[kbase + d]); }\n"
        "        partial = simd_sum(partial);\n"
        "        if (sl == 0) sh[sg] = partial;\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        if (sg == 0) {\n"
        "            float sv = (sl < sct) ? sh[sl] : 0.0f;\n"
        "            sv = simd_sum(sv);\n"
        "            if (sl == 0) sh[0] = sv;\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        float sc = sh[0] * scale;\n"
        "        float alpha = 0.0f, p = 1.0f;\n"
        "        if (t == 0) {\n"
        "            rmax = sc; lsum = 1.0f;\n"
        "        } else {\n"
        "            float nm = max(rmax, sc);\n"
        "            alpha = exp(rmax - nm);\n"
        "            p = exp(sc - nm);\n"
        "            lsum = lsum * alpha + p;\n"
        "            rmax = nm;\n"
        "        }\n"
        "        for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) {\n"
        "            o_reg[i] = (t == 0 ? 0.0f : o_reg[i] * alpha) + p * float(v[vbase + d]);\n"
        "        }\n"
        "    }\n"
        "    float inv_sum = 1.0f / lsum;\n"
        "    for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) {\n"
        "        dst[dbase + d] = o_reg[i] * inv_sum;\n"
        "    }\n"
        "}\n"
        "kernel void xu_causal_attn_f16(device const float *q [[buffer(0)]], device const half *k [[buffer(1)]],\n"
        "    device const half *v [[buffer(2)]], device float *dst [[buffer(3)]],\n"
        "    constant uint &seq_len [[buffer(4)]], constant uint &kv_len [[buffer(5)]],\n"
        "    constant uint &n_heads [[buffer(6)]], constant uint &kv_mul [[buffer(7)]],\n"
        "    constant uint &head_dim [[buffer(8)]], constant uint &qs [[buffer(9)]],\n"
        "    constant uint &ks [[buffer(10)]], constant uint &vs [[buffer(11)]],\n"
        "    constant uint &ds [[buffer(12)]], constant float &scale [[buffer(13)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sct [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint h = tgid.x, q_row = tgid.y;\n"
        "    uint tid = tid2.x, tg = tg2.x;\n"
        "    if (h >= n_heads || q_row >= seq_len) return;\n"
        "    uint kv_h = h / kv_mul;\n"
        "    uint max_t = (kv_len - seq_len) + q_row;\n"
        "    uint qbase = q_row * qs + h * head_dim;\n"
        "    uint dbase = q_row * ds + h * head_dim;\n"
        "    float o_reg[8];\n"
        "    for (uint i = 0; i < 8; ++i) { o_reg[i] = 0.0f; }\n"
        "    float rmax = -INFINITY;\n"
        "    float lsum = 0.0f;\n"
        "    for (uint t = 0; t <= max_t; ++t) {\n"
        "        uint kbase = t * ks + kv_h * head_dim;\n"
        "        uint vbase = t * vs + kv_h * head_dim;\n"
        "        float partial = 0.0f;\n"
        "        for (uint d = tid; d < head_dim; d += tg) { partial += q[qbase + d] * float(k[kbase + d]); }\n"
        "        partial = simd_sum(partial);\n"
        "        if (sl == 0) sh[sg] = partial;\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        if (sg == 0) {\n"
        "            float sv = (sl < sct) ? sh[sl] : 0.0f;\n"
        "            sv = simd_sum(sv);\n"
        "            if (sl == 0) sh[0] = sv;\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        float sc = sh[0] * scale;\n"
        "        float alpha = 0.0f, p = 1.0f;\n"
        "        if (t == 0) { rmax = sc; lsum = 1.0f; }\n"
        "        else { float nm = max(rmax, sc); alpha = exp(rmax - nm); p = exp(sc - nm); lsum = lsum * alpha + p; rmax = nm; }\n"
        "        for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) {\n"
        "            o_reg[i] = (t == 0 ? 0.0f : o_reg[i] * alpha) + p * float(v[vbase + d]);\n"
        "        }\n"
        "    }\n"
        "    float inv_sum = 1.0f / lsum;\n"
        "    for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) { dst[dbase + d] = o_reg[i] * inv_sum; }\n"
        "}\n"
        "kernel void xu_moe_gemv_q4_0_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device const float *eids [[buffer(2)]], device const float *ewts [[buffer(3)]],\n"
        "    device float *dst [[buffer(4)]], constant uint &out_dim [[buffer(5)]],\n"
        "    constant uint &cols [[buffer(6)]], constant uint &m_cnt [[buffer(7)]],\n"
        "    constant uint &top_k [[buffer(8)]], constant uint &n_experts [[buffer(9)]],\n"
        "    constant uint &x_row_mode [[buffer(10)]], constant uint &xs [[buffer(11)]],\n"
        "    constant uint &is [[buffer(12)]], constant uint &ws [[buffer(13)]],\n"
        "    constant uint &ds [[buffer(14)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= out_dim || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 5;\n"
        "    uint row_bytes = n_blocks * 18;\n"
        "    float acc = 0.0f;\n"
        "    for (uint ek = 0; ek < top_k; ++ek) {\n"
        "        int eid = int(rint(eids[m * is + ek]));\n"
        "        float ew = ewts[m * ws + ek];\n"
        "        if (eid < 0 || uint(eid) >= n_experts || ew == 0.0f) continue;\n"
        "        uint w_row_off = (uint(eid) * out_dim + n) * row_bytes;\n"
        "        uint x_row_off = (x_row_mode != 0 ? (m * top_k + ek) : m) * xs;\n"
        "        float eacc = 0.0f;\n"
        "        for (uint b = tid; b < n_blocks; b += tg) {\n"
        "            device const uchar *blk = w + w_row_off + b * 18;\n"
        "            ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "            float d = float(as_type<half>(d_bits));\n"
        "            device const float *xp = x + x_row_off + (b << 5);\n"
        "            float bsum = 0.0f;\n"
        "            for (uint j = 0; j < 16; ++j) {\n"
        "                uchar bv = blk[2 + j];\n"
        "                bsum += float(int(bv & 15u) - 8) * xp[j] + float(int(bv >> 4) - 8) * xp[16 + j];\n"
        "            }\n"
        "            eacc += bsum * d;\n"
        "        }\n"
        "        acc += ew * eacc;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_moe_gemv_q8_0_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device const float *eids [[buffer(2)]], device const float *ewts [[buffer(3)]],\n"
        "    device float *dst [[buffer(4)]], constant uint &out_dim [[buffer(5)]],\n"
        "    constant uint &cols [[buffer(6)]], constant uint &m_cnt [[buffer(7)]],\n"
        "    constant uint &top_k [[buffer(8)]], constant uint &n_experts [[buffer(9)]],\n"
        "    constant uint &x_row_mode [[buffer(10)]], constant uint &xs [[buffer(11)]],\n"
        "    constant uint &is [[buffer(12)]], constant uint &ws [[buffer(13)]],\n"
        "    constant uint &ds [[buffer(14)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= out_dim || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 5;\n"
        "    uint row_bytes = n_blocks * 34;\n"
        "    float acc = 0.0f;\n"
        "    for (uint ek = 0; ek < top_k; ++ek) {\n"
        "        int eid = int(rint(eids[m * is + ek]));\n"
        "        float ew = ewts[m * ws + ek];\n"
        "        if (eid < 0 || uint(eid) >= n_experts || ew == 0.0f) continue;\n"
        "        uint w_row_off = (uint(eid) * out_dim + n) * row_bytes;\n"
        "        uint x_row_off = (x_row_mode != 0 ? (m * top_k + ek) : m) * xs;\n"
        "        float eacc = 0.0f;\n"
        "        for (uint b = tid; b < n_blocks; b += tg) {\n"
        "            device const uchar *blk = w + w_row_off + b * 34;\n"
        "            ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "            float d = float(as_type<half>(d_bits));\n"
        "            device const float *xp = x + x_row_off + (b << 5);\n"
        "            float bsum = 0.0f;\n"
        "            for (uint j = 0; j < 32; ++j) {\n"
        "                bsum += float(as_type<char>(blk[2 + j])) * xp[j];\n"
        "            }\n"
        "            eacc += bsum * d;\n"
        "        }\n"
        "        acc += ew * eacc;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_moe_gemv_q4_k_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device const float *eids [[buffer(2)]], device const float *ewts [[buffer(3)]],\n"
        "    device float *dst [[buffer(4)]], constant uint &out_dim [[buffer(5)]],\n"
        "    constant uint &cols [[buffer(6)]], constant uint &m_cnt [[buffer(7)]],\n"
        "    constant uint &top_k [[buffer(8)]], constant uint &n_experts [[buffer(9)]],\n"
        "    constant uint &x_row_mode [[buffer(10)]], constant uint &xs [[buffer(11)]],\n"
        "    constant uint &is [[buffer(12)]], constant uint &ws [[buffer(13)]],\n"
        "    constant uint &ds [[buffer(14)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= out_dim || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 8;\n"
        "    uint row_bytes = n_blocks * 144;\n"
        "    float acc = 0.0f;\n"
        "    for (uint ek = 0; ek < top_k; ++ek) {\n"
        "        int eid = int(rint(eids[m * is + ek]));\n"
        "        float ew = ewts[m * ws + ek];\n"
        "        if (eid < 0 || uint(eid) >= n_experts || ew == 0.0f) continue;\n"
        "        uint w_row_off = (uint(eid) * out_dim + n) * row_bytes;\n"
        "        uint x_row_off = (x_row_mode != 0 ? (m * top_k + ek) : m) * xs;\n"
        "        float eacc = 0.0f;\n"
        "        for (uint b = tid; b < n_blocks; b += tg) {\n"
        "            device const uchar *blk = w + w_row_off + b * 144;\n"
        "            ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "            ushort dm_bits = ushort(blk[2]) | (ushort(blk[3]) << 8);\n"
        "            float d = float(as_type<half>(d_bits));\n"
        "            float dmin = float(as_type<half>(dm_bits));\n"
        "            device const uchar *sc = blk + 4;\n"
        "            device const uchar *qs = blk + 16;\n"
        "            device const float *xp_base = x + x_row_off + (b << 8);\n"
        "            for (uint c = 0; c < 4; ++c) {\n"
        "                uint is0 = c << 1, is1 = is0 + 1;\n"
        "                uchar sc1, m1, sc2, m2;\n"
        "                if (c < 2) {\n"
        "                    sc1 = sc[is0] & 63u; m1 = sc[is0 + 4] & 63u;\n"
        "                    sc2 = sc[is1] & 63u; m2 = sc[is1 + 4] & 63u;\n"
        "                } else {\n"
        "                    sc1 = (sc[is0 + 4] & 15u) | ((sc[is0 - 4] >> 6) << 4);\n"
        "                    m1  = (sc[is0 + 4] >> 4)  | ((sc[is0]     >> 6) << 4);\n"
        "                    sc2 = (sc[is1 + 4] & 15u) | ((sc[is1 - 4] >> 6) << 4);\n"
        "                    m2  = (sc[is1 + 4] >> 4)  | ((sc[is1]     >> 6) << 4);\n"
        "                }\n"
        "                float d1 = d * float(sc1), min1 = dmin * float(m1);\n"
        "                float d2 = d * float(sc2), min2 = dmin * float(m2);\n"
        "                device const uchar *qsp = qs + (c << 5);\n"
        "                device const float *xp = xp_base + (c << 6);\n"
        "                float sum1 = 0.0f, xsum1 = 0.0f, sum2 = 0.0f, xsum2 = 0.0f;\n"
        "                for (uint l = 0; l < 32; ++l) {\n"
        "                    uchar qv = qsp[l];\n"
        "                    float x0 = xp[l], x1 = xp[32 + l];\n"
        "                    sum1 += x0 * float(qv & 15u); xsum1 += x0;\n"
        "                    sum2 += x1 * float(qv >> 4);  xsum2 += x1;\n"
        "                }\n"
        "                eacc += (d1 * sum1 - min1 * xsum1) + (d2 * sum2 - min2 * xsum2);\n"
        "            }\n"
        "        }\n"
        "        acc += ew * eacc;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_moe_gemv_q6_k_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device const float *eids [[buffer(2)]], device const float *ewts [[buffer(3)]],\n"
        "    device float *dst [[buffer(4)]], constant uint &out_dim [[buffer(5)]],\n"
        "    constant uint &cols [[buffer(6)]], constant uint &m_cnt [[buffer(7)]],\n"
        "    constant uint &top_k [[buffer(8)]], constant uint &n_experts [[buffer(9)]],\n"
        "    constant uint &x_row_mode [[buffer(10)]], constant uint &xs [[buffer(11)]],\n"
        "    constant uint &is [[buffer(12)]], constant uint &ws [[buffer(13)]],\n"
        "    constant uint &ds [[buffer(14)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= out_dim || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 8;\n"
        "    uint row_bytes = n_blocks * 210;\n"
        "    float acc = 0.0f;\n"
        "    for (uint ek = 0; ek < top_k; ++ek) {\n"
        "        int eid = int(rint(eids[m * is + ek]));\n"
        "        float ew = ewts[m * ws + ek];\n"
        "        if (eid < 0 || uint(eid) >= n_experts || ew == 0.0f) continue;\n"
        "        uint w_row_off = (uint(eid) * out_dim + n) * row_bytes;\n"
        "        uint x_row_off = (x_row_mode != 0 ? (m * top_k + ek) : m) * xs;\n"
        "        float eacc = 0.0f;\n"
        "        for (uint b = tid; b < n_blocks; b += tg) {\n"
        "            device const uchar *blk = w + w_row_off + b * 210;\n"
        "            ushort d_bits = ushort(blk[208]) | (ushort(blk[209]) << 8);\n"
        "            float d = float(as_type<half>(d_bits));\n"
        "            device const uchar *ql = blk;\n"
        "            device const uchar *qh = blk + 128;\n"
        "            device const uchar *sc = blk + 192;\n"
        "            device const float *xp_base = x + x_row_off + (b << 8);\n"
        "            float bsum = 0.0f;\n"
        "            for (uint h = 0; h < 2; ++h) {\n"
        "                device const uchar *qlp = ql + (h << 6);\n"
        "                device const uchar *qhp = qh + (h << 5);\n"
        "                device const uchar *scp = sc + (h << 3);\n"
        "                device const float *yp = xp_base + (h << 7);\n"
        "                for (uint l = 0; l < 32; ++l) {\n"
        "                    uint is_idx = l >> 4;\n"
        "                    uchar ql0 = qlp[l], ql1 = qlp[32 + l], qhv = qhp[l];\n"
        "                    int q1 = int((ql0 & 15u) | ((qhv & 3u) << 4)) - 32;\n"
        "                    int q2 = int((ql1 & 15u) | (((qhv >> 2) & 3u) << 4)) - 32;\n"
        "                    int q3 = int(((ql0 >> 4) & 15u) | (((qhv >> 4) & 3u) << 4)) - 32;\n"
        "                    int q4 = int(((ql1 >> 4) & 15u) | (((qhv >> 6) & 3u) << 4)) - 32;\n"
        "                    int s1 = int(as_type<char>(scp[is_idx])) * q1;\n"
        "                    int s2 = int(as_type<char>(scp[is_idx + 2])) * q2;\n"
        "                    int s3 = int(as_type<char>(scp[is_idx + 4])) * q3;\n"
        "                    int s4 = int(as_type<char>(scp[is_idx + 6])) * q4;\n"
        "                    bsum += float(s1) * yp[l] + float(s2) * yp[l + 32] +\n"
        "                            float(s3) * yp[l + 64] + float(s4) * yp[l + 96];\n"
        "                }\n"
        "            }\n"
        "            eacc += d * bsum;\n"
        "        }\n"
        "        acc += ew * eacc;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_paged_attn_decode_f16(device const float *q [[buffer(0)]], device const half *k [[buffer(1)]],\n"
        "    device const half *v [[buffer(2)]], device const float *bt [[buffer(3)]],\n"
        "    device float *dst [[buffer(4)]], constant uint &kv_len [[buffer(5)]],\n"
        "    constant uint &block_size [[buffer(6)]], constant uint &num_blocks [[buffer(7)]],\n"
        "    constant uint &n_heads [[buffer(8)]], constant uint &kv_mul [[buffer(9)]],\n"
        "    constant uint &head_dim [[buffer(10)]], constant uint &qs [[buffer(11)]],\n"
        "    constant uint &ks [[buffer(12)]], constant uint &vs [[buffer(13)]],\n"
        "    constant uint &bts [[buffer(14)]], constant uint &ds [[buffer(15)]],\n"
        "    constant float &scale [[buffer(16)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sct [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint h = tgid.x, b = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (h >= n_heads) return;\n"
        "    uint kv_h = h / kv_mul;\n"
        "    uint qbase = b * qs + h * head_dim;\n"
        "    uint dbase = b * ds + h * head_dim;\n"
        "    float o_reg[8];\n"
        "    for (uint i = 0; i < 8; ++i) { o_reg[i] = 0.0f; }\n"
        "    float rmax = -INFINITY;\n"
        "    float lsum = 0.0f;\n"
        "    uint btbase = b * bts;\n"
        "    bool valid_any = false;\n"
        "    for (uint t = 0; t < kv_len; ++t) {\n"
        "        int pb = int(rint(bt[btbase + (t / block_size)]));\n"
        "        if (pb < 0 || uint(pb) >= num_blocks) continue;\n"
        "        uint row_idx = uint(pb) * block_size + (t % block_size);\n"
        "        uint kbase = row_idx * ks + kv_h * head_dim;\n"
        "        uint vbase = row_idx * vs + kv_h * head_dim;\n"
        "        float partial = 0.0f;\n"
        "        for (uint d = tid; d < head_dim; d += tg) { partial += q[qbase + d] * float(k[kbase + d]); }\n"
        "        partial = simd_sum(partial);\n"
        "        if (sl == 0) sh[sg] = partial;\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        if (sg == 0) {\n"
        "            float sv = (sl < sct) ? sh[sl] : 0.0f;\n"
        "            sv = simd_sum(sv);\n"
        "            if (sl == 0) sh[0] = sv;\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        float sc = sh[0] * scale;\n"
        "        float alpha = 0.0f, p = 1.0f;\n"
        "        if (!valid_any) { rmax = sc; lsum = 1.0f; valid_any = true; }\n"
        "        else { float nm = max(rmax, sc); alpha = exp(rmax - nm); p = exp(sc - nm); lsum = lsum * alpha + p; rmax = nm; }\n"
        "        for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) {\n"
        "            o_reg[i] = (!valid_any ? 0.0f : o_reg[i] * alpha) + p * float(v[vbase + d]);\n"
        "        }\n"
        "    }\n"
        "    float inv_sum = (lsum > 0.0f) ? (1.0f / lsum) : 0.0f;\n"
        "    for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) {\n"
        "        dst[dbase + d] = o_reg[i] * inv_sum;\n"
        "    }\n"
        "}\n"
        "kernel void xu_gemv_q5_k_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= rows || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 8;\n"
        "    uint w_row_off = n * n_blocks * 176;\n"
        "    uint x_row_off = m * xs;\n"
        "    float acc = 0.0f;\n"
        "    for (uint b = tid; b < n_blocks; b += tg) {\n"
        "        device const uchar *blk = w + w_row_off + b * 176;\n"
        "        ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "        ushort dm_bits = ushort(blk[2]) | (ushort(blk[3]) << 8);\n"
        "        float d = float(as_type<half>(d_bits));\n"
        "        float dmin = float(as_type<half>(dm_bits));\n"
        "        device const uchar *sc = blk + 4;\n"
        "        device const uchar *qh = blk + 16;\n"
        "        device const uchar *qs = blk + 48;\n"
        "        device const float *xp_base = x + x_row_off + (b << 8);\n"
        "        for (uint c = 0; c < 4; ++c) {\n"
        "            uint is0 = c << 1, is1 = is0 + 1;\n"
        "            uchar sc1, m1, sc2, m2;\n"
        "            if (c < 2) {\n"
        "                sc1 = sc[is0] & 63u; m1 = sc[is0 + 4] & 63u;\n"
        "                sc2 = sc[is1] & 63u; m2 = sc[is1 + 4] & 63u;\n"
        "            } else {\n"
        "                sc1 = (sc[is0 + 4] & 15u) | ((sc[is0 - 4] >> 6) << 4);\n"
        "                m1  = (sc[is0 + 4] >> 4)  | ((sc[is0]     >> 6) << 4);\n"
        "                sc2 = (sc[is1 + 4] & 15u) | ((sc[is1 - 4] >> 6) << 4);\n"
        "                m2  = (sc[is1 + 4] >> 4)  | ((sc[is1]     >> 6) << 4);\n"
        "            }\n"
        "            float d1 = d * float(sc1), min1 = dmin * float(m1);\n"
        "            float d2 = d * float(sc2), min2 = dmin * float(m2);\n"
        "            device const uchar *qsp = qs + (c << 5);\n"
        "            device const float *xp = xp_base + (c << 6);\n"
        "            uint hm0 = 1u << is0, hm1 = 1u << is1;\n"
        "            float sum1 = 0.0f, xsum1 = 0.0f, sum2 = 0.0f, xsum2 = 0.0f;\n"
        "            for (uint l = 0; l < 32; ++l) {\n"
        "                uchar qv = qsp[l], qhb = qh[l];\n"
        "                uint q0 = (qv & 15u) | ((qhb & hm0) != 0 ? 16u : 0u);\n"
        "                uint q1 = (qv >> 4)  | ((qhb & hm1) != 0 ? 16u : 0u);\n"
        "                float x0 = xp[l], x1 = xp[32 + l];\n"
        "                sum1 += x0 * float(q0); xsum1 += x0;\n"
        "                sum2 += x1 * float(q1); xsum2 += x1;\n"
        "            }\n"
        "            acc += (d1 * sum1 - min1 * xsum1) + (d2 * sum2 - min2 * xsum2);\n"
        "        }\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_moe_gemv_q5_k_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device const float *eids [[buffer(2)]], device const float *ewts [[buffer(3)]],\n"
        "    device float *dst [[buffer(4)]], constant uint &out_dim [[buffer(5)]],\n"
        "    constant uint &cols [[buffer(6)]], constant uint &m_cnt [[buffer(7)]],\n"
        "    constant uint &top_k [[buffer(8)]], constant uint &n_experts [[buffer(9)]],\n"
        "    constant uint &x_row_mode [[buffer(10)]], constant uint &xs [[buffer(11)]],\n"
        "    constant uint &is [[buffer(12)]], constant uint &ws [[buffer(13)]],\n"
        "    constant uint &ds [[buffer(14)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]],\n"
        "    uint simd_lane [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],\n"
        "    uint simd_cnt [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint n = tgid.x, m = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (n >= out_dim || m >= m_cnt) return;\n"
        "    uint n_blocks = cols >> 8;\n"
        "    uint row_bytes = n_blocks * 176;\n"
        "    float acc = 0.0f;\n"
        "    for (uint ek = 0; ek < top_k; ++ek) {\n"
        "        int eid = int(rint(eids[m * is + ek]));\n"
        "        float ew = ewts[m * ws + ek];\n"
        "        if (eid < 0 || uint(eid) >= n_experts || ew == 0.0f) continue;\n"
        "        uint w_row_off = (uint(eid) * out_dim + n) * row_bytes;\n"
        "        uint x_row_off = (x_row_mode != 0 ? (m * top_k + ek) : m) * xs;\n"
        "        float eacc = 0.0f;\n"
        "        for (uint b = tid; b < n_blocks; b += tg) {\n"
        "            device const uchar *blk = w + w_row_off + b * 176;\n"
        "            ushort d_bits = ushort(blk[0]) | (ushort(blk[1]) << 8);\n"
        "            ushort dm_bits = ushort(blk[2]) | (ushort(blk[3]) << 8);\n"
        "            float d = float(as_type<half>(d_bits));\n"
        "            float dmin = float(as_type<half>(dm_bits));\n"
        "            device const uchar *sc = blk + 4;\n"
        "            device const uchar *qh = blk + 16;\n"
        "            device const uchar *qs = blk + 48;\n"
        "            device const float *xp_base = x + x_row_off + (b << 8);\n"
        "            for (uint c = 0; c < 4; ++c) {\n"
        "                uint is0 = c << 1, is1 = is0 + 1;\n"
        "                uchar sc1, m1, sc2, m2;\n"
        "                if (c < 2) {\n"
        "                    sc1 = sc[is0] & 63u; m1 = sc[is0 + 4] & 63u;\n"
        "                    sc2 = sc[is1] & 63u; m2 = sc[is1 + 4] & 63u;\n"
        "                } else {\n"
        "                    sc1 = (sc[is0 + 4] & 15u) | ((sc[is0 - 4] >> 6) << 4);\n"
        "                    m1  = (sc[is0 + 4] >> 4)  | ((sc[is0]     >> 6) << 4);\n"
        "                    sc2 = (sc[is1 + 4] & 15u) | ((sc[is1 - 4] >> 6) << 4);\n"
        "                    m2  = (sc[is1 + 4] >> 4)  | ((sc[is1]     >> 6) << 4);\n"
        "                }\n"
        "                float d1 = d * float(sc1), min1 = dmin * float(m1);\n"
        "                float d2 = d * float(sc2), min2 = dmin * float(m2);\n"
        "                device const uchar *qsp = qs + (c << 5);\n"
        "                device const float *xp = xp_base + (c << 6);\n"
        "                uint hm0 = 1u << is0, hm1 = 1u << is1;\n"
        "                float sum1 = 0.0f, xsum1 = 0.0f, sum2 = 0.0f, xsum2 = 0.0f;\n"
        "                for (uint l = 0; l < 32; ++l) {\n"
        "                    uchar qv = qsp[l], qhb = qh[l];\n"
        "                    uint q0 = (qv & 15u) | ((qhb & hm0) != 0 ? 16u : 0u);\n"
        "                    uint q1 = (qv >> 4)  | ((qhb & hm1) != 0 ? 16u : 0u);\n"
        "                    float x0 = xp[l], x1 = xp[32 + l];\n"
        "                    sum1 += x0 * float(q0); xsum1 += x0;\n"
        "                    sum2 += x1 * float(q1); xsum2 += x1;\n"
        "                }\n"
        "                eacc += (d1 * sum1 - min1 * xsum1) + (d2 * sum2 - min2 * xsum2);\n"
        "            }\n"
        "        }\n"
        "        acc += ew * eacc;\n"
        "    }\n"
        "    acc = simd_sum(acc);\n"
        "    if (simd_lane == 0) sh[simd_gid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float v = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        v = simd_sum(v);\n"
        "        if (simd_lane == 0) dst[m * ds + n] = v;\n"
        "    }\n"
        "}\n"
        "kernel void xu_paged_kv_append_f16(device const float *src [[buffer(0)]], device const float *bt [[buffer(1)]],\n"
        "    device half *pages [[buffer(2)]], constant uint &seq_len [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &n_seqs [[buffer(5)]], constant uint &pos_off [[buffer(6)]],\n"
        "    constant uint &block_size [[buffer(7)]], constant uint &num_blocks [[buffer(8)]],\n"
        "    constant uint &ss [[buffer(9)]], constant uint &bts [[buffer(10)]], constant uint &ds [[buffer(11)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint c = gid.x, t = gid.y, b = gid.z;\n"
        "    if (c < cols && t < seq_len && b < n_seqs) {\n"
        "        uint logic_pos = pos_off + t;\n"
        "        int pb = int(rint(bt[b * bts + (logic_pos / block_size)]));\n"
        "        if (pb >= 0 && uint(pb) < num_blocks) {\n"
        "            uint row_idx = uint(pb) * block_size + (logic_pos % block_size);\n"
        "            pages[row_idx * ds + c] = half(src[(b * seq_len + t) * ss + c]);\n"
        "        }\n"
        "    }\n"
        "}\n";
    NSError *error = nil;
    id<MTLLibrary> lib = [d.gpu newLibraryWithSource:src options:nil error:&error];
    if (!lib) return NO;
    id<MTLComputePipelineState> p0 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_f16_to_f32"] error:&error];
    id<MTLComputePipelineState> p1 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_f32_to_f16"] error:&error];
    id<MTLComputePipelineState> p2 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q4_0_f32"] error:&error];
    id<MTLComputePipelineState> p3 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q8_0_f32"] error:&error];
    id<MTLComputePipelineState> p4 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q4_k_f32"] error:&error];
    id<MTLComputePipelineState> p5 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q5_0_f32"] error:&error];
    id<MTLComputePipelineState> p6 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q6_k_f32"] error:&error];
    id<MTLComputePipelineState> p7 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_kv_cache_append_f16"] error:&error];
    id<MTLComputePipelineState> p8 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_attention_decode_f16"] error:&error];
    id<MTLComputePipelineState> p9 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_causal_attn_f16"] error:&error];
    id<MTLComputePipelineState> p10 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_moe_gemv_q4_0_f32"] error:&error];
    id<MTLComputePipelineState> p11 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_moe_gemv_q8_0_f32"] error:&error];
    id<MTLComputePipelineState> p12 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_moe_gemv_q4_k_f32"] error:&error];
    id<MTLComputePipelineState> p13 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_moe_gemv_q6_k_f32"] error:&error];
    id<MTLComputePipelineState> p14 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_paged_attn_decode_f16"] error:&error];
    id<MTLComputePipelineState> p15 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q5_k_f32"] error:&error];
    id<MTLComputePipelineState> p16 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_moe_gemv_q5_k_f32"] error:&error];
    id<MTLComputePipelineState> p17 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_paged_kv_append_f16"] error:&error];
    if (!p0 || !p1 || !p2 || !p3 || !p4 || !p5 || !p6 || !p7 || !p8 || !p9 || !p10 || !p11 || !p12 || !p13 || !p14 || !p15 || !p16 || !p17) return NO;
    d.f16ToF32Pipeline = p0;
    d.f32ToF16Pipeline = p1;
    d.gemvQ40Pipeline = p2;
    d.gemvQ80Pipeline = p3;
    d.gemvQ4KPipeline = p4;
    d.gemvQ50Pipeline = p5;
    d.gemvQ6KPipeline = p6;
    d.kvAppendF16Pipeline = p7;
    d.attnDecodeF16Pipeline = p8;
    d.causalAttnF16Pipeline = p9;
    d.moeGemvQ40Pipeline = p10;
    d.moeGemvQ80Pipeline = p11;
    d.moeGemvQ4KPipeline = p12;
    d.moeGemvQ6KPipeline = p13;
    d.pagedAttnDecodeF16Pipeline = p14;
    d.gemvQ5KPipeline = p15;
    d.moeGemvQ5KPipeline = p16;
    d.pagedKvAppendF16Pipeline = p17;
    return YES;
}

static BOOL f16_normal_inputs(XMMatrix *m) {
    if (m.numberClass != MATRIX_CLASS_UNKNOWN) return m.numberClass == MATRIX_CLASS_NORMAL;
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    const uint8_t *source = m.memory.contents;
    for (NSUInteger row = 0; row < rows; ++row) for (NSUInteger col = 0; col < columns; ++col) {
        uint16_t bits;
        memcpy(&bits, source + row * stride + col * 2, 2);
        if ((bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) {
            m.numberClass = MATRIX_CLASS_UNSUPPORTED;
            return NO;
        }
    }
    m.numberClass = MATRIX_CLASS_NORMAL;
    return YES;
}

static inline BOOL batch_f16_normal_input(XMBatch *batch, XMMatrix *m) {
    if (batch && [batch.writeBuffers indexOfObjectIdenticalTo:m] != NSNotFound) return YES;
    return f16_normal_inputs(m);
}

xu_int XM_API(matrix_f16_new)(xu_int h, xu_int rows, xu_int columns) { @autoreleasepool {
    if (rows < 0 || columns < 0 || rows > MATRIX_MAX_DIM || columns > MATRIX_MAX_DIM) return XM_BOUNDS;
    uint64_t cells = (uint64_t)rows * (uint64_t)columns;
    if (cells > MATRIX_MAX_CELLS) return XM_BOUNDS;
    NSUInteger stride = [MPSMatrixDescriptor rowBytesFromColumns:MAX((NSUInteger)1, (NSUInteger)columns)
        dataType:MPSDataTypeFloat16];
    if (!stride || stride < (uint64_t)columns * 2 || (rows && stride > SIZE_MAX / (NSUInteger)rows)) return XM_BOUNDS;
    NSUInteger bytes = cells ? stride * (NSUInteger)rows : 16;
    ENTER; XMDevice *d = lookup(h, DEVICE);
    if (!d) RETURN(XM_HANDLE);
    if (!MPSSupportsMTLDevice(d.gpu)) RETURN(XM_UNAVAILABLE);
    if (bytes > d.gpu.maxBufferLength) RETURN(XM_BOUNDS);
    d.children++;
    os_unfair_lock_unlock(&gate);
    XMMatrix *m = nil;
    @try {
        id<MTLBuffer> memory = [d.gpu newBufferWithLength:MAX((NSUInteger)16, bytes) options:MTLResourceStorageModeShared];
        if (memory) {
            m = [XMMatrix new];
            m.owner = d; m.memory = memory; m.words = cells;
            m.rows = (NSUInteger)rows; m.columns = (NSUInteger)columns; m.rowBytes = stride;
            memset(memory.contents, 0, memory.length);
            m.numberClass = MATRIX_CLASS_NORMAL;
            if (cells) {
                MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:m.rows columns:m.columns
                    rowBytes:stride dataType:MPSDataTypeFloat16];
                m.matrix = [[MPSMatrix alloc] initWithBuffer:memory descriptor:desc];
                if (!m.matrix) m = nil;
            }
        }
    } @catch (NSException *exception) { (void)exception; m = nil; }
    ENTER;
    xu_int result = m ? insert(m, MATRIX_F16) : XM_ALLOC;
    if (result < 0) d.children--;
    RETURN(result);
} }

xu_int XM_API(matrix_f16_info)(xu_int h, xu_int key) { @autoreleasepool {
    ENTER; XMMatrix *m = lookup(h, MATRIX_F16);
    if (!m) RETURN(XM_HANDLE);
    if (key == 1) RETURN((xu_int)m.rows);
    if (key == 2) RETURN((xu_int)m.columns);
    if (key == 3) RETURN((xu_int)m.rowBytes);
    if (key == 4) RETURN((xu_int)m.words);
    RETURN(XM_BOUNDS);
} }

xu_int XM_API(matrix_f16_write)(xu_int h, const xu_i64_span *source) { @autoreleasepool {
    if (!source || !host_range(source->len, source->data, 0, source->len)) return XM_BOUNDS;
    ENTER; XMMatrix *m = lookup(h, MATRIX_F16);
    if (!m) RETURN(XM_HANDLE);
    if ((uint64_t)source->len != m.words) RETURN(XM_BOUNDS);
    if (!can_host_write(m)) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    const xu_int *input = source->data;
    int numberClass = MATRIX_CLASS_NORMAL;
    for (xu_int i = 0; i < source->len; ++i) {
        if (input[i] < 0 || (xu_uint)input[i] > UINT16_MAX) RETURN(XM_BOUNDS);
        uint16_t bits = (uint16_t)input[i];
        if ((bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) numberClass = MATRIX_CLASS_UNSUPPORTED;
    }
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    uint8_t *base = m.memory.contents;
    if (m.words) for (NSUInteger row = 0; row < rows; ++row) {
        uint8_t *output = base + row * stride;
        for (NSUInteger col = 0; col < columns; ++col) {
            uint16_t bits = (uint16_t)input[row * columns + col];
            memcpy(output + col * 2, &bits, 2);
        }
    }
    m.numberClass = numberClass;
    RETURN(0);
} }

xu_int XM_API(matrix_f16_write_bytes)(xu_int h, const xu_u8_span *bytes) { @autoreleasepool {
    if (!bytes || bytes->len < 0 || (bytes->len > 0 && !bytes->data)) return XM_BOUNDS;
    ENTER; XMMatrix *m = lookup(h, MATRIX_F16);
    if (!m) RETURN(XM_HANDLE);
    if ((uint64_t)bytes->len != m.words * 2ULL) RETURN(XM_BOUNDS);
    if (!can_host_write(m)) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    const uint8_t *src = bytes->data;
    int numberClass = MATRIX_CLASS_NORMAL;
    for (uint64_t i = 0; i < m.words; ++i) {
        uint16_t bits;
        memcpy(&bits, src + i * 2ULL, 2);
        if ((bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) {
            numberClass = MATRIX_CLASS_UNSUPPORTED;
            break;
        }
    }
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    uint8_t *base = m.memory.contents;
    if (m.words && columns) {
        size_t rowCopy = (size_t)columns * 2;
        for (NSUInteger row = 0; row < rows; ++row) {
            memcpy(base + row * stride, src + (size_t)row * rowCopy, rowCopy);
        }
    }
    m.numberClass = numberClass;
    RETURN(0);
} }

xu_int XM_API(matrix_f16_copy)(xu_int h, const xu_i64_span_mut *destination) { @autoreleasepool {
    if (!destination || !host_range(destination->len, destination->data, 0, destination->len)) return XM_BOUNDS;
    ENTER; XMMatrix *m = lookup(h, MATRIX_F16);
    if (!m) RETURN(XM_HANDLE);
    if ((uint64_t)destination->len < m.words) RETURN(XM_BOUNDS);
    if (!can_host_read(m)) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    if (!m.words) RETURN(0);
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    const uint8_t *input = m.memory.contents;
    xu_int *output = destination->data;
    for (NSUInteger row = 0; row < rows; ++row) {
        const uint8_t *rowp = input + row * stride;
        for (NSUInteger col = 0; col < columns; ++col) {
            uint16_t bits;
            memcpy(&bits, rowp + col * 2, 2);
            output[row * columns + col] = (xu_int)bits;
        }
    }
    RETURN(0);
} }

xu_int XM_API(matrix_f16_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_F16, NO); } }
xu_int XM_API(matrix_f16_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_F16, YES); } }

static xu_int gemm_f16_impl(xu_int bh, xu_int dst, xu_int ah, xu_int bh_m, BOOL transposeRight) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F16), *a = lookup(ah, MATRIX_F16), *b = lookup(bh_m, MATRIX_F16);
    if (!c || !a || !b) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (a.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    NSUInteger innerB = transposeRight ? b.columns : b.rows;
    NSUInteger outColsB = transposeRight ? b.rows : b.columns;
    if (a.columns != innerB || c.rows != a.rows || c.columns != outColsB) RETURN(XM_SHAPE);
    if (c == a || c == b) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (a == b) ? @[a] : @[a, b];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || a.poisoned || b.poisoned) RETURN(XM_COMMAND);
    if (!batch_f16_normal_input(batch, a) || !batch_f16_normal_input(batch, b)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!MPSSupportsMTLDevice(d.gpu)) RETURN(XM_UNAVAILABLE);
    if (!c.words || !a.columns) {
        xu_int fence = finish_op(batch, d, c, inputs, nil, nil);
        if (fence >= 0) {
            if (c.words && !batch) memset(c.memory.contents, 0, c.memory.length);
            c.numberClass = MATRIX_CLASS_NORMAL;
        }
        RETURN(fence);
    }
    id<MTLCommandBuffer> command = nil;
    MPSMatrixMultiplication *kernel = nil;
    @try {
        kernel = matrix_kernel_typed(d, c, a, NO, transposeRight, MPSDataTypeFloat16);
        if (!kernel) RETURN(XM_PIPELINE);
        command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
        if (!command) RETURN(XM_COMMAND);
        [kernel encodeToCommandBuffer:command leftMatrix:a.matrix rightMatrix:b.matrix resultMatrix:c.matrix];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, kernel);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(gemm_f16)(xu_int dst, xu_int a, xu_int b) { return gemm_f16_impl(-1, dst, a, b, NO); }
xu_int XM_API(gemm_f16_nt)(xu_int dst, xu_int a, xu_int b) { return gemm_f16_impl(-1, dst, a, b, YES); }
xu_int XM_API(batch_gemm_f16)(xu_int batch, xu_int dst, xu_int a, xu_int b) { return gemm_f16_impl(batch < 0 ? 0 : batch, dst, a, b, NO); }
xu_int XM_API(batch_gemm_f16_nt)(xu_int batch, xu_int dst, xu_int a, xu_int b) { return gemm_f16_impl(batch < 0 ? 0 : batch, dst, a, b, YES); }

static xu_int convert_f16_f32_impl(xu_int bh, xu_int dst, xu_int src, BOOL to_f16) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, to_f16 ? MATRIX_F16 : MATRIX_F32);
    XMMatrix *x = lookup(src, to_f16 ? MATRIX_F32 : MATRIX_F16);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (to_f16) {
        if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    } else {
        if (!batch_f16_normal_input(batch, x)) RETURN(XM_NUMERICS);
    }
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLComputePipelineState> pipe = to_f16 ? d.f32ToF16Pipeline : d.f16ToF32Pipeline;
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns;
        uint32_t ss = (uint32_t)(x.rowBytes / (to_f16 ? 4 : 2));
        uint32_t ds = (uint32_t)(c.rowBytes / (to_f16 ? 2 : 4));
        NSUInteger tx = MIN((NSUInteger)16, pipe.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, pipe.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:pipe];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&rows length:sizeof(rows) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&ss length:sizeof(ss) atIndex:4];
        [enc setBytes:&ds length:sizeof(ds) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, pipe);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(f16_to_f32)(xu_int dst, xu_int src) { return convert_f16_f32_impl(-1, dst, src, NO); }
xu_int XM_API(f32_to_f16)(xu_int dst, xu_int src) { return convert_f16_f32_impl(-1, dst, src, YES); }
xu_int XM_API(batch_f16_to_f32)(xu_int batch, xu_int dst, xu_int src) { return convert_f16_f32_impl(batch < 0 ? 0 : batch, dst, src, NO); }
xu_int XM_API(batch_f32_to_f16)(xu_int batch, xu_int dst, xu_int src) { return convert_f16_f32_impl(batch < 0 ? 0 : batch, dst, src, YES); }

static xu_int quant_matrix_new(xu_int h, xu_int rows, xu_int columns, xu_int block_cols,
                               xu_int block_bytes, int kind) { @autoreleasepool {
    if (rows < 0 || columns < 0 || rows > MATRIX_MAX_DIM || columns > MATRIX_MAX_DIM) return XM_BOUNDS;
    if ((columns % block_cols) != 0) return XM_SHAPE;
    uint64_t cells = (uint64_t)rows * (uint64_t)columns;
    if (cells > MATRIX_MAX_CELLS) return XM_BOUNDS;
    NSUInteger bpr = (NSUInteger)(columns / block_cols);
    uint64_t byteLen = (uint64_t)rows * (uint64_t)bpr * (uint64_t)block_bytes;
    ENTER; XMDevice *d = lookup(h, DEVICE);
    if (!d) RETURN(XM_HANDLE);
    if (byteLen > d.gpu.maxBufferLength) RETURN(XM_BOUNDS);
    d.children++;
    os_unfair_lock_unlock(&gate);
    XMQ4Matrix *m = nil;
    @try {
        id<MTLBuffer> memory = [d.gpu newBufferWithLength:MAX((NSUInteger)16, (NSUInteger)byteLen) options:MTLResourceStorageModeShared];
        if (memory) {
            m = [XMQ4Matrix new];
            m.owner = d; m.memory = memory; m.words = byteLen;
            m.rows = (NSUInteger)rows; m.columns = (NSUInteger)columns;
            m.blocksPerRow = bpr; m.byteLen = (NSUInteger)byteLen;
            memset(memory.contents, 0, memory.length);
            m.numberClass = MATRIX_CLASS_NORMAL;
        }
    } @catch (NSException *exception) { (void)exception; m = nil; }
    ENTER;
    xu_int result = m ? insert(m, kind) : XM_ALLOC;
    if (result < 0) d.children--;
    RETURN(result);
} }

static xu_int quant_matrix_info(xu_int h, xu_int key, int kind) { @autoreleasepool {
    ENTER; XMQ4Matrix *m = lookup(h, kind);
    if (!m) RETURN(XM_HANDLE);
    if (key == 1) RETURN((xu_int)m.rows);
    if (key == 2) RETURN((xu_int)m.columns);
    if (key == 3) RETURN((xu_int)m.byteLen);
    if (key == 4) RETURN((xu_int)m.blocksPerRow);
    RETURN(XM_BOUNDS);
} }

static xu_int quant_matrix_write(xu_int h, const xu_u8_span *bytes, int kind,
                                 NSUInteger block_bytes, NSUInteger d_offset, BOOL check_second_f16) { @autoreleasepool {
    if (!bytes || bytes->len < 0 || (bytes->len > 0 && !bytes->data)) return XM_BOUNDS;
    ENTER; XMQ4Matrix *m = lookup(h, kind);
    if (!m) RETURN(XM_HANDLE);
    if ((xu_uint)bytes->len != m.byteLen) RETURN(XM_BOUNDS);
    if (!can_host_write(m)) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    int numberClass = MATRIX_CLASS_NORMAL;
    NSUInteger nblocks = m.rows * m.blocksPerRow;
    const uint8_t *src = bytes->data;
    for (NSUInteger b = 0; b < nblocks; ++b) {
        NSUInteger off = b * block_bytes;
        uint16_t d_bits = (uint16_t)src[off + d_offset] | ((uint16_t)src[off + d_offset + 1] << 8);
        if ((d_bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) {
            numberClass = MATRIX_CLASS_UNSUPPORTED;
            break;
        }
        if (check_second_f16) {
            uint16_t dm_bits = (uint16_t)src[off + 2] | ((uint16_t)src[off + 3] << 8);
            if ((dm_bits & UINT16_C(0x7c00)) == UINT16_C(0x7c00)) {
                numberClass = MATRIX_CLASS_UNSUPPORTED;
                break;
            }
        }
    }
    if (m.byteLen) memcpy(m.memory.contents, src, m.byteLen);
    m.numberClass = numberClass;
    RETURN(0);
} }

static xu_int gemv_quant_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int wh, int kind) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    XMQ4Matrix *w = lookup(wh, kind);
    if (!c || !x || !w) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner || w.owner != c.owner) RETURN(XM_DEVICE);
    if (x.columns != w.columns || c.rows != x.rows || c.columns != w.rows) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x, w];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || w.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x) || w.numberClass != MATRIX_CLASS_NORMAL) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words || !w.columns) {
        xu_int fence = finish_op(batch, d, c, inputs, nil, nil);
        if (fence >= 0) {
            if (c.words && !batch) memset(c.memory.contents, 0, c.memory.length);
            c.numberClass = MATRIX_CLASS_NORMAL;
        }
        RETURN(fence);
    }
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLComputePipelineState> pipe = d.gemvQ40Pipeline;
    if (kind == MATRIX_Q8_0) pipe = d.gemvQ80Pipeline;
    else if (kind == MATRIX_Q4_K) pipe = d.gemvQ4KPipeline;
    else if (kind == MATRIX_Q5_0) pipe = d.gemvQ50Pipeline;
    else if (kind == MATRIX_Q6_K) pipe = d.gemvQ6KPipeline;
    else if (kind == MATRIX_Q5_K) pipe = d.gemvQ5KPipeline;
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)w.rows, cols = (uint32_t)w.columns, m_cnt = (uint32_t)x.rows;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(pipe);
        [enc setComputePipelineState:pipe];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:w.memory offset:0 atIndex:1];
        [enc setBuffer:c.memory offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&m_cnt length:sizeof(m_cnt) atIndex:5];
        [enc setBytes:&xs length:sizeof(xs) atIndex:6];
        [enc setBytes:&ds length:sizeof(ds) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(rows, m_cnt, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, pipe);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(matrix_q4_0_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 32, 18, MATRIX_Q4_0); }
xu_int XM_API(matrix_q4_0_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q4_0); }
xu_int XM_API(matrix_q4_0_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q4_0, 18, 0, NO); }
xu_int XM_API(matrix_q4_0_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_0, NO); } }
xu_int XM_API(matrix_q4_0_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_0, YES); } }
xu_int XM_API(gemv_q4_0_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q4_0); }
xu_int XM_API(batch_gemv_q4_0_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q4_0); }

xu_int XM_API(matrix_q8_0_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 32, 34, MATRIX_Q8_0); }
xu_int XM_API(matrix_q8_0_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q8_0); }
xu_int XM_API(matrix_q8_0_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q8_0, 34, 0, NO); }
xu_int XM_API(matrix_q8_0_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q8_0, NO); } }
xu_int XM_API(matrix_q8_0_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q8_0, YES); } }
xu_int XM_API(gemv_q8_0_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q8_0); }
xu_int XM_API(batch_gemv_q8_0_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q8_0); }

xu_int XM_API(matrix_q4_k_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 256, 144, MATRIX_Q4_K); }
xu_int XM_API(matrix_q4_k_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q4_K); }
xu_int XM_API(matrix_q4_k_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q4_K, 144, 0, YES); }
xu_int XM_API(matrix_q4_k_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_K, NO); } }
xu_int XM_API(matrix_q4_k_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_K, YES); } }
xu_int XM_API(gemv_q4_k_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q4_K); }
xu_int XM_API(batch_gemv_q4_k_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q4_K); }

xu_int XM_API(matrix_q5_0_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 32, 22, MATRIX_Q5_0); }
xu_int XM_API(matrix_q5_0_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q5_0); }
xu_int XM_API(matrix_q5_0_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q5_0, 22, 0, NO); }
xu_int XM_API(matrix_q5_0_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q5_0, NO); } }
xu_int XM_API(matrix_q5_0_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q5_0, YES); } }
xu_int XM_API(gemv_q5_0_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q5_0); }
xu_int XM_API(batch_gemv_q5_0_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q5_0); }

xu_int XM_API(matrix_q6_k_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 256, 210, MATRIX_Q6_K); }
xu_int XM_API(matrix_q6_k_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q6_K); }
xu_int XM_API(matrix_q6_k_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q6_K, 210, 208, NO); }
xu_int XM_API(matrix_q6_k_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q6_K, NO); } }
xu_int XM_API(matrix_q6_k_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q6_K, YES); } }
xu_int XM_API(gemv_q6_k_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q6_K); }
xu_int XM_API(batch_gemv_q6_k_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q6_K); }

xu_int XM_API(matrix_q5_k_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 256, 176, MATRIX_Q5_K); }
xu_int XM_API(matrix_q5_k_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q5_K); }
xu_int XM_API(matrix_q5_k_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q5_K, 176, 0, YES); }
xu_int XM_API(matrix_q5_k_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q5_K, NO); } }
xu_int XM_API(matrix_q5_k_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q5_K, YES); } }
xu_int XM_API(gemv_q5_k_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q5_K); }
xu_int XM_API(batch_gemv_q5_k_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q5_K); }

static xu_int kv_cache_append_f16_impl(xu_int bh, xu_int cacheh, xu_int srch, xu_int pos_offset) { @autoreleasepool {
    if (pos_offset < 0) return XM_BOUNDS;
    ENTER;
    XMMatrix *c = lookup(cacheh, MATRIX_F16), *s = lookup(srch, MATRIX_F32);
    if (!c || !s) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (s.owner != c.owner) RETURN(XM_DEVICE);
    if (s.columns != c.columns || (uint64_t)pos_offset + (uint64_t)s.rows > (uint64_t)c.rows) RETURN(XM_SHAPE);
    NSArray<XMBuffer *> *inputs = @[s];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || s.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, s)) RETURN(XM_NUMERICS);
    BOOL keep_normal = (!batch && c.numberClass == MATRIX_CLASS_NORMAL && s.numberClass == MATRIX_CLASS_NORMAL);
    XMDevice *d = c.owner;
    if (!s.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)s.rows, cols = (uint32_t)s.columns;
        uint32_t ss = (uint32_t)(s.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 2);
        uint32_t po = (uint32_t)pos_offset;
        [enc setComputePipelineState:d.kvAppendF16Pipeline];
        [enc setBuffer:s.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&rows length:sizeof(rows) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&ss length:sizeof(ss) atIndex:4];
        [enc setBytes:&ds length:sizeof(ds) atIndex:5];
        [enc setBytes:&po length:sizeof(po) atIndex:6];
        NSUInteger tx = 16, ty = 16;
        if (d.kvAppendF16Pipeline.maxTotalThreadsPerThreadgroup < tx * ty) { tx = 8; ty = 8; }
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.kvAppendF16Pipeline);
    if (handle >= 0) c.numberClass = keep_normal ? MATRIX_CLASS_NORMAL : MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(kv_cache_append_f16)(xu_int cache, xu_int src, xu_int pos_offset) {
    return kv_cache_append_f16_impl(-1, cache, src, pos_offset);
}
xu_int XM_API(batch_kv_cache_append_f16)(xu_int batch, xu_int cache, xu_int src, xu_int pos_offset) {
    return kv_cache_append_f16_impl(batch < 0 ? 0 : batch, cache, src, pos_offset);
}

static xu_int attention_decode_f16_impl(xu_int bh, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                          xu_int kv_len, xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) { @autoreleasepool {
    if (kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || (n_heads % n_kv_heads) != 0) return XM_SHAPE;
    if (scale_bits < 0 || (xu_uint)scale_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t sb = (uint32_t)scale_bits;
    float scale = 0.0f;
    memcpy(&scale, &sb, 4);
    if (!isfinite(scale) || scale <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *q = lookup(qh, MATRIX_F32),
             *k = lookup(kh, MATRIX_F16), *v = lookup(vh, MATRIX_F16);
    if (!c || !q || !k || !v) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (q.owner != c.owner || k.owner != c.owner || v.owner != c.owner) RETURN(XM_DEVICE);
    if (q.rows != 1 || c.rows != 1 || c.columns != q.columns || q.columns == 0 || (q.columns % (NSUInteger)n_heads) != 0) RETURN(XM_SHAPE);
    NSUInteger head_dim = q.columns / (NSUInteger)n_heads;
    if (k.columns != (NSUInteger)n_kv_heads * head_dim || v.columns != k.columns ||
        (NSUInteger)kv_len > k.rows || (NSUInteger)kv_len > v.rows) RETURN(XM_SHAPE);
    if (c == q) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[q, k, v];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || q.poisoned || k.poisoned || v.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, q) || !batch_f16_normal_input(batch, k) || !batch_f16_normal_input(batch, v)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    NSUInteger tg = ai_row_tg_width(d.attnDecodeF16Pipeline);
    if (head_dim > 8 * tg) RETURN(XM_SHAPE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t kvl = (uint32_t)kv_len, nh = (uint32_t)n_heads, kvm = (uint32_t)(n_heads / n_kv_heads);
        uint32_t hd = (uint32_t)head_dim, ks = (uint32_t)(k.rowBytes / 2), vs = (uint32_t)(v.rowBytes / 2);
        [enc setComputePipelineState:d.attnDecodeF16Pipeline];
        [enc setBuffer:q.memory offset:0 atIndex:0];
        [enc setBuffer:k.memory offset:0 atIndex:1];
        [enc setBuffer:v.memory offset:0 atIndex:2];
        [enc setBuffer:c.memory offset:0 atIndex:3];
        [enc setBytes:&kvl length:sizeof(kvl) atIndex:4];
        [enc setBytes:&nh length:sizeof(nh) atIndex:5];
        [enc setBytes:&kvm length:sizeof(kvm) atIndex:6];
        [enc setBytes:&hd length:sizeof(hd) atIndex:7];
        [enc setBytes:&ks length:sizeof(ks) atIndex:8];
        [enc setBytes:&vs length:sizeof(vs) atIndex:9];
        [enc setBytes:&scale length:sizeof(scale) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(nh, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.attnDecodeF16Pipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(attention_decode_f16)(xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                    xu_int kv_len, xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return attention_decode_f16_impl(-1, dst, qh, kh, vh, kv_len, n_heads, n_kv_heads, scale_bits);
}
xu_int XM_API(batch_attention_decode_f16)(xu_int batch, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                          xu_int kv_len, xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return attention_decode_f16_impl(batch < 0 ? 0 : batch, dst, qh, kh, vh, kv_len, n_heads, n_kv_heads, scale_bits);
}

static xu_int causal_attention_f16_impl(xu_int bh, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                          xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) { @autoreleasepool {
    if (n_heads <= 0 || n_kv_heads <= 0 || (n_heads % n_kv_heads) != 0) return XM_SHAPE;
    if (scale_bits < 0 || (xu_uint)scale_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t sb = (uint32_t)scale_bits;
    float scale = 0.0f;
    memcpy(&scale, &sb, 4);
    if (!isfinite(scale) || scale <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *q = lookup(qh, MATRIX_F32),
             *k = lookup(kh, MATRIX_F16), *v = lookup(vh, MATRIX_F16);
    if (!c || !q || !k || !v) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, c, &berr); if (!batch) RETURN(berr); }
    if (q.owner != c.owner || k.owner != c.owner || v.owner != c.owner) RETURN(XM_DEVICE);
    if (q.columns == 0 || (q.columns % (NSUInteger)n_heads) != 0) RETURN(XM_SHAPE);
    NSUInteger head_dim = q.columns / (NSUInteger)n_heads;
    if (c.rows != q.rows || c.columns != q.columns ||
        k.columns != (NSUInteger)n_kv_heads * head_dim ||
        v.rows != k.rows || v.columns != k.columns || k.rows < q.rows) RETURN(XM_SHAPE);
    if (c == q) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[q, k, v];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || q.poisoned || k.poisoned || v.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, q) || !batch_f16_normal_input(batch, k) || !batch_f16_normal_input(batch, v)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    NSUInteger tg = ai_row_tg_width(d.causalAttnF16Pipeline);
    if (head_dim > 8 * tg) RETURN(XM_SHAPE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t seq_len = (uint32_t)q.rows, kv_len = (uint32_t)k.rows;
        uint32_t nh = (uint32_t)n_heads, kvm = (uint32_t)(n_heads / n_kv_heads), hd = (uint32_t)head_dim;
        uint32_t qs = (uint32_t)(q.rowBytes / 4), ks = (uint32_t)(k.rowBytes / 2);
        uint32_t vs = (uint32_t)(v.rowBytes / 2), ds = (uint32_t)(c.rowBytes / 4);
        [enc setComputePipelineState:d.causalAttnF16Pipeline];
        [enc setBuffer:q.memory offset:0 atIndex:0];
        [enc setBuffer:k.memory offset:0 atIndex:1];
        [enc setBuffer:v.memory offset:0 atIndex:2];
        [enc setBuffer:c.memory offset:0 atIndex:3];
        [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:4];
        [enc setBytes:&kv_len length:sizeof(kv_len) atIndex:5];
        [enc setBytes:&nh length:sizeof(nh) atIndex:6];
        [enc setBytes:&kvm length:sizeof(kvm) atIndex:7];
        [enc setBytes:&hd length:sizeof(hd) atIndex:8];
        [enc setBytes:&qs length:sizeof(qs) atIndex:9];
        [enc setBytes:&ks length:sizeof(ks) atIndex:10];
        [enc setBytes:&vs length:sizeof(vs) atIndex:11];
        [enc setBytes:&ds length:sizeof(ds) atIndex:12];
        [enc setBytes:&scale length:sizeof(scale) atIndex:13];
        [enc dispatchThreadgroups:MTLSizeMake(nh, seq_len, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.causalAttnF16Pipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(causal_attention_f16)(xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                    xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return causal_attention_f16_impl(-1, dst, qh, kh, vh, n_heads, n_kv_heads, scale_bits);
}
xu_int XM_API(batch_causal_attention_f16)(xu_int batch, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                          xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return causal_attention_f16_impl(batch < 0 ? 0 : batch, dst, qh, kh, vh, n_heads, n_kv_heads, scale_bits);
}

static xu_int moe_gemv_quant_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int wh,
                                      xu_int eidh, xu_int ewh, xu_int n_experts, int kind) { @autoreleasepool {
    if (n_experts <= 0) return XM_SHAPE;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32),
             *eids = lookup(eidh, MATRIX_F32), *ewts = lookup(ewh, MATRIX_F32);
    XMQ4Matrix *w = lookup(wh, kind);
    if (!c || !x || !eids || !ewts || !w) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner || w.owner != c.owner || eids.owner != c.owner || ewts.owner != c.owner) RETURN(XM_DEVICE);
    NSUInteger m_cnt = c.rows, out_dim = c.columns, top_k = eids.columns;
    if (top_k == 0 || eids.rows != m_cnt || ewts.rows != m_cnt || ewts.columns != top_k) RETURN(XM_SHAPE);
    if (x.columns != w.columns || w.rows != (NSUInteger)n_experts * out_dim) RETURN(XM_SHAPE);
    BOOL shared_x = (x.rows == m_cnt);
    BOOL per_exp_x = (x.rows == m_cnt * top_k);
    if (!shared_x && !per_exp_x) RETURN(XM_SHAPE);
    if (c == x || c == eids || c == ewts) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x, w, eids, ewts];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || w.poisoned || eids.poisoned || ewts.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x) || !batch_normal_input(batch, eids) || !batch_normal_input(batch, ewts) ||
        w.numberClass != MATRIX_CLASS_NORMAL) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words || !w.columns) {
        xu_int fence = finish_op(batch, d, c, inputs, nil, nil);
        if (fence >= 0) {
            if (c.words && !batch) memset(c.memory.contents, 0, c.memory.length);
            c.numberClass = MATRIX_CLASS_NORMAL;
        }
        RETURN(fence);
    }
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLComputePipelineState> pipe = d.moeGemvQ40Pipeline;
    if (kind == MATRIX_Q8_0) pipe = d.moeGemvQ80Pipeline;
    else if (kind == MATRIX_Q4_K) pipe = d.moeGemvQ4KPipeline;
    else if (kind == MATRIX_Q6_K) pipe = d.moeGemvQ6KPipeline;
    else if (kind == MATRIX_Q5_K) pipe = d.moeGemvQ5KPipeline;
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t od = (uint32_t)out_dim, cols = (uint32_t)w.columns, mc = (uint32_t)m_cnt;
        uint32_t tk = (uint32_t)top_k, ne = (uint32_t)n_experts;
        uint32_t x_mode = per_exp_x ? 1u : 0u;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), is = (uint32_t)(eids.rowBytes / 4);
        uint32_t ws = (uint32_t)(ewts.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(pipe);
        [enc setComputePipelineState:pipe];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:w.memory offset:0 atIndex:1];
        [enc setBuffer:eids.memory offset:0 atIndex:2];
        [enc setBuffer:ewts.memory offset:0 atIndex:3];
        [enc setBuffer:c.memory offset:0 atIndex:4];
        [enc setBytes:&od length:sizeof(od) atIndex:5];
        [enc setBytes:&cols length:sizeof(cols) atIndex:6];
        [enc setBytes:&mc length:sizeof(mc) atIndex:7];
        [enc setBytes:&tk length:sizeof(tk) atIndex:8];
        [enc setBytes:&ne length:sizeof(ne) atIndex:9];
        [enc setBytes:&x_mode length:sizeof(x_mode) atIndex:10];
        [enc setBytes:&xs length:sizeof(xs) atIndex:11];
        [enc setBytes:&is length:sizeof(is) atIndex:12];
        [enc setBytes:&ws length:sizeof(ws) atIndex:13];
        [enc setBytes:&ds length:sizeof(ds) atIndex:14];
        [enc dispatchThreadgroups:MTLSizeMake(od, mc, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, pipe);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(moe_gemv_q4_0_f32)(xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(-1, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q4_0);
}
xu_int XM_API(batch_moe_gemv_q4_0_f32)(xu_int batch, xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q4_0);
}
xu_int XM_API(moe_gemv_q8_0_f32)(xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(-1, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q8_0);
}
xu_int XM_API(batch_moe_gemv_q8_0_f32)(xu_int batch, xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q8_0);
}
xu_int XM_API(moe_gemv_q4_k_f32)(xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(-1, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q4_K);
}
xu_int XM_API(batch_moe_gemv_q4_k_f32)(xu_int batch, xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q4_K);
}
xu_int XM_API(moe_gemv_q6_k_f32)(xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(-1, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q6_K);
}
xu_int XM_API(batch_moe_gemv_q6_k_f32)(xu_int batch, xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q6_K);
}
xu_int XM_API(moe_gemv_q5_k_f32)(xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(-1, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q5_K);
}
xu_int XM_API(batch_moe_gemv_q5_k_f32)(xu_int batch, xu_int dst, xu_int x, xu_int w, xu_int expert_ids, xu_int expert_weights, xu_int n_experts) {
    return moe_gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, x, w, expert_ids, expert_weights, n_experts, MATRIX_Q5_K);
}

static xu_int paged_attention_decode_f16_impl(xu_int bh, xu_int dst, xu_int qh, xu_int kh, xu_int vh, xu_int bth,
                                                 const xu_i64_span *params) { @autoreleasepool {
    if (!params || !host_range(params->len, params->data, 0, params->len) || params->len < 5) return XM_BOUNDS;
    const xu_int *pv = params->data;
    xu_int kv_len = pv[0], block_size = pv[1], n_heads = pv[2], n_kv_heads = pv[3], scale_bits = pv[4];
    if (kv_len <= 0 || block_size <= 0 || n_heads <= 0 || n_kv_heads <= 0 || (n_heads % n_kv_heads) != 0) return XM_SHAPE;
    if (scale_bits < 0 || (xu_uint)scale_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t sb = (uint32_t)scale_bits;
    float scale = 0.0f;
    memcpy(&scale, &sb, 4);
    if (!isfinite(scale) || scale <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *q = lookup(qh, MATRIX_F32),
             *k = lookup(kh, MATRIX_F16), *v = lookup(vh, MATRIX_F16), *bt = lookup(bth, MATRIX_F32);
    if (!c || !q || !k || !v || !bt) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, c, &berr); if (!batch) RETURN(berr); }
    if (q.owner != c.owner || k.owner != c.owner || v.owner != c.owner || bt.owner != c.owner) RETURN(XM_DEVICE);
    if (q.rows == 0 || c.rows != q.rows || c.columns != q.columns || q.columns == 0 || (q.columns % (NSUInteger)n_heads) != 0) RETURN(XM_SHAPE);
    NSUInteger head_dim = q.columns / (NSUInteger)n_heads;
    NSUInteger kv_blocks_needed = ((NSUInteger)kv_len + (NSUInteger)block_size - 1) / (NSUInteger)block_size;
    if (k.columns != (NSUInteger)n_kv_heads * head_dim || v.columns != k.columns ||
        k.rows != v.rows || k.rows == 0 || (k.rows % (NSUInteger)block_size) != 0 ||
        bt.rows != q.rows || bt.columns < kv_blocks_needed) RETURN(XM_SHAPE);
    if (c == q || c == bt) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[q, k, v, bt];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || q.poisoned || k.poisoned || v.poisoned || bt.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, q) || !batch_f16_normal_input(batch, k) ||
        !batch_f16_normal_input(batch, v) || !batch_normal_input(batch, bt)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    NSUInteger tg = ai_row_tg_width(d.pagedAttnDecodeF16Pipeline);
    if (head_dim > 8 * tg) RETURN(XM_SHAPE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t kvl = (uint32_t)kv_len, bs_u = (uint32_t)block_size, nb = (uint32_t)(k.rows / (NSUInteger)block_size);
        uint32_t nh = (uint32_t)n_heads, kvm = (uint32_t)(n_heads / n_kv_heads), hd = (uint32_t)head_dim;
        uint32_t qs = (uint32_t)(q.rowBytes / 4), ks = (uint32_t)(k.rowBytes / 2), vs = (uint32_t)(v.rowBytes / 2);
        uint32_t bts = (uint32_t)(bt.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        [enc setComputePipelineState:d.pagedAttnDecodeF16Pipeline];
        [enc setBuffer:q.memory offset:0 atIndex:0];
        [enc setBuffer:k.memory offset:0 atIndex:1];
        [enc setBuffer:v.memory offset:0 atIndex:2];
        [enc setBuffer:bt.memory offset:0 atIndex:3];
        [enc setBuffer:c.memory offset:0 atIndex:4];
        [enc setBytes:&kvl length:sizeof(kvl) atIndex:5];
        [enc setBytes:&bs_u length:sizeof(bs_u) atIndex:6];
        [enc setBytes:&nb length:sizeof(nb) atIndex:7];
        [enc setBytes:&nh length:sizeof(nh) atIndex:8];
        [enc setBytes:&kvm length:sizeof(kvm) atIndex:9];
        [enc setBytes:&hd length:sizeof(hd) atIndex:10];
        [enc setBytes:&qs length:sizeof(qs) atIndex:11];
        [enc setBytes:&ks length:sizeof(ks) atIndex:12];
        [enc setBytes:&vs length:sizeof(vs) atIndex:13];
        [enc setBytes:&bts length:sizeof(bts) atIndex:14];
        [enc setBytes:&ds length:sizeof(ds) atIndex:15];
        [enc setBytes:&scale length:sizeof(scale) atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake(nh, q.rows, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.pagedAttnDecodeF16Pipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(paged_attention_decode_f16)(xu_int dst, xu_int q, xu_int k_pages, xu_int v_pages, xu_int block_table,
                                           const xu_i64_span *params) {
    return paged_attention_decode_f16_impl(-1, dst, q, k_pages, v_pages, block_table, params);
}
xu_int XM_API(batch_paged_attention_decode_f16)(xu_int batch, xu_int dst, xu_int q, xu_int k_pages, xu_int v_pages, xu_int block_table,
                                                 const xu_i64_span *params) {
    return paged_attention_decode_f16_impl(batch < 0 ? 0 : batch, dst, q, k_pages, v_pages, block_table, params);
}

static xu_int paged_kv_cache_append_f16_impl(xu_int bh, xu_int pagesh, xu_int srch, xu_int bth,
                                             const xu_i64_span *params) { @autoreleasepool {
    if (!params || !host_range(params->len, params->data, 0, params->len) || params->len < 2) return XM_BOUNDS;
    const xu_int *pv = params->data;
    xu_int pos_offset = pv[0], block_size = pv[1];
    if (pos_offset < 0 || block_size <= 0) return XM_SHAPE;
    ENTER;
    XMMatrix *pages = lookup(pagesh, MATRIX_F16), *src = lookup(srch, MATRIX_F32), *bt = lookup(bth, MATRIX_F32);
    if (!pages || !src || !bt) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, pages, &berr); if (!batch) RETURN(berr); }
    if (src.owner != pages.owner || bt.owner != pages.owner) RETURN(XM_DEVICE);
    if (bt.rows == 0 || pages.rows == 0 || (pages.rows % (NSUInteger)block_size) != 0 ||
        src.columns != pages.columns || (src.rows > 0 && (src.rows % bt.rows) != 0)) RETURN(XM_SHAPE);
    NSUInteger n_seqs = bt.rows, seq_len = src.rows / n_seqs;
    NSUInteger blocks_needed = (seq_len == 0) ? 0 : (((NSUInteger)pos_offset + seq_len + (NSUInteger)block_size - 1) / (NSUInteger)block_size);
    if (blocks_needed > bt.columns) RETURN(XM_SHAPE);
    NSArray<XMBuffer *> *inputs = (src == bt) ? @[src] : @[src, bt];
    if (!op_can_access(batch, pages, inputs)) RETURN(XM_BUSY);
    if (pages.poisoned || src.poisoned || bt.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, src) || !batch_normal_input(batch, bt)) RETURN(XM_NUMERICS);
    BOOL keep_normal = (!batch && pages.numberClass == MATRIX_CLASS_NORMAL);
    XMDevice *d = pages.owner;
    if (!src.words || !pages.columns) RETURN(finish_op(batch, d, pages, inputs, nil, nil));
    if (!ensure_quant_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t sl = (uint32_t)seq_len, cols = (uint32_t)pages.columns, ns = (uint32_t)n_seqs;
        uint32_t po = (uint32_t)pos_offset, bs_u = (uint32_t)block_size, nb = (uint32_t)(pages.rows / (NSUInteger)block_size);
        uint32_t ss = (uint32_t)(src.rowBytes / 4), bts = (uint32_t)(bt.rowBytes / 4), ds = (uint32_t)(pages.rowBytes / 2);
        [enc setComputePipelineState:d.pagedKvAppendF16Pipeline];
        [enc setBuffer:src.memory offset:0 atIndex:0];
        [enc setBuffer:bt.memory offset:0 atIndex:1];
        [enc setBuffer:pages.memory offset:0 atIndex:2];
        [enc setBytes:&sl length:sizeof(sl) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&ns length:sizeof(ns) atIndex:5];
        [enc setBytes:&po length:sizeof(po) atIndex:6];
        [enc setBytes:&bs_u length:sizeof(bs_u) atIndex:7];
        [enc setBytes:&nb length:sizeof(nb) atIndex:8];
        [enc setBytes:&ss length:sizeof(ss) atIndex:9];
        [enc setBytes:&bts length:sizeof(bts) atIndex:10];
        [enc setBytes:&ds length:sizeof(ds) atIndex:11];
        NSUInteger maxT = d.pagedKvAppendF16Pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger tx = MIN((NSUInteger)16, maxT);
        NSUInteger ty = MIN((NSUInteger)16, maxT / MAX((NSUInteger)1, tx));
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (sl + ty - 1) / ty, ns)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, pages, inputs, command, d.pagedKvAppendF16Pipeline);
    if (handle >= 0) pages.numberClass = keep_normal ? MATRIX_CLASS_NORMAL : MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(paged_kv_cache_append_f16)(xu_int pages, xu_int src, xu_int block_table, const xu_i64_span *params) {
    return paged_kv_cache_append_f16_impl(-1, pages, src, block_table, params);
}
xu_int XM_API(batch_paged_kv_cache_append_f16)(xu_int batch, xu_int pages, xu_int src, xu_int block_table, const xu_i64_span *params) {
    return paged_kv_cache_append_f16_impl(batch < 0 ? 0 : batch, pages, src, block_table, params);
}
