// Private implementation included after the common registry/fence machinery.
// Logical matrices are row-major binary32; MPS padding never crosses the ABI.
enum { MATRIX_MAX_DIM = 16384, MATRIX_MAX_CELLS = 16777216, MATRIX_CACHE_LIMIT = 8 };
enum { MATRIX_CLASS_UNKNOWN, MATRIX_CLASS_NORMAL, MATRIX_CLASS_UNSUPPORTED };

@interface XMMatrix : XMBuffer
@property(nonatomic) NSUInteger rows;
@property(nonatomic) NSUInteger columns;
@property(nonatomic) NSUInteger rowBytes;
@property(nonatomic, strong) MPSMatrix *matrix;
@property(nonatomic) int numberClass;
@end
@implementation XMMatrix
@end

xu_int XM_API(matrix_f32_new)(xu_int h, xu_int rows, xu_int columns) { @autoreleasepool {
    if (rows < 0 || columns < 0 || rows > MATRIX_MAX_DIM || columns > MATRIX_MAX_DIM) return XM_BOUNDS;
    uint64_t cells = (uint64_t)rows * (uint64_t)columns;
    if (cells > MATRIX_MAX_CELLS) return XM_BOUNDS;
    NSUInteger stride = [MPSMatrixDescriptor rowBytesFromColumns:MAX((NSUInteger)1, (NSUInteger)columns)
        dataType:MPSDataTypeFloat32];
    if (!stride || stride < (uint64_t)columns * 4 || (rows && stride > SIZE_MAX / (NSUInteger)rows)) return XM_BOUNDS;
    NSUInteger bytes = cells ? stride * (NSUInteger)rows : 16;
    // Stay well below 32-bit addressing boundaries as well as the device cap.
    if (bytes > UINT32_MAX / 4) return XM_BOUNDS;
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
                    rowBytes:stride dataType:MPSDataTypeFloat32];
                m.matrix = [[MPSMatrix alloc] initWithBuffer:memory descriptor:desc];
                if (!m.matrix) m = nil;
            }
        }
    } @catch (NSException *exception) { (void)exception; m = nil; }
    ENTER;
    xu_int result = m ? insert(m, MATRIX_F32) : XM_ALLOC;
    if (result < 0) d.children--;
    RETURN(result);
} }

xu_int XM_API(matrix_f32_info)(xu_int h, xu_int key) { @autoreleasepool {
    ENTER; XMMatrix *m = lookup(h, MATRIX_F32);
    if (!m) RETURN(XM_HANDLE);
    if (key == 1) RETURN((xu_int)m.rows);
    if (key == 2) RETURN((xu_int)m.columns);
    if (key == 3) RETURN((xu_int)m.rowBytes);
    if (key == 4) RETURN((xu_int)m.words);
    RETURN(XM_BOUNDS);
} }

xu_int XM_API(matrix_f32_write)(xu_int h, const xu_i64_span *source) { @autoreleasepool {
    if (!source || !host_range(source->len, source->data, 0, source->len)) return XM_BOUNDS;
    ENTER; XMMatrix *m = lookup(h, MATRIX_F32);
    if (!m) RETURN(XM_HANDLE);
    if ((uint64_t)source->len != m.words) RETURN(XM_BOUNDS);
    if (m.busy) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    const xu_int *input = source->data;
    int numberClass = MATRIX_CLASS_NORMAL;
    for (xu_int i = 0; i < source->len; ++i) {
        if (input[i] < 0 || (xu_uint)input[i] > UINT32_MAX) RETURN(XM_BOUNDS);
        uint32_t bits = (uint32_t)input[i], exponent = bits & UINT32_C(0x7f800000);
        if (exponent == UINT32_C(0x7f800000) || (!exponent && (bits & UINT32_C(0x007fffff))))
            numberClass = MATRIX_CLASS_UNSUPPORTED;
    }
    // Complete input validation precedes every destination write.
    // Shape/storage are immutable and gate protects lifetime: fetch Objective-C
    // properties once, leaving plain contiguous C loops for the host compiler.
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    uint8_t *base = m.memory.contents;
    if (m.words) for (NSUInteger row = 0; row < rows; ++row) {
        uint8_t *output = base + row * stride;
        for (NSUInteger col = 0; col < columns; ++col) {
            uint32_t bits = (uint32_t)input[row * columns + col];
            memcpy(output + col * 4, &bits, 4);
        }
    }
    m.numberClass = numberClass;
    RETURN(0);
} }

xu_int XM_API(matrix_f32_snapshot)(xu_int h) { @autoreleasepool {
    ENTER; XMMatrix *m = lookup(h, MATRIX_F32);
    if (!m) RETURN(XM_HANDLE);
    if (m.busy) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)m.words * 4];
    if (!data) RETURN(XM_ALLOC);
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    const uint8_t *source = m.memory.contents;
    uint8_t *destination = data.mutableBytes;
    if (m.words) for (NSUInteger row = 0; row < rows; ++row)
        memcpy(destination + row * columns * 4, source + row * stride, columns * 4);
    // No mutable reference escapes this scope; subsequent matrix writes cannot
    // alter the separately allocated, logically immutable snapshot.
    RETURN(snapshot(data));
} }

// Exact bit expansion, with no floating-point conversion or alignment
// assumption about NSData. Both readback paths use this representation rule.
static void copy_f32_bits(const uint8_t *input, xu_int *output, NSUInteger count) {
    for (NSUInteger i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, input + i * 4, 4);
        output[i] = (xu_int)bits;
    }
}

xu_int XM_API(matrix_f32_copy)(xu_int h, const xu_i64_span_mut *destination) { @autoreleasepool {
    if (!destination || !host_range(destination->len, destination->data, 0, destination->len)) return XM_BOUNDS;
    ENTER; XMMatrix *m = lookup(h, MATRIX_F32);
    if (!m) RETURN(XM_HANDLE);
    if ((uint64_t)destination->len < m.words) RETURN(XM_BOUNDS);
    if (m.busy) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    // Empty shapes may have nonzero row counts but only a 16-byte backing
    // allocation. Do not even form row pointers when there are no elements.
    if (!m.words) RETURN(0);
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    const uint8_t *input = m.memory.contents;
    xu_int *output = destination->data;
    for (NSUInteger row = 0; row < rows; ++row)
        copy_f32_bits(input + row * stride, output + row * columns, columns);
    RETURN(0);
} }

xu_int XM_API(snapshot_f32_copy)(xu_int h, const xu_i64_span_mut *destination) { @autoreleasepool {
    if (!destination || !host_range(destination->len, destination->data, 0, destination->len)) return XM_BOUNDS;
    ENTER; NSData *data = lookup(h, SNAPSHOT);
    if (!data) RETURN(XM_HANDLE);
    if ((data.length & 3) || (uint64_t)destination->len < data.length / 4) RETURN(XM_BOUNDS);
    NSUInteger count = data.length / 4;
    const uint8_t *input = data.bytes;
    xu_int *output = destination->data;
    copy_f32_bits(input, output, count);
    RETURN(0);
} }

xu_int XM_API(matrix_f32_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_F32, NO); } }
xu_int XM_API(matrix_f32_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_F32, YES); } }

// Storage accepts all bit patterns. This first compute contract deliberately
// admits only finite normals/zero, since MPS does not promise strict handling
// of subnormal/nonfinite operands or sequential reduction/FMA order.
static BOOL matrix_normal_inputs(XMMatrix *m) {
    // Every mutation is private and serialized by gate: upload classifies all
    // bits, zero-fill establishes normal/zero, and GPU output invalidates it.
    // An unchanged input therefore needs no repeated O(rows*columns) scan.
    if (m.numberClass != MATRIX_CLASS_UNKNOWN) return m.numberClass == MATRIX_CLASS_NORMAL;
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    const uint8_t *source = m.memory.contents;
    for (NSUInteger row = 0; row < rows; ++row) for (NSUInteger col = 0; col < columns; ++col) {
        uint32_t bits;
        memcpy(&bits, source + row * stride + col * 4, 4);
        uint32_t exponent = bits & UINT32_C(0x7f800000);
        if (exponent == UINT32_C(0x7f800000) || (!exponent && (bits & UINT32_C(0x007fffff)))) {
            m.numberClass = MATRIX_CLASS_UNSUPPORTED;
            return NO;
        }
    }
    m.numberClass = MATRIX_CLASS_NORMAL;
    return YES;
}

// Called with gate held. Per-device FIFO is bounded; pending fences retain
// their own operation after eviction. Keys cover every varying GEMM property.
static MPSMatrixMultiplication *matrix_kernel_ex(XMDevice *d, XMMatrix *c, XMMatrix *a, BOOL transposeRight) {
    NSString *key = [NSString stringWithFormat:@"%llu:%llu:%llu:%d",
        (unsigned long long)c.rows, (unsigned long long)c.columns,
        (unsigned long long)a.columns, transposeRight ? 1 : 0];
    MPSMatrixMultiplication *kernel = d.matrixKernels[key];
    if (kernel) return kernel;
    kernel = [[MPSMatrixMultiplication alloc] initWithDevice:d.gpu transposeLeft:NO transposeRight:transposeRight
        resultRows:c.rows resultColumns:c.columns interiorColumns:a.columns alpha:1 beta:0];
    if (!kernel) return nil;
    if (!d.matrixKernels) d.matrixKernels = [NSMutableDictionary new];
    if (!d.matrixKeys) d.matrixKeys = [NSMutableArray new];
    if (!d.matrixKernels || !d.matrixKeys) return nil;
    if (d.matrixKeys.count >= MATRIX_CACHE_LIMIT) {
        [d.matrixKernels removeObjectForKey:d.matrixKeys.firstObject];
        [d.matrixKeys removeObjectAtIndex:0];
    }
    d.matrixKernels[key] = kernel;
    [d.matrixKeys addObject:key];
    d.matrixKernelBuilds++;
    return kernel;
}

static BOOL ensure_ai_pipelines(XMDevice *d) {
    if (d.rmsnormPipeline && d.softmaxPipeline && d.swigluPipeline && d.addPipeline &&
        d.layernormPipeline && d.geluPipeline && d.mulPipeline && d.scalePipeline &&
        d.ropePipeline && d.causalAttnPipeline) return YES;
    NSString *src =
        @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void xu_rmsnorm_f32(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &xs [[buffer(4)]], constant uint &ds [[buffer(5)]], constant float &eps [[buffer(6)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh[256];\n"
        "    float acc = 0.0f;\n"
        "    uint xbase = row * xs;\n"
        "    for (uint c = tid; c < cols; c += tg) { float v = x[xbase + c]; acc += v * v; }\n"
        "    sh[tid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] += sh[tid + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    float inv = rsqrt(sh[0] / float(cols) + eps);\n"
        "    uint dbase = row * ds;\n"
        "    for (uint c = tid; c < cols; c += tg) { dst[dbase + c] = x[xbase + c] * inv * w[c]; }\n"
        "}\n"
        "kernel void xu_layernorm_f32(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
        "    device const float *b [[buffer(2)]], device float *dst [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &xs [[buffer(5)]], constant uint &ds [[buffer(6)]], constant float &eps [[buffer(7)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh[256];\n"
        "    uint xbase = row * xs;\n"
        "    float lsum = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { lsum += x[xbase + c]; }\n"
        "    sh[tid] = lsum;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] += sh[tid + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    float mean = sh[0] / float(cols);\n"
        "    float lvar = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { float d = x[xbase + c] - mean; lvar += d * d; }\n"
        "    sh[tid] = lvar;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] += sh[tid + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    float inv = rsqrt(sh[0] / float(cols) + eps);\n"
        "    uint dbase = row * ds;\n"
        "    for (uint c = tid; c < cols; c += tg) { dst[dbase + c] = ((x[xbase + c] - mean) * inv) * w[c] + b[c]; }\n"
        "}\n"
        "kernel void xu_softmax_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &cols [[buffer(2)]], constant uint &xs [[buffer(3)]], constant uint &ds [[buffer(4)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh[256];\n"
        "    uint xbase = row * xs;\n"
        "    float lmax = -INFINITY;\n"
        "    for (uint c = tid; c < cols; c += tg) { lmax = max(lmax, x[xbase + c]); }\n"
        "    sh[tid] = lmax;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]);\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    float rmax = sh[0];\n"
        "    float lsum = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { lsum += exp(x[xbase + c] - rmax); }\n"
        "    sh[tid] = lsum;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] += sh[tid + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    float inv_sum = 1.0f / sh[0];\n"
        "    uint dbase = row * ds;\n"
        "    for (uint c = tid; c < cols; c += tg) { dst[dbase + c] = exp(x[xbase + c] - rmax) * inv_sum; }\n"
        "}\n"
        "kernel void xu_swiglu_f32(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &gs [[buffer(5)]], constant uint &us [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        float gv = g[gid.y * gs + gid.x];\n"
        "        float silu = gv < -88.0f ? 0.0f : (gv / (1.0f + exp(-gv)));\n"
        "        dst[gid.y * ds + gid.x] = silu * u[gid.y * us + gid.x];\n"
        "    }\n"
        "}\n"
        "kernel void xu_add_f32(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &as [[buffer(5)]], constant uint &bs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        dst[gid.y * ds + gid.x] = a[gid.y * as + gid.x] + b[gid.y * bs + gid.x];\n"
        "    }\n"
        "}\n"
        "kernel void xu_mul_f32(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &as [[buffer(5)]], constant uint &bs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        dst[gid.y * ds + gid.x] = a[gid.y * as + gid.x] * b[gid.y * bs + gid.x];\n"
        "    }\n"
        "}\n"
        "kernel void xu_gelu_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &xs [[buffer(4)]], constant uint &ds [[buffer(5)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        float v = x[gid.y * xs + gid.x];\n"
        "        float t = tanh(0.7978845608f * (v + 0.044715f * v * v * v));\n"
        "        dst[gid.y * ds + gid.x] = 0.5f * v * (1.0f + t);\n"
        "    }\n"
        "}\n"
        "kernel void xu_scale_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &xs [[buffer(4)]], constant uint &ds [[buffer(5)]], constant float &alpha [[buffer(6)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        dst[gid.y * ds + gid.x] = x[gid.y * xs + gid.x] * alpha;\n"
        "    }\n"
        "}\n"
        "kernel void xu_rope_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &seq_len [[buffer(2)]], constant uint &n_heads [[buffer(3)]],\n"
        "    constant uint &head_dim [[buffer(4)]], constant uint &xs [[buffer(5)]],\n"
        "    constant uint &ds [[buffer(6)]], constant uint &pos_off [[buffer(7)]],\n"
        "    constant float &log_theta [[buffer(8)]], uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint half_d = head_dim >> 1;\n"
        "    uint j = gid.x, h = gid.y, r = gid.z;\n"
        "    if (j < half_d && h < n_heads && r < seq_len) {\n"
        "        float freq = exp(-log_theta * (float(2 * j) / float(head_dim)));\n"
        "        float angle = float(pos_off + r) * freq;\n"
        "        float c = cos(angle), s = sin(angle);\n"
        "        uint xbase = r * xs + h * head_dim;\n"
        "        uint dbase = r * ds + h * head_dim;\n"
        "        float v0 = x[xbase + j];\n"
        "        float v1 = x[xbase + half_d + j];\n"
        "        dst[dbase + j] = v0 * c - v1 * s;\n"
        "        dst[dbase + half_d + j] = v0 * s + v1 * c;\n"
        "    }\n"
        "}\n"
        "kernel void xu_causal_attn_f32(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],\n"
        "    device const float *v [[buffer(2)]], device float *dst [[buffer(3)]],\n"
        "    constant uint &seq_len [[buffer(4)]], constant uint &kv_len [[buffer(5)]],\n"
        "    constant uint &n_heads [[buffer(6)]], constant uint &kv_mul [[buffer(7)]],\n"
        "    constant uint &head_dim [[buffer(8)]], constant uint &qs [[buffer(9)]],\n"
        "    constant uint &ks [[buffer(10)]], constant uint &vs [[buffer(11)]],\n"
        "    constant uint &ds [[buffer(12)]], constant float &scale [[buffer(13)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh[256];\n"
        "    uint h = tgid.x, q_row = tgid.y;\n"
        "    uint tid = tid2.x, tg = tg2.x;\n"
        "    if (h >= n_heads || q_row >= seq_len) return;\n"
        "    uint kv_h = h / kv_mul;\n"
        "    uint max_t = (kv_len - seq_len) + q_row;\n"
        "    uint qbase = q_row * qs + h * head_dim;\n"
        "    uint dbase = q_row * ds + h * head_dim;\n"
        "    float rmax = -INFINITY;\n"
        "    float lsum = 0.0f;\n"
        "    for (uint t = 0; t <= max_t; ++t) {\n"
        "        uint kbase = t * ks + kv_h * head_dim;\n"
        "        float partial = 0.0f;\n"
        "        for (uint d = tid; d < head_dim; d += tg) { partial += q[qbase + d] * k[kbase + d]; }\n"
        "        sh[tid] = partial;\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "            if (tid < s) sh[tid] += sh[tid + s];\n"
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        }\n"
        "        float sc = sh[0] * scale;\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        if (t == 0) { rmax = sc; lsum = 1.0f; }\n"
        "        else { float nm = max(rmax, sc); lsum = lsum * exp(rmax - nm) + exp(sc - nm); rmax = nm; }\n"
        "    }\n"
        "    float inv_sum = 1.0f / lsum;\n"
        "    for (uint d = tid; d < head_dim; d += tg) { dst[dbase + d] = 0.0f; }\n"
        "    for (uint t = 0; t <= max_t; ++t) {\n"
        "        uint kbase = t * ks + kv_h * head_dim;\n"
        "        uint vbase = t * vs + kv_h * head_dim;\n"
        "        float partial = 0.0f;\n"
        "        for (uint d = tid; d < head_dim; d += tg) { partial += q[qbase + d] * k[kbase + d]; }\n"
        "        sh[tid] = partial;\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "            if (tid < s) sh[tid] += sh[tid + s];\n"
        "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        }\n"
        "        float p = exp(sh[0] * scale - rmax) * inv_sum;\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        for (uint d = tid; d < head_dim; d += tg) { dst[dbase + d] += p * v[vbase + d]; }\n"
        "    }\n"
        "}\n";
    NSError *error = nil;
    id<MTLLibrary> lib = [d.gpu newLibraryWithSource:src options:nil error:&error];
    if (!lib) return NO;
    id<MTLComputePipelineState> p0 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_rmsnorm_f32"] error:&error];
    id<MTLComputePipelineState> p1 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_softmax_f32"] error:&error];
    id<MTLComputePipelineState> p2 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_swiglu_f32"] error:&error];
    id<MTLComputePipelineState> p3 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_add_f32"] error:&error];
    id<MTLComputePipelineState> p4 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_layernorm_f32"] error:&error];
    id<MTLComputePipelineState> p5 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gelu_f32"] error:&error];
    id<MTLComputePipelineState> p6 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_mul_f32"] error:&error];
    id<MTLComputePipelineState> p7 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_scale_f32"] error:&error];
    id<MTLComputePipelineState> p8 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_rope_f32"] error:&error];
    id<MTLComputePipelineState> p9 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_causal_attn_f32"] error:&error];
    if (!p0 || !p1 || !p2 || !p3 || !p4 || !p5 || !p6 || !p7 || !p8 || !p9) return NO;
    d.rmsnormPipeline = p0;
    d.softmaxPipeline = p1;
    d.swigluPipeline = p2;
    d.addPipeline = p3;
    d.layernormPipeline = p4;
    d.geluPipeline = p5;
    d.mulPipeline = p6;
    d.scalePipeline = p7;
    d.ropePipeline = p8;
    d.causalAttnPipeline = p9;
    return YES;
}

static NSUInteger ai_row_tg_width(id<MTLComputePipelineState> pipe) {
    NSUInteger w = 256;
    while (w > 1 && w > pipe.maxTotalThreadsPerThreadgroup) w >>= 1;
    return w;
}

static xu_int gemm_f32_impl(xu_int dst, xu_int ah, xu_int bh, BOOL transposeRight) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *a = lookup(ah, MATRIX_F32), *b = lookup(bh, MATRIX_F32);
    if (!c || !a || !b) RETURN(XM_HANDLE);
    if (a.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    NSUInteger innerB = transposeRight ? b.columns : b.rows;
    NSUInteger outColsB = transposeRight ? b.rows : b.columns;
    if (a.columns != innerB || c.rows != a.rows || c.columns != outColsB) RETURN(XM_SHAPE);
    if (c == a || c == b) RETURN(XM_ALIAS);
    if (c.busy || a.busy || b.busy) RETURN(XM_BUSY);
    if (c.poisoned || a.poisoned || b.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(a) || !matrix_normal_inputs(b)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!MPSSupportsMTLDevice(d.gpu)) RETURN(XM_UNAVAILABLE);
    if (!c.words || !a.columns) {
        xu_int fence = submit_command(d, c, a, b, nil, nil);
        // Failure to allocate the completed fence must leave output unchanged.
        if (fence >= 0) {
            if (c.words) memset(c.memory.contents, 0, c.memory.length);
            c.numberClass = MATRIX_CLASS_NORMAL;
        }
        RETURN(fence);
    }
    id<MTLCommandBuffer> command = nil;
    MPSMatrixMultiplication *kernel = nil;
    @try {
        kernel = matrix_kernel_ex(d, c, a, transposeRight);
        if (!kernel) RETURN(XM_PIPELINE);
        command = [d.queue commandBuffer];
        if (!command) RETURN(XM_COMMAND);
        [kernel encodeToCommandBuffer:command leftMatrix:a.matrix rightMatrix:b.matrix resultMatrix:c.matrix];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, a, b, command, kernel);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(gemm_f32)(xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(dst, ah, bh, NO); }
xu_int XM_API(gemm_f32_nt)(xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(dst, ah, bh, YES); }

xu_int XM_API(rmsnorm_f32)(xu_int dst, xu_int xh, xu_int wh, xu_int eps_bits) { @autoreleasepool {
    if (eps_bits < 0 || (xu_uint)eps_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t eb = (uint32_t)eps_bits;
    float eps = 0.0f;
    memcpy(&eps, &eb, 4);
    if (!isfinite(eps) || eps <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32), *w = lookup(wh, MATRIX_F32);
    if (!c || !x || !w) RETURN(XM_HANDLE);
    if (x.owner != c.owner || w.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns || w.rows != 1 || w.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x || c == w) RETURN(XM_ALIAS);
    if (c.busy || x.busy || w.busy) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || w.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(x) || !matrix_normal_inputs(w)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command(d, c, x, w, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t cols = (uint32_t)x.columns, xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.rmsnormPipeline);
        [enc setComputePipelineState:d.rmsnormPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:w.memory offset:0 atIndex:1];
        [enc setBuffer:c.memory offset:0 atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&xs length:sizeof(xs) atIndex:4];
        [enc setBytes:&ds length:sizeof(ds) atIndex:5];
        [enc setBytes:&eps length:sizeof(eps) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(x.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, x, w, command, d.rmsnormPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(softmax_f32)(xu_int dst, xu_int xh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    if (c.busy || x.busy) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command(d, c, x, x, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t cols = (uint32_t)x.columns, xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.softmaxPipeline);
        [enc setComputePipelineState:d.softmaxPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc setBytes:&xs length:sizeof(xs) atIndex:3];
        [enc setBytes:&ds length:sizeof(ds) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(x.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, x, x, command, d.softmaxPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

static xu_int binary_elem_f32(xu_int dst, xu_int ah, xu_int bh, int mode) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *a = lookup(ah, MATRIX_F32), *b = lookup(bh, MATRIX_F32);
    if (!c || !a || !b) RETURN(XM_HANDLE);
    if (a.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != a.rows || c.columns != a.columns || b.rows != a.rows || b.columns != a.columns) RETURN(XM_SHAPE);
    if (c == a || c == b) RETURN(XM_ALIAS);
    if (c.busy || a.busy || b.busy) RETURN(XM_BUSY);
    if (c.poisoned || a.poisoned || b.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(a) || !matrix_normal_inputs(b)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command(d, c, a, b, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLComputePipelineState> pipe = mode == 1 ? d.swigluPipeline : (mode == 2 ? d.mulPipeline : d.addPipeline);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns;
        uint32_t as = (uint32_t)(a.rowBytes / 4), bs = (uint32_t)(b.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, pipe.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, pipe.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:pipe];
        [enc setBuffer:a.memory offset:0 atIndex:0];
        [enc setBuffer:b.memory offset:0 atIndex:1];
        [enc setBuffer:c.memory offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&as length:sizeof(as) atIndex:5];
        [enc setBytes:&bs length:sizeof(bs) atIndex:6];
        [enc setBytes:&ds length:sizeof(ds) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, a, b, command, pipe);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(swiglu_f32)(xu_int dst, xu_int gate, xu_int up) { return binary_elem_f32(dst, gate, up, 1); }
xu_int XM_API(add_f32)(xu_int dst, xu_int a, xu_int b) { return binary_elem_f32(dst, a, b, 0); }
xu_int XM_API(mul_f32)(xu_int dst, xu_int a, xu_int b) { return binary_elem_f32(dst, a, b, 2); }

xu_int XM_API(layernorm_f32)(xu_int dst, xu_int xh, xu_int wh, xu_int bh, xu_int eps_bits) { @autoreleasepool {
    if (eps_bits < 0 || (xu_uint)eps_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t eb = (uint32_t)eps_bits;
    float eps = 0.0f;
    memcpy(&eps, &eb, 4);
    if (!isfinite(eps) || eps <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32),
             *w = lookup(wh, MATRIX_F32), *b = lookup(bh, MATRIX_F32);
    if (!c || !x || !w || !b) RETURN(XM_HANDLE);
    if (x.owner != c.owner || w.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns ||
        w.rows != 1 || w.columns != x.columns ||
        b.rows != 1 || b.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x || c == w || c == b) RETURN(XM_ALIAS);
    if (c.busy || x.busy || w.busy || b.busy) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || w.poisoned || b.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(x) || !matrix_normal_inputs(w) || !matrix_normal_inputs(b)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command_many(d, c, @[c, x, w, b], nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t cols = (uint32_t)x.columns, xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.layernormPipeline);
        [enc setComputePipelineState:d.layernormPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:w.memory offset:0 atIndex:1];
        [enc setBuffer:b.memory offset:0 atIndex:2];
        [enc setBuffer:c.memory offset:0 atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&xs length:sizeof(xs) atIndex:5];
        [enc setBytes:&ds length:sizeof(ds) atIndex:6];
        [enc setBytes:&eps length:sizeof(eps) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(x.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command_many(d, c, @[c, x, w, b], command, d.layernormPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(gelu_f32)(xu_int dst, xu_int xh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    if (c.busy || x.busy) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command(d, c, x, x, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, d.geluPipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, d.geluPipeline.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:d.geluPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&rows length:sizeof(rows) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&xs length:sizeof(xs) atIndex:4];
        [enc setBytes:&ds length:sizeof(ds) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, x, x, command, d.geluPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(scale_f32)(xu_int dst, xu_int xh, xu_int alpha_bits) { @autoreleasepool {
    if (alpha_bits < 0 || (xu_uint)alpha_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t ab = (uint32_t)alpha_bits;
    float alpha = 0.0f;
    memcpy(&alpha, &ab, 4);
    if (!isfinite(alpha)) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    if (c.busy || x.busy) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command(d, c, x, x, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, d.scalePipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, d.scalePipeline.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:d.scalePipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&rows length:sizeof(rows) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&xs length:sizeof(xs) atIndex:4];
        [enc setBytes:&ds length:sizeof(ds) atIndex:5];
        [enc setBytes:&alpha length:sizeof(alpha) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, x, x, command, d.scalePipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(rope_f32)(xu_int dst, xu_int xh, xu_int pos_offset, xu_int head_dim, xu_int theta_bits) { @autoreleasepool {
    if (pos_offset < 0 || pos_offset > UINT32_MAX || head_dim <= 0 || (head_dim & 1) || head_dim > MATRIX_MAX_DIM) return XM_BOUNDS;
    if (theta_bits < 0 || (xu_uint)theta_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t tb = (uint32_t)theta_bits;
    float theta = 0.0f;
    memcpy(&theta, &tb, 4);
    if (!isfinite(theta) || theta <= 0.0f) return XM_NUMERICS;
    float log_theta = logf(theta);
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns || (x.columns % (NSUInteger)head_dim) != 0) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    if (c.busy || x.busy) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command(d, c, x, x, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t seq_len = (uint32_t)c.rows, hd = (uint32_t)head_dim;
        uint32_t nh = (uint32_t)(c.columns / hd);
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        uint32_t poff = (uint32_t)pos_offset;
        uint32_t half_d = hd >> 1;
        NSUInteger tx = MIN((NSUInteger)MAX(1u, MIN(half_d, 64u)), d.ropePipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:d.ropePipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:2];
        [enc setBytes:&nh length:sizeof(nh) atIndex:3];
        [enc setBytes:&hd length:sizeof(hd) atIndex:4];
        [enc setBytes:&xs length:sizeof(xs) atIndex:5];
        [enc setBytes:&ds length:sizeof(ds) atIndex:6];
        [enc setBytes:&poff length:sizeof(poff) atIndex:7];
        [enc setBytes:&log_theta length:sizeof(log_theta) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake((half_d + tx - 1) / tx, nh, seq_len)
            threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, x, x, command, d.ropePipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(causal_attention_f32)(xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                    xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) { @autoreleasepool {
    if (n_heads <= 0 || n_kv_heads <= 0 || (n_heads % n_kv_heads) != 0) return XM_SHAPE;
    if (scale_bits < 0 || (xu_uint)scale_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t sb = (uint32_t)scale_bits;
    float scale = 0.0f;
    memcpy(&scale, &sb, 4);
    if (!isfinite(scale) || scale <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *q = lookup(qh, MATRIX_F32),
             *k = lookup(kh, MATRIX_F32), *v = lookup(vh, MATRIX_F32);
    if (!c || !q || !k || !v) RETURN(XM_HANDLE);
    if (q.owner != c.owner || k.owner != c.owner || v.owner != c.owner) RETURN(XM_DEVICE);
    if (q.columns == 0 || (q.columns % (NSUInteger)n_heads) != 0) RETURN(XM_SHAPE);
    NSUInteger head_dim = q.columns / (NSUInteger)n_heads;
    if (c.rows != q.rows || c.columns != q.columns ||
        k.columns != (NSUInteger)n_kv_heads * head_dim ||
        v.rows != k.rows || v.columns != k.columns || k.rows < q.rows) RETURN(XM_SHAPE);
    if (c == q || c == k || c == v) RETURN(XM_ALIAS);
    if (c.busy || q.busy || k.busy || v.busy) RETURN(XM_BUSY);
    if (c.poisoned || q.poisoned || k.poisoned || v.poisoned) RETURN(XM_COMMAND);
    if (!matrix_normal_inputs(q) || !matrix_normal_inputs(k) || !matrix_normal_inputs(v)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(submit_command_many(d, c, @[c, q, k, v], nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t seq_len = (uint32_t)q.rows, kv_len = (uint32_t)k.rows;
        uint32_t nh = (uint32_t)n_heads, kvm = (uint32_t)(n_heads / n_kv_heads), hd = (uint32_t)head_dim;
        uint32_t qs = (uint32_t)(q.rowBytes / 4), ks = (uint32_t)(k.rowBytes / 4);
        uint32_t vs = (uint32_t)(v.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.causalAttnPipeline);
        [enc setComputePipelineState:d.causalAttnPipeline];
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
    xu_int handle = submit_command_many(d, c, @[c, q, k, v], command, d.causalAttnPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }
