// Private implementation included after the common registry/fence machinery.
// Logical matrices are row-major binary32; MPS padding never crosses the ABI.
enum { MATRIX_MAX_DIM = 262144, MATRIX_MAX_CELLS = 536870912, MATRIX_CACHE_LIMIT = 8 };
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
    if (!can_host_write(m)) RETURN(XM_BUSY);
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

xu_int XM_API(matrix_f32_write_bytes)(xu_int h, const xu_u8_span *bytes) { @autoreleasepool {
    if (!bytes || bytes->len < 0 || (bytes->len > 0 && !bytes->data)) return XM_BOUNDS;
    ENTER; XMMatrix *m = lookup(h, MATRIX_F32);
    if (!m) RETURN(XM_HANDLE);
    if ((uint64_t)bytes->len != m.words * 4ULL) RETURN(XM_BOUNDS);
    if (!can_host_write(m)) RETURN(XM_BUSY);
    if (m.poisoned) RETURN(XM_COMMAND);
    const uint8_t *src = bytes->data;
    int numberClass = MATRIX_CLASS_NORMAL;
    for (uint64_t i = 0; i < m.words; ++i) {
        uint32_t bits;
        memcpy(&bits, src + i * 4ULL, 4);
        uint32_t exponent = bits & UINT32_C(0x7f800000);
        if (exponent == UINT32_C(0x7f800000) || (!exponent && (bits & UINT32_C(0x007fffff)))) {
            numberClass = MATRIX_CLASS_UNSUPPORTED;
            break;
        }
    }
    NSUInteger rows = m.rows, columns = m.columns, stride = m.rowBytes;
    uint8_t *base = m.memory.contents;
    if (m.words && columns) {
        size_t rowCopy = (size_t)columns * 4;
        for (NSUInteger row = 0; row < rows; ++row) {
            memcpy(base + row * stride, src + (size_t)row * rowCopy, rowCopy);
        }
    }
    m.numberClass = numberClass;
    RETURN(0);
} }

xu_int XM_API(matrix_f32_snapshot)(xu_int h) { @autoreleasepool {
    ENTER; XMMatrix *m = lookup(h, MATRIX_F32);
    if (!m) RETURN(XM_HANDLE);
    if (!can_host_read(m)) RETURN(XM_BUSY);
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
    if (!can_host_read(m)) RETURN(XM_BUSY);
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
static MPSMatrixMultiplication *matrix_kernel_typed(XMDevice *d, XMMatrix *c, XMMatrix *a,
                                                      BOOL transposeLeft, BOOL transposeRight, MPSDataType dataType) {
    const char *tag = (dataType == MPSDataTypeFloat16) ? "f16" : "f32";
    NSUInteger interior = transposeLeft ? a.rows : a.columns;
    NSString *key = [NSString stringWithFormat:@"%s:%llu:%llu:%llu:%d:%d",
        tag, (unsigned long long)c.rows, (unsigned long long)c.columns,
        (unsigned long long)interior, transposeLeft ? 1 : 0, transposeRight ? 1 : 0];
    MPSMatrixMultiplication *kernel = d.matrixKernels[key];
    if (kernel) return kernel;
    kernel = [[MPSMatrixMultiplication alloc] initWithDevice:d.gpu transposeLeft:transposeLeft transposeRight:transposeRight
        resultRows:c.rows resultColumns:c.columns interiorColumns:interior alpha:1 beta:0];
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

static MPSMatrixMultiplication *matrix_kernel_ex(XMDevice *d, XMMatrix *c, XMMatrix *a, BOOL transposeLeft, BOOL transposeRight) {
    return matrix_kernel_typed(d, c, a, transposeLeft, transposeRight, MPSDataTypeFloat32);
}

static inline BOOL batch_normal_input(XMBatch *batch, XMMatrix *m) {
    if (batch && [batch.writeBuffers indexOfObjectIdenticalTo:m] != NSNotFound) return YES;
    return matrix_normal_inputs(m);
}

static xu_int finish_op(XMBatch *batch, XMDevice *d, XMBuffer *dst, NSArray<XMBuffer *> *inputs,
                        id<MTLCommandBuffer> command, id operation) {
    if (batch) {
        if (command && !batch.command) batch.command = command;
        batch_acquire(batch, dst, inputs, operation);
        return 0;
    }
    return submit_command_rw(d, dst ? @[dst] : @[], inputs ?: @[], command, operation ? @[operation] : nil, NO);
}

static xu_int finish_op2(XMBatch *batch, XMDevice *d, NSArray<XMBuffer *> *writes, NSArray<XMBuffer *> *inputs,
                         id<MTLCommandBuffer> command, id operation) {
    if (batch) {
        if (command && !batch.command) batch.command = command;
        batch_acquire_many(batch, writes, inputs, operation);
        return 0;
    }
    return submit_command_rw(d, writes ?: @[], inputs ?: @[], command, operation ? @[operation] : nil, NO);
}

static BOOL ensure_ai_pipelines(XMDevice *d) {
    if (d.rmsnormPipeline && d.softmaxPipeline && d.swigluPipeline && d.addPipeline &&
        d.layernormPipeline && d.geluPipeline && d.mulPipeline && d.scalePipeline &&
        d.ropePipeline && d.ropeInterleavedPipeline && d.causalAttnPipeline && d.qkNormPipeline &&
        d.kvAppendPipeline && d.attnDecodePipeline && d.embeddingPipeline &&
        d.argmaxPipeline && d.addRmsnormPipeline && d.gegluPipeline && d.siluPipeline &&
        d.logSoftmaxPipeline && d.moeTopkPipeline && d.moeWeightedAddPipeline &&
        d.ropeYarnPipeline && d.loraAddPipeline && d.adamwStepPipeline &&
        d.crossEntropyGradPipeline && d.rmsnormBwdDxPipeline &&
        d.rmsnormBwdDwPipeline && d.swigluBwdPipeline) return YES;
    NSString *src =
        @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "inline float tg_sum(float v, threadgroup float *sh, uint simd_lane, uint simd_gid, uint simd_cnt) {\n"
        "    v = simd_sum(v);\n"
        "    if (simd_lane == 0) sh[simd_gid] = v;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float t = (simd_lane < simd_cnt) ? sh[simd_lane] : 0.0f;\n"
        "        t = simd_sum(t);\n"
        "        if (simd_lane == 0) sh[0] = t;\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    return sh[0];\n"
        "}\n"
        "inline float tg_max(float v, threadgroup float *sh, uint simd_lane, uint simd_gid, uint simd_cnt) {\n"
        "    v = simd_max(v);\n"
        "    if (simd_lane == 0) sh[simd_gid] = v;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    if (simd_gid == 0) {\n"
        "        float t = (simd_lane < simd_cnt) ? sh[simd_lane] : -INFINITY;\n"
        "        t = simd_max(t);\n"
        "        if (simd_lane == 0) sh[0] = t;\n"
        "    }\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    return sh[0];\n"
        "}\n"
        "kernel void xu_rmsnorm_f32(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &xs [[buffer(4)]], constant uint &ds [[buffer(5)]], constant float &eps [[buffer(6)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    float acc = 0.0f;\n"
        "    uint xbase = row * xs;\n"
        "    for (uint c = tid; c < cols; c += tg) { float v = x[xbase + c]; acc += v * v; }\n"
        "    float tot = tg_sum(acc, sh, sl, sg, sc);\n"
        "    float inv = rsqrt(tot / float(cols) + eps);\n"
        "    uint dbase = row * ds;\n"
        "    for (uint c = tid; c < cols; c += tg) { dst[dbase + c] = x[xbase + c] * inv * w[c]; }\n"
        "}\n"
        "kernel void xu_add_rmsnorm_f32(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],\n"
        "    device const float *w [[buffer(2)]], device float *dst_norm [[buffer(3)]], device float *dst_sum [[buffer(4)]],\n"
        "    constant uint &cols [[buffer(5)]], constant uint &as [[buffer(6)]], constant uint &bs [[buffer(7)]],\n"
        "    constant uint &dns [[buffer(8)]], constant uint &dss [[buffer(9)]], constant float &eps [[buffer(10)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    float acc = 0.0f;\n"
        "    uint ab = row * as, bb = row * bs, sb = row * dss, nb = row * dns;\n"
        "    for (uint c = tid; c < cols; c += tg) {\n"
        "        float v = a[ab + c] + b[bb + c];\n"
        "        dst_sum[sb + c] = v;\n"
        "        acc += v * v;\n"
        "    }\n"
        "    float tot = tg_sum(acc, sh, sl, sg, sc);\n"
        "    float inv = rsqrt(tot / float(cols) + eps);\n"
        "    for (uint c = tid; c < cols; c += tg) { dst_norm[nb + c] = dst_sum[sb + c] * inv * w[c]; }\n"
        "}\n"
        "kernel void xu_layernorm_f32(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
        "    device const float *b [[buffer(2)]], device float *dst [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &xs [[buffer(5)]], constant uint &ds [[buffer(6)]], constant float &eps [[buffer(7)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint xbase = row * xs;\n"
        "    float lsum = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { lsum += x[xbase + c]; }\n"
        "    float mean = tg_sum(lsum, sh, sl, sg, sc) / float(cols);\n"
        "    float lvar = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { float d = x[xbase + c] - mean; lvar += d * d; }\n"
        "    float inv = rsqrt(tg_sum(lvar, sh, sl, sg, sc) / float(cols) + eps);\n"
        "    uint dbase = row * ds;\n"
        "    for (uint c = tid; c < cols; c += tg) { dst[dbase + c] = ((x[xbase + c] - mean) * inv) * w[c] + b[c]; }\n"
        "}\n"
        "kernel void xu_softmax_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &cols [[buffer(2)]], constant uint &xs [[buffer(3)]], constant uint &ds [[buffer(4)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint xbase = row * xs;\n"
        "    float lmax = -INFINITY;\n"
        "    for (uint c = tid; c < cols; c += tg) { lmax = max(lmax, x[xbase + c]); }\n"
        "    float rmax = tg_max(lmax, sh, sl, sg, sc);\n"
        "    float lsum = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { lsum += exp(x[xbase + c] - rmax); }\n"
        "    float inv_sum = 1.0f / tg_sum(lsum, sh, sl, sg, sc);\n"
        "    uint dbase = row * ds;\n"
        "    for (uint c = tid; c < cols; c += tg) { dst[dbase + c] = exp(x[xbase + c] - rmax) * inv_sum; }\n"
        "}\n"
        "kernel void xu_log_softmax_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &cols [[buffer(2)]], constant uint &xs [[buffer(3)]], constant uint &ds [[buffer(4)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint xbase = row * xs;\n"
        "    float lmax = -INFINITY;\n"
        "    for (uint c = tid; c < cols; c += tg) { lmax = max(lmax, x[xbase + c]); }\n"
        "    float rmax = tg_max(lmax, sh, sl, sg, sc);\n"
        "    float lsum = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { lsum += exp(x[xbase + c] - rmax); }\n"
        "    float log_z = rmax + log(tg_sum(lsum, sh, sl, sg, sc));\n"
        "    uint dbase = row * ds;\n"
        "    for (uint c = tid; c < cols; c += tg) { dst[dbase + c] = x[xbase + c] - log_z; }\n"
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
        "kernel void xu_geglu_f32(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &gs [[buffer(5)]], constant uint &us [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        float gv = g[gid.y * gs + gid.x];\n"
        "        float t = tanh(0.7978845608f * (gv + 0.044715f * gv * gv * gv));\n"
        "        dst[gid.y * ds + gid.x] = (0.5f * gv * (1.0f + t)) * u[gid.y * us + gid.x];\n"
        "    }\n"
        "}\n"
        "kernel void xu_silu_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &xs [[buffer(4)]], constant uint &ds [[buffer(5)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        float v = x[gid.y * xs + gid.x];\n"
        "        dst[gid.y * ds + gid.x] = v < -88.0f ? 0.0f : (v / (1.0f + exp(-v)));\n"
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
        "kernel void xu_rope_interleaved_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
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
        "        float v0 = x[xbase + 2 * j];\n"
        "        float v1 = x[xbase + 2 * j + 1];\n"
        "        dst[dbase + 2 * j] = v0 * c - v1 * s;\n"
        "        dst[dbase + 2 * j + 1] = v0 * s + v1 * c;\n"
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
        "        for (uint d = tid; d < head_dim; d += tg) { partial += q[qbase + d] * k[kbase + d]; }\n"
        "        float sc = tg_sum(partial, sh, sl, sg, sct) * scale;\n"
        "        float alpha = 0.0f, p = 1.0f;\n"
        "        if (t == 0) { rmax = sc; lsum = 1.0f; }\n"
        "        else { float nm = max(rmax, sc); alpha = exp(rmax - nm); p = exp(sc - nm); lsum = lsum * alpha + p; rmax = nm; }\n"
        "        for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) {\n"
        "            o_reg[i] = (t == 0 ? 0.0f : o_reg[i] * alpha) + p * v[vbase + d];\n"
        "        }\n"
        "    }\n"
        "    float inv_sum = 1.0f / lsum;\n"
        "    for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) { dst[dbase + d] = o_reg[i] * inv_sum; }\n"
        "}\n"
        "kernel void xu_qk_norm_f32(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &n_heads [[buffer(3)]],\n"
        "    constant uint &head_dim [[buffer(4)]], constant uint &xs [[buffer(5)]],\n"
        "    constant uint &ds [[buffer(6)]], constant float &eps [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sct [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint h = tgid.x, row = tgid.y, tid = tid2.x, tg = tg2.x;\n"
        "    if (h >= n_heads) return;\n"
        "    uint xbase = row * xs + h * head_dim;\n"
        "    uint dbase = row * ds + h * head_dim;\n"
        "    float acc = 0.0f;\n"
        "    for (uint c = tid; c < head_dim; c += tg) { float v = x[xbase + c]; acc += v * v; }\n"
        "    float inv = rsqrt(tg_sum(acc, sh, sl, sg, sct) / float(head_dim) + eps);\n"
        "    for (uint c = tid; c < head_dim; c += tg) { dst[dbase + c] = x[xbase + c] * inv * w[c]; }\n"
        "}\n"
        "kernel void xu_kv_cache_append_f32(device const float *src [[buffer(0)]], device float *cache [[buffer(1)]],\n"
        "    constant uint &rows [[buffer(2)]], constant uint &cols [[buffer(3)]],\n"
        "    constant uint &ss [[buffer(4)]], constant uint &ds [[buffer(5)]],\n"
        "    constant uint &pos_off [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        cache[(pos_off + gid.y) * ds + gid.x] = src[gid.y * ss + gid.x];\n"
        "    }\n"
        "}\n"
        "kernel void xu_attention_decode_f32(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],\n"
        "    device const float *v [[buffer(2)]], device float *dst [[buffer(3)]],\n"
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
        "        for (uint d = tid; d < head_dim; d += tg) { partial += q[qbase + d] * k[kbase + d]; }\n"
        "        float sc = tg_sum(partial, sh, sl, sg, sct) * scale;\n"
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
        "            o_reg[i] = (t == 0 ? 0.0f : o_reg[i] * alpha) + p * v[vbase + d];\n"
        "        }\n"
        "    }\n"
        "    float inv_sum = 1.0f / lsum;\n"
        "    for (uint d = tid, i = 0; d < head_dim && i < 8; d += tg, ++i) {\n"
        "        dst[dbase + d] = o_reg[i] * inv_sum;\n"
        "    }\n"
        "}\n"
        "kernel void xu_moe_topk_f32(device const float *logits [[buffer(0)]], device float *out_idx [[buffer(1)]],\n"
        "    device float *out_w [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &n_exp [[buffer(4)]],\n"
        "    constant uint &k [[buffer(5)]], constant uint &use_sigmoid [[buffer(6)]],\n"
        "    constant uint &norm_topk [[buffer(7)]], constant float &wscale [[buffer(8)]],\n"
        "    constant uint &ls [[buffer(9)]], constant uint &is [[buffer(10)]], constant uint &ws [[buffer(11)]],\n"
        "    uint row [[thread_position_in_grid]]) {\n"
        "    if (row >= rows) return;\n"
        "    uint lbase = row * ls, ibase = row * is, wbase = row * ws;\n"
        "    float rmax = -INFINITY;\n"
        "    for (uint e = 0; e < n_exp; ++e) { rmax = max(rmax, logits[lbase + e]); }\n"
        "    float esum = 0.0f;\n"
        "    if (!use_sigmoid) {\n"
        "        for (uint e = 0; e < n_exp; ++e) { esum += exp(logits[lbase + e] - rmax); }\n"
        "    }\n"
        "    float wsum = 0.0f;\n"
        "    for (uint j = 0; j < k; ++j) {\n"
        "        uint best_e = 0;\n"
        "        float best_s = -INFINITY;\n"
        "        for (uint e = 0; e < n_exp; ++e) {\n"
        "            bool seen = false;\n"
        "            for (uint p = 0; p < j; ++p) { if (uint(out_idx[ibase + p]) == e) { seen = true; break; } }\n"
        "            if (seen) continue;\n"
        "            float lv = logits[lbase + e];\n"
        "            float sc = use_sigmoid ? (1.0f / (1.0f + exp(-lv))) : (exp(lv - rmax) / esum);\n"
        "            if (sc > best_s) { best_s = sc; best_e = e; }\n"
        "        }\n"
        "        out_idx[ibase + j] = float(best_e);\n"
        "        out_w[wbase + j] = best_s;\n"
        "        wsum += best_s;\n"
        "    }\n"
        "    if (norm_topk && wsum > 0.0f) {\n"
        "        for (uint j = 0; j < k; ++j) { out_w[wbase + j] /= wsum; }\n"
        "    }\n"
        "    if (wscale != 1.0f) {\n"
        "        for (uint j = 0; j < k; ++j) { out_w[wbase + j] *= wscale; }\n"
        "    }\n"
        "}\n"
        "kernel void xu_moe_weighted_add_f32(device const float *eouts [[buffer(0)]], device const float *weights [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &k [[buffer(5)]], constant uint &es [[buffer(6)]], constant uint &ws [[buffer(7)]],\n"
        "    constant uint &ds [[buffer(8)]], uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        float acc = 0.0f;\n"
        "        uint wbase = gid.y * ws;\n"
        "        uint ebase = gid.y * k;\n"
        "        for (uint j = 0; j < k; ++j) {\n"
        "            acc += weights[wbase + j] * eouts[(ebase + j) * es + gid.x];\n"
        "        }\n"
        "        dst[gid.y * ds + gid.x] = acc;\n"
        "    }\n"
        "}\n"
        "kernel void xu_embedding_f32(device const float *table [[buffer(0)]], device const uint *ids [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &ts [[buffer(5)]], constant uint &ds [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        uint row_idx = ids[gid.y];\n"
        "        dst[gid.y * ds + gid.x] = table[row_idx * ts + gid.x];\n"
        "    }\n"
        "}\n"
        "kernel void xu_argmax_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &cols [[buffer(2)]], constant uint &xs [[buffer(3)]], constant uint &ds [[buffer(4)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh_v[256];\n"
        "    threadgroup uint sh_i[256];\n"
        "    uint xbase = row * xs;\n"
        "    float best_v = -INFINITY;\n"
        "    uint best_i = 0xFFFFFFFFu;\n"
        "    for (uint c = tid; c < cols; c += tg) {\n"
        "        float v = x[xbase + c];\n"
        "        if (v > best_v || (v == best_v && c < best_i)) { best_v = v; best_i = c; }\n"
        "    }\n"
        "    sh_v[tid] = best_v;\n"
        "    sh_i[tid] = best_i;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) {\n"
        "            float ov = sh_v[tid + s]; uint oi = sh_i[tid + s];\n"
        "            if (ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid])) { sh_v[tid] = ov; sh_i[tid] = oi; }\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    if (tid == 0) dst[row * ds] = float(sh_i[0]);\n"
        "}\n"
        "kernel void xu_rope_yarn_f32(device const float *x [[buffer(0)]], device float *dst [[buffer(1)]],\n"
        "    constant uint &seq_len [[buffer(2)]], constant uint &n_heads [[buffer(3)]],\n"
        "    constant uint &head_dim [[buffer(4)]], constant uint &rope_dim [[buffer(5)]],\n"
        "    constant uint &interleaved [[buffer(6)]], constant uint &xs [[buffer(7)]],\n"
        "    constant uint &ds [[buffer(8)]], constant uint &pos_off [[buffer(9)]],\n"
        "    constant float &log_theta [[buffer(10)]], constant float &eff_scale [[buffer(11)]],\n"
        "    constant uint &use_ntk [[buffer(12)]], constant float &low [[buffer(13)]],\n"
        "    constant float &high [[buffer(14)]], constant float &mscale [[buffer(15)]],\n"
        "    uint3 gid [[thread_position_in_grid]]) {\n"
        "    uint half_hd = head_dim >> 1;\n"
        "    uint half_r = rope_dim >> 1;\n"
        "    uint j = gid.x, h = gid.y, r = gid.z;\n"
        "    if (j >= half_hd || h >= n_heads || r >= seq_len) return;\n"
        "    uint xbase = r * xs + h * head_dim;\n"
        "    uint dbase = r * ds + h * head_dim;\n"
        "    if (j >= half_r) {\n"
        "        uint k0 = 2 * j, k1 = 2 * j + 1;\n"
        "        dst[dbase + k0] = x[xbase + k0];\n"
        "        dst[dbase + k1] = x[xbase + k1];\n"
        "        return;\n"
        "    }\n"
        "    float base_freq = exp(-log_theta * (float(2 * j) / float(rope_dim)));\n"
        "    float freq = base_freq / eff_scale;\n"
        "    if (use_ntk != 0u) {\n"
        "        float ramp = clamp((float(j) - low) / (high - low), 0.0f, 1.0f);\n"
        "        float gamma = 1.0f - ramp;\n"
        "        freq = (1.0f - gamma) * (base_freq / eff_scale) + gamma * base_freq;\n"
        "    }\n"
        "    float angle = float(pos_off + r) * freq;\n"
        "    float c = cos(angle) * mscale, s = sin(angle) * mscale;\n"
        "    uint idx0 = interleaved != 0u ? (2 * j) : j;\n"
        "    uint idx1 = interleaved != 0u ? (2 * j + 1) : (half_r + j);\n"
        "    float v0 = x[xbase + idx0], v1 = x[xbase + idx1];\n"
        "    dst[dbase + idx0] = v0 * c - v1 * s;\n"
        "    dst[dbase + idx1] = v0 * s + v1 * c;\n"
        "}\n"
        "kernel void xu_lora_add_f32(device const float *x [[buffer(0)]], device const float *a [[buffer(1)]],\n"
        "    device const float *b [[buffer(2)]], device float *dst [[buffer(3)]],\n"
        "    constant uint &m_cnt [[buffer(4)]], constant uint &in_dim [[buffer(5)]],\n"
        "    constant uint &out_dim [[buffer(6)]], constant uint &rank [[buffer(7)]],\n"
        "    constant uint &xs [[buffer(8)]], constant uint &as [[buffer(9)]],\n"
        "    constant uint &bs [[buffer(10)]], constant uint &ds [[buffer(11)]],\n"
        "    constant float &scale [[buffer(12)]],\n"
        "    uint m [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    if (m >= m_cnt) return;\n"
        "    threadgroup float sh[32];\n"
        "    threadgroup float h_sh[256];\n"
        "    device const float *xp = x + m * xs;\n"
        "    for (uint r0 = 0; r0 < rank; r0 += 256) {\n"
        "        uint chunk = min(256u, rank - r0);\n"
        "        for (uint rk = 0; rk < chunk; ++rk) {\n"
        "            device const float *ap = a + (r0 + rk) * as;\n"
        "            float partial = 0.0f;\n"
        "            for (uint k = tid; k < in_dim; k += tg) { partial += ap[k] * xp[k]; }\n"
        "            float tot = tg_sum(partial, sh, sl, sg, sc);\n"
        "            if (tid == 0) h_sh[rk] = tot * scale;\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        for (uint o = tid; o < out_dim; o += tg) {\n"
        "            device const float *bp = b + o * bs + r0;\n"
        "            float acc = 0.0f;\n"
        "            for (uint rk = 0; rk < chunk; ++rk) { acc += bp[rk] * h_sh[rk]; }\n"
        "            dst[m * ds + o] += acc;\n"
        "        }\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "}\n"
        "kernel void xu_adamw_step_f32(device float *p [[buffer(0)]], device const float *g [[buffer(1)]],\n"
        "    device float *mb [[buffer(2)]], device float *vb [[buffer(3)]],\n"
        "    constant uint &rows [[buffer(4)]], constant uint &cols [[buffer(5)]],\n"
        "    constant uint &ps [[buffer(6)]], constant uint &gs [[buffer(7)]],\n"
        "    constant uint &ms [[buffer(8)]], constant uint &vs [[buffer(9)]],\n"
        "    constant float &lr [[buffer(10)]], constant float &b1 [[buffer(11)]],\n"
        "    constant float &b2 [[buffer(12)]], constant float &eps [[buffer(13)]],\n"
        "    constant float &wd [[buffer(14)]], constant float &inv_bc1 [[buffer(15)]],\n"
        "    constant float &inv_bc2 [[buffer(16)]], uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        float gv = g[gid.y * gs + gid.x];\n"
        "        float mi = b1 * mb[gid.y * ms + gid.x] + (1.0f - b1) * gv;\n"
        "        float vi = b2 * vb[gid.y * vs + gid.x] + (1.0f - b2) * gv * gv;\n"
        "        mb[gid.y * ms + gid.x] = mi;\n"
        "        vb[gid.y * vs + gid.x] = vi;\n"
        "        float mh = mi * inv_bc1;\n"
        "        float vh = vi * inv_bc2;\n"
        "        float pv = p[gid.y * ps + gid.x];\n"
        "        p[gid.y * ps + gid.x] = pv * (1.0f - lr * wd) - lr * mh / (sqrt(vh) + eps);\n"
        "    }\n"
        "}\n"
        "kernel void xu_cross_entropy_grad_f32(device const float *logits [[buffer(0)]],\n"
        "    device const float *targets [[buffer(1)]], device float *dlogits [[buffer(2)]],\n"
        "    device float *losses [[buffer(3)]], constant uint &cols [[buffer(4)]],\n"
        "    constant uint &ls [[buffer(5)]], constant uint &ts [[buffer(6)]],\n"
        "    constant uint &ds [[buffer(7)]], constant uint &los [[buffer(8)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint xbase = row * ls, dbase = row * ds;\n"
        "    float lmax = -INFINITY;\n"
        "    for (uint c = tid; c < cols; c += tg) { lmax = max(lmax, logits[xbase + c]); }\n"
        "    float rmax = tg_max(lmax, sh, sl, sg, sc);\n"
        "    float lsum = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) { lsum += exp(logits[xbase + c] - rmax); }\n"
        "    float tot = tg_sum(lsum, sh, sl, sg, sc);\n"
        "    float inv_sum = 1.0f / tot;\n"
        "    float log_z = rmax + log(tot);\n"
        "    int tgt = int(rint(targets[row * ts]));\n"
        "    for (uint c = tid; c < cols; c += tg) {\n"
        "        float p = exp(logits[xbase + c] - rmax) * inv_sum;\n"
        "        dlogits[dbase + c] = (int(c) == tgt) ? (p - 1.0f) : p;\n"
        "    }\n"
        "    if (tid == 0) {\n"
        "        losses[row * los] = (tgt >= 0 && uint(tgt) < cols) ? (log_z - logits[xbase + uint(tgt)]) : 0.0f;\n"
        "    }\n"
        "}\n"
        "kernel void xu_rmsnorm_bwd_dx_f32(device const float *dy [[buffer(0)]], device const float *x [[buffer(1)]],\n"
        "    device const float *w [[buffer(2)]], device float *dx [[buffer(3)]], device float *inv_rms [[buffer(4)]],\n"
        "    constant uint &cols [[buffer(5)]], constant uint &dys [[buffer(6)]], constant uint &xs [[buffer(7)]],\n"
        "    constant uint &dxs [[buffer(8)]], constant float &eps [[buffer(9)]],\n"
        "    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],\n"
        "    uint tg [[threads_per_threadgroup]], uint sl [[thread_index_in_simdgroup]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint sc [[simdgroups_per_threadgroup]]) {\n"
        "    threadgroup float sh[32];\n"
        "    uint xbase = row * xs, dybase = row * dys, dxbase = row * dxs;\n"
        "    float ss = 0.0f, sdot = 0.0f;\n"
        "    for (uint c = tid; c < cols; c += tg) {\n"
        "        float xv = x[xbase + c];\n"
        "        float gv = dy[dybase + c] * w[c];\n"
        "        ss += xv * xv;\n"
        "        sdot += gv * xv;\n"
        "    }\n"
        "    float tot_ss = tg_sum(ss, sh, sl, sg, sc);\n"
        "    float tot_dot = tg_sum(sdot, sh, sl, sg, sc);\n"
        "    float inv = rsqrt(tot_ss / float(cols) + eps);\n"
        "    float scl = (tot_dot / float(cols)) * inv * inv;\n"
        "    if (tid == 0) inv_rms[row] = inv;\n"
        "    for (uint c = tid; c < cols; c += tg) {\n"
        "        dx[dxbase + c] = inv * (dy[dybase + c] * w[c] - x[xbase + c] * scl);\n"
        "    }\n"
        "}\n"
        "kernel void xu_rmsnorm_bwd_dw_f32(device const float *dy [[buffer(0)]], device const float *x [[buffer(1)]],\n"
        "    device const float *inv_rms [[buffer(2)]], device float *dw [[buffer(3)]],\n"
        "    constant uint &rows [[buffer(4)]], constant uint &cols [[buffer(5)]],\n"
        "    constant uint &dys [[buffer(6)]], constant uint &xs [[buffer(7)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "    if (gid < cols) {\n"
        "        float acc = 0.0f;\n"
        "        for (uint r = 0; r < rows; ++r) {\n"
        "            acc += dy[r * dys + gid] * x[r * xs + gid] * inv_rms[r];\n"
        "        }\n"
        "        dw[gid] = acc;\n"
        "    }\n"
        "}\n"
        "kernel void xu_swiglu_bwd_f32(device const float *dy [[buffer(0)]], device const float *g [[buffer(1)]],\n"
        "    device const float *u [[buffer(2)]], device float *dgate [[buffer(3)]], device float *dup [[buffer(4)]],\n"
        "    constant uint &rows [[buffer(5)]], constant uint &cols [[buffer(6)]],\n"
        "    constant uint &dys [[buffer(7)]], constant uint &gs [[buffer(8)]], constant uint &us [[buffer(9)]],\n"
        "    constant uint &dgs [[buffer(10)]], constant uint &dus [[buffer(11)]],\n"
        "    uint2 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x < cols && gid.y < rows) {\n"
        "        float dyv = dy[gid.y * dys + gid.x];\n"
        "        float gv = g[gid.y * gs + gid.x];\n"
        "        float uv = u[gid.y * us + gid.x];\n"
        "        float sig = gv < -88.0f ? 0.0f : (1.0f / (1.0f + exp(-gv)));\n"
        "        float silu = gv < -88.0f ? 0.0f : (gv * sig);\n"
        "        float dsilu = gv < -88.0f ? 0.0f : (sig * (1.0f + gv * (1.0f - sig)));\n"
        "        dup[gid.y * dus + gid.x] = dyv * silu;\n"
        "        dgate[gid.y * dgs + gid.x] = dyv * uv * dsilu;\n"
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
    id<MTLComputePipelineState> p10 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_qk_norm_f32"] error:&error];
    id<MTLComputePipelineState> p11 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_kv_cache_append_f32"] error:&error];
    id<MTLComputePipelineState> p12 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_attention_decode_f32"] error:&error];
    id<MTLComputePipelineState> p13 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_embedding_f32"] error:&error];
    id<MTLComputePipelineState> p14 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_argmax_f32"] error:&error];
    id<MTLComputePipelineState> p15 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_rope_interleaved_f32"] error:&error];
    id<MTLComputePipelineState> p16 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_add_rmsnorm_f32"] error:&error];
    id<MTLComputePipelineState> p17 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_geglu_f32"] error:&error];
    id<MTLComputePipelineState> p18 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_silu_f32"] error:&error];
    id<MTLComputePipelineState> p19 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_log_softmax_f32"] error:&error];
    id<MTLComputePipelineState> p20 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_moe_topk_f32"] error:&error];
    id<MTLComputePipelineState> p21 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_moe_weighted_add_f32"] error:&error];
    id<MTLComputePipelineState> p22 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_rope_yarn_f32"] error:&error];
    id<MTLComputePipelineState> p23 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_lora_add_f32"] error:&error];
    id<MTLComputePipelineState> p24 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_adamw_step_f32"] error:&error];
    id<MTLComputePipelineState> p25 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_cross_entropy_grad_f32"] error:&error];
    id<MTLComputePipelineState> p26 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_rmsnorm_bwd_dx_f32"] error:&error];
    id<MTLComputePipelineState> p27 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_rmsnorm_bwd_dw_f32"] error:&error];
    id<MTLComputePipelineState> p28 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_swiglu_bwd_f32"] error:&error];
    if (!p0 || !p1 || !p2 || !p3 || !p4 || !p5 || !p6 || !p7 || !p8 || !p9 || !p10 || !p11 || !p12 || !p13 || !p14 || !p15 ||
        !p16 || !p17 || !p18 || !p19 || !p20 || !p21 || !p22 || !p23 || !p24 || !p25 || !p26 || !p27 || !p28) return NO;
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
    d.qkNormPipeline = p10;
    d.kvAppendPipeline = p11;
    d.attnDecodePipeline = p12;
    d.embeddingPipeline = p13;
    d.argmaxPipeline = p14;
    d.ropeInterleavedPipeline = p15;
    d.addRmsnormPipeline = p16;
    d.gegluPipeline = p17;
    d.siluPipeline = p18;
    d.logSoftmaxPipeline = p19;
    d.moeTopkPipeline = p20;
    d.moeWeightedAddPipeline = p21;
    d.ropeYarnPipeline = p22;
    d.loraAddPipeline = p23;
    d.adamwStepPipeline = p24;
    d.crossEntropyGradPipeline = p25;
    d.rmsnormBwdDxPipeline = p26;
    d.rmsnormBwdDwPipeline = p27;
    d.swigluBwdPipeline = p28;
    return YES;
}

static NSUInteger ai_row_tg_width(id<MTLComputePipelineState> pipe) {
    NSUInteger w = 256;
    while (w > 1 && w > pipe.maxTotalThreadsPerThreadgroup) w >>= 1;
    return w;
}

static XMBatch *lookup_batch_for(xu_int bh, XMBuffer *dst, xu_int *err_out) {
    if (bh <= 0) { *err_out = XM_HANDLE; return nil; }
    XMBatch *b = lookup(bh, BATCH);
    if (!b) { *err_out = XM_HANDLE; return nil; }
    if (b.submitted) { *err_out = XM_BUSY; return nil; }
    if (dst && b.owner != dst.owner) { *err_out = XM_DEVICE; return nil; }
    *err_out = 0;
    return b;
}

static xu_int gemm_f32_impl(xu_int bh, xu_int dst, xu_int ah, xu_int bh_m, BOOL transposeLeft, BOOL transposeRight) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *a = lookup(ah, MATRIX_F32), *b = lookup(bh_m, MATRIX_F32);
    if (!c || !a || !b) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (a.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    NSUInteger outRowsA = transposeLeft ? a.columns : a.rows;
    NSUInteger innerA = transposeLeft ? a.rows : a.columns;
    NSUInteger innerB = transposeRight ? b.columns : b.rows;
    NSUInteger outColsB = transposeRight ? b.rows : b.columns;
    if (innerA != innerB || c.rows != outRowsA || c.columns != outColsB) RETURN(XM_SHAPE);
    if (c == a || c == b) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (a == b) ? @[a] : @[a, b];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || a.poisoned || b.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, a) || !batch_normal_input(batch, b)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!MPSSupportsMTLDevice(d.gpu)) RETURN(XM_UNAVAILABLE);
    if (!c.words || !innerA) {
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
        kernel = matrix_kernel_ex(d, c, a, transposeLeft, transposeRight);
        if (!kernel) RETURN(XM_PIPELINE);
        command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
        if (!command) RETURN(XM_COMMAND);
        [kernel encodeToCommandBuffer:command leftMatrix:a.matrix rightMatrix:b.matrix resultMatrix:c.matrix];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, kernel);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(gemm_f32)(xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(-1, dst, ah, bh, NO, NO); }
xu_int XM_API(gemm_f32_nt)(xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(-1, dst, ah, bh, NO, YES); }
xu_int XM_API(gemm_f32_tn)(xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(-1, dst, ah, bh, YES, NO); }
xu_int XM_API(batch_gemm_f32)(xu_int batch, xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(batch < 0 ? 0 : batch, dst, ah, bh, NO, NO); }
xu_int XM_API(batch_gemm_f32_nt)(xu_int batch, xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(batch < 0 ? 0 : batch, dst, ah, bh, NO, YES); }
xu_int XM_API(batch_gemm_f32_tn)(xu_int batch, xu_int dst, xu_int ah, xu_int bh) { return gemm_f32_impl(batch < 0 ? 0 : batch, dst, ah, bh, YES, NO); }

static xu_int rmsnorm_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int wh, xu_int eps_bits) { @autoreleasepool {
    if (eps_bits < 0 || (xu_uint)eps_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t eb = (uint32_t)eps_bits;
    float eps = 0.0f;
    memcpy(&eps, &eb, 4);
    if (!isfinite(eps) || eps <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32), *w = lookup(wh, MATRIX_F32);
    if (!c || !x || !w) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner || w.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns || w.rows != 1 || w.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x || c == w) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (x == w) ? @[x] : @[x, w];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || w.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x) || !batch_normal_input(batch, w)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.rmsnormPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(rmsnorm_f32)(xu_int dst, xu_int xh, xu_int wh, xu_int eps_bits) { return rmsnorm_f32_impl(-1, dst, xh, wh, eps_bits); }
xu_int XM_API(batch_rmsnorm_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh, xu_int eps_bits) { return rmsnorm_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, eps_bits); }

static xu_int softmax_f32_impl(xu_int bh, xu_int dst, xu_int xh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.softmaxPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(softmax_f32)(xu_int dst, xu_int xh) { return softmax_f32_impl(-1, dst, xh); }
xu_int XM_API(batch_softmax_f32)(xu_int batch, xu_int dst, xu_int xh) { return softmax_f32_impl(batch < 0 ? 0 : batch, dst, xh); }

static xu_int binary_elem_f32(xu_int bh, xu_int dst, xu_int ah, xu_int bh_m, int mode) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *a = lookup(ah, MATRIX_F32), *b = lookup(bh_m, MATRIX_F32);
    if (!c || !a || !b) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (a.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != a.rows || c.columns != a.columns || b.rows != a.rows || b.columns != a.columns) RETURN(XM_SHAPE);
    if (c == a || c == b) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (a == b) ? @[a] : @[a, b];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || a.poisoned || b.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, a) || !batch_normal_input(batch, b)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLComputePipelineState> pipe = mode == 1 ? d.swigluPipeline : (mode == 2 ? d.mulPipeline : d.addPipeline);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, pipe);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(swiglu_f32)(xu_int dst, xu_int gate, xu_int up) { return binary_elem_f32(-1, dst, gate, up, 1); }
xu_int XM_API(add_f32)(xu_int dst, xu_int a, xu_int b) { return binary_elem_f32(-1, dst, a, b, 0); }
xu_int XM_API(mul_f32)(xu_int dst, xu_int a, xu_int b) { return binary_elem_f32(-1, dst, a, b, 2); }
xu_int XM_API(batch_swiglu_f32)(xu_int batch, xu_int dst, xu_int gate, xu_int up) { return binary_elem_f32(batch < 0 ? 0 : batch, dst, gate, up, 1); }
xu_int XM_API(batch_add_f32)(xu_int batch, xu_int dst, xu_int a, xu_int b) { return binary_elem_f32(batch < 0 ? 0 : batch, dst, a, b, 0); }
xu_int XM_API(batch_mul_f32)(xu_int batch, xu_int dst, xu_int a, xu_int b) { return binary_elem_f32(batch < 0 ? 0 : batch, dst, a, b, 2); }

static xu_int layernorm_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int wh, xu_int bh_bias, xu_int eps_bits) { @autoreleasepool {
    if (eps_bits < 0 || (xu_uint)eps_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t eb = (uint32_t)eps_bits;
    float eps = 0.0f;
    memcpy(&eps, &eb, 4);
    if (!isfinite(eps) || eps <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32),
             *w = lookup(wh, MATRIX_F32), *b = lookup(bh_bias, MATRIX_F32);
    if (!c || !x || !w || !b) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner || w.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns ||
        w.rows != 1 || w.columns != x.columns ||
        b.rows != 1 || b.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x || c == w || c == b) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x, w, b];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || w.poisoned || b.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x) || !batch_normal_input(batch, w) || !batch_normal_input(batch, b)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.layernormPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(layernorm_f32)(xu_int dst, xu_int xh, xu_int wh, xu_int bh, xu_int eps_bits) { return layernorm_f32_impl(-1, dst, xh, wh, bh, eps_bits); }
xu_int XM_API(batch_layernorm_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh, xu_int bh, xu_int eps_bits) { return layernorm_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, bh, eps_bits); }

static xu_int gelu_f32_impl(xu_int bh, xu_int dst, xu_int xh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.geluPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(gelu_f32)(xu_int dst, xu_int xh) { return gelu_f32_impl(-1, dst, xh); }
xu_int XM_API(batch_gelu_f32)(xu_int batch, xu_int dst, xu_int xh) { return gelu_f32_impl(batch < 0 ? 0 : batch, dst, xh); }

static xu_int scale_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int alpha_bits) { @autoreleasepool {
    if (alpha_bits < 0 || (xu_uint)alpha_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t ab = (uint32_t)alpha_bits;
    float alpha = 0.0f;
    memcpy(&alpha, &ab, 4);
    if (!isfinite(alpha)) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.scalePipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(scale_f32)(xu_int dst, xu_int xh, xu_int alpha_bits) { return scale_f32_impl(-1, dst, xh, alpha_bits); }
xu_int XM_API(batch_scale_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int alpha_bits) { return scale_f32_impl(batch < 0 ? 0 : batch, dst, xh, alpha_bits); }

static xu_int rope_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int pos_offset, xu_int head_dim, xu_int theta_bits, BOOL interleaved) { @autoreleasepool {
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
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns || (x.columns % (NSUInteger)head_dim) != 0) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLComputePipelineState> pipe = interleaved ? d.ropeInterleavedPipeline : d.ropePipeline;
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t seq_len = (uint32_t)c.rows, hd = (uint32_t)head_dim;
        uint32_t nh = (uint32_t)(c.columns / hd);
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        uint32_t poff = (uint32_t)pos_offset;
        uint32_t half_d = hd >> 1;
        NSUInteger tx = MIN((NSUInteger)MAX(1u, MIN(half_d, 64u)), pipe.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:pipe];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, pipe);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(rope_f32)(xu_int dst, xu_int xh, xu_int pos_offset, xu_int head_dim, xu_int theta_bits) { return rope_f32_impl(-1, dst, xh, pos_offset, head_dim, theta_bits, NO); }
xu_int XM_API(batch_rope_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int pos_offset, xu_int head_dim, xu_int theta_bits) { return rope_f32_impl(batch < 0 ? 0 : batch, dst, xh, pos_offset, head_dim, theta_bits, NO); }
xu_int XM_API(rope_interleaved_f32)(xu_int dst, xu_int xh, xu_int pos_offset, xu_int head_dim, xu_int theta_bits) { return rope_f32_impl(-1, dst, xh, pos_offset, head_dim, theta_bits, YES); }
xu_int XM_API(batch_rope_interleaved_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int pos_offset, xu_int head_dim, xu_int theta_bits) { return rope_f32_impl(batch < 0 ? 0 : batch, dst, xh, pos_offset, head_dim, theta_bits, YES); }

static BOOL unpack_f32_bits(xu_int bits_val, float *out_f) {
    if (bits_val < 0 || (xu_uint)bits_val > UINT32_MAX) return NO;
    uint32_t u = (uint32_t)bits_val;
    float f = 0.0f;
    memcpy(&f, &u, 4);
    if (!isfinite(f)) return NO;
    *out_f = f;
    return YES;
}

static xu_int rope_yarn_f32_impl(xu_int bh, xu_int dst, xu_int xh, const xu_i64_span *params) { @autoreleasepool {
    if (!params || params->len != 10 || !params->data) return XM_SHAPE;
    const xu_int *pv = params->data;
    xu_int pos_offset = pv[0], head_dim = pv[1], rdim_in = pv[2], il = pv[3], orig_ctx = pv[4];
    if (pos_offset < 0 || pos_offset > UINT32_MAX || head_dim <= 0 || (head_dim & 1) || head_dim > MATRIX_MAX_DIM) return XM_BOUNDS;
    xu_int rope_dim = (rdim_in <= 0 || rdim_in > head_dim) ? head_dim : rdim_in;
    if (rope_dim <= 0 || (rope_dim & 1)) return XM_SHAPE;
    float theta = 0.0f, scale = 0.0f, beta_fast = 0.0f, beta_slow = 0.0f, attn_factor = 0.0f;
    if (!unpack_f32_bits(pv[5], &theta) || theta <= 0.0f ||
        !unpack_f32_bits(pv[6], &scale) ||
        !unpack_f32_bits(pv[7], &beta_fast) ||
        !unpack_f32_bits(pv[8], &beta_slow) ||
        !unpack_f32_bits(pv[9], &attn_factor)) return XM_NUMERICS;
    float eff_scale = scale <= 0.0f ? 1.0f : scale;
    float log_theta = logf(theta);
    float two_pi = 6.283185307179586f;
    uint32_t use_ntk = (eff_scale > 1.0f && orig_ctx > 0 && beta_fast > 0.0f && beta_slow > 0.0f) ? 1u : 0u;
    uint32_t half_r = (uint32_t)(rope_dim >> 1);
    float low = 0.0f, high = (float)(half_r > 0 ? half_r - 1u : 0u);
    if (use_ntk) {
        low = ((float)rope_dim * logf((float)orig_ctx / (beta_fast * two_pi))) / (2.0f * log_theta);
        high = ((float)rope_dim * logf((float)orig_ctx / (beta_slow * two_pi))) / (2.0f * log_theta);
        if (low < 0.0f) low = 0.0f;
        if (high > (float)(half_r - 1u)) high = (float)(half_r - 1u);
        if (high <= low) high = low + 0.001f;
    }
    float mscale = attn_factor;
    if (mscale <= 0.0f) mscale = eff_scale > 1.0f ? (0.1f * logf(eff_scale) + 1.0f) : 1.0f;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns || (x.columns % (NSUInteger)head_dim) != 0) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t seq_len = (uint32_t)c.rows, hd = (uint32_t)head_dim, rd = (uint32_t)rope_dim;
        uint32_t nh = (uint32_t)(c.columns / hd), il_u = il ? 1u : 0u;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        uint32_t poff = (uint32_t)pos_offset, half_hd = hd >> 1;
        NSUInteger tx = MIN((NSUInteger)MAX(1u, MIN(half_hd, 64u)), d.ropeYarnPipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:d.ropeYarnPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:2];
        [enc setBytes:&nh length:sizeof(nh) atIndex:3];
        [enc setBytes:&hd length:sizeof(hd) atIndex:4];
        [enc setBytes:&rd length:sizeof(rd) atIndex:5];
        [enc setBytes:&il_u length:sizeof(il_u) atIndex:6];
        [enc setBytes:&xs length:sizeof(xs) atIndex:7];
        [enc setBytes:&ds length:sizeof(ds) atIndex:8];
        [enc setBytes:&poff length:sizeof(poff) atIndex:9];
        [enc setBytes:&log_theta length:sizeof(log_theta) atIndex:10];
        [enc setBytes:&eff_scale length:sizeof(eff_scale) atIndex:11];
        [enc setBytes:&use_ntk length:sizeof(use_ntk) atIndex:12];
        [enc setBytes:&low length:sizeof(low) atIndex:13];
        [enc setBytes:&high length:sizeof(high) atIndex:14];
        [enc setBytes:&mscale length:sizeof(mscale) atIndex:15];
        [enc dispatchThreadgroups:MTLSizeMake((half_hd + tx - 1) / tx, nh, seq_len)
            threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.ropeYarnPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(rope_yarn_f32)(xu_int dst, xu_int xh, const xu_i64_span *params) { return rope_yarn_f32_impl(-1, dst, xh, params); }
xu_int XM_API(batch_rope_yarn_f32)(xu_int batch, xu_int dst, xu_int xh, const xu_i64_span *params) { return rope_yarn_f32_impl(batch < 0 ? 0 : batch, dst, xh, params); }

static xu_int causal_attention_f32_impl(xu_int bh, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
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
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (q.owner != c.owner || k.owner != c.owner || v.owner != c.owner) RETURN(XM_DEVICE);
    if (q.columns == 0 || (q.columns % (NSUInteger)n_heads) != 0) RETURN(XM_SHAPE);
    NSUInteger head_dim = q.columns / (NSUInteger)n_heads;
    if (c.rows != q.rows || c.columns != q.columns ||
        k.columns != (NSUInteger)n_kv_heads * head_dim ||
        v.rows != k.rows || v.columns != k.columns || k.rows < q.rows) RETURN(XM_SHAPE);
    if (c == q || c == k || c == v) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[q, k, v];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || q.poisoned || k.poisoned || v.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, q) || !batch_normal_input(batch, k) || !batch_normal_input(batch, v)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t seq_len = (uint32_t)q.rows, kv_len = (uint32_t)k.rows;
        uint32_t nh = (uint32_t)n_heads, kvm = (uint32_t)(n_heads / n_kv_heads), hd = (uint32_t)head_dim;
        uint32_t qs = (uint32_t)(q.rowBytes / 4), ks = (uint32_t)(k.rowBytes / 4);
        uint32_t vs = (uint32_t)(v.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.causalAttnPipeline);
        if (head_dim > 8 * tg) RETURN(XM_SHAPE);
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.causalAttnPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(causal_attention_f32)(xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                    xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return causal_attention_f32_impl(-1, dst, qh, kh, vh, n_heads, n_kv_heads, scale_bits);
}
xu_int XM_API(batch_causal_attention_f32)(xu_int batch, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                          xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return causal_attention_f32_impl(batch < 0 ? 0 : batch, dst, qh, kh, vh, n_heads, n_kv_heads, scale_bits);
}

static xu_int qk_norm_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int wh, xu_int n_heads, xu_int eps_bits) { @autoreleasepool {
    if (n_heads <= 0) return XM_SHAPE;
    if (eps_bits < 0 || (xu_uint)eps_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t eb = (uint32_t)eps_bits;
    float eps = 0.0f;
    memcpy(&eps, &eb, 4);
    if (!isfinite(eps) || eps <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32), *w = lookup(wh, MATRIX_F32);
    if (!c || !x || !w) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner || w.owner != c.owner) RETURN(XM_DEVICE);
    if (x.columns == 0 || (x.columns % (NSUInteger)n_heads) != 0) RETURN(XM_SHAPE);
    NSUInteger head_dim = x.columns / (NSUInteger)n_heads;
    if (c.rows != x.rows || c.columns != x.columns || w.rows != 1 || w.columns != head_dim) RETURN(XM_SHAPE);
    if (c == x || c == w) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (x == w) ? @[x] : @[x, w];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || w.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x) || !batch_normal_input(batch, w)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t nh = (uint32_t)n_heads, hd = (uint32_t)head_dim;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.qkNormPipeline);
        [enc setComputePipelineState:d.qkNormPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:w.memory offset:0 atIndex:1];
        [enc setBuffer:c.memory offset:0 atIndex:2];
        [enc setBytes:&nh length:sizeof(nh) atIndex:3];
        [enc setBytes:&hd length:sizeof(hd) atIndex:4];
        [enc setBytes:&xs length:sizeof(xs) atIndex:5];
        [enc setBytes:&ds length:sizeof(ds) atIndex:6];
        [enc setBytes:&eps length:sizeof(eps) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(nh, x.rows, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.qkNormPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(qk_norm_f32)(xu_int dst, xu_int xh, xu_int wh, xu_int n_heads, xu_int eps_bits) {
    return qk_norm_f32_impl(-1, dst, xh, wh, n_heads, eps_bits);
}
xu_int XM_API(batch_qk_norm_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh, xu_int n_heads, xu_int eps_bits) {
    return qk_norm_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, n_heads, eps_bits);
}

static xu_int kv_cache_append_f32_impl(xu_int bh, xu_int cacheh, xu_int srch, xu_int pos_offset) { @autoreleasepool {
    if (pos_offset < 0) return XM_BOUNDS;
    ENTER;
    XMMatrix *c = lookup(cacheh, MATRIX_F32), *s = lookup(srch, MATRIX_F32);
    if (!c || !s) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (s.owner != c.owner) RETURN(XM_DEVICE);
    if (s.columns != c.columns || (uint64_t)pos_offset + (uint64_t)s.rows > (uint64_t)c.rows) RETURN(XM_SHAPE);
    if (c == s) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[s];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || s.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, s)) RETURN(XM_NUMERICS);
    BOOL keep_normal = (!batch && c.numberClass == MATRIX_CLASS_NORMAL && s.numberClass == MATRIX_CLASS_NORMAL);
    XMDevice *d = c.owner;
    if (!s.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)s.rows, cols = (uint32_t)s.columns;
        uint32_t ss = (uint32_t)(s.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        uint32_t po = (uint32_t)pos_offset;
        [enc setComputePipelineState:d.kvAppendPipeline];
        [enc setBuffer:s.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&rows length:sizeof(rows) atIndex:2];
        [enc setBytes:&cols length:sizeof(cols) atIndex:3];
        [enc setBytes:&ss length:sizeof(ss) atIndex:4];
        [enc setBytes:&ds length:sizeof(ds) atIndex:5];
        [enc setBytes:&po length:sizeof(po) atIndex:6];
        NSUInteger tx = 16, ty = 16;
        if (d.kvAppendPipeline.maxTotalThreadsPerThreadgroup < tx * ty) { tx = 8; ty = 8; }
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.kvAppendPipeline);
    if (handle >= 0) c.numberClass = keep_normal ? MATRIX_CLASS_NORMAL : MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(kv_cache_append_f32)(xu_int cache, xu_int src, xu_int pos_offset) {
    return kv_cache_append_f32_impl(-1, cache, src, pos_offset);
}
xu_int XM_API(batch_kv_cache_append_f32)(xu_int batch, xu_int cache, xu_int src, xu_int pos_offset) {
    return kv_cache_append_f32_impl(batch < 0 ? 0 : batch, cache, src, pos_offset);
}

static xu_int attention_decode_f32_impl(xu_int bh, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                          xu_int kv_len, xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) { @autoreleasepool {
    if (kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || (n_heads % n_kv_heads) != 0) return XM_SHAPE;
    if (scale_bits < 0 || (xu_uint)scale_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t sb = (uint32_t)scale_bits;
    float scale = 0.0f;
    memcpy(&scale, &sb, 4);
    if (!isfinite(scale) || scale <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *q = lookup(qh, MATRIX_F32),
             *k = lookup(kh, MATRIX_F32), *v = lookup(vh, MATRIX_F32);
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
    if (c == q || c == k || c == v) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[q, k, v];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || q.poisoned || k.poisoned || v.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, q) || !batch_normal_input(batch, k) || !batch_normal_input(batch, v)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    NSUInteger tg = ai_row_tg_width(d.attnDecodePipeline);
    if (head_dim > 8 * tg) RETURN(XM_SHAPE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t kvl = (uint32_t)kv_len, nh = (uint32_t)n_heads, kvm = (uint32_t)(n_heads / n_kv_heads);
        uint32_t hd = (uint32_t)head_dim, ks = (uint32_t)(k.rowBytes / 4), vs = (uint32_t)(v.rowBytes / 4);
        [enc setComputePipelineState:d.attnDecodePipeline];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.attnDecodePipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(attention_decode_f32)(xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                    xu_int kv_len, xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return attention_decode_f32_impl(-1, dst, qh, kh, vh, kv_len, n_heads, n_kv_heads, scale_bits);
}
xu_int XM_API(batch_attention_decode_f32)(xu_int batch, xu_int dst, xu_int qh, xu_int kh, xu_int vh,
                                          xu_int kv_len, xu_int n_heads, xu_int n_kv_heads, xu_int scale_bits) {
    return attention_decode_f32_impl(batch < 0 ? 0 : batch, dst, qh, kh, vh, kv_len, n_heads, n_kv_heads, scale_bits);
}

static xu_int embedding_f32_impl(xu_int bh, xu_int dst, xu_int tableh, const xu_i64_span *ids) { @autoreleasepool {
    if (!ids || !host_range(ids->len, ids->data, 0, ids->len)) return XM_BOUNDS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *table = lookup(tableh, MATRIX_F32);
    if (!c || !table) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (table.owner != c.owner) RETURN(XM_DEVICE);
    if ((NSUInteger)ids->len != c.rows || table.columns != c.columns) RETURN(XM_SHAPE);
    if (c == table) RETURN(XM_ALIAS);
    for (xu_int i = 0; i < ids->len; ++i) {
        xu_int row_id = ids->data[i];
        if (row_id < 0 || (uint64_t)row_id >= table.rows) RETURN(XM_BOUNDS);
    }
    NSArray<XMBuffer *> *inputs = @[table];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || table.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, table)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLBuffer> idBuf = [d.gpu newBufferWithLength:MAX((NSUInteger)16, (NSUInteger)ids->len * 4) options:MTLResourceStorageModeShared];
    if (!idBuf) RETURN(XM_ALLOC);
    uint32_t *u32_ids = (uint32_t *)idBuf.contents;
    for (xu_int i = 0; i < ids->len; ++i) u32_ids[i] = (uint32_t)ids->data[i];
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns;
        uint32_t ts = (uint32_t)(table.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        [enc setComputePipelineState:d.embeddingPipeline];
        [enc setBuffer:table.memory offset:0 atIndex:0];
        [enc setBuffer:idBuf offset:0 atIndex:1];
        [enc setBuffer:c.memory offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&ts length:sizeof(ts) atIndex:5];
        [enc setBytes:&ds length:sizeof(ds) atIndex:6];
        NSUInteger tx = 16, ty = 16;
        if (d.embeddingPipeline.maxTotalThreadsPerThreadgroup < tx * ty) { tx = 8; ty = 8; }
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.embeddingPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(embedding_f32)(xu_int dst, xu_int table, const xu_i64_span *ids) {
    return embedding_f32_impl(-1, dst, table, ids);
}
xu_int XM_API(batch_embedding_f32)(xu_int batch, xu_int dst, xu_int table, const xu_i64_span *ids) {
    return embedding_f32_impl(batch < 0 ? 0 : batch, dst, table, ids);
}

static xu_int argmax_f32_impl(xu_int bh, xu_int dst, xu_int xh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, c, &berr);
        if (!batch) RETURN(berr);
    }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (x.columns == 0 || c.rows != x.rows || c.columns != 1) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t cols = (uint32_t)x.columns, xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.argmaxPipeline);
        [enc setComputePipelineState:d.argmaxPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc setBytes:&xs length:sizeof(xs) atIndex:3];
        [enc setBytes:&ds length:sizeof(ds) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(x.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.argmaxPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(argmax_f32)(xu_int dst, xu_int xh) { return argmax_f32_impl(-1, dst, xh); }
xu_int XM_API(batch_argmax_f32)(xu_int batch, xu_int dst, xu_int xh) { return argmax_f32_impl(batch < 0 ? 0 : batch, dst, xh); }

static xu_int add_rmsnorm_f32_impl(xu_int bh, xu_int dnh, xu_int dsh, xu_int ah, xu_int bhm, xu_int wh, xu_int eps_bits) { @autoreleasepool {
    if (eps_bits < 0 || (xu_uint)eps_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t eb = (uint32_t)eps_bits;
    float eps = 0.0f;
    memcpy(&eps, &eb, 4);
    if (!isfinite(eps) || eps <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *dn = lookup(dnh, MATRIX_F32), *ds_m = lookup(dsh, MATRIX_F32),
             *a = lookup(ah, MATRIX_F32), *b = lookup(bhm, MATRIX_F32), *w = lookup(wh, MATRIX_F32);
    if (!dn || !ds_m || !a || !b || !w) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) {
        xu_int berr = 0;
        batch = lookup_batch_for(bh, dn, &berr);
        if (!batch) RETURN(berr);
    }
    if (ds_m.owner != dn.owner || a.owner != dn.owner || b.owner != dn.owner || w.owner != dn.owner) RETURN(XM_DEVICE);
    if (dn.rows != a.rows || dn.columns != a.columns || ds_m.rows != a.rows || ds_m.columns != a.columns ||
        b.rows != a.rows || b.columns != a.columns || w.rows != 1 || w.columns != a.columns) RETURN(XM_SHAPE);
    if (dn == ds_m || dn == a || dn == b || dn == w || ds_m == a || ds_m == b || ds_m == w) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (a == b) ? @[a, w] : @[a, b, w];
    NSArray<XMBuffer *> *writes = @[dn, ds_m];
    if (!op_can_access2(batch, dn, ds_m, inputs)) RETURN(XM_BUSY);
    if (dn.poisoned || ds_m.poisoned || a.poisoned || b.poisoned || w.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, a) || !batch_normal_input(batch, b) || !batch_normal_input(batch, w)) RETURN(XM_NUMERICS);
    XMDevice *d = dn.owner;
    if (!dn.words) RETURN(finish_op2(batch, d, writes, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t cols = (uint32_t)a.columns, as = (uint32_t)(a.rowBytes / 4), bs = (uint32_t)(b.rowBytes / 4);
        uint32_t dns = (uint32_t)(dn.rowBytes / 4), dss = (uint32_t)(ds_m.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.addRmsnormPipeline);
        [enc setComputePipelineState:d.addRmsnormPipeline];
        [enc setBuffer:a.memory offset:0 atIndex:0];
        [enc setBuffer:b.memory offset:0 atIndex:1];
        [enc setBuffer:w.memory offset:0 atIndex:2];
        [enc setBuffer:dn.memory offset:0 atIndex:3];
        [enc setBuffer:ds_m.memory offset:0 atIndex:4];
        [enc setBytes:&cols length:sizeof(cols) atIndex:5];
        [enc setBytes:&as length:sizeof(as) atIndex:6];
        [enc setBytes:&bs length:sizeof(bs) atIndex:7];
        [enc setBytes:&dns length:sizeof(dns) atIndex:8];
        [enc setBytes:&dss length:sizeof(dss) atIndex:9];
        [enc setBytes:&eps length:sizeof(eps) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(a.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op2(batch, d, writes, inputs, command, d.addRmsnormPipeline);
    if (handle >= 0) { dn.numberClass = MATRIX_CLASS_UNKNOWN; ds_m.numberClass = MATRIX_CLASS_UNKNOWN; }
    RETURN(handle);
} }

xu_int XM_API(add_rmsnorm_f32)(xu_int dst_norm, xu_int dst_sum, xu_int ah, xu_int bh_m, xu_int wh, xu_int eps_bits) {
    return add_rmsnorm_f32_impl(-1, dst_norm, dst_sum, ah, bh_m, wh, eps_bits);
}
xu_int XM_API(batch_add_rmsnorm_f32)(xu_int batch, xu_int dst_norm, xu_int dst_sum, xu_int ah, xu_int bh_m, xu_int wh, xu_int eps_bits) {
    return add_rmsnorm_f32_impl(batch < 0 ? 0 : batch, dst_norm, dst_sum, ah, bh_m, wh, eps_bits);
}

static xu_int geglu_f32_impl(xu_int bh, xu_int dst, xu_int gh, xu_int uh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *g = lookup(gh, MATRIX_F32), *u = lookup(uh, MATRIX_F32);
    if (!c || !g || !u) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, c, &berr); if (!batch) RETURN(berr); }
    if (g.owner != c.owner || u.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != g.rows || c.columns != g.columns || u.rows != g.rows || u.columns != g.columns) RETURN(XM_SHAPE);
    if (c == g || c == u) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (g == u) ? @[g] : @[g, u];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || g.poisoned || u.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, g) || !batch_normal_input(batch, u)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns;
        uint32_t gs = (uint32_t)(g.rowBytes / 4), us = (uint32_t)(u.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, d.gegluPipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, d.gegluPipeline.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:d.gegluPipeline];
        [enc setBuffer:g.memory offset:0 atIndex:0];
        [enc setBuffer:u.memory offset:0 atIndex:1];
        [enc setBuffer:c.memory offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&gs length:sizeof(gs) atIndex:5];
        [enc setBytes:&us length:sizeof(us) atIndex:6];
        [enc setBytes:&ds length:sizeof(ds) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.gegluPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(geglu_f32)(xu_int dst, xu_int gate, xu_int up) { return geglu_f32_impl(-1, dst, gate, up); }
xu_int XM_API(batch_geglu_f32)(xu_int batch, xu_int dst, xu_int gate, xu_int up) { return geglu_f32_impl(batch < 0 ? 0 : batch, dst, gate, up); }

static xu_int silu_f32_impl(xu_int bh, xu_int dst, xu_int xh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, c, &berr); if (!batch) RETURN(berr); }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, d.siluPipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, d.siluPipeline.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:d.siluPipeline];
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
    xu_int handle = finish_op(batch, d, c, inputs, command, d.siluPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(silu_f32)(xu_int dst, xu_int xh) { return silu_f32_impl(-1, dst, xh); }
xu_int XM_API(batch_silu_f32)(xu_int batch, xu_int dst, xu_int xh) { return silu_f32_impl(batch < 0 ? 0 : batch, dst, xh); }

static xu_int log_softmax_f32_impl(xu_int bh, xu_int dst, xu_int xh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32);
    if (!c || !x) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, c, &berr); if (!batch) RETURN(berr); }
    if (x.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != x.columns) RETURN(XM_SHAPE);
    if (c == x) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, x)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t cols = (uint32_t)x.columns, xs = (uint32_t)(x.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.logSoftmaxPipeline);
        [enc setComputePipelineState:d.logSoftmaxPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:c.memory offset:0 atIndex:1];
        [enc setBytes:&cols length:sizeof(cols) atIndex:2];
        [enc setBytes:&xs length:sizeof(xs) atIndex:3];
        [enc setBytes:&ds length:sizeof(ds) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(x.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.logSoftmaxPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(log_softmax_f32)(xu_int dst, xu_int xh) { return log_softmax_f32_impl(-1, dst, xh); }
xu_int XM_API(batch_log_softmax_f32)(xu_int batch, xu_int dst, xu_int xh) { return log_softmax_f32_impl(batch < 0 ? 0 : batch, dst, xh); }

static xu_int moe_topk_f32_impl(xu_int bh, xu_int oih, xu_int owh, xu_int lh,
                                  xu_int k, xu_int use_sig, xu_int norm_topk, xu_int wscale_bits) { @autoreleasepool {
    if (k <= 0) return XM_SHAPE;
    if (wscale_bits < 0 || (xu_uint)wscale_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t wb = (uint32_t)wscale_bits;
    float wscale = 1.0f;
    memcpy(&wscale, &wb, 4);
    if (!isfinite(wscale)) return XM_NUMERICS;
    ENTER;
    XMMatrix *oi = lookup(oih, MATRIX_F32), *ow = lookup(owh, MATRIX_F32), *lg = lookup(lh, MATRIX_F32);
    if (!oi || !ow || !lg) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, oi, &berr); if (!batch) RETURN(berr); }
    if (ow.owner != oi.owner || lg.owner != oi.owner) RETURN(XM_DEVICE);
    if (oi.rows != lg.rows || ow.rows != lg.rows || oi.columns != (NSUInteger)k || ow.columns != (NSUInteger)k || (NSUInteger)k > lg.columns) RETURN(XM_SHAPE);
    if (oi == ow || oi == lg || ow == lg) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[lg];
    NSArray<XMBuffer *> *writes = @[oi, ow];
    if (!op_can_access2(batch, oi, ow, inputs)) RETURN(XM_BUSY);
    if (oi.poisoned || ow.poisoned || lg.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, lg)) RETURN(XM_NUMERICS);
    XMDevice *d = oi.owner;
    if (!oi.words) RETURN(finish_op2(batch, d, writes, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)lg.rows, n_exp = (uint32_t)lg.columns, kv = (uint32_t)k;
        uint32_t us = use_sig ? 1u : 0u, nt = norm_topk ? 1u : 0u;
        uint32_t ls = (uint32_t)(lg.rowBytes / 4), is = (uint32_t)(oi.rowBytes / 4), ws = (uint32_t)(ow.rowBytes / 4);
        NSUInteger tw = MIN((NSUInteger)64, d.moeTopkPipeline.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:d.moeTopkPipeline];
        [enc setBuffer:lg.memory offset:0 atIndex:0];
        [enc setBuffer:oi.memory offset:0 atIndex:1];
        [enc setBuffer:ow.memory offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&n_exp length:sizeof(n_exp) atIndex:4];
        [enc setBytes:&kv length:sizeof(kv) atIndex:5];
        [enc setBytes:&us length:sizeof(us) atIndex:6];
        [enc setBytes:&nt length:sizeof(nt) atIndex:7];
        [enc setBytes:&wscale length:sizeof(wscale) atIndex:8];
        [enc setBytes:&ls length:sizeof(ls) atIndex:9];
        [enc setBytes:&is length:sizeof(is) atIndex:10];
        [enc setBytes:&ws length:sizeof(ws) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake((lg.rows + tw - 1) / tw, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op2(batch, d, writes, inputs, command, d.moeTopkPipeline);
    if (handle >= 0) { oi.numberClass = MATRIX_CLASS_UNKNOWN; ow.numberClass = MATRIX_CLASS_UNKNOWN; }
    RETURN(handle);
} }

xu_int XM_API(moe_topk_f32)(xu_int out_idx, xu_int out_w, xu_int logits, xu_int k, xu_int use_sig, xu_int norm_topk, xu_int wscale_bits) {
    return moe_topk_f32_impl(-1, out_idx, out_w, logits, k, use_sig, norm_topk, wscale_bits);
}
xu_int XM_API(batch_moe_topk_f32)(xu_int batch, xu_int out_idx, xu_int out_w, xu_int logits, xu_int k, xu_int use_sig, xu_int norm_topk, xu_int wscale_bits) {
    return moe_topk_f32_impl(batch < 0 ? 0 : batch, out_idx, out_w, logits, k, use_sig, norm_topk, wscale_bits);
}

static xu_int moe_weighted_add_f32_impl(xu_int bh, xu_int dst, xu_int eh, xu_int wh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *eo = lookup(eh, MATRIX_F32), *w = lookup(wh, MATRIX_F32);
    if (!c || !eo || !w) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, c, &berr); if (!batch) RETURN(berr); }
    if (eo.owner != c.owner || w.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != w.rows || c.columns != eo.columns || eo.rows != w.rows * w.columns || w.columns == 0) RETURN(XM_SHAPE);
    if (c == eo || c == w) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[eo, w];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || eo.poisoned || w.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, eo) || !batch_normal_input(batch, w)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)c.rows, cols = (uint32_t)c.columns, k = (uint32_t)w.columns;
        uint32_t es = (uint32_t)(eo.rowBytes / 4), ws = (uint32_t)(w.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, d.moeWeightedAddPipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, d.moeWeightedAddPipeline.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:d.moeWeightedAddPipeline];
        [enc setBuffer:eo.memory offset:0 atIndex:0];
        [enc setBuffer:w.memory offset:0 atIndex:1];
        [enc setBuffer:c.memory offset:0 atIndex:2];
        [enc setBytes:&rows length:sizeof(rows) atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&k length:sizeof(k) atIndex:5];
        [enc setBytes:&es length:sizeof(es) atIndex:6];
        [enc setBytes:&ws length:sizeof(ws) atIndex:7];
        [enc setBytes:&ds length:sizeof(ds) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.moeWeightedAddPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(moe_weighted_add_f32)(xu_int dst, xu_int expert_outs, xu_int weights) {
    return moe_weighted_add_f32_impl(-1, dst, expert_outs, weights);
}
xu_int XM_API(batch_moe_weighted_add_f32)(xu_int batch, xu_int dst, xu_int expert_outs, xu_int weights) {
    return moe_weighted_add_f32_impl(batch < 0 ? 0 : batch, dst, expert_outs, weights);
}

static xu_int lora_add_f32_impl(xu_int bh, xu_int dst, xu_int xh, xu_int ah, xu_int bmh, xu_int scale_bits) { @autoreleasepool {
    if (scale_bits < 0 || (xu_uint)scale_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t sb = (uint32_t)scale_bits;
    float scale = 0.0f;
    memcpy(&scale, &sb, 4);
    if (!isfinite(scale)) return XM_NUMERICS;
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *x = lookup(xh, MATRIX_F32),
             *a = lookup(ah, MATRIX_F32), *bm = lookup(bmh, MATRIX_F32);
    if (!c || !x || !a || !bm) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, c, &berr); if (!batch) RETURN(berr); }
    if (x.owner != c.owner || a.owner != c.owner || bm.owner != c.owner) RETURN(XM_DEVICE);
    if (c.rows != x.rows || c.columns != bm.rows || a.columns != x.columns ||
        a.rows != bm.columns || a.rows == 0) RETURN(XM_SHAPE);
    if (c == x || c == a || c == bm) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[x, a, bm];
    if (!op_can_access(batch, c, inputs)) RETURN(XM_BUSY);
    if (c.poisoned || x.poisoned || a.poisoned || bm.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, c) || !batch_normal_input(batch, x) ||
        !batch_normal_input(batch, a) || !batch_normal_input(batch, bm)) RETURN(XM_NUMERICS);
    XMDevice *d = c.owner;
    if (!c.words) RETURN(finish_op(batch, d, c, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t mc = (uint32_t)c.rows, in_d = (uint32_t)x.columns, out_d = (uint32_t)c.columns, rk = (uint32_t)a.rows;
        uint32_t xs = (uint32_t)(x.rowBytes / 4), as = (uint32_t)(a.rowBytes / 4);
        uint32_t bs = (uint32_t)(bm.rowBytes / 4), ds = (uint32_t)(c.rowBytes / 4);
        NSUInteger tg = ai_row_tg_width(d.loraAddPipeline);
        [enc setComputePipelineState:d.loraAddPipeline];
        [enc setBuffer:x.memory offset:0 atIndex:0];
        [enc setBuffer:a.memory offset:0 atIndex:1];
        [enc setBuffer:bm.memory offset:0 atIndex:2];
        [enc setBuffer:c.memory offset:0 atIndex:3];
        [enc setBytes:&mc length:sizeof(mc) atIndex:4];
        [enc setBytes:&in_d length:sizeof(in_d) atIndex:5];
        [enc setBytes:&out_d length:sizeof(out_d) atIndex:6];
        [enc setBytes:&rk length:sizeof(rk) atIndex:7];
        [enc setBytes:&xs length:sizeof(xs) atIndex:8];
        [enc setBytes:&as length:sizeof(as) atIndex:9];
        [enc setBytes:&bs length:sizeof(bs) atIndex:10];
        [enc setBytes:&ds length:sizeof(ds) atIndex:11];
        [enc setBytes:&scale length:sizeof(scale) atIndex:12];
        [enc dispatchThreadgroups:MTLSizeMake(mc, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op(batch, d, c, inputs, command, d.loraAddPipeline);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }

xu_int XM_API(lora_add_f32)(xu_int dst, xu_int x, xu_int a, xu_int b, xu_int scale_bits) {
    return lora_add_f32_impl(-1, dst, x, a, b, scale_bits);
}
xu_int XM_API(batch_lora_add_f32)(xu_int batch, xu_int dst, xu_int x, xu_int a, xu_int b, xu_int scale_bits) {
    return lora_add_f32_impl(batch < 0 ? 0 : batch, dst, x, a, b, scale_bits);
}

static xu_int adamw_step_f32_impl(xu_int bh, xu_int ph, xu_int gh, xu_int mh, xu_int vh,
                                  const xu_i64_span *params) { @autoreleasepool {
    if (!params || !host_range(params->len, params->data, 0, params->len) || params->len < 6) return XM_BOUNDS;
    const xu_int *pv = params->data;
    float fvals[5];
    for (int i = 0; i < 5; ++i) {
        if (pv[i] < 0 || (xu_uint)pv[i] > UINT32_MAX) return XM_NUMERICS;
        uint32_t ub = (uint32_t)pv[i];
        memcpy(&fvals[i], &ub, 4);
        if (!isfinite(fvals[i])) return XM_NUMERICS;
    }
    xu_int step = pv[5];
    if (step <= 0) return XM_SHAPE;
    float lr = fvals[0], b1 = fvals[1], b2 = fvals[2], eps = fvals[3], wd = fvals[4];
    double bc1d = 1.0 - pow((double)b1, (double)step);
    double bc2d = 1.0 - pow((double)b2, (double)step);
    if (!(bc1d > 0.0) || !(bc2d > 0.0) || eps <= 0.0f) return XM_NUMERICS;
    float inv_bc1 = (float)(1.0 / bc1d), inv_bc2 = (float)(1.0 / bc2d);
    ENTER;
    XMMatrix *p = lookup(ph, MATRIX_F32), *g = lookup(gh, MATRIX_F32),
             *mb = lookup(mh, MATRIX_F32), *vb = lookup(vh, MATRIX_F32);
    if (!p || !g || !mb || !vb) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, p, &berr); if (!batch) RETURN(berr); }
    if (g.owner != p.owner || mb.owner != p.owner || vb.owner != p.owner) RETURN(XM_DEVICE);
    if (p.rows != g.rows || p.columns != g.columns ||
        mb.rows != p.rows || mb.columns != p.columns ||
        vb.rows != p.rows || vb.columns != p.columns) RETURN(XM_SHAPE);
    if (p == g || p == mb || p == vb || mb == vb || mb == g || vb == g) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[g];
    NSArray<XMBuffer *> *writes = @[p, mb, vb];
    if (!op_can_access(batch, p, inputs) || !op_can_access(batch, mb, inputs) || !op_can_access(batch, vb, inputs)) RETURN(XM_BUSY);
    if (p.poisoned || g.poisoned || mb.poisoned || vb.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, p) || !batch_normal_input(batch, g) ||
        !batch_normal_input(batch, mb) || !batch_normal_input(batch, vb)) RETURN(XM_NUMERICS);
    XMDevice *d = p.owner;
    if (!p.words) RETURN(finish_op2(batch, d, writes, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)p.rows, cols = (uint32_t)p.columns;
        uint32_t ps = (uint32_t)(p.rowBytes / 4), gs = (uint32_t)(g.rowBytes / 4);
        uint32_t ms = (uint32_t)(mb.rowBytes / 4), vs = (uint32_t)(vb.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, d.adamwStepPipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, d.adamwStepPipeline.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:d.adamwStepPipeline];
        [enc setBuffer:p.memory offset:0 atIndex:0];
        [enc setBuffer:g.memory offset:0 atIndex:1];
        [enc setBuffer:mb.memory offset:0 atIndex:2];
        [enc setBuffer:vb.memory offset:0 atIndex:3];
        [enc setBytes:&rows length:sizeof(rows) atIndex:4];
        [enc setBytes:&cols length:sizeof(cols) atIndex:5];
        [enc setBytes:&ps length:sizeof(ps) atIndex:6];
        [enc setBytes:&gs length:sizeof(gs) atIndex:7];
        [enc setBytes:&ms length:sizeof(ms) atIndex:8];
        [enc setBytes:&vs length:sizeof(vs) atIndex:9];
        [enc setBytes:&lr length:sizeof(lr) atIndex:10];
        [enc setBytes:&b1 length:sizeof(b1) atIndex:11];
        [enc setBytes:&b2 length:sizeof(b2) atIndex:12];
        [enc setBytes:&eps length:sizeof(eps) atIndex:13];
        [enc setBytes:&wd length:sizeof(wd) atIndex:14];
        [enc setBytes:&inv_bc1 length:sizeof(inv_bc1) atIndex:15];
        [enc setBytes:&inv_bc2 length:sizeof(inv_bc2) atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op2(batch, d, writes, inputs, command, d.adamwStepPipeline);
    if (handle >= 0) {
        p.numberClass = MATRIX_CLASS_UNKNOWN;
        mb.numberClass = MATRIX_CLASS_UNKNOWN;
        vb.numberClass = MATRIX_CLASS_UNKNOWN;
    }
    RETURN(handle);
} }

xu_int XM_API(adamw_step_f32)(xu_int param, xu_int grad, xu_int m_buf, xu_int v_buf, const xu_i64_span *params) {
    return adamw_step_f32_impl(-1, param, grad, m_buf, v_buf, params);
}
xu_int XM_API(batch_adamw_step_f32)(xu_int batch, xu_int param, xu_int grad, xu_int m_buf, xu_int v_buf, const xu_i64_span *params) {
    return adamw_step_f32_impl(batch < 0 ? 0 : batch, param, grad, m_buf, v_buf, params);
}

static xu_int cross_entropy_grad_f32_impl(xu_int bh, xu_int dlh, xu_int lsh, xu_int lh, xu_int th) { @autoreleasepool {
    ENTER;
    XMMatrix *dl = lookup(dlh, MATRIX_F32), *ls_m = lookup(lsh, MATRIX_F32),
             *lg = lookup(lh, MATRIX_F32), *tg = lookup(th, MATRIX_F32);
    if (!dl || !ls_m || !lg || !tg) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, dl, &berr); if (!batch) RETURN(berr); }
    if (ls_m.owner != dl.owner || lg.owner != dl.owner || tg.owner != dl.owner) RETURN(XM_DEVICE);
    if (dl.rows != lg.rows || dl.columns != lg.columns || lg.columns == 0 ||
        ls_m.rows != lg.rows || ls_m.columns != 1 || tg.rows != lg.rows || tg.columns != 1) RETURN(XM_SHAPE);
    if (dl == ls_m || dl == lg || dl == tg || ls_m == lg || ls_m == tg) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[lg, tg];
    NSArray<XMBuffer *> *writes = @[dl, ls_m];
    if (!op_can_access2(batch, dl, ls_m, inputs)) RETURN(XM_BUSY);
    if (dl.poisoned || ls_m.poisoned || lg.poisoned || tg.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, lg) || !batch_normal_input(batch, tg)) RETURN(XM_NUMERICS);
    XMDevice *d = dl.owner;
    if (!dl.words) RETURN(finish_op2(batch, d, writes, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t cols = (uint32_t)lg.columns;
        uint32_t lgs = (uint32_t)(lg.rowBytes / 4), tgs = (uint32_t)(tg.rowBytes / 4);
        uint32_t ds = (uint32_t)(dl.rowBytes / 4), los = (uint32_t)(ls_m.rowBytes / 4);
        NSUInteger tgw = ai_row_tg_width(d.crossEntropyGradPipeline);
        [enc setComputePipelineState:d.crossEntropyGradPipeline];
        [enc setBuffer:lg.memory offset:0 atIndex:0];
        [enc setBuffer:tg.memory offset:0 atIndex:1];
        [enc setBuffer:dl.memory offset:0 atIndex:2];
        [enc setBuffer:ls_m.memory offset:0 atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&lgs length:sizeof(lgs) atIndex:5];
        [enc setBytes:&tgs length:sizeof(tgs) atIndex:6];
        [enc setBytes:&ds length:sizeof(ds) atIndex:7];
        [enc setBytes:&los length:sizeof(los) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(lg.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op2(batch, d, writes, inputs, command, d.crossEntropyGradPipeline);
    if (handle >= 0) { dl.numberClass = MATRIX_CLASS_UNKNOWN; ls_m.numberClass = MATRIX_CLASS_UNKNOWN; }
    RETURN(handle);
} }

xu_int XM_API(cross_entropy_grad_f32)(xu_int dlogits, xu_int losses, xu_int logits, xu_int targets) {
    return cross_entropy_grad_f32_impl(-1, dlogits, losses, logits, targets);
}
xu_int XM_API(batch_cross_entropy_grad_f32)(xu_int batch, xu_int dlogits, xu_int losses, xu_int logits, xu_int targets) {
    return cross_entropy_grad_f32_impl(batch < 0 ? 0 : batch, dlogits, losses, logits, targets);
}

static xu_int rmsnorm_backward_f32_impl(xu_int bh, xu_int dxh, xu_int dwh, xu_int dyh,
                                        xu_int xh, xu_int wh, xu_int eps_bits) { @autoreleasepool {
    if (eps_bits < 0 || (xu_uint)eps_bits > UINT32_MAX) return XM_NUMERICS;
    uint32_t eb = (uint32_t)eps_bits;
    float eps = 0.0f;
    memcpy(&eps, &eb, 4);
    if (!isfinite(eps) || eps <= 0.0f) return XM_NUMERICS;
    ENTER;
    XMMatrix *dx = lookup(dxh, MATRIX_F32), *dw = lookup(dwh, MATRIX_F32),
             *dy = lookup(dyh, MATRIX_F32), *x = lookup(xh, MATRIX_F32), *w = lookup(wh, MATRIX_F32);
    if (!dx || !dw || !dy || !x || !w) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, dx, &berr); if (!batch) RETURN(berr); }
    if (dw.owner != dx.owner || dy.owner != dx.owner || x.owner != dx.owner || w.owner != dx.owner) RETURN(XM_DEVICE);
    if (x.columns == 0 || dx.rows != x.rows || dx.columns != x.columns ||
        dy.rows != x.rows || dy.columns != x.columns ||
        dw.rows != 1 || dw.columns != x.columns ||
        w.rows != 1 || w.columns != x.columns) RETURN(XM_SHAPE);
    if (dx == dw || dx == dy || dx == x || dx == w || dw == dy || dw == x || dw == w) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = (dy == x) ? @[x, w] : @[dy, x, w];
    NSArray<XMBuffer *> *writes = @[dx, dw];
    if (!op_can_access2(batch, dx, dw, inputs)) RETURN(XM_BUSY);
    if (dx.poisoned || dw.poisoned || dy.poisoned || x.poisoned || w.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, dy) || !batch_normal_input(batch, x) || !batch_normal_input(batch, w)) RETURN(XM_NUMERICS);
    XMDevice *d = dx.owner;
    if (!dx.words) {
        xu_int fence = finish_op2(batch, d, writes, inputs, nil, nil);
        if (fence >= 0 && dw.words && !batch) memset(dw.memory.contents, 0, dw.memory.length);
        RETURN(fence);
    }
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLBuffer> invBuf = [d.gpu newBufferWithLength:MAX((NSUInteger)16, (NSUInteger)x.rows * 4) options:MTLResourceStorageModePrivate];
    if (!invBuf) RETURN(XM_ALLOC);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        uint32_t rows = (uint32_t)x.rows, cols = (uint32_t)x.columns;
        uint32_t dys = (uint32_t)(dy.rowBytes / 4), xs = (uint32_t)(x.rowBytes / 4), dxs = (uint32_t)(dx.rowBytes / 4);
        id<MTLComputeCommandEncoder> enc1 = [command computeCommandEncoder];
        if (!enc1) RETURN(XM_COMMAND);
        NSUInteger tg1 = ai_row_tg_width(d.rmsnormBwdDxPipeline);
        [enc1 setComputePipelineState:d.rmsnormBwdDxPipeline];
        [enc1 setBuffer:dy.memory offset:0 atIndex:0];
        [enc1 setBuffer:x.memory offset:0 atIndex:1];
        [enc1 setBuffer:w.memory offset:0 atIndex:2];
        [enc1 setBuffer:dx.memory offset:0 atIndex:3];
        [enc1 setBuffer:invBuf offset:0 atIndex:4];
        [enc1 setBytes:&cols length:sizeof(cols) atIndex:5];
        [enc1 setBytes:&dys length:sizeof(dys) atIndex:6];
        [enc1 setBytes:&xs length:sizeof(xs) atIndex:7];
        [enc1 setBytes:&dxs length:sizeof(dxs) atIndex:8];
        [enc1 setBytes:&eps length:sizeof(eps) atIndex:9];
        [enc1 dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg1, 1, 1)];
        [enc1 endEncoding];

        id<MTLComputeCommandEncoder> enc2 = [command computeCommandEncoder];
        if (!enc2) RETURN(XM_COMMAND);
        NSUInteger tg2 = MIN((NSUInteger)256, d.rmsnormBwdDwPipeline.maxTotalThreadsPerThreadgroup);
        [enc2 setComputePipelineState:d.rmsnormBwdDwPipeline];
        [enc2 setBuffer:dy.memory offset:0 atIndex:0];
        [enc2 setBuffer:x.memory offset:0 atIndex:1];
        [enc2 setBuffer:invBuf offset:0 atIndex:2];
        [enc2 setBuffer:dw.memory offset:0 atIndex:3];
        [enc2 setBytes:&rows length:sizeof(rows) atIndex:4];
        [enc2 setBytes:&cols length:sizeof(cols) atIndex:5];
        [enc2 setBytes:&dys length:sizeof(dys) atIndex:6];
        [enc2 setBytes:&xs length:sizeof(xs) atIndex:7];
        [enc2 dispatchThreadgroups:MTLSizeMake((cols + tg2 - 1) / tg2, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        [enc2 endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    NSArray *ops = @[d.rmsnormBwdDxPipeline, d.rmsnormBwdDwPipeline, invBuf];
    xu_int handle = finish_op2(batch, d, writes, inputs, command, ops);
    if (handle >= 0) { dx.numberClass = MATRIX_CLASS_UNKNOWN; dw.numberClass = MATRIX_CLASS_UNKNOWN; }
    RETURN(handle);
} }

xu_int XM_API(rmsnorm_backward_f32)(xu_int dx, xu_int dw, xu_int dy, xu_int x, xu_int weight, xu_int eps_bits) {
    return rmsnorm_backward_f32_impl(-1, dx, dw, dy, x, weight, eps_bits);
}
xu_int XM_API(batch_rmsnorm_backward_f32)(xu_int batch, xu_int dx, xu_int dw, xu_int dy, xu_int x, xu_int weight, xu_int eps_bits) {
    return rmsnorm_backward_f32_impl(batch < 0 ? 0 : batch, dx, dw, dy, x, weight, eps_bits);
}

static xu_int swiglu_backward_f32_impl(xu_int bh, xu_int dgh, xu_int duh, xu_int dyh,
                                       xu_int gh, xu_int uh) { @autoreleasepool {
    ENTER;
    XMMatrix *dg = lookup(dgh, MATRIX_F32), *du = lookup(duh, MATRIX_F32),
             *dy = lookup(dyh, MATRIX_F32), *g = lookup(gh, MATRIX_F32), *u = lookup(uh, MATRIX_F32);
    if (!dg || !du || !dy || !g || !u) RETURN(XM_HANDLE);
    XMBatch *batch = nil;
    if (bh >= 0) { xu_int berr = 0; batch = lookup_batch_for(bh, dg, &berr); if (!batch) RETURN(berr); }
    if (du.owner != dg.owner || dy.owner != dg.owner || g.owner != dg.owner || u.owner != dg.owner) RETURN(XM_DEVICE);
    if (dg.rows != dy.rows || dg.columns != dy.columns ||
        du.rows != dy.rows || du.columns != dy.columns ||
        g.rows != dy.rows || g.columns != dy.columns ||
        u.rows != dy.rows || u.columns != dy.columns) RETURN(XM_SHAPE);
    if (dg == du || dg == dy || dg == g || dg == u || du == dy || du == g || du == u) RETURN(XM_ALIAS);
    NSArray<XMBuffer *> *inputs = @[dy, g, u];
    NSArray<XMBuffer *> *writes = @[dg, du];
    if (!op_can_access2(batch, dg, du, inputs)) RETURN(XM_BUSY);
    if (dg.poisoned || du.poisoned || dy.poisoned || g.poisoned || u.poisoned) RETURN(XM_COMMAND);
    if (!batch_normal_input(batch, dy) || !batch_normal_input(batch, g) || !batch_normal_input(batch, u)) RETURN(XM_NUMERICS);
    XMDevice *d = dg.owner;
    if (!dg.words) RETURN(finish_op2(batch, d, writes, inputs, nil, nil));
    if (!ensure_ai_pipelines(d)) RETURN(XM_PIPELINE);
    id<MTLCommandBuffer> command = (batch && batch.command) ? batch.command : [d.queue commandBuffer];
    if (!command) RETURN(XM_COMMAND);
    @try {
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        if (!enc) RETURN(XM_COMMAND);
        uint32_t rows = (uint32_t)dg.rows, cols = (uint32_t)dg.columns;
        uint32_t dys = (uint32_t)(dy.rowBytes / 4), gs = (uint32_t)(g.rowBytes / 4), us = (uint32_t)(u.rowBytes / 4);
        uint32_t dgs = (uint32_t)(dg.rowBytes / 4), dus = (uint32_t)(du.rowBytes / 4);
        NSUInteger tx = MIN((NSUInteger)16, d.swigluBwdPipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger ty = MIN((NSUInteger)16, d.swigluBwdPipeline.maxTotalThreadsPerThreadgroup / MAX((NSUInteger)1, tx));
        [enc setComputePipelineState:d.swigluBwdPipeline];
        [enc setBuffer:dy.memory offset:0 atIndex:0];
        [enc setBuffer:g.memory offset:0 atIndex:1];
        [enc setBuffer:u.memory offset:0 atIndex:2];
        [enc setBuffer:dg.memory offset:0 atIndex:3];
        [enc setBuffer:du.memory offset:0 atIndex:4];
        [enc setBytes:&rows length:sizeof(rows) atIndex:5];
        [enc setBytes:&cols length:sizeof(cols) atIndex:6];
        [enc setBytes:&dys length:sizeof(dys) atIndex:7];
        [enc setBytes:&gs length:sizeof(gs) atIndex:8];
        [enc setBytes:&us length:sizeof(us) atIndex:9];
        [enc setBytes:&dgs length:sizeof(dgs) atIndex:10];
        [enc setBytes:&dus length:sizeof(dus) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake((cols + tx - 1) / tx, (rows + ty - 1) / ty, 1)
            threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
        [enc endEncoding];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = finish_op2(batch, d, writes, inputs, command, d.swigluBwdPipeline);
    if (handle >= 0) { dg.numberClass = MATRIX_CLASS_UNKNOWN; du.numberClass = MATRIX_CLASS_UNKNOWN; }
    RETURN(handle);
} }

xu_int XM_API(swiglu_backward_f32)(xu_int dgate, xu_int dup, xu_int dy, xu_int gate, xu_int up) {
    return swiglu_backward_f32_impl(-1, dgate, dup, dy, gate, up);
}
xu_int XM_API(batch_swiglu_backward_f32)(xu_int batch, xu_int dgate, xu_int dup, xu_int dy, xu_int gate, xu_int up) {
    return swiglu_backward_f32_impl(batch < 0 ? 0 : batch, dgate, dup, dy, gate, up);
}
