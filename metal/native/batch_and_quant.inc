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
        d.gemvQ80Pipeline && d.gemvQ4KPipeline) return YES;
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
        "    uint2 tg2 [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh[256];\n"
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
        "    sh[tid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] += sh[tid + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    if (tid == 0) dst[m * ds + n] = sh[0];\n"
        "}\n"
        "kernel void xu_gemv_q8_0_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh[256];\n"
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
        "    sh[tid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] += sh[tid + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    if (tid == 0) dst[m * ds + n] = sh[0];\n"
        "}\n"
        "kernel void xu_gemv_q4_k_f32(device const float *x [[buffer(0)]], device const uchar *w [[buffer(1)]],\n"
        "    device float *dst [[buffer(2)]], constant uint &rows [[buffer(3)]],\n"
        "    constant uint &cols [[buffer(4)]], constant uint &m_cnt [[buffer(5)]],\n"
        "    constant uint &xs [[buffer(6)]], constant uint &ds [[buffer(7)]],\n"
        "    uint2 tgid [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],\n"
        "    uint2 tg2 [[threads_per_threadgroup]]) {\n"
        "    threadgroup float sh[256];\n"
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
        "    sh[tid] = acc;\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    for (uint s = tg >> 1; s > 0; s >>= 1) {\n"
        "        if (tid < s) sh[tid] += sh[tid + s];\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    }\n"
        "    if (tid == 0) dst[m * ds + n] = sh[0];\n"
        "}\n";
    NSError *error = nil;
    id<MTLLibrary> lib = [d.gpu newLibraryWithSource:src options:nil error:&error];
    if (!lib) return NO;
    id<MTLComputePipelineState> p0 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_f16_to_f32"] error:&error];
    id<MTLComputePipelineState> p1 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_f32_to_f16"] error:&error];
    id<MTLComputePipelineState> p2 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q4_0_f32"] error:&error];
    id<MTLComputePipelineState> p3 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q8_0_f32"] error:&error];
    id<MTLComputePipelineState> p4 = [d.gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"xu_gemv_q4_k_f32"] error:&error];
    if (!p0 || !p1 || !p2 || !p3 || !p4) return NO;
    d.f16ToF32Pipeline = p0;
    d.f32ToF16Pipeline = p1;
    d.gemvQ40Pipeline = p2;
    d.gemvQ80Pipeline = p3;
    d.gemvQ4KPipeline = p4;
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
        kernel = matrix_kernel_typed(d, c, a, transposeRight, MPSDataTypeFloat16);
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
                                 NSUInteger block_bytes, BOOL check_second_f16) { @autoreleasepool {
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
        uint16_t d_bits = (uint16_t)src[off] | ((uint16_t)src[off + 1] << 8);
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
xu_int XM_API(matrix_q4_0_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q4_0, 18, NO); }
xu_int XM_API(matrix_q4_0_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_0, NO); } }
xu_int XM_API(matrix_q4_0_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_0, YES); } }
xu_int XM_API(gemv_q4_0_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q4_0); }
xu_int XM_API(batch_gemv_q4_0_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q4_0); }

xu_int XM_API(matrix_q8_0_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 32, 34, MATRIX_Q8_0); }
xu_int XM_API(matrix_q8_0_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q8_0); }
xu_int XM_API(matrix_q8_0_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q8_0, 34, NO); }
xu_int XM_API(matrix_q8_0_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q8_0, NO); } }
xu_int XM_API(matrix_q8_0_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q8_0, YES); } }
xu_int XM_API(gemv_q8_0_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q8_0); }
xu_int XM_API(batch_gemv_q8_0_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q8_0); }

xu_int XM_API(matrix_q4_k_new)(xu_int h, xu_int rows, xu_int columns) { return quant_matrix_new(h, rows, columns, 256, 144, MATRIX_Q4_K); }
xu_int XM_API(matrix_q4_k_info)(xu_int h, xu_int key) { return quant_matrix_info(h, key, MATRIX_Q4_K); }
xu_int XM_API(matrix_q4_k_write)(xu_int h, const xu_u8_span *bytes) { return quant_matrix_write(h, bytes, MATRIX_Q4_K, 144, YES); }
xu_int XM_API(matrix_q4_k_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_K, NO); } }
xu_int XM_API(matrix_q4_k_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, MATRIX_Q4_K, YES); } }
xu_int XM_API(gemv_q4_k_f32)(xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(-1, dst, xh, wh, MATRIX_Q4_K); }
xu_int XM_API(batch_gemv_q4_k_f32)(xu_int batch, xu_int dst, xu_int xh, xu_int wh) { return gemv_quant_f32_impl(batch < 0 ? 0 : batch, dst, xh, wh, MATRIX_Q4_K); }
