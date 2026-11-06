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
static MPSMatrixMultiplication *matrix_kernel(XMDevice *d, XMMatrix *c, XMMatrix *a) {
    NSString *key = [NSString stringWithFormat:@"%llu:%llu:%llu",
        (unsigned long long)c.rows, (unsigned long long)c.columns, (unsigned long long)a.columns];
    MPSMatrixMultiplication *kernel = d.matrixKernels[key];
    if (kernel) return kernel;
    kernel = [[MPSMatrixMultiplication alloc] initWithDevice:d.gpu transposeLeft:NO transposeRight:NO
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

xu_int XM_API(gemm_f32)(xu_int dst, xu_int ah, xu_int bh) { @autoreleasepool {
    ENTER;
    XMMatrix *c = lookup(dst, MATRIX_F32), *a = lookup(ah, MATRIX_F32), *b = lookup(bh, MATRIX_F32);
    if (!c || !a || !b) RETURN(XM_HANDLE);
    if (a.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    if (a.columns != b.rows || c.rows != a.rows || c.columns != b.columns) RETURN(XM_SHAPE);
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
        kernel = matrix_kernel(d, c, a);
        if (!kernel) RETURN(XM_PIPELINE);
        command = [d.queue commandBuffer];
        if (!command) RETURN(XM_COMMAND);
        [kernel encodeToCommandBuffer:command leftMatrix:a.matrix rightMatrix:b.matrix resultMatrix:c.matrix];
    } @catch (NSException *exception) { (void)exception; RETURN(XM_PIPELINE); }
    xu_int handle = submit_command(d, c, a, b, command, kernel);
    if (handle >= 0) c.numberClass = MATRIX_CLASS_UNKNOWN;
    RETURN(handle);
} }
