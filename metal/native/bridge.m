// Optional Metal runtime. Module autolink records framework dependencies in
// the native object, so ordinary Xu builds need no Metal-specific linker path.
@import Foundation;
@import Metal;
@import MetalPerformanceShaders;
#include <os/lock.h>
#include <stdint.h>
#include <string.h>
#include "api.h"

enum { DEVICE = 1, BUFFER, FENCE, SNAPSHOT, MATRIX_F32, SLOT_BITS = 12, SLOT_COUNT = 4096 };
// Rounded dispatches use at most 256 threads and must not wrap a uint gid.
static const uint64_t MAX_WORDS = UINT32_MAX - 255ULL;

@interface XMDevice : NSObject
@property(nonatomic, strong) id<MTLDevice> gpu;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property(nonatomic) xu_int children;
@property(nonatomic) xu_int dispatches;
@property(nonatomic, strong) NSMutableDictionary<NSString *, MPSMatrixMultiplication *> *matrixKernels;
@property(nonatomic, strong) NSMutableArray<NSString *> *matrixKeys;
@property(nonatomic) xu_int matrixKernelBuilds;
@end
@implementation XMDevice
@end

@interface XMBuffer : NSObject
@property(nonatomic, strong) XMDevice *owner;
@property(nonatomic, strong) id<MTLBuffer> memory;
@property(nonatomic) uint64_t words;
@property(nonatomic) xu_int busy;
@property(nonatomic) BOOL poisoned;
@end
@implementation XMBuffer
@end

@interface XMFence : NSObject
@property(nonatomic, strong) XMDevice *owner;
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, strong) NSArray<XMBuffer *> *buffers;
@property(nonatomic, strong) XMBuffer *output;
@property(nonatomic, strong) NSLock *waitGate;
@property(nonatomic) BOOL completed;
@property(nonatomic) xu_int status;
@property(nonatomic, strong) id operation;
@end
@implementation XMFence
@end

typedef struct {
    __strong id object;
    uint64_t generation;
    int kind;
} XMSlot;
static XMSlot slots[SLOT_COUNT];
static unsigned cursor;
static xu_int live;
static os_unfair_lock gate = OS_UNFAIR_LOCK_INIT;

// All registry and resource-state accesses are under gate. A strong local or
// fence retains physical resources after an owning public handle is released.
static id lookup(xu_int handle, int kind) {
    if (handle <= 0) return nil;
    XMSlot *s = &slots[(xu_uint)handle & (SLOT_COUNT - 1)];
    return s->kind == kind && s->generation == ((xu_uint)handle >> SLOT_BITS) ? s->object : nil;
}
static xu_int insert(id object, int kind) {
    if (!object) return XM_ALLOC;
    for (unsigned n = 0; n < SLOT_COUNT; ++n) {
        unsigned index = cursor++ & (SLOT_COUNT - 1);
        XMSlot *s = &slots[index];
        if (s->object || s->generation >= ((uint64_t)INT64_MAX >> SLOT_BITS)) continue;
        s->generation++;
        s->kind = kind;
        s->object = object;
        live++;
        return (xu_int)((s->generation << SLOT_BITS) | index);
    }
    return XM_LIMIT;
}
static void remove_handle(xu_int handle) {
    XMSlot *s = &slots[(xu_uint)handle & (SLOT_COUNT - 1)];
    s->object = nil;
    s->kind = 0;
    live--;
}
#define ENTER os_unfair_lock_lock(&gate)
#define RETURN(value) do { xu_int xm_return = (value); os_unfair_lock_unlock(&gate); return xm_return; } while (0)

static xu_int snapshot(NSData *data) { return insert(data, SNAPSHOT); }
static xu_int finish(XMFence *f) {
    if (!f.completed) {
        f.status = f.command.status == MTLCommandBufferStatusCompleted ? 0 : XM_COMMAND;
        f.completed = YES;
        for (XMBuffer *b in f.buffers) b.busy--;
        if (f.status < 0) f.output.poisoned = YES;
    }
    return f.status;
}

// Caller holds gate. Integer and MPS operations share lifetime, busy accounting,
// acknowledgement and failure handling. A dropped fence retains its operation.
static xu_int submit_command(XMDevice *d, XMBuffer *c, XMBuffer *a, XMBuffer *b,
                             id<MTLCommandBuffer> command, id operation) {
    XMFence *f = [XMFence new];
    if (!f) return XM_ALLOC;
    f.waitGate = [NSLock new];
    if (!f.waitGate) return XM_ALLOC;
    f.owner = d; f.output = c;
    f.buffers = a == b ? @[c,a] : @[c,a,b];
    f.command = command; f.operation = operation;
    f.completed = command == nil;
    xu_int handle = insert(f, FENCE);
    if (handle < 0) return handle;
    d.children++;
    if (command) {
        for (XMBuffer *buffer in f.buffers) buffer.busy++;
        [command commit];
        d.dispatches++;
    }
    return handle;
}

xu_int XM_API(abi_version)(void) { return 2; }
xu_int XM_API(available)(void) { @autoreleasepool { return MTLCreateSystemDefaultDevice() != nil; } }
xu_int XM_API(live_handles)(void) { ENTER; RETURN(live); }

xu_int XM_API(device_new)(void) { @autoreleasepool {
    id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
    if (!gpu) return XM_UNAVAILABLE;
    id<MTLCommandQueue> queue = [gpu newCommandQueue];
    if (!queue) return XM_ALLOC;
    NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void xu_madd(device const ulong *a [[buffer(0)]], device const ulong *b [[buffer(1)]], "
        "device ulong *c [[buffer(2)]], constant uint &n [[buffer(3)]], constant ulong &factor [[buffer(4)]], "
        "uint i [[thread_position_in_grid]]) { if (i < n) c[i] = a[i] * factor + b[i]; }";
    NSError *error = nil;
    id<MTLLibrary> library = [gpu newLibraryWithSource:source options:nil error:&error];
    if (!library) return XM_PIPELINE;
    id<MTLComputePipelineState> pipeline = [gpu newComputePipelineStateWithFunction:[library newFunctionWithName:@"xu_madd"] error:&error];
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup == 0) return XM_PIPELINE;
    XMDevice *device = [XMDevice new];
    device.gpu = gpu; device.queue = queue; device.pipeline = pipeline;
    ENTER; RETURN(insert(device, DEVICE));
} }

static xu_int release_device(xu_int handle, BOOL force) {
    ENTER;
    XMDevice *d = lookup(handle, DEVICE);
    if (!d) RETURN(XM_HANDLE);
    if (!force && d.children) RETURN(XM_BUSY);
    remove_handle(handle);
    RETURN(0);
}
xu_int XM_API(device_close)(xu_int h) { @autoreleasepool { return release_device(h, NO); } }
xu_int XM_API(device_release)(xu_int h) { @autoreleasepool { return release_device(h, YES); } }
xu_int XM_API(device_info)(xu_int h, xu_int key) { @autoreleasepool {
    ENTER; XMDevice *d = lookup(h, DEVICE);
    if (!d) RETURN(XM_HANDLE);
    if (key == 1) RETURN(d.gpu.hasUnifiedMemory);
    if (key == 2) RETURN(d.dispatches);
    if (key == 3) RETURN(1); // One successfully created, retained pipeline per device.
    if (key == 4) RETURN(MIN(MAX_WORDS, d.gpu.maxBufferLength / 8));
    if (key == 5) RETURN(d.matrixKernelBuilds);
    if (key == 6) RETURN((xu_int)d.matrixKernels.count);
    if (key == 7) RETURN(MPSSupportsMTLDevice(d.gpu));
    RETURN(XM_BOUNDS);
} }
xu_int XM_API(device_name)(xu_int h) { @autoreleasepool {
    ENTER; XMDevice *d = lookup(h, DEVICE);
    if (!d) RETURN(XM_HANDLE);
    RETURN(snapshot([d.gpu.name dataUsingEncoding:NSUTF8StringEncoding]));
} }

xu_int XM_API(buffer_new)(xu_int h, xu_int words, xu_int dtype) { @autoreleasepool {
    if (dtype != 1) return XM_DTYPE;
    if (words < 0 || (xu_uint)words > MAX_WORDS) return XM_BOUNDS;
    ENTER; XMDevice *d = lookup(h, DEVICE);
    if (!d) RETURN(XM_HANDLE);
    if ((xu_uint)words > d.gpu.maxBufferLength / 8) RETURN(XM_BOUNDS);
    // Reserve the child while allocation runs without the registry lock.
    d.children++;
    os_unfair_lock_unlock(&gate);
    NSUInteger bytes = MAX((NSUInteger)16, (NSUInteger)words * 8);
    id<MTLBuffer> memory = [d.gpu newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    XMBuffer *buffer = memory ? [XMBuffer new] : nil;
    if (buffer) {
        memset(memory.contents, 0, bytes);
        buffer.owner = d; buffer.memory = memory; buffer.words = (uint64_t)words;
    }
    ENTER;
    xu_int result = buffer ? insert(buffer, BUFFER) : XM_ALLOC;
    if (result < 0) d.children--;
    RETURN(result);
} }
xu_int XM_API(buffer_put)(xu_int h, xu_int index, xu_int value) { @autoreleasepool {
    ENTER; XMBuffer *b = lookup(h, BUFFER);
    if (!b) RETURN(XM_HANDLE);
    if (index < 0 || (xu_uint)index >= b.words) RETURN(XM_BOUNDS);
    if (b.busy) RETURN(XM_BUSY);
    if (b.poisoned) RETURN(XM_COMMAND);
    uint64_t bits = (xu_uint)value;
    memcpy((uint8_t *)b.memory.contents + (uint64_t)index * 8, &bits, 8);
    RETURN(0);
} }

// Validate the complete range before arithmetic or any observable write.
// The typed ABI supplies live caller-owned storage; this runtime never keeps
// its pointer. Even zero-count operations validate handles and busy state.
static BOOL host_range(xu_int length, const void *data, xu_int first, xu_int count) {
    if (length < 0 || first < 0 || count < 0 || first > length || count > length - first) return NO;
    if ((xu_uint)length > SIZE_MAX / 8) return NO;
    if (count && !data) return NO;
    if (count && (xu_uint)(first + count) > (UINTPTR_MAX - (uintptr_t)data) / 8) return NO;
    return YES;
}
xu_int XM_API(buffer_write)(xu_int h, xu_int offset, const xu_i64_span *source, xu_int first, xu_int count) { @autoreleasepool {
    if (!source || !host_range(source->len, source->data, first, count)) return XM_BOUNDS;
    ENTER; XMBuffer *b = lookup(h, BUFFER);
    if (!b) RETURN(XM_HANDLE);
    if (offset < 0 || (xu_uint)offset > b.words || (xu_uint)count > b.words - (xu_uint)offset) RETURN(XM_BOUNDS);
    if (b.busy) RETURN(XM_BUSY);
    if (b.poisoned) RETURN(XM_COMMAND);
    if (count) memmove((uint8_t *)b.memory.contents + (xu_uint)offset * 8,
        (const uint8_t *)source->data + (xu_uint)first * 8, (size_t)count * 8);
    RETURN(0);
} }
xu_int XM_API(buffer_snapshot)(xu_int h) { @autoreleasepool {
    ENTER; XMBuffer *b = lookup(h, BUFFER);
    if (!b) RETURN(XM_HANDLE);
    if (b.busy) RETURN(XM_BUSY);
    if (b.poisoned) RETURN(XM_COMMAND);
    RETURN(snapshot([NSData dataWithBytes:b.memory.contents length:(NSUInteger)b.words * 8]));
} }
static xu_int release_buffer_kind(xu_int handle, int kind, BOOL force) {
    ENTER; XMBuffer *b = lookup(handle, kind);
    if (!b) RETURN(XM_HANDLE);
    if (!force && b.busy) RETURN(XM_BUSY);
    b.owner.children--;
    remove_handle(handle);
    RETURN(0);
}
xu_int XM_API(buffer_close)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, BUFFER, NO); } }
xu_int XM_API(buffer_release)(xu_int h) { @autoreleasepool { return release_buffer_kind(h, BUFFER, YES); } }

xu_int XM_API(snapshot_len)(xu_int h) { @autoreleasepool {
    ENTER; NSData *data = lookup(h, SNAPSHOT);
    if (!data) RETURN(XM_HANDLE);
    RETURN((xu_int)data.length);
} }
xu_int XM_API(snapshot_byte)(xu_int h, xu_int index) { @autoreleasepool {
    ENTER; NSData *data = lookup(h, SNAPSHOT);
    if (!data) RETURN(XM_HANDLE);
    if (index < 0 || (xu_uint)index >= data.length) RETURN(XM_BOUNDS);
    RETURN(((const uint8_t *)data.bytes)[index]);
} }
static xu_int snapshot_word(xu_int h, xu_int index, BOOL high) {
    ENTER; NSData *data = lookup(h, SNAPSHOT);
    if (!data) RETURN(XM_HANDLE);
    if (index < 0 || (xu_uint)index >= data.length / 8) RETURN(XM_BOUNDS);
    uint64_t bits;
    memcpy(&bits, (const uint8_t *)data.bytes + (uint64_t)index * 8, 8);
    RETURN((xu_int)(high ? bits >> 32 : bits & UINT32_MAX));
}
xu_int XM_API(snapshot_low)(xu_int h, xu_int i) { @autoreleasepool { return snapshot_word(h, i, NO); } }
xu_int XM_API(snapshot_high)(xu_int h, xu_int i) { @autoreleasepool { return snapshot_word(h, i, YES); } }
xu_int XM_API(snapshot_copy)(xu_int h, xu_int first, const xu_i64_span_mut *destination, xu_int offset, xu_int count) { @autoreleasepool {
    if (!destination || !host_range(destination->len, destination->data, offset, count)) return XM_BOUNDS;
    ENTER; NSData *data = lookup(h, SNAPSHOT);
    if (!data) RETURN(XM_HANDLE);
    if ((data.length & 7) || first < 0 || (xu_uint)first > data.length / 8 || (xu_uint)count > data.length / 8 - (xu_uint)first) RETURN(XM_BOUNDS);
    if (count) memmove((uint8_t *)destination->data + (xu_uint)offset * 8,
        (const uint8_t *)data.bytes + (xu_uint)first * 8, (size_t)count * 8);
    RETURN(0);
} }
xu_int XM_API(snapshot_close)(xu_int h) { @autoreleasepool {
    ENTER;
    if (!lookup(h, SNAPSHOT)) RETURN(XM_HANDLE);
    remove_handle(h); RETURN(0);
} }

xu_int XM_API(madd_i64)(xu_int dst, xu_int ah, xu_int bh, xu_int factor, xu_int words) { @autoreleasepool {
    if (words < 0 || (xu_uint)words > MAX_WORDS) return XM_BOUNDS;
    ENTER;
    XMBuffer *c = lookup(dst, BUFFER), *a = lookup(ah, BUFFER), *b = lookup(bh, BUFFER);
    if (!c || !a || !b) RETURN(XM_HANDLE);
    if (a.owner != c.owner || b.owner != c.owner) RETURN(XM_DEVICE);
    if ((xu_uint)words > c.words || (xu_uint)words > a.words || (xu_uint)words > b.words) RETURN(XM_BOUNDS);
    if (c == a || c == b) RETURN(XM_ALIAS);
    if (c.busy || a.busy || b.busy) RETURN(XM_BUSY);
    if (c.poisoned || a.poisoned || b.poisoned) RETURN(XM_COMMAND);
    XMDevice *d = c.owner;
    id<MTLCommandBuffer> command = nil;
    if (words) {
        command = [d.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) RETURN(XM_COMMAND);
        [encoder setComputePipelineState:d.pipeline];
        [encoder setBuffer:a.memory offset:0 atIndex:0];
        [encoder setBuffer:b.memory offset:0 atIndex:1];
        [encoder setBuffer:c.memory offset:0 atIndex:2];
        uint32_t count = (uint32_t)words;
        uint64_t scale = (xu_uint)factor;
        [encoder setBytes:&count length:4 atIndex:3];
        [encoder setBytes:&scale length:8 atIndex:4];
        NSUInteger width = MIN((NSUInteger)256, d.pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:MTLSizeMake(((NSUInteger)words + width - 1) / width,1,1)
            threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        [encoder endEncoding];
    }
    RETURN(submit_command(d, c, a, b, command, nil));
} }

static xu_int wait_fence(xu_int handle, BOOL dispose) { @autoreleasepool {
    ENTER; XMFence *f = lookup(handle, FENCE);
    if (!f) RETURN(XM_HANDLE);
    os_unfair_lock_unlock(&gate);
    // Do not concurrently enter one Metal command buffer's blocking wait.
    // The global registry remains available to unrelated devices and jobs.
    [f.waitGate lock];
    ENTER;
    id<MTLCommandBuffer> command = f.command;
    BOOL completed = f.completed;
    os_unfair_lock_unlock(&gate);
    if (!completed) [command waitUntilCompleted];
    ENTER;
    xu_int result = f.completed ? f.status : finish(f);
    // Another waiter may already have closed this handle and reused the slot.
    if (dispose && lookup(handle, FENCE) == f) {
        f.owner.children--;
        remove_handle(handle);
    }
    os_unfair_lock_unlock(&gate);
    [f.waitGate unlock];
    return result;
} }
xu_int XM_API(fence_wait)(xu_int h) { return wait_fence(h, NO); }
xu_int XM_API(fence_dispose)(xu_int h) { return wait_fence(h, YES); }
xu_int XM_API(fence_close)(xu_int h) { @autoreleasepool {
    ENTER; XMFence *f = lookup(h, FENCE);
    if (!f) RETURN(XM_HANDLE);
    if (!f.completed) RETURN(XM_BUSY);
    f.owner.children--;
    remove_handle(h); RETURN(0);
} }

#include "matrix_f32.inc"
