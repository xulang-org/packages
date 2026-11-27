import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("XU_ROOT", str(ROOT.parent / "xu"))).resolve()
XU = Path(os.environ.get("XU", str(REPO / "bin" / "xu"))).resolve()


def xu_env(work: Path, extra=None) -> dict:
    env = dict(
        os.environ,
        XU_ROOT=str(REPO),
        XU_STDLIB_ROOT=str(REPO / "src"),
        XU_PACKAGES_ROOT=str(ROOT),
        XU_CACHE=str(work / "cache"),
        XU_TASK_POLL="0",
    )
    if extra:
        env.update(extra)
    return env


def build_xu(work: Path, name: str, source: str) -> Path:
    proj = work / name
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "xu.xun").write_text(
        f'module: {name}\nversion: !ver 0.1.0\nxu: !ver 0.12.6\nrequire:\n  "github.com/xulang-org/packages": file:{ROOT}\n'
    )
    (proj / "main.xu").write_text(source)
    exe = work / f"{name}_bin"
    res = subprocess.run(
        [str(XU), "build", str(proj / "main.xu"), "-o", str(exe)],
        cwd=proj,
        env=xu_env(work),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(res.stdout + res.stderr)
    return exe


COMMON_XU_TEST_HELPERS = r'''
fn make_f32_bits(n int, seed int) []int {
    $out := []int{}
    out.reserve(n)
    for i in 0:n {
        v := f64(((i * 11 + seed * 7) % 23) - 11) * 0.125 + 0.0625
        out.push(math.float32_bits(v))
    }
    << out
}

fn push_u16_le(out $&[]int, v int) {
    out.push(v & 255)
    out.push((v >> 8) & 255)
}

fn check(cond bool, msg string) {
    if !cond {
        eprintln("FAIL: " + msg)
        <<< 1
    }
}

fn bytes_str(bs []int) string {
    << string_from_bytes(bs)
}

fn make_q4_0_bytes(rows int, cols int) string {
    nb := cols / 32
    $bs := []int{}
    bs.reserve(rows * nb * 18)
    for r in 0:rows {
        for b in 0:nb {
            push_u16_le($&bs, 0x3400) // 0.25 in f16
            for j in 0:16 {
                q0 := (r * 3 + b + j) % 15
                q1 := (r * 5 + b * 2 + j + 4) % 15
                bs.push((q1 << 4) | q0)
            }
        }
    }
    << bytes_str(bs)
}

fn make_q8_0_bytes(rows int, cols int) string {
    nb := cols / 32
    $bs := []int{}
    bs.reserve(rows * nb * 34)
    for r in 0:rows {
        for b in 0:nb {
            push_u16_le($&bs, 0x3000) // 0.125 in f16
            for j in 0:32 {
                v := ((r * 7 + b * 3 + j) % 25) - 12
                bs.push(v & 255)
            }
        }
    }
    << bytes_str(bs)
}

fn make_q4_k_bytes(rows int, cols int) string {
    nb := cols / 256
    $bs := []int{}
    bs.reserve(rows * nb * 144)
    for r in 0:rows {
        for b in 0:nb {
            push_u16_le($&bs, 0x3000) // d=0.125
            push_u16_le($&bs, 0x2C00) // dmin=0.0625
            for s in 0:12 {
                bs.push(0x40 | ((r + b + s) % 31) + 1)
            }
            for q in 0:128 {
                bs.push(((r * 13 + b * 7 + q * 5) & 255))
            }
        }
    }
    << bytes_str(bs)
}

fn make_q5_k_bytes(rows int, cols int) string {
    nb := cols / 256
    $bs := []int{}
    bs.reserve(rows * nb * 176)
    for r in 0:rows {
        for b in 0:nb {
            push_u16_le($&bs, 0x3000)
            push_u16_le($&bs, 0x2C00)
            for s in 0:12 {
                bs.push(0x80 | ((r * 2 + b + s) % 29) + 1)
            }
            for h in 0:32 {
                bs.push((h * 19 + r + b) & 255)
            }
            for q in 0:128 {
                bs.push(((r * 11 + b * 5 + q * 3) & 255))
            }
        }
    }
    << bytes_str(bs)
}

fn make_q6_k_bytes(rows int, cols int) string {
    nb := cols / 256
    $bs := []int{}
    bs.reserve(rows * nb * 210)
    for r in 0:rows {
        for b in 0:nb {
            for q in 0:128 { bs.push((r * 7 + b + q * 3) & 255) }
            for h in 0:64 { bs.push((r * 5 + b + h * 11) & 255) }
            for s in 0:16 {
                sv := ((s + r + b) % 9) - 4
                bs.push(sv & 255)
            }
            push_u16_le($&bs, 0x2C00) // d=0.0625
        }
    }
    << bytes_str(bs)
}

fn check_close(label string, got &[]int, ref &[]int, tol f64) {
    if got.len != ref.len {
        eprintln("FAIL " + label + " len mismatch")
        <<< 1
    }
    $mx := 0.0
    for i in 0:got.len {
        a := math.float32_from_bits(got[i])
        b := math.float32_from_bits(ref[i])
        d := math.abs(a - b)
        if d > mx { mx = d }
    }
    if mx > tol {
        eprintln("FAIL " + label + " max_diff_e-6=" + str(int(mx * 1000000.0)))
        <<< 1
    }
}

fn run_15_ops(backend_name string) [][]int {
    $dev := gpu.open_backend(backend_name) >>>
    $res := [][]int{}

    // 1. gemm_f32 (3x5 @ 5x4 -> 3x4)
    $a1 := dev.matrix_f32(3, 5) >>>
    $b1 := dev.matrix_f32(5, 4) >>>
    $c1 := dev.matrix_f32(3, 4) >>>
    _ := a1.write_bits(&make_f32_bits(15, 1)) >>>
    _ := b1.write_bits(&make_f32_bits(20, 2)) >>>
    _ := gpu.gemm_f32($&dev, $&c1, &a1, &b1) >>>
    res.push(c1.bits() >>>)

    // 2. gemm_f32_nt (3x5 @ (4x5)^T -> 3x4)
    $b2 := dev.matrix_f32(4, 5) >>>
    $c2 := dev.matrix_f32(3, 4) >>>
    _ := b2.write_bits(&make_f32_bits(20, 3)) >>>
    _ := gpu.gemm_f32_nt($&dev, $&c2, &a1, &b2) >>>
    res.push(c2.bits() >>>)

    // 3. gemm_f32_tn ((5x3)^T @ 5x4 -> 3x4)
    $a3 := dev.matrix_f32(5, 3) >>>
    $c3 := dev.matrix_f32(3, 4) >>>
    _ := a3.write_bits(&make_f32_bits(15, 4)) >>>
    _ := gpu.gemm_f32_tn($&dev, $&c3, &a3, &b1) >>>
    res.push(c3.bits() >>>)

    // 4-6. add, mul, scale (2x8)
    $ea := dev.matrix_f32(2, 8) >>>
    $eb := dev.matrix_f32(2, 8) >>>
    $ec := dev.matrix_f32(2, 8) >>>
    _ := ea.write_bits(&make_f32_bits(16, 5)) >>>
    _ := eb.write_bits(&make_f32_bits(16, 6)) >>>
    _ := gpu.add_f32($&dev, $&ec, &ea, &eb) >>>
    res.push(ec.bits() >>>)
    _ := gpu.mul_f32($&dev, $&ec, &ea, &eb) >>>
    res.push(ec.bits() >>>)
    _ := gpu.scale_f32($&dev, $&ec, &ea, 0.75) >>>
    res.push(ec.bits() >>>)

    // 7. rmsnorm (2x8)
    $rw := dev.matrix_f32(1, 8) >>>
    _ := rw.write_bits(&make_f32_bits(8, 7)) >>>
    _ := gpu.rmsnorm_f32($&dev, $&ec, &ea, &rw, 1.0e-5) >>>
    res.push(ec.bits() >>>)

    // 8. softmax (2x8)
    _ := gpu.softmax_f32($&dev, $&ec, &ea) >>>
    res.push(ec.bits() >>>)

    // 9. swiglu (2x8)
    _ := gpu.swiglu_f32($&dev, $&ec, &ea, &eb) >>>
    res.push(ec.bits() >>>)

    // 10. rope (seq=2, n_heads=2, head_dim=8 => cols=16)
    $rx := dev.matrix_f32(2, 16) >>>
    $rc := dev.matrix_f32(2, 16) >>>
    _ := rx.write_bits(&make_f32_bits(32, 8)) >>>
    _ := gpu.rope_f32($&dev, $&rc, &rx, 3, 8, 10000.0) >>>
    res.push(rc.bits() >>>)

    // 11-12. q4_0, q8_0 (rows=4, cols=64, M=2)
    $qx64 := dev.matrix_f32(2, 64) >>>
    $qd64 := dev.matrix_f32(2, 4) >>>
    _ := qx64.write_bits(&make_f32_bits(128, 9)) >>>
    $wq4 := dev.matrix_q4_0(4, 64) >>>
    _ := wq4.write_bytes(&make_q4_0_bytes(4, 64)) >>>
    _ := gpu.gemv_q4_0_f32($&dev, $&qd64, &qx64, &wq4) >>>
    res.push(qd64.bits() >>>)

    $wq8 := dev.matrix_q8_0(4, 64) >>>
    _ := wq8.write_bytes(&make_q8_0_bytes(4, 64)) >>>
    _ := gpu.gemv_q8_0_f32($&dev, $&qd64, &qx64, &wq8) >>>
    res.push(qd64.bits() >>>)

    // 13-15. q4_k, q5_k, q6_k (rows=3, cols=256, M=2)
    $qx256 := dev.matrix_f32(2, 256) >>>
    $qd256 := dev.matrix_f32(2, 3) >>>
    _ := qx256.write_bits(&make_f32_bits(512, 10)) >>>
    $wq4k := dev.matrix_q4_k(3, 256) >>>
    _ := wq4k.write_bytes(&make_q4_k_bytes(3, 256)) >>>
    _ := gpu.gemv_q4_k_f32($&dev, $&qd256, &qx256, &wq4k) >>>
    res.push(qd256.bits() >>>)

    $wq5k := dev.matrix_q5_k(3, 256) >>>
    _ := wq5k.write_bytes(&make_q5_k_bytes(3, 256)) >>>
    _ := gpu.gemv_q5_k_f32($&dev, $&qd256, &qx256, &wq5k) >>>
    res.push(qd256.bits() >>>)

    $wq6k := dev.matrix_q6_k(3, 256) >>>
    _ := wq6k.write_bytes(&make_q6_k_bytes(3, 256)) >>>
    _ := gpu.gemv_q6_k_f32($&dev, $&qd256, &qx256, &wq6k) >>>
    res.push(qd256.bits() >>>)

    _ := wq6k.close()
    _ := wq5k.close()
    _ := wq4k.close()
    _ := qd256.close()
    _ := qx256.close()
    _ := wq8.close()
    _ := wq4.close()
    _ := qd64.close()
    _ := qx64.close()
    _ := rc.close()
    _ := rx.close()
    _ := rw.close()
    _ := ec.close()
    _ := eb.close()
    _ := ea.close()
    _ := c3.close()
    _ := a3.close()
    _ := c2.close()
    _ := b2.close()
    _ := c1.close()
    _ := b1.close()
    _ := a1.close()
    _ := dev.close()
    << res
}
'''


CUDA_MOCK_C = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float half_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0u) {
        if (mant == 0u) bits = sign;
        else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x0400u) == 0u) { m <<= 1; e -= 1; }
            m &= 0x03FFu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, 4);
    return out;
}

typedef struct { int id; } MockCtx;
typedef struct { int id; } MockStream;
typedef struct { int id; } MockCublas;
typedef struct { int valid; } MockModule;
typedef struct { int op_id; } MockFunc;
typedef struct { char *src; char *ptx; } MockProg;

int cuInit(unsigned int flags) { (void)flags; return 0; }
int cuDeviceGetCount(int *out) { *out = 1; return 0; }
int cuDeviceGet(int *out, int ordinal) { if (ordinal != 0) return 101; *out = 0; return 0; }
int cuDeviceGetName(char *name, int len, int dev) { (void)dev; snprintf(name, (size_t)len, "Xu Mock NVIDIA RTX 4090"); return 0; }
int cuDeviceTotalMem_v2(size_t *bytes, int dev) { (void)dev; *bytes = (size_t)24ULL * 1024ULL * 1024ULL * 1024ULL; return 0; }
int cuDeviceGetAttribute(int *pi, int attr, int dev) {
    (void)dev;
    if (attr == 16) *pi = 128;
    else if (attr == 75) *pi = 8;
    else if (attr == 76) *pi = 9;
    else *pi = 1024;
    return 0;
}
int cuCtxCreate_v2(void **pctx, unsigned int flags, int dev) { (void)flags; (void)dev; MockCtx *c = malloc(sizeof(MockCtx)); c->id = 1; *pctx = c; return 0; }
int cuCtxDestroy_v2(void *ctx) { free(ctx); return 0; }
int cuCtxSetCurrent(void *ctx) { (void)ctx; return 0; }
int cuCtxSynchronize(void) { return 0; }
int cuMemAlloc_v2(uint64_t *dptr, size_t sz) { void *p = calloc(1, sz ? sz : 16); *dptr = (uint64_t)(uintptr_t)p; return p ? 0 : 2; }
int cuMemFree_v2(uint64_t dptr) { free((void*)(uintptr_t)dptr); return 0; }
int cuMemcpyHtoD_v2(uint64_t dst, const void *src, size_t sz) { memcpy((void*)(uintptr_t)dst, src, sz); return 0; }
int cuMemcpyDtoH_v2(void *dst, uint64_t src, size_t sz) { memcpy(dst, (const void*)(uintptr_t)src, sz); return 0; }
int cuMemcpyDtoD_v2(uint64_t dst, uint64_t src, size_t sz) { memcpy((void*)(uintptr_t)dst, (const void*)(uintptr_t)src, sz); return 0; }
int cuMemsetD8_v2(uint64_t dst, unsigned char v, size_t n) { memset((void*)(uintptr_t)dst, v, n); return 0; }
int cuStreamCreate(void **pst, unsigned int flags) { (void)flags; MockStream *s = malloc(sizeof(MockStream)); s->id = 1; *pst = s; return 0; }
int cuStreamDestroy_v2(void *st) { free(st); return 0; }
int cuStreamSynchronize(void *st) { (void)st; return 0; }

int nvrtcCreateProgram(void **prog, const char *src, const char *name, int nh, const char **hdrs, const char **inames) {
    (void)name; (void)nh; (void)hdrs; (void)inames;
    MockProg *p = calloc(1, sizeof(MockProg));
    p->src = strdup(src ? src : "");
    *prog = p;
    return 0;
}
int nvrtcCompileProgram(void *prog, int nopts, const char **opts) {
    (void)nopts; (void)opts;
    MockProg *p = (MockProg*)prog;
    if (!strstr(p->src, "xu_gemv_q4_k_f32") || !strstr(p->src, "xu_rope_f32")) return 6;
    p->ptx = strdup("XU_MOCK_PTX_OK");
    return 0;
}
int nvrtcGetPTXSize(void *prog, size_t *sz) { MockProg *p = (MockProg*)prog; *sz = strlen(p->ptx) + 1; return 0; }
int nvrtcGetPTX(void *prog, char *ptx) { MockProg *p = (MockProg*)prog; strcpy(ptx, p->ptx); return 0; }
int nvrtcGetProgramLogSize(void *prog, size_t *sz) { (void)prog; *sz = 1; return 0; }
int nvrtcGetProgramLog(void *prog, char *log) { (void)prog; log[0] = 0; return 0; }
int nvrtcDestroyProgram(void **pprog) { MockProg *p = (MockProg*)*pprog; free(p->src); free(p->ptx); free(p); *pprog = NULL; return 0; }

int cuModuleLoadData(void **pmod, const void *img) {
    if (!strstr((const char*)img, "XU_MOCK_PTX_OK")) return 209;
    MockModule *m = malloc(sizeof(MockModule)); m->valid = 1; *pmod = m; return 0;
}
int cuModuleUnload(void *mod) { free(mod); return 0; }
int cuModuleGetFunction(void **pfn, void *mod, const char *name) {
    (void)mod;
    static const char *names[] = {
        "", "xu_add_f32", "xu_mul_f32", "xu_scale_f32", "xu_rmsnorm_f32",
        "xu_softmax_f32", "xu_swiglu_f32", "xu_rope_f32", "xu_gemm_f32",
        "xu_gemm_f32_nt", "xu_gemm_f32_tn", "xu_gemv_q4_0_f32", "xu_gemv_q8_0_f32",
        "xu_gemv_q4_k_f32", "xu_gemv_q5_k_f32", "xu_gemv_q6_k_f32"
    };
    for (int i = 1; i <= 15; ++i) {
        if (strcmp(name, names[i]) == 0) {
            MockFunc *f = malloc(sizeof(MockFunc)); f->op_id = i; *pfn = f; return 0;
        }
    }
    return 500;
}

int cuLaunchKernel(void *fn, unsigned int gx, unsigned int gy, unsigned int gz,
                   unsigned int bx, unsigned int by, unsigned int bz,
                   unsigned int smem, void *stream, void **kp, void **extra) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)smem; (void)stream; (void)extra;
    int op = ((MockFunc*)fn)->op_id;
    if (op == 1) { // add
        const float *a = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const float *b = (const float*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int n = *(int*)kp[3];
        for (int i = 0; i < n; ++i) dst[i] = a[i] + b[i];
    } else if (op == 2) { // mul
        const float *a = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const float *b = (const float*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int n = *(int*)kp[3];
        for (int i = 0; i < n; ++i) dst[i] = a[i] * b[i];
    } else if (op == 3) { // scale
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[1]);
        int n = *(int*)kp[2];
        float alpha = *(float*)kp[3];
        for (int i = 0; i < n; ++i) dst[i] = x[i] * alpha;
    } else if (op == 4) { // rmsnorm
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const float *w = (const float*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int rows = *(int*)kp[3], cols = *(int*)kp[4];
        float eps = *(float*)kp[5];
        for (int r = 0; r < rows; ++r) {
            float ss = 0.0f;
            for (int c = 0; c < cols; ++c) ss += x[r * cols + c] * x[r * cols + c];
            float inv = 1.0f / sqrtf(ss / (float)cols + eps);
            for (int c = 0; c < cols; ++c) dst[r * cols + c] = x[r * cols + c] * inv * w[c];
        }
    } else if (op == 5) { // softmax
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[1]);
        int rows = *(int*)kp[2], cols = *(int*)kp[3];
        for (int r = 0; r < rows; ++r) {
            float mx = x[r * cols];
            for (int c = 1; c < cols; ++c) if (x[r * cols + c] > mx) mx = x[r * cols + c];
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) { float e = expf(x[r * cols + c] - mx); dst[r * cols + c] = e; sum += e; }
            for (int c = 0; c < cols; ++c) dst[r * cols + c] /= sum;
        }
    } else if (op == 6) { // swiglu
        const float *g = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const float *u = (const float*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int n = *(int*)kp[3];
        for (int i = 0; i < n; ++i) dst[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
    } else if (op == 7) { // rope
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[1]);
        int seq_len = *(int*)kp[2], n_heads = *(int*)kp[3], head_dim = *(int*)kp[4], pos_off = *(int*)kp[5];
        float log_theta = *(float*)kp[6];
        int half_d = head_dim >> 1;
        for (int r = 0; r < seq_len; ++r) for (int h = 0; h < n_heads; ++h) {
            int base = (r * n_heads + h) * head_dim;
            for (int j = 0; j < half_d; ++j) {
                float freq = expf(-log_theta * ((float)(2 * j) / (float)head_dim));
                float angle = (float)(pos_off + r) * freq;
                float c = cosf(angle), s = sinf(angle);
                float v0 = x[base + j], v1 = x[base + half_d + j];
                dst[base + j] = v0 * c - v1 * s;
                dst[base + half_d + j] = v0 * s + v1 * c;
            }
        }
    } else if (op == 11) { // q4_0
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const unsigned char *w = (const unsigned char*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int rows = *(int*)kp[3], cols = *(int*)kp[4], m_cnt = *(int*)kp[5];
        int nb = cols >> 5;
        for (int m = 0; m < m_cnt; ++m) for (int n = 0; n < rows; ++n) {
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const unsigned char *blk = w + (n * nb + b) * 18;
                float d = half_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                const float *xp = x + m * cols + (b << 5);
                float bsum = 0.0f;
                for (int j = 0; j < 16; ++j) {
                    unsigned char bv = blk[2 + j];
                    bsum += (float)((int)(bv & 15u) - 8) * xp[j] + (float)((int)(bv >> 4) - 8) * xp[16 + j];
                }
                acc += bsum * d;
            }
            dst[m * rows + n] = acc;
        }
    } else if (op == 12) { // q8_0
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const unsigned char *w = (const unsigned char*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int rows = *(int*)kp[3], cols = *(int*)kp[4], m_cnt = *(int*)kp[5];
        int nb = cols >> 5;
        for (int m = 0; m < m_cnt; ++m) for (int n = 0; n < rows; ++n) {
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const unsigned char *blk = w + (n * nb + b) * 34;
                float d = half_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                const float *xp = x + m * cols + (b << 5);
                float bsum = 0.0f;
                for (int j = 0; j < 32; ++j) bsum += (float)((int8_t)blk[2 + j]) * xp[j];
                acc += bsum * d;
            }
            dst[m * rows + n] = acc;
        }
    } else if (op == 13) { // q4_k
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const unsigned char *w = (const unsigned char*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int rows = *(int*)kp[3], cols = *(int*)kp[4], m_cnt = *(int*)kp[5];
        int nb = cols >> 8;
        for (int m = 0; m < m_cnt; ++m) for (int n = 0; n < rows; ++n) {
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const unsigned char *blk = w + (n * nb + b) * 144;
                float d = half_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                float dmin = half_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
                const unsigned char *sc = blk + 4, *qs = blk + 16;
                const float *xpb = x + m * cols + (b << 8);
                for (int c = 0; c < 4; ++c) {
                    int is0 = c << 1, is1 = is0 + 1;
                    uint8_t sc1, m1, sc2, m2;
                    if (c < 2) { sc1 = sc[is0] & 63u; m1 = sc[is0+4] & 63u; sc2 = sc[is1] & 63u; m2 = sc[is1+4] & 63u; }
                    else {
                        sc1 = (sc[is0+4] & 15u) | ((sc[is0-4] >> 6) << 4);
                        m1 = (sc[is0+4] >> 4) | ((sc[is0] >> 6) << 4);
                        sc2 = (sc[is1+4] & 15u) | ((sc[is1-4] >> 6) << 4);
                        m2 = (sc[is1+4] >> 4) | ((sc[is1] >> 6) << 4);
                    }
                    float d1 = d * sc1, min1 = dmin * m1, d2 = d * sc2, min2 = dmin * m2;
                    const float *xp = xpb + (c << 6);
                    for (int l = 0; l < 32; ++l) {
                        uint8_t qv = qs[(c << 5) + l];
                        acc += (d1 * (qv & 15u) - min1) * xp[l] + (d2 * (qv >> 4) - min2) * xp[32 + l];
                    }
                }
            }
            dst[m * rows + n] = acc;
        }
    } else if (op == 14) { // q5_k
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const unsigned char *w = (const unsigned char*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int rows = *(int*)kp[3], cols = *(int*)kp[4], m_cnt = *(int*)kp[5];
        int nb = cols >> 8;
        for (int m = 0; m < m_cnt; ++m) for (int n = 0; n < rows; ++n) {
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const unsigned char *blk = w + (n * nb + b) * 176;
                float d = half_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
                float dmin = half_to_f32((uint16_t)blk[2] | ((uint16_t)blk[3] << 8));
                const unsigned char *sc = blk + 4, *qh = blk + 16, *qs = blk + 48;
                const float *xpb = x + m * cols + (b << 8);
                for (int c = 0; c < 4; ++c) {
                    int is0 = c << 1, is1 = is0 + 1;
                    uint8_t sc1, m1, sc2, m2;
                    if (c < 2) { sc1 = sc[is0] & 63u; m1 = sc[is0+4] & 63u; sc2 = sc[is1] & 63u; m2 = sc[is1+4] & 63u; }
                    else {
                        sc1 = (sc[is0+4] & 15u) | ((sc[is0-4] >> 6) << 4);
                        m1 = (sc[is0+4] >> 4) | ((sc[is0] >> 6) << 4);
                        sc2 = (sc[is1+4] & 15u) | ((sc[is1-4] >> 6) << 4);
                        m2 = (sc[is1+4] >> 4) | ((sc[is1] >> 6) << 4);
                    }
                    float d1 = d * sc1, min1 = dmin * m1, d2 = d * sc2, min2 = dmin * m2;
                    unsigned int hm0 = 1u << is0, hm1 = 1u << is1;
                    const float *xp = xpb + (c << 6);
                    for (int l = 0; l < 32; ++l) {
                        uint8_t qhb = qh[l], qv = qs[(c << 5) + l];
                        uint32_t q0 = (qv & 15u) | ((qhb & hm0) ? 16u : 0u);
                        uint32_t q1 = (qv >> 4) | ((qhb & hm1) ? 16u : 0u);
                        acc += (d1 * q0 - min1) * xp[l] + (d2 * q1 - min2) * xp[32 + l];
                    }
                }
            }
            dst[m * rows + n] = acc;
        }
    } else if (op == 15) { // q6_k
        const float *x = (const float*)(uintptr_t)(*(uint64_t*)kp[0]);
        const unsigned char *w = (const unsigned char*)(uintptr_t)(*(uint64_t*)kp[1]);
        float *dst = (float*)(uintptr_t)(*(uint64_t*)kp[2]);
        int rows = *(int*)kp[3], cols = *(int*)kp[4], m_cnt = *(int*)kp[5];
        int nb = cols >> 8;
        for (int m = 0; m < m_cnt; ++m) for (int n = 0; n < rows; ++n) {
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const unsigned char *blk = w + (n * nb + b) * 210;
                float d = half_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
                const unsigned char *ql = blk, *qh = blk + 128, *sc = blk + 192;
                const float *xpb = x + m * cols + (b << 8);
                float bsum = 0.0f;
                for (int h = 0; h < 2; ++h) {
                    const unsigned char *qlp = ql + (h << 6), *qhp = qh + (h << 5), *scp = sc + (h << 3);
                    const float *yp = xpb + (h << 7);
                    for (int l = 0; l < 32; ++l) {
                        int is_idx = l >> 4;
                        uint8_t ql0 = qlp[l], ql1 = qlp[32+l], qhv = qhp[l];
                        int q1 = (int)((ql0 & 15u) | ((qhv & 3u) << 4)) - 32;
                        int q2 = (int)((ql1 & 15u) | (((qhv >> 2) & 3u) << 4)) - 32;
                        int q3 = (int)(((ql0 >> 4) & 15u) | (((qhv >> 4) & 3u) << 4)) - 32;
                        int q4 = (int)(((ql1 >> 4) & 15u) | (((qhv >> 6) & 3u) << 4)) - 32;
                        bsum += (float)((int8_t)scp[is_idx] * q1) * yp[l]
                              + (float)((int8_t)scp[is_idx + 2] * q2) * yp[l + 32]
                              + (float)((int8_t)scp[is_idx + 4] * q3) * yp[l + 64]
                              + (float)((int8_t)scp[is_idx + 6] * q4) * yp[l + 96];
                    }
                }
                acc += d * bsum;
            }
            dst[m * rows + n] = acc;
        }
    }
    return 0;
}

int cublasCreate_v2(void **ph) { MockCublas *h = malloc(sizeof(MockCublas)); h->id = 1; *ph = h; return 0; }
int cublasDestroy_v2(void *h) { free(h); return 0; }
int cublasSetStream_v2(void *h, void *st) { (void)h; (void)st; return 0; }
// True column-major BLAS sgemm verifying our 14-arg FFI stack passing & row/col-major mapping!
int cublasSgemm_v2(void *handle, int transa, int transb, int m, int n, int k,
                   const float *alpha, uint64_t A_u, int lda,
                   uint64_t B_u, int ldb, const float *beta, uint64_t C_u, int ldc) {
    (void)handle;
    const float *A = (const float*)(uintptr_t)A_u;
    const float *B = (const float*)(uintptr_t)B_u;
    float *C = (float*)(uintptr_t)C_u;
    float a = *alpha, be = *beta;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) {
                float va = (transa == 0) ? A[p * lda + row] : A[row * lda + p];
                float vb = (transb == 0) ? B[col * ldb + p] : B[p * ldb + col];
                acc += va * vb;
            }
            float prev = C[col * ldc + row];
            C[col * ldc + row] = a * acc + (be == 0.0f ? 0.0f : be * prev);
        }
    }
    return 0;
}
'''


VULKAN_MOCK_C = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t sType; void *pNext; } VkBaseIn;
typedef struct { uint32_t sType; void *pNext; uint32_t flags; size_t codeSize; const uint32_t *pCode; } VkShaderModuleCreateInfo;
typedef struct { uint32_t sType; void *pNext; uint32_t flags; uint32_t stage; uint64_t module; const char *pName; void *pSpec; } VkPipelineShaderStageCreateInfo;
typedef struct { uint32_t sType; void *pNext; uint32_t flags; VkPipelineShaderStageCreateInfo stage; uint64_t layout; uint64_t basePipe; int32_t baseIdx; } VkComputePipelineCreateInfo;
typedef struct { uint64_t buffer; uint64_t offset; uint64_t range; } VkDescriptorBufferInfo;
typedef struct { uint32_t sType; void *pNext; uint64_t dstSet; uint32_t dstBinding; uint32_t dstArrayElement; uint32_t descriptorCount; uint32_t descriptorType; void *pImage; VkDescriptorBufferInfo *pBuf; void *pTexel; } VkWriteDescriptorSet;
typedef struct { uint32_t sType; void *pNext; uint32_t waitCnt; void *pWait; void *pStage; uint32_t cmdCnt; void **pCmds; uint32_t sigCnt; void *pSig; } VkSubmitInfo;

typedef struct {
    void *data;
    size_t size;
} MockMem;

typedef struct {
    size_t size;
    MockMem *mem;
} MockBuf;

typedef struct {
    int kind; // 1=add, 2=mul, 3=scale, 4=gemm
} MockShader;

typedef struct {
    int kind;
} MockPipe;

typedef struct {
    MockBuf *bufs[8];
} MockDescSet;

typedef struct {
    MockPipe *pipe;
    MockDescSet *dset;
    uint32_t push[8];
    uint32_t gx, gy, gz;
} MockCmdBuf;

static int validate_and_classify_spirv(const uint32_t *w, size_t nwords) {
    if (nwords < 5) return -1;
    if (w[0] != 0x07230203u) return -2;
    if (w[1] != 0x00010000u) return -3;
    if (w[3] <= 1u) return -4;
    size_t pos = 5;
    int seen_cap = 0, seen_mem = 0, seen_ep = 0, seen_exec = 0, seen_fn = 0, seen_end = 0;
    int has_fadd = 0, has_fmul = 0, has_bitcast = 0, has_loop = 0;
    while (pos < nwords) {
        uint32_t wc = (w[pos] >> 16) & 0xFFFFu;
        uint32_t op = w[pos] & 0xFFFFu;
        if (wc == 0 || pos + wc > nwords) return -5;
        if (op == 17) seen_cap = 1; // OpCapability
        if (op == 14) { if (!seen_cap) return -6; seen_mem = 1; }
        if (op == 15) {
            if (!seen_mem || wc < 6 || w[pos + 1] != 5u) return -7; // GLCompute=5
            seen_ep = 1;
        }
        if (op == 16) { if (!seen_ep) return -8; seen_exec = 1; }
        if (op == 54) { if (!seen_exec) return -9; seen_fn = 1; }
        if (op == 56) seen_end = 1;
        if (op == 129) has_fadd = 1;
        if (op == 133) has_fmul = 1;
        if (op == 124) has_bitcast = 1;
        if (op == 246) has_loop = 1;
        pos += wc;
    }
    if (!seen_cap || !seen_mem || !seen_ep || !seen_exec || !seen_fn || !seen_end) return -10;
    if (has_loop) return 4;
    if (has_bitcast && has_fmul) return 3;
    if (has_fmul) return 2;
    if (has_fadd) return 1;
    return -11;
}

int vkCreateInstance(const void *ci, const void *al, void **inst) { (void)ci; (void)al; *inst = malloc(8); return 0; }
void vkDestroyInstance(void *inst, const void *al) { (void)al; free(inst); }
int vkEnumeratePhysicalDevices(void *inst, uint32_t *cnt, void **devs) {
    (void)inst;
    *cnt = 1;
    if (devs) devs[0] = (void*)0x1001;
    return 0;
}
void vkGetPhysicalDeviceProperties(void *phys, void *props) {
    (void)phys;
    memset(props, 0, 1024);
    *(uint32_t*)((char*)props + 8) = 0x1002;
    snprintf((char*)props + 20, 256, "Xu Mock Vulkan Compute GPU");
}
void vkGetPhysicalDeviceMemoryProperties(void *phys, void *mp) {
    (void)phys;
    memset(mp, 0, 640);
    *(uint32_t*)mp = 1;
    *(uint32_t*)((char*)mp + 4) = 6; // HOST_VISIBLE | HOST_COHERENT
}
void vkGetPhysicalDeviceQueueFamilyProperties(void *phys, uint32_t *cnt, void *qp) {
    (void)phys;
    *cnt = 1;
    if (qp) {
        memset(qp, 0, 24);
        *(uint32_t*)qp = 2; // VK_QUEUE_COMPUTE_BIT
        *(uint32_t*)((char*)qp + 4) = 1;
    }
}
int vkCreateDevice(void *phys, const void *ci, const void *al, void **dev) { (void)phys; (void)ci; (void)al; *dev = malloc(8); return 0; }
void vkDestroyDevice(void *dev, const void *al) { (void)al; free(dev); }
void vkGetDeviceQueue(void *dev, uint32_t qf, uint32_t qi, void **q) { (void)dev; (void)qf; (void)qi; *q = (void*)0x2001; }

int vkCreateBuffer(void *dev, const void *ci, const void *al, uint64_t *buf) {
    (void)dev; (void)al;
    uint64_t sz = *(const uint64_t*)((const char*)ci + 24);
    MockBuf *b = calloc(1, sizeof(MockBuf));
    b->size = (size_t)sz;
    *buf = (uint64_t)(uintptr_t)b;
    return 0;
}
void vkDestroyBuffer(void *dev, uint64_t buf, const void *al) { (void)dev; (void)al; free((void*)(uintptr_t)buf); }
void vkGetBufferMemoryRequirements(void *dev, uint64_t buf, void *req) {
    (void)dev;
    MockBuf *b = (MockBuf*)(uintptr_t)buf;
    *(uint64_t*)((char*)req + 0) = b->size;
    *(uint64_t*)((char*)req + 8) = 16;
    *(uint32_t*)((char*)req + 16) = 1;
}
int vkAllocateMemory(void *dev, const void *ai, const void *al, uint64_t *mem) {
    (void)dev; (void)al;
    uint64_t sz = *(const uint64_t*)((const char*)ai + 16);
    MockMem *m = calloc(1, sizeof(MockMem));
    m->size = (size_t)sz;
    m->data = calloc(1, m->size ? m->size : 16);
    *mem = (uint64_t)(uintptr_t)m;
    return 0;
}
void vkFreeMemory(void *dev, uint64_t mem, const void *al) {
    (void)dev; (void)al;
    MockMem *m = (MockMem*)(uintptr_t)mem;
    if (m) { free(m->data); free(m); }
}
int vkBindBufferMemory(void *dev, uint64_t buf, uint64_t mem, uint64_t off) {
    (void)dev; (void)off;
    MockBuf *b = (MockBuf*)(uintptr_t)buf;
    b->mem = (MockMem*)(uintptr_t)mem;
    return 0;
}
int vkMapMemory(void *dev, uint64_t mem, uint64_t off, uint64_t sz, uint32_t fl, void **pp) {
    (void)dev; (void)off; (void)sz; (void)fl;
    MockMem *m = (MockMem*)(uintptr_t)mem;
    *pp = m->data;
    return 0;
}
void vkUnmapMemory(void *dev, uint64_t mem) { (void)dev; (void)mem; }

int vkCreateShaderModule(void *dev, const VkShaderModuleCreateInfo *ci, const void *al, uint64_t *mod) {
    (void)dev; (void)al;
    if (ci->sType != 16u || (ci->codeSize & 3u) != 0) return -3;
    int kind = validate_and_classify_spirv(ci->pCode, ci->codeSize / 4);
    if (kind < 0) return kind;
    MockShader *s = calloc(1, sizeof(MockShader));
    s->kind = kind;
    *mod = (uint64_t)(uintptr_t)s;
    return 0;
}
void vkDestroyShaderModule(void *dev, uint64_t mod, const void *al) { (void)dev; (void)al; free((void*)(uintptr_t)mod); }
int vkCreateDescriptorSetLayout(void *dev, const void *ci, const void *al, uint64_t *l) { (void)dev; (void)ci; (void)al; *l = 1; return 0; }
void vkDestroyDescriptorSetLayout(void *dev, uint64_t l, const void *al) { (void)dev; (void)l; (void)al; }
int vkCreatePipelineLayout(void *dev, const void *ci, const void *al, uint64_t *l) { (void)dev; (void)ci; (void)al; *l = 1; return 0; }
void vkDestroyPipelineLayout(void *dev, uint64_t l, const void *al) { (void)dev; (void)l; (void)al; }
int vkCreateComputePipelines(void *dev, uint64_t cache, uint32_t cnt, const VkComputePipelineCreateInfo *ci, const void *al, uint64_t *pipes) {
    (void)dev; (void)cache; (void)cnt; (void)al;
    if (ci->sType != 29u || ci->stage.sType != 18u || strcmp(ci->stage.pName, "main") != 0) return -3;
    MockShader *s = (MockShader*)(uintptr_t)ci->stage.module;
    MockPipe *p = calloc(1, sizeof(MockPipe));
    p->kind = s->kind;
    pipes[0] = (uint64_t)(uintptr_t)p;
    return 0;
}
void vkDestroyPipeline(void *dev, uint64_t p, const void *al) { (void)dev; (void)al; free((void*)(uintptr_t)p); }
int vkCreateDescriptorPool(void *dev, const void *ci, const void *al, uint64_t *pool) { (void)dev; (void)ci; (void)al; *pool = 1; return 0; }
void vkDestroyDescriptorPool(void *dev, uint64_t pool, const void *al) { (void)dev; (void)pool; (void)al; }
int vkAllocateDescriptorSets(void *dev, const void *ai, uint64_t *sets) {
    (void)dev; (void)ai;
    sets[0] = (uint64_t)(uintptr_t)calloc(1, sizeof(MockDescSet));
    return 0;
}
void vkUpdateDescriptorSets(void *dev, uint32_t wc, const VkWriteDescriptorSet *w, uint32_t cc, const void *c) {
    (void)dev; (void)cc; (void)c;
    for (uint32_t i = 0; i < wc; ++i) {
        MockDescSet *ds = (MockDescSet*)(uintptr_t)w[i].dstSet;
        if (w[i].dstBinding < 8) ds->bufs[w[i].dstBinding] = (MockBuf*)(uintptr_t)w[i].pBuf->buffer;
    }
}
int vkCreateCommandPool(void *dev, const void *ci, const void *al, uint64_t *cp) { (void)dev; (void)ci; (void)al; *cp = 1; return 0; }
void vkDestroyCommandPool(void *dev, uint64_t cp, const void *al) { (void)dev; (void)cp; (void)al; }
int vkAllocateCommandBuffers(void *dev, const void *ai, void **cmds) { (void)dev; (void)ai; cmds[0] = calloc(1, sizeof(MockCmdBuf)); return 0; }
int vkBeginCommandBuffer(void *cmd, const void *bi) { (void)bi; memset(cmd, 0, sizeof(MockCmdBuf)); return 0; }
int vkEndCommandBuffer(void *cmd) { (void)cmd; return 0; }
void vkCmdBindPipeline(void *cmd, int bp, uint64_t pipe) { (void)bp; ((MockCmdBuf*)cmd)->pipe = (MockPipe*)(uintptr_t)pipe; }
void vkCmdBindDescriptorSets(void *cmd, int bp, uint64_t l, uint32_t f, uint32_t cnt, const uint64_t *sets, uint32_t dc, const uint32_t *doff) {
    (void)bp; (void)l; (void)f; (void)cnt; (void)dc; (void)doff;
    ((MockCmdBuf*)cmd)->dset = (MockDescSet*)(uintptr_t)sets[0];
}
void vkCmdPushConstants(void *cmd, uint64_t l, uint32_t st, uint32_t off, uint32_t sz, const void *vals) {
    (void)l; (void)st;
    if (off + sz <= 32) memcpy(((char*)((MockCmdBuf*)cmd)->push) + off, vals, sz);
}
void vkCmdDispatch(void *cmd, uint32_t gx, uint32_t gy, uint32_t gz) {
    MockCmdBuf *cb = (MockCmdBuf*)cmd;
    cb->gx = gx; cb->gy = gy; cb->gz = gz;
}
void vkCmdPipelineBarrier(void *cmd, uint32_t s, uint32_t d, uint32_t dep, uint32_t mc, void *m, uint32_t bc, void *b, uint32_t ic, void *im) {
    (void)cmd; (void)s; (void)d; (void)dep; (void)mc; (void)m; (void)bc; (void)b; (void)ic; (void)im;
}
int vkQueueSubmit(void *q, uint32_t cnt, const VkSubmitInfo *si, uint64_t fence) {
    (void)q; (void)fence;
    for (uint32_t i = 0; i < cnt; ++i) {
        MockCmdBuf *cb = (MockCmdBuf*)(si[i].pCmds[0]);
        int k = cb->pipe->kind;
        float *dst = (float*)cb->dset->bufs[0]->mem->data;
        const float *a = (const float*)cb->dset->bufs[1]->mem->data;
        if (k == 1) {
            const float *b = (const float*)cb->dset->bufs[2]->mem->data;
            uint32_t n = cb->push[0];
            for (uint32_t idx = 0; idx < n; ++idx) dst[idx] = a[idx] + b[idx];
        } else if (k == 2) {
            const float *b = (const float*)cb->dset->bufs[2]->mem->data;
            uint32_t n = cb->push[0];
            for (uint32_t idx = 0; idx < n; ++idx) dst[idx] = a[idx] * b[idx];
        } else if (k == 3) {
            uint32_t n = cb->push[0];
            float alpha;
            memcpy(&alpha, &cb->push[1], 4);
            for (uint32_t idx = 0; idx < n; ++idx) dst[idx] = a[idx] * alpha;
        } else if (k == 4) {
            const float *b = (const float*)cb->dset->bufs[2]->mem->data;
            uint32_t M = cb->push[0], N = cb->push[1], K = cb->push[2], mode = cb->push[3];
            for (uint32_t r = 0; r < M; ++r) {
                for (uint32_t c = 0; c < N; ++c) {
                    float acc = 0.0f;
                    for (uint32_t p = 0; p < K; ++p) {
                        float va = (mode == 2) ? a[p * M + r] : a[r * K + p];
                        float vb = (mode == 1) ? b[c * K + p] : b[p * N + c];
                        acc += va * vb;
                    }
                    dst[r * N + c] = acc;
                }
            }
        }
        free(cb->dset);
        free(cb);
    }
    return 0;
}
int vkQueueWaitIdle(void *q) { (void)q; return 0; }
int vkCreateFence(void *dev, const void *ci, const void *al, uint64_t *f) { (void)dev; (void)ci; (void)al; *f = 1; return 0; }
void vkDestroyFence(void *dev, uint64_t f, const void *al) { (void)dev; (void)f; (void)al; }
int vkWaitForFences(void *dev, uint32_t cnt, const uint64_t *f, uint32_t wa, uint64_t to) { (void)dev; (void)cnt; (void)f; (void)wa; (void)to; return 0; }
int vkResetFences(void *dev, uint32_t cnt, const uint64_t *f) { (void)dev; (void)cnt; (void)f; return 0; }
'''


class GpuBackendsTests(unittest.TestCase):
    def test_unavailable_without_libraries(self):
        src = r'''@main
cuda #"github.com/xulang-org/packages/cuda_native"
vulkan #"github.com/xulang-org/packages/vulkan_native"
gpu #"github.com/xulang-org/packages/gpu"

fn run() int {
    if cuda.available() || vulkan.available() { <<< 1 }
    if cuda.device_count() != 0 || vulkan.device_count() != 0 { <<< 1 }

    $cd := gpu.open_backend("cpu") >>>
    if cd.backend_id() != gpu.BACKEND_CPU { <<< 1 }
    $a := cd.matrix_f32(1, 2) >>>
    $b := cd.matrix_f32(1, 2) >>>
    $c := cd.matrix_f32(1, 2) >>>
    _ := a.write_f32(&[f32(1.5), f32(2.5)]) >>>
    _ := b.write_f32(&[f32(3.5), f32(4.5)]) >>>
    _ := gpu.add_f32($&cd, $&c, &a, &b) >>>
    got := c.to_f32() >>>
    if got[0] != f32(5.0) || got[1] != f32(7.0) { <<< 1 }
    _ := c.close()
    _ := b.close()
    _ := a.close()
    _ := cd.close()

    $ad := gpu.open_auto() >>>
    println("auto=" + gpu.backend_name(ad.backend_id()))
    _ := ad.close()
    << 0
}

fn main() {
    _ := run() >>> (err) {
        eprintln(err.msg())
        <<< 1
    }
}
'''
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            exe = build_xu(work, "gpu_unavail", src)
            env = xu_env(work, {
                "XU_CUDA_LIBRARY": "/nonexistent/libcuda.so.1",
                "XU_VULKAN_LIBRARY": "/nonexistent/libvulkan.so.1",
            })
            res = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("auto=metal", res.stdout)

    def test_gpu_metal_and_cpu_parity(self):
        src = (
            '@main\n#math\ngpu #"github.com/xulang-org/packages/gpu"\n'
            + COMMON_XU_TEST_HELPERS
            + r'''
fn run_metal() int {
    bks := gpu.detect_backends()
    $has_metal := false
    for i in 0:bks.len { if bks[i] == "metal" { has_metal = true } }
    check(has_metal, "has_metal")
    ref := run_15_ops("cpu") >>>
    got := run_15_ops("metal") >>>
    check(got.len == 15 && ref.len == 15, "len 15")
    for i in 0:15 {
        check_close("metal_op_" + str(i), &got[i], &ref[i], 1.0e-4)
    }
    println("metal_cpu_15_ops_ok")
    << 0
}

fn main() {
    _ := run_metal() >>> (err) {
        eprintln(err.msg())
        <<< 1
    }
}
'''
        )
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            exe = build_xu(work, "metal_cpu_parity", src)
            res = subprocess.run([str(exe)], env=xu_env(work), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("metal_cpu_15_ops_ok", res.stdout)

    def test_cuda_native_and_gpu_cuda_end_to_end_abi(self):
        src = (
            '@main\n#math\ncuda #"github.com/xulang-org/packages/cuda_native"\ngpu #"github.com/xulang-org/packages/gpu"\n'
            + COMMON_XU_TEST_HELPERS
            + r'''
fn run_cuda() int {
    check(cuda.available(), "cuda.available")
    check(cuda.has_nvrtc(), "cuda.has_nvrtc")
    check(cuda.has_cublas(), "cuda.has_cublas")
    check(cuda.device_count() == 1, "cuda.device_count")
    $cd := cuda.open(0) >>>
    check(cd.name() == "Xu Mock NVIDIA RTX 4090", "cuda name")
    check(cd.sm_count() == 128, "cuda sm")
    check(cd.cc_major() == 8, "cuda cc_major")
    check(cd.cc_minor() == 9, "cuda cc_minor")
    check(cd.total_mem() > 0, "cuda total_mem")
    _ := cd.close()

    ref := run_15_ops("cpu") >>>
    got := run_15_ops("cuda") >>>
    check(got.len == 15 && ref.len == 15, "cuda len 15")
    for i in 0:15 {
        check_close("cuda_op_" + str(i), &got[i], &ref[i], 1.0e-4)
    }
    println("cuda_15_ops_abi_ok")
    << 0
}

fn main() {
    _ := run_cuda() >>> (err) {
        eprintln(err.msg())
        <<< 1
    }
}
'''
        )
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            c_path = work / "cuda_mock.c"
            dylib = work / "libcuda_mock.dylib"
            c_path.write_text(CUDA_MOCK_C)
            subprocess.run(["clang", "-shared", "-fPIC", "-O2", str(c_path), "-o", str(dylib), "-lm"], check=True)
            exe = build_xu(work, "cuda_abi_test", src)
            env = xu_env(work, {
                "XU_CUDA_LIBRARY": str(dylib),
                "XU_NVRTC_LIBRARY": str(dylib),
                "XU_CUBLAS_LIBRARY": str(dylib),
            })
            res = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("cuda_15_ops_abi_ok", res.stdout)

    def test_vulkan_native_spirv_and_gpu_vulkan_end_to_end_abi(self):
        src = (
            '@main\n#math\nvulkan #"github.com/xulang-org/packages/vulkan_native"\ngpu #"github.com/xulang-org/packages/gpu"\n'
            + COMMON_XU_TEST_HELPERS
            + r'''
fn run_vk() int {
    check(vulkan.validate_spirv(&vulkan.spv_add_f32()), "spv_add")
    check(vulkan.validate_spirv(&vulkan.spv_mul_f32()), "spv_mul")
    check(vulkan.validate_spirv(&vulkan.spv_scale_f32()), "spv_scale")
    check(vulkan.validate_spirv(&vulkan.spv_gemm_f32(0)), "spv_gemm0")
    check(vulkan.validate_spirv(&vulkan.spv_gemm_f32(1)), "spv_gemm1")
    check(vulkan.validate_spirv(&vulkan.spv_gemm_f32(2)), "spv_gemm2")

    check(vulkan.available(), "vk available")
    check(vulkan.device_count() == 1, "vk dev count")
    $vd := vulkan.open(0) >>>
    check(vd.name() == "Xu Mock Vulkan Compute GPU", "vk name")
    check(vd.vendor_id() == 0x1002, "vk vendor")
    check(vd.queue_family() == 0, "vk qfam")
    _ := vd.close()

    ref := run_15_ops("cpu") >>>
    got := run_15_ops("vulkan") >>>
    check(got.len == 15 && ref.len == 15, "vk len 15")
    for i in 0:15 {
        check_close("vulkan_op_" + str(i), &got[i], &ref[i], 1.0e-4)
    }
    println("vulkan_15_ops_spirv_abi_ok")
    << 0
}

fn main() {
    _ := run_vk() >>> (err) {
        eprintln(err.msg())
        <<< 1
    }
}
'''
        )
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            c_path = work / "vulkan_mock.c"
            dylib = work / "libvulkan_mock.dylib"
            c_path.write_text(VULKAN_MOCK_C)
            subprocess.run(["clang", "-shared", "-fPIC", "-O2", str(c_path), "-o", str(dylib)], check=True)
            exe = build_xu(work, "vulkan_abi_test", src)
            env = xu_env(work, {
                "XU_VULKAN_LIBRARY": str(dylib),
            })
            res = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("vulkan_15_ops_spirv_abi_ok", res.stdout)


if __name__ == "__main__":
    unittest.main()
