"""Verify Metal GPU AI compute kernels against CPU binary32 baselines."""
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

PACKAGES = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("XU_ROOT", PACKAGES.parent / "xu")).expanduser().resolve()
XU = Path(os.environ.get("XU", str(REPO / "bin/xu"))).resolve()


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bits_to_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def xu_compiler_env(work):
    env = dict(
        os.environ,
        XUPATH=str(PACKAGES),
        XU_CACHE=str(work / "cache"),
        XU_ROOT=str(REPO),
        XU_STDLIB_ROOT=str(REPO / "src"),
        XU_METAL_LIBRARY=str(PACKAGES / "metal/native/libxu_metal.dylib"),
    )
    for key in ("XUOS", "XUARCH", "XU_HEAP_STATS"):
        env.pop(key, None)
    return env


def run_xu(work, src_text):
    src = work / "main.xu"
    exe = work / "metal_ai_test"
    src.write_text(src_text)
    (work / "xu.xun").write_text(
        f'module: metal_ai_test\nversion: !ver 0.1.0\nrequire:\n  "github.com/xulang-org/packages": file:{PACKAGES}\n'
    )
    env = xu_compiler_env(work)
    built = subprocess.run(
        [str(XU), "build", str(work), "-o", str(exe)],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    if built.returncode != 0:
        raise AssertionError(built.stdout + built.stderr)
    if os.uname().sysname == "Darwin":
        subprocess.run(["codesign", "--force", "--sign", "-", str(exe)], check=True, capture_output=True)
    res = subprocess.run([str(exe)], env=env, text=True, capture_output=True)
    if res.returncode != 0:
        raise AssertionError(f"rc={res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res.stdout


class TestMetalAI(unittest.TestCase):
    def test_ai_kernels_and_guards(self):
        # 3x5 @ (4x5)^T -> 3x4
        a_gemm = [f32(((i * 7) % 13 - 6) * 0.25) for i in range(15)]
        b_gemm = [f32(((i * 5 + 3) % 11 - 5) * 0.2) for i in range(20)]
        ref_gemm_nt = []
        for r in range(3):
            for c in range(4):
                acc = 0.0
                for k in range(5):
                    acc += a_gemm[r * 5 + k] * b_gemm[c * 5 + k]
                ref_gemm_nt.append(f32(acc))

        # RMSNorm 3x17 (unaligned MPS stride > 17)
        x_rms = [f32(((i * 11 + 2) % 19 - 9) * 0.3) for i in range(3 * 17)]
        w_rms = [f32(0.5 + (i % 5) * 0.25) for i in range(17)]
        eps = 1e-5
        ref_rms = []
        for r in range(3):
            row = x_rms[r * 17:(r + 1) * 17]
            ms = sum(v * v for v in row) / 17.0
            inv = 1.0 / math.sqrt(ms + eps)
            for c in range(17):
                ref_rms.append(f32(row[c] * inv * w_rms[c]))

        # Softmax 4x300 (> 256 threadgroup width + unaligned stride)
        x_sm = [f32(((i * 13 + 5) % 29 - 14) * 0.35) for i in range(4 * 300)]
        ref_sm = []
        for r in range(4):
            row = x_sm[r * 300:(r + 1) * 300]
            mx = max(row)
            exps = [math.exp(v - mx) for v in row]
            sm = sum(exps)
            ref_sm.extend(f32(e / sm) for e in exps)

        # SwiGLU & Add & Mul & Scale & GELU 3x19 (unaligned stride)
        g_vec = [f32(((i * 9 + 1) % 21 - 10) * 0.4) for i in range(3 * 19)]
        u_vec = [f32(((i * 3 + 7) % 17 - 8) * 0.25) for i in range(3 * 19)]
        b_ln = [f32(((i * 2 + 1) % 9 - 4) * 0.125) for i in range(17)]
        ref_swiglu = [f32((g / (1.0 + math.exp(-g))) * u) for g, u in zip(g_vec, u_vec)]
        ref_add = [f32(g + u) for g, u in zip(g_vec, u_vec)]
        ref_mul = [f32(g * u) for g, u in zip(g_vec, u_vec)]
        ref_scale = [f32(g * 0.375) for g in g_vec]
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        ref_gelu = [
            f32(0.5 * v * (1.0 + math.tanh(sqrt_2_over_pi * (v + 0.044715 * v * v * v))))
            for v in g_vec
        ]
        ref_ln = []
        for r in range(3):
            row = x_rms[r * 17:(r + 1) * 17]
            mean = sum(row) / 17.0
            var = sum((v - mean) ** 2 for v in row) / 17.0
            inv = 1.0 / math.sqrt(var + eps)
            for c in range(17):
                ref_ln.append(f32(((row[c] - mean) * inv) * w_rms[c] + b_ln[c]))

        # RoPE [4, 16] (2 heads * head_dim=8, pos_offset=3, theta=10000)
        x_rope = [f32(((i * 5 + 3) % 13 - 6) * 0.25) for i in range(4 * 16)]
        ref_rope = []
        log_theta = math.log(10000.0)
        for r in range(4):
            pos = 3 + r
            for h in range(2):
                base = r * 16 + h * 8
                row_out = [0.0] * 8
                for j in range(4):
                    freq = math.exp(-log_theta * (2 * j / 8.0))
                    ang = pos * freq
                    c, s = math.cos(ang), math.sin(ang)
                    v0, v1 = x_rope[base + j], x_rope[base + 4 + j]
                    row_out[j] = f32(v0 * c - v1 * s)
                    row_out[4 + j] = f32(v0 * s + v1 * c)
                ref_rope.extend(row_out)

        # Causal GQA Attention: seq_len=3, kv_len=5, n_heads=4, n_kv_heads=2, head_dim=8
        q_att = [f32(((i * 7 + 1) % 15 - 7) * 0.2) for i in range(3 * 32)]
        k_att = [f32(((i * 3 + 5) % 11 - 5) * 0.2) for i in range(5 * 16)]
        v_att = [f32(((i * 11 + 2) % 13 - 6) * 0.2) for i in range(5 * 16)]
        att_scale = 1.0 / math.sqrt(8.0)
        ref_att = []
        for qr in range(3):
            max_t = (5 - 3) + qr
            for h in range(4):
                kvh = h // 2
                qv = q_att[qr * 32 + h * 8:qr * 32 + (h + 1) * 8]
                scores = [
                    sum(qv[d] * k_att[t * 16 + kvh * 8 + d] for d in range(8)) * att_scale
                    for t in range(max_t + 1)
                ]
                mx = max(scores)
                exps = [math.exp(sc - mx) for sc in scores]
                sm = sum(exps)
                for d in range(8):
                    ref_att.append(f32(sum((exps[t] / sm) * v_att[t * 16 + kvh * 8 + d] for t in range(max_t + 1))))

        def xu_arr(name, vals):
            bits = ", ".join(str(f32_bits(v)) for v in vals)
            return f"    {name} := [{bits}]\n"

        xu_src = (
            '@main\n'
            'metal #"github.com/xulang-org/packages/metal"\n'
            '#errors\n\n'
            'fn expect_err(label string, ok bool) {\n'
            '    if ok { println("ERR_FAIL:" + label) } else { println("ERR_OK:" + label) }\n'
            '}\n\n'
            'fn dump(tag string, bits &[]int) {\n'
            '    $s := tag + ":"\n'
            '    for i in 0:bits.len {\n'
            '        if i > 0 { s = s + "," }\n'
            '        s = s + str(bits[i])\n'
            '    }\n'
            '    println(s)\n'
            '}\n\n'
            'fn guard_shape(mc $&metal.F32Matrix, mx &metal.F32Matrix, mb &metal.F32Matrix) bool {\n'
            '    _ := metal.gemm_f32_nt(mc, mx, mb) >>> (_) { << true }\n'
            '    << false\n'
            '}\n\n'
            'fn guard_alias(msd $&metal.F32Matrix, mx_bad &metal.F32Matrix) bool {\n'
            '    _ := metal.softmax_f32(msd, mx_bad) >>> (_) { << true }\n'
            '    << false\n'
            '}\n\n'
            'fn guard_eps(mr $&metal.F32Matrix, mx &metal.F32Matrix, mw &metal.F32Matrix) bool {\n'
            '    _ := metal.rmsnorm_f32(mr, mx, mw, 0.0) >>> (_) { << true }\n'
            '    << false\n'
            '}\n\n'
            'fn guard_dev(mo $&metal.F32Matrix, mg &metal.F32Matrix, mu &metal.F32Matrix) bool {\n'
            '    _ := metal.add_f32(mo, mg, mu) >>> (_) { << true }\n'
            '    << false\n'
            '}\n\n'
            'fn run_all() int {\n'
            '    if !metal.available() {\n'
            '        println("SKIP:NO_METAL")\n'
            '        << 0\n'
            '    }\n'
            '    $d := metal.open() >>>\n'
            '    pc := d.pipeline_count() >>>\n'
            '    if pc != 1 {\n'
            '        eprintln("bad pipeline_count")\n'
            '        << 2\n'
            '    }\n'
            + xu_arr("a_gemm", a_gemm)
            + xu_arr("b_gemm", b_gemm)
            + xu_arr("x_rms", x_rms)
            + xu_arr("w_rms", w_rms)
            + xu_arr("x_sm", x_sm)
            + xu_arr("g_vec", g_vec)
            + xu_arr("u_vec", u_vec)
            + xu_arr("b_ln", b_ln)
            + xu_arr("x_rope", x_rope)
            + xu_arr("q_att", q_att)
            + xu_arr("k_att", k_att)
            + xu_arr("v_att", v_att)
            + '    $ma := d.matrix_f32(3, 5) >>>\n'
            '    $mb := d.matrix_f32(4, 5) >>>\n'
            '    $mc := d.matrix_f32(3, 4) >>>\n'
            '    _ := ma.write_bits(&a_gemm) >>>\n'
            '    _ := mb.write_bits(&b_gemm) >>>\n'
            '    $fg := metal.gemm_f32_nt($&mc, &ma, &mb) >>>\n'
            '    _ := fg.wait() >>>\n'
            '    _ := fg.close() >>>\n'
            '    b_gemm_nt := mc.bits() >>>\n'
            '    dump("GEMM_NT", &b_gemm_nt)\n\n'
            '    $mx := d.matrix_f32(3, 17) >>>\n'
            '    $mw := d.matrix_f32(1, 17) >>>\n'
            '    $mr := d.matrix_f32(3, 17) >>>\n'
            '    _ := mx.write_bits(&x_rms) >>>\n'
            '    _ := mw.write_bits(&w_rms) >>>\n'
            '    $fr := metal.rmsnorm_f32($&mr, &mx, &mw, 0.00001) >>>\n'
            '    _ := fr.wait() >>>\n'
            '    _ := fr.close() >>>\n'
            '    b_rms := mr.bits() >>>\n'
            '    dump("RMSNORM", &b_rms)\n\n'
            '    $msx := d.matrix_f32(4, 300) >>>\n'
            '    $msd := d.matrix_f32(4, 300) >>>\n'
            '    _ := msx.write_bits(&x_sm) >>>\n'
            '    $fs := metal.softmax_f32($&msd, &msx) >>>\n'
            '    _ := fs.wait() >>>\n'
            '    _ := fs.close() >>>\n'
            '    b_sm := msd.bits() >>>\n'
            '    dump("SOFTMAX", &b_sm)\n\n'
            '    $mg := d.matrix_f32(3, 19) >>>\n'
            '    $mu := d.matrix_f32(3, 19) >>>\n'
            '    $md1 := d.matrix_f32(3, 19) >>>\n'
            '    $md2 := d.matrix_f32(3, 19) >>>\n'
            '    _ := mg.write_bits(&g_vec) >>>\n'
            '    _ := mu.write_bits(&u_vec) >>>\n'
            '    $fw := metal.swiglu_f32($&md1, &mg, &mu) >>>\n'
            '    _ := fw.wait() >>>\n'
            '    _ := fw.close() >>>\n'
            '    b_swiglu := md1.bits() >>>\n'
            '    dump("SWIGLU", &b_swiglu)\n'
            '    $fa := metal.add_f32($&md2, &mg, &mu) >>>\n'
            '    _ := fa.wait() >>>\n'
            '    _ := fa.close() >>>\n'
            '    b_add := md2.bits() >>>\n'
            '    dump("ADD", &b_add)\n\n'
            '    $mf := d.matrix_f32(3, 19) >>>\n'
            '    $fmul := metal.mul_f32($&mf, &mg, &mu) >>>\n'
            '    _ := fmul.wait() >>>\n'
            '    _ := fmul.close() >>>\n'
            '    b_mul := mf.bits() >>>\n'
            '    dump("MUL", &b_mul)\n'
            '    $fsc := metal.scale_f32($&mf, &mg, 0.375) >>>\n'
            '    _ := fsc.wait() >>>\n'
            '    _ := fsc.close() >>>\n'
            '    b_scale := mf.bits() >>>\n'
            '    dump("SCALE", &b_scale)\n'
            '    $fgelu := metal.gelu_f32($&mf, &mg) >>>\n'
            '    _ := fgelu.wait() >>>\n'
            '    _ := fgelu.close() >>>\n'
            '    b_gelu := mf.bits() >>>\n'
            '    dump("GELU", &b_gelu)\n\n'
            '    $mb_ln := d.matrix_f32(1, 17) >>>\n'
            '    _ := mb_ln.write_bits(&b_ln) >>>\n'
            '    $mln := d.matrix_f32(3, 17) >>>\n'
            '    $fln := metal.layernorm_f32($&mln, &mx, &mw, &mb_ln, 0.00001) >>>\n'
            '    _ := fln.wait() >>>\n'
            '    _ := fln.close() >>>\n'
            '    b_ln_out := mln.bits() >>>\n'
            '    dump("LAYERNORM", &b_ln_out)\n\n'
            '    $mrp_in := d.matrix_f32(4, 16) >>>\n'
            '    $mrp_out := d.matrix_f32(4, 16) >>>\n'
            '    _ := mrp_in.write_bits(&x_rope) >>>\n'
            '    $frp := metal.rope_f32($&mrp_out, &mrp_in, 3, 8, 10000.0) >>>\n'
            '    _ := frp.wait() >>>\n'
            '    _ := frp.close() >>>\n'
            '    b_rope := mrp_out.bits() >>>\n'
            '    dump("ROPE", &b_rope)\n\n'
            '    $mq := d.matrix_f32(3, 32) >>>\n'
            '    $mk := d.matrix_f32(5, 16) >>>\n'
            '    $mv := d.matrix_f32(5, 16) >>>\n'
            '    $ma_out := d.matrix_f32(3, 32) >>>\n'
            '    _ := mq.write_bits(&q_att) >>>\n'
            '    _ := mk.write_bits(&k_att) >>>\n'
            '    _ := mv.write_bits(&v_att) >>>\n'
            '    $fatt := metal.causal_attention_f32($&ma_out, &mq, &mk, &mv, 4, 2, 0.35355339059327373) >>>\n'
            '    _ := fatt.wait() >>>\n'
            '    _ := fatt.close() >>>\n'
            '    b_att := ma_out.bits() >>>\n'
            '    dump("CAUSAL_ATTN", &b_att)\n\n'
            '    // Guards: shape mismatch, aliasing, invalid eps, cross-device\n'
            '    expect_err("shape_gemm_nt", !guard_shape($&mc, &mx, &mb))\n'
            '    expect_err("shape_softmax", !guard_alias($&msd, &mx))\n'
            '    expect_err("bad_eps", !guard_eps($&mr, &mx, &mw))\n'
            '    $d2 := metal.open() >>>\n'
            '    $m_other := d2.matrix_f32(3, 19) >>>\n'
            '    expect_err("cross_device", !guard_dev($&m_other, &mg, &mu))\n\n'
            '    _ := m_other.close() >>>\n'
            '    _ := d2.close() >>>\n'
            '    _ := ma.close() >>>\n'
            '    _ := mb.close() >>>\n'
            '    _ := mc.close() >>>\n'
            '    _ := mx.close() >>>\n'
            '    _ := mw.close() >>>\n'
            '    _ := mr.close() >>>\n'
            '    _ := msx.close() >>>\n'
            '    _ := msd.close() >>>\n'
            '    _ := mg.close() >>>\n'
            '    _ := mu.close() >>>\n'
            '    _ := md1.close() >>>\n'
            '    _ := md2.close() >>>\n'
            '    _ := mf.close() >>>\n'
            '    _ := mb_ln.close() >>>\n'
            '    _ := mln.close() >>>\n'
            '    _ := mrp_in.close() >>>\n'
            '    _ := mrp_out.close() >>>\n'
            '    _ := mq.close() >>>\n'
            '    _ := mk.close() >>>\n'
            '    _ := mv.close() >>>\n'
            '    _ := ma_out.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n'
            '    << 0\n'
            '}\n\n'
            'fn main() int {\n'
            '    << run_all() >>> (err) {\n'
            '        eprintln(err.msg())\n'
            '        << 1\n'
            '    }\n'
            '}\n'
        )

        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)

        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")

        parsed = {}
        err_tags = {}
        for line in out.splitlines():
            if line.startswith("ERR_OK:"):
                err_tags[line.split(":", 1)[1]] = True
            elif line.startswith("ERR_FAIL:"):
                err_tags[line.split(":", 1)[1]] = False
            elif ":" in line:
                k, v = line.split(":", 1)
                parsed[k] = v

        self.assertEqual(parsed.get("LIVE"), "0")
        for guard in ("shape_gemm_nt", "shape_softmax", "bad_eps", "cross_device"):
            self.assertTrue(err_tags.get(guard), f"Guard {guard} did not fail as expected")

        checks = [
            ("GEMM_NT", ref_gemm_nt),
            ("RMSNORM", ref_rms),
            ("SOFTMAX", ref_sm),
            ("SWIGLU", ref_swiglu),
            ("ADD", ref_add),
            ("MUL", ref_mul),
            ("SCALE", ref_scale),
            ("GELU", ref_gelu),
            ("LAYERNORM", ref_ln),
            ("ROPE", ref_rope),
            ("CAUSAL_ATTN", ref_att),
        ]
        for key, ref in checks:
            got = [bits_to_f32(int(x)) for x in parsed[key].split(",")]
            self.assertEqual(len(got), len(ref), key)
            for idx, (gv, rv) in enumerate(zip(got, ref)):
                self.assertLess(abs(gv - rv), 1e-4, f"{key}[{idx}]: gpu={gv} cpu={rv}")

    def test_full_transformer_block_gpu_vs_cpu(self):
        # End-to-end GQA Transformer block: seq_len=3, dim=16, n_heads=4, n_kv_heads=2, head_dim=4, hidden=24
        seq_len, dim, hidden, n_heads, n_kv_heads, head_dim = 3, 16, 24, 4, 2, 4
        kv_dim = n_kv_heads * head_dim
        eps = 1e-5
        scale = 1.0 / math.sqrt(float(head_dim))

        def mk(n, s):
            return [f32((((i * 13 + 7) % 23) - 11) * s) for i in range(n)]

        x = mk(seq_len * dim, 0.15)
        wn1 = [f32(0.8 + (i % 4) * 0.1) for i in range(dim)]
        wn2 = [f32(0.9 + (i % 3) * 0.1) for i in range(dim)]
        wq = mk(dim * dim, 0.08)
        wk = mk(kv_dim * dim, 0.08)
        wv = mk(kv_dim * dim, 0.08)
        wo = mk(dim * dim, 0.08)
        wg = mk(hidden * dim, 0.08)
        wu = mk(hidden * dim, 0.08)
        wd = mk(dim * hidden, 0.08)

        def cpu_rmsnorm(mat, w, rows, cols):
            out = []
            for r in range(rows):
                row = mat[r * cols:(r + 1) * cols]
                inv = 1.0 / math.sqrt(sum(v * v for v in row) / cols + eps)
                out.extend(f32(row[c] * inv * w[c]) for c in range(cols))
            return out

        def cpu_gemm_nt(a, b, m, n, k):
            out = []
            for r in range(m):
                for c in range(n):
                    out.append(f32(sum(a[r * k + p] * b[c * k + p] for p in range(k))))
            return out

        def cpu_rope(mat, rows, heads, hd):
            out = []
            half = hd // 2
            lt = math.log(10000.0)
            for r in range(rows):
                for h in range(heads):
                    base = r * heads * hd + h * hd
                    seg = [0.0] * hd
                    for j in range(half):
                        ang = r * math.exp(-lt * (2 * j / float(hd)))
                        co, si = math.cos(ang), math.sin(ang)
                        v0, v1 = mat[base + j], mat[base + half + j]
                        seg[j] = f32(v0 * co - v1 * si)
                        seg[half + j] = f32(v0 * si + v1 * co)
                    out.extend(seg)
            return out

        n1 = cpu_rmsnorm(x, wn1, seq_len, dim)
        q = cpu_rope(cpu_gemm_nt(n1, wq, seq_len, dim, dim), seq_len, n_heads, head_dim)
        k = cpu_rope(cpu_gemm_nt(n1, wk, seq_len, kv_dim, dim), seq_len, n_kv_heads, head_dim)
        v = cpu_gemm_nt(n1, wv, seq_len, kv_dim, dim)
        att = []
        for qr in range(seq_len):
            for h in range(n_heads):
                kvh = h // (n_heads // n_kv_heads)
                qv = q[qr * dim + h * head_dim:qr * dim + (h + 1) * head_dim]
                sc = [sum(qv[d] * k[t * kv_dim + kvh * head_dim + d] for d in range(head_dim)) * scale for t in range(qr + 1)]
                mx = max(sc)
                ex = [math.exp(s - mx) for s in sc]
                sm = sum(ex)
                for d in range(head_dim):
                    att.append(f32(sum((ex[t] / sm) * v[t * kv_dim + kvh * head_dim + d] for t in range(qr + 1))))
        h_att = [f32(a + b) for a, b in zip(x, cpu_gemm_nt(att, wo, seq_len, dim, dim))]
        n2 = cpu_rmsnorm(h_att, wn2, seq_len, dim)
        fg = cpu_gemm_nt(n2, wg, seq_len, hidden, dim)
        fu = cpu_gemm_nt(n2, wu, seq_len, hidden, dim)
        fsw = [f32((g / (1.0 + math.exp(-g))) * u) for g, u in zip(fg, fu)]
        ref_out = [f32(a + b) for a, b in zip(h_att, cpu_gemm_nt(fsw, wd, seq_len, dim, hidden))]

        def xu_arr(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn dump(t string, v &[]int) {\n'
            '    $s := t + ":"\n    for i in 0:v.len { if i > 0 { s = s + "," }\n s = s + str(v[i]) }\n    println(s)\n'
            '}\n'
            'fn mat(d &metal.Device, r int, c int, vals &[]int) metal.F32Matrix {\n'
            '    $m := d.matrix_f32(r, c) >>>\n'
            '    _ := m.write_bits(vals) >>>\n'
            '    << m\n'
            '}\n'
            'fn run_layer() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_arr("x", x) + xu_arr("wn1", wn1) + xu_arr("wn2", wn2)
            + xu_arr("wq", wq) + xu_arr("wk", wk) + xu_arr("wv", wv) + xu_arr("wo", wo)
            + xu_arr("wg", wg) + xu_arr("wu", wu) + xu_arr("wd", wd)
            + '    $mx := mat(&d, 3, 16, &x) >>>\n'
            '    $mwn1 := mat(&d, 1, 16, &wn1) >>>\n'
            '    $mwn2 := mat(&d, 1, 16, &wn2) >>>\n'
            '    $mwq := mat(&d, 16, 16, &wq) >>>\n'
            '    $mwk := mat(&d, 8, 16, &wk) >>>\n'
            '    $mwv := mat(&d, 8, 16, &wv) >>>\n'
            '    $mwo := mat(&d, 16, 16, &wo) >>>\n'
            '    $mwg := mat(&d, 24, 16, &wg) >>>\n'
            '    $mwu := mat(&d, 24, 16, &wu) >>>\n'
            '    $mwd := mat(&d, 16, 24, &wd) >>>\n'
            '    $mn1 := d.matrix_f32(3, 16) >>>\n'
            '    $mq_raw := d.matrix_f32(3, 16) >>>\n'
            '    $mk_raw := d.matrix_f32(3, 8) >>>\n'
            '    $mv := d.matrix_f32(3, 8) >>>\n'
            '    $mq := d.matrix_f32(3, 16) >>>\n'
            '    $mk := d.matrix_f32(3, 8) >>>\n'
            '    $matt := d.matrix_f32(3, 16) >>>\n'
            '    $mo_proj := d.matrix_f32(3, 16) >>>\n'
            '    $mh_att := d.matrix_f32(3, 16) >>>\n'
            '    $mn2 := d.matrix_f32(3, 16) >>>\n'
            '    $mfg := d.matrix_f32(3, 24) >>>\n'
            '    $mfu := d.matrix_f32(3, 24) >>>\n'
            '    $mfsw := d.matrix_f32(3, 24) >>>\n'
            '    $mffn := d.matrix_f32(3, 16) >>>\n'
            '    $mout := d.matrix_f32(3, 16) >>>\n'
            '    $f1 := metal.rmsnorm_f32($&mn1, &mx, &mwn1, 0.00001) >>>\n    _ := f1.wait() >>>\n    _ := f1.close() >>>\n'
            '    $f2 := metal.gemm_f32_nt($&mq_raw, &mn1, &mwq) >>>\n    _ := f2.wait() >>>\n    _ := f2.close() >>>\n'
            '    $f3 := metal.gemm_f32_nt($&mk_raw, &mn1, &mwk) >>>\n    _ := f3.wait() >>>\n    _ := f3.close() >>>\n'
            '    $f4 := metal.gemm_f32_nt($&mv, &mn1, &mwv) >>>\n    _ := f4.wait() >>>\n    _ := f4.close() >>>\n'
            '    $f5 := metal.rope_f32($&mq, &mq_raw, 0, 4, 10000.0) >>>\n    $f6 := metal.rope_f32($&mk, &mk_raw, 0, 4, 10000.0) >>>\n'
            '    _ := f5.wait() >>>\n    _ := f6.wait() >>>\n    _ := f5.close() >>>\n    _ := f6.close() >>>\n'
            '    $f7 := metal.causal_attention_f32($&matt, &mq, &mk, &mv, 4, 2, 0.5) >>>\n    _ := f7.wait() >>>\n    _ := f7.close() >>>\n'
            '    $f8 := metal.gemm_f32_nt($&mo_proj, &matt, &mwo) >>>\n    _ := f8.wait() >>>\n    _ := f8.close() >>>\n'
            '    $f9 := metal.add_f32($&mh_att, &mx, &mo_proj) >>>\n    _ := f9.wait() >>>\n    _ := f9.close() >>>\n'
            '    $f10 := metal.rmsnorm_f32($&mn2, &mh_att, &mwn2, 0.00001) >>>\n    _ := f10.wait() >>>\n    _ := f10.close() >>>\n'
            '    $f11 := metal.gemm_f32_nt($&mfg, &mn2, &mwg) >>>\n    _ := f11.wait() >>>\n    _ := f11.close() >>>\n'
            '    $f12 := metal.gemm_f32_nt($&mfu, &mn2, &mwu) >>>\n    _ := f12.wait() >>>\n    _ := f12.close() >>>\n'
            '    $f13 := metal.swiglu_f32($&mfsw, &mfg, &mfu) >>>\n    _ := f13.wait() >>>\n    _ := f13.close() >>>\n'
            '    $f14 := metal.gemm_f32_nt($&mffn, &mfsw, &mwd) >>>\n    _ := f14.wait() >>>\n    _ := f14.close() >>>\n'
            '    $f15 := metal.add_f32($&mout, &mh_att, &mffn) >>>\n    _ := f15.wait() >>>\n    _ := f15.close() >>>\n'
            '    bits_out := mout.bits() >>>\n'
            '    $s := "OUT:"\n    for i in 0:bits_out.len { if i > 0 { s = s + "," }\n s = s + str(bits_out[i]) }\n    println(s)\n'
            '    _ := mx.close() >>>\n    _ := mwn1.close() >>>\n    _ := mwn2.close() >>>\n'
            '    _ := mwq.close() >>>\n    _ := mwk.close() >>>\n    _ := mwv.close() >>>\n    _ := mwo.close() >>>\n'
            '    _ := mwg.close() >>>\n    _ := mwu.close() >>>\n    _ := mwd.close() >>>\n'
            '    _ := mn1.close() >>>\n    _ := mq_raw.close() >>>\n    _ := mk_raw.close() >>>\n    _ := mv.close() >>>\n'
            '    _ := mq.close() >>>\n    _ := mk.close() >>>\n    _ := matt.close() >>>\n    _ := mo_proj.close() >>>\n'
            '    _ := mh_att.close() >>>\n    _ := mn2.close() >>>\n    _ := mfg.close() >>>\n    _ := mfu.close() >>>\n'
            '    _ := mfsw.close() >>>\n    _ := mffn.close() >>>\n    _ := mout.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_layer() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        got = [bits_to_f32(int(x)) for x in kv["OUT"].split(",")]
        self.assertEqual(len(got), len(ref_out))
        for i, (gv, rv) in enumerate(zip(got, ref_out)):
            self.assertLess(abs(gv - rv), 1e-3, f"layer[{i}]: gpu={gv} cpu={rv}")

    def test_concurrent_reads_and_rw_guards(self):
        # 2x4 @ (3x4)^T -> 2x3, three concurrent reads on the same input matrix mn1
        a = [f32((i + 1) * 0.25) for i in range(8)]
        wq = [f32(((i * 3) % 7 - 3) * 0.2) for i in range(12)]
        wk = [f32(((i * 5) % 9 - 4) * 0.2) for i in range(12)]
        wv = [f32(((i * 7) % 11 - 5) * 0.2) for i in range(12)]

        def cpu_mm_nt(x, w):
            out = []
            for r in range(2):
                for c in range(3):
                    out.append(f32(sum(x[r * 4 + k] * w[c * 4 + k] for k in range(4))))
            return out

        ref_q = cpu_mm_nt(a, wq)
        ref_k = cpu_mm_nt(a, wk)
        ref_v = cpu_mm_nt(a, wv)

        def xu_arr(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn dump(t string, v &[]int) {\n'
            '    $s := t + ":"\n    for i in 0:v.len { if i > 0 { s = s + "," }\n s = s + str(v[i]) }\n    println(s)\n'
            '}\n'
            'fn mat(d &metal.Device, r int, c int, vals &[]int) metal.F32Matrix {\n'
            '    $m := d.matrix_f32(r, c) >>>\n'
            '    _ := m.write_bits(vals) >>>\n'
            '    << m\n'
            '}\n'
            'fn try_write_busy(m $&metal.F32Matrix, vals &[]int) bool {\n'
            '    _ := m.write_bits(vals) >>> (_) { << true }\n'
            '    << false\n'
            '}\n'
            'fn try_read_busy(m &metal.F32Matrix) bool {\n'
            '    _ := m.bits() >>> (_) { << true }\n'
            '    << false\n'
            '}\n'
            'fn run_conc() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_arr("a", a) + xu_arr("wq", wq) + xu_arr("wk", wk) + xu_arr("wv", wv)
            + '    $mn1 := mat(&d, 2, 4, &a) >>>\n'
            '    $mwq := mat(&d, 3, 4, &wq) >>>\n'
            '    $mwk := mat(&d, 3, 4, &wk) >>>\n'
            '    $mwv := mat(&d, 3, 4, &wv) >>>\n'
            '    $out_q := d.matrix_f32(2, 3) >>>\n'
            '    $out_k := d.matrix_f32(2, 3) >>>\n'
            '    $out_v := d.matrix_f32(2, 3) >>>\n'
            '    // Deterministic R/W lock verification using an active Batch:\n'
            '    $b_chk := d.begin_batch() >>>\n'
            '    _ := b_chk.gemm_f32_nt($&out_q, &mn1, &mwq) >>>\n'
            '    if !try_write_busy($&mn1, &a) { << 2 }\n'
            '    if !try_read_busy(&out_q) { << 3 }\n'
            '    _ := mn1.bits() >>>\n'
            '    _ := b_chk.close() >>>\n'
            '    // Now launch 3 concurrent standalone GPU dispatches all reading mn1 simultaneously:\n'
            '    $fq := metal.gemm_f32_nt($&out_q, &mn1, &mwq) >>>\n'
            '    $fk := metal.gemm_f32_nt($&out_k, &mn1, &mwk) >>>\n'
            '    $fv := metal.gemm_f32_nt($&out_v, &mn1, &mwv) >>>\n'
            '    _ := fq.wait() >>>\n    _ := fq.close() >>>\n'
            '    _ := fk.wait() >>>\n    _ := fk.close() >>>\n'
            '    _ := fv.wait() >>>\n    _ := fv.close() >>>\n'
            '    bq := out_q.bits() >>>\n    bk := out_k.bits() >>>\n    bv := out_v.bits() >>>\n'
            '    dump("Q", &bq)\n    dump("K", &bk)\n    dump("V", &bv)\n'
            '    _ := mn1.close() >>>\n    _ := mwq.close() >>>\n    _ := mwk.close() >>>\n    _ := mwv.close() >>>\n'
            '    _ := out_q.close() >>>\n    _ := out_k.close() >>>\n    _ := out_v.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_conc() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        for key, ref in (("Q", ref_q), ("K", ref_k), ("V", ref_v)):
            got = [bits_to_f32(int(x)) for x in kv[key].split(",")]
            for i, (gv, rv) in enumerate(zip(got, ref)):
                self.assertLess(abs(gv - rv), 1e-4, f"{key}[{i}]: {gv} vs {rv}")

    def test_f16_and_q4_0_matrices(self):
        # Test F16Matrix conversions + GEMM + GEMM_NT, and Q4_0Matrix GEMV (M=1 and M=4)
        def f16_bits(v):
            return struct.unpack("<H", struct.pack("<e", v))[0]

        def bits_to_f16(b):
            return struct.unpack("<e", struct.pack("<H", b & 0xFFFF))[0]

        a_vals = [bits_to_f16(f16_bits(((i * 7) % 13 - 6) * 0.25)) for i in range(12)]  # 3x4
        b_vals = [bits_to_f16(f16_bits(((i * 5 + 1) % 11 - 5) * 0.25)) for i in range(20)]  # 5x4
        ref_f16_nt = []
        for r in range(3):
            for c in range(5):
                acc = 0.0
                for k in range(4):
                    acc += a_vals[r * 4 + k] * b_vals[c * 4 + k]
                ref_f16_nt.append(bits_to_f16(f16_bits(acc)))

        # Q4_0: rows=12, cols=64 (2 blocks/row), x: 4x64
        x_q4 = [f32(((i * 11 + 3) % 19 - 9) * 0.125) for i in range(4 * 64)]
        w_raw_src = [f32(((i * 7 + 5) % 23 - 11) * 0.25) for i in range(12 * 64)]
        q4_bytes = bytearray()
        w_dequant = []
        for row in range(12):
            for blk in range(2):
                seg = w_raw_src[(row * 2 + blk) * 32:(row * 2 + blk + 1) * 32]
                amax = max(abs(v) for v in seg)
                d = f32(amax / 7.0) if amax > 0 else 0.0
                d16_b = struct.pack("<e", d)
                d16 = f32(struct.unpack("<e", d16_b)[0])
                q4_bytes.extend(d16_b)
                inv_d = 1.0 / d16 if d16 != 0 else 0.0
                qs = [min(15, max(0, int(round(v * inv_d)) + 8)) for v in seg]
                for j in range(16):
                    q4_bytes.append((qs[j] & 15) | ((qs[16 + j] & 15) << 4))
                w_dequant.extend(f32((q - 8) * d16) for q in qs)
        ref_q4_m4 = []
        for m in range(4):
            for r in range(12):
                ref_q4_m4.append(f32(sum(x_q4[m * 64 + k] * w_dequant[r * 64 + k] for k in range(64))))
        ref_q4_m1 = ref_q4_m4[:12]

        def xu_int_arr(name, ints):
            return f"    {name} := [{', '.join(str(x) for x in ints)}]\n"

        q4_byte_ints = ", ".join(str(b) for b in q4_bytes)
        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn dump(tag string, bits &[]int) {\n'
            '    $s := tag + ":"\n    for i in 0:bits.len { if i > 0 { s = s + "," }\n s = s + str(bits[i]) }\n    println(s)\n'
            '}\n'
            'fn guard_q4_shape(d &metal.Device) bool {\n'
            '    _ := d.matrix_q4_0(4, 30) >>> (_) { << true }\n    << false\n'
            '}\n'
            'fn guard_q4_len(q $&metal.Q4Matrix) bool {\n'
            '    bad := "short"\n'
            '    _ := q.write_bytes(&bad) >>> (_) { << true }\n    << false\n'
            '}\n'
            'fn run_quant() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_int_arr("a_h", [f16_bits(v) for v in a_vals])
            + xu_int_arr("b_h", [f16_bits(v) for v in b_vals])
            + xu_int_arr("x_f", [f32_bits(v) for v in x_q4])
            + xu_int_arr("x_m1_f", [f32_bits(v) for v in x_q4[:64]])
            + f'    q4_b_arr := [{q4_byte_ints}]\n'
            '    $q4_str := ""\n    for b in q4_b_arr { q4_str.push_str(string_from_byte(b)) }\n'
            '    $ma := d.matrix_f16(3, 4) >>>\n    $mb := d.matrix_f16(5, 4) >>>\n    $mc := d.matrix_f16(3, 5) >>>\n'
            '    _ := ma.write_bits(&a_h) >>>\n    _ := mb.write_bits(&b_h) >>>\n'
            '    $fg := metal.gemm_f16_nt($&mc, &ma, &mb) >>>\n    _ := fg.wait() >>>\n    _ := fg.close() >>>\n'
            '    out_c := mc.bits() >>>\n    dump("F16_NT", &out_c)\n'
            '    $mf32 := d.matrix_f32(3, 5) >>>\n'
            '    $fc := metal.f16_to_f32($&mf32, &mc) >>>\n    _ := fc.wait() >>>\n    _ := fc.close() >>>\n'
            '    $mc2 := d.matrix_f16(3, 5) >>>\n'
            '    $fc2 := metal.f32_to_f16($&mc2, &mf32) >>>\n    _ := fc2.wait() >>>\n    _ := fc2.close() >>>\n'
            '    out_c2 := mc2.bits() >>>\n    dump("F16_RT", &out_c2)\n'
            '    if !guard_q4_shape(&d) { << 2 }\n'
            '    $mq4 := d.matrix_q4_0(12, 64) >>>\n'
            '    if !guard_q4_len($&mq4) { << 3 }\n'
            '    _ := mq4.write_bytes(&q4_str) >>>\n'
            '    $mx4 := d.matrix_f32(4, 64) >>>\n    _ := mx4.write_bits(&x_f) >>>\n'
            '    $md4 := d.matrix_f32(4, 12) >>>\n'
            '    $fq4 := metal.gemv_q4_0_f32($&md4, &mx4, &mq4) >>>\n    _ := fq4.wait() >>>\n    _ := fq4.close() >>>\n'
            '    out_q4_m4 := md4.bits() >>>\n    dump("Q4_M4", &out_q4_m4)\n'
            '    $mx1 := d.matrix_f32(1, 64) >>>\n    _ := mx1.write_bits(&x_m1_f) >>>\n'
            '    $md1 := d.matrix_f32(1, 12) >>>\n'
            '    $fq1 := metal.gemv_q4_0_f32($&md1, &mx1, &mq4) >>>\n    _ := fq1.wait() >>>\n    _ := fq1.close() >>>\n'
            '    out_q4_m1 := md1.bits() >>>\n    dump("Q4_M1", &out_q4_m1)\n'
            '    _ := ma.close() >>>\n    _ := mb.close() >>>\n    _ := mc.close() >>>\n    _ := mc2.close() >>>\n    _ := mf32.close() >>>\n'
            '    _ := mq4.close() >>>\n    _ := mx4.close() >>>\n    _ := md4.close() >>>\n    _ := mx1.close() >>>\n    _ := md1.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_quant() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        got_f16 = [bits_to_f16(int(x)) for x in kv["F16_NT"].split(",")]
        got_f16_rt = [bits_to_f16(int(x)) for x in kv["F16_RT"].split(",")]
        self.assertEqual(got_f16, got_f16_rt)
        for i, (gv, rv) in enumerate(zip(got_f16, ref_f16_nt)):
            self.assertLess(abs(gv - rv), 5e-3, f"F16[{i}]: {gv} vs {rv}")
        got_m4 = [bits_to_f32(int(x)) for x in kv["Q4_M4"].split(",")]
        got_m1 = [bits_to_f32(int(x)) for x in kv["Q4_M1"].split(",")]
        for i, (gv, rv) in enumerate(zip(got_m4, ref_q4_m4)):
            self.assertLess(abs(gv - rv), 1e-3, f"Q4_M4[{i}]: {gv} vs {rv}")
        for i, (gv, rv) in enumerate(zip(got_m1, ref_q4_m1)):
            self.assertLess(abs(gv - rv), 1e-3, f"Q4_M1[{i}]: {gv} vs {rv}")

    def test_single_batch_full_transformer_layer(self):
        # Encode all 15 Transformer layer ops into ONE Batch -> dispatch_count() increases by 1!
        seq_len, dim, hidden, n_heads, n_kv_heads, head_dim = 3, 16, 24, 4, 2, 4
        kv_dim = n_kv_heads * head_dim
        eps = 1e-5
        scale = 1.0 / math.sqrt(float(head_dim))

        def mk(n, s):
            return [f32((((i * 13 + 7) % 23) - 11) * s) for i in range(n)]

        x = mk(seq_len * dim, 0.15)
        wn1 = [f32(0.8 + (i % 4) * 0.1) for i in range(dim)]
        wn2 = [f32(0.9 + (i % 3) * 0.1) for i in range(dim)]
        wq = mk(dim * dim, 0.08)
        wk = mk(kv_dim * dim, 0.08)
        wv = mk(kv_dim * dim, 0.08)
        wo = mk(dim * dim, 0.08)
        wg = mk(hidden * dim, 0.08)
        wu = mk(hidden * dim, 0.08)
        wd = mk(dim * hidden, 0.08)

        def cpu_rmsnorm(mat, w, rows, cols):
            out = []
            for r in range(rows):
                row = mat[r * cols:(r + 1) * cols]
                inv = 1.0 / math.sqrt(sum(v * v for v in row) / cols + eps)
                out.extend(f32(row[c] * inv * w[c]) for c in range(cols))
            return out

        def cpu_gemm_nt(a, b, m, n, k):
            return [f32(sum(a[r * k + p] * b[c * k + p] for p in range(k))) for r in range(m) for c in range(n)]

        def cpu_rope(mat, rows, heads, hd):
            out = []
            half = hd // 2
            lt = math.log(10000.0)
            for r in range(rows):
                for h in range(heads):
                    base = r * heads * hd + h * hd
                    seg = [0.0] * hd
                    for j in range(half):
                        ang = r * math.exp(-lt * (2 * j / float(hd)))
                        co, si = math.cos(ang), math.sin(ang)
                        v0, v1 = mat[base + j], mat[base + half + j]
                        seg[j] = f32(v0 * co - v1 * si)
                        seg[half + j] = f32(v0 * si + v1 * co)
                    out.extend(seg)
            return out

        n1 = cpu_rmsnorm(x, wn1, seq_len, dim)
        q = cpu_rope(cpu_gemm_nt(n1, wq, seq_len, dim, dim), seq_len, n_heads, head_dim)
        k = cpu_rope(cpu_gemm_nt(n1, wk, seq_len, kv_dim, dim), seq_len, n_kv_heads, head_dim)
        v = cpu_gemm_nt(n1, wv, seq_len, kv_dim, dim)
        att = []
        for qr in range(seq_len):
            for h in range(n_heads):
                kvh = h // (n_heads // n_kv_heads)
                qv = q[qr * dim + h * head_dim:qr * dim + (h + 1) * head_dim]
                sc = [sum(qv[d] * k[t * kv_dim + kvh * head_dim + d] for d in range(head_dim)) * scale for t in range(qr + 1)]
                mx = max(sc)
                ex = [math.exp(s - mx) for s in sc]
                sm = sum(ex)
                for d in range(head_dim):
                    att.append(f32(sum((ex[t] / sm) * v[t * kv_dim + kvh * head_dim + d] for t in range(qr + 1))))
        h_att = [f32(a + b) for a, b in zip(x, cpu_gemm_nt(att, wo, seq_len, dim, dim))]
        n2 = cpu_rmsnorm(h_att, wn2, seq_len, dim)
        fg = cpu_gemm_nt(n2, wg, seq_len, hidden, dim)
        fu = cpu_gemm_nt(n2, wu, seq_len, hidden, dim)
        fsw = [f32((g / (1.0 + math.exp(-g))) * u) for g, u in zip(fg, fu)]
        ref_out = [f32(a + b) for a, b in zip(h_att, cpu_gemm_nt(fsw, wd, seq_len, dim, hidden))]

        def xu_arr(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn dump(t string, v &[]int) {\n'
            '    $s := t + ":"\n    for i in 0:v.len { if i > 0 { s = s + "," }\n s = s + str(v[i]) }\n    println(s)\n'
            '}\n'
            'fn mat(d &metal.Device, r int, c int, vals &[]int) metal.F32Matrix {\n'
            '    $m := d.matrix_f32(r, c) >>>\n'
            '    _ := m.write_bits(vals) >>>\n'
            '    << m\n'
            '}\n'
            'fn run_batch_layer() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_arr("x", x) + xu_arr("wn1", wn1) + xu_arr("wn2", wn2)
            + xu_arr("wq", wq) + xu_arr("wk", wk) + xu_arr("wv", wv) + xu_arr("wo", wo)
            + xu_arr("wg", wg) + xu_arr("wu", wu) + xu_arr("wd", wd)
            + '    $mx := mat(&d, 3, 16, &x) >>>\n'
            '    $mwn1 := mat(&d, 1, 16, &wn1) >>>\n'
            '    $mwn2 := mat(&d, 1, 16, &wn2) >>>\n'
            '    $mwq := mat(&d, 16, 16, &wq) >>>\n'
            '    $mwk := mat(&d, 8, 16, &wk) >>>\n'
            '    $mwv := mat(&d, 8, 16, &wv) >>>\n'
            '    $mwo := mat(&d, 16, 16, &wo) >>>\n'
            '    $mwg := mat(&d, 24, 16, &wg) >>>\n'
            '    $mwu := mat(&d, 24, 16, &wu) >>>\n'
            '    $mwd := mat(&d, 16, 24, &wd) >>>\n'
            '    $mn1 := d.matrix_f32(3, 16) >>>\n'
            '    $mq_raw := d.matrix_f32(3, 16) >>>\n'
            '    $mk_raw := d.matrix_f32(3, 8) >>>\n'
            '    $mv := d.matrix_f32(3, 8) >>>\n'
            '    $mq := d.matrix_f32(3, 16) >>>\n'
            '    $mk := d.matrix_f32(3, 8) >>>\n'
            '    $matt := d.matrix_f32(3, 16) >>>\n'
            '    $mo_proj := d.matrix_f32(3, 16) >>>\n'
            '    $mh_att := d.matrix_f32(3, 16) >>>\n'
            '    $mn2 := d.matrix_f32(3, 16) >>>\n'
            '    $mfg := d.matrix_f32(3, 24) >>>\n'
            '    $mfu := d.matrix_f32(3, 24) >>>\n'
            '    $mfsw := d.matrix_f32(3, 24) >>>\n'
            '    $mffn := d.matrix_f32(3, 16) >>>\n'
            '    $mout := d.matrix_f32(3, 16) >>>\n'
            '    // First test aborting an uncommitted batch cleanly releases buffer locks\n'
            '    $b_abort := d.begin_batch() >>>\n'
            '    _ := b_abort.rmsnorm_f32($&mn1, &mx, &mwn1, 0.00001) >>>\n'
            '    _ := b_abort.close() >>>\n'
            '    dc0 := d.dispatch_count() >>>\n'
            '    $b := d.begin_batch() >>>\n'
            '    _ := b.rmsnorm_f32($&mn1, &mx, &mwn1, 0.00001) >>>\n'
            '    _ := b.gemm_f32_nt($&mq_raw, &mn1, &mwq) >>>\n'
            '    _ := b.gemm_f32_nt($&mk_raw, &mn1, &mwk) >>>\n'
            '    _ := b.gemm_f32_nt($&mv, &mn1, &mwv) >>>\n'
            '    _ := b.rope_f32($&mq, &mq_raw, 0, 4, 10000.0) >>>\n'
            '    _ := b.rope_f32($&mk, &mk_raw, 0, 4, 10000.0) >>>\n'
            '    _ := b.causal_attention_f32($&matt, &mq, &mk, &mv, 4, 2, 0.5) >>>\n'
            '    _ := b.gemm_f32_nt($&mo_proj, &matt, &mwo) >>>\n'
            '    _ := b.add_f32($&mh_att, &mx, &mo_proj) >>>\n'
            '    _ := b.rmsnorm_f32($&mn2, &mh_att, &mwn2, 0.00001) >>>\n'
            '    _ := b.gemm_f32_nt($&mfg, &mn2, &mwg) >>>\n'
            '    _ := b.gemm_f32_nt($&mfu, &mn2, &mwu) >>>\n'
            '    _ := b.swiglu_f32($&mfsw, &mfg, &mfu) >>>\n'
            '    _ := b.gemm_f32_nt($&mffn, &mfsw, &mwd) >>>\n'
            '    _ := b.add_f32($&mout, &mh_att, &mffn) >>>\n'
            '    $f := b.commit() >>>\n'
            '    _ := f.wait() >>>\n'
            '    _ := f.close() >>>\n'
            '    dc1 := d.dispatch_count() >>>\n'
            '    println("DELTA:" + str(dc1 - dc0))\n'
            '    bits_out := mout.bits() >>>\n'
            '    $s := "OUT:"\n    for i in 0:bits_out.len { if i > 0 { s = s + "," }\n s = s + str(bits_out[i]) }\n    println(s)\n'
            '    _ := mx.close() >>>\n    _ := mwn1.close() >>>\n    _ := mwn2.close() >>>\n'
            '    _ := mwq.close() >>>\n    _ := mwk.close() >>>\n    _ := mwv.close() >>>\n    _ := mwo.close() >>>\n'
            '    _ := mwg.close() >>>\n    _ := mwu.close() >>>\n    _ := mwd.close() >>>\n'
            '    _ := mn1.close() >>>\n    _ := mq_raw.close() >>>\n    _ := mk_raw.close() >>>\n    _ := mv.close() >>>\n'
            '    _ := mq.close() >>>\n    _ := mk.close() >>>\n    _ := matt.close() >>>\n    _ := mo_proj.close() >>>\n'
            '    _ := mh_att.close() >>>\n    _ := mn2.close() >>>\n    _ := mfg.close() >>>\n    _ := mfu.close() >>>\n'
            '    _ := mfsw.close() >>>\n    _ := mffn.close() >>>\n    _ := mout.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_batch_layer() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        self.assertEqual(kv.get("DELTA"), "1")
        got = [bits_to_f32(int(x)) for x in kv["OUT"].split(",")]
        self.assertEqual(len(got), len(ref_out))
        for i, (gv, rv) in enumerate(zip(got, ref_out)):
            self.assertLess(abs(gv - rv), 1e-3, f"batch_layer[{i}]: gpu={gv} cpu={rv}")


if __name__ == "__main__":
    unittest.main()
