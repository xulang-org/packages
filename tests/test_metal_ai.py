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

    def test_v14_ops_and_quant_parity(self):
        # Pack Q8_0 (rows=4, cols=64) and Q4_K (rows=3, cols=256)
        x_q8 = [f32(((i * 5 + 3) % 17 - 8) * 0.25) for i in range(2 * 64)]
        w_q8_src = [f32(((i * 7 + 1) % 21 - 10) * 0.125) for i in range(4 * 64)]
        q8_bytes = bytearray()
        w_q8_dq = []
        for blk in range(len(w_q8_src) // 32):
            seg = w_q8_src[blk * 32:(blk + 1) * 32]
            amax = max(abs(v) for v in seg)
            d = f32(amax / 127.0) if amax > 0 else 0.0
            d16_b = struct.pack("<e", d)
            d16 = f32(struct.unpack("<e", d16_b)[0])
            q8_bytes.extend(d16_b)
            inv_d = 1.0 / d16 if d16 != 0 else 0.0
            for v in seg:
                q = max(-127, min(127, int(round(v * inv_d))))
                q8_bytes.append(q & 0xFF)
                w_q8_dq.append(f32(q * d16))
        ref_q8 = [
            f32(sum(x_q8[m * 64 + k] * w_q8_dq[r * 64 + k] for k in range(64)))
            for m in range(2) for r in range(4)
        ]

        x_q4k = [f32(((i * 3 + 2) % 15 - 7) * 0.125) for i in range(256)]
        w_q4k_src = [f32(((i * 11 + 5) % 19 - 9) * 0.125) for i in range(3 * 256)]
        q4k_bytes = bytearray()
        w_q4k_dq = []
        for r in range(3):
            blk = w_q4k_src[r * 256:(r + 1) * 256]
            sub_sc, sub_mn = [], []
            for s in range(8):
                sub = blk[s * 32:(s + 1) * 32]
                lo, hi = min(0.0, min(sub)), max(sub)
                rng = max(hi - lo, 0.0)
                sub_sc.append(rng / 15.0 if rng > 0.0 else 0.0)
                sub_mn.append(-lo)
            max_sc, max_mn = max(sub_sc), max(sub_mn)
            d16_b = struct.pack("<e", f32(max_sc / 63.0) if max_sc > 0 else 0.0)
            dm16_b = struct.pack("<e", f32(max_mn / 63.0) if max_mn > 0 else 0.0)
            d16 = f32(struct.unpack("<e", d16_b)[0])
            dm16 = f32(struct.unpack("<e", dm16_b)[0])
            q4k_bytes.extend(d16_b + dm16_b)
            sc_q = [min(63, max(0, int(round(s / d16)))) if d16 > 0 else 0 for s in sub_sc]
            mn_q = [min(63, max(0, int(round(m / dm16)))) if dm16 > 0 else 0 for m in sub_mn]
            scales_b = bytearray(12)
            for j in range(4):
                scales_b[j] = sc_q[j] & 63
                scales_b[j + 4] = mn_q[j] & 63
            for j in range(4, 8):
                scales_b[j + 4] = (sc_q[j] & 15) | ((mn_q[j] & 15) << 4)
                scales_b[j - 4] |= ((sc_q[j] >> 4) & 3) << 6
                scales_b[j] |= ((mn_q[j] >> 4) & 3) << 6
            q4k_bytes.extend(scales_b)
            qs_b = bytearray(128)
            for ch in range(4):
                s0, s1 = ch * 2, ch * 2 + 1
                d1, m1 = f32(d16 * sc_q[s0]), f32(dm16 * mn_q[s0])
                d2, m2 = f32(d16 * sc_q[s1]), f32(dm16 * mn_q[s1])
                sub0, sub1 = blk[s0 * 32:(s0 + 1) * 32], blk[s1 * 32:(s1 + 1) * 32]
                dq0, dq1 = [], []
                for l in range(32):
                    q0 = min(15, max(0, int(round((sub0[l] + m1) / d1)))) if d1 > 0 else 0
                    q1 = min(15, max(0, int(round((sub1[l] + m2) / d2)))) if d2 > 0 else 0
                    qs_b[ch * 32 + l] = (q0 & 15) | ((q1 & 15) << 4)
                    dq0.append(f32(f32(d1 * q0) - m1))
                    dq1.append(f32(f32(d2 * q1) - m2))
                w_q4k_dq.extend(dq0)
                w_q4k_dq.extend(dq1)
            q4k_bytes.extend(qs_b)
        ref_q4k = [f32(sum(x_q4k[k] * w_q4k_dq[r * 256 + k] for k in range(256))) for r in range(3)]

        # Reference qk_norm_f32 on [2, 8] with n_heads=2, head_dim=4, eps=1e-5
        qk_x = [f32(v) for v in [1.0, -2.0, 3.0, 0.5, 0.25, -1.5, 2.5, 1.0,
                                  -1.0, 0.5, 1.5, -2.0, 2.0, 1.0, -0.5, 0.25]]
        qk_w = [f32(1.25), f32(0.75), f32(1.5), f32(0.5)]
        ref_qkn = []
        for r in range(2):
            for h in range(2):
                seg = qk_x[r * 8 + h * 4:r * 8 + (h + 1) * 4]
                ms = f32(sum(v * v for v in seg) / 4.0)
                inv = f32(1.0 / math.sqrt(ms + 1e-5))
                ref_qkn.extend(f32(v * inv * wv) for v, wv in zip(seg, qk_w))

        # Reference attention_decode_f32 (kv_len=3 of 8-row cache, n_heads=4, n_kv_heads=2, head_dim=4, scale=0.5)
        dec_q = [f32(((i * 5) % 9 - 4) * 0.25) for i in range(16)]
        kc = [f32(((i * 7 + 1) % 11 - 5) * 0.25) for i in range(8 * 8)]
        vc = [f32(((i * 3 + 2) % 13 - 6) * 0.25) for i in range(8 * 8)]
        ref_dec = []
        for h in range(4):
            kvh = h // 2
            qv = dec_q[h * 4:(h + 1) * 4]
            scores = [sum(qv[d] * kc[t * 8 + kvh * 4 + d] for d in range(4)) * 0.5 for t in range(3)]
            mx = max(scores)
            ex = [math.exp(s - mx) for s in scores]
            sm = sum(ex)
            for d in range(4):
                ref_dec.append(f32(sum((ex[t] / sm) * vc[t * 8 + kvh * 4 + d] for t in range(3))))

        # Reference embedding_f32 & argmax_f32 (with tie)
        emb_tbl = [f32(r * 10 + c + 0.5) for r in range(5) for c in range(4)]
        emb_ids = [3, 0, 4]
        ref_emb = [emb_tbl[idx * 4 + c] for idx in emb_ids for c in range(4)]
        am_x = [f32(1.0), f32(5.0), f32(5.0), f32(2.0),  -3.0, -1.0, -2.0, -4.0]
        ref_am = [1.0, 1.0]

        def bstr_lit(raw):
            return '"' + "".join(f"\\{b:03o}" for b in raw) + '"'

        def xu_arr(name, vals):
            bits = ", ".join(str(f32_bits(v)) for v in vals)
            return f"    {name}_b := [{bits}]\n"

        q8_hex = ", ".join(str(b) for b in q8_bytes)
        q4k_hex = ", ".join(str(b) for b in q4k_bytes)
        wb_src = struct.pack("<4f", 1.5, -2.5, 3.25, 0.0)
        wb_ints = ", ".join(str(b) for b in wb_src)
        f16b_raw = struct.pack("<4e", 1.5, -2.5, 3.25, 0.0)
        f16b_ints = ", ".join(str(b) for b in f16b_raw)

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn to_bytes(vals &[]int) string {\n'
            '    $out := ""\n    for i in 0:vals.len { out.push_str(string_from_byte(vals[i] & 255)) }\n'
            '    << out\n}\n'
            'fn dump(tag string, bits &[]int) {\n'
            '    $s := tag + ":"\n    for i in 0:bits.len { if i > 0 { s = s + "," }\n s = s + str(bits[i]) }\n    println(s)\n}\n'
            'fn mat(d &metal.Device, r int, c int, b &[]int) metal.F32Matrix {\n'
            '    $m := d.matrix_f32(r, c) >>>\n    _ := m.write_bits(b) >>>\n    << m\n}\n'
            'fn guard_emb(d &metal.Device, tbl &metal.F32Matrix) bool {\n'
            '    $dst := d.matrix_f32(1, 4) >>>\n    bad := [99]\n'
            '    _ := metal.embedding_f32($&dst, tbl, &bad) >>> (_) { _ := dst.close() >>>\n << true }\n'
            '    _ := dst.close() >>>\n    << false\n}\n'
            'fn run_v14() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            '    // 1. MATRIX_MAX_DIM raised (>16384): allocate 32768 x 64 F32Matrix\n'
            '    $big := d.matrix_f32(32768, 64) >>>\n'
            '    println("BIG:" + str(big.rows()) + "x" + str(big.columns()))\n'
            '    _ := big.close() >>>\n'
            f'    wbf_arr := [{wb_ints}]\n    wbf := to_bytes(&wbf_arr)\n'
            '    $mf32b := d.matrix_f32(2, 2) >>>\n    _ := mf32b.write_bytes(&wbf) >>>\n'
            '    b_wb32 := mf32b.bits() >>>\n    dump("WB32", &b_wb32)\n    _ := mf32b.close() >>>\n'
            f'    wbf16_arr := [{f16b_ints}]\n    wbf16 := to_bytes(&wbf16_arr)\n'
            '    $mf16b := d.matrix_f16(2, 2) >>>\n    _ := mf16b.write_bytes(&wbf16) >>>\n'
            '    b_wb16 := mf16b.bits() >>>\n    dump("WB16", &b_wb16)\n    _ := mf16b.close() >>>\n'
            + xu_arr("xq8", x_q8)
            + f'    q8_arr := [{q8_hex}]\n    q8_bytes := to_bytes(&q8_arr)\n'
            + '    $mxq8 := mat(&d, 2, 64, &xq8_b) >>>\n'
            + '    $mwq8 := d.matrix_q8_0(4, 64) >>>\n    _ := mwq8.write_bytes(&q8_bytes) >>>\n'
            + '    $mdq8 := d.matrix_f32(2, 4) >>>\n'
            + '    $fq8 := metal.gemv_q8_0_f32($&mdq8, &mxq8, &mwq8) >>>\n    _ := fq8.wait() >>>\n    _ := fq8.close() >>>\n'
            + '    b_q8 := mdq8.bits() >>>\n    dump("Q8", &b_q8)\n'
            + xu_arr("xq4k", x_q4k)
            + f'    q4k_arr := [{q4k_hex}]\n    q4k_bytes := to_bytes(&q4k_arr)\n'
            + '    $mxq4k := mat(&d, 1, 256, &xq4k_b) >>>\n'
            + '    $mwq4k := d.matrix_q4_k(3, 256) >>>\n    _ := mwq4k.write_bytes(&q4k_bytes) >>>\n'
            + '    $mdq4k := d.matrix_f32(1, 3) >>>\n'
            + '    $fq4k := metal.gemv_q4_k_f32($&mdq4k, &mxq4k, &mwq4k) >>>\n    _ := fq4k.wait() >>>\n    _ := fq4k.close() >>>\n'
            + '    b_q4k := mdq4k.bits() >>>\n    dump("Q4K", &b_q4k)\n'
            + xu_arr("qkx", qk_x) + xu_arr("qkw", qk_w)
            + '    $mqkx := mat(&d, 2, 8, &qkx_b) >>>\n    $mqkw := mat(&d, 1, 4, &qkw_b) >>>\n'
            + '    $mqkn := d.matrix_f32(2, 8) >>>\n'
            + '    $fqkn := metal.qk_norm_f32($&mqkn, &mqkx, &mqkw, 2, 0.00001) >>>\n    _ := fqkn.wait() >>>\n    _ := fqkn.close() >>>\n'
            + '    b_qkn := mqkn.bits() >>>\n    dump("QKN", &b_qkn)\n'
            + xu_arr("dq", dec_q) + xu_arr("kc", kc) + xu_arr("vc", vc)
            + '    $mdq := mat(&d, 1, 16, &dq_b) >>>\n    $mkc := mat(&d, 8, 8, &kc_b) >>>\n    $mvc := mat(&d, 8, 8, &vc_b) >>>\n'
            + '    $mdec := d.matrix_f32(1, 16) >>>\n'
            + '    $fdec := metal.attention_decode_f32($&mdec, &mdq, &mkc, &mvc, 3, 4, 2, 0.5) >>>\n    _ := fdec.wait() >>>\n    _ := fdec.close() >>>\n'
            + '    b_dec := mdec.bits() >>>\n    dump("DEC", &b_dec)\n'
            + xu_arr("tbl", emb_tbl)
            + '    $mtbl := mat(&d, 5, 4, &tbl_b) >>>\n'
            + '    if !guard_emb(&d, &mtbl) { << 9 }\n'
            + '    $memb := d.matrix_f32(3, 4) >>>\n    eids := [3, 0, 4]\n'
            + '    $femb := metal.embedding_f32($&memb, &mtbl, &eids) >>>\n    _ := femb.wait() >>>\n    _ := femb.close() >>>\n'
            + '    b_emb := memb.bits() >>>\n    dump("EMB", &b_emb)\n'
            + xu_arr("amx", am_x)
            + '    $mamx := mat(&d, 2, 4, &amx_b) >>>\n    $mam := d.matrix_f32(2, 1) >>>\n'
            + '    $fam := metal.argmax_f32($&mam, &mamx) >>>\n    _ := fam.wait() >>>\n    _ := fam.close() >>>\n'
            + '    b_am := mam.bits() >>>\n    dump("AM", &b_am)\n'
            + '    _ := mxq8.close() >>>\n    _ := mwq8.close() >>>\n    _ := mdq8.close() >>>\n'
            + '    _ := mxq4k.close() >>>\n    _ := mwq4k.close() >>>\n    _ := mdq4k.close() >>>\n'
            + '    _ := mqkx.close() >>>\n    _ := mqkw.close() >>>\n    _ := mqkn.close() >>>\n'
            + '    _ := mdq.close() >>>\n    _ := mkc.close() >>>\n    _ := mvc.close() >>>\n    _ := mdec.close() >>>\n'
            + '    _ := mtbl.close() >>>\n    _ := memb.close() >>>\n    _ := mamx.close() >>>\n    _ := mam.close() >>>\n'
            + '    _ := d.close() >>>\n'
            + '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            + 'fn main() int { << run_v14() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        self.assertEqual(kv.get("BIG"), "32768x64")
        wb32 = [bits_to_f32(int(x)) for x in kv["WB32"].split(",")]
        self.assertEqual(wb32, [1.5, -2.5, 3.25, 0.0])
        wb16 = [int(x) for x in kv["WB16"].split(",")]
        self.assertEqual(wb16, list(struct.unpack("<4H", f16b_raw)))
        for key, ref in (("Q8", ref_q8), ("Q4K", ref_q4k), ("QKN", ref_qkn), ("DEC", ref_dec), ("EMB", ref_emb), ("AM", ref_am)):
            got = [bits_to_f32(int(x)) for x in kv[key].split(",")]
            self.assertEqual(len(got), len(ref), key)
            for i, (gv, rv) in enumerate(zip(got, ref)):
                self.assertLess(abs(gv - rv), 1e-3, f"{key}[{i}]: gpu={gv} cpu={rv}")

    def test_v14_end_to_end_gpu_llm_decode(self):
        # End-to-end 3-step autoregressive 1-layer GQA decode loop in a Metal Batch with KV cache append
        dim, kv_dim, hidden, vocab = 32, 16, 32, 16
        n_heads, n_kv_heads, head_dim = 4, 2, 8

        def pack_q8(rows, cols, scale):
            src = [f32(((i * 7 + 3) % 15 - 7) * scale) for i in range(rows * cols)]
            raw = bytearray()
            dq = []
            for blk in range(len(src) // 32):
                seg = src[blk * 32:(blk + 1) * 32]
                amax = max(abs(v) for v in seg)
                d = f32(amax / 127.0) if amax > 0 else 0.0
                d16_b = struct.pack("<e", d)
                d16 = f32(struct.unpack("<e", d16_b)[0])
                raw.extend(d16_b)
                inv_d = 1.0 / d16 if d16 != 0 else 0.0
                for v in seg:
                    q = max(-127, min(127, int(round(v * inv_d))))
                    raw.append(q & 0xFF)
                    dq.append(f32(q * d16))
            return raw, dq

        embed_tbl = [f32(((i * 11 + 1) % 13 - 6) * 0.125) for i in range(vocab * dim)]
        wn1 = [f32(1.0 + (i % 5) * 0.05) for i in range(dim)]
        wn2 = [f32(1.0 - (i % 4) * 0.05) for i in range(dim)]
        wn_f = [f32(1.0) for _ in range(dim)]
        qnorm_w = [f32(1.1) for _ in range(head_dim)]
        knorm_w = [f32(0.9) for _ in range(head_dim)]

        wq_b, wq_dq = pack_q8(dim, dim, 0.05)
        wk_b, wk_dq = pack_q8(kv_dim, dim, 0.04)
        wv_b, wv_dq = pack_q8(kv_dim, dim, 0.05)
        wo_b, wo_dq = pack_q8(dim, dim, 0.05)
        wg_b, wg_dq = pack_q8(hidden, dim, 0.05)
        wu_b, wu_dq = pack_q8(hidden, dim, 0.04)
        wd_b, wd_dq = pack_q8(dim, hidden, 0.05)
        whead_b, whead_dq = pack_q8(vocab, dim, 0.15)

        def cpu_rmsnorm(v, w):
            inv = 1.0 / math.sqrt(sum(x * x for x in v) / len(v) + 1e-5)
            return [v[i] * inv * w[i] for i in range(len(v))]

        def cpu_qknorm(v, w, nh):
            out = []
            for h in range(nh):
                out.extend(cpu_rmsnorm(v[h * head_dim:(h + 1) * head_dim], w))
            return out

        def cpu_rope(v, nh, pos):
            out = list(v)
            half = head_dim // 2
            lt = math.log(10000.0)
            for h in range(nh):
                base = h * head_dim
                for j in range(half):
                    ang = pos * math.exp(-lt * (2.0 * j) / head_dim)
                    c, s = math.cos(ang), math.sin(ang)
                    v0, v1 = out[base + j], out[base + half + j]
                    out[base + j] = v0 * c - v1 * s
                    out[base + half + j] = v0 * s + v1 * c
            return out

        def cpu_matvec(x, w_dq, rows, cols):
            return [sum(x[c] * w_dq[r * cols + c] for c in range(cols)) for r in range(rows)]

        k_hist, v_hist = [], []
        cur_tok = 2
        cpu_tokens = []
        for pos in range(3):
            x = embed_tbl[cur_tok * dim:(cur_tok + 1) * dim]
            n1 = cpu_rmsnorm(x, wn1)
            q = cpu_rope(cpu_qknorm(cpu_matvec(n1, wq_dq, dim, dim), qnorm_w, n_heads), n_heads, pos)
            k = cpu_rope(cpu_qknorm(cpu_matvec(n1, wk_dq, kv_dim, dim), knorm_w, n_kv_heads), n_kv_heads, pos)
            v = cpu_matvec(n1, wv_dq, kv_dim, dim)
            k_hist.append(k)
            v_hist.append(v)
            att = []
            sc = 1.0 / math.sqrt(head_dim)
            for h in range(n_heads):
                kvh = h // 2
                qh = q[h * head_dim:(h + 1) * head_dim]
                scores = [sum(qh[d] * k_hist[t][kvh * head_dim + d] for d in range(head_dim)) * sc for t in range(pos + 1)]
                mx = max(scores)
                ex = [math.exp(sv - mx) for sv in scores]
                sm = sum(ex)
                for d in range(head_dim):
                    att.append(sum((ex[t] / sm) * v_hist[t][kvh * head_dim + d] for t in range(pos + 1)))
            h_att = [x[i] + o for i, o in enumerate(cpu_matvec(att, wo_dq, dim, dim))]
            n2 = cpu_rmsnorm(h_att, wn2)
            g = cpu_matvec(n2, wg_dq, hidden, dim)
            u = cpu_matvec(n2, wu_dq, hidden, dim)
            sw = [(gv / (1.0 + math.exp(-gv))) * uv for gv, uv in zip(g, u)]
            hout = [h_att[i] + o for i, o in enumerate(cpu_matvec(sw, wd_dq, dim, hidden))]
            nf = cpu_rmsnorm(hout, wn_f)
            logits = cpu_matvec(nf, whead_dq, vocab, dim)
            nxt = max(range(vocab), key=lambda idx: logits[idx])
            cpu_tokens.append(nxt)
            cur_tok = nxt

        def xu_f32_arr(name, vals):
            return f"    {name}_b := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        def xu_bytes_def(name, raw):
            ints = ", ".join(str(b) for b in raw)
            return f"    {name}_arr := [{ints}]\n    {name}_bs := to_bytes(&{name}_arr)\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n#math\n'
            'fn to_bytes(vals &[]int) string {\n'
            '    $out := ""\n    for i in 0:vals.len { out.push_str(string_from_byte(vals[i] & 255)) }\n'
            '    << out\n}\n'
            'fn mat(d &metal.Device, r int, c int, b &[]int) metal.F32Matrix {\n'
            '    $m := d.matrix_f32(r, c) >>>\n    _ := m.write_bits(b) >>>\n    << m\n}\n'
            'fn q8(d &metal.Device, r int, c int, bs &string) metal.Q8Matrix {\n'
            '    $m := d.matrix_q8_0(r, c) >>>\n    _ := m.write_bytes(bs) >>>\n    << m\n}\n'
            'fn run_decode() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_f32_arr("et", embed_tbl) + xu_f32_arr("wn1", wn1) + xu_f32_arr("wn2", wn2)
            + xu_f32_arr("wnf", wn_f) + xu_f32_arr("qnw", qnorm_w) + xu_f32_arr("knw", knorm_w)
            + xu_bytes_def("wq", wq_b) + xu_bytes_def("wk", wk_b) + xu_bytes_def("wv", wv_b)
            + xu_bytes_def("wo", wo_b) + xu_bytes_def("wg", wg_b) + xu_bytes_def("wu", wu_b)
            + xu_bytes_def("wd", wd_b) + xu_bytes_def("wh", whead_b)
            + '    $met := mat(&d, 16, 32, &et_b) >>>\n'
            + '    $mwn1 := mat(&d, 1, 32, &wn1_b) >>>\n    $mwn2 := mat(&d, 1, 32, &wn2_b) >>>\n'
            + '    $mwnf := mat(&d, 1, 32, &wnf_b) >>>\n    $mqnw := mat(&d, 1, 8, &qnw_b) >>>\n    $mknw := mat(&d, 1, 8, &knw_b) >>>\n'
            + '    $mwq := q8(&d, 32, 32, &wq_bs) >>>\n    $mwk := q8(&d, 16, 32, &wk_bs) >>>\n    $mwv := q8(&d, 16, 32, &wv_bs) >>>\n'
            + '    $mwo := q8(&d, 32, 32, &wo_bs) >>>\n    $mwg := q8(&d, 32, 32, &wg_bs) >>>\n    $mwu := q8(&d, 32, 32, &wu_bs) >>>\n'
            + '    $mwd := q8(&d, 32, 32, &wd_bs) >>>\n    $mwh := q8(&d, 16, 32, &wh_bs) >>>\n'
            + '    $k_cache := d.matrix_f32(8, 16) >>>\n    $v_cache := d.matrix_f32(8, 16) >>>\n'
            + '    $x := d.matrix_f32(1, 32) >>>\n    $n1 := d.matrix_f32(1, 32) >>>\n'
            + '    $qr := d.matrix_f32(1, 32) >>>\n    $qn := d.matrix_f32(1, 32) >>>\n    $q := d.matrix_f32(1, 32) >>>\n'
            + '    $kr := d.matrix_f32(1, 16) >>>\n    $kn := d.matrix_f32(1, 16) >>>\n    $k := d.matrix_f32(1, 16) >>>\n'
            + '    $v := d.matrix_f32(1, 16) >>>\n    $att := d.matrix_f32(1, 32) >>>\n    $oproj := d.matrix_f32(1, 32) >>>\n'
            + '    $hatt := d.matrix_f32(1, 32) >>>\n    $n2 := d.matrix_f32(1, 32) >>>\n'
            + '    $fg := d.matrix_f32(1, 32) >>>\n    $fu := d.matrix_f32(1, 32) >>>\n    $fsw := d.matrix_f32(1, 32) >>>\n'
            + '    $ffn := d.matrix_f32(1, 32) >>>\n    $hout := d.matrix_f32(1, 32) >>>\n'
            + '    $nf := d.matrix_f32(1, 32) >>>\n    $logits := d.matrix_f32(1, 16) >>>\n    $idxm := d.matrix_f32(1, 1) >>>\n'
            + '    $tok := 2\n    $out_toks := []int{}\n'
            + '    scale := 1.0 / math.sqrt(8.0)\n'
            + '    for pos in 0:3 {\n'
            + '        ids := [tok]\n'
            + '        $b := d.begin_batch() >>>\n'
            + '        _ := b.embedding_f32($&x, &met, &ids) >>>\n'
            + '        _ := b.rmsnorm_f32($&n1, &x, &mwn1, 0.00001) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&qr, &n1, &mwq) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&kr, &n1, &mwk) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&v, &n1, &mwv) >>>\n'
            + '        _ := b.qk_norm_f32($&qn, &qr, &mqnw, 4, 0.00001) >>>\n'
            + '        _ := b.qk_norm_f32($&kn, &kr, &mknw, 2, 0.00001) >>>\n'
            + '        _ := b.rope_f32($&q, &qn, pos, 8, 10000.0) >>>\n'
            + '        _ := b.rope_f32($&k, &kn, pos, 8, 10000.0) >>>\n'
            + '        _ := b.kv_cache_append_f32($&k_cache, &k, pos) >>>\n'
            + '        _ := b.kv_cache_append_f32($&v_cache, &v, pos) >>>\n'
            + '        _ := b.attention_decode_f32($&att, &q, &k_cache, &v_cache, pos + 1, 4, 2, scale) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&oproj, &att, &mwo) >>>\n'
            + '        _ := b.add_f32($&hatt, &x, &oproj) >>>\n'
            + '        _ := b.rmsnorm_f32($&n2, &hatt, &mwn2, 0.00001) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&fg, &n2, &mwg) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&fu, &n2, &mwu) >>>\n'
            + '        _ := b.swiglu_f32($&fsw, &fg, &fu) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&ffn, &fsw, &mwd) >>>\n'
            + '        _ := b.add_f32($&hout, &hatt, &ffn) >>>\n'
            + '        _ := b.rmsnorm_f32($&nf, &hout, &mwnf, 0.00001) >>>\n'
            + '        _ := b.gemv_q8_0_f32($&logits, &nf, &mwh) >>>\n'
            + '        _ := b.argmax_f32($&idxm, &logits) >>>\n'
            + '        $f := b.commit() >>>\n'
            + '        _ := f.wait() >>>\n'
            + '        _ := f.close() >>>\n'
            + '        b0 := idxm.bits() >>>\n'
            + '        tok = int(math.float32_from_bits(b0[0]) + 0.5)\n'
            + '        out_toks.push(tok)\n'
            + '    }\n'
            + '    $ts := "TOKS:"\n    for i in 0:out_toks.len { if i > 0 { ts = ts + "," }\n ts = ts + str(out_toks[i]) }\n    println(ts)\n'
            + '    _ := met.close() >>>\n    _ := mwn1.close() >>>\n    _ := mwn2.close() >>>\n    _ := mwnf.close() >>>\n'
            + '    _ := mqnw.close() >>>\n    _ := mknw.close() >>>\n'
            + '    _ := mwq.close() >>>\n    _ := mwk.close() >>>\n    _ := mwv.close() >>>\n    _ := mwo.close() >>>\n'
            + '    _ := mwg.close() >>>\n    _ := mwu.close() >>>\n    _ := mwd.close() >>>\n    _ := mwh.close() >>>\n'
            + '    _ := k_cache.close() >>>\n    _ := v_cache.close() >>>\n'
            + '    _ := x.close() >>>\n    _ := n1.close() >>>\n    _ := qr.close() >>>\n    _ := qn.close() >>>\n    _ := q.close() >>>\n'
            + '    _ := kr.close() >>>\n    _ := kn.close() >>>\n    _ := k.close() >>>\n    _ := v.close() >>>\n'
            + '    _ := att.close() >>>\n    _ := oproj.close() >>>\n    _ := hatt.close() >>>\n    _ := n2.close() >>>\n'
            + '    _ := fg.close() >>>\n    _ := fu.close() >>>\n    _ := fsw.close() >>>\n    _ := ffn.close() >>>\n    _ := hout.close() >>>\n'
            + '    _ := nf.close() >>>\n    _ := logits.close() >>>\n    _ := idxm.close() >>>\n'
            + '    _ := d.close() >>>\n'
            + '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            + 'fn main() int { << run_decode() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        gpu_toks = [int(x) for x in kv["TOKS"].split(",")]
        self.assertEqual(gpu_toks, cpu_tokens)

    def test_q5_q6k_interleaved_rope_and_f16_flash_decode(self):
        # 1. CPU quantizers matching GGML Q5_0 (22B / 32) and Q6_K (210B / 256)
        def pack_q5_0(vals):
            raw = bytearray()
            dq = []
            for b in range(len(vals) // 32):
                seg = vals[b * 32:(b + 1) * 32]
                amax = max(abs(v) for v in seg)
                d = f32(amax / 15.0) if amax > 0 else 0.0
                d_b = struct.pack("<e", d)
                d16 = f32(struct.unpack("<e", d_b)[0])
                inv = 1.0 / d16 if d16 != 0 else 0.0
                qs = [min(31, max(0, int(round(v * inv)) + 16)) for v in seg]
                qh = 0
                for j in range(32):
                    if qs[j] & 16:
                        qh |= 1 << j
                raw.extend(d_b)
                raw.extend(struct.pack("<I", qh))
                for j in range(16):
                    raw.append((qs[j] & 15) | ((qs[16 + j] & 15) << 4))
                dq.extend(f32((q - 16) * d16) for q in qs)
            return bytes(raw), dq

        def pack_q6_k(vals):
            raw = bytearray()
            dq = []
            for b in range(len(vals) // 256):
                blk = vals[b * 256:(b + 1) * 256]
                amax_subs = [max(abs(v) for v in blk[s * 16:(s + 1) * 16]) for s in range(16)]
                max_sub = max(amax_subs)
                d = f32(max_sub / (31.0 * 127.0)) if max_sub > 0 else 0.0
                d_b = struct.pack("<e", d)
                d16 = f32(struct.unpack("<e", d_b)[0])
                sc_q = [min(127, max(1, int(round(a / (31.0 * d16))))) if (d16 > 0 and a > 0) else 0 for a in amax_subs]
                ql, qh = bytearray(128), bytearray(64)
                dq_blk = [0.0] * 256
                for h in range(2):
                    y_off, ql_off, qh_off, sc_off = h * 128, h * 64, h * 32, h * 8
                    for l in range(32):
                        is_idx = l // 16
                        qs_l = []
                        for quad, off_s in enumerate((0, 32, 64, 96)):
                            idx = y_off + l + off_s
                            scv = d16 * sc_q[sc_off + is_idx + quad * 2]
                            q = min(31, max(-32, int(round(blk[idx] / scv)))) if scv != 0 else 0
                            qs_l.append(q + 32)
                            dq_blk[idx] = f32(scv * q)
                        q1, q2, q3, q4 = qs_l
                        ql[ql_off + l] = (q1 & 15) | ((q3 & 15) << 4)
                        ql[ql_off + l + 32] = (q2 & 15) | ((q4 & 15) << 4)
                        qh[qh_off + l] = ((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) | (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6)
                raw.extend(ql)
                raw.extend(qh)
                raw.extend(struct.pack("<16b", *sc_q))
                raw.extend(d_b)
                dq.extend(dq_blk)
            return bytes(raw), dq

        cols, rows_q = 256, 3
        x_q = [f32(((i * 7 + 3) % 19 - 9) * 0.25) for i in range(2 * cols)]
        w5_src = [f32(((i * 5 + 1) % 17 - 8) * 0.3) for i in range(rows_q * cols)]
        w6_src = [f32(((i * 11 + 2) % 23 - 11) * 0.2) for i in range(rows_q * cols)]
        w5_raw, w5_dq = pack_q5_0(w5_src)
        w6_raw, w6_dq = pack_q6_k(w6_src)
        ref_q5 = [f32(sum(x_q[m * cols + c] * w5_dq[r * cols + c] for c in range(cols))) for m in range(2) for r in range(rows_q)]
        ref_q6 = [f32(sum(x_q[m * cols + c] * w6_dq[r * cols + c] for c in range(cols))) for m in range(2) for r in range(rows_q)]

        # Interleaved RoPE [3, 16] (2 heads * 8 dim, pos_offset=2)
        x_ri = [f32(((i * 3 + 2) % 15 - 7) * 0.3) for i in range(3 * 16)]
        ref_ri = []
        lt = math.log(10000.0)
        for r in range(3):
            pos = 2 + r
            for h in range(2):
                base = r * 16 + h * 8
                for j in range(4):
                    ang = pos * math.exp(-lt * (2.0 * j) / 8.0)
                    c, s = math.cos(ang), math.sin(ang)
                    v0, v1 = x_ri[base + 2 * j], x_ri[base + 2 * j + 1]
                    ref_ri.extend([f32(v0 * c - v1 * s), f32(v0 * s + v1 * c)])

        # F16 KV cache append + FlashDecode GQA: seq_len kv=4, 4 heads, 2 kv_heads, head_dim=8
        k_kv = [f32(((i * 5 + 1) % 13 - 6) * 0.25) for i in range(4 * 16)]
        v_kv = [f32(((i * 7 + 3) % 11 - 5) * 0.25) for i in range(4 * 16)]
        q_dec = [f32(((i * 3 + 4) % 17 - 8) * 0.2) for i in range(32)]
        k16 = [f32(struct.unpack("<e", struct.pack("<e", v))[0]) for v in k_kv]
        v16 = [f32(struct.unpack("<e", struct.pack("<e", v))[0]) for v in v_kv]
        sc_dec = 1.0 / math.sqrt(8.0)
        ref_dec16 = []
        for h in range(4):
            kvh = h // 2
            qh = q_dec[h * 8:(h + 1) * 8]
            scores = [sum(qh[d] * k16[t * 16 + kvh * 8 + d] for d in range(8)) * sc_dec for t in range(4)]
            mx = max(scores)
            ex = [math.exp(sv - mx) for sv in scores]
            sm = sum(ex)
            for d in range(8):
                ref_dec16.append(f32(sum((ex[t] / sm) * v16[t * 16 + kvh * 8 + d] for t in range(4))))

        def xu_f32_list(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        def xu_u8_str(name, bs):
            return f"    {name}_a := [{', '.join(str(b) for b in bs)}]\n    {name} := to_bytes(&{name}_a)\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n#math\n'
            'fn to_bytes(vals &[]int) string {\n'
            '    $out := ""\n    for i in 0:vals.len { out.push_str(string_from_byte(vals[i] & 255)) }\n    << out\n}\n'
            'fn dump(tag string, bits &[]int) {\n'
            '    $s := tag + ":"\n    for i in 0:bits.len { if i > 0 { s = s + "," }\n s = s + str(bits[i]) }\n    println(s)\n}\n'
            'fn run_v15() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_f32_list("xq_b", x_q) + xu_f32_list("xri_b", x_ri)
            + xu_f32_list("kkv_b", k_kv) + xu_f32_list("vkv_b", v_kv) + xu_f32_list("qdec_b", q_dec)
            + xu_u8_str("w5_s", w5_raw) + xu_u8_str("w6_s", w6_raw)
            + '    $mxq := d.matrix_f32(2, 256) >>>\n    _ := mxq.write_bits(&xq_b) >>>\n'
            '    $mw5 := d.matrix_q5_0(3, 256) >>>\n    _ := mw5.write_bytes(&w5_s) >>>\n'
            '    $mw6 := d.matrix_q6_k(3, 256) >>>\n    _ := mw6.write_bytes(&w6_s) >>>\n'
            '    $mo5 := d.matrix_f32(2, 3) >>>\n    $mo6 := d.matrix_f32(2, 3) >>>\n'
            '    $mri_in := d.matrix_f32(3, 16) >>>\n    _ := mri_in.write_bits(&xri_b) >>>\n'
            '    $mri_out := d.matrix_f32(3, 16) >>>\n'
            '    $mk_in := d.matrix_f32(4, 16) >>>\n    _ := mk_in.write_bits(&kkv_b) >>>\n'
            '    $mv_in := d.matrix_f32(4, 16) >>>\n    _ := mv_in.write_bits(&vkv_b) >>>\n'
            '    $mk16 := d.matrix_f16(8, 16) >>>\n    $mv16 := d.matrix_f16(8, 16) >>>\n'
            '    $mqdec := d.matrix_f32(1, 32) >>>\n    _ := mqdec.write_bits(&qdec_b) >>>\n'
            '    $mdec := d.matrix_f32(1, 32) >>>\n'
            '    $b := d.begin_batch() >>>\n'
            '    _ := b.gemv_q5_0_f32($&mo5, &mxq, &mw5) >>>\n'
            '    _ := b.gemv_q6_k_f32($&mo6, &mxq, &mw6) >>>\n'
            '    _ := b.rope_interleaved_f32($&mri_out, &mri_in, 2, 8, 10000.0) >>>\n'
            '    _ := b.kv_cache_append_f16($&mk16, &mk_in, 0) >>>\n'
            '    _ := b.kv_cache_append_f16($&mv16, &mv_in, 0) >>>\n'
            '    _ := b.attention_decode_f16($&mdec, &mqdec, &mk16, &mv16, 4, 4, 2, 1.0 / math.sqrt(8.0)) >>>\n'
            '    $f := b.commit() >>>\n    _ := f.wait() >>>\n    _ := f.close() >>>\n'
            '    b5 := mo5.bits() >>>\n    b6 := mo6.bits() >>>\n'
            '    bri := mri_out.bits() >>>\n    bdec := mdec.bits() >>>\n'
            '    dump("Q5", &b5)\n    dump("Q6K", &b6)\n    dump("ROPE_INT", &bri)\n    dump("DEC16", &bdec)\n'
            '    _ := mxq.close() >>>\n    _ := mw5.close() >>>\n    _ := mw6.close() >>>\n'
            '    _ := mo5.close() >>>\n    _ := mo6.close() >>>\n    _ := mri_in.close() >>>\n    _ := mri_out.close() >>>\n'
            '    _ := mk_in.close() >>>\n    _ := mv_in.close() >>>\n    _ := mk16.close() >>>\n    _ := mv16.close() >>>\n'
            '    _ := mqdec.close() >>>\n    _ := mdec.close() >>>\n    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_v15() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        for tag, ref, tol in (
            ("Q5", ref_q5, 1e-4),
            ("Q6K", ref_q6, 1e-4),
            ("ROPE_INT", ref_ri, 1e-5),
            ("DEC16", ref_dec16, 1e-4),
        ):
            got = [bits_to_f32(int(x)) for x in kv[tag].split(",")]
            self.assertEqual(len(got), len(ref), tag)
            for idx, (g, r) in enumerate(zip(got, ref)):
                self.assertAlmostEqual(g, r, delta=tol, msg=f"{tag}[{idx}]: got={g} ref={r}")

    def test_v16_flash_and_moe_ops(self):
        # 1. Fused Add+RMSNorm [2, 19], GeGLU [2, 19], SiLU [2, 19], LogSoftmax [2, 19]
        rows, cols = 2, 19
        a_v = [f32(((i * 7 + 1) % 15 - 7) * 0.25) for i in range(rows * cols)]
        b_v = [f32(((i * 5 + 3) % 13 - 6) * 0.3) for i in range(rows * cols)]
        w_v = [f32(0.5 + (i % 5) * 0.2) for i in range(cols)]
        eps = 1e-5
        ref_sum = [f32(x + y) for x, y in zip(a_v, b_v)]
        ref_norm = []
        for r in range(rows):
            row = ref_sum[r * cols:(r + 1) * cols]
            inv = 1.0 / math.sqrt(sum(v * v for v in row) / cols + eps)
            ref_norm.extend(f32(v * inv * w_v[c]) for c, v in enumerate(row))
        ref_geglu = [
            f32(0.5 * g * (1.0 + math.tanh(0.7978845608 * (g + 0.044715 * g * g * g))) * u)
            for g, u in zip(a_v, b_v)
        ]
        ref_silu = [f32(v / (1.0 + math.exp(-v))) for v in a_v]
        ref_lsm = []
        for r in range(rows):
            row = a_v[r * cols:(r + 1) * cols]
            mx = max(row)
            lz = mx + math.log(sum(math.exp(v - mx) for v in row))
            ref_lsm.extend(f32(v - lz) for v in row)

        # 2. Long-sequence FlashAttention F32 & Causal Attention F16 (seq_len=300 > 256, 2 heads, 1 kv head, head_dim=8)
        seq_len, nh, nkvh, hd = 300, 2, 1, 8
        sc_attn = 1.0 / math.sqrt(hd)
        q_fa = [f32(((i * 11 + 3) % 17 - 8) * 0.1) for i in range(seq_len * nh * hd)]
        k_fa = [f32(((i * 7 + 5) % 19 - 9) * 0.1) for i in range(seq_len * nkvh * hd)]
        v_fa = [f32(((i * 13 + 1) % 15 - 7) * 0.1) for i in range(seq_len * nkvh * hd)]
        k16_fa = [f32(struct.unpack("<e", struct.pack("<e", x))[0]) for x in k_fa]
        v16_fa = [f32(struct.unpack("<e", struct.pack("<e", x))[0]) for x in v_fa]
        # Verify rows 0, 150, 299 of F32 & F16 FlashAttention on CPU
        check_rows = (0, 150, 299)
        ref_fa32_rows, ref_fa16_rows = [], []
        for r in check_rows:
            for h in range(nh):
                qh = q_fa[(r * nh + h) * hd:(r * nh + h + 1) * hd]
                for ref_dst, ksrc, vsrc in ((ref_fa32_rows, k_fa, v_fa), (ref_fa16_rows, k16_fa, v16_fa)):
                    scores = [sum(qh[d] * ksrc[t * hd + d] for d in range(hd)) * sc_attn for t in range(r + 1)]
                    mx = max(scores)
                    ex = [math.exp(s - mx) for s in scores]
                    sm = sum(ex)
                    for d in range(hd):
                        ref_dst.append(f32(sum((ex[t] / sm) * vsrc[t * hd + d] for t in range(r + 1))))

        # 3. MoE TopK [2, 4] top-2 softmax/normalized + MoE weighted add ([2*2, 5] x [2, 2] -> [2, 5])
        moe_logits = [f32(0.1), f32(2.0), f32(-1.0), f32(1.0),
                      f32(3.0), f32(0.5), f32(2.0), f32(-0.5)]
        moe_eouts = [f32(((i * 5 + 2) % 11 - 5) * 0.25) for i in range(4 * 5)]
        ref_idx, ref_w = [], []
        for r in range(2):
            row = moe_logits[r * 4:(r + 1) * 4]
            mx = max(row)
            ex = [math.exp(x - mx) for x in row]
            sm = sum(ex)
            probs = [f32(x / sm) for x in ex]
            picked = sorted(range(4), key=lambda idx: -probs[idx])[:2]
            wsum = sum(probs[i] for i in picked)
            ref_idx.extend([float(i) for i in picked])
            ref_w.extend([f32(probs[i] / wsum * 1.5) for i in picked])
        ref_moe_out = []
        for r in range(2):
            for c in range(5):
                acc = sum(ref_w[r * 2 + j] * moe_eouts[(r * 2 + j) * 5 + c] for j in range(2))
                ref_moe_out.append(f32(acc))

        def xu_f32_list(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n#math\n'
            'fn dump(tag string, bits &[]int) {\n'
            '    $s := tag + ":"\n    for i in 0:bits.len { if i > 0 { s = s + "," }\n s = s + str(bits[i]) }\n    println(s)\n}\n'
            'fn run_v16() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_f32_list("av_b", a_v) + xu_f32_list("bv_b", b_v) + xu_f32_list("wv_b", w_v)
            + xu_f32_list("qfa_b", q_fa) + xu_f32_list("kfa_b", k_fa) + xu_f32_list("vfa_b", v_fa)
            + xu_f32_list("mlog_b", moe_logits) + xu_f32_list("meo_b", moe_eouts)
            + '    $ma := d.matrix_f32(2, 19) >>>\n    _ := ma.write_bits(&av_b) >>>\n'
            '    $mb := d.matrix_f32(2, 19) >>>\n    _ := mb.write_bits(&bv_b) >>>\n'
            '    $mw := d.matrix_f32(1, 19) >>>\n    _ := mw.write_bits(&wv_b) >>>\n'
            '    $mdnorm := d.matrix_f32(2, 19) >>>\n    $mdsum := d.matrix_f32(2, 19) >>>\n'
            '    $mgel := d.matrix_f32(2, 19) >>>\n    $msil := d.matrix_f32(2, 19) >>>\n    $mlsm := d.matrix_f32(2, 19) >>>\n'
            '    $mqfa := d.matrix_f32(300, 16) >>>\n    _ := mqfa.write_bits(&qfa_b) >>>\n'
            '    $mkfa := d.matrix_f32(300, 8) >>>\n    _ := mkfa.write_bits(&kfa_b) >>>\n'
            '    $mvfa := d.matrix_f32(300, 8) >>>\n    _ := mvfa.write_bits(&vfa_b) >>>\n'
            '    $mk16 := d.matrix_f16(300, 8) >>>\n    $mv16 := d.matrix_f16(300, 8) >>>\n'
            '    $mfa32 := d.matrix_f32(300, 16) >>>\n    $mfa16 := d.matrix_f32(300, 16) >>>\n'
            '    $mlog := d.matrix_f32(2, 4) >>>\n    _ := mlog.write_bits(&mlog_b) >>>\n'
            '    $moidx := d.matrix_f32(2, 2) >>>\n    $mow := d.matrix_f32(2, 2) >>>\n'
            '    $meo := d.matrix_f32(4, 5) >>>\n    _ := meo.write_bits(&meo_b) >>>\n'
            '    $mmout := d.matrix_f32(2, 5) >>>\n'
            '    // Immediate calls first\n'
            '    $f0 := metal.add_rmsnorm_f32($&mdnorm, $&mdsum, &ma, &mb, &mw, 1e-5) >>>\n    _ := f0.wait() >>>\n    _ := f0.close() >>>\n'
            '    $f1 := metal.geglu_f32($&mgel, &ma, &mb) >>>\n    _ := f1.wait() >>>\n    _ := f1.close() >>>\n'
            '    $f2 := metal.silu_f32($&msil, &ma) >>>\n    _ := f2.wait() >>>\n    _ := f2.close() >>>\n'
            '    $f3 := metal.log_softmax_f32($&mlsm, &ma) >>>\n    _ := f3.wait() >>>\n    _ := f3.close() >>>\n'
            '    $f4 := metal.causal_attention_f32($&mfa32, &mqfa, &mkfa, &mvfa, 2, 1, 1.0 / math.sqrt(8.0)) >>>\n    _ := f4.wait() >>>\n    _ := f4.close() >>>\n'
            '    // Batched calls exercising all 7 new batch wrappers + f32_to_f16 + causal_attention_f16\n'
            '    $b := d.begin_batch() >>>\n'
            '    _ := b.f32_to_f16($&mk16, &mkfa) >>>\n'
            '    _ := b.f32_to_f16($&mv16, &mvfa) >>>\n'
            '    _ := b.causal_attention_f16($&mfa16, &mqfa, &mk16, &mv16, 2, 1, 1.0 / math.sqrt(8.0)) >>>\n'
            '    _ := b.add_rmsnorm_f32($&mdnorm, $&mdsum, &ma, &mb, &mw, 1e-5) >>>\n'
            '    _ := b.geglu_f32($&mgel, &ma, &mb) >>>\n'
            '    _ := b.silu_f32($&msil, &ma) >>>\n'
            '    _ := b.log_softmax_f32($&mlsm, &ma) >>>\n'
            '    _ := b.moe_topk_f32($&moidx, $&mow, &mlog, 2, false, true, 1.5) >>>\n'
            '    _ := b.moe_weighted_add_f32($&mmout, &meo, &mow) >>>\n'
            '    $fb := b.commit() >>>\n    _ := fb.wait() >>>\n    _ := fb.close() >>>\n'
            '    $fm0 := metal.moe_topk_f32($&moidx, $&mow, &mlog, 2, false, true, 1.5) >>>\n    _ := fm0.wait() >>>\n    _ := fm0.close() >>>\n'
            '    $fm1 := metal.moe_weighted_add_f32($&mmout, &meo, &mow) >>>\n    _ := fm1.wait() >>>\n    _ := fm1.close() >>>\n'
            '    bdn := mdnorm.bits() >>>\n    bds := mdsum.bits() >>>\n'
            '    bgl := mgel.bits() >>>\n    bsi := msil.bits() >>>\n    bls := mlsm.bits() >>>\n'
            '    bfa32 := mfa32.bits() >>>\n    bfa16 := mfa16.bits() >>>\n'
            '    bidx := moidx.bits() >>>\n    bw := mow.bits() >>>\n    bmo := mmout.bits() >>>\n'
            '    dump("DN", &bdn)\n    dump("DS", &bds)\n    dump("GEGLU", &bgl)\n    dump("SILU", &bsi)\n    dump("LSM", &bls)\n'
            '    dump("FA32", &bfa32)\n    dump("FA16", &bfa16)\n    dump("MOE_I", &bidx)\n    dump("MOE_W", &bw)\n    dump("MOE_O", &bmo)\n'
            '    _ := ma.close() >>>\n    _ := mb.close() >>>\n    _ := mw.close() >>>\n'
            '    _ := mdnorm.close() >>>\n    _ := mdsum.close() >>>\n    _ := mgel.close() >>>\n    _ := msil.close() >>>\n    _ := mlsm.close() >>>\n'
            '    _ := mqfa.close() >>>\n    _ := mkfa.close() >>>\n    _ := mvfa.close() >>>\n    _ := mk16.close() >>>\n    _ := mv16.close() >>>\n'
            '    _ := mfa32.close() >>>\n    _ := mfa16.close() >>>\n    _ := mlog.close() >>>\n    _ := moidx.close() >>>\n    _ := mow.close() >>>\n'
            '    _ := meo.close() >>>\n    _ := mmout.close() >>>\n    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_v16() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        for tag, ref, tol in (
            ("DN", ref_norm, 1e-4),
            ("DS", ref_sum, 1e-5),
            ("GEGLU", ref_geglu, 1e-4),
            ("SILU", ref_silu, 1e-5),
            ("LSM", ref_lsm, 1e-4),
            ("MOE_I", ref_idx, 1e-5),
            ("MOE_W", ref_w, 1e-5),
            ("MOE_O", ref_moe_out, 1e-5),
        ):
            got = [bits_to_f32(int(x)) for x in kv[tag].split(",")]
            self.assertEqual(len(got), len(ref), tag)
            for idx, (g, r) in enumerate(zip(got, ref)):
                self.assertAlmostEqual(g, r, delta=tol, msg=f"{tag}[{idx}]: got={g} ref={r}")
        for tag, ref_rows in (("FA32", ref_fa32_rows), ("FA16", ref_fa16_rows)):
            all_vals = [bits_to_f32(int(x)) for x in kv[tag].split(",")]
            got_rows = []
            for r in check_rows:
                got_rows.extend(all_vals[r * nh * hd:(r + 1) * nh * hd])
            self.assertEqual(len(got_rows), len(ref_rows), tag)
            for idx, (g, r) in enumerate(zip(got_rows, ref_rows)):
                self.assertAlmostEqual(g, r, delta=1e-3, msg=f"{tag}[{idx}]: got={g} ref={r}")

    def test_v17_yarn_and_fused_moe_gpu(self):
        import random
        rng = random.Random(20261207)

        def f32_to_f16_bits(val):
            return struct.unpack("<H", struct.pack("<e", val))[0]

        def f16_bits_to_f32(bits):
            return struct.unpack("<e", struct.pack("<H", bits))[0]

        def cpu_rope_yarn(mat, seq_len, n_heads, head_dim, rope_dim, interleaved,
                          pos_off, theta, scale, orig_ctx, beta_fast, beta_slow, attn_factor):
            rdim = rope_dim if 0 < rope_dim <= head_dim else head_dim
            half_r = rdim // 2
            eff_scale = scale if scale > 0.0 else 1.0
            log_theta = math.log(theta)
            two_pi = 6.283185307179586
            use_ntk = eff_scale > 1.0 and orig_ctx > 0 and beta_fast > 0.0 and beta_slow > 0.0
            low = 0.0
            high = float(half_r - 1)
            if use_ntk:
                low = (rdim * math.log(orig_ctx / (beta_fast * two_pi))) / (2.0 * log_theta)
                high = (rdim * math.log(orig_ctx / (beta_slow * two_pi))) / (2.0 * log_theta)
                low = max(0.0, low)
                high = min(float(half_r - 1), high)
                if high <= low:
                    high = low + 0.001
            mscale = attn_factor if attn_factor > 0.0 else ((0.1 * math.log(eff_scale) + 1.0) if eff_scale > 1.0 else 1.0)
            out = list(mat)
            for r in range(seq_len):
                for h in range(n_heads):
                    base = r * n_heads * head_dim + h * head_dim
                    for j in range(half_r):
                        bf = math.exp(-log_theta * (2.0 * j / float(rdim)))
                        freq = bf / eff_scale
                        if use_ntk:
                            ramp = max(0.0, min(1.0, (j - low) / (high - low)))
                            gamma = 1.0 - ramp
                            freq = (1.0 - gamma) * (bf / eff_scale) + gamma * bf
                        ang = (pos_off + r) * freq
                        co, si = math.cos(ang) * mscale, math.sin(ang) * mscale
                        i0 = base + (2 * j if interleaved else j)
                        i1 = base + (2 * j + 1 if interleaved else half_r + j)
                        v0, v1 = mat[i0], mat[i1]
                        out[i0] = f32(v0 * co - v1 * si)
                        out[i1] = f32(v0 * si + v1 * co)
            return out

        def make_q4_0_bytes(weights_2d):
            raw = bytearray()
            dq = []
            for row in weights_2d:
                assert len(row) % 32 == 0
                dq_row = [0.0] * len(row)
                for b in range(len(row) // 32):
                    blk = row[b * 32:(b + 1) * 32]
                    amax = max(abs(x) for x in blk)
                    d = amax / 7.0
                    d_bits = f32_to_f16_bits(d)
                    d_f32 = f16_bits_to_f32(d_bits)
                    raw += struct.pack("<H", d_bits)
                    inv = (1.0 / d_f32) if d_f32 != 0.0 else 0.0
                    for j in range(16):
                        q0 = max(-8, min(7, int(round(blk[j] * inv)))) + 8
                        q1 = max(-8, min(7, int(round(blk[16 + j] * inv)))) + 8
                        raw.append((q0 & 15) | ((q1 & 15) << 4))
                        dq_row[b * 32 + j] = (q0 - 8) * d_f32
                        dq_row[b * 32 + 16 + j] = (q1 - 8) * d_f32
                dq.append(dq_row)
            return bytes(raw), dq

        def make_q8_0_bytes(weights_2d):
            raw = bytearray()
            dq = []
            for row in weights_2d:
                assert len(row) % 32 == 0
                dq_row = [0.0] * len(row)
                for b in range(len(row) // 32):
                    blk = row[b * 32:(b + 1) * 32]
                    amax = max(abs(x) for x in blk)
                    d = amax / 127.0
                    d_bits = f32_to_f16_bits(d)
                    d_f32 = f16_bits_to_f32(d_bits)
                    raw += struct.pack("<H", d_bits)
                    inv = (1.0 / d_f32) if d_f32 != 0.0 else 0.0
                    for j in range(32):
                        q = max(-127, min(127, int(round(blk[j] * inv))))
                        raw += struct.pack("<b", q)
                        dq_row[b * 32 + j] = q * d_f32
                dq.append(dq_row)
            return bytes(raw), dq

        # YaRN inputs
        rx = [f32(rng.uniform(-1.5, 1.5)) for _ in range(2 * 3 * 8)]
        ref_yarn_split = cpu_rope_yarn(rx, 2, 3, 8, 6, False, 5, 10000.0, 2.0, 32, 32.0, 1.0, 0.0)
        rx2 = [f32(rng.uniform(-1.5, 1.5)) for _ in range(2 * 2 * 8)]
        ref_yarn_il = cpu_rope_yarn(rx2, 2, 2, 8, 8, True, 3, 10000.0, 4.0, 64, 32.0, 1.0, 1.15)

        # Fused MoE Q4_0 (shared x) & Q8_0 (per-expert x)
        m_cnt, n_exp, top_k, in_dim, out_dim = 2, 4, 2, 64, 16
        logits = [f32(rng.uniform(-2.0, 2.0)) for _ in range(m_cnt * n_exp)]
        x_shared = [f32(rng.uniform(-1.0, 1.0)) for _ in range(m_cnt * in_dim)]
        x_per_exp = [f32(rng.uniform(-1.0, 1.0)) for _ in range(m_cnt * top_k * in_dim)]
        w_q4_rows = [[f32(rng.uniform(-0.8, 0.8)) for _ in range(in_dim)] for _ in range(n_exp * out_dim)]
        w_q8_rows = [[f32(rng.uniform(-0.8, 0.8)) for _ in range(in_dim)] for _ in range(n_exp * out_dim)]
        q4_bytes, w_q4_dq = make_q4_0_bytes(w_q4_rows)
        q8_bytes, w_q8_dq = make_q8_0_bytes(w_q8_rows)

        # CPU topk (softmax, normalize=True, scale=1.0)
        ref_moe_q4 = []
        ref_moe_q8 = []
        for m in range(m_cnt):
            row_l = logits[m * n_exp:(m + 1) * n_exp]
            mx = max(row_l)
            ex = [math.exp(v - mx) for v in row_l]
            sm = sum(ex)
            probs = [v / sm for v in ex]
            ranked = sorted(range(n_exp), key=lambda i: (-probs[i], i))[:top_k]
            ws = [probs[i] for i in ranked]
            wsum = sum(ws)
            ws = [w / wsum for w in ws]
            for n in range(out_dim):
                acc4 = 0.0
                acc8 = 0.0
                for ek, eid in enumerate(ranked):
                    ew = ws[ek]
                    xs = x_shared[m * in_dim:(m + 1) * in_dim]
                    xp = x_per_exp[(m * top_k + ek) * in_dim:(m * top_k + ek + 1) * in_dim]
                    w4r = w_q4_dq[eid * out_dim + n]
                    w8r = w_q8_dq[eid * out_dim + n]
                    acc4 += ew * sum(a * b for a, b in zip(xs, w4r))
                    acc8 += ew * sum(a * b for a, b in zip(xp, w8r))
                ref_moe_q4.append(f32(acc4))
                ref_moe_q8.append(f32(acc8))

        def xu_f32_list(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        def xu_u8_bytes(name, raw):
            ints = ", ".join(str(b) for b in raw)
            return f"    {name}_a := [{ints}]\n    {name} := to_bytes(&{name}_a)\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn to_bytes(vals &[]int) string {\n'
            '    $out := ""\n    for i in 0:vals.len { out.push_str(string_from_byte(vals[i] & 255)) }\n    << out\n'
            '}\n'
            'fn dump(t string, v &[]int) {\n'
            '    $s := t + ":"\n    for i in 0:v.len { if i > 0 { s = s + "," }\n s = s + str(v[i]) }\n    println(s)\n'
            '}\n'
            'fn run_v17() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_f32_list("rx_b", rx) + xu_f32_list("rx2_b", rx2)
            + xu_f32_list("log_b", logits) + xu_f32_list("xs_b", x_shared) + xu_f32_list("xp_b", x_per_exp)
            + xu_u8_bytes("q4_raw", q4_bytes) + xu_u8_bytes("q8_raw", q8_bytes)
            + '    $mrx := d.matrix_f32(2, 24) >>>\n    _ := mrx.write_bits(&rx_b) >>>\n'
            '    $mry1 := d.matrix_f32(2, 24) >>>\n'
            '    $mrx2 := d.matrix_f32(2, 16) >>>\n    _ := mrx2.write_bits(&rx2_b) >>>\n'
            '    $mry2 := d.matrix_f32(2, 16) >>>\n'
            '    $mlog := d.matrix_f32(2, 4) >>>\n    _ := mlog.write_bits(&log_b) >>>\n'
            '    $mxs := d.matrix_f32(2, 64) >>>\n    _ := mxs.write_bits(&xs_b) >>>\n'
            '    $mxp := d.matrix_f32(4, 64) >>>\n    _ := mxp.write_bits(&xp_b) >>>\n'
            '    $mwq4 := d.matrix_q4_0(64, 64) >>>\n    _ := mwq4.write_bytes(&q4_raw) >>>\n'
            '    $mwq8 := d.matrix_q8_0(64, 64) >>>\n    _ := mwq8.write_bytes(&q8_raw) >>>\n'
            '    $meids := d.matrix_f32(2, 2) >>>\n    $mews := d.matrix_f32(2, 2) >>>\n'
            '    $mdq4 := d.matrix_f32(2, 16) >>>\n    $mdq8 := d.matrix_f32(2, 16) >>>\n'
            '    $mdq4_b := d.matrix_f32(2, 16) >>>\n    $mdq8_b := d.matrix_f32(2, 16) >>>\n'
            '    $f0 := metal.rope_yarn_f32($&mry1, &mrx, 5, 8, 6, false, 10000.0, 2.0, 32, 32.0, 1.0, 0.0) >>>\n'
            '    _ := f0.wait() >>>\n    _ := f0.close() >>>\n'
            '    $b := d.begin_batch() >>>\n'
            '    _ := b.rope_yarn_f32($&mry2, &mrx2, 3, 8, 8, true, 10000.0, 4.0, 64, 32.0, 1.0, 1.15) >>>\n'
            '    _ := b.moe_topk_f32($&meids, $&mews, &mlog, 2, false, true, 1.0) >>>\n'
            '    _ := b.moe_gemv_q4_0_f32($&mdq4_b, &mxs, &mwq4, &meids, &mews, 4) >>>\n'
            '    _ := b.moe_gemv_q8_0_f32($&mdq8_b, &mxp, &mwq8, &meids, &mews, 4) >>>\n'
            '    $fb := b.commit() >>>\n    _ := fb.wait() >>>\n    _ := fb.close() >>>\n'
            '    $fm4 := metal.moe_gemv_q4_0_f32($&mdq4, &mxs, &mwq4, &meids, &mews, 4) >>>\n    _ := fm4.wait() >>>\n    _ := fm4.close() >>>\n'
            '    $fm8 := metal.moe_gemv_q8_0_f32($&mdq8, &mxp, &mwq8, &meids, &mews, 4) >>>\n    _ := fm8.wait() >>>\n    _ := fm8.close() >>>\n'
            '    by1 := mry1.bits() >>>\n    by2 := mry2.bits() >>>\n'
            '    bq4i := mdq4.bits() >>>\n    bq4b := mdq4_b.bits() >>>\n'
            '    bq8i := mdq8.bits() >>>\n    bq8b := mdq8_b.bits() >>>\n'
            '    dump("Y1", &by1)\n    dump("Y2", &by2)\n'
            '    dump("Q4I", &bq4i)\n    dump("Q4B", &bq4b)\n'
            '    dump("Q8I", &bq8i)\n    dump("Q8B", &bq8b)\n'
            '    _ := mrx.close() >>>\n    _ := mry1.close() >>>\n    _ := mrx2.close() >>>\n    _ := mry2.close() >>>\n'
            '    _ := mlog.close() >>>\n    _ := mxs.close() >>>\n    _ := mxp.close() >>>\n'
            '    _ := mwq4.close() >>>\n    _ := mwq8.close() >>>\n    _ := meids.close() >>>\n    _ := mews.close() >>>\n'
            '    _ := mdq4.close() >>>\n    _ := mdq8.close() >>>\n    _ := mdq4_b.close() >>>\n    _ := mdq8_b.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_v17() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        for tag, ref in (
            ("Y1", ref_yarn_split),
            ("Y2", ref_yarn_il),
            ("Q4I", ref_moe_q4),
            ("Q4B", ref_moe_q4),
            ("Q8I", ref_moe_q8),
            ("Q8B", ref_moe_q8),
        ):
            got = [bits_to_f32(int(x)) for x in kv[tag].split(",")]
            self.assertEqual(len(got), len(ref), tag)
            for idx, (g, r) in enumerate(zip(got, ref)):
                self.assertAlmostEqual(g, r, delta=1e-4, msg=f"{tag}[{idx}]: got={g} ref={r}")

    def test_v18_moe_kquant_paged_attn_and_training_gpu(self):
        import random

        rng = random.Random(20261126)

        def make_q4_k_row(vals):
            assert len(vals) == 256
            sub_sc, sub_mn = [], []
            for s in range(8):
                sub = vals[s * 32:(s + 1) * 32]
                lo, hi = min(0.0, min(sub)), max(sub)
                r = max(hi - lo, 0.0)
                sub_sc.append(r / 15.0 if r > 0.0 else 0.0)
                sub_mn.append(-lo)
            max_sc, max_mn = max(sub_sc), max(sub_mn)
            d16_b = struct.pack("<e", f32(max_sc / 63.0) if max_sc > 0 else 0.0)
            dm16_b = struct.pack("<e", f32(max_mn / 63.0) if max_mn > 0 else 0.0)
            d16 = f32(struct.unpack("<e", d16_b)[0])
            dm16 = f32(struct.unpack("<e", dm16_b)[0])
            sc_q = [min(63, max(0, int(round(s / d16)))) if d16 > 0 else 0 for s in sub_sc]
            mn_q = [min(63, max(0, int(round(m / dm16)))) if dm16 > 0 else 0 for m in sub_mn]
            scales_b = bytearray(12)
            for j in range(4):
                scales_b[j] = sc_q[j] & 63
                scales_b[j + 4] = mn_q[j] & 63
            for j in range(4, 8):
                scales_b[j + 4] = (sc_q[j] & 15) | ((mn_q[j] & 15) << 4)
                scales_b[j - 4] |= ((sc_q[j] >> 4) & 3) << 6
                scales_b[j] |= ((mn_q[j] >> 4) & 3) << 6
            qs_b = bytearray(128)
            dq = [0.0] * 256
            for ch in range(4):
                s0, s1 = ch * 2, ch * 2 + 1
                d1, m1 = f32(d16 * sc_q[s0]), f32(dm16 * mn_q[s0])
                d2, m2 = f32(d16 * sc_q[s1]), f32(dm16 * mn_q[s1])
                sub0, sub1 = vals[s0 * 32:(s0 + 1) * 32], vals[s1 * 32:(s1 + 1) * 32]
                for l in range(32):
                    q0 = min(15, max(0, int(round((sub0[l] + m1) / d1)))) if d1 > 0 else 0
                    q1 = min(15, max(0, int(round((sub1[l] + m2) / d2)))) if d2 > 0 else 0
                    qs_b[ch * 32 + l] = (q0 & 15) | ((q1 & 15) << 4)
                    dq[ch * 64 + l] = f32(f32(d1 * q0) - m1)
                    dq[ch * 64 + 32 + l] = f32(f32(d2 * q1) - m2)
            return bytes(d16_b + dm16_b + scales_b + qs_b), dq

        def make_q6_k_row(vals):
            assert len(vals) == 256
            amax = max(abs(v) for v in vals)
            d16_b = struct.pack("<e", f32(amax / 127.0) if amax > 0 else 0.0)
            d16 = f32(struct.unpack("<e", d16_b)[0])
            ql = bytearray(128)
            qh = bytearray(64)
            sc = bytearray(16)
            dq = [0.0] * 256
            for h in range(2):
                sub = vals[h * 128:(h + 1) * 128]
                for g in range(8):
                    gvals = sub[g * 16:(g + 1) * 16]
                    gmax = max(abs(v) for v in gvals)
                    # compute integer scale -128..127 such that scale*d16*31 >= gmax
                    sc_i = max(-127, min(127, int(round((gmax / (31.0 * d16)))))) if d16 > 0 else 1
                    if sc_i == 0:
                        sc_i = 1
                    sc[h * 8 + g] = sc_i & 0xFF
                for l in range(32):
                    is_idx = l >> 4
                    qvals = []
                    for pos, sc_off in ((l, is_idx), (l + 32, is_idx + 2), (l + 64, is_idx + 4), (l + 96, is_idx + 6)):
                        sv = struct.unpack("<b", bytes([sc[h * 8 + sc_off]]))[0]
                        denom = d16 * sv
                        qi = max(-32, min(31, int(round(sub[pos] / denom)))) if denom != 0 else 0
                        qvals.append((qi + 32, qi * denom))
                        dq[h * 128 + pos] = f32(qi * denom)
                    u0, u1, u2, u3 = qvals[0][0], qvals[1][0], qvals[2][0], qvals[3][0]
                    ql[h * 64 + l] = (u0 & 15) | ((u2 & 15) << 4)
                    ql[h * 64 + 32 + l] = (u1 & 15) | ((u3 & 15) << 4)
                    qh[h * 32 + l] = ((u0 >> 4) & 3) | (((u1 >> 4) & 3) << 2) | (((u2 >> 4) & 3) << 4) | (((u3 >> 4) & 3) << 6)
            return bytes(ql + qh + sc + d16_b), dq

        # 1. MoE GEMV Q4_K & Q6_K (n_experts=3, out_dim=4, in_dim=256, top_k=2, M=1)
        n_exp, out_dim, in_dim, top_k = 3, 4, 256, 2
        logits = [f32(rng.uniform(-1.5, 1.5)) for _ in range(n_exp)]
        x_moe = [f32(rng.uniform(-0.8, 0.8)) for _ in range(in_dim)]
        q4k_raw = bytearray()
        w4k_dq = []
        q6k_raw = bytearray()
        w6k_dq = []
        for _ in range(n_exp * out_dim):
            r4, dq4 = make_q4_k_row([f32(rng.uniform(-0.8, 0.8)) for _ in range(in_dim)])
            q4k_raw.extend(r4)
            w4k_dq.append(dq4)
            r6, dq6 = make_q6_k_row([f32(rng.uniform(-0.8, 0.8)) for _ in range(in_dim)])
            q6k_raw.extend(r6)
            w6k_dq.append(dq6)
        mx = max(logits)
        ex = [math.exp(v - mx) for v in logits]
        probs = [v / sum(ex) for v in ex]
        ranked = sorted(range(n_exp), key=lambda i: (-probs[i], i))[:top_k]
        ws = [probs[i] for i in ranked]
        ws = [w / sum(ws) for w in ws]
        ref_m4k = [f32(sum(ws[ek] * sum(x_moe[k] * w4k_dq[eid * out_dim + n][k] for k in range(in_dim)) for ek, eid in enumerate(ranked))) for n in range(out_dim)]
        ref_m6k = [f32(sum(ws[ek] * sum(x_moe[k] * w6k_dq[eid * out_dim + n][k] for k in range(in_dim)) for ek, eid in enumerate(ranked))) for n in range(out_dim)]

        # 2. Paged Attention Decode F16: B=2, kv_len=5, block_size=2 (num_blocks=6), n_heads=2, n_kv_heads=1, head_dim=4
        pa_B, pa_kv_len, pa_bs, pa_nb, pa_nh, pa_nkv, pa_hd, pa_scale = 2, 5, 2, 6, 2, 1, 4, 0.5
        pa_q = [f32(rng.uniform(-1.0, 1.0)) for _ in range(pa_B * pa_nh * pa_hd)]
        pa_k_rows = [[f32(struct.unpack("<e", struct.pack("<e", rng.uniform(-1.0, 1.0)))[0]) for _ in range(pa_nkv * pa_hd)] for _ in range(pa_nb * pa_bs)]
        pa_v_rows = [[f32(struct.unpack("<e", struct.pack("<e", rng.uniform(-1.0, 1.0)))[0]) for _ in range(pa_nkv * pa_hd)] for _ in range(pa_nb * pa_bs)]
        pa_bt = [[4.0, 1.0, 3.0], [5.0, 0.0, 2.0]]
        pa_k_u16 = [struct.unpack("<H", struct.pack("<e", v))[0] for row in pa_k_rows for v in row]
        pa_v_u16 = [struct.unpack("<H", struct.pack("<e", v))[0] for row in pa_v_rows for v in row]
        ref_pa = []
        for b in range(pa_B):
            for h in range(pa_nh):
                qv = pa_q[b * pa_nh * pa_hd + h * pa_hd:b * pa_nh * pa_hd + (h + 1) * pa_hd]
                scs = []
                vs_seq = []
                for t in range(pa_kv_len):
                    pb = int(pa_bt[b][t // pa_bs])
                    ri = pb * pa_bs + (t % pa_bs)
                    scs.append(pa_scale * sum(qv[d] * pa_k_rows[ri][d] for d in range(pa_hd)))
                    vs_seq.append(pa_v_rows[ri])
                mx_s = max(scs)
                ex_s = [math.exp(s - mx_s) for s in scs]
                sm_s = sum(ex_s)
                for d in range(pa_hd):
                    ref_pa.append(f32(sum((ex_s[t] / sm_s) * vs_seq[t][d] for t in range(pa_kv_len))))

        # 3. GPU LoRA Add, Cross-Entropy + Grad, AdamW step
        lx = [f32(1.0), f32(-2.0), f32(0.5)]
        la = [f32(0.5), f32(-1.0), f32(2.0), f32(1.5), f32(0.25), f32(-0.5)]
        lb = [f32(1.0), f32(-0.5), f32(0.25), f32(2.0), f32(-1.0), f32(0.5), f32(0.5), f32(1.5)]
        l_init = [f32(0.1), f32(-0.2), f32(0.3), f32(-0.4)]
        lh = [sum(la[r * 3 + j] * lx[j] for j in range(3)) for r in range(2)]
        ref_logits = [f32(l_init[i] + 0.5 * sum(lb[i * 2 + r] * lh[r] for r in range(2))) for i in range(4)]
        # Cross entropy on logits [ref_logits, [0.5, -0.5, 1.5, 0.0]], targets=[2.0, 0.0]
        ce_row1 = [f32(0.5), f32(-0.5), f32(1.5), f32(0.0)]
        ce_logits = ref_logits + ce_row1
        ce_targets = [f32(2.0), f32(0.0)]
        ref_losses = []
        ref_dlogits = []
        for row, tgt in ((ref_logits, 2), (ce_row1, 0)):
            m_r = max(row)
            ex_r = [math.exp(v - m_r) for v in row]
            s_r = sum(ex_r)
            ref_losses.append(f32(m_r + math.log(s_r) - row[tgt]))
            ref_dlogits.extend(f32((ex_r[j] / s_r) - (1.0 if j == tgt else 0.0)) for j in range(4))
        # AdamW on lb [4, 2] using grad=0.1 everywhere for 1 step (lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0.01)
        g_aw = [f32(0.1)] * 8
        ref_aw = [f32(p * (1.0 - 0.1 * 0.01) - 0.1 * 0.1 / (math.sqrt(0.01) + 1e-8)) for p in lb]

        def xu_f32_list(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        def xu_u8_bytes(name, raw):
            ints = ", ".join(str(b) for b in raw)
            return f"    {name}_a := [{ints}]\n    {name} := to_bytes(&{name}_a)\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn to_bytes(vals &[]int) string {\n'
            '    $out := ""\n    for i in 0:vals.len { out.push_str(string_from_byte(vals[i] & 255)) }\n    << out\n'
            '}\n'
            'fn dump(t string, v &[]int) {\n'
            '    $s := t + ":"\n    for i in 0:v.len { if i > 0 { s = s + "," }\n s = s + str(v[i]) }\n    println(s)\n'
            '}\n'
            'fn run_v18() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_f32_list("log_b", logits) + xu_f32_list("xmoe_b", x_moe)
            + xu_u8_bytes("q4k_raw", q4k_raw) + xu_u8_bytes("q6k_raw", q6k_raw)
            + xu_f32_list("paq_b", pa_q)
            + f"    pak_u := [{', '.join(str(v) for v in pa_k_u16)}]\n    pav_u := [{', '.join(str(v) for v in pa_v_u16)}]\n"
            + xu_f32_list("pabt_b", [f32(v) for row in pa_bt for v in row])
            + xu_f32_list("lx_b", lx) + xu_f32_list("la_b", la) + xu_f32_list("lb_b", lb)
            + xu_f32_list("linit_b", l_init) + xu_f32_list("cer1_b", ce_row1)
            + xu_f32_list("cet_b", ce_targets) + xu_f32_list("gaw_b", g_aw)
            + '    $mlog := d.matrix_f32(1, 3) >>>\n    _ := mlog.write_bits(&log_b) >>>\n'
            '    $mxmoe := d.matrix_f32(1, 256) >>>\n    _ := mxmoe.write_bits(&xmoe_b) >>>\n'
            '    $mq4k := d.matrix_q4_k(12, 256) >>>\n    _ := mq4k.write_bytes(&q4k_raw) >>>\n'
            '    $mq6k := d.matrix_q6_k(12, 256) >>>\n    _ := mq6k.write_bytes(&q6k_raw) >>>\n'
            '    $meids := d.matrix_f32(1, 2) >>>\n    $mews := d.matrix_f32(1, 2) >>>\n'
            '    $md4k := d.matrix_f32(1, 4) >>>\n    $md6k := d.matrix_f32(1, 4) >>>\n'
            '    $mpaq := d.matrix_f32(2, 8) >>>\n    _ := mpaq.write_bits(&paq_b) >>>\n'
            '    $mpak := d.matrix_f16(12, 4) >>>\n    _ := mpak.write_bits(&pak_u) >>>\n'
            '    $mpav := d.matrix_f16(12, 4) >>>\n    _ := mpav.write_bits(&pav_u) >>>\n'
            '    $mpabt := d.matrix_f32(2, 3) >>>\n    _ := mpabt.write_bits(&pabt_b) >>>\n'
            '    $mpadst := d.matrix_f32(2, 8) >>>\n'
            '    $mlx := d.matrix_f32(1, 3) >>>\n    _ := mlx.write_bits(&lx_b) >>>\n'
            '    $mla := d.matrix_f32(2, 3) >>>\n    _ := mla.write_bits(&la_b) >>>\n'
            '    $mlb := d.matrix_f32(4, 2) >>>\n    _ := mlb.write_bits(&lb_b) >>>\n'
            '    $mlout := d.matrix_f32(1, 4) >>>\n    _ := mlout.write_bits(&linit_b) >>>\n'
            '    $mcelog := d.matrix_f32(2, 4) >>>\n'
            '    $mcetgt := d.matrix_f32(2, 1) >>>\n    _ := mcetgt.write_bits(&cet_b) >>>\n'
            '    $mcedl := d.matrix_f32(2, 4) >>>\n    $mcelos := d.matrix_f32(2, 1) >>>\n'
            '    $maw_g := d.matrix_f32(4, 2) >>>\n    _ := maw_g.write_bits(&gaw_b) >>>\n'
            '    $maw_m := d.matrix_f32(4, 2) >>>\n    $maw_v := d.matrix_f32(4, 2) >>>\n'
            '    $fb := d.begin_batch() >>>\n'
            '    _ := fb.moe_topk_f32($&meids, $&mews, &mlog, 2, false, true, 1.0) >>>\n'
            '    _ := fb.moe_gemv_q4_k_f32($&md4k, &mxmoe, &mq4k, &meids, &mews, 3) >>>\n'
            '    _ := fb.moe_gemv_q6_k_f32($&md6k, &mxmoe, &mq6k, &meids, &mews, 3) >>>\n'
            '    _ := fb.paged_attention_decode_f16($&mpadst, &mpaq, &mpak, &mpav, &mpabt, 5, 2, 2, 1, 0.5) >>>\n'
            '    _ := fb.lora_add_f32($&mlout, &mlx, &mla, &mlb, 0.5) >>>\n'
            '    _ := fb.adamw_step_f32($&mlb, &maw_g, $&maw_m, $&maw_v, 0.1, 0.9, 0.999, 1e-8, 0.01, 1) >>>\n'
            '    $f1 := fb.commit() >>>\n    _ := f1.wait() >>>\n    _ := f1.close() >>>\n'
            + xu_f32_list("cel_b", ce_logits)
            + '    _ := mcelog.write_bits(&cel_b) >>>\n'
            '    $f2 := metal.cross_entropy_grad_f32($&mcedl, $&mcelos, &mcelog, &mcetgt) >>>\n'
            '    _ := f2.wait() >>>\n    _ := f2.close() >>>\n'
            '    b4k := md4k.bits() >>>\n    b6k := md6k.bits() >>>\n'
            '    bpa := mpadst.bits() >>>\n    blout := mlout.bits() >>>\n'
            '    bdl := mcedl.bits() >>>\n    blos := mcelos.bits() >>>\n    baw := mlb.bits() >>>\n'
            '    dump("M4K", &b4k)\n    dump("M6K", &b6k)\n    dump("PA", &bpa)\n'
            '    dump("LORA", &blout)\n    dump("CEDL", &bdl)\n    dump("CELOS", &blos)\n    dump("ADAMW", &baw)\n'
            '    _ := mlog.close() >>>\n    _ := mxmoe.close() >>>\n    _ := mq4k.close() >>>\n    _ := mq6k.close() >>>\n'
            '    _ := meids.close() >>>\n    _ := mews.close() >>>\n    _ := md4k.close() >>>\n    _ := md6k.close() >>>\n'
            '    _ := mpaq.close() >>>\n    _ := mpak.close() >>>\n    _ := mpav.close() >>>\n    _ := mpabt.close() >>>\n    _ := mpadst.close() >>>\n'
            '    _ := mlx.close() >>>\n    _ := mla.close() >>>\n    _ := mlb.close() >>>\n    _ := mlout.close() >>>\n'
            '    _ := mcelog.close() >>>\n    _ := mcetgt.close() >>>\n    _ := mcedl.close() >>>\n    _ := mcelos.close() >>>\n'
            '    _ := maw_g.close() >>>\n    _ := maw_m.close() >>>\n    _ := maw_v.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_v18() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        for tag, ref in (
            ("M4K", ref_m4k),
            ("M6K", ref_m6k),
            ("PA", ref_pa),
            ("LORA", ref_logits),
            ("CEDL", ref_dlogits),
            ("CELOS", ref_losses),
            ("ADAMW", ref_aw),
        ):
            got = [bits_to_f32(int(x)) for x in kv[tag].split(",")]
            self.assertEqual(len(got), len(ref), tag)
            for idx, (g, r) in enumerate(zip(got, ref)):
                self.assertAlmostEqual(g, r, delta=2e-3 if tag in ("M4K", "M6K", "PA") else 1e-4, msg=f"{tag}[{idx}]: got={g} ref={r}")

    def test_v19_q5k_paged_kv_append_and_backprop_gpu(self):
        # Verify v1.9.0 Metal ops: Q5KMatrix + gemv_q5_k_f32 + moe_gemv_q5_k_f32,
        # paged_kv_cache_append_f16 -> paged_attention_decode_f16 round-trip,
        # gemm_f32_tn, rmsnorm_backward_f32, and swiglu_backward_f32 (sync + Batch).
        import random
        rng = random.Random(20261129)

        def pack_q5_k_row(vals):
            assert len(vals) == 256
            sub_scales, sub_mins, q_subs = [], [], []
            for s in range(8):
                sub = vals[s * 32:(s + 1) * 32]
                lo = min(0.0, min(sub))
                hi = max(sub)
                sc = (hi - lo) / 31.0 if hi > lo else 0.0
                sub_scales.append(sc)
                sub_mins.append(-lo)
                q_subs.append(sub)
            max_sc = max(sub_scales) if sub_scales else 0.0
            max_mn = max(sub_mins) if sub_mins else 0.0
            d = f32(max_sc / 63.0) if max_sc > 0.0 else 0.0
            dmin = f32(max_mn / 63.0) if max_mn > 0.0 else 0.0
            d16_b = struct.pack("<e", d)
            dm16_b = struct.pack("<e", dmin)
            d16 = f32(struct.unpack("<e", d16_b)[0])
            dm16 = f32(struct.unpack("<e", dm16_b)[0])
            sc_q = [min(63, max(0, int(round(s / d16)))) if d16 > 0.0 else 0 for s in sub_scales]
            mn_q = [min(63, max(0, int(round(m / dm16)))) if dm16 > 0.0 else 0 for m in sub_mins]
            scales_b = bytearray(12)
            for j in range(4):
                scales_b[j] = sc_q[j] & 63
                scales_b[j + 4] = mn_q[j] & 63
            for j in range(4, 8):
                scales_b[j + 4] = (sc_q[j] & 15) | ((mn_q[j] & 15) << 4)
                scales_b[j - 4] |= ((sc_q[j] >> 4) & 3) << 6
                scales_b[j] |= ((mn_q[j] >> 4) & 3) << 6
            qh_b = bytearray(32)
            qs_b = bytearray(128)
            dq = []
            for chunk in range(4):
                s_lo = chunk * 2
                s_hi = s_lo + 1
                d1, m1 = f32(d16 * sc_q[s_lo]), f32(dm16 * mn_q[s_lo])
                d2, m2 = f32(d16 * sc_q[s_hi]), f32(dm16 * mn_q[s_hi])
                ql_vals, qh_vals = [], []
                for l in range(32):
                    q0 = min(31, max(0, int(round((q_subs[s_lo][l] + m1) / d1)))) if d1 > 0.0 else 0
                    q1 = min(31, max(0, int(round((q_subs[s_hi][l] + m2) / d2)))) if d2 > 0.0 else 0
                    qs_b[chunk * 32 + l] = (q0 & 15) | ((q1 & 15) << 4)
                    if q0 & 16:
                        qh_b[l] |= 1 << (chunk * 2)
                    if q1 & 16:
                        qh_b[l] |= 1 << (chunk * 2 + 1)
                    ql_vals.append(f32(f32(d1 * q0) - m1))
                    qh_vals.append(f32(f32(d2 * q1) - m2))
                dq.extend(ql_vals)
                dq.extend(qh_vals)
            return bytes(d16_b + dm16_b + scales_b + qh_b + qs_b), dq

        def f16_u16(val):
            return struct.unpack("<H", struct.pack("<e", float(val)))[0]

        def f16_val(u):
            return f32(struct.unpack("<e", struct.pack("<H", u))[0])

        # 1. gemv_q5_k_f32 [2, 3] + moe_gemv_q5_k_f32 (3 experts, out_dim=4, top_k=2)
        xq5 = [f32(rng.uniform(-0.8, 0.8)) for _ in range(2 * 256)]
        w_dense_rows = [[f32(rng.uniform(-1.0, 1.0)) for _ in range(256)] for _ in range(3)]
        w_q5k_raw = bytearray()
        w_q5k_dq = []
        for row in w_dense_rows:
            rb, dq = pack_q5_k_row(row)
            w_q5k_raw.extend(rb)
            w_q5k_dq.append(dq)
        ref_q5k = []
        for m in range(2):
            xr = xq5[m * 256:(m + 1) * 256]
            for n in range(3):
                ref_q5k.append(f32(sum(xr[i] * w_q5k_dq[n][i] for i in range(256))))

        # 3 experts * 4 out_dim = 12 rows
        moe_raw = bytearray()
        moe_dq = []
        for _ in range(12):
            rb, dq = pack_q5_k_row([f32(rng.uniform(-0.9, 0.9)) for _ in range(256)])
            moe_raw.extend(rb)
            moe_dq.append(dq)
        eids = [f32(2.0), f32(0.0)]
        ewts = [f32(0.7), f32(0.3)]
        xmoe = xq5[:256]
        ref_m5k = []
        for n in range(4):
            d2 = sum(xmoe[i] * moe_dq[2 * 4 + n][i] for i in range(256))
            d0 = sum(xmoe[i] * moe_dq[0 * 4 + n][i] for i in range(256))
            ref_m5k.append(f32(0.7 * d2 + 0.3 * d0))

        # 2. paged_kv_cache_append_f16 (2 seqs, seq_len=2, pos_offset=1, block_size=2, num_blocks=6, kv_dim=4)
        #    followed by paged_attention_decode_f16 (kv_len=3, n_heads=2, n_kv_heads=1, head_dim=4, scale=0.5)
        pa_bt = [[3.0, 1.0], [5.0, 2.0]]  # pos 0 -> blk 3 / 5, pos 1,2 -> blk 3@1,1@0 and 5@1,2@0
        k_pages_u16 = [0] * (12 * 4)
        v_pages_u16 = [0] * (12 * 4)
        # seed pos=0 in blocks 3 and 5
        for c in range(4):
            k_pages_u16[(3 * 2 + 0) * 4 + c] = f16_u16(0.25 * (c + 1))
            v_pages_u16[(3 * 2 + 0) * 4 + c] = f16_u16(0.5 + 0.1 * c)
            k_pages_u16[(5 * 2 + 0) * 4 + c] = f16_u16(-0.2 * (c + 1))
            v_pages_u16[(5 * 2 + 0) * 4 + c] = f16_u16(0.8 - 0.15 * c)
        append_k = [f32(rng.uniform(-0.7, 0.7)) for _ in range(2 * 2 * 4)]
        append_v = [f32(rng.uniform(-0.7, 0.7)) for _ in range(2 * 2 * 4)]
        pa_q = [f32(rng.uniform(-0.6, 0.6)) for _ in range(2 * 8)]
        # CPU simulate paged_kv_cache_append_f16 then paged_attention_decode_f16
        k_cpu = list(k_pages_u16)
        v_cpu = list(v_pages_u16)
        for b in range(2):
            for t in range(2):
                pos = 1 + t
                pb = int(round(pa_bt[b][pos // 2]))
                row_idx = pb * 2 + (pos % 2)
                for c in range(4):
                    k_cpu[row_idx * 4 + c] = f16_u16(append_k[(b * 2 + t) * 4 + c])
                    v_cpu[row_idx * 4 + c] = f16_u16(append_v[(b * 2 + t) * 4 + c])
        ref_pa = []
        for b in range(2):
            for h in range(2):
                q_h = pa_q[b * 8 + h * 4:b * 8 + (h + 1) * 4]
                scores = []
                vs_rows = []
                for t in range(3):
                    pb = int(round(pa_bt[b][t // 2]))
                    row_idx = pb * 2 + (t % 2)
                    kr = [f16_val(k_cpu[row_idx * 4 + c]) for c in range(4)]
                    vr = [f16_val(v_cpu[row_idx * 4 + c]) for c in range(4)]
                    scores.append(0.5 * sum(q_h[c] * kr[c] for c in range(4)))
                    vs_rows.append(vr)
                mx = max(scores)
                ex = [math.exp(s - mx) for s in scores]
                sm = sum(ex)
                for c in range(4):
                    ref_pa.append(f32(sum(ex[t] * vs_rows[t][c] for t in range(3)) / sm))

        # 3. gemm_f32_tn: A [4, 2], B [4, 3] -> C = A^T @ B [2, 3]
        ga = [f32(rng.uniform(-1.0, 1.0)) for _ in range(8)]
        gb = [f32(rng.uniform(-1.0, 1.0)) for _ in range(12)]
        ref_tn = [
            f32(sum(ga[k * 2 + m] * gb[k * 3 + n] for k in range(4)))
            for m in range(2) for n in range(3)
        ]

        # 4. rmsnorm_backward_f32: [3, 6], eps = 1e-5
        rx = [f32(rng.uniform(-1.2, 1.2)) for _ in range(18)]
        rdy = [f32(rng.uniform(-0.8, 0.8)) for _ in range(18)]
        rw = [f32(0.7 + 0.1 * c) for c in range(6)]
        reps = 1e-5
        ref_dx = []
        ref_dw = [0.0] * 6
        for r in range(3):
            xr = rx[r * 6:(r + 1) * 6]
            dyr = rdy[r * 6:(r + 1) * 6]
            inv = 1.0 / math.sqrt(sum(v * v for v in xr) / 6.0 + reps)
            sdot = sum((dyr[c] * rw[c]) * xr[c] for c in range(6)) / 6.0
            sc = sdot * inv * inv
            for c in range(6):
                ref_dx.append(f32(inv * (dyr[c] * rw[c] - xr[c] * sc)))
                ref_dw[c] += dyr[c] * xr[c] * inv
        ref_dw = [f32(v) for v in ref_dw]

        # 5. swiglu_backward_f32: [2, 5]
        sg = [f32(rng.uniform(-2.0, 2.0)) for _ in range(10)]
        su = [f32(rng.uniform(-1.5, 1.5)) for _ in range(10)]
        sdy = [f32(rng.uniform(-1.0, 1.0)) for _ in range(10)]
        ref_dgate = []
        ref_dup = []
        for g, u, dyv in zip(sg, su, sdy):
            sig = 1.0 / (1.0 + math.exp(-g))
            silu = g * sig
            dsilu = sig * (1.0 + g * (1.0 - sig))
            ref_dup.append(f32(dyv * silu))
            ref_dgate.append(f32(dyv * u * dsilu))

        def xu_f32_list(name, vals):
            return f"    {name} := [{', '.join(str(f32_bits(v)) for v in vals)}]\n"

        def xu_u8_bytes(name, raw):
            ints = ", ".join(str(b) for b in raw)
            return f"    {name}_a := [{ints}]\n    {name} := to_bytes(&{name}_a)\n"

        xu_src = (
            '@main\nmetal #"github.com/xulang-org/packages/metal"\n#errors\n'
            'fn to_bytes(vals &[]int) string {\n'
            '    $out := ""\n    for i in 0:vals.len { out.push_str(string_from_byte(vals[i] & 255)) }\n    << out\n'
            '}\n'
            'fn dump(t string, v &[]int) {\n'
            '    $s := t + ":"\n    for i in 0:v.len { if i > 0 { s = s + "," }\n s = s + str(v[i]) }\n    println(s)\n'
            '}\n'
            'fn run_v19() int {\n'
            '    if !metal.available() { println("SKIP:NO_METAL")\n << 0 }\n'
            '    $d := metal.open() >>>\n'
            + xu_f32_list("xq5_b", xq5) + xu_u8_bytes("wq5k_raw", w_q5k_raw)
            + xu_f32_list("xm_b", xmoe)
            + xu_f32_list("eids_b", eids) + xu_f32_list("ewts_b", ewts)
            + xu_u8_bytes("moeq5k_raw", moe_raw)
            + f"    kp_u := [{', '.join(str(v) for v in k_pages_u16)}]\n"
            + f"    vp_u := [{', '.join(str(v) for v in v_pages_u16)}]\n"
            + xu_f32_list("ak_b", append_k) + xu_f32_list("av_b", append_v)
            + xu_f32_list("paq_b", pa_q) + xu_f32_list("pabt_b", [f32(v) for row in pa_bt for v in row])
            + xu_f32_list("ga_b", ga) + xu_f32_list("gb_b", gb)
            + xu_f32_list("rx_b", rx) + xu_f32_list("rdy_b", rdy) + xu_f32_list("rw_b", rw)
            + xu_f32_list("sg_b", sg) + xu_f32_list("su_b", su) + xu_f32_list("sdy_b", sdy)
            + '    $mxq5 := d.matrix_f32(2, 256) >>>\n    _ := mxq5.write_bits(&xq5_b) >>>\n'
            '    $mwq5 := d.matrix_q5_k(3, 256) >>>\n    _ := mwq5.write_bytes(&wq5k_raw) >>>\n'
            '    $mdq5 := d.matrix_f32(2, 3) >>>\n'
            '    $mxm := d.matrix_f32(1, 256) >>>\n    _ := mxm.write_bits(&xm_b) >>>\n'
            '    $meids := d.matrix_f32(1, 2) >>>\n    _ := meids.write_bits(&eids_b) >>>\n'
            '    $mews := d.matrix_f32(1, 2) >>>\n    _ := mews.write_bits(&ewts_b) >>>\n'
            '    $mmoew := d.matrix_q5_k(12, 256) >>>\n    _ := mmoew.write_bytes(&moeq5k_raw) >>>\n'
            '    $mdm5 := d.matrix_f32(1, 4) >>>\n'
            '    $mkp := d.matrix_f16(12, 4) >>>\n    _ := mkp.write_bits(&kp_u) >>>\n'
            '    $mvp := d.matrix_f16(12, 4) >>>\n    _ := mvp.write_bits(&vp_u) >>>\n'
            '    $mak := d.matrix_f32(4, 4) >>>\n    _ := mak.write_bits(&ak_b) >>>\n'
            '    $mav := d.matrix_f32(4, 4) >>>\n    _ := mav.write_bits(&av_b) >>>\n'
            '    $mpaq := d.matrix_f32(2, 8) >>>\n    _ := mpaq.write_bits(&paq_b) >>>\n'
            '    $mpabt := d.matrix_f32(2, 2) >>>\n    _ := mpabt.write_bits(&pabt_b) >>>\n'
            '    $mpadst := d.matrix_f32(2, 8) >>>\n'
            '    $mga := d.matrix_f32(4, 2) >>>\n    _ := mga.write_bits(&ga_b) >>>\n'
            '    $mgb := d.matrix_f32(4, 3) >>>\n    _ := mgb.write_bits(&gb_b) >>>\n'
            '    $mgc := d.matrix_f32(2, 3) >>>\n'
            '    $mrx := d.matrix_f32(3, 6) >>>\n    _ := mrx.write_bits(&rx_b) >>>\n'
            '    $mrdy := d.matrix_f32(3, 6) >>>\n    _ := mrdy.write_bits(&rdy_b) >>>\n'
            '    $mrw := d.matrix_f32(1, 6) >>>\n    _ := mrw.write_bits(&rw_b) >>>\n'
            '    $mrdx := d.matrix_f32(3, 6) >>>\n    $mrdw := d.matrix_f32(1, 6) >>>\n'
            '    $msg := d.matrix_f32(2, 5) >>>\n    _ := msg.write_bits(&sg_b) >>>\n'
            '    $msu := d.matrix_f32(2, 5) >>>\n    _ := msu.write_bits(&su_b) >>>\n'
            '    $msdy := d.matrix_f32(2, 5) >>>\n    _ := msdy.write_bits(&sdy_b) >>>\n'
            '    $msdg := d.matrix_f32(2, 5) >>>\n    $msdu := d.matrix_f32(2, 5) >>>\n'
            '    $fb := d.begin_batch() >>>\n'
            '    _ := fb.gemv_q5_k_f32($&mdq5, &mxq5, &mwq5) >>>\n'
            '    _ := fb.moe_gemv_q5_k_f32($&mdm5, &mxm, &mmoew, &meids, &mews, 3) >>>\n'
            '    _ := fb.paged_kv_cache_append_f16($&mkp, &mak, &mpabt, 1, 2) >>>\n'
            '    _ := fb.paged_kv_cache_append_f16($&mvp, &mav, &mpabt, 1, 2) >>>\n'
            '    _ := fb.paged_attention_decode_f16($&mpadst, &mpaq, &mkp, &mvp, &mpabt, 3, 2, 2, 1, 0.5) >>>\n'
            '    _ := fb.gemm_f32_tn($&mgc, &mga, &mgb) >>>\n'
            '    _ := fb.rmsnorm_backward_f32($&mrdx, $&mrdw, &mrdy, &mrx, &mrw, 1e-5) >>>\n'
            '    _ := fb.swiglu_backward_f32($&msdg, $&msdu, &msdy, &msg, &msu) >>>\n'
            '    $f := fb.commit() >>>\n    _ := f.wait() >>>\n    _ := f.close() >>>\n'
            '    bq5 := mdq5.bits() >>>\n    bm5 := mdm5.bits() >>>\n'
            '    bpa := mpadst.bits() >>>\n    btn := mgc.bits() >>>\n'
            '    bdx := mrdx.bits() >>>\n    bdw := mrdw.bits() >>>\n'
            '    bsg := msdg.bits() >>>\n    bsu := msdu.bits() >>>\n'
            '    dump("Q5K", &bq5)\n    dump("M5K", &bm5)\n    dump("PA", &bpa)\n'
            '    dump("TN", &btn)\n    dump("RDX", &bdx)\n    dump("RDW", &bdw)\n'
            '    dump("SDG", &bsg)\n    dump("SDU", &bsu)\n'
            '    _ := mxq5.close() >>>\n    _ := mwq5.close() >>>\n    _ := mdq5.close() >>>\n'
            '    _ := mxm.close() >>>\n    _ := meids.close() >>>\n    _ := mews.close() >>>\n'
            '    _ := mmoew.close() >>>\n    _ := mdm5.close() >>>\n'
            '    _ := mkp.close() >>>\n    _ := mvp.close() >>>\n    _ := mak.close() >>>\n    _ := mav.close() >>>\n'
            '    _ := mpaq.close() >>>\n    _ := mpabt.close() >>>\n    _ := mpadst.close() >>>\n'
            '    _ := mga.close() >>>\n    _ := mgb.close() >>>\n    _ := mgc.close() >>>\n'
            '    _ := mrx.close() >>>\n    _ := mrdy.close() >>>\n    _ := mrw.close() >>>\n'
            '    _ := mrdx.close() >>>\n    _ := mrdw.close() >>>\n'
            '    _ := msg.close() >>>\n    _ := msu.close() >>>\n    _ := msdy.close() >>>\n'
            '    _ := msdg.close() >>>\n    _ := msdu.close() >>>\n'
            '    _ := d.close() >>>\n'
            '    println("LIVE:" + str(metal.live_handles()))\n    << 0\n}\n'
            'fn main() int { << run_v19() >>> (err) { eprintln(err.msg())\n << 1 } }\n'
        )
        with tempfile.TemporaryDirectory() as td:
            out = run_xu(Path(td), xu_src)
        if "SKIP:NO_METAL" in out:
            self.skipTest("Metal device unavailable on host")
        kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(kv.get("LIVE"), "0")
        for tag, ref in (
            ("Q5K", ref_q5k),
            ("M5K", ref_m5k),
            ("PA", ref_pa),
            ("TN", ref_tn),
            ("RDX", ref_dx),
            ("RDW", ref_dw),
            ("SDG", ref_dgate),
            ("SDU", ref_dup),
        ):
            got = [bits_to_f32(int(x)) for x in kv[tag].split(",")]
            self.assertEqual(len(got), len(ref), tag)
            for idx, (g, r) in enumerate(zip(got, ref)):
                self.assertAlmostEqual(g, r, delta=2e-3 if tag in ("Q5K", "M5K", "PA") else 1e-4, msg=f"{tag}[{idx}]: got={g} ref={r}")


if __name__ == "__main__":
    unittest.main()
