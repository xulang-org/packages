import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_v116_batch1_hardening import ROOT, REPO, XU, build_xu_no_codesign, xu_env

sys.path.insert(0, str(REPO / "scripts"))
from test_llm_engine import (  # noqa: E402
    CFG,
    REQUESTS,
    bits_to_f32,
    f32,
    gen_weights,
    generate_weight_dict,
    pack_gguf_str,
    parse_bench,
    py_run_request,
    quantize_q4_k_matrix,
    write_gguf_q4_0,
    write_safetensors,
)


def metal_env(work: Path) -> dict:
    env = xu_env(work)
    env["XU_METAL_LIBRARY"] = str(ROOT / "metal" / "native" / "libxu_metal.dylib")
    return env


def run_metal_bin(exe: Path, work: Path, args: list[str]) -> str:
    res = subprocess.run(
        [str(exe), *args],
        cwd=work,
        env=metal_env(work),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(f"Exit {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res.stdout


def write_fast_q4k_bench_gguf(path: Path, cfg: dict) -> None:
    """Write a deterministic valid GGUF v3 Q4_K model fast by tiling quantized super-blocks."""
    dim = cfg["dim"]
    hidden = cfg["hidden_dim"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    head_dim = dim // n_heads
    kv_dim = n_kv * head_dim
    vocab = cfg["vocab_size"]

    rng = random.Random(20261130)
    tile_raw, _ = quantize_q4_k_matrix(gen_weights(rng, 4096, 0.04))
    norms = [f32(0.95 + 0.1 * rng.random()) for _ in range(dim)]
    norm_raw = struct.pack(f"<{dim}f", *norms)
    vocab_pieces = [f"tok_{i}" for i in range(vocab)]

    kv_entries = [
        ("general.architecture", 8, pack_gguf_str("llama")),
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("llama.embedding_length", 4, struct.pack("<I", dim)),
        ("llama.feed_forward_length", 4, struct.pack("<I", hidden)),
        ("llama.block_count", 4, struct.pack("<I", n_layers)),
        ("llama.attention.head_count", 4, struct.pack("<I", n_heads)),
        ("llama.attention.head_count_kv", 4, struct.pack("<I", n_kv)),
        ("llama.context_length", 4, struct.pack("<I", cfg["max_seq_len"])),
        ("llama.rope.freq_base", 6, struct.pack("<f", float(cfg["rope_theta"]))),
        ("llama.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-5)),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, vocab) + b"".join(pack_gguf_str(p) for p in vocab_pieces)),
    ]

    tensor_specs = []

    def add_q4k_tiled(gname: str, cols: int, rows: int) -> None:
        n_blk = (rows * cols) // 256
        rep = (n_blk + 15) // 16
        raw = (tile_raw * rep)[: n_blk * 144]
        tensor_specs.append((gname, [cols, rows], 12, raw))

    add_q4k_tiled("token_embd.weight", dim, vocab)
    for l in range(n_layers):
        gp = f"blk.{l}."
        tensor_specs.append((gp + "attn_norm.weight", [dim], 0, norm_raw))
        add_q4k_tiled(gp + "attn_q.weight", dim, dim)
        add_q4k_tiled(gp + "attn_k.weight", dim, kv_dim)
        add_q4k_tiled(gp + "attn_v.weight", dim, kv_dim)
        add_q4k_tiled(gp + "attn_output.weight", dim, dim)
        tensor_specs.append((gp + "ffn_norm.weight", [dim], 0, norm_raw))
        add_q4k_tiled(gp + "ffn_gate.weight", dim, hidden)
        add_q4k_tiled(gp + "ffn_up.weight", dim, hidden)
        add_q4k_tiled(gp + "ffn_down.weight", hidden, dim)
    tensor_specs.append(("output_norm.weight", [dim], 0, norm_raw))
    add_q4k_tiled("output.weight", dim, vocab)

    buf = bytearray(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, len(tensor_specs), len(kv_entries)))
    for k, vtype, vbytes in kv_entries:
        buf.extend(pack_gguf_str(k))
        buf.extend(struct.pack("<I", vtype))
        buf.extend(vbytes)

    payload = bytearray()
    for tname, dims, gtype, raw in tensor_specs:
        tpad = (32 - (len(payload) % 32)) % 32
        payload.extend(b"\x00" * tpad)
        off = len(payload)
        payload.extend(raw)
        buf.extend(pack_gguf_str(tname))
        buf.extend(struct.pack("<I", len(dims)))
        for dv in dims:
            buf.extend(struct.pack("<Q", dv))
        buf.extend(struct.pack("<IQ", gtype, off))

    pad = (32 - (len(buf) % 32)) % 32
    buf.extend(b"\x00" * pad)
    buf.extend(payload)
    path.write_bytes(buf)


class TestV117LlmMetal(unittest.TestCase):
    def test_safetensors_f32_contiguous_and_paged_parity(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            tensors = generate_weight_dict(CFG)
            st_path = work / "tiny_f32.safetensors"
            write_safetensors(st_path, CFG, tensors)

            py0_toks, _, py0_logits = py_run_request(
                tensors, REQUESTS[0][0], 1, return_last_logits=True, cfg=CFG
            )
            py0_all, _ = py_run_request(tensors, REQUESTS[0][0], REQUESTS[0][1], cfg=CFG)
            py1_all, _ = py_run_request(tensors, REQUESTS[1][0], REQUESTS[1][1], cfg=CFG)

            src = '''@main
#"github.com/xulang-org/packages/llm_metal" as llm_metal
#"github.com/xulang-org/packages/metal" as metal
#math
#os

fn join_ints(xs &[]int) string {
    $s := ""
    for i in 0:xs.len {
        if i > 0 { s = s + "," }
        s = s + str(xs[i])
    }
    << s
}

fn run() int {
    args := os.args()
    $m := llm_metal.load_safetensors(args[1], 64) >>>
    p0 := [1, 7, 42]
    p1 := [3, 19]
    g0 := llm_metal.generate(&m, &p0, 3) >>>
    g1 := llm_metal.generate(&m, &p1, 5) >>>
    gp0 := llm_metal.generate_paged(&m, &p0, 3, 2) >>>
    gp1 := llm_metal.generate_paged(&m, &p1, 5, 2) >>>
    $kv := llm_metal.new_kv_cache(&m, 16) >>>
    _ := llm_metal.forward_token_argmax(&m, $&kv, 1) >>>
    _ := llm_metal.forward_token_argmax(&m, $&kv, 7) >>>
    lg := llm_metal.forward_token_logits(&m, $&kv, 42) >>>
    _ := kv.close()
    _ := m.close()

    $lb := ""
    for i in 0:lg.len {
        if i > 0 { lb = lb + "," }
        lb = lb + str(math.float32_bits(f64(lg[i])))
    }
    println("G0:" + join_ints(&g0))
    println("G1:" + join_ints(&g1))
    println("GP0:" + join_ints(&gp0))
    println("GP1:" + join_ints(&gp1))
    println("LOGITS:" + lb)
    println("LIVE:" + str(metal.live_handles()))
    << 0
}

fn main() int {
    << run() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
'''
            exe = build_xu_no_codesign(work, "st_parity", src)
            out = run_metal_bin(exe, work, [str(st_path)])
            kv_out = dict(line.split(":", 1) for line in out.strip().splitlines())

            g0 = [int(x) for x in kv_out["G0"].split(",")]
            g1 = [int(x) for x in kv_out["G1"].split(",")]
            gp0 = [int(x) for x in kv_out["GP0"].split(",")]
            gp1 = [int(x) for x in kv_out["GP1"].split(",")]
            gpu_logits = [bits_to_f32(int(x)) for x in kv_out["LOGITS"].split(",")]

            self.assertEqual(g0, py0_all)
            self.assertEqual(g1, py1_all)
            self.assertEqual(gp0, py0_all)
            self.assertEqual(gp1, py1_all)
            max_diff = max(abs(a - b) for a, b in zip(gpu_logits, py0_logits))
            self.assertLess(max_diff, 2e-3, f"logits max_diff={max_diff}")
            self.assertEqual(g0[0], py0_toks[0])
            self.assertEqual(int(kv_out["LIVE"]), 0)

    def test_gguf_quant_modes_parity(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            src = '''@main
#"github.com/xulang-org/packages/llm_metal" as llm_metal
#"github.com/xulang-org/packages/metal" as metal
#math
#os

fn join_ints(xs &[]int) string {
    $s := ""
    for i in 0:xs.len {
        if i > 0 { s = s + "," }
        s = s + str(xs[i])
    }
    << s
}

fn run() int {
    args := os.args()
    $m := llm_metal.load_gguf(args[1], 128) >>>
    p0 := [1, 7, 42]
    p1 := [3, 19]
    g0 := llm_metal.generate(&m, &p0, 3) >>>
    g1 := llm_metal.generate(&m, &p1, 5) >>>
    $kv := llm_metal.new_kv_cache(&m, 16) >>>
    _ := llm_metal.forward_token_argmax(&m, $&kv, 1) >>>
    _ := llm_metal.forward_token_argmax(&m, $&kv, 7) >>>
    lg := llm_metal.forward_token_logits(&m, $&kv, 42) >>>
    _ := kv.close()
    _ := m.close()

    $lb := ""
    for i in 0:lg.len {
        if i > 0 { lb = lb + "," }
        lb = lb + str(math.float32_bits(f64(lg[i])))
    }
    println("G0:" + join_ints(&g0))
    println("G1:" + join_ints(&g1))
    println("LOGITS:" + lb)
    println("LIVE:" + str(metal.live_handles()))
    << 0
}

fn main() int {
    << run() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
'''
            exe = build_xu_no_codesign(work, "gguf_quant", src)
            kcfg = dict(CFG, n_kv_heads=4, max_seq_len=128)
            base_tensors = generate_weight_dict(CFG)
            k_tensors = generate_weight_dict(kcfg)

            cases = [
                ("q4_0", CFG, base_tensors),
                ("q8_0", CFG, base_tensors),
                ("q4_k", kcfg, k_tensors),
                ("q5_k", kcfg, k_tensors),
                ("q6_k", kcfg, k_tensors),
            ]
            for fmt, cfg, tensors in cases:
                with self.subTest(fmt=fmt):
                    path = work / f"model_{fmt}.gguf"
                    first_ret, _ = write_gguf_q4_0(
                        path, cfg, tensors, quant_mode=(None if fmt == "q4_0" else fmt)
                    )
                    q4_ref = first_ret if fmt == "q4_0" else None
                    dq_ref = None if fmt == "q4_0" else first_ret
                    py0_toks, _, py0_logits = py_run_request(
                        tensors,
                        REQUESTS[0][0],
                        1,
                        q4_data=q4_ref,
                        use_q8_act=False,
                        dequant_data=dq_ref,
                        return_last_logits=True,
                        cfg=cfg,
                    )
                    py0_all, _ = py_run_request(
                        tensors,
                        REQUESTS[0][0],
                        REQUESTS[0][1],
                        q4_data=q4_ref,
                        use_q8_act=False,
                        dequant_data=dq_ref,
                        cfg=cfg,
                    )
                    py1_all, _ = py_run_request(
                        tensors,
                        REQUESTS[1][0],
                        REQUESTS[1][1],
                        q4_data=q4_ref,
                        use_q8_act=False,
                        dequant_data=dq_ref,
                        cfg=cfg,
                    )

                    out = run_metal_bin(exe, work, [str(path)])
                    kv_out = dict(line.split(":", 1) for line in out.strip().splitlines())
                    g0 = [int(x) for x in kv_out["G0"].split(",")]
                    g1 = [int(x) for x in kv_out["G1"].split(",")]
                    gpu_logits = [bits_to_f32(int(x)) for x in kv_out["LOGITS"].split(",")]

                    self.assertEqual(g0, py0_all, f"{fmt} g0 mismatch")
                    self.assertEqual(g1, py1_all, f"{fmt} g1 mismatch")
                    self.assertEqual(g0[0], py0_toks[0])
                    max_diff = max(abs(a - b) for a, b in zip(gpu_logits, py0_logits))
                    self.assertLess(max_diff, 5e-3, f"{fmt} max_diff={max_diff}")
                    self.assertEqual(int(kv_out["LIVE"]), 0)

    def test_gpu_lora_train_and_infer(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            kcfg = dict(CFG, n_kv_heads=4, max_seq_len=64)
            tensors = generate_weight_dict(kcfg)
            q4k_path = work / "lora_base_q4k.gguf"
            write_gguf_q4_0(q4k_path, kcfg, tensors, quant_mode="q4_k")

            src = '''@main
#"github.com/xulang-org/packages/llm_metal" as llm_metal
#"github.com/xulang-org/packages/metal" as metal
#os
#strconv

fn run() int {
    args := os.args()
    $m := llm_metal.load_gguf(args[1], 64) >>>
    $lora := llm_metal.new_lora(&m, 4, 8.0) >>>
    inp := [1, 7, 42]
    tgt := [7, 42, 19]
    $loss0 := 0.0
    $loss_last := 0.0
    for step in 0:25 {
        lv := llm_metal.train_lora_step(&m, $&lora, &inp, &tgt, 0.05, 0.0) >>>
        if step == 0 { loss0 = lv }
        loss_last = lv
    }
    $kv := llm_metal.new_kv_cache(&m, 16) >>>
    _ := llm_metal.forward_token_lora_argmax(&m, $&kv, &lora, 1) >>>
    t1 := llm_metal.forward_token_lora_argmax(&m, $&kv, &lora, 7) >>>
    t2 := llm_metal.forward_token_lora_argmax(&m, $&kv, &lora, 42) >>>
    _ := kv.close()
    _ := lora.close()
    _ := m.close()
    println("LOSS0:" + strconv.format_float(loss0, 102, 8))
    println("LOSSN:" + strconv.format_float(loss_last, 102, 8))
    println("PRED:" + str(t1) + "," + str(t2))
    println("LIVE:" + str(metal.live_handles()))
    << 0
}

fn main() int {
    << run() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
'''
            exe = build_xu_no_codesign(work, "gpu_lora", src)
            out = run_metal_bin(exe, work, [str(q4k_path)])
            kv_out = dict(line.split(":", 1) for line in out.strip().splitlines())
            loss0 = float(kv_out["LOSS0"])
            loss_n = float(kv_out["LOSSN"])
            self.assertLess(loss_n, loss0 * 0.15, f"loss0={loss0}, loss_n={loss_n}")
            self.assertEqual(kv_out["PRED"], "42,19")
            self.assertEqual(int(kv_out["LIVE"]), 0)

    def test_metal_gpu_vs_cpu_q4k_benchmark(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            bcfg = {
                "dim": 2048,
                "hidden_dim": 5632,
                "n_layers": 8,
                "n_heads": 16,
                "n_kv_heads": 8,
                "vocab_size": 2048,
                "max_seq_len": 128,
                "rope_theta": 10000,
            }
            gguf_path = work / "bench_q4k_2048.gguf"
            write_fast_q4k_bench_gguf(gguf_path, bcfg)

            gpu_src = '''@main
#"github.com/xulang-org/packages/llm_metal" as llm_metal
#"github.com/xulang-org/packages/metal" as metal
#os
#time

fn run() int {
    args := os.args()
    $m := llm_metal.load_gguf(args[1], 128) >>>
    $kv := llm_metal.new_kv_cache(&m, 128) >>>
    $prompt := []int{}
    for pi in 0:32 {
        prompt.push((pi * 17 + 1) % 2048)
    }
    _ := llm_metal.prefill_tokens(&m, $&kv, &prompt) >>>
    kv.reset()
    t_pre := time.now()
    _ := llm_metal.prefill_tokens(&m, $&kv, &prompt) >>>
    prefill_ns := time.now() - t_pre

    kv.reset()
    $tok := 1
    for _ in 0:2 {
        tok = llm_metal.forward_token_argmax(&m, $&kv, tok) >>>
    }
    kv.reset()
    tok = llm_metal.forward_token_argmax(&m, $&kv, 1) >>>
    t_dec := time.now()
    for _ in 0:16 {
        tok = llm_metal.forward_token_argmax(&m, $&kv, tok) >>>
    }
    decode_ns := time.now() - t_dec

    _ := kv.close()
    _ := m.close()
    println("PREFILL_NS:" + str(prefill_ns))
    println("DECODE_NS:" + str(decode_ns))
    println("LAST_TOK:" + str(tok))
    println("LIVE:" + str(metal.live_handles()))
    << 0
}

fn main() int {
    << run() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
'''
            gpu_exe = build_xu_no_codesign(work, "gpu_bench", gpu_src)
            gpu_out = run_metal_bin(gpu_exe, work, [str(gguf_path)])
            kv_gpu = dict(line.split(":", 1) for line in gpu_out.strip().splitlines())
            gpu_prefill_tok_s = 32 * 1e9 / max(int(kv_gpu["PREFILL_NS"]), 1)
            gpu_decode_tok_s = 16 * 1e9 / max(int(kv_gpu["DECODE_NS"]), 1)
            self.assertEqual(int(kv_gpu["LIVE"]), 0)

            cpu_exe = work / "cpu_llm_engine"
            cpu_build_env = dict(
                os.environ,
                XU_STDLIB_ROOT=str(REPO / "src"),
                XU_CACHE=str(work / "cpu_cache"),
                XU_TENSOR_FUSION="1",
                XU_TENSOR_FAST_MATH="1",
                XU_TASK_POLL="0",
            )
            cpu_src = REPO / "examples" / "llm_engine" / "main.xu"
            bres = subprocess.run(
                [str(XU), "build", str(cpu_src), "-o", str(cpu_exe)],
                cwd=REPO,
                env=cpu_build_env,
                capture_output=True,
                text=True,
            )
            if bres.returncode != 0:
                raise AssertionError(bres.stdout + bres.stderr)

            cres = subprocess.run(
                [str(cpu_exe), "bench", str(gguf_path), "16", "single", "32", "--q8k-act"],
                cwd=work,
                env=cpu_build_env,
                capture_output=True,
                text=True,
            )
            if cres.returncode != 0:
                raise AssertionError(cres.stdout + cres.stderr)
            cpu_stats = parse_bench(cres.stdout)
            cpu_prefill_tok_s = cpu_stats["prefill_tokens"] * 1e9 / max(cpu_stats["prefill_ns"], 1)
            cpu_decode_tok_s = cpu_stats["decode_steps"] * 1e9 / max(cpu_stats["decode_total_ns"], 1)
            prefill_speedup = gpu_prefill_tok_s / max(cpu_prefill_tok_s, 1e-9)
            decode_speedup = gpu_decode_tok_s / max(cpu_decode_tok_s, 1e-9)
            print(
                f"\n[Metal vs CPU Q4_K dim=2048 L=8 Prefill P=32] CPU(ARM64-SDOT)={cpu_prefill_tok_s:.1f} tok/s"
                f" | GPU(llm_metal)={gpu_prefill_tok_s:.1f} tok/s | speedup={prefill_speedup:.2f}x\n"
                f"[Metal vs CPU Q4_K dim=2048 L=8 Decode 16] CPU(ARM64-SDOT)={cpu_decode_tok_s:.1f} tok/s"
                f" | GPU(llm_metal)={gpu_decode_tok_s:.1f} tok/s | speedup={decode_speedup:.2f}x",
                flush=True,
            )
            self.assertGreater(gpu_prefill_tok_s, cpu_prefill_tok_s)
            self.assertGreater(gpu_decode_tok_s, cpu_decode_tok_s)
