import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("XU_ROOT", str(ROOT.parent / "xu"))).resolve()
XU = Path(os.environ.get("XU", str(REPO / "bin" / "xu"))).resolve()


def xu_env(work: Path) -> dict:
    return dict(
        os.environ,
        XU_STDLIB_ROOT=str(REPO / "src"),
        XU_PACKAGES_ROOT=str(ROOT),
        XU_CACHE=str(work / "cache"),
        XU_TASK_POLL="0",
    )


def build_xu(work: Path, name: str, source: str) -> Path:
    proj = work / name
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "xu.xun").write_text(
        "module: " + name + "\n"
        "version: !ver 0.1.0\n"
        "xu: !ver 0.12.6\n"
        "require:\n"
        f'  "github.com/xulang-org/packages": file:{ROOT}\n'
    )
    (proj / "main.xu").write_text(source)
    exe = work / (name + "_bin")
    res = subprocess.run(
        [str(XU), "build", str(proj / "main.xu"), "-o", str(exe)],
        cwd=proj,
        env=xu_env(work),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(res.stdout + res.stderr)
    if os.uname().sysname == "Darwin":
        subprocess.run(["codesign", "--force", "--sign", "-", str(exe)], check=True, capture_output=True)
    return exe


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class MockV114State:
    def __init__(self):
        self.lock = threading.Lock()
        self.paths = []
        self.primary_fails = 0
        self.weighted_models = []


class MockV114Handler(http.server.BaseHTTPRequestHandler):
    state = None

    def log_message(self, format, *args):
        return

    def _read_raw(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n) if n > 0 else b""

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        st = self.__class__.state
        raw = self._read_raw()
        body = json.loads(raw.decode("utf-8"))
        with st.lock:
            st.paths.append(self.path)

        if self.path == "/gw_fail/v1/chat/completions":
            with st.lock:
                st.primary_fails += 1
            self._send_json(503, {"error": {"message": "upstream overloaded"}})
            return

        if self.path in ("/gw_ok/v1/chat/completions", "/gw_alt/v1/chat/completions"):
            m = body.get("model", "")
            with st.lock:
                st.weighted_models.append(m)
            self._send_json(200, {
                "id": "gw-114",
                "model": m,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "gw-ok:" + m}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })
            return

        if self.path in ("/openai/v1/chat/completions", "/openai/chat/completions"):
            msgs = body.get("messages", [])
            sys_txt = msgs[0]["content"] if msgs else ""
            usr_txt = msgs[-1]["content"] if msgs else ""
            m = body.get("model", "gpt-4o-mini")
            if "synthetic dataset generator" in sys_txt:
                content = json.dumps([
                    {"instruction": "What is Xu?", "input": "", "output": "Xu is a compiled systems and AI language."},
                    {"instruction": "Explain LoRA.", "input": "", "output": "Low-rank adaptation freezes base weights and updates low-rank matrices."}
                ])
            elif "You situate a document chunk" in sys_txt:
                content = "Explains HNSW graph indexing in PostgreSQL."
            elif "You are agent 'triage'" in sys_txt:
                content = "HANDOFF:coder implement fast parser"
            elif "You are agent 'coder'" in sys_txt:
                content = "Done: fast parser implemented."
            elif "weak-model" in m:
                content = "Maybe check docs."
            elif "strong-model" in m:
                content = "Use HNSW cosine index with m=16, ef_construction=64."
            else:
                content = "distilled:" + usr_txt
            self._send_json(200, {
                "id": "chatcmpl-114",
                "model": m,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 14, "completion_tokens": 9, "total_tokens": 23},
            })
            return

        self._send_json(404, {"error": {"message": "not found"}})


class TestV114Ecosystem(unittest.TestCase):
    def test_v114_synth_swarm_a2a_stream_gateway_rag_eval_and_live_sft(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import test_llm_engine as tle

        state = MockV114State()
        MockV114Handler.state = state
        srv = ThreadingHTTPServer(("127.0.0.1", 0), MockV114Handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{port}"

        with tempfile.TemporaryDirectory() as td:
            work = Path(td)

            # Start live A2A server supporting both JSON-RPC and tasks/sendSubscribe SSE
            a2a_srv_src = r'''@main

#net/http
a2a #"github.com/xulang-org/packages/a2a"

fn handle_a2a_task(tid &string, text &string) string {
    << "worker[" + (*tid) + "]:" + (*text)
}

fn main() int {
    $card := a2a.new_agent_card("xu-swarm-worker", "A2A streaming worker", "http://127.0.0.1", "1.14.0")
    card.add_skill("review", "Code Review", "Reviews Xu modules")
    $asrv := a2a.new_server(card, handle_a2a_task)
    $hs := http.new_server()
    _ := hs.listen("127.0.0.1:0") >>> (err) {
        eprintln(err.msg())
        << 1
    }
    println("A2A_ADDR:" + hs.addr())
    _ := asrv.serve_http_server($&hs) >>> (err) {
        eprintln(err.msg())
        << 1
    }
    << 0
}
'''
            a2a_exe = build_xu(work, "a2a_srv_v114", a2a_srv_src)
            a2a_proc = subprocess.Popen(
                [str(a2a_exe)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                a2a_addr = ""
                for _ in range(100):
                    line = a2a_proc.stdout.readline()
                    if line.startswith("A2A_ADDR:"):
                        a2a_addr = line.strip()[len("A2A_ADDR:"):]
                        break
                self.assertTrue(bool(a2a_addr), "A2A streaming server failed to bind")

                cli_src = r'''@main

#errors
#math
#os
#strings
openai #"github.com/xulang-org/packages/openai"
a2a #"github.com/xulang-org/packages/a2a"
agent #"github.com/xulang-org/packages/agent"
synth #"github.com/xulang-org/packages/synth"
gateway #"github.com/xulang-org/packages/gateway"
rag #"github.com/xulang-org/packages/rag"
eval #"github.com/xulang-org/packages/eval"

fn url_join(base &string, suffix string) string {
    << (*base) + suffix
}

fn approx(a f64, b f64, tol f64) bool {
    << (a - b) <= tol && (b - a) <= tol
}

fn hit_rate_limit(g $&gateway.Gateway, req &openai.ChatRequest) bool {
    _ := g.chat_weighted(req) >>> (_) { << true }
    << false
}

fn run_all(mock_base string, a2a_url string, work_dir string) int {
    oc := openai.new_client(url_join(&mock_base, "/openai"), "ok")

    // 1. A2A SSE streaming (tasks/sendSubscribe) + TaskArtifact + Agent A2A delegation
    ac := a2a.new_client(a2a_url + "")
    evs := ac.send_task_stream("t-sse-1", "inspect kernels") >>>
    if evs.len != 3 || evs[0].state != "submitted" || evs[1].state != "working" || evs[2].state != "completed" || evs[2].artifacts.len != 1 || evs[2].artifacts[0].text != "worker[t-sse-1]:inspect kernels" {
        >>> errors.new("bad A2A SSE stream events")
    }
    skills_txt := agent.discover_a2a_skills_prompt(a2a_url + "") >>>
    if !strings.contains(&skills_txt, "[review] Code Review") {
        >>> errors.new("bad discover_a2a_skills_prompt: " + skills_txt)
    }
    del_out := agent.delegate_to_a2a(a2a_url + "", "", "t-del-1", "audit") >>>
    if del_out != "worker[t-del-1]:audit" {
        >>> errors.new("bad delegate_to_a2a: " + del_out)
    }

    // 2. Agent Swarm / Handoff orchestration
    $team := agent.new_handoff_team("triage")
    team.add_agent("triage", "Classify requests and hand off coding tasks to coder.")
    team.add_agent("coder", "Implement the requested coding task.")
    hres := agent.run_handoff_openai(&team, &oc, "gpt-4o-mini", "Build fast parser", 3) >>>
    if hres.final_agent != "coder" || hres.handoffs != 1 || hres.turns != 2 || hres.route_trace.len != 2 || !strings.contains(&hres.content, "fast parser implemented") {
        >>> errors.new("bad run_handoff_openai")
    }

    // 3. Synth data generation, quality filtering, dedup, DPO/RLVR formatting, and JSONL write
    gen_ex := synth.generate_sft_openai(&oc, "gpt-4o-mini", "Xu language", 2, "Be helpful.") >>>
    if gen_ex.len != 2 || gen_ex[0].instruction != "What is Xu?" {
        >>> errors.new("bad generate_sft_openai")
    }
    dp := synth.generate_dpo_pair_openai(&oc, "strong-model", "weak-model", "Index tuning?") >>>
    if !strings.contains(&dp.chosen, "HNSW") || !strings.contains(&dp.rejected, "Maybe") {
        >>> errors.new("bad generate_dpo_pair_openai")
    }
    d_prompts := ["Summarize VLM support"]
    distilled := synth.distill_dataset_openai(&oc, "teacher-model", &d_prompts, "System") >>>
    if distilled.len != 1 || !strings.contains(&distilled[0].output, "distilled:") {
        >>> errors.new("bad distill_dataset_openai")
    }
    raw_items := ["xu inference engine", "xu inference engine", "alpha beta gamma"]
    d1 := synth.dedup_exact(&raw_items)
    d2 := synth.dedup_near_jaccard(&["one two three four", "one two three five", "completely different"], 2, 0.4)
    if d1.len != 2 || d2.len != 2 {
        >>> errors.new("bad synth dedup")
    }
    $ex_list := [gen_ex[0], synth.SftExample{instruction: "What is Xu?", input: "", output: "Xu is a compiled systems and AI language.", system: "Be helpful."}, synth.SftExample{instruction: "Too short", input: "", output: "no", system: ""}, gen_ex[1]]
    $fstats := synth.FilterStats{total_input: 0, kept: 0, dropped_empty: 0, dropped_duplicate: 0}
    filtered := synth.filter_sft_quality(&ex_list, 5, 0.9, $&fstats)
    if filtered.len != 2 || fstats.dropped_empty != 1 || fstats.dropped_duplicate != 1 {
        >>> errors.new("bad filter_sft_quality")
    }
    sft_jsonl := synth.format_sft_jsonl(&filtered)
    round_sft := synth.parse_sft_jsonl(&sft_jsonl) >>>
    if round_sft.len != 2 || round_sft[0].instruction != "What is Xu?" {
        >>> errors.new("bad parse_sft_jsonl")
    }
    sft_file := work_dir + "/synth_sft.jsonl"
    n_lines := synth.save_jsonl(sft_file + "", &sft_jsonl) >>>
    dpo_txt := synth.format_dpo_jsonl(&[dp])
    rlvr_txt := synth.format_rlvr_jsonl(&[synth.RlvrTask{id: "r1", prompt: "1+1?", expected: "2", verifier: "exact"}])
    if n_lines != 2 || !strings.contains(&dpo_txt, "\"chosen\":") || !strings.contains(&rlvr_txt, "\"verifier\":\"exact\"") {
        >>> errors.new("bad save/format jsonl")
    }

    // 4. Gateway circuit breaker, weighted routing, rate limiting, and Prometheus metrics
    $gw := gateway.new_gateway()
    gw.add_weighted_route("bad", openai.new_client(url_join(&mock_base, "/gw_fail"), "k"), "m-bad", 1.0, 2.0, 1)
    gw.add_weighted_route("good", openai.new_client(url_join(&mock_base, "/gw_ok"), "k"), "m-good", 1.0, 2.0, 2)
    gw.set_circuit_breaker(1, 60000)
    creq := openai.chat_request("", [openai.message("user", "ping")])
    r1 := gw.chat_with_fallback(&creq) >>>
    r2 := gw.chat_with_fallback(&creq) >>>
    if r1.content != "gw-ok:m-good" || r2.content != "gw-ok:m-good" {
        >>> errors.new("bad gateway circuit breaker fallback")
    }
    $gw_w := gateway.new_gateway()
    gw_w.add_weighted_route("w1", openai.new_client(url_join(&mock_base, "/gw_ok"), "k"), "m-a", 1.0, 2.0, 1)
    gw_w.add_weighted_route("w2", openai.new_client(url_join(&mock_base, "/gw_alt"), "k"), "m-b", 1.0, 2.0, 2)
    wa := gw_w.chat_weighted(&creq) >>>
    wb := gw_w.chat_weighted(&creq) >>>
    wc := gw_w.chat_weighted(&creq) >>>
    if wa.model != "m-a" || wb.model != "m-b" || wc.model != "m-b" {
        >>> errors.new("bad chat_weighted distribution: " + wa.model + "," + wb.model + "," + wc.model)
    }
    gw_w.set_rate_limit(1)
    _ := gw_w.chat_weighted(&creq) >>>
    if !hit_rate_limit($&gw_w, &creq) {
        >>> errors.new("expected gateway rate limit error")
    }
    gw_prom := gw.format_metrics_prometheus()
    if !strings.contains(&gw_prom, "xu_gateway_requests_total 2") || !strings.contains(&gw_prom, "xu_gateway_fallbacks_total 1") {
        >>> errors.new("bad gateway Prometheus metrics: " + gw_prom)
    }

    // 5. RAG Contextual Retrieval + Hybrid GraphRAG search
    raw_c := ["pgvector adds HNSW indexes."]
    ctx_c := rag.contextualize_chunks_openai(&oc, "gpt-4o-mini", "Vector DB", "Postgres guide", &raw_c) >>>
    if ctx_c.len != 1 || !strings.contains(&ctx_c[0], "[Context: Explains HNSW graph indexing") {
        >>> errors.new("bad contextualize_chunks_openai")
    }
    $store := rag.new_store(3)
    _ := store.add_chunk("c1", "doc1", ctx_c[0] + "", "pg.md", "{}", &[1.0, 0.0, 0.0]) >>>
    _ := store.add_chunk("c2", "doc2", "Unrelated topic", "misc.md", "{}", &[0.0, 1.0, 0.0]) >>>
    $kg := rag.new_knowledge_graph()
    kg.add_triple("pgvector", "supports", "HNSW", "pg.md")
    kg.add_triple("HNSW", "improves", "Recall", "bench.md")
    hgr := rag.hybrid_graph_vector_search(&store, &kg, "HNSW", &[1.0, 0.0, 0.0], &["pgvector"], 1, 2)
    if hgr.hits.len != 1 || hgr.graph_triples.len != 2 || !strings.contains(&hgr.formatted_context, "Knowledge Graph Context:") {
        >>> errors.new("bad hybrid_graph_vector_search")
    }

    // 6. Eval JSON schema check + ECE + regression gate + Markdown/JSON reporting
    rules := [eval.SchemaFieldRule{name: "name", expected_type: "string", required: true}, eval.SchemaFieldRule{name: "score", expected_type: "number", required: true}, eval.SchemaFieldRule{name: "opt", expected_type: "bool", required: false}]
    sc1 := eval.evaluate_json_schema(&"```json\n{\"name\":\"xu\",\"score\":99}\n```", &rules)
    sc2 := eval.evaluate_json_schema(&"{\"name\":123}", &rules)
    if !sc1.valid || !approx(sc1.pass_rate, 1.0, 1e-9) || sc2.valid || sc2.missing_fields.len != 1 || sc2.type_mismatches.len != 1 {
        >>> errors.new("bad evaluate_json_schema")
    }
    ece := eval.expected_calibration_error(&[0.9, 0.1], &[true, false], 2)
    if !approx(ece, 0.1, 1e-9) {
        >>> errors.new("bad expected_calibration_error")
    }
    base_sum := eval.EvalSummary{total: 10, exact_match_avg: 0.80, token_f1_avg: 0.85, rouge_l_avg: 0.82}
    cand_ok := eval.EvalSummary{total: 10, exact_match_avg: 0.81, token_f1_avg: 0.84, rouge_l_avg: 0.83}
    cand_bad := eval.EvalSummary{total: 10, exact_match_avg: 0.65, token_f1_avg: 0.80, rouge_l_avg: 0.78}
    g_ok := eval.compare_summaries(&base_sum, &cand_ok, 0.02)
    g_bad := eval.compare_summaries(&base_sum, &cand_bad, 0.02)
    if !g_ok.passed || g_bad.passed || !strings.contains(&g_bad.summary, "FAIL") {
        >>> errors.new("bad compare_summaries gate")
    }
    md := eval.format_summary_markdown("v1.14", &cand_ok)
    js := eval.format_summary_json("v1.14", &cand_ok)
    if !strings.contains(&md, "exact_match_avg") || !strings.contains(&js, "\"total\":10") {
        >>> errors.new("bad eval summary format")
    }

    println("V114_ECOSYSTEM_OK")
    << 0
}

fn main() int {
    args := os.argv()
    if args.len < 4 { << 1 }
    run_all(args[1], args[2], args[3]) >>> (err) {
        eprintln(err.msg())
        << 1
    }
    << 0
}
'''
                cli_exe = build_xu(work, "v114_cli", cli_src)
                res = subprocess.run(
                    [str(cli_exe), base, f"http://{a2a_addr}", str(work)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                self.assertIn("V114_ECOSYSTEM_OK", res.stdout)
                # Confirm circuit breaker opened on /gw_fail after 1st failure (not called on 2nd request)
                self.assertEqual(state.primary_fails, 1)

                # 7. Cross-repo E2E: consume synth_sft.jsonl directly inside xu/examples/llm_engine train-lora --dataset!
                sft_jsonl_path = work / "synth_sft.jsonl"
                self.assertTrue(sft_jsonl_path.is_file())
                llm_exe = work / "llm_v114"
                b_res = subprocess.run(
                    [str(XU), "build", str(REPO / "examples/llm_engine/main.xu"), "-o", str(llm_exe)],
                    cwd=REPO,
                    env=xu_env(work),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(b_res.returncode, 0, b_res.stdout + b_res.stderr)
                if os.uname().sysname == "Darwin":
                    subprocess.run(["codesign", "--force", "--sign", "-", str(llm_exe)], check=True, capture_output=True)
                sf_path = work / "v114_base.safetensors"
                lora_out = work / "v114_synth_lora.safetensors"
                tle.write_safetensors(sf_path, tle.CFG, tle.generate_weight_dict(tle.CFG))
                tr = subprocess.run(
                    [
                        str(llm_exe), "train-lora", str(sf_path), str(lora_out),
                        "--dataset", str(sft_jsonl_path), "--steps", "2", "--lora-rank", "4",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(tr.returncode, 0, tr.stdout + tr.stderr)
                self.assertTrue(lora_out.is_file())
                self.assertIn("step:2 loss:", tr.stdout)
                self.assertIn("kv_free_after:4/4", tr.stdout)
            finally:
                a2a_proc.terminate()
                a2a_proc.wait(timeout=5)
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
