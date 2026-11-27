import http.server
import json
import os
import socketserver
import subprocess
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


class MockV111State:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []
        self.eval_round = 0
        self.json_round = 0
        self.cache_calls = 0


class MockV111Handler(http.server.BaseHTTPRequestHandler):
    state = None

    def log_message(self, format, *args):
        return

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n) if n > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _chat_resp(self, text, tool_calls=None, ptoks=100, ctoks=50):
        msg = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {
            "id": "chatcmpl-v111",
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": ptoks,
                "completion_tokens": ctoks,
                "total_tokens": ptoks + ctoks,
            },
        }

    def do_POST(self):
        req = self._read_json()
        st = self.__class__.state
        with st.lock:
            st.calls.append((self.path, req.get("model", "")))

        if self.path == "/primary/v1/chat/completions":
            self._send_json(503, {"error": {"message": "primary overloaded"}})
            return

        if self.path == "/backup/v1/chat/completions":
            msgs = req.get("messages", [])
            last = msgs[-1].get("content", "") if msgs else ""
            if "cacheable:" in last:
                with st.lock:
                    st.cache_calls += 1
                    cnt = st.cache_calls
                self._send_json(200, self._chat_resp(f"cached-reply-{cnt}", ptoks=1000, ctoks=500))
                return
            self._send_json(200, self._chat_resp(f"backup-ok:{req.get('model','')}", ptoks=200, ctoks=100))
            return

        if self.path == "/v1/chat/completions":
            msgs = req.get("messages", [])
            last_msg = msgs[-1] if msgs else {}
            role = last_msg.get("role", "")
            content = last_msg.get("content", "")
            model = req.get("model", "")

            if model == "rag-mq":
                self._send_json(200, self._chat_resp("1. xu tensor simd\n2. xu metal gpu\n3. xu rag hybrid"))
                return

            if model == "rag-hyde":
                self._send_json(200, self._chat_resp("Xu uses #tensor SIMD ops and #metal kernels for fast RAG scoring."))
                return

            if model == "rag-eval":
                self._send_json(200, self._chat_resp('{"context_precision": 0.95, "context_recall": 0.94, "faithfulness": 0.92, "answer_relevancy": 0.88}'))
                return

            if model == "approval-model":
                if role == "tool":
                    results = [m.get("content", "") for m in msgs if m.get("role") == "tool"]
                    self._send_json(200, self._chat_resp("done:" + "|".join(results)))
                    return
                self._send_json(200, self._chat_resp("", tool_calls=[
                    {"id": "tc1", "type": "function", "function": {"name": "safe_tool", "arguments": "{\"x\":1}"}},
                    {"id": "tc2", "type": "function", "function": {"name": "danger_tool", "arguments": "{\"x\":2}"}},
                ]))
                return

            if model == "gen-model":
                if "Revise the draft" in content:
                    self._send_json(200, self._chat_resp("draft v2 with citations"))
                else:
                    self._send_json(200, self._chat_resp("draft v1"))
                return

            if model == "eval-model":
                with st.lock:
                    st.eval_round += 1
                    r = st.eval_round
                if r == 1:
                    self._send_json(200, self._chat_resp("Needs citations"))
                else:
                    self._send_json(200, self._chat_resp("PASS: good draft"))
                return

            if model == "json-model":
                with st.lock:
                    st.json_round += 1
                    jr = st.json_round
                if jr == 1:
                    self._send_json(200, self._chat_resp('{"answer": "partial_only"}'))
                else:
                    self._send_json(200, self._chat_resp('```json\n{"answer":"ok","score":95}\n```'))
                return

            self._send_json(200, self._chat_resp(f"echo:{model}:{content}"))
            return

        self._send_json(404, {"error": "unknown route"})


class V111AIWorkflowsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = MockV111State()
        MockV111Handler.state = cls.state
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), MockV111Handler)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_v111_prompt_gateway_rag_eval_agent_workflows(self):
        base = f"http://127.0.0.1:{self.port}"
        src = f"""@main
#encoding/json
prompt #github.com/xulang-org/packages/prompt
gateway #github.com/xulang-org/packages/gateway
openai #github.com/xulang-org/packages/openai
rag #github.com/xulang-org/packages/rag
eval #github.com/xulang-org/packages/eval
agent #github.com/xulang-org/packages/agent

fn safe_handler(args &string) string {{
    << "ran_safe:" + (*args)
}}

fn danger_handler(args &string) string {{
    << "SHOULD_NOT_RUN"
}}

fn my_approval(name &string, args_json &string) bool {{
    << (*name) == "safe_tool"
}}

fn check_missing_var(reg &prompt.PromptRegistry, qname &string, qver &string, bad_vars &map[string]string) bool {{
    _ := reg.render_messages(qname, qver, bad_vars) >>> (_) {{
        << true
    }}
    << false
}}

fn run_all() int {{
    // 1. prompt registry, required var validation, few-shot, extract_json_block
    $reg := prompt.new_registry()
    $req_vars := []string{{}}
    req_vars.push("topic")
    tpl := prompt.new_template("qa", "v1", "You are a {{{{role}}}} helper.", "Explain {{{{topic}}}}.", &req_vars)
    reg.register(~tpl)
    $vars := map[string]string{{}}
    vars["role"] = "concise"
    vars["topic"] = "SIMD"
    qname := "qa"
    qver := "v1"
    pmsgs := reg.render_messages(&qname, &qver, &vars) >>>
    if pmsgs.len != 2 || pmsgs[1].content != "Explain SIMD." {{
        println("ERR:prompt_render")
        << 1
    }}
    $bad_vars := map[string]string{{}}
    bad_vars["role"] = "x"
    missing_err := check_missing_var(&reg, &qname, &qver, &bad_vars)
    if !missing_err {{
        println("ERR:prompt_missing_check")
        << 1
    }}

    $ex := []prompt.FewShotExample{{}}
    $e1 := []f32{{}}
    e1.push(f32(1))
    e1.push(f32(0))
    $e2 := []f32{{}}
    e2.push(f32(0))
    e2.push(f32(1))
    ex.push(prompt.FewShotExample{{input: "tensor math", output: "fast", embedding: ~e1}})
    ex.push(prompt.FewShotExample{{input: "http server", output: "net", embedding: ~e2}})
    $qe := []f32{{}}
    qe.push(f32(0.99))
    qe.push(f32(0.1))
    picked := prompt.select_few_shot(&ex, &qe, 1)
    if picked.len != 1 || picked[0].input != "tensor math" {{
        println("ERR:prompt_few_shot")
        << 1
    }}
    fs_msgs := prompt.format_few_shot_messages("", &picked, "new q")
    if fs_msgs.len != 3 {{
        println("ERR:prompt_few_shot_msgs")
        << 1
    }}
    jraw := "prefix ```json\\n{{\\"k\\":42}}\\n``` suffix"
    jext := prompt.extract_json_block(&jraw) >>>
    if jext != "{{\\"k\\":42}}" {{
        println("ERR:prompt_extract_json=" + jext)
        << 1
    }}

    // 2. gateway: failover + cost + semantic cache
    $gw := gateway.new_gateway()
    $c_prim := openai.new_client("{base}/primary", "k1")
    c_prim.set_max_retries(0)
    $c_back := openai.new_client("{base}/backup", "k2")
    c_back.set_max_retries(0)
    gw.add_route("primary", ~c_prim, "m-prim", 10.0, 20.0)
    gw.add_route("backup", ~c_back, "m-back", 1.0, 2.0)
    gw.enable_semantic_cache(0.95, 16)

    $msgs1 := []openai.Message{{}}
    msgs1.push(openai.message("user", "hello fallback"))
    req1 := openai.chat_request("ignored", ~msgs1)
    resp1 := gw.chat_with_fallback(&req1) >>>
    if resp1.content != "backup-ok:m-back" {{
        println("ERR:gw_fallback=" + resp1.content)
        << 1
    }}

    $msgs2 := []openai.Message{{}}
    msgs2.push(openai.message("user", "cacheable: explain xu"))
    req2 := openai.chat_request("ignored", ~msgs2)
    $emb_a := []f32{{}}
    emb_a.push(f32(1))
    emb_a.push(f32(0))
    $emb_b := []f32{{}}
    emb_b.push(f32(0.99))
    emb_b.push(f32(0.05))
    r_c1 := gw.chat_cached(&req2, &emb_a) >>>
    r_c2 := gw.chat_cached(&req2, &emb_b) >>>
    if r_c1.content != "cached-reply-1" || r_c2.content != "cached-reply-1" {{
        println("ERR:gw_cache=" + r_c1.content + "|" + r_c2.content)
        << 1
    }}
    gst := gw.get_stats()
    if gst.fallbacks != 2 || gst.cache_hits != 1 || gst.estimated_cost_usd <= 0.0 {{
        println("ERR:gw_stats")
        << 1
    }}

    // 3. rag advanced: Parent-Child, Multi-Query, HyDE, Context Compression, Cited Format
    long_doc := "Xu language has native SIMD tensor ops and Metal kernels. Second paragraph covers async runtime and pure Xu AI packages."
    pc := rag.chunk_parent_child("docA", "srcA", long_doc + "", 120, 55, 10)
    if pc.children.len < 2 {{
        println("ERR:rag_pc_len=" + str(pc.children.len))
        << 1
    }}
    $store := rag.new_store(2)
    for i in 0:pc.children.len {{
        ch := &pc.children[i]
        $v := []f64{{}}
        if i == 0 {{
            v.push(1.0)
            v.push(0.0)
        }} else {{
            v.push(0.2)
            v.push(0.9)
        }}
        store.add_chunk(ch.id + "", ch.doc_id + "", ch.text + "", ch.source + "", ch.metadata_json + "", &v)
    }}
    $qv := []f64{{}}
    qv.push(1.0)
    qv.push(0.0)
    hits := store.search(&qv, 2, "")
    parents := rag.expand_to_parents(&hits, &pc.parents)
    if parents.len != 1 || parents[0].chunk.text != long_doc {{
        println("ERR:rag_expand_parents")
        << 1
    }}
    c_main := openai.new_client("{base}", "k")
    eqs := rag.expand_queries_openai(&c_main, "rag-mq", "xu tensor", 3) >>>
    if eqs.len < 2 {{
        println("ERR:rag_multi_query")
        << 1
    }}
    hyde := rag.hyde_document_openai(&c_main, "rag-hyde", "xu tensor") >>>
    if hyde.len == 0 {{
        println("ERR:rag_hyde")
        << 1
    }}
    cq := "SIMD tensor"
    comp := rag.compress_context(&cq, &hits, 1)
    cited := rag.format_context_cited(&hits)
    if comp.len == 0 || cited.len == 0 {{
        println("ERR:rag_comp_cited")
        << 1
    }}

    // 4. eval: Ragas Triad + Judge
    $ctxs := []string{{}}
    ctxs.push("Xu uses SIMD tensor ops and Metal kernels.")
    ctxs.push("Unrelated weather note.")
    eq := "How does Xu use SIMD tensor ops?"
    eans := "Xu uses SIMD tensor ops."
    egt := "SIMD tensor"
    triad := eval.evaluate_rag_triad(&eq, &eans, &ctxs, &egt)
    if triad.context_recall <= 0.0 || triad.faithfulness <= 0.0 || triad.answer_relevancy <= 0.0 {{
        println("ERR:eval_triad")
        << 1
    }}
    jtriad := eval.evaluate_rag_judge_openai(&c_main, "rag-eval", "What accelerates Xu?", "Xu uses SIMD tensor ops.", &ctxs) >>>
    if jtriad.faithfulness < 0.9 || jtriad.context_precision < 0.9 {{
        println("ERR:eval_judge")
        << 1
    }}

    // 5. agent: approval hook + checkpoint + parallel workflow + evaluator-optimizer + run_json_checked
    $ag_app := agent.new_agent(openai.new_client("{base}", "k"), "approval-model")
    ag_app.add_tool("safe_tool", "safe", "{{\\"type\\":\\"object\\"}}", safe_handler)
    ag_app.add_tool("danger_tool", "danger", "{{\\"type\\":\\"object\\"}}", danger_handler)
    ag_app.set_approval_hook(my_approval)
    res_app := ag_app.run("run both tools") >>>
    if res_app.content != "done:ran_safe:{{\\"x\\":1}}|denied by approval hook" {{
        println("ERR:agent_approval=" + res_app.content)
        << 1
    }}

    $sess := agent.new_session(agent.new_agent(openai.new_client("{base}", "k"), "sess-model"))
    _ := sess.send("turn1") >>>
    ck_path := "/tmp/xu_v111_ckpt.json"
    _ := sess.save_checkpoint_json(ck_path + "") >>>
    $sess2 := agent.load_session_checkpoint(agent.new_agent(openai.new_client("{base}", "k"), "sess-model"), ck_path + "") >>>
    if sess2.history().len != 2 {{
        println("ERR:agent_ckpt_len")
        << 1
    }}
    r_s2 := sess2.send("turn2") >>>
    if r_s2.content != "echo:sess-model:turn2" {{
        println("ERR:agent_ckpt_send=" + r_s2.content)
        << 1
    }}

    $par_agents := []agent.Agent{{}}
    par_agents.push(agent.new_agent(openai.new_client("{base}", "k"), "worker-a"))
    par_agents.push(agent.new_agent(openai.new_client("{base}", "k"), "worker-b"))
    $par_names := []string{{}}
    par_names.push("wa")
    par_names.push("wb")
    $par_prompts := []string{{}}
    par_prompts.push("p1")
    par_prompts.push("p2")
    par_out := agent.run_parallel_workflow(&par_agents, &par_names, &par_prompts) >>>
    if par_out.len != 2 || par_out[0].result.content != "echo:worker-a:p1" || par_out[1].result.content != "echo:worker-b:p2" {{
        println("ERR:agent_parallel")
        << 1
    }}

    ag_gen := agent.new_agent(openai.new_client("{base}", "k"), "gen-model")
    ag_eval := agent.new_agent(openai.new_client("{base}", "k"), "eval-model")
    eores := agent.run_evaluator_optimizer(&ag_gen, &ag_eval, "write note", 4, "PASS") >>>
    if !eores.passed || eores.rounds != 2 || eores.content != "draft v2 with citations" {{
        println("ERR:agent_eval_opt")
        << 1
    }}

    ag_json := agent.new_agent(openai.new_client("{base}", "k"), "json-model")
    $req_keys := []string{{}}
    req_keys.push("answer")
    req_keys.push("score")
    jres := ag_json.run_json_checked("give json", &req_keys, 2) >>>
    if jres.content != "{{\\"answer\\":\\"ok\\",\\"score\\":95}}" {{
        println("ERR:agent_json_checked=" + jres.content)
        << 1
    }}

    println("ALL_V111_OK")
    << 0
}}

fn main() {{
    rc := run_all() >>> (err) {{
        eprintln(err.msg())
        <<< 1
    }}
    if rc != 0 {{
        <<< rc
    }}
}}
"""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            exe = build_xu(work, "v111_workflows", src)
            res = subprocess.run([str(exe)], env=xu_env(work), capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertEqual(res.stdout.strip(), "ALL_V111_OK", res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
