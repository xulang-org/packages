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


class MockV112State:
    def __init__(self):
        self.lock = threading.Lock()
        self.paths = []
        self.files = {}
        self.batches = {}
        self.otlp_payloads = []
        self.cohere_reqs = []


class MockV112Handler(http.server.BaseHTTPRequestHandler):
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

    def _send_text(self, status, text):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        st = self.__class__.state
        with st.lock:
            st.paths.append(("GET", self.path))
        if self.path.startswith("/v1/files/") and self.path.endswith("/content"):
            fid = self.path[len("/v1/files/"):-len("/content")]
            fobj = st.files.get(fid)
            if not fobj:
                self._send_json(404, {"error": {"message": "not found"}})
                return
            self._send_text(200, fobj["raw"])
            return
        if self.path.startswith("/v1/files/"):
            fid = self.path[len("/v1/files/"):]
            fobj = st.files.get(fid)
            if not fobj:
                self._send_json(404, {"error": {"message": "not found"}})
                return
            self._send_json(200, fobj["meta"])
            return
        if self.path.startswith("/v1/files"):
            items = [v["meta"] for v in st.files.values()]
            self._send_json(200, {"object": "list", "data": items})
            return
        if self.path.startswith("/v1/batches/"):
            bid = self.path[len("/v1/batches/"):]
            bobj = st.batches.get(bid)
            if not bobj:
                self._send_json(404, {"error": {"message": "not found"}})
                return
            self._send_json(200, bobj)
            return
        if self.path.startswith("/v1/batches"):
            self._send_json(200, {"object": "list", "data": list(st.batches.values())})
            return
        self._send_json(404, {"error": {"message": "unknown path"}})

    def do_DELETE(self):
        st = self.__class__.state
        with st.lock:
            st.paths.append(("DELETE", self.path))
        if self.path.startswith("/v1/files/"):
            fid = self.path[len("/v1/files/"):]
            existed = fid in st.files
            st.files.pop(fid, None)
            self._send_json(200, {"id": fid, "object": "file", "deleted": existed})
            return
        self._send_json(404, {"error": {"message": "unknown path"}})

    def do_POST(self):
        raw = self._read_raw()
        st = self.__class__.state
        with st.lock:
            st.paths.append(("POST", self.path))

        # OpenAI Files multipart upload
        if self.path == "/v1/files":
            text_body = raw.decode("utf-8", errors="replace")
            # Extract uploaded file content after filename=...\r\nContent-Type:...\r\n\r\n
            marker = "Content-Type: application/octet-stream\r\n\r\n"
            idx = text_body.find(marker)
            assert idx >= 0, text_body
            tail = text_body[idx + len(marker):]
            end_idx = tail.find("\r\n--")
            payload_txt = tail[:end_idx] if end_idx >= 0 else tail
            fid = "file-v112-1"
            meta = {
                "id": fid,
                "object": "file",
                "bytes": len(payload_txt.encode("utf-8")),
                "created_at": 1730000100,
                "filename": "jobs.jsonl",
                "purpose": "batch",
                "status": "processed",
            }
            st.files[fid] = {"meta": meta, "raw": payload_txt}
            self._send_json(200, meta)
            return

        req = json.loads(raw.decode("utf-8")) if raw else {}

        # OpenAI Batches
        if self.path == "/v1/batches":
            bid = "batch-v112-1"
            bobj = {
                "id": bid,
                "object": "batch",
                "endpoint": req.get("endpoint", ""),
                "input_file_id": req.get("input_file_id", ""),
                "output_file_id": "file-out-1",
                "error_file_id": "",
                "completion_window": req.get("completion_window", ""),
                "status": "validating",
                "created_at": 1730000200,
                "request_counts": {"total": 4, "completed": 1, "failed": 0},
            }
            st.batches[bid] = bobj
            self._send_json(200, bobj)
            return

        if self.path.startswith("/v1/batches/") and self.path.endswith("/cancel"):
            bid = self.path[len("/v1/batches/"):-len("/cancel")]
            bobj = dict(st.batches[bid])
            bobj["status"] = "cancelling"
            st.batches[bid] = bobj
            self._send_json(200, bobj)
            return

        # Cohere native v2 API
        if self.path == "/cohere/v2/chat":
            with st.lock:
                st.cohere_reqs.append(("chat", req))
            auth = self.headers.get("Authorization", "")
            assert auth == "Bearer cohere-key", auth
            self._send_json(200, {
                "id": "cohere-chat-1",
                "finish_reason": "COMPLETE",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Cohere v2 ok"}]},
                "usage": {"tokens": {"input_tokens": 11, "output_tokens": 7}},
            })
            return

        if self.path == "/cohere/v2/embed":
            with st.lock:
                st.cohere_reqs.append(("embed", req))
            self._send_json(200, {
                "id": "cohere-emb-1",
                "embeddings": {"float": [[0.5, 0.25, 0.75], [1.0, 0.0, 0.0]]},
            })
            return

        if self.path == "/cohere/v2/rerank":
            with st.lock:
                st.cohere_reqs.append(("rerank", req))
            self._send_json(200, {
                "id": "cohere-rerank-1",
                "results": [
                    {"index": 1, "relevance_score": 0.96},
                    {"index": 0, "relevance_score": 0.42},
                ],
            })
            return

        # Anthropic judge
        if self.path == "/anthropic/v1/messages":
            ans_prompt = req["messages"][0]["content"]
            sc = 0.91 if "Paris is the capital." in ans_prompt else 0.25
            reas = "Accurate" if sc > 0.5 else "Incorrect"
            self._send_json(200, {
                "id": "msg_eval_1",
                "model": req.get("model", ""),
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": f'{{"score": {sc}, "reasoning": "{reas}"}}'}],
                "usage": {"input_tokens": 10, "output_tokens": 8},
            })
            return

        # Gemini judge
        if self.path.startswith("/gemini/v1beta/models/") and ":generateContent" in self.path:
            usr_txt = req["contents"][0]["parts"][0]["text"]
            sc = 0.88 if "Answer:\nParis." in usr_txt else 0.30
            reas = "Solid" if sc > 0.5 else "Weak"
            self._send_json(200, {
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": f'```json\n{{"score": {sc}, "reasoning": "{reas}"}}\n```'}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16},
            })
            return

        # OTLP trace ingest
        if self.path == "/otlp/v1/traces":
            with st.lock:
                st.otlp_payloads.append(req)
            self._send_json(200, {"partialSuccess": {}})
            return

        self._send_json(404, {"error": {"message": f"unexpected path: {self.path}"}})


class TestV112AiEcosystem(unittest.TestCase):
    def test_cohere_files_batches_skills_graph_otlp_judges_pgvector(self):
        state = MockV112State()
        MockV112Handler.state = state
        srv = ThreadingHTTPServer(("127.0.0.1", 0), MockV112Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        try:
            with tempfile.TemporaryDirectory() as td:
                work = Path(td)
                skills_dir = work / "skills"
                (skills_dir / "code_review").mkdir(parents=True)
                (skills_dir / "code_review" / "SKILL.md").write_text(
                    "---\nname: review-helper\ndescription: Inspect pull request security issues\n---\nAlways check auth boundaries.\n"
                )
                (skills_dir / "sql_tuning.md").write_text(
                    "---\nname: sql-tuning\ndescription: Optimize database SQL indexes\n---\nPrefer EXPLAIN ANALYZE.\n"
                )

                src = r'''@main
#errors
#math
#os
#strconv
#strings
openai #github.com/xulang-org/packages/openai
cohere #github.com/xulang-org/packages/cohere
anthropic #github.com/xulang-org/packages/anthropic
gemini #github.com/xulang-org/packages/gemini
agent #github.com/xulang-org/packages/agent
eval #github.com/xulang-org/packages/eval
rag #github.com/xulang-org/packages/rag

fn node_draft(st &string) string {
    << (*st) + "->draft"
}

fn node_refine(st &string) string {
    << (*st) + "|step"
}

fn route_refine(st &string) string {
    if strings.contains(st, "|step|step") { << "END" }
    << "refine"
}

fn node_dummy(st &string) string {
    << *st
}

fn check_bad_graph(g &agent.StateGraph) bool {
    _ := g.run("x", 4) >>> (_) { << true }
    << false
}

fn check_bad_pg() bool {
    _ := rag.new_pgvector_store("bad;drop", 3, "cosine") >>> (_) { << true }
    << false
}

fn url_join(base &string, suffix string) string {
    << (*base) + suffix
}

fn run_all(base_url string, skills_dir string) int {
    // 1. Cohere native client (/v2/chat, /v2/embed, /v2/rerank)
    coh := cohere.new_client_with_base(url_join(&base_url, "/cohere"), "cohere-key")
    cmsg := [cohere.message("user", "Say hi")]
    cresp := coh.chat("command-a", &cmsg, 64, 0.1) >>>
    if cresp.text != "Cohere v2 ok" || cresp.input_tokens != 11 || cresp.output_tokens != 7 {
        >>> errors.new("bad cohere chat: " + cresp.text)
    }
    emb_texts := ["alpha doc", "beta doc"]
    eresp := coh.embed("embed-v4.0", &emb_texts, "search_document") >>>
    if eresp.embeddings.len != 2 || math.abs(eresp.embeddings[0][2] - 0.75) > 1e-6 {
        >>> errors.new("bad cohere embed")
    }
    docs := ["doc zero", "doc one"]
    rr := coh.rerank("rerank-v3.5", "query", &docs, 2) >>>
    if rr.len != 2 || rr[0].index != 1 || rr[0].document != "doc one" || math.abs(rr[0].relevance_score - 0.96) > 1e-6 {
        >>> errors.new("bad cohere rerank")
    }

    // 2. OpenAI new provider presets + Files & Batches API
    qw := openai.new_qwen_client("qk")
    ms := openai.new_mistral_client("mk")
    mm := openai.new_minimax_client("mmk")
    if qw.base_url != "https://dashscope.aliyuncs.com/compatible-mode/v1" ||
       ms.base_url != "https://api.mistral.ai/v1" ||
       mm.base_url != "https://api.minimax.chat/v1" {
        >>> errors.new("bad openai preset base URLs")
    }

    oc := openai.new_client(url_join(&base_url, ""), "oai-key")
    jsonl_payload := "{\"custom_id\":\"r1\",\"method\":\"POST\"}\n"
    fobj := oc.upload_file("jobs.jsonl", "batch", &jsonl_payload) >>>
    if fobj.id != "file-v112-1" || fobj.bytes != jsonl_payload.len {
        >>> errors.new("bad upload_file")
    }
    fgot := oc.get_file(fobj.id + "") >>>
    if fgot.filename != "jobs.jsonl" { >>> errors.new("bad get_file") }
    flist := oc.list_files("batch") >>>
    if flist.len != 1 || flist[0].id != fobj.id { >>> errors.new("bad list_files") }
    fraw := oc.get_file_content(fobj.id + "") >>>
    if fraw != jsonl_payload { >>> errors.new("bad get_file_content: " + fraw) }

    bobj := oc.create_batch(fobj.id + "", "/v1/chat/completions", "24h", "{\"env\":\"test\"}") >>>
    if bobj.id != "batch-v112-1" || bobj.status != "validating" || bobj.request_total != 4 {
        >>> errors.new("bad create_batch")
    }
    bgot := oc.get_batch(bobj.id + "") >>>
    if bgot.output_file_id != "file-out-1" { >>> errors.new("bad get_batch") }
    blist := oc.list_batches(10) >>>
    if blist.len != 1 { >>> errors.new("bad list_batches") }
    bcan := oc.cancel_batch(bobj.id + "") >>>
    if bcan.status != "cancelling" { >>> errors.new("bad cancel_batch") }
    deleted := oc.delete_file(fobj.id + "") >>>
    if !deleted { >>> errors.new("bad delete_file") }

    // 3. Agent skills directory discovery + StateGraph engine + OTLP trace export
    found_skills := agent.discover_skills_dir(skills_dir + "") >>>
    if found_skills.len != 2 || found_skills[0].name != "review-helper" || found_skills[1].name != "sql-tuning" {
        >>> errors.new("bad discover_skills_dir: " + str(found_skills.len))
    }
    $ag := agent.new_agent(oc.clone_client(), "gpt-4o-mini")
    loaded_cnt := ag.load_skills_dir(skills_dir + "") >>>
    if loaded_cnt != 2 { >>> errors.new("bad load_skills_dir count") }
    prompt_hit := agent.format_skills_prompt(&found_skills, &"Please check security of auth boundaries")
    if !strings.contains(&prompt_hit, "review-helper") {
        >>> errors.new("skill prompt injection failed")
    }

    $g := agent.new_state_graph("draft")
    g.add_node("draft", node_draft)
    g.add_node("refine", node_refine)
    g.add_edge("draft", "refine")
    g.add_conditional_edge("refine", route_refine)
    gres := g.run("init", 10) >>>
    if !gres.finished || gres.steps.len != 3 || gres.last_node != "refine" || gres.final_state != "init->draft|step|step" {
        >>> errors.new("bad StateGraph run: " + gres.final_state)
    }

    $bad_g := agent.new_state_graph("missing_entry")
    bad_g.add_node("ok", node_dummy)
    if !check_bad_graph(&bad_g) { >>> errors.new("missing node should fail") }

    spans := [agent.TraceSpan{trace_id: "trace-v112", kind: "llm", name: "turn_1", turn: 1, duration_ms: 42, tokens: 120, is_error: false, detail: "ok"}]
    otlp_json := agent.format_trace_otlp_json("xu-test-svc", &spans)
    if !strings.contains(&otlp_json, "resourceSpans") || !strings.contains(&otlp_json, "xu-test-svc") {
        >>> errors.new("bad OTLP JSON")
    }
    otlp_st := agent.export_traces_otlp_http(url_join(&base_url, "/otlp/v1/traces"), "xu-test-svc", &spans) >>>
    if otlp_st != 200 { >>> errors.new("bad OTLP status: " + str(otlp_st)) }

    // 4. Eval Anthropic & Gemini judges
    ac := anthropic.new_client_with_base(url_join(&base_url, "/anthropic"), "anth-key")
    aj1 := eval.judge_anthropic(&ac, "claude-3-7", "Capital of France?", "Paris is the capital.", "Mention Paris", 0.7) >>>
    aj2 := eval.judge_anthropic(&ac, "claude-3-7", "Capital of France?", "Lyon.", "Mention Paris", 0.7) >>>
    if !aj1.passed || aj1.score < 0.85 || aj2.passed || aj2.score > 0.4 {
        >>> errors.new("bad judge_anthropic")
    }

    gc := gemini.new_client_with_base(url_join(&base_url, "/gemini"), "gem-key")
    gj1 := eval.judge_gemini(&gc, "gemini-2.5-flash", "Capital of France?", "Paris.", "Mention Paris", 0.7) >>>
    gj2 := eval.judge_gemini(&gc, "gemini-2.5-flash", "Capital of France?", "Nice.", "Mention Paris", 0.7) >>>
    if !gj1.passed || gj1.score < 0.85 || gj2.passed || gj2.score > 0.4 {
        >>> errors.new("bad judge_gemini")
    }

    // 5. RAG PgVectorStore SQL adapter
    pgv := rag.new_pgvector_store("kb_chunks", 3, "cosine") >>>
    ddl := pgv.create_table_sql()
    idx_sql := pgv.create_index_sql("idx_kb_emb", "hnsw") >>>
    ch := rag.Chunk{id: "c1", doc_id: "d1", text: "O'Reilly guide", source: "book", metadata_json: "{\"p\":1}", embedding: [f32(0.1), f32(0.2), f32(0.3)]}
    ups := pgv.upsert_chunk_sql(&ch) >>>
    qvec := [0.1, 0.2, 0.3]
    qsql := pgv.search_sql(&qvec, 5, "d1") >>>
    if !strings.contains(&ddl, "embedding vector(3) NOT NULL") ||
       !strings.contains(&idx_sql, "USING hnsw (embedding vector_cosine_ops)") ||
       !strings.contains(&ups, "'O''Reilly guide'") ||
       !strings.contains(&qsql, "embedding <=>") ||
       !strings.contains(&qsql, "WHERE doc_id = 'd1'") ||
       !strings.contains(&qsql, "LIMIT 5") {
        >>> errors.new("bad PgVectorStore SQL")
    }

    if !check_bad_pg() { >>> errors.new("expected invalid identifier rejection") }

    println("V112_ECOSYSTEM_OK")
    << 0
}

fn main() int {
    raw_args := os.argv()
    if raw_args.len < 3 { << 1 }
    run_all(raw_args[1], raw_args[2]) >>> (err) {
        eprintln(err.msg())
        << 1
    }
    << 0
}
'''
                exe = build_xu(work, "v112_eco", src)
                res = subprocess.run([str(exe), base, str(skills_dir)], capture_output=True, text=True, check=False)
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                self.assertIn("V112_ECOSYSTEM_OK", res.stdout)
                self.assertEqual(len(state.otlp_payloads), 1)
                self.assertEqual(
                    state.otlp_payloads[0]["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"],
                    "xu-test-svc",
                )
                self.assertEqual([r[0] for r in state.cohere_reqs], ["chat", "embed", "rerank"])
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
