import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from llm_client import LLMClient
from memory_store import MemoryStore


def load_env_files() -> None:
    for filename in [".env", ".env.example"]:
        path = Path(filename)
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                os.environ.setdefault(key, value)


class ChatEngine:
    def __init__(self) -> None:
        load_env_files()
        self.memory_store = MemoryStore(path="memories.json")
        self.llm = LLMClient()
        self.history: List[Dict[str, str]] = []
        self.lock = threading.Lock()

    def _context_facts(self, message: str) -> List[str]:
        relevant = self.memory_store.search(message, top_k=5, min_score=0.0)
        relevant_facts = [m.fact for m in relevant]
        all_facts = [m.fact for m in self.memory_store.memories]
        out: List[str] = []
        for fact in relevant_facts + all_facts:
            if fact not in out:
                out.append(fact)
        return out

    def new_session(self) -> str:
        with self.lock:
            self.history = []
        return "Started a new session. I still remember saved facts."

    def clear_memories(self) -> str:
        with self.lock:
            self.memory_store.clear()
            self.history = []
        return "Cleared all saved memories."

    def list_memories(self) -> List[str]:
        with self.lock:
            return [m.fact for m in self.memory_store.memories]

    def handle_message(self, message: str) -> Dict[str, object]:
        user_message = message.strip()
        if not user_message:
            return {"reply": "Please type a message.", "saved": 0}

        normalized_command = user_message.lower().strip()
        command_aliases = {
            "new": "/new",
            "mem": "/mem",
            "clear": "/clear",
            "quit": "/quit",
            "exit": "/quit",
        }
        if normalized_command in command_aliases:
            user_message = command_aliases[normalized_command]

        if user_message == "/new":
            return {"reply": self.new_session(), "saved": 0}
        if user_message == "/clear":
            return {"reply": self.clear_memories(), "saved": 0}
        if user_message == "/mem":
            memories = self.list_memories()
            if not memories:
                return {"reply": "No memories saved yet.", "saved": 0}
            return {"reply": "Stored memories:\n- " + "\n- ".join(memories), "saved": 0}
        if user_message == "/quit":
            return {"reply": "Use your browser tab to close the app.", "saved": 0}

        with self.lock:
            context_facts = self._context_facts(user_message)
            self.history.append({"role": "user", "content": user_message})
            reply = self.llm.generate_reply(user_message, context_facts, self.history)
            self.history.append({"role": "assistant", "content": reply})

            extracted = self.llm.extract_facts(user_message)
            added = self.memory_store.add_facts(extracted, source_user_message=user_message)
            return {"reply": reply, "saved": added}


ENGINE = ChatEngine()


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Memory Chatbot UI</title>
  <style>
    :root {
      --bg1: #f7f7f7;
      --bg2: #e9e9e9;
      --panel: #ffffff;
      --ink: #0b0b0b;
      --accent: #111111;
      --accent2: #000000;
      --terminal-bg: #050505;
      --terminal-ink: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      min-height: 100vh;
      background: linear-gradient(120deg, var(--bg1), var(--bg2));
      overflow-x: hidden;
    }
    .blob {
      position: fixed;
      border-radius: 999px;
      filter: blur(1px);
      opacity: 0.4;
      z-index: 0;
      animation: float 8s ease-in-out infinite;
    }
    .blob.one { width: 220px; height: 220px; background: #d0d0d0; top: 4%; left: 6%; }
    .blob.two { width: 280px; height: 280px; background: #d5d5d5; top: 56%; right: 8%; animation-delay: -3s; }
    .blob.three { width: 180px; height: 180px; background: #c6c6c6; bottom: 8%; left: 20%; animation-delay: -5s; }
    @keyframes float {
      0%, 100% { transform: translateY(0px) translateX(0px); }
      50% { transform: translateY(-18px) translateX(10px); }
    }
    .wrap {
      position: relative;
      z-index: 1;
      max-width: 980px;
      margin: 28px auto;
      padding: 14px;
    }
    .card {
      background: var(--panel);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.1);
      padding: 18px;
      border: 2px solid #111;
    }
    h1 {
      margin: 6px 0 10px;
      font-size: 30px;
      line-height: 1.2;
      letter-spacing: 0.5px;
    }
    .sub { margin: 0 0 14px; opacity: 0.85; }
    .switches {
      display: inline-flex;
      gap: 8px;
      padding: 6px;
      border-radius: 999px;
      background: #ececec;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .sw {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
      font-weight: 700;
      background: transparent;
    }
    .sw.active {
      background: var(--accent2);
      color: #fff;
      transform: scale(1.02);
    }
    .panel {
      border-radius: 16px;
      padding: 12px;
      min-height: 370px;
      max-height: 54vh;
      overflow: auto;
      margin-bottom: 12px;
      border: 2px dashed rgba(0,0,0,0.08);
    }
    .panel.terminal {
      background: var(--terminal-bg);
      color: var(--terminal-ink);
      font-family: "Courier New", monospace;
    }
    .panel.simple {
      background: #ffffff;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .input {
      flex: 1 1 520px;
      min-width: 220px;
      border: 2px solid #111;
      border-radius: 12px;
      padding: 12px;
      font-size: 16px;
      color: #000;
      background: #fff;
      outline: none;
    }
    .btn {
      border: 0;
      border-radius: 12px;
      padding: 12px 16px;
      font-weight: 800;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      box-shadow: 0 6px 0 #2d2d2d;
      transition: transform 0.12s ease;
    }
    .btn:active { transform: translateY(2px); box-shadow: 0 4px 0 #2d2d2d; }
    .btn.alt { background: #303030; box-shadow: 0 6px 0 #151515; }
    .entry {
      margin: 10px 0;
      animation: pop 0.22s ease;
    }
    @keyframes pop {
      from { transform: scale(0.96); opacity: 0; }
      to { transform: scale(1.0); opacity: 1; }
    }
    .label { font-weight: 800; margin-right: 6px; }
    .bubble {
      display: inline-block;
      padding: 10px 12px;
      border-radius: 14px;
      max-width: 90%;
      white-space: pre-wrap;
    }
    .user .bubble { background: #f0f0f0; color: #000; border: 1px solid #cfcfcf; }
    .bot .bubble { background: #ffffff; color: #000; border: 1px solid #cfcfcf; }
    .panel.terminal .bubble { background: #000; color: #fff; border: 1px solid #333; }
    .tiny { font-size: 12px; opacity: 0.75; margin-top: 4px; }
  </style>
</head>
<body>
  <div class="blob one"></div>
  <div class="blob two"></div>
  <div class="blob three"></div>

  <div class="wrap">
    <div class="card">
      <h1>AI Memory Chatbot</h1>
      <p class="sub">Switch between a terminal-style chat and a simple chat UI. Both use the same saved memory.</p>

      <div class="switches">
        <button class="sw active" id="terminalSwitch" onclick="setMode('terminal')">Terminal View</button>
        <button class="sw" id="simpleSwitch" onclick="setMode('simple')">Simple UI</button>
      </div>

      <div id="chatPanel" class="panel terminal"></div>

      <div class="row">
        <input id="userInput" class="input" placeholder="Type your message..." />
        <button class="btn" onclick="askBot()">Ask Bot</button>
        <button class="btn alt" onclick="newSession()">New Session</button>
        <button class="btn alt" onclick="showMemories()">Show Memories</button>
        <button class="btn alt" onclick="clearMemories()">Clear Memories</button>
      </div>
    </div>
  </div>

  <script>
    let mode = "terminal";

    function setMode(next) {
      mode = next;
      const panel = document.getElementById("chatPanel");
      const termBtn = document.getElementById("terminalSwitch");
      const simpleBtn = document.getElementById("simpleSwitch");
      if (mode === "terminal") {
        panel.classList.remove("simple");
        panel.classList.add("terminal");
        termBtn.classList.add("active");
        simpleBtn.classList.remove("active");
      } else {
        panel.classList.remove("terminal");
        panel.classList.add("simple");
        simpleBtn.classList.add("active");
        termBtn.classList.remove("active");
      }
    }

    function addMessage(role, text, saved = 0) {
      const panel = document.getElementById("chatPanel");
      const entry = document.createElement("div");
      entry.className = "entry " + (role === "You" ? "user" : "bot");
      const bubble = document.createElement("div");
      bubble.className = "bubble";

      if (mode === "terminal") {
        bubble.textContent = role + ": " + text;
      } else {
        bubble.innerHTML = "<span class='label'>" + role + ":</span>" + text.replace(/\\n/g, "<br>");
      }
      entry.appendChild(bubble);

      if (saved > 0 && role === "AI") {
        const tiny = document.createElement("div");
        tiny.className = "tiny";
        tiny.textContent = "[memory] saved " + saved + " new fact(s)";
        entry.appendChild(tiny);
      }

      panel.appendChild(entry);
      panel.scrollTop = panel.scrollHeight;
    }

    async function sendMessage(message) {
      addMessage("You", message);
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });
      const data = await res.json();
      addMessage("AI", data.reply || "No response", data.saved || 0);
    }

    async function askBot() {
      const input = document.getElementById("userInput");
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      await sendMessage(text);
    }

    async function newSession() {
      const res = await fetch("/api/new", { method: "POST" });
      const data = await res.json();
      addMessage("AI", data.reply);
    }

    async function clearMemories() {
      const res = await fetch("/api/clear", { method: "POST" });
      const data = await res.json();
      addMessage("AI", data.reply);
    }

    async function showMemories() {
      const res = await fetch("/api/mem");
      const data = await res.json();
      if (!data.memories || data.memories.length === 0) {
        addMessage("AI", "No memories saved yet.");
        return;
      }
      addMessage("AI", "Stored memories:\\n- " + data.memories.join("\\n- "));
    }

    document.getElementById("userInput").addEventListener("keydown", function(ev) {
      if (ev.key === "Enter") askBot();
    });

    addMessage("AI", "Ready. Type a message and click Ask Bot.");
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/mem":
            self._send_json(200, {"memories": ENGINE.list_memories()})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            body = {}

        if parsed.path == "/api/chat":
            message = str(body.get("message", ""))
            out = ENGINE.handle_message(message)
            self._send_json(200, out)
            return
        if parsed.path == "/api/new":
            self._send_json(200, {"reply": ENGINE.new_session()})
            return
        if parsed.path == "/api/clear":
            self._send_json(200, {"reply": ENGINE.clear_memories()})
            return

        self._send_json(404, {"error": "Not found"})

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("AI Memory Chatbot UI running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
