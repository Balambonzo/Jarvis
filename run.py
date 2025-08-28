#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local JARVIS backend for Llama 3.2 via Ollama with persistent memory and
filtered action handling.

- Serves index.html (visit http://127.0.0.1:5000)
- /api/chat: non-streaming chat (returns a human-friendly announcement only;
  action JSON blocks are processed server-side and removed from the spoken text)
- /api/stream_chat: streaming chat (streams assistant text up until any JSON
  action block; the JSON action block is parsed server-side)
- Uses Ollama's REST API at http://localhost:11434
- Configure model via env OLLAMA_MODEL (default: "llama3.2")
- Memory is stored in memory.json and injected into the system prompt
"""

import os
import json
import requests
import traceback
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from calendar_auth import get_upcoming_events  # <-- Google Calendar

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = APP_DIR
MEMORY_FILE = os.path.join(APP_DIR, "memory.json")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"

# ---------------- MEMORY ----------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(memory_obj):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory_obj, f, indent=2)
    except Exception as e:
        print("Error saving memory:", e)

memory = load_memory()

def build_system_prompt():
    return (
        "You are JARVIS, a precise, courteous and slightly witty engineering assistant for a workshop. "
        "Always address the human user as 'sir' (e.g., 'Yes, sir' or 'Certainly, sir'), and keep responses concise. "
        "When the user provides personal information that should be remembered (name, birthday, preferences, calendar items, tasks), "
        "include a JSON object in your reply of the exact form: { \"memory_update\": { \"key\": \"value\" }, \"announce\": \"...\" }\n\n"
        "When you want the client to perform a real-world action (open a URL, toggle a device, navigate a robot), include a JSON object in your reply like: "
        "{ \"action\": \"open_url\", \"url\": \"https://youtube.com\", \"announce\": \"Opening YouTube, sir.\" }\n\n"
        "IMPORTANT: The backend will parse and APPLY any such JSON object and will NOT read it aloud. "
        "Instead, only the 'announce' text (or a short human-friendly substitute) will be returned/sent to the user.\n\n"
        f"Current persistent memory: {json.dumps(memory)}"
    )

# ---------------- FLASK APP ----------------
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")

def to_ollama_messages(client_messages):
    out = [{"role": "system", "content": build_system_prompt()}]
    for m in client_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out

def extract_action_and_human_text(full_text: str):
    if not full_text:
        return None, full_text
    first = full_text.find("{")
    last = full_text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None, full_text
    candidate = full_text[first:last + 1]
    try:
        parsed = json.loads(candidate)
        human = full_text[:first].strip()
        return parsed, human
    except Exception:
        return None, full_text

def handle_action_json(action_json):
    try:
        if not isinstance(action_json, dict):
            return None
        if "memory_update" in action_json and isinstance(action_json["memory_update"], dict):
            mem = action_json["memory_update"]
            memory.update(mem)
            save_memory(memory)
            return action_json.get("announce") or "Memory updated, sir."
        if action_json.get("action") == "open_url":
            return action_json.get("announce") or "Opening the requested website, sir."
        if "announce" in action_json:
            return action_json.get("announce")
        return "Action completed, sir."
    except Exception:
        traceback.print_exc()
        return "I performed the requested action, sir."

# ---------------- CHAT ENDPOINT ----------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages") or []

    if not messages:
        return jsonify({"content": "No message received, sir.", "action": None, "raw": {}})

    # --- Google Calendar interception ---
    user_message = messages[-1]["content"].lower()
    if any(kw in user_message for kw in ["calendar", "events", "schedule", "agenda"]):
        try:
            events_text = get_upcoming_events(max_results=5)
            return jsonify({"content": events_text, "action": None, "raw": {}})
        except Exception as e:
            return jsonify({"content": f"Failed to fetch calendar: {e}", "action": None, "raw": {}})

    # --- LLM fallback ---
    temperature = float(data.get("temperature", 0.2))
    top_p = float(data.get("top_p", 0.9))

    payload = {
        "model": OLLAMA_MODEL,
        "messages": to_ollama_messages(messages),
        "options": {"temperature": temperature, "top_p": top_p},
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=600)
        r.raise_for_status()
        resp = r.json()
        content = (resp.get("message") or {}).get("content", "")

        action_json, human_text = extract_action_and_human_text(content)
        result_text = handle_action_json(action_json) if action_json else content or "I have nothing to add, sir."

        return jsonify({"content": result_text, "action": action_json, "raw": resp})
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

# ---------------- STREAM CHAT (unchanged) ----------------
@app.route("/api/stream_chat", methods=["POST"])
def stream_chat():
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages") or []
    temperature = float(data.get("temperature", 0.2))
    top_p = float(data.get("top_p", 0.9))
    payload = {"model": OLLAMA_MODEL, "messages": to_ollama_messages(messages), "options": {"temperature": temperature, "top_p": top_p}, "stream": True}

    def event_stream():
        try:
            with requests.post(OLLAMA_CHAT_URL, json=payload, stream=True, timeout=600) as r:
                r.raise_for_status()
                full_text = ""
                yielded_until = 0
                json_started = False
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    delta = msg.get("content", "") or ""
                    full_text += delta
                    if not json_started:
                        idx = full_text.find("{")
                        if idx == -1:
                            to_yield = full_text[yielded_until:]
                            if to_yield:
                                yield f"data: {json.dumps({'delta': to_yield})}\n\n"
                                yielded_until = len(full_text)
                        else:
                            pre_json = full_text[:idx]
                            if len(pre_json) > yielded_until:
                                to_yield = pre_json[yielded_until:]
                                yield f"data: {json.dumps({'delta': to_yield})}\n\n"
                                yielded_until = len(pre_json)
                            json_started = True
                    if chunk.get("done"):
                        action_json, human_text = extract_action_and_human_text(full_text)
                        final_text = handle_action_json(action_json) if action_json else (human_text if human_text else full_text)
                        yield f"data: {json.dumps({'delta': final_text, 'done': True, 'action': action_json})}\n\n"
                        break
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")

# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": OLLAMA_MODEL, "ollama": OLLAMA_BASE, "memory_loaded": bool(memory)})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True)
