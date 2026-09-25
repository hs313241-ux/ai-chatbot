"""
AI Chatbot - Backend (Flask + Google Gemini API)
Ab yeh cloud pe deploy ho sakta hai kyunke local Ollama model ki zaroorat nahi.

Run karne ke liye (local test):
    set GEMINI_API_KEY=your_key_here      (Windows PowerShell: $env:GEMINI_API_KEY="your_key_here")
    python3 server.py

Browser mein khulega: http://localhost:5000

Extra installs needed:
    pip install flask google-genai ddgs
"""

from flask import Flask, request, Response, jsonify, send_from_directory
from google import genai
from google.genai import types
import json
import os
import uuid

app = Flask(__name__, static_folder="static")

MODEL_NAME = "gemini-3.6-flash"
DATA_FILE = "chats_data.json"

# API key environment variable se aati hai (kabhi bhi code mein seedha mat likhna)
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """You are a friendly and helpful AI assistant. 
Always respond in English by default, unless the user writes to you in a 
different language (like Roman Urdu/Hindi) - in that case, match their language.
Explain things clearly and simply. Keep answers concise unless the user asks 
for more detail."""


# ============================================
# Simple JSON file storage (chats persist between restarts)
# Note: Render ke free tier pe yeh file restart hone par reset ho sakti hai
# ============================================
def load_chats():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_chats(chats):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


chats = load_chats()


# ============================================
# Serve the frontend
# ============================================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ============================================
# Chat list management
# ============================================
@app.route("/api/chats", methods=["GET"])
def get_chats():
    chat_list = [
        {"id": cid, "title": c["title"]}
        for cid, c in sorted(chats.items(), key=lambda x: x[1].get("order", 0))
    ]
    return jsonify(chat_list)


@app.route("/api/chats", methods=["POST"])
def create_chat():
    chat_id = str(uuid.uuid4())[:8]
    chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "uploaded_context": None,
        "uploaded_filename": None,
        "order": len(chats),
    }
    save_chats(chats)
    return jsonify({"id": chat_id, "title": "New chat"})


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "not found"}), 404
    return jsonify(chats[chat_id])


@app.route("/api/chats/<chat_id>", methods=["PUT"])
def rename_chat(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "not found"}), 404
    new_title = request.json.get("title", "").strip()
    if new_title:
        chats[chat_id]["title"] = new_title
        save_chats(chats)
    return jsonify({"ok": True})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    if chat_id in chats:
        del chats[chat_id]
        save_chats(chats)
    return jsonify({"ok": True})


# ============================================
# File upload (attach to a chat)
# ============================================
@app.route("/api/upload/<chat_id>", methods=["POST"])
def upload_file(chat_id):
    if chat_id not in chats:
        return jsonify({"error": "not found"}), 404
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400
    content = file.read().decode("utf-8", errors="ignore")
    chats[chat_id]["uploaded_context"] = content[:6000]
    chats[chat_id]["uploaded_filename"] = file.filename
    save_chats(chats)
    return jsonify({"ok": True, "filename": file.filename})


# ============================================
# Web search helper
# ============================================
def do_web_search(query, max_results=3):
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        summary = "Web search results:\n"
        for r in results:
            summary += f"- {r.get('title', '')}: {r.get('body', '')}\n"
        return summary
    except Exception as e:
        return f"[Web search error: {e}]"


# ============================================
# Main chat endpoint - streams the response back
# ============================================
@app.route("/api/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify({"error": "GEMINI_API_KEY not set on server"}), 500

    data = request.json
    chat_id = data.get("chat_id")
    user_message = data.get("message", "")
    web_search_enabled = data.get("web_search", False)

    if chat_id not in chats:
        return jsonify({"error": "invalid chat_id"}), 400

    current_chat = chats[chat_id]
    current_chat["messages"].append({"role": "user", "content": user_message})

    # Auto-title the chat
    if current_chat["title"] == "New chat" and user_message:
        current_chat["title"] = user_message[:30] + ("..." if len(user_message) > 30 else "")

    # Build the system instruction (Gemini takes ONE system instruction string,
    # so file context + search results get merged into it)
    system_instruction = SYSTEM_PROMPT

    if current_chat.get("uploaded_context"):
        system_instruction += f"\n\nReference document ({current_chat['uploaded_filename']}):\n{current_chat['uploaded_context']}"

    if web_search_enabled:
        search_summary = do_web_search(user_message)
        if search_summary:
            system_instruction += f"\n\n{search_summary}"

    # Build conversation history in Gemini's format (role: user/model)
    contents = []
    for m in current_chat["messages"]:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    def generate():
        full_response = ""
        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
            )
            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
        except Exception as e:
            error_msg = f"\n\n⚠️ Error: {e}"
            full_response += error_msg
            yield error_msg
        finally:
            current_chat["messages"].append({"role": "assistant", "content": full_response})
            save_chats(chats)

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
