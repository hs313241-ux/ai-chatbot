"""
AI Chatbot - Backend (Flask + Google Gemini API)
Features: email/password login, Google login, per-account private chats,
file attach, web search, auto-retry on busy errors.

Run karne ke liye (local test):
    $env:GEMINI_API_KEY="your_key_here"
    $env:GOOGLE_CLIENT_ID="your_google_client_id"
    $env:GOOGLE_CLIENT_SECRET="your_google_client_secret"
    python3 server.py

Browser mein khulega: http://localhost:5000

Extra installs needed:
    pip install flask google-genai ddgs Authlib
"""

from flask import Flask, request, Response, jsonify, send_from_directory, session, redirect, url_for
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from authlib.integrations.flask_client import OAuth
from google import genai
from google.genai import types
import json
import os
import re
import sqlite3
import time
import uuid

app = Flask(__name__, static_folder="static")

# Railway (aur zyada tar cloud hosts) ek proxy ke peeche hote hain, isliye
# Flask ko batana zaroori hai ke asal request HTTPS thi - warna Google OAuth
# redirect URL "http://" ban jayega aur Google mismatch error dega.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Login session cookie se track hota hai.
# Railway pe SECRET_KEY env var zaroor set karo taake restart ke baad bhi
# log-in valid rahe (na ho to sabko dobara login karna padega - data safe
# rehta hai, bas session expire ho jati hai).
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24).hex()

MODEL_NAME = "gemini-3.6-flash"
DATA_FILE = "chats_data.json"
DB_FILE = "users.db"

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """You are a friendly and helpful AI assistant. 
Always respond in English by default, unless the user writes to you in a 
different language (like Roman Urdu/Hindi) - in that case, match their language.
Explain things clearly and simply. Keep answers concise unless the user asks 
for more detail."""


# ============================================
# Google OAuth setup
# ============================================
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ============================================
# User database (SQLite)
# ============================================
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE,
            password_hash TEXT,
            google_id TEXT UNIQUE
        )
    """)
    conn.commit()
    conn.close()


init_db()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_user_email(name, email, password):
    conn = get_db()
    try:
        uid = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO users (id, name, email, password_hash, google_id) VALUES (?, ?, ?, ?, NULL)",
            (uid, name, email.lower(), generate_password_hash(password)),
        )
        conn.commit()
        return uid
    except sqlite3.IntegrityError:
        return None  # email already registered
    finally:
        conn.close()


def get_user_by_email(email):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    conn.close()
    return row


def get_user_by_id(uid):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    conn.close()
    return row


def get_or_create_google_user(email, name, google_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE google_id = ?", (google_id,)).fetchone()
    if row:
        conn.close()
        return row["id"]

    row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
    if row:
        conn.execute("UPDATE users SET google_id = ? WHERE id = ?", (google_id, row["id"]))
        conn.commit()
        conn.close()
        return row["id"]

    uid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO users (id, name, email, password_hash, google_id) VALUES (?, ?, ?, NULL, ?)",
        (uid, name, email.lower(), google_id),
    )
    conn.commit()
    conn.close()
    return uid


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


# ============================================
# Auth pages
# ============================================
@app.route("/login")
def login_page():
    if session.get("user_id"):
        return redirect("/")
    return send_from_directory("static", "login.html")


@app.route("/signup")
def signup_page():
    if session.get("user_id"):
        return redirect("/")
    return send_from_directory("static", "signup.html")


@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "All fields are required."}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    uid = create_user_email(name, email, password)
    if uid is None:
        return jsonify({"error": "An account with this email already exists."}), 409

    session["user_id"] = uid
    return jsonify({"ok": True})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    row = get_user_by_email(email)
    if not row or not row["password_hash"] or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Incorrect email or password."}), 401

    session["user_id"] = row["id"]
    return jsonify({"ok": True})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
def api_me():
    if not session.get("user_id"):
        return jsonify({"error": "unauthorized"}), 401
    row = get_user_by_id(session["user_id"])
    if not row:
        session.clear()
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"name": row["name"], "email": row["email"]})


@app.route("/auth/google")
def auth_google():
    redirect_uri = url_for("auth_google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def auth_google_callback():
    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo")
        email = userinfo["email"]
        name = userinfo.get("name") or email.split("@")[0]
        google_id = userinfo["sub"]
    except Exception:
        return redirect("/login?error=google_failed")

    uid = get_or_create_google_user(email, name, google_id)
    session["user_id"] = uid
    return redirect("/")


# ============================================
# Chat data storage (JSON file)
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


def get_owned_chat(chat_id):
    chat = chats.get(chat_id)
    if not chat or chat.get("owner") != session.get("user_id"):
        return None
    return chat


# ============================================
# Serve the chat app (protected)
# ============================================
@app.route("/")
def index():
    if not session.get("user_id"):
        return redirect("/login")
    return send_from_directory("static", "index.html")


# ============================================
# Chat list management
# ============================================
@app.route("/api/chats", methods=["GET"])
@login_required
def get_chats():
    uid = session["user_id"]
    chat_list = [
        {"id": cid, "title": c["title"]}
        for cid, c in sorted(chats.items(), key=lambda x: x[1].get("order", 0))
        if c.get("owner") == uid
    ]
    return jsonify(chat_list)


@app.route("/api/chats", methods=["POST"])
@login_required
def create_chat():
    chat_id = str(uuid.uuid4())[:8]
    chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "uploaded_context": None,
        "uploaded_filename": None,
        "order": len(chats),
        "owner": session["user_id"],
    }
    save_chats(chats)
    return jsonify({"id": chat_id, "title": "New chat"})


@app.route("/api/chats/<chat_id>", methods=["GET"])
@login_required
def get_chat(chat_id):
    chat = get_owned_chat(chat_id)
    if chat is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(chat)


@app.route("/api/chats/<chat_id>", methods=["PUT"])
@login_required
def rename_chat(chat_id):
    chat = get_owned_chat(chat_id)
    if chat is None:
        return jsonify({"error": "not found"}), 404
    new_title = (request.json or {}).get("title", "").strip()
    if new_title:
        chat["title"] = new_title
        save_chats(chats)
    return jsonify({"ok": True})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
@login_required
def delete_chat(chat_id):
    chat = get_owned_chat(chat_id)
    if chat is None:
        return jsonify({"error": "not found"}), 404
    del chats[chat_id]
    save_chats(chats)
    return jsonify({"ok": True})


# ============================================
# File upload (attach to a chat)
# ============================================
@app.route("/api/upload/<chat_id>", methods=["POST"])
@login_required
def upload_file(chat_id):
    chat = get_owned_chat(chat_id)
    if chat is None:
        return jsonify({"error": "not found"}), 404
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400
    content = file.read().decode("utf-8", errors="ignore")
    chat["uploaded_context"] = content[:6000]
    chat["uploaded_filename"] = file.filename
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
@login_required
def chat():
    data = request.json or {}
    chat_id = data.get("chat_id")
    user_message = data.get("message", "")
    web_search_enabled = data.get("web_search", False)

    current_chat = get_owned_chat(chat_id)
    if current_chat is None:
        return jsonify({"error": "invalid chat_id"}), 400

    if client is None:
        return jsonify({"error": "GEMINI_API_KEY not set on server"}), 500

    current_chat["messages"].append({"role": "user", "content": user_message})

    if current_chat["title"] == "New chat" and user_message:
        current_chat["title"] = user_message[:30] + ("..." if len(user_message) > 30 else "")

    system_instruction = SYSTEM_PROMPT
    if current_chat.get("uploaded_context"):
        system_instruction += f"\n\nReference document ({current_chat['uploaded_filename']}):\n{current_chat['uploaded_context']}"
    if web_search_enabled:
        search_summary = do_web_search(user_message)
        if search_summary:
            system_instruction += f"\n\n{search_summary}"

    contents = []
    for m in current_chat["messages"]:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    def is_transient_error(e):
        msg = str(e).upper()
        return "503" in msg or "UNAVAILABLE" in msg or "OVERLOADED" in msg

    def generate():
        full_response = ""
        try:
            attempts = 0
            while True:
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
                    break
                except Exception as e:
                    attempts += 1
                    if full_response or attempts >= 3 or not is_transient_error(e):
                        raise
                    time.sleep(2 ** attempts)
        except Exception:
            app.logger.exception("Gemini chat request failed")
            friendly = "⚠️ Server busy, try again."
            full_response += "\n\n" + friendly
            yield "\n\n" + friendly
        finally:
            current_chat["messages"].append({"role": "assistant", "content": full_response})
            save_chats(chats)

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
