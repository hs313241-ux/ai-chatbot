"""
AI Chatbot (Web Version) - Full featured, ChatGPT-style
Features: multiple chats (rename/delete), inline file attach, web search, stop generation

Run karne ke liye: streamlit run app.py
Browser mein khulega: http://localhost:8501

Extra installs needed:
    pip install ddgs
"""

import streamlit as st
import ollama
import uuid

MODEL_NAME = "llama3.2"

SYSTEM_PROMPT = """You are a friendly and helpful AI assistant. 
Always respond in English by default, unless the user writes to you in a 
different language (like Roman Urdu/Hindi) - in that case, match their language.
Explain things clearly and simply. Keep answers concise unless the user asks 
for more detail."""

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    .stApp { background-color: #1a1a1a; color: #e5e5e5; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        max-width: 46rem;
        padding-top: 3rem;
        padding-bottom: 9rem;
        margin: 0 auto;
    }

    section[data-testid="stSidebar"] {
        background-color: #141414;
        border-right: 1px solid #2a2a2a;
    }
    section[data-testid="stSidebar"] * { color: #d4d4d4; }
    section[data-testid="stSidebar"] h3 { color: #f5f5f5; font-weight: 600; }

    .stButton button {
        background-color: transparent;
        color: #d4d4d4;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.87rem;
        padding: 0.45rem 0.8rem;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        background-color: #262626;
        border-color: #4a4a4a;
        color: #ffffff;
    }

    /* Chat list row buttons - make them look like list items, left aligned */
    .chat-row .stButton button {
        text-align: left;
        justify-content: flex-start;
        border: none;
        background-color: transparent;
        font-weight: 400;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .chat-row .stButton button:hover { background-color: #232323; }
    .chat-row-active .stButton button {
        background-color: #262626;
        color: #ffffff !important;
        font-weight: 500;
    }

    .empty-state { text-align: center; margin-top: 18vh; }
    .empty-state h1 {
        font-size: 1.9rem; font-weight: 600; color: #f5f5f5;
        margin-bottom: 0.4rem; letter-spacing: -0.02em;
    }
    .empty-state p { color: #8a8a8a; font-size: 0.95rem; font-weight: 400; }

    [data-testid="stChatMessage"] {
        background-color: transparent;
        padding: 1.1rem 0;
        border-bottom: 1px solid #222222;
    }
    [data-testid="stChatMessageAvatarUser"] { background-color: #4f46e5 !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #059669 !important; }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] ol,
    [data-testid="stChatMessageContent"] ul,
    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3,
    [data-testid="stChatMessageContent"] h4,
    [data-testid="stChatMessageContent"] strong,
    [data-testid="stChatMessageContent"] em,
    [data-testid="stChatMessageContent"] span {
        color: #e5e5e5 !important;
        font-size: 0.96rem;
        line-height: 1.7;
    }

    div[data-testid="stBottomBlockContainer"] { background-color: #1a1a1a !important; }
    div[data-testid="stBottom"] { background-color: #1a1a1a !important; }
    .stChatFloatingInputContainer { background-color: #1a1a1a !important; }

    [data-testid="stChatInput"] {
        background-color: #262626 !important;
        border-radius: 24px;
        border: 1px solid #3a3a3a;
        max-width: 46rem;
        margin: 0 auto;
    }
    [data-testid="stChatInput"] > div { background-color: #262626 !important; }
    [data-testid="stChatInput"] textarea {
        background-color: #262626 !important;
        color: #e5e5e5 !important;
        font-size: 0.95rem !important;
        -webkit-text-fill-color: #e5e5e5 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8a8a8a !important;
        -webkit-text-fill-color: #8a8a8a !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInputSubmitButton"] { background-color: #e5e5e5 !important; }
    [data-testid="stChatInputSubmitButton"] svg { fill: #1a1a1a !important; }

    code { background-color: #262626 !important; border-radius: 5px; padding: 2px 6px; font-size: 0.88rem; }
    pre { background-color: #141414 !important; border-radius: 10px; border: 1px solid #2a2a2a; }
    hr { border-color: #2a2a2a; }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; }
    ::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ============================================
# Session state init - multi-chat structure
# ============================================
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "active_stream" not in st.session_state:
    st.session_state.active_stream = None
if "active_response" not in st.session_state:
    st.session_state.active_response = ""
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "renaming_chat_id" not in st.session_state:
    st.session_state.renaming_chat_id = None


def create_new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "uploaded_context": None,
        "uploaded_filename": None,
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.active_stream = None
    st.session_state.active_response = ""


def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    if st.session_state.current_chat_id == chat_id:
        remaining = list(st.session_state.chats.keys())
        if remaining:
            st.session_state.current_chat_id = remaining[-1]
        else:
            create_new_chat()


# Make sure at least one chat exists
if not st.session_state.chats:
    create_new_chat()
if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]

current_chat = st.session_state.chats[st.session_state.current_chat_id]


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
# Sidebar - ChatGPT-style chat list
# ============================================
with st.sidebar:
    st.markdown("### AI Chatbot")
    st.caption(f"{MODEL_NAME} · Running offline")
    st.write("")

    if st.button("＋  New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.write("")
    web_search_enabled = st.checkbox("🌐 Web search (needs internet)")

    st.write("")
    st.divider()
    st.caption("CHATS")

    # List all chats, newest first
    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[chat_id]
        is_active = chat_id == st.session_state.current_chat_id
        row_class = "chat-row-active" if is_active else "chat-row"

        if st.session_state.renaming_chat_id == chat_id:
            new_title = st.text_input(
                "Rename", value=chat["title"], key=f"rename_input_{chat_id}",
                label_visibility="collapsed"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Save", key=f"save_{chat_id}", use_container_width=True):
                    chat["title"] = new_title.strip() or "New chat"
                    st.session_state.renaming_chat_id = None
                    st.rerun()
            with col_b:
                if st.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):
                    st.session_state.renaming_chat_id = None
                    st.rerun()
        else:
            st.markdown(f'<div class="{row_class}">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
            with c1:
                if st.button(chat["title"], key=f"select_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with c2:
                if st.button("✏️", key=f"editbtn_{chat_id}"):
                    st.session_state.renaming_chat_id = chat_id
                    st.rerun()
            with c3:
                if st.button("🗑️", key=f"delbtn_{chat_id}"):
                    delete_chat(chat_id)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.divider()
    if current_chat["uploaded_filename"]:
        st.caption(f"📄 Attached: {current_chat['uploaded_filename']}")
    st.caption("Fully local model — koi data internet pe nahi jaata (web search on hone par sirf search query jaati hai).")

# ============================================
# Empty state
# ============================================
if len(current_chat["messages"]) == 0 and st.session_state.active_stream is None:
    st.markdown("""
    <div class="empty-state">
        <h1>AI Chatbot</h1>
        <p>Ask me anything — I'm here to help.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Show chat history
# ============================================
for message in current_chat["messages"]:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ============================================
# Active generation - pull one chunk per rerun (enables working Stop button)
# ============================================
if st.session_state.active_stream is not None:
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown(st.session_state.active_response + "▌")
        st.button(
            "⏹  Stop generating", key="stop_btn",
            on_click=lambda: st.session_state.update(stop_requested=True)
        )

    if st.session_state.stop_requested:
        current_chat["messages"].append(
            {"role": "assistant", "content": st.session_state.active_response}
        )
        st.session_state.active_stream = None
        st.session_state.active_response = ""
        st.session_state.stop_requested = False
        st.rerun()
    else:
        try:
            chunk = next(st.session_state.active_stream)
            st.session_state.active_response += chunk["message"]["content"]
            st.rerun()
        except StopIteration:
            current_chat["messages"].append(
                {"role": "assistant", "content": st.session_state.active_response}
            )
            st.session_state.active_stream = None
            st.session_state.active_response = ""
            st.rerun()
        except Exception as e:
            current_chat["messages"].append({"role": "assistant", "content": f"⚠️ Error: {e}"})
            st.session_state.active_stream = None
            st.session_state.active_response = ""
            st.rerun()

# ============================================
# Chat input - with inline file attach (paperclip icon, built into Streamlit)
# ============================================
else:
    prompt = st.chat_input(
        "Message AI Chatbot...",
        accept_file=True,
        file_type=["txt", "md"],
    )

    if prompt and (prompt.text or prompt.files):
        user_text = prompt.text or "(file attached)"

        # Handle attached file
        if prompt.files:
            uploaded = prompt.files[0]
            content = uploaded.read().decode("utf-8", errors="ignore")
            current_chat["uploaded_context"] = content[:6000]
            current_chat["uploaded_filename"] = uploaded.name

        current_chat["messages"].append({"role": "user", "content": user_text})

        # Auto-title the chat from the first message
        if current_chat["title"] == "New chat":
            current_chat["title"] = user_text[:30] + ("..." if len(user_text) > 30 else "")

        full_conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        if current_chat["uploaded_context"]:
            full_conversation.append({
                "role": "system",
                "content": f"Reference document ({current_chat['uploaded_filename']}):\n{current_chat['uploaded_context']}"
            })

        if web_search_enabled:
            search_summary = do_web_search(user_text)
            if search_summary:
                full_conversation.append({"role": "system", "content": search_summary})

        full_conversation.extend(current_chat["messages"])

        st.session_state.active_stream = ollama.chat(
            model=MODEL_NAME, messages=full_conversation, stream=True,
        )
        st.session_state.active_response = ""
        st.rerun()
