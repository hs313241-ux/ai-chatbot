"""
Local AI Chatbot (Terminal Version) - Upgraded
Features:
  - Chat history save hoti hai (JSON file mein)
  - Custom personality/system prompt
  - "exit"/"quit" se band karo
"""

import ollama
import json
import os
from datetime import datetime

MODEL_NAME = "llama3.2"
HISTORY_FILE = "chat_history.json"

# ============================================
# YAHAN APNE CHATBOT KI PERSONALITY SET KARO
# ============================================
SYSTEM_PROMPT = """You are a friendly and helpful AI assistant. 
You explain things clearly and simply. You can respond in both English 
and Roman Urdu/Hindi depending on how the user talks to you. 
Keep your answers concise unless the user asks for more detail."""


def load_history():
    """Pichli saved history load karo agar file exist karti hai."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    """Conversation ko JSON file mein save karo."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 50)
    print("  Mera Local AI Chatbot")
    print("  (Type 'exit' ya 'quit' band karne ke liye)")
    print("  (Type 'clear' history saaf karne ke liye)")
    print("=" * 50)
    print()

    # Purani history load karo, aur system prompt hamesha sabse upar rakho
    saved_history = load_history()
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation_history.extend(saved_history)

    if saved_history:
        print(f"[{len(saved_history)} purane messages load hue]\n")

    while True:
        user_input = input("Aap: ").strip()

        if user_input.lower() in ["exit", "quit", "/bye"]:
            print("\nChatbot: Allah Hafiz! 👋")
            break

        if user_input.lower() == "clear":
            conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            save_history([])
            print("[History clear ho gayi]\n")
            continue

        if not user_input:
            continue

        conversation_history.append({"role": "user", "content": user_input})

        print("Chatbot: ", end="", flush=True)

        full_response = ""
        try:
            stream = ollama.chat(
                model=MODEL_NAME,
                messages=conversation_history,
                stream=True,
            )

            for chunk in stream:
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
                full_response += content

        except Exception as e:
            print(f"\n[Error: {e}]")
            print("Check karo ke Ollama chal raha hai ya nahi.")
            continue

        print("\n")

        conversation_history.append({"role": "assistant", "content": full_response})

        # Har response ke baad history save karo (system prompt chhod kar)
        save_history(conversation_history[1:])


if __name__ == "__main__":
    main()
