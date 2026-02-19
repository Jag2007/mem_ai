
import os
from pathlib import Path

from llm_client import LLMClient
from memory_store import MemoryStore


HELP_TEXT = (
    "Commands:\n"
    "  /new    Start a new conversation session (memory remains on disk)\n"
    "  /mem    Show all stored memories\n"
    "  /clear  Delete all stored memories\n"
    "  /quit   Exit\n"
)


def load_env_files() -> None:
    """Load KEY=VALUE pairs from .env, then .env.example as fallback."""
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
            if not key:
                continue

            # Respect variables already exported in the shell.
            os.environ.setdefault(key, value)


def main() -> None:
    load_env_files()
    memory_store = MemoryStore(path="memories.json")
    llm = LLMClient()
    conversation_history = []

    print("AI Memory Chatbot")
    print(HELP_TEXT)
    if llm.enabled:
        print(f"LLM mode: enabled ({llm.model} via {llm.base_url})")
    else:
        print("LLM mode: no API key found (set GROK_API_KEY); using fallback behavior")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_message:
            print("AI: Please type a message, or use /quit to exit.")
            continue

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

        if user_message == "/quit":
            print("Goodbye.")
            return

        if user_message == "/new":
            conversation_history = []
            print("AI: Started a new session. I still remember past saved facts.")
            continue

        if user_message == "/mem":
            if not memory_store.memories:
                print("AI: No memories saved yet.")
            else:
                print("AI: Stored memories:")
                for idx, memory in enumerate(memory_store.memories, start=1):
                    print(f"  {idx}. {memory.fact}")
            continue

        if user_message == "/clear":
            memory_store.clear()
            conversation_history = []
            print("AI: Cleared all saved memories.")
            continue

        relevant = memory_store.search(user_message, top_k=5, min_score=0.0)
        relevant_facts = [m.fact for m in relevant]
        all_facts = [m.fact for m in memory_store.memories]
        context_facts = []
        for fact in relevant_facts + all_facts:
            if fact not in context_facts:
                context_facts.append(fact)

        conversation_history.append({"role": "user", "content": user_message})
        reply = llm.generate_reply(user_message, context_facts, conversation_history)
        conversation_history.append({"role": "assistant", "content": reply})

        print(f"AI: {reply}")

        extracted = llm.extract_facts(user_message)
        added = memory_store.add_facts(extracted, source_user_message=user_message)
        if added:
            print(f"[memory] saved {added} new fact(s)")


if __name__ == "__main__":
    main()
