# Simple AI Memory Chatbot

A Python terminal chatbot prototype that can:

1. extract useful facts from conversations,
2. store them persistently,
3. retrieve relevant memories in future sessions,
4. use those memories to generate more personalized answers.

## Features

- Terminal chat interface.
- Simple dual-mode web UI:
  - `Terminal View` (terminal-style chat look)
  - `Simple UI` (chat bubble look)
- Automatic memory extraction after each user message.
- Persistent memory storage in `memories.json`.
- Memory retrieval with cosine similarity over token-frequency vectors.
- Personalized response generation with retrieved memories.
- Works in two modes:
  - `GROK_API_KEY` set: uses Grok/xAI chat completions API for extraction and reply generation.
  - no API key: falls back to lightweight heuristic extraction + simple fallback reply.

## Architecture

- `app.py`
  - Main chat loop.
  - Handles commands (`/new`, `/mem`, `/quit`).
  - Orchestrates retrieve -> generate -> extract -> store.
- `web_ui.py`
  - Local HTTP server at `http://127.0.0.1:8000`.
  - Two frontend modes with a mode switch.
  - Uses the same memory and LLM pipeline as the terminal app.
- `llm_client.py`
  - `extract_facts(user_message)`: asks the LLM to output durable facts in JSON.
  - `generate_reply(user_message, memory_facts, conversation_history)`: generates answer using relevant memories.
  - Includes fallback logic when API key/network is unavailable.
- `memory_store.py`
  - Loads/saves `memories.json`.
  - Deduplicates facts.
  - Implements simple vectorization and cosine similarity search.

## Memory Lifecycle

1. User sends a message.
2. System retrieves relevant stored facts with similarity search.
3. System generates response with those facts in prompt context.
4. System extracts durable facts from the user message.
5. New facts are deduplicated and saved to `memories.json`.

## Setup

```bash
cd /Users/jagruthipulumati/Desktop/task1
python3 -m venv .venv
source .venv/bin/activate
```

Create a `.env` file (recommended) or edit `.env.example` (also supported by this app):
```bash
cp .env.example .env
```

Then set:
```env
GROK_API_KEY=your_key_here
GROK_MODEL=grok-2-latest
LLM_BASE_URL=https://api.x.ai/v1
```

## Run

Terminal mode:

```bash
python3 app.py
```

Web UI mode:

```bash
python3 web_ui.py
```

Then open: `http://127.0.0.1:8000`

In-chat commands:
- `/new` start a new session (keeps saved memories)
- `/mem` list saved memories
- `/clear` wipe all saved memories
- `/quit` exit

## Example

Session 1:

- User: `Hey, I'm Priya. I'm allergic to peanuts and I love Italian food.`
- AI: responds and saves facts.

Session 2 (fresh process or `/new`):

- User: `Can you suggest something for dinner tonight?`
- AI: uses retrieved memories (name/allergy/food preference) to personalize suggestion.

## Storage Format

`memories.json` stores a list of records like:

```json
[
  {
    "fact": "User is allergic to peanuts",
    "created_at": "2026-02-19T...Z",
    "source_user_message": "Hey, I'm Priya..."
  }
]
```

## Tradeoffs

- Pros:
  - Very simple and understandable memory pipeline.
  - No external vector DB required.
  - Persistent cross-session memory in a human-readable file.
- Cons:
  - Similarity search is lexical, not true semantic retrieval.
  - Extraction quality depends on LLM output quality.
  - No privacy filtering/redaction layer.

## What I'd Improve With More Time

- Use embedding-based retrieval (e.g., OpenAI embeddings + FAISS/Chroma) for better semantic matching.
- Add confidence scoring + memory aging/pruning.
- Separate memory categories (profile, preferences, constraints, goals).
- Add tests for extraction, deduplication, and retrieval ranking.
- Add optional lightweight web UI.
