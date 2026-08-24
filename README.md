# LLM_Api-

this is a repository showing all things i learned about LLMS 
## /src/logic/RagChatBot.py
# Local RAG Chatbot (`RagChatBot.py`)

A lightweight Retrieval-Augmented Generation (RAG) script that performs semantic product searches using **PostgreSQL (`pgvector`)** and queries a local **Ollama** LLM with strict closed-domain guardrails.

---

## What It Does
1. **Semantic Search:** Converts user queries into vector embeddings via Ollama and fetches the top 5 closest product reviews using cosine distance (`pgvector`).
2. **Context Formatting:** Structures database rows into a readable context block for the LLM.
3. **Guardrailed Inference:** Sends the context to a local LLM (`qwen`) with strict instructions to answer *only* using the provided data (otherwise outputting `"I don't know"`).
4. **Streaming CLI:** Runs an interactive terminal loop with real-time token streaming.

---

## Dependencies & Setup
Requires a running PostgreSQL instance with `pgvector` and Ollama running locally (`http://localhost:11434/v1/`).

```bash
pip install fastapi openai pandas numpy sqlalchemy pgvector
```
## Starting the Chatbot 
```bash
uv run python -m src.llm_api.logic.start
```
