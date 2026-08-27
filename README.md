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
# RAG Evaluation Module (`deepeval`)

An automated evaluation pipeline for measuring Retrieval-Augmented Generation (RAG) performance using hybrid search, local LLM generation, and DeepEval metrics.

---

## Overview

This module runs end-to-end evaluation on a set of benchmark questions to evaluate both retrieval quality and LLM answer fidelity:

1. **Retrieval**: Uses `hybrid_search_with_rerank` to fetch relevant context for each query.
2. **Generation**: Queries a local LLM via Ollama (`qwen`) using a strict context-bound system prompt.
3. **Evaluation**: Evaluates responses using standard LLM-as-a-judge metrics powered by Anthropic's `claude-3-haiku`.

---

## Evaluated Metrics

| Metric | Threshold | Evaluator Model | Description |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | `0.7` | `claude-3-haiku-20240307` | Measures if the generated output stays strictly within the retrieved context (prevents hallucinations). |
| **Answer Relevancy** | `0.7` | `claude-3-haiku-20240307` | Measures how directly and accurately the response addresses the input question. |

---

## Prerequisites

* **Python 3.10+**
* **Ollama**: Running locally at `http://localhost:11434` with the specified `qwen` model pulled.
* **Anthropic API Key**: Required for running `deepeval` metrics with Claude.
* **`questions.json`**: Input file containing test benchmark cases.

---

## Expected Data Format (`questions.json`)

The evaluation script expects a local JSON file formatted as follows:

```json
[
  {
    "question": "What is the return policy?",
    "expected_answer": "Items can be returned within 30 days of purchase."
  }
]

```
# Hybrid Search vs. Vector Baseline Evaluation (`Recall@5`)

An evaluation benchmark module that measures retrieval accuracy by comparing a pure **Vector-Only Search** baseline against a **Hybrid Search + Reranking** pipeline (BM25 + Vector + Cohere Rerank) using `Recall@5`.

---

## Overview

This module benchmarks search effectiveness across different query patterns (keyword-heavy, semantic-heavy, distractor-heavy, and ambiguous queries):

1. **Vector Search Baseline**: Uses `NomicModel` embeddings via Ollama and SQLAlchemy (pgvector cosine distance) to retrieve `Top-5` matches.
2. **Hybrid Search Pipeline (`bm25_Rag`)**: Combines BM25 lexical search (`bm25s` with English stemming) and vector similarity search to collect candidate pools (`Top-10` each).
3. **Deduplication & Reranking**: Deduplicates candidate results and uses Cohere’s `rerank-v4.0-pro` model to output the final `Top-5` ranked IDs.
4. **Recall Evaluation**: Computes `Recall@5` for both strategies across defined test query sets to verify measurable retrieval improvements.

---

## Benchmark Query Test Cases

The evaluation script tests four specific query categories:

| Category | Example Query | Search Strengths |
| :--- | :--- | :--- |
| **Keyword-focused** | `"Robitussin"`, `"Dolce Gusto machine"` | Favors BM25 (rare exact match terms) |
| **Semantic-focused** | `"A candy that reminds someone of a fantasy novel..."` | Favors Vector (conceptual matches without exact keywords) |
| **Distractor-heavy** | `"Dog food that helped with itching..."` | Tests multi-relevant candidate retrieval precision |
| **Ambiguous** | `"candy that had no real flavor..."` | Evaluates real-user natural language queries |

---

## Prerequisites

* **Python 3.10+**
* **PostgreSQL + pgvector**: Database connection supplying `EmbeddedItems` records and embeddings.
* **Ollama**: Local instance running at `http://localhost:11434/v1/` for local embeddings (`NomicModel`).
* **Cohere API Key**: Configured in environment variables (`cohere`).
* **Pre-built BM25 Index**: Existing `bm25Retriever` index directory loadable by `bm25s`.

---

## Setup & Execution

1. **Install dependencies**:
   ```bash
   pip install bm25s PyStemmer cohere openai sqlalchemy numpy python-dotenv
   ```

   # Synthetic Evaluation Dataset Generator (`questions.json`)

An automated pipeline for generating synthetic Question & Answer benchmark pairs from database product reviews using a locally hosted LLM via Ollama.

---

## Overview

This module automates the creation of test benchmarks for search and RAG evaluation pipelines:

1. **Database Extraction**: Fetches product reviews (`EmbeddedItems`) from PostgreSQL using SQLAlchemy.
2. **Synthetic Pair Generation**: Prompts a local LLM (`qwen`) via Ollama to construct natural search queries and accurate expected answers derived strictly from each review's content.
3. **JSON Parsing & Formatting**: Strips Markdown code blocks, validates JSON output formatting, and maps questions to their corresponding review IDs.
4. **Dataset Export**: Outputs the formatted evaluation dataset directly to `questions.json` for downstream testing.

---

## Prerequisites

* **Python 3.10+**
* **PostgreSQL Database**: Configured connection serving the `EmbeddedItems` table model.
* **Ollama**: Local instance running at `http://localhost:11434/v1/` with the target `qwen` model pulled.

---

## Setup & Execution

1. **Install dependencies**:
   ```bash
   pip install openai sqlalchemy

``
## Data format
``` json
[
  {
    "question": "What candy tasted like plain red sugar with no flavor?",
    "expected_answer": "The Twizzlers order tasted completely bland with no actual flavor.",
    "id": 27
  }
]

```
