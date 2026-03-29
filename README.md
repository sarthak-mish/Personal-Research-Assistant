# Personal Research Assistant

An AI-powered research assistant that answers questions using your uploaded PDF documents and web search. Built with LangChain, LangGraph, and ChromaDB.

## How It Works

1. **PDF Ingestion** -- Load and chunk PDF documents into a ChromaDB vector store using HuggingFace embeddings.
2. **Router Agent** -- An LLM-based router determines whether to use RAG, deep search, or web search based on the question.
3. **RAG Tool** -- Retrieves relevant chunks from your documents to answer queries.
4. **Deep Search Tool** -- A multi-layered retrieval pipeline for complex questions: multi-query generation, hybrid search (BM25 + vector), cross-encoder reranking, and contextual compression.
5. **Web Search Tool** -- Uses Tavily web search for topics needing current information.
6. **Final Answer Tool** -- Synthesizes retrieved context into a clear, sourced answer using an LLM via OpenRouter.
7. **Conversational Memory** -- Maintains chat history across questions using LangGraph's checkpoint system.

## Project Structure

```
agent/
  graph.py          # LangGraph agent with router, RAG, deep search, web search, and answer nodes
  rag_pipeline.py   # PDF loading, chunking, and vector store creation
  tools.py          # LangChain tools (RAG, deep search, web search, final answer)
  deep_search.py    # Multi-layered deep search pipeline (multi-query, hybrid retrieval, reranking, compression)
server/
  mcp_server.py     # FastAPI MCP-compatible server exposing the agent as API endpoints
docs/               # Place your research PDFs here
vectorstore/        # Auto-generated ChromaDB storage
```

## Setup

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key
- A [Tavily](https://tavily.com/) API key (for web search)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
openrouter_api_key=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### 1. Ingest PDFs

Place your PDF files in the `docs/` directory, then run:

```bash
python agent/rag_pipeline.py
```

This loads, chunks, and indexes the documents into the vector store.

### 2. Run the Agent

```bash
python agent/graph.py
```

The agent automatically routes questions to the appropriate tool (RAG, deep search, or web search), generates answers, and maintains conversational memory across queries.

### 3. Use Individual Tools

```python
from agent.tools import rag_tool, deep_search_tool, web_search_tool, final_answer_tool

# Search your documents
context = rag_tool.invoke("What is multi-head attention?")

# Deep search (multi-query + hybrid retrieval + reranking + compression)
context = deep_search_tool.invoke("Explain the transformer architecture in detail")

# Search the web
results = web_search_tool.invoke("Latest LLM models 2025")

# Generate a final answer
answer = final_answer_tool.invoke({
    "question": "What is multi-head attention?",
    "context": context
})
```

### 4. Run the MCP Server

```bash
python server/mcp_server.py
```

This starts a FastAPI server on `http://localhost:8000` that exposes the research agent as MCP-compatible API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tools` | GET | List available tools (MCP standard) |
| `/invoke` | POST | Send a query to the agent (auto-routes to RAG or web search) |
| `/rag_search` | POST | Direct RAG search over your PDFs |
| `/web_search` | POST | Direct web search |

**Example request:**

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "What is multi-head attention?", "session_id": "my-session"}'
```

## CI

Linting (ruff) and tests (pytest) run automatically on push/PR via GitHub Actions.
