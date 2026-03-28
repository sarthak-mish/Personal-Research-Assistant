# Personal Research Assistant

An AI-powered research assistant that answers questions using your uploaded PDF documents and web search. Built with LangChain, LangGraph, and ChromaDB.

## How It Works

1. **PDF Ingestion** -- Load and chunk PDF documents into a ChromaDB vector store using HuggingFace embeddings.
2. **RAG Tool** -- Retrieves relevant chunks from your documents to answer queries.
3. **Web Search Tool** -- Falls back to Tavily web search for topics not covered in your documents.
4. **Final Answer Tool** -- Synthesizes retrieved context into a clear, sourced answer using an LLM via OpenRouter.

## Project Structure

```
agent/
  rag_pipeline.py   # PDF loading, chunking, and vector store creation
  tools.py          # LangChain tools (RAG, web search, final answer)
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

### 2. Use the Tools

```python
from agent.tools import rag_tool, web_search_tool, final_answer_tool

# Search your documents
context = rag_tool.invoke("What is multi-head attention?")

# Search the web
results = web_search_tool.invoke("Latest LLM models 2025")

# Generate a final answer
answer = final_answer_tool.invoke({
    "question": "What is multi-head attention?",
    "context": context
})
```

## CI

Linting (ruff) and tests (pytest) run automatically on push/PR via GitHub Actions.
