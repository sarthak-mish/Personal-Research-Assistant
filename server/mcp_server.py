import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agent'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from datetime import datetime

from graph import build_graph

# App Setup
app = FastAPI(
    title="Reseach Agent MCP Server",
    description="LangGraph RAG Agent exposed as an MCP-compatible tool server",
    version="1.0.0"
)

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Build agent when server starts
print("Initializing agent graph...")
app_agent = build_graph()
print("Agent graph initialized successfully.")

# REQUEST / RESPONSE MODEL
class InvokeRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default-session"

class InvokeResponse(BaseModel):
    answer: str
    tool_used: str
    session_id: str
    timestamp: str

class ToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict

# API ENDPOINTS ----- #1 Health Check
@app.get('/health')
def heath_check():
    """Check if the server is running"""
    return {
        "status": "ok", 
        "message": "Research Agent MCP Server is running and ready to accept requests.",
        "timestamp": datetime.now().isoformat()
        }

# API ENDPOINTS ----- #2 Get Tool List (MCP Standard)
@app.get('/tools', response_model=list[ToolInfo])
def list_tools():
    """
    MCP standard endpoint to list all the available tools.
    Any MCP client calls this first to discover what tools are available and how to use them.
    """

    return [
        {
            "name": "research_agent",
            "description": (
                "Answers questions by searching research PDFs (RAG)"
                "or on web. Supports multiturn conversations with memory via session_id."
            ),
            "input_schema": {
                "type": "object",
                "properties" : {
                    "query" : {
                        "type": "string",
                        "description": "The user's question or query to the research agent."
                    },
                    "session_id": {
                        "type": "string",
                        "description": (
                            "Session ID for conversation memory"
                        )
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "rag_search",
            "description": "Tool to search through uploaded research PDFs.",
            "input_schema": {
                "type": "object",
                "properties" : {
                    "query" : {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "web_search",
            "description": "Tool to search the web for up-to-date information.",
            "input_schema": {
                "type": "object",
                "properties" : {
                    "query" : {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    ]

# API ENDPOINTS ----- #3 Invoke Agent
@app.post('/invoke', response_model=InvokeResponse)
def invoke_agent(request: InvokeRequest):
    """
    Main MCP endpoint - sends a question to the agent and gets an answer.
    The agent auto decides whether to use RAG or web search.
    """

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        print(f"\nReceived query: '{request.query}' with session_id: '{request.session_id}'")

        config = { "configurable" : { "thread_id": request.session_id }}
    
        initial_state = {
            "question": request.query,
            "context": "",
            "answer": "",
            "next_step": "",
            "chat_history": []
        }

        result = app_agent.invoke(initial_state, config=config)

        return InvokeResponse(
            answer=result['answer'],
            tool_used=result['next_step'],
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")
    
# API ENDPOINTS ----- #4 Direct RAG Search
@app.post('/rag_search', response_model=InvokeResponse)
def rag_search(request: InvokeRequest):
    """
    Bypass the router and directly invoke the RAG tool.
    """

    from tools import rag_tool
    try:
        context = rag_tool.invoke(request.query)
        return { "query": request.query, "context": context }
    except Exception as e:
        print(f"Error during RAG search: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while performing RAG search.")
    
# API ENDPOINTS ----- #5 Direct Web Search
@app.post('/web_search', response_model=InvokeResponse)
def web_search(request: InvokeRequest):
    """
    Bypass the router and directly invoke the Web Search tool.
    """

    from tools import web_search_tool
    try:
        context = web_search_tool.invoke(request.query)
        return { "query": request.query, "context": context }
    except Exception as e:
        print(f"Error during Web search: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while performing Web search.")



# ── RUN SERVER ─────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "mcp_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True        # auto-restart on code changes
    )