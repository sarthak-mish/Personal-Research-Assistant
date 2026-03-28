import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VECTORSTORE_DIR = "vectorstore"

# Shared LLM Client
def get_llm():
    return ChatOpenAI(
        model= "google/gemma-3-27b-it:free",
        open_api_key=OPENROUTER_API_KEY,
        open_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referrer": "http://localhost",
            "X-Title": "RAG POC"
        }
    )

# Shared: Retriever
def get_retriver():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# --- Tool#1: RAG TOOL ---
@tool
def rag_tool(query: str) -> str:
    """
    Search though the uploaded research PDFs to answer questions.
    Use this tool when the question is about:
    - Transformer architecture or attention mechanism
    - RAG (Retrieval-Augmented Generation)
    - ReAct framework or agent reasoning
    - Any topic likely covered in the uploaded document
    """
    retreiver = get_retriver()
    results = retreiver.invoke(query)

    if not results:
        return "No relevant information found in the documents."
    
    # Format the retrieved results for better readability
    context_parts = []
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", " ? ")
        filename = os.path.basename(source)
        context_parts.append(
            f"Source: {filename}, Page: {page}\nContent: {doc.page_content}\n"
        )

    context = "\n\n---\n\n".join(context_parts)
    print(f"\nRetrieved {len(results)} relevant chunks for the query.'{query}'")
    return context



if __name__ == "__main__":
    print("=" * 50)
    print("TEST 1: RAG Tool")
    print("=" * 50)
    rag_result = rag_tool.invoke("What is LLM?")
    print(rag_result[:500])