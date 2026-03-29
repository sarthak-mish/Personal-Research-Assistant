import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VECTORSTORE_DIR = "vectorstore"

# ── SHARED UTILITIES ──────────────────────────────────────────────
def get_llm():
    return ChatOpenAI(
        model="mistralai/mistral-small-2603",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG POC"
        }
    )

def get_vectostore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )
    return vectorstore

# --- LAYER 1: Muti Query Generation ---
def generate_multi_queries(question: str, n: int = 3) -> List[str]:
    """
    Generate N alternative version of the question
    to improve retrieval coverage
    """

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate {n} different version of user's question
         to retrieve more relevant information from a vector database.
         Each version should approcah the question from a different angle.
         Return only the questions, one per line, No numbering, no explanation."""),
         ("humam", "Original Question: {question}")
    ])

    chain = prompt | llm
    response = chain.invoke({"question": question})

    #Parse response in to list of questions
    variants = [q.strip() for q in response.content.split("\n") if q.strip()]
    all_queries = [question] + variants[:n]

    print(f"🔄 Multi-Query variants generated:")
    for q in all_queries:
        print(f"   → {q}")

    return all_queries

# --- LAYER 2: Hybrid Search (Vector + BM25) ---
def build_hybrid_retriever(chunks: List[Document], k: int = 6):
    """
    Combines BM25 keyword search with vector semantic search.
    Equal weigh give to both.
    """

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Vector - Semantic retriever
    