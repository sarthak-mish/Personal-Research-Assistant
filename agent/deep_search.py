import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder

load_dotenv()

OPENROUTER_API_KEY = os.getenv("openrouter_api_key")
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
        ("system", f"""Generate {n} different version of user's question
         to retrieve more relevant information from a vector database.
         Each version should approcah the question from a different angle.
         Return only the questions, one per line, No numbering, no explanation."""),
         ("human", "Original Question: {question}")
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
    vectorstore = get_vectostore()
    vectorstore_retriver = vectorstore.as_retriever(search_kwargs={"k": k})

    # Ensemble retriever that combines both
    hybrid_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, vectorstore_retriver],
        weights = [0.5, 0.5]
    )

    print("Hybrid retriever built with BM25 and Vector search.")
    return hybrid_retriever

# --- LAYER 3: Reranking with Cross-Encoder ---
def rerank_documents(
        query: str,
        documents: List[Document],
        top_n: int = 4
) -> List[Document]:
    """
    Use a cross-encoder to rerank retrieved documents.
    Much more accurate than cosine similarity alone.
    """

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Score each document against the query
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)

    # Sort by score decending
    scored_docs = sorted (
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Keep only Top N
    reranked = [doc for doc, _ in scored_docs[:top_n]]

    print(f"Reranked {len(documents)} chunks → kept top {top_n}")

    for i, (doc, score) in enumerate(scored_docs[:top_n]):
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  #{i+1} score={score:.3f} | {src} p{doc.metadata.get('page','?')}")

    return reranked

# ── LAYER 4: CONTEXTUAL COMPRESSION ──────────────────────────────
def compress_documents(
    query: str,
    documents: List[Document]
) -> List[Document]:
    """
    Extract only the relevant parts of each chunk
    relative to the query — strips noise.
    """
    llm = get_llm()
    compressor = LLMChainExtractor.from_llm(llm)

    compressed = []
    for doc in documents:
        try:
            result = compressor.compress_documents([doc], query)
            if result:
                compressed.extend(result)
            else:
                compressed.append(doc)   # fallback: keep original
        except Exception:
            compressed.append(doc)       # fallback: keep original

    print(f"Compressed {len(documents)} chunks → {len(compressed)} focused chunks")
    return compressed

# ── MAIN: DEEP SEARCH PIPELINE ────────────────────────────────────
def deep_search(question: str, chunks: List[Document]) -> str:
    """
    Full deep search pipeline:
    Multi-Query → Hybrid Search → Rerank → Compress → Return context
    """
    print(f"\n{'='*60}")
    print(f"Deep Search: '{question}'")
    print(f"{'='*60}")

    # ── Layer 1: Generate multiple query variants
    queries = generate_multi_queries(question, n=3)

    # ── Layer 2: Hybrid search for each query variant
    hybrid_retriever = build_hybrid_retriever(chunks, k=6)

    all_docs = []
    seen_contents = set()

    for query in queries:
        results = hybrid_retriever.invoke(query)
        for doc in results:
            # Deduplicate by content
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)

    print(f"\n Total unique chunks after hybrid search: {len(all_docs)}")

    # ── Layer 3: Rerank all collected chunks
    reranked_docs = rerank_documents(question, all_docs, top_n=5)

    # ── Layer 4: Contextual compression
    compressed_docs = compress_documents(question, reranked_docs)

    # ── Format final context
    context_parts = []
    for i, doc in enumerate(compressed_docs):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source: {source} | Page: {page}]\n{doc.page_content}"
        )

    final_context = "\n\n---\n\n".join(context_parts)
    print(f"\nDeep search complete — {len(compressed_docs)} high-quality chunks ready")
    return final_context


if __name__ == "__main__":
    # Need to load chunks first for BM25
    from rag_pipeline import load_documents, chunk_documents

    docs = load_documents("docs")
    chunks = chunk_documents(docs)

    # Run deep search
    context = deep_search("What is multi-head attention?", chunks)
    print("\nFinal Context Preview:")
    print(context[:1000])


