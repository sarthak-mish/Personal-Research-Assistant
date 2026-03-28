import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OPENROUTER_API_KEY = os.getenv("openrouter_api_key")
VECTORSTORE_DIR = "vectorstore"

# Shared LLM Client
def get_llm():
    return ChatOpenAI(
        model = "google/gemma-3-27b-it:free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={
            "HTTP-Referrer": "http://localhost",
            "X-Title": "Personal Research Assistant",
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

# --- Tool#2: Web Search ---
@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for current or recent information.
    Use this tool when question is about:
    - Recent AI models or releases (2024 onwards)
    - News or events not coveered in the uploaded documents
    - Topics where up-to-date information is crucial and required
    - Anything that is likely not covered in the uploaded documents
    """
    search = TavilySearchResults(
        max_results=3,
        search_depth="basic"
    )
    results = search.invoke(query)

    if not results:
        return "No relevant information found on the web."
    
    # Format results
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append(
            f"[Web Result {i+1}]\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"Content: {result.get('content', 'N/A')[:400]}" 
        )
    output = "\n\n---\n\n".join(formatted_results)
    print(f"\nRetrieved {len(results)} web results for the query.'{query}'")
    return output

# --- Tool#3: Final Answer Tool ---
@tool
def final_answer_tool(question: str, context: str) -> str:
    """
    Generate a clean, structured final answer using the provided context.
    Always use this tool last after you have gather context
    from either the RAG tool or the Web Search tool. This tool is responsible for synthesizing the final answer.
    Never use this tool without context. If you don't have enough information to answer the question, say "I don't know" instead of making up an answer.
    """

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant.
         Answer the user's question using only the provided context.
         Be concise and clear.
         Always mention which sources you are using to answer the question.
         If the context does not contain enough information to answer the question, say "I don't know"""),
         ("human", "context:\n{context}\n\nQuestion: {question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"\nGenerated final answer for the question: '{question}'")
    return response.content

if __name__ == "__main__":
    print("=" * 50)
    print("TEST 1: RAG Tool")
    print("=" * 50)
    rag_result = rag_tool.invoke("What is LLM?")
    print(rag_result[:500])

    print("\n" + "=" * 50)
    print("TEST 2: Web Search Tool")
    print("=" * 50)
    web_result = web_search_tool.invoke("Latest LLM models released in 2025")
    print(web_result[:500])  # preview


    print("\n" + "=" * 50)
    print("TEST 3: Final Answer Tool")
    print("=" * 50)
    answer = final_answer_tool.invoke({
        "question": "What is multi-head attention?",
        "context": rag_result
    })
    print(answer)