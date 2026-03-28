import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

OPENROUTER_API_KEY = os.getenv("openrouter_api_key")

# Open Router LLM Client
def get_llm():
    return ChatOpenAI(
        model = "deepseek/deepseek-r1-0528:free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={
            "HTTP-Referrer": "http://localhost",
            "X-Title": "Personal Research Assistant",
        }
    )

# Load Documents
def load_documents(docs_folder: str):
    all_docs = []
    for filename in os.listdir(docs_folder):
        if filename.endswith('.pdf'):
            path = os.path.join(docs_folder, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {filename}")
    print(f"Total documents loaded: {len(all_docs)}")
    return all_docs

# Doc Splitting
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators =["\n\n", "\n", ".", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# Create Vector Store
def create_vector_store(chunks, persist_dir: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Vector store created with {len(chunks)} chunks.")
    return vectorstore

#Retrieve relevant chunks
def get_retriever(persist_dir: str, k: int = 4):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    print(f"Retriever created with top {k} results.")
    return retriever

# Test Similarity Search
def test_retriever(retiever, query: str):
    print(f"Query: {query}")
    results = retiever.invoke(query)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:200]}...")
        print(doc.page_content[:300])

if __name__ == "__main__":
    DOCS_DIR = "docs"
    VECTORSTORE_DIR = "vectorstore"

    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs)
    create_vector_store(chunks, VECTORSTORE_DIR)

    retriever = get_retriever(VECTORSTORE_DIR, k=4)
    test_retriever(retriever, "What is multi-head attention?")


