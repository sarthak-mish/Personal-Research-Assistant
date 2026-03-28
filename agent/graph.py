import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tools import rag_tool, web_search_tool, final_answer_tool

load_dotenv()

OPENROUTER_API_KEY = os.getenv("openrouter_api_key")

# Define State
class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    next_step: str
    chat_history: str

# LLM Client
def get_llm():
    return ChatOpenAI(
        model="google/gemma-3-27b-it:free",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG POC"
        }
    )

# Define Router Node - Node#1
def router_node(state: AgentState) -> AgentState:
    print(f"Router Node: Received question: {state['question']}")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant.
        Decide the best tool to answer the user's question based on the following criteria:
        Reply with ONLY one word:
        - "rag"        → if the answer is likely in uploaded research papers 
                         (transformers, attention, RAG, ReAct, LLMs concepts)
        - "web_search" → if the answer needs current/recent information
                         (latest models, news, events after 2023)
        
        Reply with ONLY the word. No explanation.
        """),
        ("human", "Question: {question}")
    ])

    chain = prompt | llm
    decision = chain.invoke({"question": state["question"]})
    next_step = decision.strip().lower()

    if next_step not in ["rag", "web_search"]:
        print(f"Router Node: Invalid decision '{decision}', defaulting to 'web_search'")
        next_step = "rag"
    
    print(f"Router Node: LLM decision: {decision}")
    return {**state, "next_step": next_step}

# Define Node#2: RAG Node
def rag_node(state: AgentState) -> AgentState:
    print("RAG Node: Invoking RAG tool...")
    context = rag_tool.invoke(state["question"])
    return {**state, "context": context}

# Define Node#3: Web Search Node
def web_search_node(state: AgentState) -> AgentState:
    print("Web Search Node: Invoking Web Search tool...")
    context = web_search_tool.invoke(state["question"])
    return {**state, "context": context}

# Define Node#4: Final Answer Node
def final_answer_node(state: AgentState) -> AgentState:
    print("Final Answer Node: Generating final answer...")
    answer = final_answer_tool.invoke({
        "question": state["question"],
        "context": state["context"]
    })

    updated_history = state.get("chat_history, []")
    updated_history.append({
        "question": state["question"],
        "answer": answer,
        "source": state["context"]
    })

    return {
        **state,
        "answer": answer,
        "chat_history": updated_history
    }

# Define Conditional Edge
def route_decision(state: AgentState) -> Literal["rag_node", "web_search_node"]:
    if state["next_step"] == "rag":
        return "rag_node"
    else:
        return "web_search_node"
    
# Build Graph
def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes

    graph.add_node("router_node", router_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("final_answer_node", final_answer_node)

    #Define edges
    graph.set_entry_point("router_node")

    # Conditional edges from router_node
    graph.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "rag_node": "rag_node",
            "web_search_node": "web_search_node"
        }
    ) 

    # Both RAG and Web Search nodes lead to Final Answer node
    graph.add_edge("rag_node", "final_answer_node")
    graph.add_edge("web_search_node", "final_answer_node")

    # Final Answer node leads to END
    graph.add_edge("final_answer_node", END)

    # Add memory saver checkpoint after final answer node
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Define Chat Loop
def chat(app, question: str, session_id: str = "user-1"):
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print(f"{'='*60}")

    config = { "configurable" : { "thread_id": session_id } }

    initial_state = {
        "question":question,
        "context":"",
        "answer":"",
        "next_step":"",
        "chat_history":[]
    }

    result = app.invoke(initial_state, config=config)
    print(f"\nFinal Answer: {result['answer']}")
    print(f"Sources used:\n{result['context']}")
    return result

if __name__ == "__main__":
    app = build_graph()

    chat(app, "What is multi-head attention?")

    chat(app, "What are the two RAG formulations in the original RAG paper?")

    chat(app, "What are the latest LLM models released in 2025?")

    chat(app, "Can you compare that with what the RAG paper says?")
