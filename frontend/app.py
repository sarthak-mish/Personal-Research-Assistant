import streamlit as st
import requests
import uuid
import pandas as pd
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────
MCP_SERVER_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL STYLING ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background-color: #0d0f14;
        color: #c9d1d9;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0a0c10 !important;
        border-right: 1px solid #1e2530;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d0f14;
        border-bottom: 1px solid #1e2530;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6b7280;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #161b22 !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #1e2530;
        border-radius: 10px;
        padding: 4px 8px;
        margin-bottom: 12px;
    }

    /* ── Input ── */
    .stChatInput textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        border-radius: 10px !important;
    }
    .stChatInput textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
        border-radius: 6px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1f2937;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem !important;
        color: #58a6ff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #1e2530;
        border-radius: 10px;
        padding: 16px;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e2530;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Divider ── */
    hr {
        border-color: #1e2530 !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #1e2530 !important;
        border-radius: 6px !important;
        color: #8b949e !important;
        font-size: 12px !important;
    }

    /* ── Custom Badges ── */
    .badge-rag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #0d1f3c;
        color: #58a6ff;
        border: 1px solid #1e3a5f;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-web {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #0d2818;
        color: #3fb950;
        border: 1px solid #1e4a2a;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-unknown {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background-color: #1e1e2e;
        color: #8b949e;
        border: 1px solid #2a2a3e;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ── Source box ── */
    .source-box {
        background-color: #0d1117;
        border-left: 3px solid #58a6ff;
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        color: #8b949e;
        margin-top: 8px;
    }
    .source-box-web {
        border-left-color: #3fb950;
    }

    /* ── Timestamp ── */
    .ts {
        font-size: 10px;
        color: #3d444d;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 4px;
    }

    /* ── Score pill ── */
    .score-good  { color: #3fb950; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
    .score-mid   { color: #d29922; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }
    .score-bad   { color: #f85149; font-family: 'IBM Plex Mono', monospace; font-weight: 600; }

    /* ── Sidebar title ── */
    .sidebar-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 18px;
        font-weight: 600;
        color: #58a6ff;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .sidebar-sub {
        font-size: 11px;
        color: #3d444d;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 16px;
    }

    /* ── Section headers ── */
    .section-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #3d444d;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 8px;
    }

    /* ── Status dot ── */
    .dot-green { color: #3fb950; font-size: 10px; }
    .dot-red   { color: #f85149; font-size: 10px; }
    .dot-grey  { color: #3d444d; font-size: 10px; }

    /* ── Sample question button ── */
    .stButton > button[kind="secondary"] {
        text-align: left;
        font-size: 12px;
        padding: 6px 10px;
    }

    /* ── Welcome banner ── */
    .welcome-banner {
        background: linear-gradient(135deg, #0d1f3c 0%, #0d2818 100%);
        border: 1px solid #1e2530;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .welcome-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 22px;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 6px;
    }
    .welcome-sub {
        font-size: 13px;
        color: #8b949e;
        line-height: 1.6;
    }

    /* ── Weak question card ── */
    .weak-card {
        background-color: #1a0e0e;
        border: 1px solid #3f1e1e;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 13px;
    }

    /* ── Progress bar override ── */
    .stProgress > div > div {
        background-color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_ok" not in st.session_state:
    st.session_state.server_ok = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── API HELPERS ───────────────────────────────────────────────────
def check_health() -> bool:
    try:
        r = requests.get(f"{MCP_SERVER_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_tools() -> list:
    try:
        r = requests.get(f"{MCP_SERVER_URL}/tools", timeout=3)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def invoke_agent(query: str, session_id: str):
    try:
        r = requests.post(
            f"{MCP_SERVER_URL}/invoke",
            json={"query": query, "session_id": session_id},
            timeout=120
        )
        if r.status_code == 200:
            return r.json(), None
        return None, f"Server returned {r.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to MCP server. Is it running on port 8000?"
    except requests.exceptions.Timeout:
        return None, "Request timed out after 120s."
    except Exception as e:
        return None, str(e)


def get_metrics():
    try:
        r = requests.get(f"{MCP_SERVER_URL}/metrics", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ── BADGE HELPERS ─────────────────────────────────────────────────
def tool_badge_html(tool: str) -> str:
    if tool == "rag":
        return '<span class="badge-rag">📚 RAG</span>'
    elif tool == "web_search":
        return '<span class="badge-web">🌐 Web Search</span>'
    return '<span class="badge-unknown">🔧 unknown</span>'


def score_color(val) -> str:
    if not isinstance(val, (int, float)):
        return val
    if val >= 0.8:
        return f'<span class="score-good">{val}</span>'
    elif val >= 0.6:
        return f'<span class="score-mid">{val}</span>'
    return f'<span class="score-bad">{val}</span>'


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⬡ ResearchAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">LangGraph · RAG · DeepSearch · MCP</div>', unsafe_allow_html=True)
    st.divider()

    # ── Server Status
    st.markdown('<div class="section-label">Server</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        if st.session_state.server_ok is True:
            st.markdown('<span class="dot-green">●</span> MCP Server Online', unsafe_allow_html=True)
        elif st.session_state.server_ok is False:
            st.markdown('<span class="dot-red">●</span> Server Offline', unsafe_allow_html=True)
        else:
            st.markdown('<span class="dot-grey">●</span> Not checked', unsafe_allow_html=True)
    with col_s2:
        if st.button("Ping", use_container_width=True):
            st.session_state.server_ok = check_health()
            st.rerun()

    if st.session_state.server_ok is False:
        st.code("python server/mcp_server.py", language="bash")

    st.divider()

    # ── Session
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:13px; '
        f'color:#58a6ff; background:#0d1f3c; padding:6px 10px; '
        f'border-radius:6px; letter-spacing:1px;">'
        f'ID: {st.session_state.session_id}</div>',
        unsafe_allow_html=True
    )
    st.caption(f"{len(st.session_state.messages)} messages in history")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()
    with col_n2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # ── Available Tools
    st.markdown('<div class="section-label">Available Tools</div>', unsafe_allow_html=True)
    tools = get_tools()
    if tools:
        for t in tools:
            with st.expander(f"🔧 {t['name']}"):
                st.caption(t.get("description", ""))
    else:
        st.caption("Start server to discover tools")

    st.divider()

    # ── Sample Questions
    st.markdown('<div class="section-label">Sample Questions</div>', unsafe_allow_html=True)
    samples = [
        "What is multi-head attention?",
        "How does ReAct combine reasoning and acting?",
        "What are the two RAG formulations?",
        "What LLMs were released in 2025?",
        "Compare transformer attention with RAG retrieval",
    ]
    for q in samples:
        if st.button(q[:45] + ("..." if len(q) > 45 else ""),
                     use_container_width=True,
                     key=f"sq_{q[:20]}"):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:10px; color:#3d444d; font-family:\'IBM Plex Mono\',monospace;">'
        'RAG · LangGraph · DeepSearch<br>RAGAS · MCP · OpenRouter</div>',
        unsafe_allow_html=True
    )


# ── MAIN TABS ─────────────────────────────────────────────────────
tab_chat, tab_metrics, tab_debug = st.tabs([
    "💬  Chat",
    "📊  Metrics Dashboard",
    "🛠️  Debug & Tools"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════
with tab_chat:

    # Welcome banner (only when no messages)
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-banner">
            <div class="welcome-title">Research Assistant</div>
            <div class="welcome-sub">
                Ask questions about your uploaded research papers or anything on the web.<br>
                The agent automatically decides whether to search your documents (RAG) or the live web.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Stats row
    if st.session_state.messages:
        rag_count = sum(1 for m in st.session_state.messages if m.get("tool_used") == "rag")
        web_count = sum(1 for m in st.session_state.messages if m.get("tool_used") == "web_search")
        total = len(st.session_state.messages)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions Asked", total)
        c2.metric("📚 RAG", rag_count)
        c3.metric("🌐 Web", web_count)
        c4.metric("Session", st.session_state.session_id)
        st.divider()

    # ── Render chat history
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg["question"])
            st.markdown(f'<div class="ts">{msg.get("timestamp", "")}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            st.write(msg["answer"])

            # Tool badge + source
            col_b, col_s = st.columns([2, 8])
            with col_b:
                st.markdown(tool_badge_html(msg.get("tool_used", "")), unsafe_allow_html=True)
            with col_s:
                if msg.get("tool_used") == "rag":
                    st.markdown(
                        '<div class="source-box">📄 Sourced from uploaded research PDFs via Deep Search</div>',
                        unsafe_allow_html=True
                    )
                elif msg.get("tool_used") == "web_search":
                    st.markdown(
                        '<div class="source-box source-box-web">🌐 Sourced from live web search via Tavily</div>',
                        unsafe_allow_html=True
                    )

            # RAGAS scores (if available)
            if msg.get("ragas_scores"):
                scores = msg["ragas_scores"]
                with st.expander("📊 RAGAS Evaluation Scores", expanded=False):
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Faithfulness", scores.get("faithfulness", "—"))
                    sc2.metric("Context Precision", scores.get("context_precision", "—"))
                    sc3.metric("Answer Relevancy", scores.get("answer_relevancy", "—"))
                    sc4.metric("Overall", scores.get("overall_score", "—"))

    # ── Handle sample question click
    pending = st.session_state.pop("pending_question", None)

    # ── Chat input
    user_input = st.chat_input("Ask about your research papers or anything on the web...")
    query = pending or user_input

    if query:
        timestamp = datetime.now().strftime("%H:%M:%S")

        with st.chat_message("user"):
            st.write(query)
            st.markdown(f'<div class="ts">{timestamp}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Running deep search pipeline..."):
                result, error = invoke_agent(query, st.session_state.session_id)

            if error:
                st.error(f"**Error:** {error}")
                st.info("Make sure the MCP server is running: `python server/mcp_server.py`")
            else:
                st.write(result["answer"])

                col_b, col_s = st.columns([2, 8])
                with col_b:
                    st.markdown(tool_badge_html(result.get("tool_used", "")), unsafe_allow_html=True)
                with col_s:
                    if result.get("tool_used") == "rag":
                        st.markdown(
                            '<div class="source-box">📄 Sourced from uploaded research PDFs via Deep Search</div>',
                            unsafe_allow_html=True
                        )
                    elif result.get("tool_used") == "web_search":
                        st.markdown(
                            '<div class="source-box source-box-web">🌐 Sourced from live web search via Tavily</div>',
                            unsafe_allow_html=True
                        )

                # Show RAGAS scores if returned
                ragas = result.get("ragas_scores")
                if ragas:
                    with st.expander("📊 RAGAS Evaluation Scores", expanded=True):
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Faithfulness", ragas.get("faithfulness", "—"))
                        sc2.metric("Context Precision", ragas.get("context_precision", "—"))
                        sc3.metric("Answer Relevancy", ragas.get("answer_relevancy", "—"))
                        sc4.metric("Overall", ragas.get("overall_score", "—"))

                # Save to history
                st.session_state.messages.append({
                    "question": query,
                    "answer": result["answer"],
                    "tool_used": result.get("tool_used", "unknown"),
                    "timestamp": timestamp,
                    "ragas_scores": result.get("ragas_scores")
                })
                st.rerun()


# ══════════════════════════════════════════════════════════════════
# TAB 2 — METRICS DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("### 📊 RAGAS Observability Dashboard")
    st.caption("Scores are automatically computed after every question answered via RAG")

    col_r, _ = st.columns([1, 5])
    with col_r:
        refresh = st.button("↻ Refresh", use_container_width=True)

    metrics_data = get_metrics()
    summary = metrics_data.get("summary", {})
    evaluations = metrics_data.get("evaluations", [])

    if not summary:
        st.markdown("""
        <div style="background:#0d1117; border:1px dashed #30363d; border-radius:10px;
                    padding:40px; text-align:center; color:#3d444d; margin-top:20px;">
            <div style="font-size:32px; margin-bottom:12px;">📊</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; color:#6b7280;">
                No evaluations yet
            </div>
            <div style="font-size:12px; margin-top:8px;">
                Ask questions in the Chat tab — RAGAS scores will appear here automatically
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── KPI Row ───────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">Average Scores Across All Evaluations</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Faithfulness",
            summary.get("avg_faithfulness", "—"),
            help="Is the answer grounded in retrieved context? Higher = less hallucination"
        )
        k2.metric(
            "Context Precision",
            summary.get("avg_context_precision", "—"),
            help="Are the retrieved chunks actually relevant to the question?"
        )
        k3.metric(
            "Answer Relevancy",
            summary.get("avg_answer_relevancy", "—"),
            help="Does the answer directly address what was asked?"
        )
        k4.metric(
            "Overall Score",
            summary.get("avg_overall_score", "—"),
            help="Average of all metrics"
        )

        st.divider()

        # ── Score Bars ────────────────────────────────────────────
        st.markdown('<div class="section-label">Score Breakdown</div>', unsafe_allow_html=True)

        metric_map = {
            "Faithfulness": summary.get("avg_faithfulness", 0),
            "Context Precision": summary.get("avg_context_precision", 0),
            "Answer Relevancy": summary.get("avg_answer_relevancy", 0),
            "Overall": summary.get("avg_overall_score", 0),
        }
        for label, val in metric_map.items():
            if isinstance(val, (int, float)):
                col_l, col_p = st.columns([2, 8])
                with col_l:
                    color = "#3fb950" if val >= 0.8 else "#d29922" if val >= 0.6 else "#f85149"
                    st.markdown(
                        f'<div style="font-family:\'IBM Plex Mono\',monospace; '
                        f'font-size:12px; color:{color}; padding-top:6px;">{label}</div>',
                        unsafe_allow_html=True
                    )
                with col_p:
                    st.progress(float(val))

        st.divider()

        # ── History Table ─────────────────────────────────────────
        total_evals = summary.get("total_evaluations", len(evaluations))
        st.markdown(f'<div class="section-label">Evaluation History — {total_evals} runs</div>', unsafe_allow_html=True)

        if evaluations:
            df = pd.DataFrame(evaluations)

            # Select and rename columns
            display_cols = {
                "timestamp": "Time",
                "question": "Question",
                "faithfulness": "Faithfulness",
                "context_precision": "Ctx Precision",
                "answer_relevancy": "Ans Relevancy",
                "overall_score": "Overall"
            }
            available = {k: v for k, v in display_cols.items() if k in df.columns}
            df_display = df[list(available.keys())].rename(columns=available)

            # Truncate question
            if "Question" in df_display.columns:
                df_display["Question"] = df_display["Question"].str[:80] + "..."

            # Color numeric columns
            numeric_cols = ["Faithfulness", "Ctx Precision", "Ans Relevancy", "Overall"]
            existing_numeric = [c for c in numeric_cols if c in df_display.columns]

            def color_cell(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 0.8:
                    return "background-color: #0d2818; color: #3fb950"
                elif val >= 0.6:
                    return "background-color: #1f1a0d; color: #d29922"
                return "background-color: #1a0e0e; color: #f85149"

            styled = (
                df_display.style
                .applymap(color_cell, subset=existing_numeric)
                .set_properties(**{
                    "font-family": "IBM Plex Mono, monospace",
                    "font-size": "12px"
                })
            )
            st.dataframe(styled, use_container_width=True, height=300)

        st.divider()

        # ── Min / Max ─────────────────────────────────────────────
        st.markdown('<div class="section-label">Score Range</div>', unsafe_allow_html=True)
        range_cols = st.columns(3)
        range_metrics = [
            ("faithfulness", "Faithfulness"),
            ("context_precision", "Ctx Precision"),
            ("overall_score", "Overall"),
        ]
        for i, (key, label) in enumerate(range_metrics):
            with range_cols[i]:
                mn = summary.get(f"min_{key}", "—")
                mx = summary.get(f"max_{key}", "—")
                st.markdown(
                    f'<div style="background:#161b22; border:1px solid #1e2530; '
                    f'border-radius:8px; padding:12px;">'
                    f'<div style="font-size:10px; color:#3d444d; font-family:\'IBM Plex Mono\',monospace; '
                    f'text-transform:uppercase; letter-spacing:1px;">{label}</div>'
                    f'<div style="margin-top:8px; font-family:\'IBM Plex Mono\',monospace;">'
                    f'<span style="color:#f85149;">↓ {mn}</span>'
                    f'&nbsp;&nbsp;'
                    f'<span style="color:#3fb950;">↑ {mx}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # ── Weak Questions ─────────────────────────────────────────
        weak = [e for e in evaluations if isinstance(e.get("overall_score"), float) and e["overall_score"] < 0.6]
        st.markdown(f'<div class="section-label">⚠️ Questions Needing Improvement — {len(weak)} found</div>', unsafe_allow_html=True)

        if weak:
            for e in weak:
                st.markdown(
                    f'<div class="weak-card">'
                    f'<span style="color:#8b949e;">Q:</span> {e["question"][:120]}<br>'
                    f'<span style="color:#f85149; font-family:\'IBM Plex Mono\',monospace; font-size:11px;">'
                    f'Overall: {e.get("overall_score", "?")} &nbsp;|&nbsp; '
                    f'Faithfulness: {e.get("faithfulness", "?")} &nbsp;|&nbsp; '
                    f'Precision: {e.get("context_precision", "?")}'
                    f'</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="background:#0d2818; border:1px solid #1e4a2a; border-radius:8px; '
                'padding:12px 16px; color:#3fb950; font-size:13px;">'
                '✅ All evaluated questions scoring above 0.6'
                '</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — DEBUG & TOOLS
# ══════════════════════════════════════════════════════════════════
with tab_debug:
    st.markdown("### 🛠️ Debug & System Tools")
    st.caption("Inspect server state, test endpoints directly, and view raw responses")

    # ── Server Info
    st.markdown('<div class="section-label">Server Endpoints</div>', unsafe_allow_html=True)
    endpoints = [
        ("GET",  "/health",     "Health check"),
        ("GET",  "/tools",      "List available MCP tools"),
        ("POST", "/invoke",     "Run full agent (RAG or Web)"),
        ("POST", "/rag",        "Direct RAG search (bypass router)"),
        ("POST", "/web-search", "Direct web search (bypass router)"),
        ("GET",  "/metrics",    "RAGAS evaluation history"),
        ("GET",  "/docs",       "FastAPI Swagger UI"),
    ]
    for method, path, desc in endpoints:
        color = "#3fb950" if method == "GET" else "#d29922"
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:12px; '
            f'padding:8px 12px; border-bottom:1px solid #1e2530; font-size:13px;">'
            f'<span style="font-family:\'IBM Plex Mono\',monospace; color:{color}; '
            f'font-size:11px; width:36px;">{method}</span>'
            f'<span style="font-family:\'IBM Plex Mono\',monospace; color:#58a6ff;">{path}</span>'
            f'<span style="color:#6b7280; font-size:12px;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Direct RAG Test
    st.markdown('<div class="section-label">Direct RAG Search Test</div>', unsafe_allow_html=True)
    rag_query = st.text_input("RAG Query", placeholder="What is multi-head attention?", key="rag_debug")
    if st.button("Run RAG Search", key="run_rag"):
        if rag_query:
            try:
                r = requests.post(
                    f"{MCP_SERVER_URL}/rag",
                    json={"query": rag_query},
                    timeout=60
                )
                if r.status_code == 200:
                    st.success("✅ RAG search completed")
                    st.json(r.json())
                else:
                    st.error(f"Error {r.status_code}")
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ── Direct Web Search Test
    st.markdown('<div class="section-label">Direct Web Search Test</div>', unsafe_allow_html=True)
    web_query = st.text_input("Web Query", placeholder="Latest LLM releases 2025", key="web_debug")
    if st.button("Run Web Search", key="run_web"):
        if web_query:
            try:
                r = requests.post(
                    f"{MCP_SERVER_URL}/web-search",
                    json={"query": web_query},
                    timeout=30
                )
                if r.status_code == 200:
                    st.success("✅ Web search completed")
                    st.json(r.json())
                else:
                    st.error(f"Error {r.status_code}")
            except Exception as e:
                st.error(str(e))

    st.divider()

    # ── Raw Session State
    st.markdown('<div class="section-label">Session State</div>', unsafe_allow_html=True)
    with st.expander("View raw session state"):
        safe_state = {
            "session_id": st.session_state.session_id,
            "message_count": len(st.session_state.messages),
            "server_ok": st.session_state.server_ok,
            "messages_preview": [
                {"q": m["question"][:60], "tool": m.get("tool_used")}
                for m in st.session_state.messages[-5:]
            ]
        }
        st.json(safe_state)

    st.divider()

    # ── How to run
    st.markdown('<div class="section-label">How to Run</div>', unsafe_allow_html=True)
    st.code("""
# Terminal 1 — Start MCP server
cd research-assistant
python server/mcp_server.py

# Terminal 2 — Start Streamlit frontend
cd research-assistant
streamlit run frontend/app.py
    """, language="bash")

    st.markdown(
        '<div style="font-size:12px; color:#6b7280; font-family:\'IBM Plex Mono\',monospace;">'
        'MCP Server → http://localhost:8000<br>'
        'Swagger UI → http://localhost:8000/docs<br>'
        'Streamlit  → http://localhost:8501'
        '</div>',
        unsafe_allow_html=True
    )