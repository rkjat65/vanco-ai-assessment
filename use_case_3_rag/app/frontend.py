"""
Streamlit frontend for the Hybrid RAG application.

Live demo interface for the Vanco assessment.
Shows: question input, answer with citations, retrieval evidence, graph concepts.

Run:
    streamlit run app/frontend.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="NCERT Physics RAG",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import pipeline components ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval indexes...")
def load_pipeline():
    from retrieval.hybrid_retriever import HybridRetriever
    from generation.answer_generator import AnswerGenerator
    from dotenv import load_dotenv
    load_dotenv()

    index_dir = os.getenv("INDEX_DIR", "indexes/")
    retriever = HybridRetriever(index_dir=index_dir)
    retriever.load()
    generator = AnswerGenerator()
    return retriever, generator


# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("⚛️ NCERT Physics RAG")
st.sidebar.markdown("""
**Hybrid Retrieval Architecture:**
- Semantic search (FAISS + MiniLM)
- Keyword search (BM25)
- Knowledge graph (NetworkX)
- Re-ranking (FlashRank)
- Generation (Claude)
""")

top_n = st.sidebar.slider("Chunks to retrieve", min_value=3, max_value=10, value=6)
show_debug = st.sidebar.checkbox("Show retrieval evidence", value=True)
show_graph = st.sidebar.checkbox("Show graph concepts", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sample questions:**")
sample_qs = [
    "What is Gauss's Law?",
    "State and derive Coulomb's Law",
    "What is the electric field inside a conductor?",
    "Explain the principle of superposition",
    "What is the relation between electric field and potential?",
    "Derive the expression for capacitance of a parallel plate capacitor",
    "Explain Kirchhoff's laws",
    "What is electromagnetic induction?",
    "State Faraday's law of induction",
    "What is the photoelectric effect?",
]
for q in sample_qs:
    if st.sidebar.button(q, key=f"sample_{q[:20]}"):
        st.session_state["question"] = q

# ── Main content ───────────────────────────────────────────────────────────
st.title("NCERT Class 12 Physics — Grounded Q&A")
st.caption("Answers are grounded exclusively in the NCERT Physics textbook. "
           "Questions outside the document will be flagged.")

# Question input
question = st.text_input(
    "Ask a physics question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is Gauss's Law and how is it derived?",
    key="main_question_input",
)

col1, col2 = st.columns([1, 5])
ask_btn = col1.button("Ask", type="primary", use_container_width=True)
clear_btn = col2.button("Clear", use_container_width=False)

if clear_btn:
    st.session_state["question"] = ""
    st.rerun()

if ask_btn and question.strip():
    try:
        retriever, generator = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load indexes: {e}\n\nRun `python ingestion/ingest_pipeline.py --pdf <pdf>` first.")
        st.stop()

    with st.spinner("Retrieving and generating answer..."):
        retrieval_result = retriever.retrieve(question, final_top_n=top_n)
        answer_data = generator.generate_with_retrieval_trace(question, retrieval_result)

    # ── Answer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Answer")
    st.markdown(answer_data["answer"])

    # ── Citations ──────────────────────────────────────────────────────────
    if answer_data.get("citations"):
        st.markdown("**Sources:**")
        for cit in answer_data["citations"]:
            st.markdown(f"- {cit}")

    # ── Token usage ────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Input tokens", answer_data.get("input_tokens", 0))
    col_b.metric("Output tokens", answer_data.get("output_tokens", 0))
    col_c.metric("Chunks used", answer_data.get("chunks_used", 0))

    # ── Retrieval evidence ─────────────────────────────────────────────────
    if show_debug:
        st.markdown("---")
        st.subheader("Retrieval Evidence")

        debug = retrieval_result.get("debug", {})
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Semantic results", debug.get("semantic_count", 0))
        d2.metric("Keyword results", debug.get("keyword_count", 0))
        d3.metric("Graph results", debug.get("graph_count", 0))
        d4.metric("After merge+rerank", debug.get("final_count", 0))

        for i, chunk in enumerate(retrieval_result.get("chunks", []), 1):
            score = chunk.get("rerank_score") or chunk.get("rrf_score", 0)
            source = chunk.get("retrieval_source", "")
            citation = chunk.get("citation", "")

            with st.expander(f"Chunk {i} — {citation} | {source} | score={score:.3f}"):
                st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id')}`")
                st.markdown(f"**Block types:** {', '.join(chunk.get('block_types', []))}")
                st.text_area(
                    "Text",
                    chunk.get("text", ""),
                    height=150,
                    key=f"chunk_text_{i}",
                    disabled=True,
                )

    # ── Graph concepts ─────────────────────────────────────────────────────
    if show_graph:
        st.markdown("---")
        st.subheader("Knowledge Graph — Related Concepts")

        retriever_obj, _ = load_pipeline()
        graph_results = retrieval_result.get("debug", {}).get("graph_results", [])
        if graph_results:
            for gr in graph_results:
                matched = gr.get("matched_concepts", [])
                if matched:
                    st.markdown(f"**Matched concepts:** {', '.join(matched)}")
        else:
            st.info("No graph concepts matched this query directly. "
                    "The semantic and keyword results cover the answer.")

elif ask_btn and not question.strip():
    st.warning("Please enter a question.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Hybrid RAG System | Vector DB: FAISS | Graph: NetworkX/Neo4j | "
    "Keyword: BM25 | LLM: Claude | Built for Vanco AI Assessment"
)
