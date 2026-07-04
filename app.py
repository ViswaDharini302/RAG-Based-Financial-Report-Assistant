# The user interface of the  project

import streamlit as st
import os
import time
import pandas as pd
from ragengine import RAGEngine
from ingest import ingest_pdf
from database import init_db, save_query, get_all_queries, get_top_questions
st.set_page_config(
    page_title="Financial Report AI",
    page_icon="📈",
    layout="wide",                 # Use full browser width
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* ── Overall background ── */
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3040;
    }
    /* ── Sidebar text ── */
    [data-testid="stSidebar"] * {
    color: #ffffff !important;
    }

    /* ── Big title at the top ── */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #38bdf8  !important;
        margin-bottom: 0.2rem;
        background: none !important;
        -webkit-text-fill-color: #38bdf8 !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #00d4aa;
        border-bottom: 2px solid #00d4aa30;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Answer box ── */
    .answer-box {
        background: #1a2235;
        border: 1px solid #00d4aa40;
        border-left: 4px solid #00d4aa;
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        line-height: 1.7;
        font-size: 0.97rem;
    }

    /* ── Source tag pills ── */
    .source-tag {
        display: inline-block;
        background: #0099ff20;
        border: 1px solid #0099ff50;
        color: #0099ff;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.82rem;
        margin: 0.2rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #1a2235;
        border: 1px solid #2a3040;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.2rem;
    }

    /* ── Info box ── */
    .info-box {
        background: #1a2235;
        border: 1px solid #2a3040;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin: 0.8rem 0;
    }

    /* ── Step box (for tutorial steps) ── */
.step-box {
    background: #161b27;
    border-left: 3px solid #0099ff;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}

/* ── Sample Question Buttons ── */
.stButton > button {
    background-color: #1a2235;      /* Dark box background */
    color: #38bdf8 !important;      /* Bright blue text */
    border: 1px solid #38bdf8;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
}

/* ── Hover effect when mouse comes over button ── */
.stButton > button:hover {
    background-color: #38bdf8;
    color: white !important;
    border-color: #7dd3fc;
}
            /* ── This Session expander box ── */
[data-testid="stExpander"] {
    background-color: transparent !important;
    border: 1px solid #2a3040;
    border-radius: 8px;
}

/* Expander title (question text) */
[data-testid="stExpander"] summary {
    color: #00d4ff !important;
    font-weight: 600;
}

/* Expander content (answer area) */
[data-testid="stExpander"] div[role="region"] {
    background-color: transparent !important;
    color: #e0e0e0 !important;
}
           
/* ── Hide Streamlit default footer ── */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

init_db()
if "rag_engine" not in st.session_state:
    with st.spinner(" Loading AI engine..."):
        st.session_state.rag_engine = RAGEngine()
# Keep track of the conversation (question-answer history this session)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# the sample question selected by the user
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# SIDEBAR — Navigation + quick status
with st.sidebar:
    st.markdown("## 📈 Financial Report AI")
    st.markdown("---")
    # Navigation menu
    page = st.radio(
        "Navigate",
        ["🏠 Ask Questions", "📤 Upload Reports", "📊 Analytics"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    # System status indicator
    engine = st.session_state.rag_engine
    if engine.is_ready:
        st.success(f"✅ System Ready")
        sources = engine.get_available_sources()
        st.caption(f"📚 {len(sources)} report(s) loaded")
        #st.caption(f" {engine.index.ntotal} chunks indexed")
        st.caption(f"🔢 {engine.index.ntotal:,} chunks indexed")
    else:
        st.error("⚠️ System not ready")
        st.caption("Upload PDFs and run ingest.py")
    st.markdown("---")
    st.caption("Built with Python · LangChain · FAISS · Gemini")

# PAGE 1: ASK QUESTIONS 
if page == "🏠 Ask Questions":
    st.markdown('<div class="main-title">📈 Financial Report Assistant</div>', unsafe_allow_html=True)
    st.markdown("Ask questions in plain English — I'll search the uploaded financial reports for you.")
    st.markdown("---")
    engine = st.session_state.rag_engine
    col1, col2 = st.columns([3, 1])
    with col2:
        sources = engine.get_available_sources()
        source_options = ["All Reports"] + sources
        #selected_source = st.selectbox("🗂 Search in:", source_options)
        st.markdown(
        "<p style='color:white; margin-bottom:0px;'>🗂 Search in:</p>",
        unsafe_allow_html=True)
        selected_source = st.selectbox(
        "",
        source_options,
        label_visibility="collapsed")

    with col1:
        # Sample question buttons
        st.markdown('<div class="section-header"> Try a sample question</div>', unsafe_allow_html=True)
        sample_questions = [
            "What was the revenue growth last year?",
            "What are the main business risks?",
            "How much profit did the company make?",
            "What are the future growth plans?",
            "What is the total debt of the company?",
            ]
        cols = st.columns(len(sample_questions))
        for i, q in enumerate(sample_questions):
            if cols[i].button(q[:30] + "…", key=f"sample_{i}"):
                st.session_state.current_question=q
    st.markdown("---")

    # Main question input Section
    st.markdown('<div class="section-header">🔍 Ask Your Question</div>', unsafe_allow_html=True)
    question = st.text_area(
        label="Your question",
        key="current_question",
        placeholder="e.g. What was the company's revenue in 2024?",
        height=80,
        label_visibility="collapsed"
        )           
    ask_btn = st.button(" Get Answer", type="primary", use_container_width=False)

    #  Process question 
    if ask_btn and question.strip():
        if not engine.is_ready:
            st.error("❌ System not ready. Please check the setup instructions in 'About & Setup'.")
        else:
            with st.spinner("🔍 Searching reports and generating answer..."):
                start_time = time.time()
                # THE CORE CALL — this does the RAG magic!
                result = engine.ask(question, source_filter=selected_source)

                elapsed = time.time() - start_time
            # Display answer
            st.markdown('<div class="section-header">💬 Answer</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True
            )
            # Show which reports were used 
            if result["sources"]:
                st.markdown("**📂 Sources used:**")
                source_html = " ".join(
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in result["sources"]
                )
                st.markdown(source_html, unsafe_allow_html=True)
            st.caption(f"⏱️ Answer generated in {elapsed:.2f} seconds")
            #show raw retrieved chunks
            with st.expander("🔬 See retrieved context (advanced)"):
                for i, (chunk, src) in enumerate(result["chunks"], 1):
                    st.markdown(f"**Chunk {i}** · from `{src}`")
                    #st.text(chunk[:400] + ("..." if len(chunk) > 400 else ""))
                    st.markdown(f"<p style='color:white;'>{chunk[:400] + ('...' if len(chunk) > 400 else '')}</p>",
                    unsafe_allow_html=True)
                    st.markdown("---")
            # Save to database & chat history
            save_query(
                report_name=selected_source,
                question=question,
                answer=result["answer"]
            )
            st.session_state.chat_history.append({
                "question": question,
                "answer":   result["answer"],
                "sources":  result["sources"]
            })
    elif ask_btn and not question.strip():
        st.warning("⚠️ Please type a question first!")
    #Chat history this session
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown('<div class="section-header">🕐 This Session</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.chat_history[-5:]):   # Show last 5
            with st.expander(f"Q: {item['question'][:70]}..."):
                st.markdown(item["answer"])

# PAGE 2: UPLOAD REPORTS
elif page == "📤 Upload Reports":
    st.markdown('<div class="main-title">📤 Upload Financial Reports</div>', unsafe_allow_html=True)
    st.markdown("Upload PDF files (annual reports, balance sheets, etc.)")
    st.markdown("---")
    # File uploader
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,  # Allow multiple files at once
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        process_btn = st.button(" Process & Index PDFs", type="primary")
        if process_btn:
            os.makedirs("reports", exist_ok=True)
            progress_bar = st.progress(0)
            status_text  = st.empty()
            for i, uploaded_file in enumerate(uploaded_files):
                pdf_path = os.path.join("reports", uploaded_file.name)
                # Save uploaded file to disk
                status_text.text(f"💾 Saving {uploaded_file.name}...")
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                # Run the ingest pipeline
                status_text.text(f" Processing {uploaded_file.name}...")
                try:
                    ingest_pdf(pdf_path)
                    st.success(f"✅ {uploaded_file.name} — indexed successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text("🔄 Reloading AI engine with new data...")
            st.session_state.rag_engine = RAGEngine()   # Reload with new data
            st.balloons()
            st.success(" All done! Go to 'Ask Questions' to start querying.")
    st.markdown("---")
    st.markdown('<div class="section-header">📂 Currently Indexed Reports</div>', unsafe_allow_html=True)
    engine = st.session_state.rag_engine
    if engine.is_ready:
        sources = engine.get_available_sources()
        if sources:
            for s in sources:
                st.markdown(f'<div class="info-box">📄 {s}</div>', unsafe_allow_html=True)
        else:
            st.info("No reports indexed yet.")
    else:
        st.info("No reports indexed yet. Upload PDFs above.")

# PAGE 3: ANALYTICS  
elif page == "📊 Analytics":
    st.markdown('<div class="main-title">📊 Query Analytics</div>', unsafe_allow_html=True)
    st.markdown("See what questions have been asked and system statistics.")
    st.markdown("---")
    # Summary metrics
    all_queries = get_all_queries()
    engine      = st.session_state.rag_engine
    total_chunks = engine.index.ntotal if engine.is_ready else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{len(all_queries)}</div>
                <div class="metric-label">Total Questions Asked</div>
            </div>""", unsafe_allow_html=True)
    with col2:
        sources = engine.get_available_sources() if engine.is_ready else []
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{len(sources)}</div>
                <div class="metric-label">Reports Indexed</div>
            </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{total_chunks:,}</div>
                <div class="metric-label">Chunks in Database</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    # Recent query history
    st.markdown('<div class="section-header">📋 Recent Questions</div>', unsafe_allow_html=True)
    if all_queries:
        df = pd.DataFrame(
            all_queries,
            columns=["ID", "Report", "Question", "Answer", "Time"]
        )
        # table with just question and time
        display_df = df[["Time", "Report", "Question"]].head(20)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        # Most popular questions 
        st.markdown('<div class="section-header"> Most Asked Questions</div>', unsafe_allow_html=True)
        top_q = get_top_questions(5)
        if top_q:
            for q, count in top_q:
                st.markdown(
                    f'<div class="info-box">🔁 {count}x — {q}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No questions asked yet. Go to 'Ask Questions' to get started!")



