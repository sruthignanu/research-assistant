# ============================================================
# AI-Powered Research Paper Intelligence Assistant
# Backend functions preserved. UI/UX refined while keeping
# Streamlit's default appearance intact.
# ============================================================

import streamlit as st
import pickle
import faiss
import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(
    page_title="AI-Powered Research Paper Intelligence Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Light CSS: only spacing, cards, hierarchy ----------------
st.markdown("""
<style>
    /* Generous, calm spacing — extra top padding so the hero is fully visible */
    .block-container {
        padding-top: 4rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hide Streamlit's default header bar */
    header[data-testid="stHeader"] {
        height: 0rem;
        background: transparent;
    }

    /* Global typography — slightly larger and more readable */
    html, body, [class*="css"] {
        font-size: 16px;
    }

    /* Hero block — larger and more prominent */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1.1rem;
        opacity: 0.85;
        margin: 0 0 1.4rem 0;
        line-height: 1.5;
    }

    /* Stat chips */
    .chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .chip {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 20px;
        padding: 0.4rem 0.95rem;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .chip b { font-weight: 700; }

    /* Section label — larger, more readable */
    .section-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.8;
        font-weight: 700;
        margin: 1.6rem 0 0.6rem 0;
    }

    /* Card */
    .card {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        line-height: 1.4;
    }
    .card-meta { font-size: 0.88rem; opacity: 0.75; }

    /* Source card */
    .source-card {
        border: 1px solid rgba(128,128,128,0.22);
        border-left: 3px solid #FF4B4B;
        border-radius: 8px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.7rem;
    }
    .source-head {
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.5rem;
    }
    .source-title {
        font-weight: 600; font-size: 1rem; line-height: 1.45;
    }
    .source-badge {
        font-size: 0.75rem; font-weight: 600; padding: 3px 10px;
        border-radius: 999px; white-space: nowrap;
        border: 1px solid rgba(128,128,128,0.3);
    }
    .source-badge.high { color: #166534; border-color: #16A34A; }
    .source-badge.mid  { color: #92400E; border-color: #D97706; }
    .source-badge.low  { color: #991B1B; border-color: #DC2626; }
    .source-excerpt {
        font-size: 0.94rem; line-height: 1.6; margin-top: 0.4rem;
    }
    .source-file {
        font-size: 0.78rem; opacity: 0.7; margin-top: 0.6rem;
        font-family: ui-monospace, "SF Mono", Consolas, monospace;
    }

    /* Empty state */
    .empty-state {
        text-align: center; padding: 2.6rem 1.6rem;
        border: 1px dashed rgba(128,128,128,0.35);
        border-radius: 10px;
    }
    .empty-state h3 {
        margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 600;
    }
    .empty-state p {
        margin: 0; opacity: 0.8; font-size: 1rem; line-height: 1.55;
    }

    /* Footer */
    .footer {
        text-align: center; margin-top: 3rem; padding-top: 1.3rem;
        border-top: 1px solid rgba(128,128,128,0.2);
        font-size: 0.88rem; opacity: 0.75;
    }

    /* Sidebar tweaks — slightly larger text */
    section[data-testid="stSidebar"] h3 { font-size: 1.1rem; line-height: 1.4; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BACKEND — unchanged
# ============================================================

@st.cache_resource
def load_resources():
    with open("data/all_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("data/all_index.faiss")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    load_dotenv(".env")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return chunks, index, model, client


def get_paper_title(pdf_name):
    for c in chunks:
        if c["source"] == pdf_name:
            lines = [l.strip() for l in c["text"].split("\n") if l.strip()]
            for line in lines[:15]:
                if line.startswith("arXiv:") or "[cs." in line:
                    continue
                if len(line) > 20:
                    return line[:90]
            return pdf_name
    return pdf_name


def search(query, k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_emb, dtype="float32"), k)
    return [
        {"score": float(s), "text": chunks[i]["text"], "source": chunks[i]["source"]}
        for s, i in zip(scores[0], indices[0])
    ]


def ask(query, k=5):
    results = search(query, k=k)
    context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(results))
    prompt = f"""You are a research assistant. Answer using the context below.
If the context doesn't contain the answer, say "I don't have enough information."
Cite sources as [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text, results


def summarize_paper(pdf_name):
    paper_chunks = [c["text"] for c in chunks if c["source"] == pdf_name]
    if not paper_chunks:
        return f"No chunks found for {pdf_name}"
    n = len(paper_chunks)
    sample_idx = list(range(0, min(10, n))) + \
                 list(range(n//2-5, min(n//2+5, n))) + \
                 list(range(max(0, n-10), n))
    sample_idx = sorted(set(sample_idx))
    context = "\n\n".join(paper_chunks[i] for i in sample_idx)
    prompt = f"""Summarize this research paper:

**Problem:** (1-2 sentences)
**Method:** (2-3 sentences)
**Dataset:** (1 sentence or "Not specified")
**Results:** (1-2 sentences)

Paper text:
{context}

Summary:"""
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text


def score_class(score):
    if score >= 0.6: return "high"
    if score >= 0.45: return "mid"
    return "low"


# ============================================================
# DATA
# ============================================================

chunks, index, model, client = load_resources()
available_papers = sorted(set(c["source"] for c in chunks))
paper_titles = {p: get_paper_title(p) for p in available_papers}

# ============================================================
# AI-Powered Research Paper Intelligence Assistant
# Backend functions preserved. UI/UX refined while keeping
# Streamlit's default appearance intact.
# ============================================================

import streamlit as st
import pickle
import faiss
import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(
    page_title="AI-Powered Research Paper Intelligence Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Light CSS: only spacing, cards, hierarchy ----------------
st.markdown("""
<style>
    /* Generous, calm spacing — extra top padding so the hero is fully visible */
    .block-container {
        padding-top: 4rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hide Streamlit's default header bar */
    header[data-testid="stHeader"] {
        height: 0rem;
        background: transparent;
    }

    /* Global typography baseline */
    html, body, [class*="css"] {
        font-size: 16px;
    }

    /* Stat chips */
    .chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .chip {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 20px;
        padding: 0.4rem 0.95rem;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .chip b { font-weight: 700; }

    /* Section label */
    .section-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.8;
        font-weight: 700;
        margin: 1.6rem 0 0.6rem 0;
    }

    /* Card */
    .card {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        line-height: 1.4;
    }
    .card-meta { font-size: 0.88rem; opacity: 0.75; }

    /* Source card */
    .source-card {
        border: 1px solid rgba(128,128,128,0.22);
        border-left: 3px solid #FF4B4B;
        border-radius: 8px;
        padding: 0.95rem 1.2rem;
        margin-bottom: 0.7rem;
    }
    .source-head {
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 0.6rem; flex-wrap: wrap; margin-bottom: 0.5rem;
    }
    .source-title {
        font-weight: 600; font-size: 1rem; line-height: 1.45;
    }
    .source-badge {
        font-size: 0.75rem; font-weight: 600; padding: 3px 10px;
        border-radius: 999px; white-space: nowrap;
        border: 1px solid rgba(128,128,128,0.3);
    }
    .source-badge.high { color: #166534; border-color: #16A34A; }
    .source-badge.mid  { color: #92400E; border-color: #D97706; }
    .source-badge.low  { color: #991B1B; border-color: #DC2626; }
    .source-excerpt {
        font-size: 0.94rem; line-height: 1.6; margin-top: 0.4rem;
    }
    .source-file {
        font-size: 0.78rem; opacity: 0.7; margin-top: 0.6rem;
        font-family: ui-monospace, "SF Mono", Consolas, monospace;
    }

    /* Empty state */
    .empty-state {
        text-align: center; padding: 2.6rem 1.6rem;
        border: 1px dashed rgba(128,128,128,0.35);
        border-radius: 10px;
    }
    .empty-state h3 {
        margin: 0 0 0.5rem 0; font-size: 1.2rem; font-weight: 600;
    }
    .empty-state p {
        margin: 0; opacity: 0.8; font-size: 1rem; line-height: 1.55;
    }

    /* Footer */
    .footer {
        text-align: center; margin-top: 3rem; padding-top: 1.3rem;
        border-top: 1px solid rgba(128,128,128,0.2);
        font-size: 0.88rem; opacity: 0.75;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] h3 { font-size: 1.1rem; line-height: 1.4; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BACKEND — unchanged
# ============================================================

@st.cache_resource
def load_resources():
    with open("data/all_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("data/all_index.faiss")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    load_dotenv(".env")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return chunks, index, model, client


def get_paper_title(pdf_name):
    for c in chunks:
        if c["source"] == pdf_name:
            lines = [l.strip() for l in c["text"].split("\n") if l.strip()]
            for line in lines[:15]:
                if line.startswith("arXiv:") or "[cs." in line:
                    continue
                if len(line) > 20:
                    return line[:90]
            return pdf_name
    return pdf_name


def search(query, k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_emb, dtype="float32"), k)
    return [
        {"score": float(s), "text": chunks[i]["text"], "source": chunks[i]["source"]}
        for s, i in zip(scores[0], indices[0])
    ]


def ask(query, k=5):
    results = search(query, k=k)
    context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(results))
    prompt = f"""You are a research assistant. Answer using the context below.
If the context doesn't contain the answer, say "I don't have enough information."
Cite sources as [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text, results


def summarize_paper(pdf_name):
    paper_chunks = [c["text"] for c in chunks if c["source"] == pdf_name]
    if not paper_chunks:
        return f"No chunks found for {pdf_name}"
    n = len(paper_chunks)
    sample_idx = list(range(0, min(10, n))) + \
                 list(range(n//2-5, min(n//2+5, n))) + \
                 list(range(max(0, n-10), n))
    sample_idx = sorted(set(sample_idx))
    context = "\n\n".join(paper_chunks[i] for i in sample_idx)
    prompt = f"""Summarize this research paper:

**Problem:** (1-2 sentences)
**Method:** (2-3 sentences)
**Dataset:** (1 sentence or "Not specified")
**Results:** (1-2 sentences)

Paper text:
{context}

Summary:"""
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text


def score_class(score):
    if score >= 0.6: return "high"
    if score >= 0.45: return "mid"
    return "low"


# ============================================================
# DATA
# ============================================================

chunks, index, model, client = load_resources()
available_papers = sorted(set(c["source"] for c in chunks))
paper_titles = {p: get_paper_title(p) for p in available_papers}


# ============================================================
# UI HELPERS
# ============================================================

def render_header():
    # Inline styles on h1/p guarantee the sizes win against Streamlit's
    # default paragraph rules, which otherwise override class-based CSS.
    st.markdown(f"""
    <div style="margin-bottom: 0.6rem;">
        <h1 style="font-size: 2.6rem; font-weight: 700; line-height: 1.15;
                   letter-spacing: -0.02em; margin: 0 0 0.7rem 0;">
            📚 AI-Powered Research Paper Intelligence Assistant
        </h1>
        <p style="font-size: 1.15rem; opacity: 0.85; margin: 0 0 1.5rem 0; line-height: 1.55;">
            Explore, understand, and discover insights across your research papers.
        </p>
        <div class="chip-row">
            <span class="chip">📄 <b>{len(available_papers)}</b> Research Papers</span>
            <span class="chip">🧩 <b>{len(chunks):,}</b> Indexed Passages</span>
            <span class="chip">🔍 Semantic Search</span>
            <span class="chip">🤖 AI Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📚 AI-Powered Research Paper Intelligence Assistant")
        st.caption("An AI-powered workspace for searching, understanding, and exploring research papers.")
        st.divider()

        st.markdown("#### Corpus")
        c1, c2 = st.columns(2)
        c1.metric("Papers", len(available_papers))
        c2.metric("Passages", f"{len(chunks):,}")

        st.divider()
        st.markdown("#### Capabilities")
        st.markdown(
            "- ✓ Semantic Search\n"
            "- ✓ AI Question Answering\n"
            "- ✓ Paper Summarization\n"
            "- ✓ Evidence Retrieval"
        )
        st.divider()
        st.caption("Streamlit · Sentence Transformers · FAISS · Gemini")


def render_source_card(i, source, score, title, text):
    cls = score_class(score)
    label = {"high": "High relevance", "mid": "Medium relevance", "low": "Low relevance"}[cls]
    excerpt = text[:350] + ("..." if len(text) > 350 else "")
    st.markdown(f"""
    <div class="source-card">
        <div class="source-head">
            <span class="source-title">[{i}] {title}</span>
            <span class="source-badge {cls}">{label} · {score:.2f}</span>
        </div>
        <div class="source-excerpt">{excerpt}</div>
        <div class="source-file">Source: {source}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("View full passage"):
        st.write(text)


def render_empty_state(title, description):
    st.markdown(f"""
    <div class="empty-state">
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def render_paper_card(pdf_name):
    title = paper_titles.get(pdf_name, pdf_name)
    count = sum(1 for c in chunks if c["source"] == pdf_name)
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📄 {title}</div>
        <div class="card-meta">{pdf_name} · <b>{count}</b> indexed passages</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN UI
# ============================================================

render_header()
render_sidebar()

tab_ask, tab_summarize, tab_library = st.tabs([
    "🔍 Ask Research",
    "📝 Paper Summaries",
    "📚 Research Library",
])

# ---------------- ASK ----------------
with tab_ask:
    st.markdown('<div class="section-label">Ask your research question</div>', unsafe_allow_html=True)
    st.caption("Search across your indexed research papers and get an AI-generated answer with supporting evidence.")

    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        query = st.text_input(
            "Your question",
            placeholder="Ask a question about your research papers...",
            label_visibility="collapsed",
        )

        st.markdown('<div class="section-label">Suggested Questions</div>', unsafe_allow_html=True)
        examples = [
            "What is retrieval augmented generation?",
            "How does MusT-RAG handle music question answering?",
            "What is the main limitation of standard RAG?",
            "Compare the retrieval approaches discussed in these papers.",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                query = ex

        ask_clicked = st.button("Ask Research Assistant", type="primary", use_container_width=True)

    with col_right:
        if ask_clicked and query:
            with st.spinner("Searching research papers and generating answer..."):
                try:
                    answer, results = ask(query)

                    st.markdown('<div class="section-label">Research Answer</div>', unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown(answer)
                        st.caption("Generated from indexed research papers.")

                    st.markdown('<div class="section-label">Research Evidence</div>', unsafe_allow_html=True)
                    st.caption(f"**{len(results)}** relevant passages found · ranked by semantic similarity")
                    for i, r in enumerate(results, 1):
                        render_source_card(
                            i, r["source"], r["score"],
                            paper_titles.get(r["source"], r["source"]),
                            r["text"],
                        )
                except Exception as e:
                    st.error("### Something went wrong\nUnable to complete the request. Please try again.")
                    with st.expander("Technical details"):
                        st.code(str(e))
        else:
            render_empty_state(
                "Start exploring your research",
                "Ask a question to search across your indexed papers and generate an AI-powered answer with supporting evidence.",
            )
            st.markdown('<div class="section-label">Try one of these</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, ex in enumerate([
                "What is retrieval augmented generation?",
                "How does MusT-RAG handle music question answering?",
                "What is the main limitation of standard RAG?",
                "Compare the retrieval approaches discussed in these papers.",
            ]):
                with cols[i % 2]:
                    st.button(ex, key=f"empty_{ex}", use_container_width=True)

# ---------------- SUMMARIZE ----------------
with tab_summarize:
    st.markdown('<div class="section-label">Understand a Research Paper</div>', unsafe_allow_html=True)
    st.caption("Select a research paper and generate a concise AI-powered summary.")

    selected = st.selectbox(
        "Choose a paper",
        options=available_papers,
        format_func=lambda p: paper_titles.get(p, p),
        label_visibility="collapsed",
    )

    if selected:
        title = paper_titles.get(selected, selected)
        count = sum(1 for c in chunks if c["source"] == selected)
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📄 {title}</div>
            <div class="card-meta">{selected} · <b>{count}</b> indexed passages</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Analyzing paper and generating summary..."):
            try:
                summary = summarize_paper(selected)
                st.markdown('<div class="section-label">Structured Summary</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(summary)
            except Exception as e:
                st.error("### Something went wrong\nUnable to generate the summary. Please try again.")
                with st.expander("Technical details"):
                    st.code(str(e))

# ---------------- LIBRARY ----------------
with tab_library:
    st.markdown('<div class="section-label">Research Library</div>', unsafe_allow_html=True)
    st.caption("Browse the research papers currently available in the knowledge base.")
    st.caption(f"**{len(available_papers)}** papers indexed · **{len(chunks):,}** total passages")

    for pdf_name in available_papers:
        render_paper_card(pdf_name)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <b>AI-Powered Research Paper Intelligence Assistant</b><br>
    Semantic Search • AI Analysis • Research Discovery
</div>
""", unsafe_allow_html=True)
