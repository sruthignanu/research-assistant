# ============================================================
# AI-Powered Research Paper Intelligence Assistant
# Chat assistant + PDF upload + summaries  (Streamlit · SBERT · FAISS · Gemini)
# Run from anywhere:  streamlit run app.py
# ============================================================
import html
import json
import os
import pickle
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:        # app still works with embeddings only
    HAS_BM25 = False

BASE = Path(__file__).parent
DATA = BASE / "data"
PAPERS_DIR = DATA / "papers"
CHUNKS_F = DATA / "all_chunks_v2.pkl"
INDEX_F = DATA / "all_index_v2.faiss"
TITLES_F = DATA / "titles.json"

DEFAULT_MODEL = "gemini-3.8-flash"   # same model your notebooks use
TARGET, MAX_CHARS, OVERLAP_SENT = 800, 1000, 1   # same chunking as 02b_rechunk.ipynb
POOL, RRF_K, W_SPARSE = 30, 60, 2.0               # hybrid search (weight 2 tested in 05_evaluation_v2)
STOP = set("what is the role of in a an and or to how does do are were was for with on by from "
           "this that these those it its as at be used use".split())
MIN_SCORE = 0.30                     # below this, the question is not about the papers (saves API calls)

st.set_page_config(page_title="AI-Powered Research Paper Intelligence Assistant", page_icon="📚",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1100px; }
.chip-row { display:flex; gap:.6rem; flex-wrap:wrap; margin-bottom:.5rem; }
.chip { border:1px solid rgba(128,128,128,.25); border-radius:20px; padding:.35rem .9rem; font-size:.9rem; }
.card { border:1px solid rgba(128,128,128,.22); border-radius:10px; padding:1rem 1.2rem; margin-bottom:.8rem; }
.card-title { font-size:1.05rem; font-weight:600; margin-bottom:.3rem; }
.card-meta { font-size:.88rem; opacity:.75; }
.source-card { border:1px solid rgba(128,128,128,.22); border-left:3px solid #FF4B4B;
               border-radius:8px; padding:.8rem 1rem; margin-bottom:.6rem; }
.source-head { display:flex; justify-content:space-between; gap:.6rem; flex-wrap:wrap; margin-bottom:.4rem; }
.source-title { font-weight:600; }
.badge { font-size:.75rem; font-weight:600; padding:2px 10px; border-radius:999px; border:1px solid; white-space:nowrap; }
.badge.high { color:#16A34A; } .badge.mid { color:#D97706; } .badge.low { color:#DC2626; }
.source-excerpt { font-size:.92rem; line-height:1.55; }
.source-file { font-size:.78rem; opacity:.65; margin-top:.5rem; font-family:ui-monospace,Consolas,monospace; }
.footer { text-align:center; margin-top:3rem; padding-top:1rem; border-top:1px solid rgba(128,128,128,.2);
          font-size:.88rem; opacity:.7; }
</style>
""", unsafe_allow_html=True)


# ---------------- Resources ----------------
@st.cache_resource(show_spinner="Loading models and index...")
def load_resources():
    load_dotenv(BASE / ".env")
    with open(CHUNKS_F, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(INDEX_F))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    key = os.getenv("GEMINI_API_KEY")
    client = (genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120_000))  # 120 s, in ms
              if key else None)
    custom_titles = json.loads(TITLES_F.read_text()) if TITLES_F.exists() else {}
    return chunks, index, model, client, custom_titles


chunks, index, model, client, custom_titles = load_resources()


def save_corpus():
    """Persist chunks, FAISS index and titles so uploads survive a restart."""
    with open(CHUNKS_F, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, str(INDEX_F))
    TITLES_F.write_text(json.dumps(custom_titles, indent=2))


def pdf_title(doc, fallback):
    """Title from PDF metadata, else the largest text on page 1."""
    meta = ((doc.metadata or {}).get("title") or "").strip()
    bad = ("untitled", "microsoft word", ".pdf", ".tex", ".dvi", "noname")
    if len(meta) > 8 and not any(b in meta.lower() for b in bad):
        return meta[:120]
    try:
        spans = []
        for b in doc[0].get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                if abs(ln.get("dir", (1, 0))[0]) < 0.99:        # skip rotated margin text
                    continue
                for sp in ln["spans"]:
                    t = sp["text"].strip()
                    if len(t) > 3 and not t.lower().startswith("arxiv"):
                        spans.append((round(sp["size"], 1), sp["bbox"][1], t))
        top = max(sp[0] for sp in spans)
        title = " ".join(t for size, y, t in sorted(spans, key=lambda x: x[1]) if size >= top - 0.5)
        if len(title) > 8:
            return title[:120]
    except Exception:
        pass
    return fallback


def guess_title(text, fallback):
    for line in [l.strip() for l in text.split("\n") if l.strip()][:15]:
        if len(line) > 20 and not line.startswith("arXiv") and "[cs." not in line and "@" not in line:
            return line[:90]
    return fallback


def build_titles():
    titles, changed = {}, False
    for c in chunks:
        s = c["source"]
        if s in titles:
            continue
        t = custom_titles.get(s)
        if not t or t == s or t.lower().startswith("arxiv"):
            pdf = PAPERS_DIR / s
            if pdf.exists():
                try:
                    t = pdf_title(pymupdf.open(str(pdf)), s)
                except Exception:
                    t = None
            t = t or guess_title(c["text"], s)
            custom_titles[s] = t
            changed = True
        titles[s] = t
    if changed:
        TITLES_F.write_text(json.dumps(custom_titles, indent=2))
    return titles


titles = build_titles()
sources = sorted(titles)


# ---------------- Gemini ----------------
def generate(prompt, model_name):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY not found. Put it in a `.env` file next to app.py.")
    for attempt in range(3):
        try:
            return (client.models.generate_content(model=model_name, contents=prompt).text or "").strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                raise RuntimeError(
                    "Gemini quota exhausted (the free tier allows only ~20 requests/day per model). "
                    "Try a different model name in the sidebar, wait for the reset, or enable billing.")
            if attempt < 2 and any(s in msg for s in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED",
                                                       "disconnected", "timed out")):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"Gemini request failed: {msg[:300]}")


# ---------------- Retrieval + QA ----------------
COMPARE_WORDS = ("compare", "comparison", "versus", " vs ", "differ", "across", "all papers", "each paper")


TOKEN = re.compile(r"[a-z0-9\-]+")


def tok2(text):
    return [w for w in TOKEN.findall(text.lower()) if w not in STOP]


@st.cache_resource
def _bm25_holder():
    return {"n": -1, "bm25": None}


def get_bm25():
    """BM25 keyword index; rebuilt automatically after new papers are uploaded."""
    h = _bm25_holder()
    if h["n"] != len(chunks):
        h["bm25"] = BM25Okapi([tok2(c["text"]) for c in chunks])
        h["n"] = len(chunks)
    return h["bm25"]


def search(query, k=6, only=None, diverse=False, hybrid=True):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q, index.ntotal)        # flat index on ~700 vectors: cheap
    cos = {int(i): float(s) for i, s in zip(idxs[0], scores[0]) if i >= 0}
    allowed = lambda i: not only or chunks[i]["source"] in only
    dense = [int(i) for i in idxs[0] if i >= 0 and allowed(int(i))]
    order = dense
    if hybrid and HAS_BM25:                              # embeddings + keyword search (RRF fusion)
        sp = get_bm25().get_scores(tok2(query))
        sparse = [int(i) for i in np.argsort(sp)[::-1] if sp[i] > 0 and allowed(int(i))][:POOL]
        fused = {}
        for r, i in enumerate(dense[:POOL]):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + r + 1)
        for r, i in enumerate(sparse):
            fused[i] = fused.get(i, 0) + W_SPARSE / (RRF_K + r + 1)
        order = sorted(fused, key=fused.get, reverse=True)
    out, per_source = [], {}
    for i in order:
        c = chunks[i]
        if diverse and per_source.get(c["source"], 0) >= 2:   # spread across papers for comparisons
            continue
        per_source[c["source"]] = per_source.get(c["source"], 0) + 1
        out.append({"score": cos[i], "text": c["text"], "source": c["source"]})
        if len(out) == k:
            break
    return out


def answer_question(q, history, k, only, model_name, on_step=lambda m: None, hybrid=True):
    prev_users = [m["content"] for m in history if m["role"] == "user"]
    # short follow-ups ("what about its dataset?") borrow the previous question for retrieval
    retrieval_q = q if (len(q.split()) >= 8 or not prev_users) else f"{prev_users[-1]} {q}"
    diverse = any(w in q.lower() for w in COMPARE_WORDS)
    on_step("1/2 Searching passages...")
    results = search(retrieval_q, k=max(k, 8) if diverse else k, only=only, diverse=diverse, hybrid=hybrid)
    if not results or max(r["score"] for r in results) < MIN_SCORE:
        return ("I couldn't find anything relevant to that in your papers. Try rephrasing, "
                "or upload a paper that covers it."), []
    on_step("2/2 Waiting for Gemini...")
    context = "\n\n".join(f"[{i+1}] ({titles.get(r['source'], r['source'])})\n{r['text']}"
                          for i, r in enumerate(results))
    convo = "\n".join(f"{m['role'].title()}: {m['content'][:600]}" for m in history[-6:]) or "(none)"
    prompt = f"""You are a research assistant helping a user understand academic papers.
Answer using ONLY the context below. If the context doesn't contain the answer, say
"I don't have enough information in the papers to answer that."
Cite sources inline as [1], [2], etc. Mention the paper by topic when comparing papers.
Be clear and well organised; use short bullets when helpful.

Earlier conversation:
{convo}

Context:
{context}

Question: {q}

Answer:"""
    try:
        return generate(prompt, model_name), results
    except Exception as e:   # keep the evidence visible even if Gemini fails
        return f"⚠️ {e}\n\nThe retrieved passages are shown below.", results


def summarize_paper(pdf_name, model_name):
    pc = [c["text"] for c in chunks if c["source"] == pdf_name]
    if not pc:
        return f"No text found for {pdf_name}."
    n = len(pc)
    idx = (set(range(min(12, n))) | set(range(max(0, n // 2 - 6), min(n, n // 2 + 6)))
           | set(range(max(0, n - 10), n)))
    context = "\n\n".join(pc[i] for i in sorted(idx))
    prompt = f"""Summarize this research paper using exactly this structure. Use only the text below.

**Problem:** (1-2 sentences)
**Method:** (2-3 sentences)
**Dataset:** (1 sentence, or "Not specified")
**Results:** (1-2 sentences)
**Key takeaway:** (1 sentence)

Paper text:
{context}

Summary:"""
    return generate(prompt, model_name)


# ---------------- Upload ----------------
def cut_references(raw):
    m = None
    for m in re.finditer(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", raw):
        pass
    if m and m.start() > 0.5 * len(raw):
        return raw[:m.start()]
    return raw


def split_sentences(t):
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\[(“"])', t)
    out = []
    for s_ in parts:
        s_ = s_.strip()
        out += [s_[i:i + MAX_CHARS] for i in range(0, len(s_), MAX_CHARS)]
    return [s_ for s_ in out if s_]


def chunk_sentences(text, target=TARGET, overlap=OVERLAP_SENT):
    chunks_, cur, n = [], [], 0
    for s_ in split_sentences(text):
        if cur and n + len(s_) > target:
            chunks_.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            if cur and sum(len(x) for x in cur) + len(s_) > MAX_CHARS:
                cur = []
            n = sum(len(x) + 1 for x in cur)
        cur.append(s_)
        n += len(s_) + 1
    if cur:
        chunks_.append(" ".join(cur))
    return [c for c in chunks_ if len(c) >= 100]


def ingest_pdf(name, data):
    """Extract -> chunk -> embed -> add to FAISS -> persist. Returns (ok, message)."""
    name = os.path.basename(name)
    if any(c["source"] == name for c in chunks):
        return False, f"{name} is already in the library."
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        text = "".join(p.get_text() for p in doc)
    except Exception as e:
        return False, f"Could not read {name}: {e}"
    if len(text.strip()) < 200:
        return False, f"{name} has no extractable text (scanned PDF?). OCR it first."

    text = cut_references(text)
    text = re.sub(r"\s+", " ", re.sub(r"-\n(?=[a-z])", "", text))
    pieces = chunk_sentences(text)
    if not pieces:
        return False, f"{name}: no usable text after cleaning."
    emb = model.encode(pieces, normalize_embeddings=True, batch_size=32)
    index.add(np.array(emb, dtype="float32"))                      # index first,
    chunks.extend({"text": t, "source": name} for t in pieces)     # then chunks: keeps them aligned
    custom_titles[name] = pdf_title(doc, name)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    (PAPERS_DIR / name).write_bytes(data)
    save_corpus()
    return True, f"Added {name} ({len(pieces)} passages)."


# ---------------- UI helpers ----------------
def score_class(s):
    return "high" if s >= 0.6 else "mid" if s >= 0.45 else "low"


def source_card(i, r):
    cls = score_class(r["score"])
    label = {"high": "High", "mid": "Medium", "low": "Low"}[cls]
    excerpt = html.escape(re.sub(r"\s+", " ", r["text"]).strip()[:600])
    title = html.escape(titles.get(r["source"], r["source"]))
    # single-line HTML, escaped text: blank lines or '<' in PDF text can't break the layout
    st.markdown(
        f'<div class="source-card"><div class="source-head"><span class="source-title">[{i}] {title}</span>'
        f'<span class="badge {cls}">{label} relevance · {r["score"]:.2f}</span></div>'
        f'<div class="source-excerpt">{excerpt}</div>'
        f'<div class="source-file">{html.escape(r["source"])}</div></div>',
        unsafe_allow_html=True)


def render_evidence(results):
    with st.expander(f"📎 Evidence ({len(results)} passages)"):
        for i, r in enumerate(results, 1):
            source_card(i, r)


def paper_card(src):
    count = sum(1 for c in chunks if c["source"] == src)
    st.markdown(
        f'<div class="card"><div class="card-title">📄 {html.escape(titles[src])}</div>'
        f'<div class="card-meta">{html.escape(src)} · <b>{count}</b> indexed passages</div></div>',
        unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📚 AI-Powered Research Paper Intelligence Assistant")
        st.caption("Ask questions, upload papers, and get summaries.")
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Papers", len(sources))
        c2.metric("Passages", f"{len(chunks):,}")
        st.divider()
        st.markdown("#### Settings")
        model_name = st.text_input("Gemini model", DEFAULT_MODEL,
                                   help="Change this if you hit a daily quota on one model.")
        k = st.slider("Passages to retrieve", 3, 12, 6)
        scope = st.multiselect("Limit to papers (empty = all)", sources,
                               format_func=lambda p: titles.get(p, p))
        hybrid = st.toggle("Hybrid keyword search", value=HAS_BM25, disabled=not HAS_BM25,
                           help="Embeddings + BM25. Helps with exact terms such as tool or dataset names.")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("Streamlit · Sentence Transformers · FAISS · Gemini")
    return model_name, k, scope, hybrid


# ---------------- State ----------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("summaries", {})
if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))

model_name, k, scope, hybrid = render_sidebar()

st.title("📚 AI-Powered Research Paper Intelligence Assistant")
st.markdown(
    f'<div class="chip-row"><span class="chip">📄 <b>{len(sources)}</b> papers</span>'
    f'<span class="chip">🧩 <b>{len(chunks):,}</b> passages</span>'
    f'<span class="chip">💬 Chat</span><span class="chip">📤 Upload</span><span class="chip">📝 Summaries</span></div>',
    unsafe_allow_html=True)

tab_chat, tab_sum, tab_upload, tab_lib = st.tabs(
    ["💬 Ask", "📝 Summaries", "📤 Upload papers", "📚 Library"])

# ---------------- Chat tab ----------------
with tab_chat:
    hist_box = st.container()
    sug_box = st.container()
    typed = st.chat_input("Ask anything about your papers...")
    q = st.session_state.pop("pending", None) or typed

    with hist_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("results"):
                    render_evidence(m["results"])

    if not st.session_state.messages and not q:
        with sug_box:
            st.markdown("**Try asking:**")
            for ex in ["What is retrieval augmented generation?",
                       "How does MusT-RAG handle music question answering?",
                       "What is the main limitation of standard RAG?",
                       "Compare the retrieval approaches discussed in these papers."]:
                st.button(ex, key=f"ex_{ex}", use_container_width=True,
                          on_click=lambda t=ex: st.session_state.update(pending=t))

    if q:
        history = list(st.session_state.messages)
        with hist_box:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.status("Working...", expanded=True) as status:
                    try:
                        answer, results = answer_question(q, history, k, scope or None, model_name,
                                                          on_step=lambda m: status.update(label=m), hybrid=hybrid)
                        status.update(label="Done", state="complete", expanded=False)
                    except Exception as e:
                        answer, results = f"⚠️ {e}", []
                        status.update(label="Failed", state="error", expanded=False)
                st.markdown(answer)
                if results:
                    render_evidence(results)
        st.session_state.messages += [{"role": "user", "content": q},
                                      {"role": "assistant", "content": answer, "results": results}]

# ---------------- Summaries tab ----------------
with tab_sum:
    st.caption("Pick any paper (including ones you uploaded) and generate a structured summary.")
    selected = st.selectbox("Paper", sources, format_func=lambda p: titles.get(p, p))
    if selected:
        paper_card(selected)
        if st.button("Generate summary", type="primary"):
            with st.spinner("Reading the paper..."):
                try:
                    st.session_state.summaries[selected] = summarize_paper(selected, model_name)
                except Exception as e:
                    st.error(str(e))
        summary = st.session_state.summaries.get(selected)
        if summary:
            with st.container(border=True):
                st.markdown(summary)
            st.download_button("Download summary", summary, file_name=f"{Path(selected).stem}_summary.md")

# ---------------- Upload tab ----------------
with tab_upload:
    st.caption("Upload PDFs. They are parsed, chunked, embedded and added to the search index immediately.")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if files and st.button("Add to library", type="primary"):
        msgs, any_ok = [], False
        with st.spinner("Processing PDFs (embedding can take a minute)..."):
            for f in files:
                ok, msg = ingest_pdf(f.name, f.getvalue())
                any_ok |= ok
                msgs.append(("✅ " if ok else "⚠️ ") + msg)
        st.session_state["flash"] = "\n".join(msgs)
        if any_ok:
            st.rerun()
        else:
            for m in msgs:
                st.warning(m)

# ---------------- Library tab ----------------
with tab_lib:
    st.caption(f"**{len(sources)}** papers · **{len(chunks):,}** passages")
    for s in sources:
        paper_card(s)

st.markdown('<div class="footer"><b>AI-Powered Research Paper Intelligence Assistant</b><br>'
            'Semantic Search • AI Analysis • Research Discovery</div>', unsafe_allow_html=True)
