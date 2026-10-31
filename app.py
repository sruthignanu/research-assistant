# ============================================================
# AI-Powered Research Paper Intelligence Assistant
# Chat assistant + PDF upload + summaries + paper suggestions (arXiv, Semantic Scholar) + SQLite
# (Streamlit · SBERT · FAISS · BM25 · Gemini · SQLite)
# Run from anywhere:  streamlit run app.py
# ============================================================
import html
import json
import os
import pickle
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
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

HAS_ARXIV = True           # arXiv is searched through its public web API (no extra package needed)

BASE = Path(__file__).parent
DATA = BASE / "data"
PAPERS_DIR = DATA / "papers"
CHUNKS_F = DATA / "all_chunks_v2.pkl"
INDEX_F = DATA / "all_index_v2.faiss"
TITLES_F = DATA / "titles.json"            # old title file, only read for migration
DB_F = DATA / "assistant.db"               # SQLite: paper details, saved summaries, question history
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

DEFAULT_MODEL = "gemini-3.8-flash"   # same model your notebooks use
TARGET, MAX_CHARS, OVERLAP_SENT = 800, 1000, 1   # same chunking as 02b_rechunk.ipynb
POOL, RRF_K, W_SPARSE = 30, 60, 2.0               # hybrid search (weight 2 tested in 05_evaluation_v2)
STOP = set("what is the role of in a an and or to how does do are were was for with on by from "
           "this that these those it its as at be used use".split())
ARXIV_MIN_SIM = 0.40                 # arXiv suggestions must be at least this similar to the question
SHOW_RELATED_MIN = 0.45              # show 'related papers in your library' only above this score
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
    client = (genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))  # 60 s, in ms
              if key else None)
    custom_titles = json.loads(TITLES_F.read_text()) if TITLES_F.exists() else {}
    return chunks, index, model, client, custom_titles


chunks, index, model, client, custom_titles = load_resources()


def save_corpus():
    """Persist chunks and the FAISS index so uploads survive a restart (details are in SQLite)."""
    with open(CHUNKS_F, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, str(INDEX_F))


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


def arxiv_url(source):
    m = re.match(r"^(\d{4}\.\d{4,5}(?:v\d+)?)\.pdf$", source)
    if m:
        return f"https://arxiv.org/abs/{m.group(1)}"
    m = re.match(r"^([a-z\-]+)_(\d{7}(?:v\d+)?)\.pdf$", source)       # old style, e.g. cs_0107006v1.pdf
    return f"https://arxiv.org/abs/{m.group(1)}/{m.group(2)}" if m else None


# ---------------- SQLite (paper details, saved summaries, question history) ----------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def db_conn():
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_F)
    con.row_factory = sqlite3.Row
    return con


def db_exec(sql, params=()):
    con = db_conn()
    try:
        with con:
            con.execute(sql, params)
    finally:
        con.close()


def db_query(sql, params=()):
    con = db_conn()
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def db_init():
    con = db_conn()
    try:
        with con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    source TEXT PRIMARY KEY, title TEXT, authors TEXT, year INTEGER,
                    url TEXT, origin TEXT, added_at TEXT);
                CREATE TABLE IF NOT EXISTS summaries (
                    source TEXT PRIMARY KEY, summary TEXT, model TEXT, created_at TEXT);
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT,
                    sources TEXT, created_at TEXT);
            """)
    finally:
        con.close()


def db_upsert_paper(source, title, authors="", year=None, url=None, origin="library"):
    db_exec("""INSERT INTO papers(source, title, authors, year, url, origin, added_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(source) DO UPDATE SET
                 title = excluded.title,
                 authors = CASE WHEN excluded.authors <> '' THEN excluded.authors ELSE papers.authors END,
                 year = COALESCE(excluded.year, papers.year),
                 url = COALESCE(excluded.url, papers.url)""",
            (source, title, authors or "", year, url, origin, now()))


def db_save_summary(source, text, model_name):
    db_exec("INSERT OR REPLACE INTO summaries(source, summary, model, created_at) VALUES(?,?,?,?)",
            (source, text, model_name, now()))


def db_get_summary(source):
    rows = db_query("SELECT * FROM summaries WHERE source = ?", (source,))
    return rows[0] if rows else None


def db_log(question, answer, results):
    srcs = ", ".join(sorted({r["source"] for r in results}))
    db_exec("INSERT INTO history(question, answer, sources, created_at) VALUES(?,?,?,?)",
            (question, answer, srcs, now()))


db_init()


def build_titles():
    """Titles come from SQLite; missing ones are read from the PDF (or the old titles.json) once."""
    meta = {r["source"]: r for r in db_query("SELECT * FROM papers")}
    tried = st.session_state.setdefault("tried_titles", set())
    bad = lambda t, s_: (not t) or t == s_ or t.lower().startswith("arxiv")
    titles_, seen = {}, set()
    for c in chunks:
        s_ = c["source"]
        if s_ in seen:
            continue
        seen.add(s_)
        t = (meta.get(s_) or {}).get("title")
        if bad(t, s_) and s_ not in tried:
            tried.add(s_)
            t = custom_titles.get(s_)
            if bad(t, s_):
                t = None
                pdf = PAPERS_DIR / s_
                if pdf.exists():
                    try:
                        t = pdf_title(pymupdf.open(str(pdf)), s_)
                    except Exception:
                        t = None
                t = t or guess_title(c["text"], s_)
            db_upsert_paper(s_, t, url=arxiv_url(s_), origin=(meta.get(s_) or {}).get("origin") or "library")
        titles_[s_] = t or s_
    return titles_


titles = build_titles()
paper_meta = {r["source"]: r for r in db_query("SELECT * FROM papers")}
sources = sorted(titles)


# ---------------- Gemini ----------------
def generate(prompt, model_name):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY not found. Put it in a `.env` file next to app.py.")
    backup = (st.session_state.get("backup_model") or "").strip()
    models = [model_name] + ([backup] if backup and backup != model_name else [])
    slow = ("504", "deadline", "timed out", "timeout")                  # the model is too slow: one retry, then the backup
    busy = ("503", "unavailable", "overloaded", "disconnected", "eof", "ssl", "connection", "reset",
            "temporary failure", "remote end closed", "remoteprotocol", "readerror", "connecterror")
    waits = (4, 8)                                                     # seconds before the 2nd and 3rd attempt
    last = ""
    cfg = None
    if st.session_state.get("fast_mode"):     # ask for less "thinking" so the answer comes back sooner
        try:
            cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"))
        except Exception:
            cfg = None                        # this SDK version does not know the option
    for name in models:                       # main model first, then the optional backup model
        attempts = 0
        while attempts < 3:
            try:
                return (client.models.generate_content(model=name, contents=prompt, config=cfg).text or "").strip()
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                low = last.lower()
                if cfg is not None and ("invalid_argument" in low or "thinking" in low):
                    cfg = None                # the model rejected the option: retry without it
                    continue
                if "429" in low or "resource_exhausted" in low:
                    break                     # the quota is per model, so the backup model may still work
                attempts += 1
                is_slow = any(t in low for t in slow)
                if (is_slow or any(t in low for t in busy)) and attempts < (2 if is_slow else 3):
                    time.sleep(waits[attempts - 1])
                    continue
                break
    if "429" in last or "RESOURCE_EXHAUSTED" in last:
        raise RuntimeError("Gemini quota exhausted (the free tier allows only ~20 requests/day per model). "
                           "Set a backup model in the sidebar, wait for the reset, or enable billing.")
    raise RuntimeError(f"Gemini request failed: {last[:300]}")


def list_gemini_models():
    """Gemini models that this API key can use for text generation."""
    if client is None:
        raise RuntimeError("GEMINI_API_KEY not found.")
    names = []
    for m in client.models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []) and "gemini" in m.name.lower():
            names.append(m.name.replace("models/", ""))
    return sorted(names)


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


def answer_question(q, history, k, only, model_name, on_step=lambda m: None, hybrid=True, web=None):
    """Library passages + (optional) online abstracts -> combined evidence -> Gemini -> cited answer.
    Returns (answer, library_results). Online items are numbered after the library items."""
    prev_users = [m["content"] for m in history if m["role"] == "user"]
    # short follow-ups ("what about its dataset?") borrow the previous question for retrieval
    retrieval_q = q if (len(q.split()) >= 8 or not prev_users) else f"{prev_users[-1]} {q}"
    diverse = any(w in q.lower() for w in COMPARE_WORDS)
    on_step("Searching your library...")
    results = search(retrieval_q, k=max(k, 8) if diverse else k, only=only, diverse=diverse, hybrid=hybrid)
    if not results or max(r["score"] for r in results) < MIN_SCORE:
        results = []                                    # nothing relevant in the library
    web = web or []
    if not results and not web:
        return ("I couldn't find anything relevant to that in your papers. Try rephrasing, "
                "turn on the online search in the sidebar, or upload a paper that covers it."), []
    on_step("Combining the evidence and asking Gemini...")
    parts = [f"[{i+1}] (Library: {titles.get(r['source'], r['source'])})\n{r['text']}" for i, r in enumerate(results)]
    for j, p in enumerate(web, len(results) + 1):
        parts.append(f"[{j}] ({p['source']} abstract, paper is not in the library) {p['title']} ({p['year']})\n{p['abstract']}")
    context = "\n\n".join(parts)
    convo = "\n".join(f"{m['role'].title()}: {m['content'][:600]}" for m in history[-6:]) or "(none)"
    prompt = f"""You are a research assistant helping a user understand academic papers.
Answer using ONLY the context below. The context has up to three kinds of items:
- "Library": passages from the user's own papers. Prefer these, they are the most reliable.
- "arXiv" or "Semantic Scholar": abstracts only (not the full text) of papers that are not in the library.
  When you rely on one, say that it comes from the abstract of an online paper and name the paper.
If the context doesn't contain the answer, say "I don't have enough information in the papers to answer that."
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
        return f"⚠️ {e}\n\nThe retrieved evidence is shown below.", results


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


def ingest_pdf(name, data, meta=None):
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
    meta = meta or {}
    db_upsert_paper(name, meta.get("title") or pdf_title(doc, name), meta.get("authors", ""), meta.get("year"),
                    meta.get("url") or arxiv_url(name), meta.get("origin", "upload"))
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
    with st.expander(f"📎 Library evidence ({len(results)} passages)"):
        for i, r in enumerate(results, 1):
            source_card(i, r)


def paper_card(src):
    count = sum(1 for c in chunks if c["source"] == src)
    m = paper_meta.get(src, {})
    bits = []
    if m.get("authors"):
        bits.append(html.escape(m["authors"]))
    if m.get("year"):
        bits.append(str(m["year"]))
    bits += [html.escape(src), f"<b>{count}</b> indexed passages", f"added via {html.escape(m.get('origin') or 'library')}"]
    url = m.get("url") or ""
    link = f' · <a href="{html.escape(url)}" target="_blank">open</a>' if url.startswith("http") else ""
    st.markdown(
        f'<div class="card"><div class="card-title">📄 {html.escape(titles[src])}</div>'
        f'<div class="card-meta">{" · ".join(bits)}{link}</div></div>',
        unsafe_allow_html=True)


QSTOP = STOP | set("explain tell me about why when which who whom whose give compare comparison difference "
                   "differences between versus vs paper papers research discussed these those there their can could "
                   "would should please describe define overview main key you your using work works help understand handle handles improve improves differ differs discussed used uses "
                   "approaches approach exist exists methods method techniques technique ways way type types kind kinds "
                   "applications application examples example".split())


def md_escape(t):
    return re.sub(r"([\\`*_\[\]$])", r"\\\1", t)


def suggest_from_results(results, n=3):
    """Rank the library papers behind the retrieved passages."""
    best = {}
    for r in results:
        b = best.setdefault(r["source"], {"score": 0.0, "n": 0})
        b["score"] = max(b["score"], r["score"])
        b["n"] += 1
    ranked = sorted(best.items(), key=lambda kv: (kv[1]["n"], kv[1]["score"]), reverse=True)[:n]
    return [{"source": s_, "title": titles.get(s_, s_), "n": v["n"], "url": arxiv_url(s_)} for s_, v in ranked]


def topic_query(q):
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", q) if w.lower() not in QSTOP and len(w) > 2]
    return " ".join(words[:8])


@st.cache_resource
def _arxiv_clock():
    return {"last": 0.0}


@st.cache_data(ttl=3600, show_spinner=False)
def arxiv_search(query, n=8):
    """arXiv search through the public API: 15 s timeout, and at least 3 s between requests (arXiv's rule)."""
    clock = _arxiv_clock()
    wait = 3.0 - (time.time() - clock["last"])
    if wait > 0:
        time.sleep(wait)
    params = urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": n, "sortBy": "relevance"})
    req = urllib.request.Request(f"{ARXIV_API}?{params}", headers={"User-Agent": "research-assistant/1.0"})
    for attempt in range(2):
        try:
            clock["last"] = time.time()
            with urllib.request.urlopen(req, timeout=20) as resp:
                root = ET.fromstring(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt == 0:
                time.sleep(3)
                continue
            raise
        except (TimeoutError, urllib.error.URLError):          # slow or briefly unreachable: try once more
            if attempt == 0:
                time.sleep(2)
                continue
            raise
    clean = lambda t: re.sub(r"\s+", " ", t or "").strip()
    out = []
    for e in root.findall(ATOM + "entry"):
        url = clean(e.findtext(ATOM + "id"))
        if "/api/errors" in url or "/abs/" not in url:
            continue
        sid = url.split("/abs/")[-1]
        names = [clean(a.findtext(ATOM + "name")) for a in e.findall(ATOM + "author")]
        abstract = clean(e.findtext(ATOM + "summary"))
        pdf = next((l.get("href") for l in e.findall(ATOM + "link") if l.get("title") == "pdf"),
                   f"https://arxiv.org/pdf/{sid}")
        year = int((e.findtext(ATOM + "published") or "0000")[:4] or 0) or None
        out.append({"id": sid, "title": clean(e.findtext(ATOM + "title")),
                    "authors": ", ".join(names[:3]) + (" et al." if len(names) > 3 else ""),
                    "year": year, "summary": abstract[:300], "abstract": abstract[:1000],
                    "url": url, "pdf": pdf, "source": "arXiv", "citations": None,
                    "file": f"{sid.replace('/', '_')}.pdf"})
    return out


def s2_key():
    """API key from .env with everything except letters, digits, _ and - removed (curly quotes, spaces...)."""
    return re.sub(r"[^A-Za-z0-9_\-]", "", os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "")


def s2_ping():
    """One tiny request that reports exactly why Semantic Scholar works or fails."""
    headers = {"User-Agent": "research-assistant/1.0"}
    if s2_key():
        headers["x-api-key"] = s2_key()
    url = f"{S2_API}?{urllib.parse.urlencode({'query': 'question answering', 'limit': 1, 'fields': 'title'})}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        return True, "Works. Example result: " + ((data.get("data") or [{}])[0].get("title") or "none")[:60]
    except urllib.error.HTTPError as e:
        hint = {400: "bad request", 403: "the key is not active yet or was copied wrongly",
                429: "rate limited, wait a minute and test again"}.get(e.code, "")
        return False, f"HTTP {e.code} {hint}".strip()
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


@st.cache_data(ttl=3600, show_spinner=False)
def s2_search(query, n=15):
    """Semantic Scholar paper search. Works without a key; SEMANTIC_SCHOLAR_API_KEY in .env raises the limits."""
    params = urllib.parse.urlencode({"query": query, "limit": n,
                                     "fields": "title,abstract,year,authors,citationCount,openAccessPdf,externalIds,url"})
    headers = {"User-Agent": "research-assistant/1.0"}
    key = s2_key()
    if key:
        headers["x-api-key"] = key
    req = urllib.request.Request(f"{S2_API}?{params}", headers=headers)
    for attempt in range(3):                   # the free, keyless API is shared and often says "429 too many requests"
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    out = []
    for p in data.get("data") or []:
        title = re.sub(r"\s+", " ", p.get("title") or "").strip()
        abstract = re.sub(r"\s+", " ", p.get("abstract") or "").strip()
        if not title or not abstract or not p.get("paperId"):
            continue
        names = [a.get("name", "") for a in p.get("authors") or []]
        arx = (p.get("externalIds") or {}).get("ArXiv")
        pdf = (p.get("openAccessPdf") or {}).get("url") or (f"https://arxiv.org/pdf/{arx}" if arx else None)
        out.append({"id": arx or p["paperId"], "title": title,
                    "authors": ", ".join(names[:3]) + (" et al." if len(names) > 3 else ""),
                    "year": p.get("year"), "summary": abstract[:300], "abstract": abstract[:1000],
                    "url": p.get("url") or f"https://www.semanticscholar.org/paper/{p['paperId']}",
                    "pdf": pdf, "source": "Semantic Scholar", "citations": p.get("citationCount"),
                    "file": f"{arx.replace('/', '_')}.pdf" if arx else f"s2_{p['paperId'][:12]}.pdf"})
    return out


def on_topic(p, terms):
    """True if the paper's title+abstract contain the question's main terms (word stems, hyphens ignored)."""
    text = (p["title"] + " " + p["abstract"]).lower().replace("-", " ")

    def hit(t):
        parts = [x for x in t.lower().split("-") if len(x) >= 3] or [t.lower()]
        return all(x[:max(4, len(x) - 2)] in text for x in parts)

    need = len(terms) if len(terms) <= 3 else -(-3 * len(terms) // 4)       # all of up to 3 terms, else 75%
    return sum(hit(t) for t in terms) >= need


def find_new_papers(q, n=4, min_sim=ARXIV_MIN_SIM, use_arxiv=True, use_s2=True):
    """New papers from arXiv and Semantic Scholar that are not in the library, ranked by similarity to q.
    Returns (papers, notes) where notes explain a source that could not be reached."""
    notes = []
    tq = topic_query(q)
    if len(tq) < 3:
        return [], notes
    found = []
    terms = tq.split()
    plain = [t for t in terms if not re.search(r"[a-z][A-Z]", t)] or terms   # drop names such as MusT-RAG
    ptq = " ".join(plain)
    if use_arxiv and HAS_ARXIV:
        queries = []
        if len(plain) >= 2:
            queries.append(" AND ".join(f"all:{t}" for t in plain[:4]))    # the main terms must all appear
        queries.append(ptq)                                                # loose search if that finds too little
        try:
            for aq in queries:
                res = arxiv_search(aq, 15)
                if len(res) >= 3:
                    break
            found += res
        except Exception as e:
            notes.append(f"arXiv could not be reached ({type(e).__name__}: {str(e)[:90]}).")
    if use_s2:
        try:
            found += s2_search(ptq, 15)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                notes.append("Semantic Scholar rate limit reached even with your key (about 1 request per second), "
                             "try again in a few seconds." if s2_key() else
                             "Semantic Scholar rate limit reached, try again in a minute "
                             "(a free API key in .env raises the limit).")
            else:
                notes.append(f"Semantic Scholar returned error {e.code}.")
        except Exception as e:
            notes.append(f"Semantic Scholar could not be reached ({type(e).__name__}: {str(e)[:90]}).")
    base = lambda i: re.sub(r"v\d+$", "", i.replace("/", "_"))
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    have = {base(s_[:-4]) for s_ in sources if s_.endswith(".pdf")} | {norm(t) for t in titles.values()}
    seen, uniq = set(), []
    for p in found:                       # drop papers already in the library and duplicates across sources
        keys = {base(p["file"][:-4]), norm(p["title"])}
        if keys & have or keys & seen:
            continue
        seen |= keys
        uniq.append(p)
    found_any = bool(uniq)
    uniq = [p for p in uniq if on_topic(p, plain)]       # drop papers that only share one or two words
    if not uniq:
        if not notes:
            notes.append(f"No closely related new papers were found for “{ptq}”." if found_any
                         else "No new papers were found online for this topic.")
        return [], notes
    emb = model.encode([q] + [f"{p['title']}. {p['abstract']}" for p in uniq], normalize_embeddings=True)
    sims = emb[1:] @ emb[0]
    ranked = sorted(zip(sims, uniq), key=lambda x: -x[0])
    cut = min_sim + (0.15 if len(plain) <= 1 else 0.05 if len(plain) == 2 else 0.0)   # short questions match loosely: be stricter
    keep = [dict(p, sim=float(sim)) for sim, p in ranked if sim >= cut][:n]
    if not keep:
        s0, p0 = ranked[0]
        notes.append(f"No new papers reached the relevance cutoff ({cut:.2f}). The closest was "
                     f"“{p0['title'][:80]}” ({float(s0):.2f}). Lower the cutoff in the sidebar to see more.")
    return keep, notes


def is_refusal(a):
    a = a.lower()
    return "don't have enough information" in a or "do not have enough information" in a


def add_from_web(p):
    if not p.get("pdf"):
        st.warning("No free PDF is available for this paper. Download it yourself and use the Upload tab.")
        return
    name = p["file"]
    try:
        with st.spinner("Downloading and indexing the paper..."):
            req = urllib.request.Request(p["pdf"], headers={"User-Agent": "research-assistant/1.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if not data.startswith(b"%PDF"):
                raise ValueError("the link did not return a PDF file")
            ok, msg = ingest_pdf(name, data, meta={"title": p["title"], "authors": p["authors"], "year": p["year"],
                                                  "url": p["url"], "origin": p["source"]})
    except Exception as e:
        st.error(f"Could not download the paper: {e}")
        return
    st.session_state["flash"] = ("✅ " if ok else "⚠️ ") + msg
    st.rerun()


def render_suggestions(m, idx):
    if m.get("papers"):
        st.markdown("**📚 Related papers in your library**")
        for p in m["papers"]:
            link = f" · [open]({p['url']})" if p["url"] else ""
            plural = "s" if p["n"] != 1 else ""
            st.markdown(f"- **{md_escape(p['title'])}** · {p['n']} matching passage{plural}{link}")
    if m.get("new"):
        offset = m.get("web_offset")
        st.markdown("**🔎 Online evidence used in this answer (abstracts from arXiv and Semantic Scholar)**"
                    if offset is not None else
                    "**🔎 Related papers from arXiv and Semantic Scholar (not in your library yet)**")
        for i, p in enumerate(m["new"]):
            with st.container(border=True):
                cites = f" · {p['citations']} citations" if p.get("citations") is not None else ""
                tag = f"[{offset + i + 1}] " if offset is not None else ""
                st.markdown(f"**{tag}{md_escape(p['title'])}**  \n{md_escape(p['authors'])} · {p['year']} · "
                            f"{p['source']}{cites} · [open]({p['url']}) · relevance {p.get('sim', 0):.2f}")
                st.caption(p["summary"] + "...")
                if p.get("pdf"):
                    if st.button("➕ Add to library", key=f"add_{idx}_{p['file']}"):
                        add_from_web(p)
                else:
                    st.caption("No free PDF available. Download it yourself and use the Upload tab.")
    for note in m.get("notes") or []:
        st.caption("ℹ️ " + note)


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
        st.text_input("Backup Gemini model (optional)", key="backup_model",
                      help="Used automatically when the main model is busy (503) or out of quota (429).")
        st.toggle("Fast mode (less thinking, experimental)", key="fast_mode",
                  help="Asks Gemini to think less so answers arrive sooner. Ignored if your model does not support it.")
        gc1, gc2 = st.columns(2)
        if gc1.button("Test Gemini", use_container_width=True):
            t0 = time.time()
            try:
                out = generate("Reply with the single word OK.", model_name)[:40]
                st.success(f"Works: {out} ({time.time() - t0:.1f} s)")
            except Exception as e:
                st.error(f"After {time.time() - t0:.0f} s: {str(e)[:200]}")
        if gc2.button("List models", use_container_width=True):
            try:
                st.code("\n".join(list_gemini_models()) or "no models found")
            except Exception as e:
                st.error(str(e)[:200])
        k = st.slider("Passages to retrieve", 3, 12, 6)
        scope = st.multiselect("Limit to papers (empty = all)", sources,
                               format_func=lambda p: titles.get(p, p), key="scope")
        suggest_new = st.toggle("Search arXiv and Semantic Scholar", value=True,
                                help="After each question, search both sites for related papers.")
        web_opts = (["arXiv"] if HAS_ARXIV else []) + ["Semantic Scholar"]
        web_sources = st.multiselect("Search in", web_opts, default=web_opts, disabled=not suggest_new)
        st.caption("Semantic Scholar key: " + (f"found ({len(s2_key())} characters)" if s2_key() else "not found in .env"))
        if st.button("Test Semantic Scholar", use_container_width=True):
            ok, info = s2_ping()
            (st.success if ok else st.error)(info)
        arxiv_cut = st.slider("Relevance cutoff for suggestions", 0.25, 0.65, ARXIV_MIN_SIM, 0.01,
                              disabled=not suggest_new,
                              help="Higher = fewer but more relevant suggestions. Lower = more suggestions.")
        use_web = st.toggle("Use online abstracts as evidence for the answer", value=True,
                            disabled=not suggest_new,
                            help="On: the abstracts of the relevant online papers are sent to Gemini together with your "
                                 "library passages. Off: Gemini sees only your library and the online papers are suggestions.")
        hybrid = st.toggle("Hybrid keyword search", value=HAS_BM25, disabled=not HAS_BM25,
                           help="Embeddings + BM25. Helps with exact terms such as tool or dataset names.")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("Streamlit · Sentence Transformers · FAISS · SQLite · Gemini")
    return model_name, k, scope, hybrid, suggest_new, arxiv_cut, web_sources, use_web


# ---------------- State ----------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("summaries", {})
if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))

model_name, k, scope, hybrid, suggest_new, arxiv_cut, web_sources, use_web = render_sidebar()

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
    st.caption("Question → library + arXiv + Semantic Scholar → relevance filtering → combined evidence → Gemini → answer with citations")
    hist_box = st.container()
    sug_box = st.container()
    typed = st.chat_input("Ask anything about your papers...")
    q = st.session_state.pop("pending", None) or typed

    with hist_box:
        for idx, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("results"):
                    render_evidence(m["results"])
                render_suggestions(m, idx)
                if (m["role"] == "assistant" and m["content"].startswith("⚠️")
                        and idx == len(st.session_state.messages) - 1 and idx >= 1):
                    if st.button("🔁 Retry the answer", key=f"retry_{idx}"):
                        question = st.session_state.messages[idx - 1]["content"]
                        with st.spinner("Asking Gemini again..."):
                            web = m.get("new") if m.get("web_offset") is not None else None
                            ans, res = answer_question(question, st.session_state.messages[:idx - 1], k,
                                                       scope or None, model_name, hybrid=hybrid, web=web)
                        m["content"], m["results"] = ans, res
                        m["web_offset"] = len(res) if web else None
                        if not ans.startswith("⚠️"):
                            show = res and not is_refusal(ans) and max(r["score"] for r in res) >= SHOW_RELATED_MIN
                            m["papers"] = suggest_from_results(res) if show else []
                            db_log(question, ans, res)
                        st.rerun()

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
                    failed = False
                    new, notes = [], []
                    if suggest_new and not scope:
                        status.update(label="Searching arXiv and Semantic Scholar...")
                        new, notes = find_new_papers(q, min_sim=arxiv_cut, use_arxiv="arXiv" in web_sources,
                                                     use_s2="Semantic Scholar" in web_sources)
                        status.write(f"✓ Online search: {len(new)} relevant paper(s) kept")
                    elif suggest_new and scope:
                        notes.append("Online search is skipped while \"Limit to papers\" is set in the sidebar.")
                    try:
                        answer, results = answer_question(q, history, k, scope or None, model_name,
                                                          on_step=lambda m: status.update(label=m), hybrid=hybrid,
                                                          web=new if use_web else None)
                    except Exception as e:
                        answer, results, failed = f"⚠️ {e}", [], True
                    failed = failed or answer.startswith("⚠️")          # Gemini errors come back as a warning text
                    status.write(f"✓ Library: {len(results)} passage(s)")
                    show_related = (results and not failed and not is_refusal(answer)
                                    and max(r["score"] for r in results) >= SHOW_RELATED_MIN)
                    papers = suggest_from_results(results) if show_related else []
                    if failed:
                        status.write("⚠️ Gemini request failed, use the Retry button below")
                    else:
                        db_log(q, answer, results)
                    status.update(label="Gemini failed" if failed else "Done",
                                  state="error" if failed else "complete", expanded=failed)
                msg = {"role": "assistant", "content": answer, "results": results, "papers": papers, "new": new,
                       "notes": notes, "web_offset": len(results) if (use_web and new) else None}
                st.markdown(answer)
                if results:
                    render_evidence(results)
                render_suggestions(msg, len(st.session_state.messages) + 1)
        st.session_state.messages += [{"role": "user", "content": q}, msg]
        if msg["content"].startswith("⚠️"):
            st.rerun()                    # redraw from the saved history so the Retry button shows immediately

# ---------------- Summaries tab ----------------
with tab_sum:
    st.caption("Pick any paper (including ones you uploaded) and generate a structured summary.")
    selected = st.selectbox("Paper", sources, format_func=lambda p: titles.get(p, p))
    if selected:
        paper_card(selected)
        saved = db_get_summary(selected)
        if st.button("Regenerate summary" if saved else "Generate summary", type="primary"):
            with st.spinner("Reading the paper..."):
                try:
                    text = summarize_paper(selected, model_name)
                    if not text.startswith("No text found"):
                        db_save_summary(selected, text, model_name)
                    saved = {"summary": text, "model": model_name, "created_at": now()}
                except Exception as e:
                    st.error(str(e))
        if saved:
            st.caption(f"Saved {saved['created_at']} · {saved['model']} · stored in the local database, "
                       "so showing it again costs no API request")
            with st.container(border=True):
                st.markdown(saved["summary"])
            st.download_button("Download summary", saved["summary"], file_name=f"{Path(selected).stem}_summary.md")

# ---------------- Upload tab ----------------
with tab_upload:
    st.caption("Upload PDFs. They are parsed, chunked, embedded and added to the search index immediately.")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if files and st.button("Add to library", type="primary"):
        msgs, added = [], []
        with st.spinner("Processing PDFs (embedding can take a minute)..."):
            for f in files:
                ok, msg = ingest_pdf(f.name, f.getvalue())
                if ok:
                    added.append(os.path.basename(f.name))
                msgs.append(("✅ " if ok else "⚠️ ") + msg)
        any_ok = bool(added)
        if added:
            msgs.append("The paper is now part of the library. Use \"Limit to papers\" in the sidebar if you want answers from it only.")
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
    st.markdown("#### 🕘 Recent questions")
    hist = db_query("SELECT * FROM history ORDER BY id DESC LIMIT 20")
    if not hist:
        st.caption("Questions you ask are saved here.")
    for h in hist:
        with st.expander(f"{h['created_at']} · {h['question'][:90]}"):
            st.markdown(h["answer"])
            if h["sources"]:
                st.caption("Papers used: " + h["sources"])
    if hist and st.button("Clear question history"):
        db_exec("DELETE FROM history")
        st.rerun()

st.markdown('<div class="footer"><b>AI-Powered Research Paper Intelligence Assistant</b><br>'
            'Semantic Search • AI Analysis • Research Discovery</div>', unsafe_allow_html=True)
