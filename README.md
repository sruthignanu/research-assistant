# AI-Powered Research Paper Intelligence Assistant

A Retrieval-Augmented Generation (RAG) assistant for research papers. Ask questions in a chat interface, upload your own PDFs, generate structured summaries, and discover related papers on arXiv and Semantic Scholar. Answers cite the passages of your own papers and, by default, the abstracts of relevant online papers; a sidebar switch limits them to your library only.

## Features

- **Chat Q&A with combined evidence**: each question is searched in your library, on arXiv and on Semantic Scholar. The online papers are filtered for relevance, and Gemini answers from the combined evidence with inline citations `[1]`, `[2]`. Library passages come first; online papers contribute their abstracts and are numbered after them.
- **Evidence display**: the library passages, the related papers in your library, and the online papers used (with source, citations and a relevance score) are shown under each answer. A sidebar switch limits Gemini to the library only.
- **Add papers with one click**: online papers with a free PDF can be added to the library directly from the evidence list
- **PDF upload**: new papers are parsed, chunked, embedded and added to the search index immediately
- **Paper summaries** (problem, method, dataset, results) for any paper, saved in a local SQLite database so they cost no API request next time
- **Hybrid search**: sentence embeddings plus BM25 keyword search (can be switched off in the sidebar)
- **Relevance filtering**: online papers must contain the question’s main terms and be semantically close to the question; very short questions use a stricter cutoff
- **Off-topic guard**: a question with no relevant library passage and no relevant online paper is rejected without calling the LLM, so it costs no API request
- **Limit to papers**: a sidebar filter restricts retrieval to chosen papers (online search is then skipped)
- **SQLite storage**: paper details, saved summaries, the history of questions and answers, and a daily count of Gemini requests
- **Resilience**: retries with waits for 503, 504, timeout and connection errors, an optional backup Gemini model, an adjustable number of attempts and a daily request counter to protect the free quota, and an optional Fast mode (less model thinking, experimental). Answers wait up to 60 seconds and summaries up to 120 seconds. A Retry button appears after a Gemini failure and after a failed online search, and the evidence still shows when Gemini is busy.
- **Diagnostics**: sidebar buttons test the Gemini connection (with timing), list the Gemini models the key can use, and test the Semantic Scholar key

## Architecture

```
Question → library (FAISS + BM25)  ┐
Question → arXiv                   ├→ relevance filtering → combined evidence → Gemini → answer with citations
Question → Semantic Scholar        ┘
```

1. Papers are downloaded from arXiv and parsed with PyMuPDF.
2. Text is cleaned (reference list removed, hyphenation fixed) and split into sentence-aware chunks of about 800 characters.
3. Chunks are embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and stored in a FAISS inner-product index (cosine similarity on normalized vectors). Chunk text and source are stored in a pickle file; paper details, summaries and history are stored in SQLite (`data/assistant.db`).
4. A question is embedded, and the top library passages are retrieved (dense search fused with BM25 by reciprocal rank fusion).
5. The question’s topic words are searched on arXiv and Semantic Scholar. Candidates already in the library are removed, the rest must contain the topic terms, are ranked by embedding similarity to the question, and kept above a relevance cutoff.
6. The library passages and the abstracts of the kept online papers are combined into one numbered evidence list and sent to Gemini, which writes a cited answer and says when it relies on an abstract.
7. The Streamlit app shows the answer and all the evidence.

## Tech stack

Python 3.12 · Streamlit · PyMuPDF · Sentence-Transformers (PyTorch) · FAISS (`faiss-cpu`) · rank-bm25 · SQLite (`sqlite3`) · Google Gemini API (`google-genai`, model `gemini-3.8-flash`) · arXiv API · Semantic Scholar API · python-dotenv · Pandas (the notebooks also use the `arxiv` package; the app calls the arXiv API directly)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

The Semantic Scholar key is free and optional. Without it the service is shared with all anonymous users and is often rate-limited; the arXiv suggestions and the library answers work either way. If something does not work, use the **Test Gemini**, **List models** and **Test Semantic Scholar** buttons in the sidebar; they show the real error.

## Build the corpus

The PDFs and index files are not stored in the repository. Run these notebooks in `notebooks/` once, in order:

1. `01_discovery.ipynb` - explore arXiv search
2. `02_corpus.ipynb` - download the papers and build the first index
3. `02b_rechunk.ipynb` - build the improved sentence-aware index (`data/all_chunks_v2.pkl`, `data/all_index_v2.faiss`) that the app uses

## Run the app

```bash
streamlit run app.py
```

## Evaluation

`05_evaluation.ipynb` (baseline) and `05_evaluation_v2.ipynb` (improved chunks) use the same 10 questions: 8 that the papers answer and 2 off-topic. Answers were graded by hand (1 = correct, 0.5 = partial, 0 = wrong).

| Setup | Hit@1 | Hit@5 | Answerable questions answered | Answer score (answerable only) | Fully correct |
|---|---|---|---|---|---|
| Baseline (500-character chunks) | 88% (7/8) | 88% (7/8) | 5/8 (62%) | 0.56 | 4/8 |
| v2 (sentence-aware chunks) | 88% (7/8) | 88% (7/8) | 7/8 (88%) | 0.88 | 7/8 |
| v2 + hybrid search (BM25 + embeddings, keyword weight 2) | 88% (7/8) | 100% (8/8) | 8/8 (100%) | 1.00 | 8/8 |

All three setups refused both off-topic questions (2/2).

- Sentence-aware chunking was the biggest improvement: two questions the baseline refused were answered correctly.
- One exact-term question (the role of FAISS) was still missed. The only passage that mentions FAISS ranked 189th of 726 with embeddings alone, but first with BM25. Hybrid search puts it in the top 5, and the question is then answered correctly.
- Off-topic questions score about 0.20 in similarity against at least 0.45 for the weakest real question, so a 0.30 threshold rejects them without calling the language model.

**Caveats:** the test set is small (8 answerable questions). The keyword weight was chosen after seeing the FAISS failure, and the expected source papers for two questions were chosen after seeing the results, so the hybrid result should be confirmed on a larger, independent set. Gemini answers also vary slightly between runs.

## Limitations

- The evaluation is small (8 answerable questions), so each question moves a score by 12 points. The hybrid weight was chosen on the same questions.
- Dense retrieval alone misses exact-term questions (for example "role of FAISS"); hybrid search addresses this on the test set, but the fix needs wider testing.
- The corpus is 10 papers on mixed topics, so general questions have thin coverage.
- Scanned PDFs without a text layer cannot be uploaded.
- The evaluation measured answers from the library only, with a stricter prompt. The combined evidence from arXiv and Semantic Scholar, and the app’s prompt (which lets Gemini answer partly and say what is missing), were checked by hand on a few questions and have not been measured.
- Online papers contribute only their title and abstract, not the full text, and some have no free PDF, so they cannot be added automatically.
- Gemini’s free tier is the main reliability problem: it allows about 20 requests per day per model, and the model is often busy (503/504 errors) or slow. The app mitigates this but cannot remove it.
- Semantic Scholar needs a free API key to be reliable, and even with a key it allows about one request per second and can still refuse a request. arXiv allows one request every 3 seconds and is sometimes slow. The app retries both and offers a retry button, but a failed source simply gives fewer suggestions.

## Future work

- Larger and more focused corpus, and a bigger evaluation set with automatic answer checks
- Re-ranking of retrieved passages
- Page numbers in citations
- A Compare tab in the app (paper comparison currently exists only in notebook 04)
- A measured evaluation of the suggestions, and another free source such as OpenAlex as a fallback

## Project structure

```
app.py                  Streamlit application
notebooks/              01 discovery, 02 corpus, 02b rechunk, 03 summarize, 04 compare, 05 evaluation (baseline + v2)
data/                   papers, index files, SQLite database and evaluation results (index, PDFs and database are git-ignored)
requirements.txt
```
