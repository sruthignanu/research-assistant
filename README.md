# AI-Powered Research Paper Intelligence Assistant

A Retrieval-Augmented Generation (RAG) assistant for research papers. Ask questions in a chat interface, upload your own PDFs, and generate structured summaries. Answers are grounded in the indexed papers and cite the passages they use.

## Features

- **Chat Q&A** over the indexed papers, with inline citations `[1]`, `[2]` and an expandable evidence list
- **PDF upload**: new papers are parsed, chunked, embedded and added to the search index immediately
- **Paper summaries** (problem, method, dataset, results) for any paper, including uploaded ones
- **Hybrid search**: sentence embeddings plus BM25 keyword search (can be switched off in the sidebar)
- **Off-topic guard**: questions with no relevant passages are rejected without calling the LLM

## Architecture

1. Papers are downloaded from arXiv (`arxiv` API) and parsed with PyMuPDF.
2. Text is cleaned (reference list removed, hyphenation fixed) and split into sentence-aware chunks of about 800 characters.
3. Chunks are embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and stored in a FAISS inner-product index (cosine similarity on normalized vectors). Chunk text and source are stored in a pickle file.
4. A question is embedded, the top passages are retrieved (dense search, fused with BM25 by reciprocal rank fusion), and the passages plus the question are sent to Gemini, which writes a cited answer.
5. The Streamlit app shows the answer with the retrieved evidence.

## Tech stack

Python 3.12 · Streamlit · PyMuPDF · Sentence-Transformers (PyTorch) · FAISS (`faiss-cpu`) · rank-bm25 · Google Gemini API (`google-genai`, model `gemini-3.8-flash`) · python-dotenv · Pandas

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

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

| Setup | Hit@1 | Answerable questions answered | Answer score (answerable only) | Fully correct |
|---|---|---|---|---|
| Baseline (500-character chunks) | 88% (7/8) | 5/8 (62%) | 0.56 | 4/8 |
| v2 (sentence-aware chunks) | 88% (7/8) | 7/8 (88%) | 0.88 | 7/8 |

Both setups refused both off-topic questions (2/2).

Retrieval-only test of hybrid search (BM25 plus embeddings, keyword weight 2): the correct paper appears in the top 5 for 8/8 questions, against 7/8 for embeddings alone, and the FAISS passage that embeddings ranked 189th of 726 is retrieved. Answer-level results for hybrid search are not yet part of the table above.

## Limitations

- The evaluation is small (8 answerable questions), so each question moves a score by 12 points. The hybrid weight was chosen on the same questions.
- Dense retrieval alone misses exact-term questions (for example "role of FAISS"); hybrid search addresses this.
- The corpus is 10 papers on mixed topics, so general questions have thin coverage.
- The free Gemini tier allows only 20 requests per day and 5 per minute, and the API sometimes returns 503/504 errors.
- Scanned PDFs without a text layer cannot be uploaded.

## Future work

- Larger and more focused corpus, and a bigger evaluation set with automatic answer checks
- Re-ranking of retrieved passages
- Page numbers in citations
- Storing paper metadata in SQLite

## Project structure

```
app.py                  Streamlit application
notebooks/              01 discovery, 02 corpus, 02b rechunk, 03 summarize, 04 compare, 05 evaluation (baseline + v2)
data/                   papers, index files and evaluation results (index and PDFs are git-ignored)
requirements.txt
```
