# AI-Powered Research Paper Intelligence Assistant

A Retrieval-Augmented Generation (RAG) assistant for research papers. Ask questions in a chat interface, upload your own PDFs, and generate structured summaries. Answers are grounded in the indexed papers and cite the passages they use.

## Features

- **Online paper discovery**: searches arXiv and Semantic Scholar for relevant papers outside the local library
- **Relevance filtering**: ranks and filters online papers before showing related-paper suggestions
- **Online abstracts as evidence**: relevant online abstracts can optionally be included with local library evidence
- **SQLite persistence**: stores paper metadata, summaries, and chat history locally
- **API error handling**: handles rate limits, temporary server errors, and connection timeouts

- **Chat Q&A** over the indexed papers, with inline citations `[1]`, `[2]` and an expandable evidence list
- **PDF upload**: new papers are parsed, chunked, embedded and added to the search index immediately
- **Paper summaries** (problem, method, dataset, results) for any paper, including uploaded ones
- **Hybrid search**: sentence embeddings plus BM25 keyword search (can be switched off in the sidebar)
- **Off-topic guard**: questions with no relevant passages are rejected without calling the LLM

## Architecture

1. Papers are downloaded from arXiv (`arxiv` API) and parsed with PyMuPDF.
2. Text is cleaned (reference list removed, hyphenation fixed) and split into sentence-aware chunks of about 800 characters.
3. Chunks are embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and stored in a FAISS inner-product index (cosine similarity on normalized vectors). Chunk text and source are stored in a pickle file.
4. A question is embedded and relevant local passages are retrieved using dense search, optionally fused with BM25 using Reciprocal Rank Fusion.
5. When enabled, arXiv and Semantic Scholar are searched for additional relevant papers. Papers already in the local library are excluded from new-paper suggestions.
6. Relevant local passages and, optionally, online abstracts are provided to Gemini to generate a grounded answer.
7. Streamlit displays the answer, citations, evidence, and related online papers.
8. SQLite stores persistent paper metadata, summaries, and chat history.

## Tech stack

Python 3.12 · Streamlit · PyMuPDF · Sentence-Transformers (PyTorch) · FAISS (`faiss-cpu`) · rank-bm25 · Google Gemini API (`google-genai`) · arXiv API · Semantic Scholar API · SQLite · python-dotenv · Pandas

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
- The initial local corpus is 10 papers on mixed topics, so general questions may have thin local coverage.
- Semantic Scholar can apply API rate limits when no API key is available.
- Online search depends on external API availability and limits.
- The free Gemini tier allows only 20 requests per day and 5 per minute, and the API sometimes returns 503/504 errors.
- Scanned PDFs without a text layer cannot be uploaded.

## SQLite database

The application uses SQLite as a lightweight local database for persistent application data. It stores paper metadata, generated summaries, and chat history. The database is created locally as `data/assistant.db`. SQLite is used for persistence; FAISS and BM25 are used for retrieval.

## Future work

- Larger and more focused corpus, and a bigger evaluation set with automatic answer checks
- Re-ranking of retrieved passages
- Page numbers in citations
- Improved metadata extraction and online-paper ranking
- Richer literature comparison and analysis features

## Project structure

```
app.py                  Streamlit application
notebooks/              01 discovery, 02 corpus, 02b rechunk, 03 summarize, 04 compare, 05 evaluation (baseline + v2)
data/                   papers, indexes, SQLite database and evaluation results (generated files can be git-ignored)
requirements.txt
```
