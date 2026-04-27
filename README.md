# MusicRAG

An AI-powered study companion for Western music history, built on Burkholder's *A History of Western Music* (10th edition). Uses Retrieval-Augmented Generation (RAG) to answer questions grounded in textbook content.

## What It Does

Ask any question about Western music history — composers, styles, forms, historical developments — and get an accurate, source-cited answer based on the textbook.

```
Enter your question: Who is Bach?

📚 Sources:
  - [0.0308] [bm25+vector] Chapter 19: German Composers of the Late Baroque, chunk 8
  - [0.0288] [bm25+vector] Chapter 19: German Composers of the Late Baroque, chunk 23
  - [0.0283] [bm25+vector] Chapter 19: German Composers of the Late Baroque, chunk 65

Johann Sebastian Bach (1685–1750) was a German composer now considered one of
the greatest in the Western music tradition. He was renowned as an organ virtuoso,
keyboard composer, and writer of learned contrapuntal works... (Chapter 19)
```

## How It Works

```
User Question
      │
      ▼
┌──────────────────────┐
│  Query Normalization │  Strip filler phrases ("expand on", "explain")
└──────┬───────────────┘
       ▼
┌──────────────┐
│   Embedding  │  all-MiniLM-L6-v2 (384-dim, local CPU)
└──────┬───────┘
       ▼
┌─────────────────────────────────────────────┐
│         Hybrid Retrieval (parallel)         │
│                                             │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Vector Path    │  │   BM25 Path      │  │
│  │                 │  │                  │  │
│  │  Summary embeds │  │  Lexical search  │  │
│  │  → top chapters │  │  over all chunks │  │
│  │  → chunk search │  │                  │  │
│  │  → keyword rank │  │                  │  │
│  └────────┬────────┘  └────────┬─────────┘  │
│           └────────┬───────────┘             │
│                    ▼                         │
│        Reciprocal Rank Fusion (RRF)          │
└──────────────────┬──────────────────────────┘
                   ▼
┌──────────────────────┐
│  Filtering           │
│  Dynamic threshold   │  Adapt cutoff per query
│  Chapter diversity   │  Max 3 chunks per chapter
│  Top-k selection     │  Final 5 candidates
└──────┬───────────────┘
       ▼
┌──────────────┐
│   LLM API    │  Generate answer with source citations
└──────┬───────┘
       ▼
   Cited Answer
```

## Setup

### Prerequisites

- Python 3.10+
- A PDF of Burkholder's *A History of Western Music* (10th edition)
- An LLM API key for answer generation (default: [MiniMax](https://www.minimaxi.com/) via Anthropic SDK)
- An OpenAI API key for chapter summary generation

### Installation

```bash
git clone https://github.com/XL9610/MusicHistoryRAG.git
cd MusicHistoryRAG
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
MINIMAX_API_KEY=your_minimax_key_here
OPENAI_API_KEY=your_openai_key_here
```

To use a different LLM provider for answer generation, edit the API settings in `config.py`.

### Build the Database

Place your PDF in the project root (named `A History of Western Music Tenth.pdf`), then run:

```bash
python build_database.py
```

This will:
1. Read PDF bookmarks (table of contents) to identify all 39 chapters
2. Extract chapter titles and page ranges directly from the bookmark tree
3. Extract text for each chapter using the bookmark page boundaries
4. Generate LLM-powered chapter summaries via OpenAI (cached to `chapter_summaries.json` after first run)
5. Chunk each chapter (~1,500 chars with 200-char overlap)
6. Generate embeddings for all chunks and summaries
7. Store ~2,300 chunks + 39 chapter summaries in a local ChromaDB database (cosine distance)

First run takes a few minutes (mostly LLM summary generation). Subsequent runs reuse cached summaries and complete in ~1 minute.

### Query

```bash
python query.py
```

## Project Structure

```
├── config.py                # All tunable parameters (models, thresholds, API keys)
├── retriever.py             # Hybrid retrieval: vector + BM25 → RRF → filter
├── generator.py             # LLM prompt construction and API call
├── query.py                 # Interactive CLI with retrieval strategy selector
├── build_database.py        # Ingestion pipeline: PDF bookmarks → chunks → ChromaDB
├── chapter_summaries.json   # Cached LLM-generated chapter summaries (auto-generated)
├── requirements.txt
└── .env                     # API keys (not committed)
```

## Retrieval Strategy

The retrieval pipeline went through five iterations:

**v1 — Keyword gate**: Used keyword filtering (`$contains`) as an entry point before vector search. Failed on natural language queries like "expand on galant style" because irrelevant keywords ("expand") narrowed the search space.

**v2 — Improved keyword gate**: Added stop words, case-variant matching, term sorting by length. Better, but the fundamental architecture was still fragile.

**v3 — Vector-first with keyword reranking**: Always starts with broad vector search (top 20), then applies keyword-based reranking, dynamic thresholds, and chapter diversity filtering. Most stable across diverse query types.

**v4 — Hierarchical retrieval**: Added a chapter summary layer. First retrieves the top 3 most relevant chapters via summary embeddings, then searches chunks only within those chapters.

**v5 — Hybrid search (current)**: Runs two parallel retrieval paths — hierarchical vector search and BM25 lexical search — then merges results using Reciprocal Rank Fusion (RRF). This combines semantic understanding with exact keyword matching for robust recall across both natural language queries and specific identifiers (BWV numbers, opus numbers, composer names).

### Key features

- **Bookmark-based ingestion** — chapters are split using the PDF's built-in table of contents, not fragile text pattern matching
- **LLM-generated summaries** — GPT-4o-mini generates dense, retrieval-optimized chapter summaries covering all composer names, work titles, and key terms (cached after first generation)
- **Hybrid search** — vector and BM25 paths run in parallel; vector excels at semantic matching, BM25 catches exact terms that embeddings may miss
- **Reciprocal Rank Fusion** — merges results from both paths using rank positions rather than raw scores, which aren't directly comparable across retrieval methods
- **Hierarchical vector path** — chapter-level summary retrieval narrows scope before chunk search
- **Query normalization** — strips filler phrases before embedding
- **Keyword reranking** — boosts chunks containing query terms: `0.7 × semantic_score + 0.2 × keyword_coverage + phrase_bonus`
- **Dynamic threshold** — `max_score × 0.5` adapts to each query's score distribution
- **Chapter diversity** — max 3 chunks per chapter to avoid redundant results
- **Fallback** — returns top 5 candidates if primary filtering returns empty

## Tech Stack

| Component | Tool |
|-----------|------|
| PDF extraction | PyMuPDF (with bookmark-based chapter splitting) |
| Text chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | ChromaDB (cosine distance) |
| BM25 search | rank-bm25 |
| Chapter summaries | OpenAI GPT-4o-mini |
| Answer generation | MiniMax-M2.7 (via Anthropic SDK) |

## Limitations

- Chunking is fixed-size, not section-aware — some chunks split mid-paragraph
- Single-turn only — no conversation memory
- Reranking is lexical keyword overlap, not a cross-encoder model
- No formal evaluation framework yet

## Planned Features

- **Section-aware chunking** — use level-3 PDF bookmarks (sub-sections) as chunk boundaries instead of fixed character counts
- **Cross-encoder reranking** — replace keyword overlap scoring with a neural reranker for more accurate result ordering
- **Quiz Mode** — auto-generate multiple-choice questions from textbook content
- **Composer Profile** — input a name, get a structured summary from all mentions across chapters
- **Concept Tracker** — trace a concept's evolution across chapters and eras
- **Web UI** — Streamlit/Gradio frontend with Markdown rendering
- **Evaluation framework** — standardized question set with expected chapters for automated retrieval quality measurement

## License

MIT
